"""
End-to-end pipeline for Business Entity Resolution.

Usage:
    python pipeline.py --mode train   # train + evaluate
    python pipeline.py --mode test    # run on test set (requires saved model)
    python pipeline.py --mode full    # train on all data, then test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root discovery ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE                 # code/business_entity_resolution/
_STUDENT_ROOT = _PROJECT_ROOT.parent.parent  # student_resource/

TRAIN_DIR = _STUDENT_ROOT / "dataset" / "train"
TEST_DIR = _STUDENT_ROOT / "dataset" / "test"
OUTPUT_DIR = _STUDENT_ROOT / "output"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts"

OUTPUT_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


# ── Imports ────────────────────────────────────────────────────────────────────
from src.data.loader import (
    load_all_train, load_all_test, ground_truth_to_pairs,
)
from src.data.validation import run_validation
from src.blocking.candidate_generator import CandidateGenerator
from src.evaluation.blocking_eval import evaluate_blocking_recall
from src.evaluation.metrics import ground_truth_dict, optimize_threshold, macro_f05
from src.model.train import (
    build_idf_dicts, build_training_pairs, train_model, predict_scores,
)
from src.features.pair_features import FEATURE_NAMES


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_lookup(df: pd.DataFrame) -> dict[str, dict]:
    return {row["entity_id"]: row.to_dict() for _, row in df.iterrows()}


def write_output(
    s1_ids: list[str],
    predictions: dict[str, set[str]],
    candidates: dict[str, set[str]],
    out_dir: Path,
):
    out_dir.mkdir(exist_ok=True)

    # matching_results.tsv
    match_path = out_dir / "matching_results.tsv"
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matched = predictions.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(matched))}\n")
    print(f"[Output] Written {match_path}")

    # candidate_pairs.tsv
    cand_path = out_dir / "candidate_pairs.tsv"
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cands = candidates.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")
    print(f"[Output] Written {cand_path}")


# ── Main modes ─────────────────────────────────────────────────────────────────

def run_train_eval(val_fraction: float = 0.20, seed: int = 42):
    """Train on a split of the training data, evaluate on the validation split."""
    print("=" * 60)
    print("[Pipeline] Starting TRAIN+EVAL mode")

    # 1. Load
    s1, s2, s3, gt = load_all_train(TRAIN_DIR)
    all_s1_ids = list(s1["entity_id"])
    run_validation(s1, s2, s3, gt)

    gt_full_dict = ground_truth_dict(gt)

    # 2. Train/val split on S1 entities
    rng = np.random.RandomState(seed)
    s1_ids_arr = np.array(all_s1_ids)
    rng.shuffle(s1_ids_arr)
    n_val = int(len(s1_ids_arr) * val_fraction)
    val_s1_ids = set(s1_ids_arr[:n_val].tolist())
    train_s1_ids = set(s1_ids_arr[n_val:].tolist())

    s1_train = s1[s1["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_s1_ids)].reset_index(drop=True)

    print(f"[Pipeline] Train S1: {len(s1_train)}, Val S1: {len(s1_val)}")

    gt_val_dict = {k: v for k, v in gt_full_dict.items() if k in val_s1_ids}

    # 3. Build IDF over all data
    print("[Pipeline] Building IDF dicts ...")
    name_idf, addr_idf = build_idf_dicts(s1, s2, s3)
    with open(ARTIFACT_DIR / "idf.pkl", "wb") as f:
        pickle.dump((name_idf, addr_idf), f)

    # 4. Blocking
    print("[Pipeline] Building blocking indexes ...")
    gen = CandidateGenerator()
    gen.build(s2, s3)
    with open(ARTIFACT_DIR / "candidate_generator.pkl", "wb") as f:
        pickle.dump(gen, f)

    # 5. Generate candidates for train subset
    print("[Pipeline] Generating train candidates ...")
    train_candidates = gen.generate_candidates(s1_train, max_candidates_per_s1=300)
    # Generate val candidates
    print("[Pipeline] Generating val candidates ...")
    val_candidates = gen.generate_candidates(s1_val, max_candidates_per_s1=300)

    # 6. Blocking evaluation on val
    val_block_metrics = evaluate_blocking_recall(val_candidates, gt[gt["source1_entity_id"].isin(val_s1_ids)])
    print(f"[Blocking] Val recall={val_block_metrics['recall']:.4f}, "
          f"avg_candidates={val_block_metrics['avg_candidates']:.1f}")

    # 7. Build S2/S3 lookup
    s23_lookup = build_lookup(s2)
    s23_lookup.update(build_lookup(s3))

    # 8. Build training pairs
    print("[Pipeline] Building training pairs ...")
    gt_train_dict = {k: v for k, v in gt_full_dict.items() if k in train_s1_ids}
    X, y, groups = build_training_pairs(
        s1_train, train_candidates, s23_lookup,
        gt_train_dict, name_idf, addr_idf,
        neg_ratio=4, seed=seed,
    )
    print(f"[Pipeline] Training data: {X.shape[0]} pairs, "
          f"{y.sum()} positives ({100*y.mean():.1f}%)")

    # 9. Train model
    print("[Pipeline] Training LightGBM model ...")
    booster = train_model(X, y, groups, val_fraction=0.15, seed=seed)
    booster.save_model(str(ARTIFACT_DIR / "lgb_model.txt"))

    # 10. Score val candidates
    print("[Pipeline] Scoring validation candidates ...")
    scores_by_s1: dict[str, list[tuple[str, float]]] = {}
    s1_val_lookup = build_lookup(s1_val)

    for s1_id, cand_ids in val_candidates.items():
        s1_row = s1_val_lookup.get(s1_id)
        if s1_row is None:
            continue
        cand_rows = [s23_lookup[cid] for cid in cand_ids if cid in s23_lookup]
        scored = predict_scores(booster, s1_row, cand_rows, name_idf, addr_idf)
        scores_by_s1[s1_id] = scored

    # 11. Threshold optimization
    print("[Pipeline] Optimizing threshold ...")
    best_thresh, thresh_results = optimize_threshold(scores_by_s1, gt_val_dict)
    with open(ARTIFACT_DIR / "threshold.json", "w") as f:
        json.dump({"best_threshold": best_thresh, "results": thresh_results}, f, indent=2)

    # 12. Final val predictions
    val_preds = {}
    for s1_id, cand_scores in scores_by_s1.items():
        val_preds[s1_id] = {cid for cid, sc in cand_scores if sc >= best_thresh}

    final_f05 = macro_f05(val_preds, gt_val_dict)
    print(f"\n[Result] Validation macro F0.5 = {final_f05:.4f}  (threshold={best_thresh})")

    return booster, gen, name_idf, addr_idf, best_thresh


def run_full_train(seed: int = 42):
    """Train on ALL training data, then run inference on test set."""
    print("=" * 60)
    print("[Pipeline] Starting FULL TRAIN mode")

    # 1. Load all training data
    s1, s2, s3, gt = load_all_train(TRAIN_DIR)
    run_validation(s1, s2, s3, gt)
    gt_full_dict = ground_truth_dict(gt)

    # 2. IDF
    print("[Pipeline] Building IDF dicts ...")
    name_idf, addr_idf = build_idf_dicts(s1, s2, s3)

    # 3. Blocking over train S2/S3
    print("[Pipeline] Building blocking indexes (train) ...")
    gen = CandidateGenerator()
    gen.build(s2, s3)

    # 4. Generate train candidates & training pairs
    print("[Pipeline] Generating train candidates ...")
    all_candidates = gen.generate_candidates(s1, max_candidates_per_s1=300)

    s23_lookup = build_lookup(s2)
    s23_lookup.update(build_lookup(s3))

    X, y, groups = build_training_pairs(
        s1, all_candidates, s23_lookup, gt_full_dict,
        name_idf, addr_idf, neg_ratio=4, seed=seed,
    )
    print(f"[Pipeline] Full training set: {X.shape[0]} pairs")

    # 5. Train LightGBM (no val split for full train)
    import lightgbm as lgb
    from src.features.pair_features import FEATURE_NAMES
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_depth": 8,
        "is_unbalance": True,
        "seed": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    booster = lgb.train(params, dtrain, num_boost_round=500)
    booster.save_model(str(ARTIFACT_DIR / "lgb_model_full.txt"))

    # 6. Get threshold from previous eval run (or use default)
    thresh_path = ARTIFACT_DIR / "threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresh_data = json.load(f)
        best_thresh = thresh_data.get("best_threshold", 0.5)
        print(f"[Pipeline] Using saved threshold: {best_thresh}")
    else:
        best_thresh = 0.5
        print("[Pipeline] No saved threshold; using 0.5")

    # 7. Run on test set
    _run_test_inference(booster, gen, name_idf, addr_idf, best_thresh)


def _run_test_inference(booster, gen, name_idf, addr_idf, best_thresh):
    """Run test set inference and write outputs."""
    print("[Pipeline] Loading test data ...")
    s1_test, s2_test, s3_test = load_all_test(TEST_DIR)
    test_s1_ids = list(s1_test["entity_id"])

    print("[Pipeline] Building test blocking indexes ...")
    gen_test = CandidateGenerator()
    gen_test.build(s2_test, s3_test)

    print("[Pipeline] Generating test candidates ...")
    test_candidates = gen_test.generate_candidates(s1_test, max_candidates_per_s1=300)

    s23_test_lookup = build_lookup(s2_test)
    s23_test_lookup.update(build_lookup(s3_test))
    s1_test_lookup = build_lookup(s1_test)

    print("[Pipeline] Scoring test candidates ...")
    test_preds: dict[str, set[str]] = {}
    total = len(test_s1_ids)
    for i, s1_id in enumerate(test_s1_ids):
        if i % 50000 == 0:
            print(f"  {i}/{total} ...")
        s1_row = s1_test_lookup.get(s1_id)
        cand_ids = test_candidates.get(s1_id, set())
        if not cand_ids or s1_row is None:
            test_preds[s1_id] = set()
            continue
        cand_rows = [s23_test_lookup[c] for c in cand_ids if c in s23_test_lookup]
        scored = predict_scores(booster, s1_row, cand_rows, name_idf, addr_idf)
        test_preds[s1_id] = {cid for cid, sc in scored if sc >= best_thresh}

    write_output(test_s1_ids, test_preds, test_candidates, OUTPUT_DIR)
    print("[Pipeline] Done.")


def run_test_only(seed: int = 42):
    """Load saved model artifacts and run on test set."""
    import lightgbm as lgb
    print("[Pipeline] Loading saved artifacts ...")
    booster = lgb.Booster(model_file=str(ARTIFACT_DIR / "lgb_model_full.txt"))
    with open(ARTIFACT_DIR / "idf.pkl", "rb") as f:
        name_idf, addr_idf = pickle.load(f)
    with open(ARTIFACT_DIR / "threshold.json") as f:
        thresh_data = json.load(f)
    best_thresh = thresh_data.get("best_threshold", 0.5)

    # Rebuild test blocking (can't easily serialize/deserialize at scale)
    s1_test, s2_test, s3_test = load_all_test(TEST_DIR)
    gen_test = CandidateGenerator()
    gen_test.build(s2_test, s3_test)

    _run_test_inference(booster, gen_test, name_idf, addr_idf, best_thresh)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "full", "test"],
        default="train",
        help="train: train+eval on split; full: full train then test; test: test only",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "train":
        run_train_eval(seed=args.seed)
    elif args.mode == "full":
        run_full_train(seed=args.seed)
    elif args.mode == "test":
        run_test_only(seed=args.seed)
