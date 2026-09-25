"""
Matching model training with LightGBM.
Supports positive/negative pair construction and IDF computation.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from ..features.pair_features import build_pair_features, FEATURE_NAMES


# ── IDF computation ────────────────────────────────────────────────────────────

def _compute_idf(token_lists: list[list[str]], smooth: bool = True) -> dict[str, float]:
    """Compute IDF over a corpus of token lists."""
    n = len(token_lists)
    df_count: Counter = Counter()
    for tokens in token_lists:
        df_count.update(set(tokens))
    idf = {}
    for term, cnt in df_count.items():
        if smooth:
            idf[term] = math.log((n + 1) / (cnt + 1)) + 1.0
        else:
            idf[term] = math.log(n / cnt)
    return idf


def build_idf_dicts(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame
) -> tuple[dict, dict]:
    """Build name_idf and addr_idf over all sources."""
    from ..normalization.names import normalize_name
    from ..normalization.addresses import normalize_address

    combined = pd.concat([s1, s2, s3], ignore_index=True)

    name_token_lists = []
    addr_token_lists = []
    for _, row in combined.iterrows():
        nm = normalize_name(str(row.get("business_name", "") or ""))
        ad = normalize_address(str(row.get("business_address", "") or ""))
        name_token_lists.append(nm.get("tokens", []))
        addr_token_lists.append(ad.get("tokens", []))

    name_idf = _compute_idf(name_token_lists)
    addr_idf = _compute_idf(addr_token_lists)
    return name_idf, addr_idf


# ── Training pair builder ──────────────────────────────────────────────────────

def build_training_pairs(
    s1: pd.DataFrame,
    candidates: dict[str, set[str]],
    s23_lookup: dict[str, dict],
    gt_dict: dict[str, set[str]],
    name_idf: dict,
    addr_idf: dict,
    neg_ratio: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, groups) for LightGBM training.

    s1           : Source 1 DataFrame
    candidates   : {s1_id -> set of candidate ids}
    s23_lookup   : {entity_id -> row dict} for S2 and S3
    gt_dict      : {s1_id -> set of true target ids}
    neg_ratio    : negatives per positive
    """
    rng = random.Random(seed)
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1.iterrows()}

    X_rows = []
    y_rows = []
    group_ids = []

    group_counter = 0

    for s1_id, cand_set in candidates.items():
        true_ids = gt_dict.get(s1_id, set())
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        positives = list(true_ids & cand_set)
        all_negatives = list(cand_set - true_ids)
        rng.shuffle(all_negatives)
        num_neg = min(len(all_negatives), max(neg_ratio * len(positives), 2))
        selected_negatives = all_negatives[:num_neg]

        pairs = [(p, 1) for p in positives] + [(n, 0) for n in selected_negatives]
        if not pairs:
            continue

        group_counter += 1
        for cand_id, label in pairs:
            cand_row = s23_lookup.get(cand_id)
            if cand_row is None:
                continue
            feats = build_pair_features(s1_row, cand_row, name_idf, addr_idf)
            feat_vec = [feats.get(fn, 0.0) for fn in FEATURE_NAMES]
            X_rows.append(feat_vec)
            y_rows.append(label)
            group_ids.append(group_counter)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)
    g = np.array(group_ids, dtype=np.int32)
    return X, y, g


# ── Model training ─────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> lgb.Booster:
    """Train a LightGBM binary classifier. Returns the booster."""
    if params is None:
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

    # Group-based train/val split (don't split same S1 group across sets)
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    n_val = max(1, int(len(unique_groups) * val_fraction))
    val_groups = set(rng.choice(unique_groups, n_val, replace=False).tolist())
    train_mask = np.array([g not in val_groups for g in groups])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    print(f"[Model] Best iteration: {booster.best_iteration}")
    return booster


def predict_scores(
    booster: lgb.Booster,
    s1_row: dict,
    candidate_rows: list[dict],
    name_idf: dict,
    addr_idf: dict,
) -> list[tuple[str, float]]:
    """
    Return [(candidate_id, match_score), ...] sorted descending.
    """
    if not candidate_rows:
        return []

    X_rows = []
    cand_ids = []
    for cand_row in candidate_rows:
        feats = build_pair_features(s1_row, cand_row, name_idf, addr_idf)
        feat_vec = [feats.get(fn, 0.0) for fn in FEATURE_NAMES]
        X_rows.append(feat_vec)
        cand_ids.append(cand_row["entity_id"])

    X = np.array(X_rows, dtype=np.float32)
    scores = booster.predict(X)
    return sorted(zip(cand_ids, scores.tolist()), key=lambda x: -x[1])
