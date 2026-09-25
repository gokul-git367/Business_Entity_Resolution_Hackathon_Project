"""
F0.5 evaluation and threshold optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def f05_score(precision: float, recall: float) -> float:
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_entity_f05(
    pred_ids: set[str],
    true_ids: set[str],
) -> float:
    """
    Compute F0.5 for a single S1 entity.
    pred_ids: set of predicted target IDs (can be empty)
    true_ids: set of ground-truth target IDs (can be empty)
    """
    if not true_ids and not pred_ids:
        return 1.0  # Correctly predicted no match
    if not true_ids and pred_ids:
        return 0.0  # False merge on singleton
    if true_ids and not pred_ids:
        recall = 0.0
        precision = 1.0  # vacuous
        return f05_score(precision, recall)

    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return f05_score(precision, recall)


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """
    Macro-average F0.5 over all S1 entities.
    predictions : {s1_id -> set of predicted target ids}
    ground_truth: {s1_id -> set of true target ids}
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(compute_entity_f05(pred_ids, true_ids))
    return float(np.mean(scores)) if scores else 0.0


def optimize_threshold(
    scores_by_s1: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, set[str]],
    thresholds: list[float] | None = None,
) -> tuple[float, dict]:
    """
    Sweep thresholds and return the best one for macro F0.5.

    scores_by_s1: {s1_id -> [(candidate_id, score), ...]}
    ground_truth: {s1_id -> set of true target ids}

    Returns (best_threshold, results_dict).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.20, 0.96, 0.02)]

    best_thresh = 0.5
    best_f05 = -1.0
    results = {}

    for thresh in thresholds:
        preds = {}
        for s1_id, cand_scores in scores_by_s1.items():
            accepted = {cid for cid, sc in cand_scores if sc >= thresh}
            preds[s1_id] = accepted

        score = macro_f05(preds, ground_truth)
        results[thresh] = score

        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    print(f"[Threshold] Best threshold={best_thresh:.2f}, macro F0.5={best_f05:.4f}")
    return best_thresh, results


def ground_truth_dict(gt: pd.DataFrame) -> dict[str, set[str]]:
    """Build {s1_id -> set(targets)} from the raw GT DataFrame."""
    d: dict[str, set[str]] = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        if raw == "" or pd.isna(raw):
            d[s1] = set()
        else:
            d[s1] = {t.strip() for t in raw.split(",") if t.strip()}
    return d
