"""
Blocking evaluation: measures recall of the candidate generator against ground truth.
"""

from __future__ import annotations

import pandas as pd


def evaluate_blocking_recall(
    candidates: dict[str, set[str]],
    gt: pd.DataFrame,
) -> dict:
    """
    Compute blocking recall given:
        candidates  : {s1_id -> set of candidate ids}
        gt          : ground truth DataFrame (source1_entity_id, matched_entity_ids)

    Returns a dict with:
        total_true_pairs   - number of true (s1, target) pairs in GT
        recovered          - how many true pairs are in the candidate set
        recall             - recovered / total_true_pairs
        s1_with_zero_match - how many S1 have no true matches
        avg_candidates     - average candidate set size
        max_candidates     - largest candidate set
    """
    total_true = 0
    recovered = 0
    s1_zero = 0

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        if raw == "" or pd.isna(raw):
            s1_zero += 1
            continue

        true_targets = {t.strip() for t in raw.split(",") if t.strip()}
        cands = candidates.get(s1_id, set())

        total_true += len(true_targets)
        recovered += len(true_targets & cands)

    recall = recovered / total_true if total_true > 0 else 1.0
    avg_cands = (
        sum(len(v) for v in candidates.values()) / len(candidates)
        if candidates else 0.0
    )
    max_cands = max((len(v) for v in candidates.values()), default=0)

    return {
        "total_true_pairs": total_true,
        "recovered": recovered,
        "recall": recall,
        "s1_with_zero_match": s1_zero,
        "avg_candidates": avg_cands,
        "max_candidates": max_cands,
    }
