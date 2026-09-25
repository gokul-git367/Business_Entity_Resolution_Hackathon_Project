"""
Input validation: verify TSV structure, ID prefixes, duplicates, and ground-truth integrity.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def validate_source(df: pd.DataFrame, expected_prefix: str, name: str) -> list[str]:
    issues = []
    required = {"entity_id", "business_name", "business_address", "country"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        issues.append(f"{name}: missing columns {missing_cols}")
        return issues

    bad_prefix = df[~df["entity_id"].str.startswith(expected_prefix)]
    if not bad_prefix.empty:
        issues.append(
            f"{name}: {len(bad_prefix)} rows with unexpected ID prefix "
            f"(expected '{expected_prefix}')"
        )

    dups = df[df["entity_id"].duplicated()]
    if not dups.empty:
        issues.append(f"{name}: {len(dups)} duplicate entity_ids")

    return issues


def validate_ground_truth(
    gt: pd.DataFrame,
    s1_ids: set[str],
    s2_ids: set[str],
    s3_ids: set[str],
) -> list[str]:
    issues = []
    required_cols = {"source1_entity_id", "matched_entity_ids"}
    if not required_cols.issubset(set(gt.columns)):
        issues.append(f"Ground truth missing columns: {required_cols - set(gt.columns)}")
        return issues

    # Every S1 must appear once
    gt_s1 = set(gt["source1_entity_id"])
    missing_from_gt = s1_ids - gt_s1
    extra_in_gt = gt_s1 - s1_ids
    if missing_from_gt:
        issues.append(
            f"Ground truth: {len(missing_from_gt)} S1 IDs missing from labels"
        )
    if extra_in_gt:
        issues.append(
            f"Ground truth: {len(extra_in_gt)} S1 IDs in labels not in source1"
        )

    dups = gt[gt["source1_entity_id"].duplicated()]
    if not dups.empty:
        issues.append(f"Ground truth: {len(dups)} duplicate source1_entity_id rows")

    # Check all target IDs are valid
    valid_target_ids = s2_ids | s3_ids
    bad_targets: list[str] = []
    for raw in gt["matched_entity_ids"]:
        if raw == "":
            continue
        for tid in raw.split(","):
            tid = tid.strip()
            if tid and tid not in valid_target_ids:
                bad_targets.append(tid)
    if bad_targets:
        issues.append(
            f"Ground truth: {len(bad_targets)} target IDs not found in S2/S3 "
            f"(first 5: {bad_targets[:5]})"
        )

    return issues


def run_validation(s1, s2, s3, gt) -> bool:
    """Print all issues and return True if clean."""
    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])
    s1_ids = set(s1["entity_id"])

    issues = []
    issues += validate_source(s1, "S1-", "Source1")
    issues += validate_source(s2, "S2-", "Source2")
    issues += validate_source(s3, "S3-", "Source3")
    issues += validate_ground_truth(gt, s1_ids, s2_ids, s3_ids)

    if issues:
        for issue in issues:
            print(f"[ISSUE] {issue}")
        return False
    print(f"[OK] Validation passed: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, GT={len(gt)}")
    return True
