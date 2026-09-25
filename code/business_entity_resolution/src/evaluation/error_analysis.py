"""
Error analysis: classifies false positives and false negatives by category.
"""

from __future__ import annotations

import pandas as pd
from collections import Counter

from ..normalization.names import normalize_name
from ..normalization.addresses import normalize_address
from ..normalization.scripts import dominant_script


def analyze_errors(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    s23_lookup: dict[str, dict],
    s1_lookup: dict[str, dict],
) -> dict:
    """
    Classify false positive and false negative pairs by error category.
    Returns a summary dict.
    """
    fp_categories: Counter = Counter()
    fn_categories: Counter = Counter()

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())

        # False positives
        for fp_id in pred_ids - true_ids:
            s1_row = s1_lookup.get(s1_id, {})
            cand_row = s23_lookup.get(fp_id, {})
            cat = _classify_fp(s1_row, cand_row)
            fp_categories[cat] += 1

        # False negatives
        for fn_id in true_ids - pred_ids:
            s1_row = s1_lookup.get(s1_id, {})
            cand_row = s23_lookup.get(fn_id, {})
            cat = _classify_fn(s1_row, cand_row)
            fn_categories[cat] += 1

    total_fp = sum(fp_categories.values())
    total_fn = sum(fn_categories.values())

    return {
        "false_positives": dict(fp_categories.most_common()),
        "false_negatives": dict(fn_categories.most_common()),
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def _classify_fp(s1_row: dict, cand_row: dict) -> str:
    name1 = str(s1_row.get("business_name", "") or "")
    name2 = str(cand_row.get("business_name", "") or "")
    addr1 = str(s1_row.get("business_address", "") or "")
    addr2 = str(cand_row.get("business_address", "") or "")

    n1, n2 = normalize_name(name1), normalize_name(name2)
    core1, core2 = n1.get("core", ""), n2.get("core", "")

    # Very common names
    tokens1 = set(n1.get("tokens", []))
    _generic = {"services", "trading", "company", "enterprises", "solutions"}
    if len(tokens1) <= 2 or tokens1.issubset(_generic):
        return "common_or_short_name"

    # Postal conflict
    a1, a2 = normalize_address(addr1), normalize_address(addr2)
    if a1.get("postal") and a2.get("postal") and a1["postal"] != a2["postal"]:
        return "postal_conflict"

    # Same name, different address
    if core1 == core2 and a1.get("clean") != a2.get("clean") and a1.get("clean") and a2.get("clean"):
        return "same_name_different_address"

    if not a1.get("clean") or not a2.get("clean"):
        return "missing_address_false_merge"

    return "other_fp"


def _classify_fn(s1_row: dict, cand_row: dict) -> str:
    name1 = str(s1_row.get("business_name", "") or "")
    name2 = str(cand_row.get("business_name", "") or "")

    sc1 = dominant_script(name1)
    sc2 = dominant_script(name2)
    if sc1 != sc2 and sc1 != "Unknown" and sc2 != "Unknown":
        return "script_mismatch_or_transliteration"

    n1, n2 = normalize_name(name1), normalize_name(name2)
    if n1.get("has_dba") or n2.get("has_dba"):
        return "dba_alias"

    a1 = normalize_address(str(s1_row.get("business_address", "") or ""))
    a2 = normalize_address(str(cand_row.get("business_address", "") or ""))
    if a1.get("is_missing") or a2.get("is_missing"):
        return "missing_address"

    return "other_fn"


def print_error_summary(analysis: dict):
    print("\n[Error Analysis] False Positive Categories:")
    for cat, cnt in sorted(analysis["false_positives"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    print("\n[Error Analysis] False Negative Categories:")
    for cat, cnt in sorted(analysis["false_negatives"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    print(f"\n  Total FPs: {analysis['total_fp']}, Total FNs: {analysis['total_fn']}")
