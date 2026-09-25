"""
Data loading utilities.
All files are TSV with columns: entity_id, business_name, business_address, country.
Ground truth has: source1_entity_id, matched_entity_ids (comma-separated).
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_source(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    # Fill empty strings and strip whitespace
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df


def load_source1(path: str | Path) -> pd.DataFrame:
    return _read_source(path)


def load_source2(path: str | Path) -> pd.DataFrame:
    return _read_source(path)


def load_source3(path: str | Path) -> pd.DataFrame:
    return _read_source(path)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        source1_entity_id   str
        matched_entity_ids  str  (raw comma-separated, may be empty)
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df["matched_entity_ids"] = df["matched_entity_ids"].str.strip()
    return df


def ground_truth_to_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the ground truth into individual (s1_id, target_id) pairs.
    Returns a DataFrame with columns: source1_entity_id, target_id.
    Rows where matched_entity_ids is empty are retained with target_id = ''.
    """
    rows = []
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        if raw == "":
            rows.append({"source1_entity_id": s1, "target_id": ""})
        else:
            for tid in raw.split(","):
                tid = tid.strip()
                if tid:
                    rows.append({"source1_entity_id": s1, "target_id": tid})
    return pd.DataFrame(rows)


def load_all_train(data_dir: str | Path):
    """
    Returns (s1, s2, s3, gt) DataFrames.
    """
    d = Path(data_dir)
    s1 = load_source1(d / "train_source1.tsv")
    s2 = load_source2(d / "train_source2.tsv")
    s3 = load_source3(d / "train_source3.tsv")
    gt = load_ground_truth(d / "train_ground_truth.tsv")
    return s1, s2, s3, gt


def load_all_test(data_dir: str | Path):
    """
    Returns (s1, s2, s3) DataFrames.
    """
    d = Path(data_dir)
    s1 = load_source1(d / "test_source1.tsv")
    s2 = load_source2(d / "test_source2.tsv")
    s3 = load_source3(d / "test_source3.tsv")
    return s1, s2, s3
