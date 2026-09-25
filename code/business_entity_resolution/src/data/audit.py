"""Streaming-friendly dataset audit for ambiguity and data-quality checks."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from ..normalization.names import normalize_name
from ..normalization.addresses import normalize_address
from ..normalization.scripts import dominant_script, is_mixed_script


PLACEHOLDERS = {"", "null", "none", "n/a", "na", "unknown", "-", ".", "?"}


def audit_file(path: str | Path, chunksize: int = 100_000) -> dict:
    """Audit one source file without retaining all rows in memory."""
    stats = Counter()
    scripts = Counter()
    names = Counter()
    addresses = Counter()
    countries = Counter()
    samples: dict[str, list[str]] = {}

    def sample(key: str, value: str, limit: int = 3):
        if value and len(samples.setdefault(key, [])) < limit:
            samples[key].append(value[:160])

    rows = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
        for _, row in chunk.iterrows():
            rows += 1
            name = str(row.get("business_name", "") or "").strip()
            address = str(row.get("business_address", "") or "").strip()
            country = str(row.get("country", "") or "").strip()
            countries[country] += 1
            names[name.lower()] += 1
            addresses[address.lower()] += 1

            for field, value in (("name", name), ("address", address)):
                key = f"{field}_missing"
                if value.lower() in PLACEHOLDERS:
                    stats[key] += 1
                    continue
                norm = normalize_name(value) if field == "name" else normalize_address(value)
                if field == "name":
                    stats["name_dba"] += int(norm.get("has_dba", False))
                    stats["name_legal"] += int(norm.get("has_legal", False))
                    stats["name_url"] += int(norm.get("has_url", False))
                    scripts[dominant_script(value)] += 1
                    stats["name_mixed_script"] += int(is_mixed_script(value))
                    sample("non_latin_name", value) if dominant_script(value) != "Latin" else None
                else:
                    stats["address_postal"] += int(bool(norm.get("postal")))
                    stats["address_house_number"] += int(bool(norm.get("house_num")))
                    stats["address_digits"] += int(bool(norm.get("digit_tokens")))
                stats[f"{field}_unicode_non_ascii"] += int(any(ord(c) > 127 for c in value))
                stats[f"{field}_punctuation"] += int(any(unicodedata.category(c).startswith("P") for c in value))
                stats[f"{field}_zero_width"] += int(any(ord(c) in {0x200B, 0x200C, 0x200D, 0xFEFF} for c in value))

    result = {
        "path": str(path),
        "rows": rows,
        "stats": dict(stats),
        "scripts": dict(scripts),
        "countries": dict(countries),
        "duplicate_name_values": sum(count > 1 for count in names.values()),
        "duplicate_address_values": sum(bool(key) and count > 1 for key, count in addresses.items()),
        "top_names": names.most_common(20),
        "top_addresses": addresses.most_common(20),
        "samples": samples,
    }
    return result


def audit_directory(train_dir: str | Path, chunksize: int = 100_000) -> dict:
    train_dir = Path(train_dir)
    return {
        "source1": audit_file(train_dir / "train_source1.tsv", chunksize),
        "source2": audit_file(train_dir / "train_source2.tsv", chunksize),
        "source3": audit_file(train_dir / "train_source3.tsv", chunksize),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit training source files")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output", default="audit.json")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    result = audit_directory(args.train_dir, args.chunksize)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    for source, report in result.items():
        print(f"{source}: {report['rows']:,} rows, {report['stats']}")


if __name__ == "__main__":
    main()
