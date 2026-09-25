"""
Pair feature engineering.
Computes a fixed-length numeric feature vector for each (S1, candidate) pair.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from ..normalization.names import normalize_name, char_ngrams, _GENERIC_TOKENS
from ..normalization.addresses import normalize_address
from ..normalization.transliteration import transliterate
from ..normalization.scripts import dominant_script, is_mixed_script


# ── Similarity helpers ─────────────────────────────────────────────────────────

def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _weighted_token_overlap(
    a_tokens: list[str],
    b_tokens: list[str],
    idf: dict[str, float],
) -> float:
    """Weighted Jaccard using IDF weights."""
    sa, sb = set(a_tokens), set(b_tokens)
    inter = sa & sb
    union = sa | sb
    if not union:
        return 1.0
    w_inter = sum(idf.get(t, 1.0) for t in inter)
    w_union = sum(idf.get(t, 1.0) for t in union)
    return w_inter / w_union if w_union > 0 else 0.0


def _ngram_sim(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ngs_a = set(char_ngrams(a, n))
    ngs_b = set(char_ngrams(b, n))
    inter = len(ngs_a & ngs_b)
    union = len(ngs_a | ngs_b)
    return inter / union if union > 0 else 0.0


def _safe_fuzz(fn, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fn(a, b) / 100.0


# ── Main feature function ──────────────────────────────────────────────────────

def build_pair_features(
    s1_row: dict,
    cand_row: dict,
    name_idf: dict[str, float] | None = None,
    addr_idf: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute all pair features for one (S1, candidate) pair.
    Both inputs are dicts with at minimum:
        entity_id, business_name, business_address, country

    name_idf / addr_idf: optional token->idf dicts for weighted features.
    """
    name_idf = name_idf or {}
    addr_idf = addr_idf or {}

    feats: dict[str, float] = {}

    # ── 1. Normalize ──────────────────────────────────────────────────────────
    n1 = normalize_name(str(s1_row.get("business_name", "") or ""))
    n2 = normalize_name(str(cand_row.get("business_name", "") or ""))
    a1 = normalize_address(str(s1_row.get("business_address", "") or ""))
    a2 = normalize_address(str(cand_row.get("business_address", "") or ""))

    country1 = str(s1_row.get("country", "") or "").strip().lower()[:2]
    country2 = str(cand_row.get("country", "") or "").strip().lower()[:2]

    # ── 2. Name features ──────────────────────────────────────────────────────
    raw1, raw2 = n1.get("raw", ""), n2.get("raw", "")
    uni1, uni2 = n1.get("unicode", ""), n2.get("unicode", "")
    clean1, clean2 = n1.get("clean", ""), n2.get("clean", "")
    core1, core2 = n1.get("core", ""), n2.get("core", "")
    toks1, toks2 = n1.get("tokens", []), n2.get("tokens", [])
    miss1, miss2 = n1.get("is_missing", False), n2.get("is_missing", False)

    feats["name_miss_s1"] = float(miss1)
    feats["name_miss_s2"] = float(miss2)

    feats["name_exact_raw"] = float(raw1.strip().lower() == raw2.strip().lower())
    feats["name_exact_clean"] = float(clean1 == clean2)
    feats["name_exact_core"] = float(core1 == core2)

    feats["name_lev"] = _safe_fuzz(fuzz.ratio, core1, core2)
    feats["name_lev_clean"] = _safe_fuzz(fuzz.ratio, clean1, clean2)
    feats["name_jw"] = JaroWinkler.normalized_similarity(core1, core2) if core1 and core2 else 0.0
    feats["name_partial"] = _safe_fuzz(fuzz.partial_ratio, core1, core2)
    feats["name_token_sort"] = _safe_fuzz(fuzz.token_sort_ratio, core1, core2)
    feats["name_token_set"] = _safe_fuzz(fuzz.token_set_ratio, core1, core2)

    feats["name_ngram3"] = _ngram_sim(core1, core2, 3)
    feats["name_ngram2"] = _ngram_sim(core1, core2, 2)
    feats["name_jaccard"] = _jaccard(toks1, toks2)
    feats["name_weighted_jaccard"] = _weighted_token_overlap(toks1, toks2, name_idf)

    feats["name_len_diff"] = abs(len(core1) - len(core2)) / (max(len(core1), len(core2)) + 1)
    feats["name_tok_cnt_s1"] = float(len(toks1))
    feats["name_tok_cnt_s2"] = float(len(toks2))

    # Rare token overlap (excluding generic)
    rare1 = [t for t in toks1 if t not in _GENERIC_TOKENS]
    rare2 = [t for t in toks2 if t not in _GENERIC_TOKENS]
    feats["name_rare_jaccard"] = _jaccard(rare1, rare2)
    feats["name_has_dba_s1"] = float(n1.get("has_dba", False))
    feats["name_has_dba_s2"] = float(n2.get("has_dba", False))
    feats["name_has_legal_s1"] = float(n1.get("has_legal", False))
    feats["name_has_legal_s2"] = float(n2.get("has_legal", False))
    feats["name_has_url_s1"] = float(n1.get("has_url", False))
    feats["name_has_url_s2"] = float(n2.get("has_url", False))

    # ── 3. Transliteration features ───────────────────────────────────────────
    def _translit_core(nm_dict, raw_name):
        sc = dominant_script(raw_name)
        if sc not in ("Latin", "Unknown", "Symbols", "Combining"):
            return transliterate(nm_dict.get("core", ""))
        return nm_dict.get("core", "")

    t1 = _translit_core(n1, raw1)
    t2 = _translit_core(n2, raw2)

    feats["translit_lev"] = _safe_fuzz(fuzz.ratio, t1, t2)
    feats["translit_jw"] = JaroWinkler.normalized_similarity(t1, t2) if t1 and t2 else 0.0
    feats["translit_ngram3"] = _ngram_sim(t1, t2, 3)
    feats["translit_exact"] = float(t1 == t2 and bool(t1))

    # Cross: raw vs transliterated
    feats["cross_lev_1t2"] = _safe_fuzz(fuzz.ratio, core1, t2)
    feats["cross_lev_2t1"] = _safe_fuzz(fuzz.ratio, core2, t1)

    # ── 4. Script features ────────────────────────────────────────────────────
    feats["script_dom_match"] = float(dominant_script(raw1) == dominant_script(raw2))
    feats["name_mixed_s1"] = float(is_mixed_script(raw1))
    feats["name_mixed_s2"] = float(is_mixed_script(raw2))

    # ── 5. Address features ───────────────────────────────────────────────────
    a_miss1 = a1.get("is_missing", False)
    a_miss2 = a2.get("is_missing", False)
    ac1, ac2 = a1.get("clean", ""), a2.get("clean", "")
    at1, at2 = a1.get("tokens", []), a2.get("tokens", [])
    postal1, postal2 = a1.get("postal", ""), a2.get("postal", "")
    house1, house2 = a1.get("house_num", ""), a2.get("house_num", "")
    dt1, dt2 = a1.get("digit_tokens", []), a2.get("digit_tokens", [])

    feats["addr_miss_s1"] = float(a_miss1)
    feats["addr_miss_s2"] = float(a_miss2)
    feats["addr_both_present"] = float(not a_miss1 and not a_miss2)

    feats["addr_lev"] = _safe_fuzz(fuzz.ratio, ac1, ac2) if not (a_miss1 or a_miss2) else 0.0
    feats["addr_partial"] = _safe_fuzz(fuzz.partial_ratio, ac1, ac2) if not (a_miss1 or a_miss2) else 0.0
    feats["addr_token_set"] = _safe_fuzz(fuzz.token_set_ratio, ac1, ac2) if not (a_miss1 or a_miss2) else 0.0
    feats["addr_jaccard"] = _jaccard(at1, at2) if not (a_miss1 or a_miss2) else 0.0
    feats["addr_weighted_jaccard"] = _weighted_token_overlap(at1, at2, addr_idf) if not (a_miss1 or a_miss2) else 0.0
    feats["addr_ngram3"] = _ngram_sim(ac1, ac2, 3) if not (a_miss1 or a_miss2) else 0.0

    # Postal/PIN
    if postal1 and postal2:
        feats["postal_exact"] = float(postal1 == postal2)
        feats["postal_present"] = 1.0
    elif postal1 or postal2:
        feats["postal_exact"] = 0.0
        feats["postal_present"] = 1.0
    else:
        feats["postal_exact"] = 0.0
        feats["postal_present"] = 0.0

    # House number
    if house1 and house2:
        feats["house_exact"] = float(house1 == house2)
        feats["house_present"] = 1.0
    else:
        feats["house_exact"] = 0.0
        feats["house_present"] = 0.0

    # Digit token overlap
    feats["digit_jaccard"] = _jaccard(dt1, dt2)

    # ── 6. Contradiction features ─────────────────────────────────────────────
    n_sim = feats["name_token_set"]
    a_sim = feats["addr_jaccard"]
    feats["strong_name_weak_addr"] = float(n_sim > 0.8 and a_sim < 0.2 and feats["addr_both_present"])
    feats["strong_addr_weak_name"] = float(a_sim > 0.8 and n_sim < 0.2)
    feats["both_strong"] = float(n_sim > 0.7 and a_sim > 0.7)
    feats["both_weak"] = float(n_sim < 0.3 and a_sim < 0.3)
    feats["postal_conflict"] = float(bool(
        postal1 and postal2 and postal1 != postal2 and
        len(postal1) == len(postal2)  # only flag exact-length mismatches
    ))

    # ── 7. Context features ───────────────────────────────────────────────────
    feats["country_match"] = float(country1 == country2 and bool(country1))
    feats["cand_is_s2"] = float(str(cand_row.get("entity_id", "")).startswith("S2-"))
    feats["cand_is_s3"] = float(str(cand_row.get("entity_id", "")).startswith("S3-"))

    return feats


FEATURE_NAMES: list[str] = [
    "name_miss_s1", "name_miss_s2",
    "name_exact_raw", "name_exact_clean", "name_exact_core",
    "name_lev", "name_lev_clean", "name_jw", "name_partial",
    "name_token_sort", "name_token_set",
    "name_ngram3", "name_ngram2",
    "name_jaccard", "name_weighted_jaccard",
    "name_len_diff", "name_tok_cnt_s1", "name_tok_cnt_s2",
    "name_rare_jaccard",
    "name_has_dba_s1", "name_has_dba_s2",
    "name_has_legal_s1", "name_has_legal_s2",
    "name_has_url_s1", "name_has_url_s2",
    "translit_lev", "translit_jw", "translit_ngram3", "translit_exact",
    "cross_lev_1t2", "cross_lev_2t1",
    "script_dom_match", "name_mixed_s1", "name_mixed_s2",
    "addr_miss_s1", "addr_miss_s2", "addr_both_present",
    "addr_lev", "addr_partial", "addr_token_set",
    "addr_jaccard", "addr_weighted_jaccard", "addr_ngram3",
    "postal_exact", "postal_present",
    "house_exact", "house_present",
    "digit_jaccard",
    "strong_name_weak_addr", "strong_addr_weak_name",
    "both_strong", "both_weak", "postal_conflict",
    "country_match", "cand_is_s2", "cand_is_s3",
]
