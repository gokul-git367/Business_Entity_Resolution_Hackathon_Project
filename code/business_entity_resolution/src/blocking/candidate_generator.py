"""
Multi-route blocking candidate generator.
Builds an inverted index for each blocking strategy and returns a union of candidates.

Scale: S2+S3 together = ~10M records.
All indexes are built in memory using Python dicts (fast lookup).
For very large datasets, we use chunked building with hashing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

import pandas as pd

from ..normalization.names import normalize_name, char_ngrams, get_rare_tokens
from ..normalization.addresses import normalize_address
from ..normalization.transliteration import transliterate
from ..normalization.scripts import dominant_script


# ── helpers ────────────────────────────────────────────────────────────────────

def _clean_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _safe(val):
    """Return empty string for NaN/None."""
    if val is None:
        return ""
    v = str(val).strip()
    return "" if v.lower() in {"nan", "none", "null", "n/a", "na"} else v


# ── Inverted index builder ─────────────────────────────────────────────────────

class InvertedIndex:
    """Maps key -> set of record IDs."""

    def __init__(self):
        self._idx: dict[str, set[str]] = defaultdict(set)

    def add(self, key: str, record_id: str):
        if key:
            self._idx[key].add(record_id)

    def query(self, key: str) -> set[str]:
        if not key:
            return set()
        return self._idx.get(key, set())

    def query_many(self, keys: list[str]) -> set[str]:
        result: set[str] = set()
        for k in keys:
            result |= self.query(k)
        return result

    def __len__(self):
        return len(self._idx)


# ── Individual blocking indexes ────────────────────────────────────────────────

class CandidateGenerator:
    """
    Builds multiple blocking indexes over S2+S3 records.
    For a given S1 record, generates the union of candidates across all routes.
    """

    def __init__(
        self,
        ngram_n: int = 3,
        top_ngrams: int = 5,
        rare_token_max_df: int = 500,   # tokens appearing in more records are too common
    ):
        self.ngram_n = ngram_n
        self.top_ngrams = top_ngrams
        self.rare_token_max_df = rare_token_max_df

        # --- indexes ---
        self._idx_core_name = InvertedIndex()        # A: exact core name + country
        self._idx_rare_token = InvertedIndex()        # B: rare name tokens
        self._idx_ngram = InvertedIndex()             # C: char n-grams of core name
        self._idx_locality = InvertedIndex()          # D: country+city
        self._idx_postal = InvertedIndex()            # E: postal code + country
        self._idx_translit = InvertedIndex()          # F: transliterated core name
        self._idx_translit_ngram = InvertedIndex()    # F2: n-grams of transliterated name

        # token document frequency for rare-token filtering
        self._token_df: dict[str, int] = defaultdict(int)
        self._built = False

    # ── Build phase ────────────────────────────────────────────────────────────

    def _process_record(self, row: pd.Series):
        """Process one S2/S3 record and add to all indexes."""
        eid = row["entity_id"]
        name = _safe(row.get("business_name", ""))
        addr = _safe(row.get("business_address", ""))
        country = _safe(row.get("country", "")).lower()[:2]   # 'us', 'in', 'fr'

        nm = normalize_name(name)
        ad = normalize_address(addr)

        core = nm.get("core", "")
        tokens = nm.get("tokens", [])

        # A: exact core name
        if core:
            self._idx_core_name.add(f"{country}|{core}", eid)
            self._idx_core_name.add(core, eid)   # without country for safety

        # B: token -> count (built first pass) — actual index built in finalize
        for t in tokens:
            self._token_df[t] += 1

        # C: character n-grams of core name
        if core:
            ngs = char_ngrams(core, self.ngram_n)
            for ng in ngs[: self.top_ngrams * 4]:   # limit per record
                self._idx_ngram.add(ng, eid)

        # D: locality (country + city approximation from address tokens)
        addr_tokens = ad.get("tokens", [])
        postal = ad.get("postal", "")
        if len(addr_tokens) >= 1:
            # Use last 1-2 significant tokens as city-like key
            city_key = " ".join(addr_tokens[-2:]) if len(addr_tokens) >= 2 else addr_tokens[-1]
            self._idx_locality.add(f"{country}|{city_key}", eid)

        # E: postal
        if postal:
            self._idx_postal.add(f"{country}|{postal}", eid)
            self._idx_postal.add(postal, eid)

        # F: transliteration
        script = dominant_script(name)
        if script not in ("Latin", "Unknown", "Symbols", "Combining"):
            translit_name = transliterate(core) if core else ""
        else:
            translit_name = core
        if translit_name:
            self._idx_translit.add(translit_name, eid)
            self._idx_translit.add(f"{country}|{translit_name}", eid)
            t_ngs = char_ngrams(translit_name, self.ngram_n)
            for ng in t_ngs[: self.top_ngrams * 3]:
                self._idx_translit_ngram.add(ng, eid)

        return tokens, eid   # returned for second-pass rare-token indexing

    def build(self, s2: pd.DataFrame, s3: pd.DataFrame):
        """Build all indexes from Source 2 and Source 3."""
        print("[Blocking] Building indexes from S2+S3 ...")
        combined = pd.concat([s2, s3], ignore_index=True)

        # First pass: count token df and compute all except rare-token index
        token_to_eids: dict[str, list[str]] = defaultdict(list)

        for _, row in combined.iterrows():
            try:
                tokens, eid = self._process_record(row)
                for t in tokens:
                    token_to_eids[t].append(eid)
            except Exception:
                pass

        # Second pass: build rare-token index
        print(f"[Blocking] {len(self._token_df)} unique name tokens; "
              f"filtering to df<={self.rare_token_max_df} ...")
        for token, eids in token_to_eids.items():
            if self._token_df.get(token, 0) <= self.rare_token_max_df:
                for eid in eids:
                    self._idx_rare_token.add(token, eid)

        self._built = True
        print(f"[Blocking] Indexes built: "
              f"core={len(self._idx_core_name)}, "
              f"ngram={len(self._idx_ngram)}, "
              f"rare_tok={len(self._idx_rare_token)}, "
              f"locality={len(self._idx_locality)}, "
              f"postal={len(self._idx_postal)}, "
              f"translit={len(self._idx_translit)}")

    # ── Query phase ────────────────────────────────────────────────────────────

    def query(self, row: pd.Series) -> set[str]:
        """Return union of candidates for one S1 record."""
        assert self._built, "Call build() first"

        name = _safe(row.get("business_name", ""))
        addr = _safe(row.get("business_address", ""))
        country = _safe(row.get("country", "")).lower()[:2]

        nm = normalize_name(name)
        ad = normalize_address(addr)

        core = nm.get("core", "")
        tokens = nm.get("tokens", [])
        postal = ad.get("postal", "")
        addr_tokens = ad.get("tokens", [])

        candidates: set[str] = set()

        # A: exact core name
        if core:
            candidates |= self._idx_core_name.query(f"{country}|{core}")
            candidates |= self._idx_core_name.query(core)

        # B: rare tokens
        rare = get_rare_tokens(tokens)
        candidates |= self._idx_rare_token.query_many(rare)

        # C: n-grams (use top discriminative ones)
        if core:
            ngs = char_ngrams(core, self.ngram_n)
            # query only a subset to keep candidate set manageable
            for ng in ngs[: self.top_ngrams]:
                candidates |= self._idx_ngram.query(ng)

        # D: locality
        if len(addr_tokens) >= 1:
            city_key = " ".join(addr_tokens[-2:]) if len(addr_tokens) >= 2 else addr_tokens[-1]
            candidates |= self._idx_locality.query(f"{country}|{city_key}")

        # E: postal
        if postal:
            candidates |= self._idx_postal.query(f"{country}|{postal}")
            candidates |= self._idx_postal.query(postal)

        # F: transliteration
        script = dominant_script(name)
        if script not in ("Latin", "Unknown", "Symbols", "Combining"):
            translit_name = transliterate(core) if core else ""
        else:
            translit_name = core
        if translit_name:
            candidates |= self._idx_translit.query(translit_name)
            candidates |= self._idx_translit.query(f"{country}|{translit_name}")
            t_ngs = char_ngrams(translit_name, self.ngram_n)
            for ng in t_ngs[: self.top_ngrams]:
                candidates |= self._idx_translit_ngram.query(ng)

        return candidates

    def generate_candidates(
        self, s1: pd.DataFrame, max_candidates_per_s1: int = 300
    ) -> dict[str, set[str]]:
        """
        For every S1 entity, return its candidate set.
        Caps candidate set at max_candidates_per_s1 (score-ranked cap not feasible
        without scoring — we rely on pre-filtering later instead).
        """
        assert self._built

        result: dict[str, set[str]] = {}
        total = len(s1)

        for i, (_, row) in enumerate(s1.iterrows()):
            if i % 50000 == 0:
                print(f"[Candidates] {i}/{total} ...")
            cands = self.query(row)
            # Cap: if too large, keep up to the limit (arbitrary but needed for memory)
            if len(cands) > max_candidates_per_s1:
                cands = set(list(cands)[:max_candidates_per_s1])
            result[row["entity_id"]] = cands

        return result
