"""
Business name normalization.
Creates multiple representations; never overwrites the original.
"""

from __future__ import annotations

import re
import unicodedata
from .unicode import normalize_unicode, is_placeholder

# ── Legal suffix patterns ──────────────────────────────────────────────────────
_LEGAL_SUFFIXES = [
    # longer first so they match before shorter subsets
    r"private\s+limited",
    r"public\s+limited",
    r"limited\s+liability\s+company",
    r"llc",
    r"l\.l\.c\.?",
    r"l\.l\.p\.?",
    r"pvt\.?\s*ltd\.?",
    r"pvt\.?",
    r"p\.?v\.?t\.?",
    r"ltd\.?",
    r"limited",
    r"inc\.?",
    r"incorporated",
    r"corp\.?",
    r"corporation",
    r"co\.?",
    r"lp\.?",
    r"llp\.?",
    r"plc\.?",
    r"gmbh",
    r"s\.?a\.?",
    r"s\.?\s*r\.?\s*l\.?",
    r"bv",
    r"nv",
    r"ag",
    r"kg",
]

_LEGAL_RE = re.compile(
    r"\b(" + "|".join(_LEGAL_SUFFIXES) + r")\b[\.\,]?",
    re.IGNORECASE,
)

# ── DBA / alias markers ────────────────────────────────────────────────────────
_DBA_RE = re.compile(
    r"\b(d/?b/?a|doing\s+business\s+as|dba|fka|formerly\s+known\s+as|also\s+known\s+as|aka)\b",
    re.IGNORECASE,
)

# ── Punctuation / symbols ──────────────────────────────────────────────────────
_AND_RE = re.compile(r"\s*&\s*", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")

# ── URL / domain ───────────────────────────────────────────────────────────────
_URL_RE = re.compile(
    r"(https?://|www\.)\S+|[\w.-]+\.(com|org|net|in|co\.in|io|biz|info)\b",
    re.IGNORECASE,
)

# ── Generic Indian/global stop words in business names ────────────────────────
_GENERIC_TOKENS = frozenset(
    [
        "services", "service", "trading", "traders", "trader", "enterprises",
        "enterprise", "company", "solutions", "solution", "international",
        "india", "group", "industries", "industry", "store", "stores",
        "market", "markets", "foods", "food", "agency", "agencies",
        "consultant", "consultants", "consultancy", "associates",
        "association", "global", "national", "management",
        "products", "product", "works", "business",
    ]
)


def _strip_legal(text: str) -> str:
    """Remove legal suffixes from text (used for core name)."""
    # First pass: remove full DBA tail (everything after the marker)
    m = _DBA_RE.search(text)
    if m:
        text = text[: m.start()].strip()
    return _LEGAL_RE.sub(" ", text).strip()


def _strip_punctuation(text: str) -> str:
    """Replace punctuation/symbols while preserving Unicode grapheme marks."""
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(("L", "M", "N")) or category.isspace():
            chars.append(char)
        else:
            chars.append(" ")
    return "".join(chars)


def normalize_name(name: str) -> dict:
    """
    Return a dict with multiple name representations.
    Keys:
        raw          - original
        unicode      - NFC cleaned
        clean        - lowercase, normalised, without legal suffixes
        clean_nopunct- clean with punctuation removed
        core         - lowercase, no legal suffix, no punct
        tokens       - list of core tokens
        has_dba      - bool
        has_legal    - bool
        has_url      - bool
        is_missing   - bool
    """
    result: dict = {"raw": name}

    if is_placeholder(name):
        empty = {
            "raw": name, "unicode": "", "clean": "", "clean_nopunct": "",
            "core": "", "tokens": [], "has_dba": False,
            "has_legal": False, "has_url": False, "is_missing": True,
        }
        return empty

    result["is_missing"] = False
    result["has_url"] = bool(_URL_RE.search(name))

    uni = normalize_unicode(name)
    result["unicode"] = uni

    # Detect DBA / FKA before stripping
    result["has_dba"] = bool(_DBA_RE.search(uni))
    result["has_legal"] = bool(_LEGAL_RE.search(uni))

    # Clean: lowercase, normalise &→and, strip legal
    clean = uni.lower()
    clean = _AND_RE.sub(" and ", clean)
    clean = _strip_legal(clean)
    result["clean"] = _MULTI_SPACE.sub(" ", clean).strip()

    # No-punct version: remove ASCII/Latin punctuation but keep native-script chars
    no_punct = _strip_punctuation(result["clean"])
    result["clean_nopunct"] = _MULTI_SPACE.sub(" ", no_punct).strip()

    # Core: use NFC (not NFKC) so Devanagari/other conjuncts are preserved intact
    core = unicodedata.normalize("NFC", result["clean_nopunct"])
    result["core"] = core

    tokens = [t for t in core.split() if len(t) >= 2]
    result["tokens"] = tokens

    return result


def get_rare_tokens(tokens: list[str], generic: frozenset = _GENERIC_TOKENS) -> list[str]:
    """Filter out generic corporate tokens."""
    return [t for t in tokens if t not in generic]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams of a string (padding optional)."""
    padded = f"#{text}#"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]
