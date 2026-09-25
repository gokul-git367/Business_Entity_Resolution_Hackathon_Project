"""
Address normalization.
Extracts postal codes, numbers, components, and creates clean representations.
"""

from __future__ import annotations

import re
from .unicode import normalize_unicode, is_placeholder

# ── Abbreviation expansions (safe, unambiguous only) ─────────────────────────
_ADDR_ABBREV = {
    r"\brd\b": "road",
    r"\bst\b": "street",   # careful: 'st' can also = 'saint'
    r"\bavenue\b": "avenue",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bh\.?no\.?\b": "house no",
    r"\bfl\b": "floor",
    r"\bpt\b": "plot",
    r"\bplt\b": "plot",
    r"\bsec\b": "sector",
    r"\bdist\b": "district",
    r"\bph\b": "phase",
    r"\bextn\b": "extension",
    r"\bextn\.\b": "extension",
    r"\bkolkata\b": "kolkata",
    r"\bcalcutta\b": "kolkata",
    r"\bbangalore\b": "bengaluru",
    r"\bbengaluru\b": "bengaluru",
    r"\bmumbai\b": "mumbai",
    r"\bbombay\b": "mumbai",
    r"\bchennai\b": "chennai",
    r"\bmadras\b": "chennai",
}

_ABBREV_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _ADDR_ABBREV.items()
]

# ── Postal/PIN code ───────────────────────────────────────────────────────────
# India PIN: 6 digits; US ZIP: 5 digits or 5+4; France: 5 digits
_PIN_RE = re.compile(r"\b(\d{5,6}(-\d{4})?)\b")

# ── House / unit / floor number extraction ────────────────────────────────────
_HOUSE_RE = re.compile(
    r"(?:no\.?|#|h\.?no\.?|plot\s*no\.?|d\.?no\.?|door\s*no\.?|flat\s*no\.?)"
    r"\s*([a-z0-9][-/a-z0-9]{0,15})",
    re.IGNORECASE,
)

# ── Digit token ───────────────────────────────────────────────────────────────
_DIGIT_TOKEN_RE = re.compile(r"\b\d[\d/-]{0,15}\b")

_MULTI_SPACE = re.compile(r"\s+")
_PUNCT_EXCEPT_DIGIT = re.compile(r"[^\w\s]", re.UNICODE)


def _expand_abbrevs(text: str) -> str:
    for pattern, replacement in _ABBREV_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def _extract_postal(text: str) -> str:
    m = _PIN_RE.search(text)
    return m.group(0) if m else ""


def _extract_house_num(text: str) -> str:
    m = _HOUSE_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_digit_tokens(text: str) -> list[str]:
    return _DIGIT_TOKEN_RE.findall(text)


def normalize_address(address: str) -> dict:
    """
    Return a dict with multiple address representations and extracted components.
    Keys:
        raw          - original
        unicode      - NFC cleaned
        clean        - lowercase, abbrevs expanded
        tokens       - word tokens
        digit_tokens - numeric tokens
        postal       - detected postal/PIN code
        house_num    - house/unit number
        is_missing   - bool
    """
    result: dict = {"raw": address}

    if is_placeholder(address):
        return {
            "raw": address, "unicode": "", "clean": "", "tokens": [],
            "digit_tokens": [], "postal": "", "house_num": "", "is_missing": True,
        }

    result["is_missing"] = False
    uni = normalize_unicode(address)
    result["unicode"] = uni

    result["postal"] = _extract_postal(uni)
    result["house_num"] = _extract_house_num(uni)
    result["digit_tokens"] = _extract_digit_tokens(uni)

    clean = uni.lower()
    clean = _expand_abbrevs(clean)
    clean = _MULT_SPACE_sub(clean)
    result["clean"] = clean.strip()

    tokens = [t for t in clean.split() if len(t) >= 2]
    result["tokens"] = tokens

    return result


def _MULT_SPACE_sub(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text)
