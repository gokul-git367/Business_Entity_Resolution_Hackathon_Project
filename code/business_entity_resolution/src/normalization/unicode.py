"""
Unicode normalization utilities.
Applies NFC/NFKC, removes invisible formatting characters,
normalises whitespace and special punctuation.
"""

from __future__ import annotations

import re
import unicodedata

# Invisible / zero-width / formatting code points to strip
_INVISIBLE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u2061\u2062\u2063\u2064]"
)

# Non-breaking / unusual spaces -> regular space
_UNUSUAL_SPACE = re.compile(
    r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
)

# Fancy dashes -> hyphen
_FANCY_DASH = re.compile(r"[\u2013\u2014\u2212]")

# Fancy quotes -> straight quote
_FANCY_QUOTE = re.compile(r"[\u2018\u2019\u201a\u201b]")
_FANCY_DQUOTE = re.compile(r'[\u201C\u201D\u201e\u201f]')

# Repeated whitespace
_MULTISPACE = re.compile(r"\s{2,}")


def normalize_unicode(text: str) -> str:
    """
    Apply NFC normalization, remove invisible chars, normalize unusual
    spaces and punctuation, and collapse repeated whitespace.
    Keeps the original language/script intact.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = _UNUSUAL_SPACE.sub(" ", text)
    text = _FANCY_DASH.sub("-", text)
    text = _FANCY_QUOTE.sub("'", text)
    text = _FANCY_DQUOTE.sub('"', text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def normalize_unicode_nfkc(text: str) -> str:
    """NFKC: also decomposes compatibility characters (e.g. fullwidth)."""
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _UNUSUAL_SPACE.sub(" ", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def is_placeholder(text: str) -> bool:
    """Return True if the text looks like a missing/placeholder value."""
    stripped = text.strip().lower()
    return stripped in {"", "null", "none", "n/a", "na", "unknown", "-", ".", "?", "x"}
