"""
Script detection utilities.
Returns Unicode script block proportions for a field value.
"""

from __future__ import annotations

import unicodedata
from collections import Counter


# Broad Unicode range -> script name mapping (simplified but practical)
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0000, 0x007F, "Latin"),      # Basic Latin
    (0x0080, 0x024F, "Latin"),      # Latin Extended
    (0x0250, 0x02AF, "Latin"),      # IPA
    (0x0300, 0x036F, "Combining"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x1700, 0x1FFF, "Other_Asian"),
    (0x2000, 0x2FFF, "Symbols"),
    (0x3000, 0x9FFF, "CJK"),
    (0xAC00, 0xD7AF, "Korean"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
]


def _script_of(cp: int) -> str:
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return "Other"


def script_counts(text: str) -> Counter:
    """Return character-level script counts, ignoring spaces/digits/punctuation."""
    c: Counter = Counter()
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):  # Letters only
            c[_script_of(ord(ch))] += 1
    return c


def dominant_script(text: str) -> str:
    """Return the most frequent script, or 'Unknown' if no letters."""
    c = script_counts(text)
    if not c:
        return "Unknown"
    return c.most_common(1)[0][0]


def is_mixed_script(text: str) -> bool:
    """Return True if more than one script is present."""
    c = script_counts(text)
    # Ignore Combining (diacritics) as a separate language
    scripts = {k for k in c if k != "Combining"}
    return len(scripts) > 1


def script_feature_dict(text: str) -> dict:
    """Return a flat dict of script proportions and flags."""
    total_letters = sum(
        1 for ch in text if unicodedata.category(ch).startswith("L")
    )
    c = script_counts(text)
    if total_letters == 0:
        return {
            "dom_script": "Unknown",
            "is_mixed": False,
            "frac_latin": 0.0,
            "frac_devanagari": 0.0,
            "frac_other_script": 0.0,
        }
    latin = c.get("Latin", 0)
    deva = c.get("Devanagari", 0)
    other = total_letters - latin - deva
    dominated = c.most_common(1)[0][0] if c else "Unknown"
    scripts = {k for k in c if k != "Combining"}
    return {
        "dom_script": dominated,
        "is_mixed": len(scripts) > 1,
        "frac_latin": latin / total_letters,
        "frac_devanagari": deva / total_letters,
        "frac_other_script": other / total_letters,
    }
