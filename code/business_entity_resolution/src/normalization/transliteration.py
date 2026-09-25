"""
Simple transliteration helper.
Provides basic Latin-to-Latin and common-Indian-script romanization for blocking.
We use a character-level mapping approach — no external library required.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari -> Latin approximate mapping
_DEVA_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "am", "अः": "ah",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क्ष": "ksh",
    "ज्ञ": "gn", "़": "", "ा": "a", "ि": "i", "ी": "ii",
    "ु": "u", "ू": "uu", "े": "e", "ै": "ai", "ो": "o",
    "ौ": "au", "ं": "m", "ः": "h", "्": "",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

# Bengali -> Latin approximate
_BENG_MAP = {
    "অ": "a", "আ": "aa", "ই": "i", "ী": "ii", "উ": "u", "ূ": "uu",
    "এ": "e", "ঐ": "ai", "ও": "o", "ঔ": "au",
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "y", "র": "r", "ল": "l", "শ": "sh", "স": "s", "হ": "h",
    "া": "a", "ি": "i", "ে": "e", "ো": "o", "ৌ": "au", "ং": "ng",
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}

# Tamil -> Latin approximate
_TAMIL_MAP = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu",
    "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au",
    "க": "k", "ங": "ng", "ச": "ch", "ஞ": "ny", "ட": "t", "ண": "n",
    "த": "th", "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r",
    "ல": "l", "வ": "v", "ழ": "zh", "ள": "ll", "ற": "r", "ன": "n",
    "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h",
    "ா": "aa", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu",
    "ெ": "e", "ே": "ee", "ை": "ai", "ொ": "o", "ோ": "oo", "ௌ": "au",
}

# Telugu -> Latin approximate
_TELUGU_MAP = {
    "అ": "a", "ఆ": "aa", "ఇ": "i","ఈ":"ii","ఉ":"u","ఊ":"uu",
    "ఎ":"e","ఏ":"ee","ఐ":"ai","ఒ":"o","ఓ":"oo","ఔ":"au",
    "క":"k","ఖ":"kh","గ":"g","ఘ":"gh","చ":"ch","జ":"j",
    "ట":"t","డ":"d","త":"th","ద":"dh","న":"n","ప":"p",
    "ఫ":"ph","బ":"b","భ":"bh","మ":"m","య":"y","ర":"r",
    "ల":"l","వ":"v","శ":"sh","ష":"sh","స":"s","హ":"h",
    "ళ":"ll","ా":"aa","ి":"i","ీ":"ii","ు":"u","ూ":"uu",
    "ె":"e","ే":"ee","ై":"ai","ొ":"o","ో":"oo","ం":"m","ః":"h",
}

# Gujarati -> Latin approximate
_GUJARATI_MAP = {
    "અ":"a","આ":"aa","ઇ":"i","ઈ":"ii","ઉ":"u","ઊ":"uu",
    "એ":"e","ઐ":"ai","ઓ":"o","ઔ":"au",
    "ક":"k","ખ":"kh","ગ":"g","ઘ":"gh","ચ":"ch","છ":"chh","જ":"j",
    "ટ":"t","ઠ":"th","ડ":"d","ઢ":"dh","ત":"t","થ":"th","દ":"d",
    "ધ":"dh","ન":"n","પ":"p","ફ":"ph","બ":"b","ભ":"bh","મ":"m",
    "ય":"y","ર":"r","લ":"l","વ":"v","શ":"sh","સ":"s","હ":"h",
    "ા":"a","િ":"i","ી":"ii","ુ":"u","ૂ":"uu","ે":"e","ૈ":"ai",
    "ો":"o","ૌ":"au","ં":"m",
}

# Kannada -> Latin approximate
_KANNADA_MAP = {
    "ಅ":"a","ಆ":"aa","ಇ":"i","ಈ":"ii","ಉ":"u","ಊ":"uu",
    "ಎ":"e","ಏ":"ee","ಐ":"ai","ಒ":"o","ಓ":"oo","ಔ":"au",
    "ಕ":"k","ಖ":"kh","ಗ":"g","ಘ":"gh","ಚ":"ch","ಜ":"j",
    "ಟ":"t","ಡ":"d","ತ":"th","ದ":"dh","ನ":"n","ಪ":"p",
    "ಫ":"ph","ಬ":"b","ಭ":"bh","ಮ":"m","ಯ":"y","ರ":"r",
    "ಲ":"l","ವ":"v","ಶ":"sh","ಷ":"sh","ಸ":"s","ಹ":"h",
    "ಳ":"ll","ಾ":"aa","ಿ":"i","ೀ":"ii","ು":"u","ೂ":"uu",
    "ೆ":"e","ೇ":"ee","ೈ":"ai","ೊ":"o","ೋ":"oo","ಂ":"m",
}

# Malayalam -> Latin approximate
_MALAYALAM_MAP = {
    "അ":"a","ആ":"aa","ഇ":"i","ഈ":"ii","ഉ":"u","ഊ":"uu",
    "എ":"e","ഏ":"ee","ഐ":"ai","ഒ":"o","ഓ":"oo","ഔ":"au",
    "ക":"k","ഖ":"kh","ഗ":"g","ഘ":"gh","ച":"ch","ജ":"j",
    "ട":"t","ഡ":"d","ത":"th","ദ":"dh","ന":"n","പ":"p",
    "ഫ":"ph","ബ":"b","ഭ":"bh","മ":"m","യ":"y","ര":"r",
    "ല":"l","വ":"v","ശ":"sh","ഷ":"sh","സ":"s","ഹ":"h",
    "ള":"ll","ാ":"aa","ി":"i","ീ":"ii","ു":"u","ൂ":"uu",
    "െ":"e","േ":"ee","ൈ":"ai","ൊ":"o","ോ":"oo","ം":"m",
}

_ALL_MAPS = [
    _DEVA_MAP, _BENG_MAP, _TAMIL_MAP, _TELUGU_MAP,
    _GUJARATI_MAP, _KANNADA_MAP, _MALAYALAM_MAP,
]

# Build combined map: char -> latin
_COMBINED_MAP: dict[str, str] = {}
for _m in _ALL_MAPS:
    _COMBINED_MAP.update(_m)


def transliterate(text: str) -> str:
    """
    Return a best-effort Latin transliteration of any Indian-script text.
    Latin characters are passed through unchanged.
    """
    result = []
    i = 0
    text_nfc = unicodedata.normalize("NFC", text)
    while i < len(text_nfc):
        # Try 2-char lookup first
        two = text_nfc[i : i + 2]
        if two in _COMBINED_MAP:
            result.append(_COMBINED_MAP[two])
            i += 2
            continue
        one = text_nfc[i]
        if one in _COMBINED_MAP:
            result.append(_COMBINED_MAP[one])
        else:
            result.append(one)  # keep Latin / digits / spaces
        i += 1
    out = "".join(result)
    return re.sub(r"\s+", " ", out).strip()
