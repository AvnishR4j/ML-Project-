"""Spam-evasion helpers: attacks (for testing) and a normalizer (defence).

Attacks follow the evasion types studied in the 2022 SMS-scam paper (character spacing, homoglyphs).
"""
import re
import unicodedata

# Latin letter -> visually identical Cyrillic/Ukrainian letter
HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
REVERSE = {v: k for k, v in HOMOGLYPH.items()}


def attack_spacing(text: str, min_len: int = 4) -> str:
    """'free prize now' -> 'f r e e  p r i z e  now' (letters spaced, words separated by 2 spaces)."""
    words = text.split(" ")
    out = [" ".join(w) if (w.isalpha() and len(w) >= min_len) else w for w in words]
    return "  ".join(out)


def attack_homoglyph(text: str, min_len: int = 4) -> str:
    def conv(w):
        return "".join(HOMOGLYPH.get(ch, ch) for ch in w) if (w.isalpha() and len(w) >= min_len) else w
    return " ".join(conv(w) for w in text.split(" "))


def fix_spacing(text: str) -> str:
    """Re-join letter-spaced words ('f r e e' -> 'free'). Safe for any language."""
    t = unicodedata.normalize("NFKC", text)
    segs = re.split(r" {2,}", t)                       # words in the attack are separated by 2+ spaces
    fixed = []
    for seg in segs:
        toks = seg.split(" ")
        if len(toks) >= 3 and all(len(x) == 1 for x in toks):   # 'f r e e' -> 'free'
            fixed.append("".join(toks))
        else:
            fixed.append(seg)
    return " ".join(fixed)


def normalize(text: str) -> str:
    """Spam defence: map look-alike letters back to Latin, then re-join letter-spaced words.
    (Only used on the English/Hinglish spam route - it would corrupt genuine Russian/Ukrainian text.)"""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(REVERSE.get(ch, ch) for ch in t)
    return fix_spacing(t)
