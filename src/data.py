"""Data loaders for the prototype.

Prototype data (all downloadable without special access):
  * language ID : NLTK UDHR corpus (Universal Declaration of Human Rights, 16 languages)
                  + a small hand-written romanized-Hindi ("Hinglish") seed file
  * spam        : UCI SMS Spam Collection (English)
  * sentiment   : NLTK movie_reviews (English, 2,000 reviews, binary)
Final-project plan replaces these with larger datasets (see docs/RESEARCH_PLAYBOOK.md).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# language code -> (UDHR file name, encoding)
UDHR_FILES = {
    "en": ("English-Latin1", "latin-1"),
    "fr": ("French_Francais-Latin1", "latin-1"),
    "de": ("German_Deutsch-Latin1", "latin-1"),
    "es": ("Spanish_Espanol-Latin1", "latin-1"),
    "it": ("Italian_Italiano-Latin1", "latin-1"),
    "pt": ("Portuguese_Portugues-Latin1", "latin-1"),
    "nl": ("Dutch_Nederlands-Latin1", "latin-1"),
    "pl": ("Polish_Polski-Latin2", "iso-8859-2"),
    "sw": ("Swahili_Kiswahili-Latin1", "latin-1"),
    "tr": ("Turkish_Turkce-UTF8", "utf-8"),
    "vi": ("Vietnamese-UTF8", "utf-8"),
    "ru": ("Russian_Russky-UTF8", "utf-8"),
    "el": ("Greek_Ellinika-UTF8", "utf-8"),
    "hi": ("Hindi-UTF8", "utf-8"),
    "ja": ("Japanese_Nihongo-UTF8", "utf-8"),
    "zh": ("Chinese_Mandarin-UTF8", "utf-8"),
}
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "sw": "Swahili", "tr": "Turkish",
    "vi": "Vietnamese", "ru": "Russian", "el": "Greek", "hi": "Hindi (Devanagari)",
    "ja": "Japanese", "zh": "Chinese", "hinglish": "Hinglish (romanized Hindi + English)",
}


def _chunks(text: str, rng: np.random.Generator, lo: int = 25, hi: int = 120):
    """Cut a document into short, message-like pieces (25-120 characters)."""
    text = re.sub(r"\s+", " ", text).strip()
    has_spaces = text.count(" ") > len(text) / 15
    out, pos = [], 0
    while pos < len(text):
        end = pos + int(rng.integers(lo, hi))
        if has_spaces and end < len(text):          # finish the current word
            nxt = text.find(" ", end)
            end = len(text) if nxt == -1 else nxt
        piece = text[pos:end].strip()
        if len(piece) >= 15:
            out.append(piece)
        pos = end
    return out


def load_language(seed: int = 42, train_frac: float = 0.7):
    """Document-order split: first 70% of each language's pieces train, rest test.
    (Avoids putting neighbouring pieces of the same passage on both sides.)"""
    rng = np.random.default_rng(seed)
    rows = []
    for code, (fname, enc) in UDHR_FILES.items():
        raw = (RAW / "udhr" / fname).read_text(encoding=enc, errors="ignore")
        pieces = _chunks(raw, rng)
        cut = int(len(pieces) * train_frac)
        rows += [(p, code, "train" if i < cut else "test") for i, p in enumerate(pieces)]
    seed_lines = [l.strip() for l in (ROOT / "data" / "hinglish_seed.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    cut = int(len(seed_lines) * train_frac)
    rows += [(l, "hinglish", "train" if i < cut else "test") for i, l in enumerate(seed_lines)]
    df = pd.DataFrame(rows, columns=["text", "label", "split"])
    return df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)


def load_spam(seed: int = 42):
    df = pd.read_csv(RAW / "sms.tsv", sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    n0 = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)   # dedupe BEFORE splitting
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True), n0 - len(df)


def load_sentiment(seed: int = 42):
    rows = []
    for lab in ("pos", "neg"):
        for f in sorted((RAW / "movie_reviews" / lab).glob("*.txt")):
            rows.append((f.read_text(encoding="latin-1"), lab))
    df = pd.DataFrame(rows, columns=["text", "label"])
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
