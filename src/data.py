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


def _read_hinglid(path):
    """L3Cube-HingLID: 'token<TAB>HI|EN' per line, blank line between sentences."""
    sents, cur = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            if len(cur) >= 3:
                sents.append(cur)
            cur = []
        else:
            cur.append(line.split("\t")[0])
    if len(cur) >= 3:
        sents.append(cur)
    return sents


def _hing_samples(path, n, rng):
    """n real Hinglish samples: half full tweets, half short 3-10 token windows (message-like)."""
    sents = _read_hinglid(path)
    idx = rng.choice(len(sents), size=min(n, len(sents)), replace=False)
    out = []
    for j, i in enumerate(idx):
        toks = sents[i]
        if j % 2 == 0 or len(toks) <= 5:
            out.append(" ".join(toks))
        else:
            w = int(rng.integers(3, min(10, len(toks)) + 1))
            st = int(rng.integers(0, len(toks) - w + 1))
            out.append(" ".join(toks[st:st + w]))
    return out


def load_language_v2(seed: int = 42, n_hing_train=1500, n_hing_test=400, n_sms=1500):
    """Language-ID data v2 (real Hinglish).

    train: UDHR pieces (15 languages) + informal English (SMS ham) + REAL romanized Hinglish (L3Cube-HingLID)
    tests: 'udhr' (other languages, easy), 'sms_en' (informal English must NOT be called Hinglish),
           'hinglid' (real Hinglish, held-out file), 'seed_ood' (my 78 hand-written Hinglish lines, never trained on)
    """
    rng = np.random.default_rng(seed)
    udhr_tr, udhr_te = load_language(seed)
    udhr_tr = udhr_tr[udhr_tr.label != "hinglish"]
    udhr_te = udhr_te[udhr_te.label != "hinglish"]
    sms_tr, sms_te, _ = load_spam(seed)
    # all SMS (ham AND spam) are English: spam text must not be mistaken for Hinglish
    sms_tr = sms_tr.sample(n_sms, random_state=seed)
    hr = RAW / "hinglid"
    tr = pd.concat([
        udhr_tr[["text", "label"]],
        pd.DataFrame({"text": sms_tr.text.values, "label": "en"}),
        pd.DataFrame({"text": _hing_samples(hr / "train.txt", n_hing_train, rng), "label": "hinglish"}),
    ], ignore_index=True)
    seed_lines = [l.strip() for l in (ROOT / "data" / "hinglish_seed.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    tests = {
        "udhr (other languages)": udhr_te[["text", "label"]].reset_index(drop=True),
        "informal English (SMS incl. spam)": pd.DataFrame({"text": sms_te.text.values, "label": "en"}),
        "real Hinglish (HingLID test)": pd.DataFrame({"text": _hing_samples(hr / "test.txt", n_hing_test, rng), "label": "hinglish"}),
        "hand-written Hinglish (OOD)": pd.DataFrame({"text": seed_lines, "label": "hinglish"}),
    }
    return tr, tests


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
