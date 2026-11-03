"""Final-project data loaders (v2).

Sentiment
  * Hinglish : IIIT-H Hindi-English code-mixed sentiment corpus (Prabhu et al., 2016) - 3,879 real,
               romanized Hinglish social-media comments, 3 classes. Source: github.com/DrImpossible/Sub-word-LSTM
  * English  : NLTK twitter_samples (10k tweets), NLTK sentence_polarity (10.6k short reviews),
               NLTK movie_reviews (2k long reviews). Different domains on purpose -> cross-domain tests.
Spam challenge sets (small, third-party hand-written; NOT used for training unless stated)
  * Hinglish scam / Hindi scam lines from github.com/gupta7050/multilingual-phishing-detection.
    The files hold 5,000 rows each but only ~101 UNIQUE texts, so only the unique lines are used.
"""
import json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import RAW

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_EMOT = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P|p)+|:'\(|<3")   # label-leaking emoticons in twitter_samples
_REPEAT = re.compile(r"(.)\1{2,}")


def clean_social(t: str) -> str:
    t = _URL.sub(" ", t)
    t = _USER.sub(" ", t)
    t = _EMOT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def squeeze(t: str) -> str:
    """'soooo gooood' -> 'soo good'  (elongation is common in Hinglish and English chat)."""
    return _REPEAT.sub(r"\1\1", t)


# ------------------------------------------------------------------ Hinglish sentiment
HING_LABELS = {"0": "negative", "1": "neutral", "2": "positive"}


def load_hinglish_sentiment(seed: int = 42, test_size: float = 0.2):
    rows = []
    for line in (RAW / "iiith_codemixed.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split("\t")
        if len(p) == 4 and p[2] == p[3] and p[2] in HING_LABELS:
            rows.append((clean_social(p[1]), HING_LABELS[p[2]]))
    df = pd.DataFrame(rows, columns=["text", "label"])
    df = df[df.text.str.len() > 0]
    df["key"] = df.text.str.lower().str.replace(r"[^a-z0-9ऀ-ॿ ]", "", regex=True)
    df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)      # dedupe before splitting
    tr, te = train_test_split(df, test_size=test_size, stratify=df.label, random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ------------------------------------------------------------------ English sentiment
def _tweets():
    rows = []
    for lab, f in (("positive", "positive_tweets.json"), ("negative", "negative_tweets.json")):
        for line in (RAW / "twitter_samples" / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((clean_social(json.loads(line)["text"]), lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def _sentences():
    rows = []
    for lab, f in (("positive", "rt-polarity.pos"), ("negative", "rt-polarity.neg")):
        for l in (RAW / "sentence_polarity" / f).read_text(encoding="latin-1").splitlines():
            rows.append((l.strip(), lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def _movies():
    rows = []
    for lab, d in (("positive", "pos"), ("negative", "neg")):
        for f in sorted((RAW / "movie_reviews" / d).glob("*.txt")):
            rows.append((f.read_text(encoding="latin-1"), lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_english_sentiment(seed: int = 42, test_size: float = 0.2):
    """{domain: (train, test)} - each domain split separately and de-duplicated first."""
    out = {}
    for name, df in (("tweets", _tweets()), ("short reviews", _sentences()), ("movie reviews", _movies())):
        df = df[df.text.str.len() > 0].copy()
        df["key"] = df.text.str.lower().str.replace(r"\W+", "", regex=True)
        df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
        tr, te = train_test_split(df, test_size=test_size, stratify=df.label, random_state=seed)
        out[name] = (tr.reset_index(drop=True), te.reset_index(drop=True))
    return out


# ------------------------------------------------------------------ Hinglish / Hindi scam challenge sets
def load_scam_challenge():
    """Unique third-party hand-written scam / genuine lines. Returns {'hinglish': df, 'hindi': df}."""
    out = {}
    for name, f in (("hinglish", "hinglish_scam_challenge_raw.csv"), ("hindi", "hindi_scam_challenge_raw.csv")):
        d = pd.read_csv(RAW / f, encoding="utf-8-sig").dropna()
        d = d.drop_duplicates("text").reset_index(drop=True)
        out[name] = d[["text", "label"]]
    return out


def prep(t: str) -> str:
    """Shared, picklable text preprocessor for the sentiment models."""
    return squeeze(t.lower())


# ------------------------------------------------------------------ TweetEval / SemEval-2017 Task 4 (English tweets, larger)
def load_tweeteval(split: str):
    """SemEval-2017 sentiment tweets (cardiffnlp/tweeteval). Keeps positive/negative only (neutral is not used by the 2-class model).
    split in {'train','val','test'}; the official test split is never used for training or model selection."""
    d = RAW / "tweeteval"
    txt = (d / f"{split}_text.txt").read_text(encoding="utf-8").splitlines()
    lab = (d / f"{split}_labels.txt").read_text(encoding="utf-8").split()
    rows = [(clean_social(t), {"0": "negative", "2": "positive"}[l]) for t, l in zip(txt, lab) if l in ("0", "2")]
    df = pd.DataFrame(rows, columns=["text", "label"])
    df = df[df.text.str.len() > 0]
    df["key"] = df.text.str.lower().str.replace(r"\W+", "", regex=True)
    return df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
