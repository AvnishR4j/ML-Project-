"""Language-agnostic structural spam cues + an optional romanized-Hindi/Devanagari scam lexicon."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_URL = re.compile(r"https?://|www\.|\b\w+\.(?:com|in|co|net|org|link|xyz|top|info)\b|\bbit\.ly|\btinyurl", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)|\b\d{5}[\s-]?\d{5}\b|\b\d{4,}\b")
_MONEY = re.compile(r"[₹$£€]|\brs\.?\s?\d|\binr\b|\blakh|\bcrore|\bpaise|रु", re.I)
_PCT = re.compile(r"\d+\s?%")
# Hand-written domain knowledge (NOT learned): common scam vocabulary in romanized Hindi / Devanagari.
_LEX = re.compile(
    r"\b(inaam|inam|jeeto|jeet\s?gaye|jeete|muft|free|lottery|lucky|winner|kyc|otp|pan\s?card|aadhaar|aadhar|"
    r"band\s?ho\s?jayega|block\s?ho|turant|jaldi|abhi\s?click|click\s?karein|click\s?kare|link\s?par|"
    r"claim|offer|subsidy|yojana|loan|cashback|refund|expire|verify|update\s?karein|apply\s?now|register\s?kar|"
    r"guaranteed|return|invest|crore|lakh|prize)\b"
    r"|इनाम|लॉटरी|मुफ्त|तुरंत|क्लिक|लिंक|केवाईसी|बंद हो|रिटर्न|अभी", re.I)


def structural_matrix(texts, lexicon=False):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        letters = [c for c in t if c.isalpha()]
        up = sum(c.isupper() for c in letters) / max(len(letters), 1)
        f = [
            float(bool(_URL.search(t))),
            float(bool(_PHONE.search(t))),
            float(bool(_MONEY.search(t))),
            float(bool(_PCT.search(t))),
            sum(c.isdigit() for c in t) / n,
            up,
            min(t.count("!"), 5) / 5.0,
            np.log1p(len(t)) / 6.0,
        ]
        if lexicon:
            f.append(min(len(_LEX.findall(t)), 4) / 4.0)
        rows.append(f)
    return np.asarray(rows, dtype=float)


class Structural(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon=False, weight=1.0):
        self.lexicon, self.weight = lexicon, weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return structural_matrix(list(X), self.lexicon) * self.weight
