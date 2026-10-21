"""End-to-end message pipeline: language -> spam -> sentiment -> theme cluster.

Prototype limits (shown to the user by the app, not hidden):
  * spam model is trained on English SMS only; for Hinglish it is run but flagged unreliable
  * sentiment model is English movie reviews only
  * cluster model is trained on English SMS only
"""
from functools import lru_cache
import joblib
import numpy as np
from sklearn.preprocessing import normalize
from data import ROOT, LANG_NAMES


@lru_cache(maxsize=1)
def _models():
    m = ROOT / "models"
    return {k: joblib.load(m / f"{k}.joblib") for k in ("language", "spam", "sentiment", "cluster")}


def analyze(text: str) -> dict:
    M = _models()
    out = {"text": text, "steps": []}
    lang = str(M["language"].predict([text])[0])
    out["language"] = lang
    out["language_name"] = LANG_NAMES.get(lang, lang)
    short = "Very short text (<8 characters): language detection is unreliable." if len(text.strip()) < 8 else None
    out["steps"].append(("Language", out["language_name"], short))

    if lang in ("en", "hinglish"):
        spam = str(M["spam"].predict([text])[0])
        note = None if lang == "en" else "Spam model is English-only in the prototype; Hinglish result is unreliable."
        out["spam"] = spam
        out["steps"].append(("Spam filter", spam.upper(), note))
    else:
        out["spam"] = None
        out["steps"].append(("Spam filter", "skipped", "Prototype spam model supports English (and experimentally Hinglish) only."))
        return out

    if spam == "spam":
        out["steps"].append(("Sentiment", "skipped", "Message filtered as spam."))
        return out

    if lang == "en":
        s = str(M["sentiment"].predict([text])[0])
        out["sentiment"] = "positive" if s == "pos" else "negative"
        out["steps"].append(("Sentiment", out["sentiment"], "Trained on movie reviews (domain mismatch with chat/customer text)."))
        C = M["cluster"]
        z = normalize(C["svd"].transform(C["vec"].transform([text])))
        cid = int(C["km"].predict(z)[0])
        out["cluster"] = cid
        out["cluster_terms"] = C["names"][cid]
        out["steps"].append(("Theme cluster", f"#{cid}", f"Top terms: {C['names'][cid]}"))
    else:
        out["steps"].append(("Sentiment", "not available", "Hinglish sentiment model planned (SemEval-2020 Task 9)."))
        out["steps"].append(("Theme cluster", "not available", "Clustering trained on English only in the prototype."))
    return out
