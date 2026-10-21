"""End-to-end message pipeline: language -> spam -> sentiment -> theme cluster.

Prototype limits (shown to the user by the app, not hidden):
  * spam model is trained on English SMS only; for Hinglish it is run but flagged unreliable
  * sentiment model is English movie reviews only
  * cluster model is trained on English SMS only
"""
from functools import lru_cache
import joblib
from sklearn.preprocessing import normalize as l2norm
from data import ROOT, LANG_NAMES
from text_norm import normalize as clean_text, fix_spacing


@lru_cache(maxsize=1)
def _models():
    m = ROOT / "models"
    d = {k: joblib.load(m / f"{k}.joblib") for k in ("language", "sentiment", "cluster")}
    d["hing_sent"] = joblib.load(m / "hinglish_sentiment.joblib") if (m / "hinglish_sentiment.joblib").exists() else None
    if (m / "spam_deploy.joblib").exists():          # calibrated model + low-false-positive threshold
        d["spam"] = joblib.load(m / "spam_deploy.joblib")
    else:                                              # fallback: plain best model
        d["spam"] = {"model": joblib.load(m / "spam.joblib"), "threshold": None, "name": "plain"}
    return d


def _spam(M, text):
    s = M["spam"]; x = clean_text(text)                # defend against spaced / look-alike-letter evasion
    if s["threshold"] is None:
        return str(s["model"].predict([x])[0]), None
    p = float(s["model"].predict_proba([x])[0, 1])
    return ("spam" if p >= s["threshold"] else "ham"), p


def analyze(text: str, force_language: str = None) -> dict:
    """force_language lets experiments route with the TRUE language (oracle) to measure error propagation."""
    M = _models()
    out = {"text": text, "steps": []}
    lang = force_language or str(M["language"].predict([fix_spacing(text)])[0])
    out["language"] = lang
    out["language_name"] = LANG_NAMES.get(lang, lang)
    short = "Very short text (<8 characters): language detection is unreliable." if len(text.strip()) < 8 else None
    out["steps"].append(("Language", out["language_name"], short))

    if lang not in ("en", "hinglish"):
        out["spam"] = None
        out["steps"].append(("Spam filter", "skipped", "Prototype spam model supports English (and experimentally Hinglish) only."))
        return out

    spam, p = _spam(M, text)
    note = None if lang == "en" else "Spam model is English-only in the prototype; Hinglish result is unreliable."
    if p is not None:
        extra = f"P(spam)={p:.2f}, threshold {M['spam']['threshold']:.2f} (chosen so <=0.5% of genuine messages were flagged on validation data)"
        note = extra if note is None else note + " " + extra
    out["spam"] = spam
    out["spam_prob"] = p
    out["steps"].append(("Spam filter", spam.upper(), note))
    if spam == "spam":
        out["steps"].append(("Sentiment", "skipped", "Message filtered as spam."))
        return out

    if lang == "en":
        s = str(M["sentiment"].predict([text])[0])
        out["sentiment"] = "positive" if s == "pos" else "negative"
        out["steps"].append(("Sentiment", out["sentiment"], "Trained on movie reviews (domain mismatch with chat/customer text)."))
        C = M["cluster"]
        z = l2norm(C["svd"].transform(C["vec"].transform([text])))
        cid = int(C["km"].predict(z)[0])
        out["cluster"] = cid
        out["cluster_terms"] = C["names"][cid]
        out["steps"].append(("Theme cluster", f"#{cid}", f"Top terms: {C['names'][cid]}"))
    else:
        if M["hing_sent"] is not None:
            hs = str(M["hing_sent"].predict([text])[0])
            out["sentiment"] = hs
            out["steps"].append(("Sentiment", hs, "Hinglish model trained on SemEval-2020 Task 9 (3 classes; noisy labels)."))
        else:
            out["steps"].append(("Sentiment", "not available", "Hinglish sentiment model planned (SemEval-2020 Task 9)."))
        out["steps"].append(("Theme cluster", "not available", "Clustering trained on English only in the prototype."))
    return out
