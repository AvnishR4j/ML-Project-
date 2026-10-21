"""End-to-end message pipeline (final): language -> spam -> sentiment -> theme.

  1. language   : calibrated char-n-gram SVM with a reject option ("uncertain" -> manual review)
  2. spam       : TF-IDF + structural cues + scam lexicon, calibrated; separate threshold per language
                  (English and Hinglish; other languages are not routed to the spam model)
  3. sentiment  : English model (tweets + short reviews, 2 classes) | Hinglish model (IIIT-H, 3 classes)
  4. theme      : word2vec + K-Means, one model per language (English, Hinglish)
"""
from functools import lru_cache
import joblib
from data import ROOT, LANG_NAMES
from text_norm import normalize as clean_text, fix_spacing

SENT_EN = {"positive": "positive", "negative": "negative"}


@lru_cache(maxsize=1)
def _models():
    m = ROOT / "models"
    return {
        "language": joblib.load(m / "language_final.joblib"),
        "spam": joblib.load(m / "spam_final.joblib"),
        "sent_en": joblib.load(m / "sentiment.joblib"),
        "sent_hi": joblib.load(m / "hinglish_sentiment.joblib"),
        "theme_en": joblib.load(m / "theme_en.joblib"),
        "theme_hi": joblib.load(m / "theme_hinglish.joblib"),
    }


def detect_language(text, M=None):
    M = M or _models()
    L = M["language"]
    x = fix_spacing(text)
    p = L["model"].predict_proba([x])[0]
    i = int(p.argmax())
    return str(L["model"].classes_[i]), float(p[i]), bool(p[i] >= L["threshold"])


def spam_score(text, lang, M=None):
    M = M or _models()
    S = M["spam"]
    p = float(S["model"].predict_proba([clean_text(text)])[0, 1])
    thr = S["thresholds"]["hinglish" if lang == "hinglish" else "en"]
    return p, thr, p >= thr


def theme(text, lang, M=None):
    M = M or _models()
    T = M["theme_hi" if lang == "hinglish" else "theme_en"]
    c = int(T["km"].predict(T["emb"]([text]))[0])
    return c, T["names"][c]


def analyze(text: str, force_language: str = None) -> dict:
    """force_language lets experiments route with the TRUE language (oracle) to measure error propagation."""
    M = _models()
    out = {"text": text, "steps": [], "needs_review": False}
    if force_language:
        lang, conf, ok = force_language, 1.0, True
    else:
        lang, conf, ok = detect_language(text, M)
    out["language_confidence"] = conf
    if not ok:
        out["language"] = "uncertain"
        out["language_name"] = f"Uncertain (best guess: {LANG_NAMES.get(lang, lang)}, {conf:.0%})"
        out["needs_review"] = True
        out["steps"].append(("Language", out["language_name"], "Confidence below the threshold, so the message is sent for manual review instead of guessed."))
        out["spam"] = None
        return out
    out["language"] = lang
    out["language_name"] = LANG_NAMES.get(lang, lang)
    note = f"Confidence {conf:.0%}." + (" Very short text, so treat with care." if len(text.strip()) < 8 else "")
    out["steps"].append(("Language", out["language_name"], note))

    if lang not in ("en", "hinglish"):
        out["spam"] = None
        out["steps"].append(("Spam filter", "not applied", "The spam model covers English and Hinglish only."))
        return out

    p, thr, is_spam = spam_score(text, lang, M)
    out["spam"] = "spam" if is_spam else "ham"
    out["spam_prob"] = p
    how = "<=0.5% of genuine English messages flagged on validation" if lang == "en" else "<=1% of real genuine Hinglish text flagged; Hinglish scam recall is limited (see Evaluation)"
    out["steps"].append(("Spam filter", out["spam"].upper(), f"P(spam)={p:.2f}, threshold {thr:.2f} ({how})."))
    if is_spam:
        out["steps"].append(("Sentiment", "skipped", "Message filtered as spam."))
        return out

    if lang == "en":
        s = str(M["sent_en"].predict([text])[0])
        out["sentiment"] = SENT_EN.get(s, s)
        out["steps"].append(("Sentiment", out["sentiment"], "English model trained on tweets and short reviews (positive / negative)."))
    else:
        s = str(M["sent_hi"].predict([text])[0])
        out["sentiment"] = s
        out["steps"].append(("Sentiment", s, "Hinglish model trained on 3,100 real comments (negative / neutral / positive)."))
    cid, names = theme(text, lang, M)
    out["cluster"] = cid
    out["cluster_terms"] = names
    out["steps"].append(("Theme", f"#{cid}", f"Top terms: {names}"))
    return out
