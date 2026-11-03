"""End-to-end message pipeline (final): language -> spam -> sentiment -> theme.

  1. language   : calibrated char-n-gram SVM with a reject option ("uncertain" -> manual review)
  2. spam       : TF-IDF + structural cues + scam lexicon, calibrated; separate threshold per language
                  (English and Hinglish; other languages are not routed to the spam model)
  3. sentiment  : English model (tweets + short reviews, 2 classes) | Hinglish model (IIIT-H, 3 classes)
  4. theme      : word2vec + K-Means, one model per language (English, Hinglish)
"""
import json
import numpy as np
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


def theme_label(lang, cid):
    f = ROOT / "results" / "theme_labels.json"
    if f.exists():
        return json.loads(f.read_text()).get("hinglish" if lang == "hinglish" else "en", {}).get(str(cid))
    return None


NO_THEME = "No clear theme"


def theme_reject_score(T, z, c):
    """distance used to decide that a message is outside every learned topic (see final_themes2.py)"""
    rs = T.get("reject_score", "centroid")
    d = float(np.linalg.norm(T["km"].cluster_centers_[c] - z[0]))
    if rs == "centroid":
        return d
    if rs == "centroid_scaled":
        return d / float(T["scale"][c])
    P = T["protos"]
    return float(np.sqrt(((P - z[0]) ** 2).sum(1).min()))


def theme(text, lang, M=None):
    """returns (cluster id, top terms, rejected). A message with no known words, or farther from every learned topic than the
    calibrated threshold, is 'rejected' (no clear theme) instead of being forced into the nearest cluster."""
    M = M or _models()
    T = M["theme_hi" if lang == "hinglish" else "theme_en"]
    z = T["emb"]([text]); c = int(T["km"].predict(z)[0])
    rejected = False
    if T.get("version") == 2:
        rejected = bool(T["emb"].known([text])[0] == 0 or theme_reject_score(T, z, c) > T["reject_threshold"])
    return c, T["names"][c], rejected


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
    cid, names, rejected = theme(text, lang, M)
    out["cluster"] = cid
    out["cluster_terms"] = names
    out["theme_rejected"] = rejected
    out["cluster_label"] = NO_THEME if rejected else theme_label(lang, cid)
    out["steps"].append(("Theme", out["cluster_label"] or f"#{cid}", "Too far from every learned topic to name one." if rejected else f"Top terms: {names}"))
    return out
