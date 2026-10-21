"""Does negation handling make sentiment more accurate?   python src/exp_negation.py
Compared on (a) the normal held-out test sets (must not get worse) and (b) a hand-written negation challenge set.
"""
import json, warnings
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from data import ROOT
from data2 import load_english_sentiment, load_hinglish_sentiment, prep
from negation import NegPrep, learn_polarity
warnings.filterwarnings("ignore")
R = ROOT / "results"
ch = pd.read_csv(ROOT / "data" / "eval" / "negation_challenge.csv")


def en_model(pp, C=0.3):
    return Pipeline([("f", TfidfVectorizer(preprocessor=pp, ngram_range=(1, 2), sublinear_tf=True, min_df=2)), ("m", LinearSVC(C=C, random_state=42))])


def hi_model(pp, C=0.3):
    w = lambda: TfidfVectorizer(preprocessor=pp, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    c = lambda: TfidfVectorizer(preprocessor=pp, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    return Pipeline([("f", FeatureUnion([("w", w()), ("c", c())])), ("m", LinearSVC(C=C, class_weight="balanced", random_state=42))])


def main():
    rows = []
    # ---------------- English: train on tweets + short reviews (the shipped training set)
    dom = load_english_sentiment()
    trd = pd.concat([dom["tweets"][0], dom["short reviews"][0]], ignore_index=True)
    pol = learn_polarity(trd.text, trd.label, "positive", "negative"); print("English polarity words:", len(pol))
    cfgs = {"baseline (no negation handling)": prep, "negation handling": NegPrep(pol)}
    chen = ch[ch.lang == "en"]; fitted = {}
    for name, pp in cfgs.items():
        m = en_model(pp).fit(trd.text, trd.label); fitted[name] = m
        for ten, (_, ted) in dom.items():
            rows.append({"language": "English", "model": name, "test": ten, "n": len(ted), "macro_F1": f1_score(ted.label, m.predict(ted.text), average="macro")})
        p = m.predict(chen.text); rows.append({"language": "English", "model": name, "test": "negation challenge", "n": len(chen), "macro_F1": f1_score(chen.label, p, average="macro"), "accuracy": float((p == chen.label.values).mean())})
        print(name, [round(r["macro_F1"], 3) for r in rows[-4:]])
    # ---------------- Hinglish
    tr, te = load_hinglish_sentiment(); chhi = ch[ch.lang == "hi"]
    polh = learn_polarity(tr.text, tr.label, "positive", "negative", top=800, min_df=2); print("Hinglish polarity words:", len(polh))
    cfgh = {"baseline (no negation handling)": prep, "negation handling": NegPrep(polh)}; fh = {}
    for name, pp in cfgh.items():
        m = hi_model(pp).fit(tr.text, tr.label); fh[name] = m
        rows.append({"language": "Hinglish", "model": name, "test": "IIIT-H test", "n": len(te), "macro_F1": f1_score(te.label, m.predict(te.text), average="macro")})
        p = m.predict(chhi.text); rows.append({"language": "Hinglish", "model": name, "test": "negation challenge", "n": len(chhi), "macro_F1": f1_score(chhi.label, p, average="macro"), "accuracy": float((p == chhi.label.values).mean())})
        print(name, [round(r["macro_F1"], 3) for r in rows[-2:]])
    pd.DataFrame(rows).to_csv(R / "sentiment_negation.csv", index=False)
    joblib.dump({"en": fitted["negation handling"], "hi": fh["negation handling"]}, ROOT / "models" / "_neg_candidates.joblib")


if __name__ == "__main__":
    main()
