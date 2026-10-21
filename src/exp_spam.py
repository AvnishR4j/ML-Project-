"""Spam experiments: how well does an English-trained spam filter transfer to Hinglish/Hindi, and what helps?

   python src/exp_spam.py
Challenge sets are small, hand-written by a third party, de-duplicated; read the numbers as indicative only.
"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from data import ROOT, load_spam
from data2 import load_scam_challenge
from spam_features import Structural
from text_norm import normalize

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42


def build(kind, sw=1.0):
    w = ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True))
    c = ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2))
    parts = {"word+char (baseline)": [w, c], "char only": [c],
             "+ structural cues": [w, c, ("s", Structural(False, sw))],
             "+ structural + lexicon": [w, c, ("s", Structural(True, sw))],
             "structural + lexicon only": [("s", Structural(True, sw))]}[kind]
    return Pipeline([("f", FeatureUnion(parts)), ("m", LinearSVC(C=0.5, random_state=SEED, class_weight="balanced"))])


def metrics(y, dec):
    y = (np.asarray(y) == "spam").astype(int); p = (dec > 0).astype(int)
    return {"AUC": roc_auc_score(y, dec), "spam_recall": recall_score(y, p), "spam_F1": f1_score(y, p),
            "false_alarm_rate": float(((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}


def main():
    tr, te, _ = load_spam(); ch = load_scam_challenge()
    print({k: v.label.value_counts().to_dict() for k, v in ch.items()})
    rows = []
    for kind in ("word+char (baseline)", "char only", "+ structural cues", "+ structural + lexicon", "structural + lexicon only"):
        m = build(kind).fit(tr.text.map(normalize), tr.label)
        for name, df in (("English SMS test", te), ("Hinglish challenge", ch["hinglish"]), ("Hindi (Devanagari) challenge", ch["hindi"])):
            r = metrics(df.label, m.decision_function(df.text.map(normalize)))
            rows.append({"variant": kind, "test_set": name, "n": len(df), **r})
    E = pd.DataFrame(rows)
    # few-shot: add a handful of Hinglish examples to training; evaluate on the REST (20 random splits)
    hl = ch["hinglish"]; fs = []
    for frac in (0.1, 0.25, 0.5):
        aucs, f1s = [], []
        for seed in range(20):
            sss = StratifiedShuffleSplit(1, train_size=frac, random_state=seed)
            a, b = next(sss.split(hl.text, hl.label))
            trn = pd.concat([tr, hl.iloc[a]], ignore_index=True)
            m = build("+ structural + lexicon").fit(trn.text.map(normalize), trn.label)
            r = metrics(hl.iloc[b].label, m.decision_function(hl.iloc[b].text.map(normalize)))
            aucs.append(r["AUC"]); f1s.append(r["spam_F1"])
        fs.append({"hinglish_training_examples": int(round(len(hl) * frac)), "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs), "spam_F1_mean": np.mean(f1s), "spam_F1_std": np.std(f1s)})
    F = pd.DataFrame(fs)
    E.to_csv(R / "spam_transfer.csv", index=False); F.to_csv(R / "spam_fewshot.csv", index=False)
    pd.set_option("display.width", 200)
    print(E.round(3).to_string(index=False)); print(F.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
