"""Hinglish sentiment from SemEval-2020 Task 9 (SentiMix). Hugging Face / Zenodo are not reachable from the
build sandbox, so download the Hinglish files yourself:
    https://zenodo.org/record/3974927   ->  put the Hinglish train/dev (+ test if present) files in data/raw/semeval/
Expected CoNLL layout (verify with `head`; adjust parse_conll if yours differs):
    meta<TAB><uid><TAB><positive|neutral|negative>
    token<TAB>Eng|Hin|O
    ...
    <blank line>
Run:  python src/semeval.py        (trains NB / SVM / KNN, prints weighted F1, saves models/hinglish_sentiment.joblib)
NOTE: this parser was tested on a synthetic file in the layout above, not on the real files.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from data import ROOT

DIR = ROOT / "data" / "raw" / "semeval"


def parse_conll(path):
    rows, toks, lab = [], [], None
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("meta"):
            parts = line.split("\t")
            lab = parts[-1].strip().lower()
            toks = []
        elif not line.strip():
            if lab and toks:
                rows.append((" ".join(toks), lab))
            toks, lab = [], None
        else:
            toks.append(line.split("\t")[0])
    if lab and toks:
        rows.append((" ".join(toks), lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def find(name_part):
    c = sorted(p for p in DIR.glob("*") if name_part in p.name.lower())
    return c[0] if c else None


if __name__ == "__main__":
    tr_p, dv_p = find("train"), find("dev") or find("valid")
    if not tr_p or not dv_p:
        sys.exit(f"Put the SemEval Hinglish train and dev files into {DIR} (see docstring).")
    tr, dv = parse_conll(tr_p), parse_conll(dv_p)
    print(len(tr), "train /", len(dv), "dev;", tr.label.value_counts().to_dict())
    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3))])
    zoo = {"Naive Bayes": (MultinomialNB(), {"m__alpha": [0.1, 0.5, 1.0]}),
           "Linear SVM": (LinearSVC(random_state=42), {"m__C": [0.05, 0.1, 0.5, 1]}),
           "KNN": (KNeighborsClassifier(metric="cosine", algorithm="brute"), {"m__n_neighbors": [5, 15, 31]})}
    best = (None, -1, None)
    for n, (clf, grid) in zoo.items():
        gs = GridSearchCV(Pipeline([("f", feats), ("m", clf)]), grid, cv=3, scoring="f1_weighted", n_jobs=-1).fit(tr.text, tr.label)
        f1 = f1_score(dv.label, gs.predict(dv.text), average="weighted")
        print(f"{n:12s} dev weighted-F1 = {f1:.3f}  {gs.best_params_}")
        if f1 > best[1]:
            best = (n, f1, gs.best_estimator_)
    print("\nBest:", best[0]); print(classification_report(dv.label, best[2].predict(dv.text)))
    print("Reference (task paper): mBERT baseline 65.4, winner 75.0 weighted F1 on the official test set.")
    joblib.dump(best[2], ROOT / "models" / "hinglish_sentiment.joblib")
