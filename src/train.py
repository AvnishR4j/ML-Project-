"""Baseline comparison of Naive Bayes / linear SVM / KNN for language detection and spam (UCI).
Sentiment and themes: see exp_sentiment.py and exp_cluster.py. Deployed models: final_language.py, final_spam.py, final_themes.py.

Run from the repo root:   python src/train.py
Outputs: models/*.joblib, results/model_comparison.csv, results/*.png, results/summary.json
"""
import json, time, warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from data import ROOT, load_language, load_language_v2, load_spam

warnings.filterwarnings("ignore")
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
SEED = 42
CV = StratifiedKFold(3, shuffle=True, random_state=SEED)


def model_zoo(features):
    """Same features for all three classifiers -> fair comparison."""
    return {
        "Naive Bayes": (Pipeline([("f", features), ("m", MultinomialNB())]), {"m__alpha": [0.01, 0.1, 0.5, 1.0]}),
        "Linear SVM": (Pipeline([("f", features), ("m", LinearSVC(random_state=SEED))]), {"m__C": [0.1, 1, 10]}),
        "KNN": (Pipeline([("f", features), ("m", KNeighborsClassifier(metric="cosine", algorithm="brute"))]),
                {"m__n_neighbors": [1, 3, 5, 9]}),
    }


def run_task(name, tr, te, features, primary, pos_label=None):
    rows, best = [], (None, -1, None)
    for mname, (pipe, grid) in model_zoo(features).items():
        gs = GridSearchCV(pipe, grid, cv=CV, scoring="f1_macro", n_jobs=-1)
        t0 = time.time(); gs.fit(tr.text, tr.label); fit_s = time.time() - t0
        t0 = time.time(); pred = gs.predict(te.text); pred_ms = (time.time() - t0) / len(te) * 1000
        r = {"task": name, "model": mname, "best_params": str(gs.best_params_),
             "accuracy": accuracy_score(te.label, pred),
             "macro_F1": f1_score(te.label, pred, average="macro"),
             "train_s": round(fit_s, 2), "predict_ms_per_msg": round(pred_ms, 3), "n_test": len(te)}
        if pos_label:
            r["spam_precision"] = precision_score(te.label, pred, pos_label=pos_label)
            r["spam_recall"] = recall_score(te.label, pred, pos_label=pos_label)
            r["spam_F1"] = f1_score(te.label, pred, pos_label=pos_label)
        rows.append(r)
        score = r[primary]
        if score > best[1]:
            best = (mname, score, gs.best_estimator_)
        print(f"  [{name}] {mname:12s} acc={r['accuracy']:.3f} macroF1={r['macro_F1']:.3f} {gs.best_params_}")
    return rows, best


if __name__ == "__main__":
    all_rows, summary = [], {}

    # ---------------- 1. language detection (v2: real Hinglish) ----------------
    print("Language detection ...")
    ltr, ltests = load_language_v2()
    feats = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, lowercase=True)
    pooled = pd.concat(ltests.values(), ignore_index=True)
    # tune on the pooled in-domain tests' structure via CV on train; report per test set
    lang_rows, best_lang = [], (None, -1, None)
    for mname, (pipe, grid) in model_zoo(feats).items():
        gs = GridSearchCV(pipe, grid, cv=CV, scoring="f1_macro", n_jobs=-1)
        t0 = time.time(); gs.fit(ltr.text, ltr.label); fit_s = time.time() - t0
        row = {"task": "Language detection", "model": mname, "best_params": str(gs.best_params_), "train_s": round(fit_s, 2)}
        t0 = time.time(); pred_all = gs.predict(pooled.text); row["predict_ms_per_msg"] = round((time.time() - t0) / len(pooled) * 1000, 3)
        for tname, tdf in ltests.items():
            row["acc: " + tname] = accuracy_score(tdf.label, gs.predict(tdf.text))
        row["accuracy"] = accuracy_score(pooled.label, pred_all)
        row["macro_F1"] = f1_score(pooled.label, pred_all, average="macro")
        row["n_test"] = len(pooled)
        lang_rows.append(row)
        sel = np.mean([row["acc: real Hinglish (HingLID test)"], row["acc: hand-written Hinglish (OOD)"], row["acc: informal English (SMS incl. spam)"]])
        if sel > best_lang[1]:
            best_lang = (mname, sel, gs.best_estimator_)
        print(f"  [lang] {mname:12s} " + " ".join(f"{k.split(':')[1].strip()[:14]}={v:.3f}" for k, v in row.items() if k.startswith("acc: ")))
    all_rows += lang_rows
    bn, bm = best_lang[0], best_lang[2]
    joblib.dump(bm, ROOT / "models" / "language_baseline.joblib")
    summary["language"] = {"best": bn, "train": len(ltr), "classes": sorted(ltr.label.unique().tolist()),
                           "test_sets": {k: len(v) for k, v in ltests.items()},
                           "selection": "mean accuracy on real Hinglish, hand-written Hinglish (OOD) and informal English"}
    pred = bm.predict(pooled.text)
    labs = sorted(pooled.label.unique())
    cm = confusion_matrix(pooled.label, pred, labels=labs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(np.log1p(cm), cmap="Blues")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Language detection - {bn} (all test sets pooled; colour = log count)")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "language_confusion.png", dpi=130); plt.close(fig)
    # accuracy by message length on the hard sets
    hard = pd.concat([ltests["real Hinglish (HingLID test)"], ltests["informal English (SMS incl. spam)"], ltests["hand-written Hinglish (OOD)"]], ignore_index=True)
    ph = bm.predict(hard.text)
    L = hard.assign(ok=(ph == hard.label), n=hard.text.str.split().str.len())
    L["words"] = pd.cut(L.n, [0, 3, 6, 10, 20, 1000], labels=["1-3", "4-6", "7-10", "11-20", "21+"])
    L.groupby("words", observed=True).ok.agg(["mean", "size"]).to_csv(ROOT / "results" / "language_accuracy_by_length.csv")
    # the errors that matter: English <-> Hinglish
    eh = hard[(hard.label != ph)].assign(pred=ph[hard.label != ph]).head(25)
    eh.to_csv(ROOT / "results" / "language_errors_sample.csv", index=False)

    # ---------------- 2. spam ----------------
    print("Spam ...")
    str_, ste, dupes = load_spam()
    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2))])
    rows, (bn, bs, bm) = run_task("Spam detection", str_, ste, feats, "spam_F1", pos_label="spam")
    all_rows += rows
    joblib.dump(bm, ROOT / "models" / "spam_baseline.joblib")
    summary["spam"] = {"best": bn, "duplicates_removed": int(dupes), "train": len(str_), "test": len(ste),
                       "spam_share_test": float((ste.label == "spam").mean())}

    pd.DataFrame(all_rows).to_csv(ROOT / "results" / "model_comparison.csv", index=False)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
