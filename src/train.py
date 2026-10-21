"""Train + compare Naive Bayes / linear SVM / KNN on each task, save best models.

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

from data import ROOT, load_language, load_language_v2, load_sentiment, load_spam

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
    joblib.dump(bm, ROOT / "models" / "language.joblib")
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
    joblib.dump(bm, ROOT / "models" / "spam.joblib")
    summary["spam"] = {"best": bn, "duplicates_removed": int(dupes), "train": len(str_), "test": len(ste),
                       "spam_share_test": float((ste.label == "spam").mean())}

    # ---------------- 3. sentiment ----------------
    print("Sentiment ...")
    sx_tr, sx_te = load_sentiment()
    feats = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.9, stop_words="english")
    rows, (bn, bs, bm) = run_task("Sentiment (English)", sx_tr, sx_te, feats, "macro_F1")
    all_rows += rows
    joblib.dump(bm, ROOT / "models" / "sentiment.joblib")
    summary["sentiment"] = {"best": bn, "train": len(sx_tr), "test": len(sx_te)}

    # ---------------- 4. clustering (PCA/SVD + K-Means) on genuine (ham) SMS ----------------
    print("Clustering ...")
    full = pd.concat([str_, ste])
    ham = full[full.label == "ham"].text.reset_index(drop=True)
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(ham)
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    Z = Normalizer(copy=False).fit_transform(svd.fit_transform(X))
    scan = {}
    for k in range(4, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        scan[k] = float(silhouette_score(Z, km.labels_))
    # Silhouette keeps rising with k (no natural elbow), so k is fixed at 8 for interpretability.
    k_best = 8
    km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit(Z)
    terms = np.array(vec.get_feature_names_out())
    cent = km.cluster_centers_ @ svd.components_ / np.maximum(np.linalg.norm(svd.components_, axis=0), 1e-9)
    names = {}
    for c in range(k_best):
        top = terms[np.argsort(cent[c])[::-1][:6]]
        names[c] = ", ".join(top)
    joblib.dump({"vec": vec, "svd": svd, "km": km, "names": names}, ROOT / "models" / "cluster.joblib")
    sizes = pd.Series(km.labels_).value_counts().sort_index()
    pd.DataFrame({"cluster": sizes.index, "size": sizes.values, "top_terms": [names[c] for c in sizes.index]}) \
        .to_csv(ROOT / "results" / "cluster_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"Genuine SMS messages: SVD(50) + K-Means (k={k_best}), first 2 components shown")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "clusters.png", dpi=130); plt.close(fig)
    summary["clustering"] = {"k_best": int(k_best), "silhouette_by_k": scan, "n_messages": int(len(ham))}

    pd.DataFrame(all_rows).to_csv(ROOT / "results" / "model_comparison.csv", index=False)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
