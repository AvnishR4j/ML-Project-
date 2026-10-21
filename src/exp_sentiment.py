"""Sentiment experiments: (1) real Hinglish sentiment, (2) English cross-domain study.

   python src/exp_sentiment.py
Selection uses cross-validation on TRAIN only; the held-out test set is touched once per final model.
"""
import json, time, warnings
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from data import ROOT
from data2 import load_hinglish_sentiment, load_english_sentiment, prep
from stats import bootstrap_ci, mcnemar

warnings.filterwarnings("ignore")
SEED = 42
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
R = ROOT / "results"; R.mkdir(exist_ok=True)


def feats(kind):
    w = lambda: TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    c = lambda: TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    return {"word": w(), "char": c(), "word+char": FeatureUnion([("w", w()), ("c", c())])}[kind]


def zoo(kind):
    return {
        "Naive Bayes": (Pipeline([("f", feats(kind)), ("m", ComplementNB())]), {"m__alpha": [0.1, 0.3, 1.0]}),
        "Linear SVM": (Pipeline([("f", feats(kind)), ("m", LinearSVC(class_weight="balanced", random_state=SEED))]), {"m__C": [0.03, 0.1, 0.3, 1]}),
        "Logistic Regression": (Pipeline([("f", feats(kind)), ("m", LogisticRegression(class_weight="balanced", max_iter=2000))]), {"m__C": [1, 3, 10, 30]}),
        "KNN": (Pipeline([("f", feats(kind)), ("m", KNeighborsClassifier(metric="cosine", algorithm="brute"))]), {"m__n_neighbors": [5, 11, 21]}),
    }


def main():
    summary = {}
    # ---------------------------------------------------------------- 1. Hinglish
    tr, te = load_hinglish_sentiment()
    print("Hinglish sentiment: train", len(tr), "test", len(te), te.label.value_counts().to_dict())
    base = f1_score(te.label, [tr.label.mode()[0]] * len(te), average="macro")
    rows, fitted = [], {}
    for kind in ("word", "char", "word+char"):
        for name, (pipe, grid) in zoo(kind).items():
            gs = GridSearchCV(pipe, grid, cv=CV, scoring="f1_macro", n_jobs=2).fit(tr.text, tr.label)
            t0 = time.time(); pred = gs.predict(te.text); ms = (time.time() - t0) / len(te) * 1000
            lo, hi = bootstrap_ci(te.label, pred, "macro_f1")
            rows.append({"features": kind, "model": name, "cv_macro_F1": gs.best_score_, "test_macro_F1": f1_score(te.label, pred, average="macro"),
                         "ci_low": lo, "ci_high": hi, "test_accuracy": accuracy_score(te.label, pred),
                         "predict_ms_per_msg": round(ms, 3), "best_params": str(gs.best_params_)})
            fitted[(kind, name)] = (gs.best_estimator_, pred)
            print(f"  {kind:10s} {name:20s} cv={gs.best_score_:.3f} test={rows[-1]['test_macro_F1']:.3f} [{lo:.3f},{hi:.3f}]")
    H = pd.DataFrame(rows); H.to_csv(R / "sentiment_hinglish.csv", index=False)
    best = H.sort_values("cv_macro_F1", ascending=False).iloc[0]          # chosen by CV, NOT by test
    bkey = (best.features, best.model); bmodel, bpred = fitted[bkey]
    print("  chosen by CV:", bkey, "test macro-F1", round(best.test_macro_F1, 3))
    # significance vs the other model families (same features)
    sig = []
    for name in zoo("word"):
        if name != best.model and (best.features, name) in fitted:
            n01, n10, p = mcnemar(te.label, bpred, fitted[(best.features, name)][1])
            sig.append({"vs": name, "best_right_other_wrong": n01, "other_right_best_wrong": n10, "p_value": p})
    pd.DataFrame(sig).to_csv(R / "sentiment_hinglish_mcnemar.csv", index=False)
    labs = ["negative", "neutral", "positive"]
    cm = confusion_matrix(te.label, bpred, labels=labs)
    pd.DataFrame(cm, index=labs, columns=labs).to_csv(R / "sentiment_hinglish_confusion.csv")
    joblib.dump(bmodel, ROOT / "models" / "hinglish_sentiment.joblib")
    summary["hinglish"] = {"train": len(tr), "test": len(te), "majority_baseline_macroF1": base, "chosen": list(bkey),
                           "test_macro_F1": float(best.test_macro_F1), "ci": [float(best.ci_low), float(best.ci_high)],
                           "published_reference": "SemEval-2020 Task 9 winner 75.0 weighted-F1 (different dataset)"}

    # ---------------------------------------------------------------- 2. English cross-domain
    dom = load_english_sentiment()
    sets = {k: v for k, v in dom.items()}
    sets["tweets + short reviews"] = (pd.concat([dom["tweets"][0], dom["short reviews"][0]], ignore_index=True), None)
    en_feats = lambda: TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    rows, models = [], {}
    for trn, (trd, _) in sets.items():
        gs = GridSearchCV(Pipeline([("f", en_feats()), ("m", LinearSVC(random_state=SEED))]), {"m__C": [0.1, 0.3, 1]}, cv=3, scoring="f1_macro", n_jobs=2).fit(trd.text, trd.label)
        models[trn] = gs.best_estimator_
        for ten, (_, ted) in dom.items():
            p = gs.predict(ted.text)
            rows.append({"trained_on": trn, "tested_on": ten, "macro_F1": f1_score(ted.label, p, average="macro"), "n_test": len(ted)})
        print("  EN trained on", trn, [round(r["macro_F1"], 3) for r in rows[-3:]])
    X = pd.DataFrame(rows); X.to_csv(R / "sentiment_english_crossdomain.csv", index=False)
    piv = X.pivot(index="trained_on", columns="tested_on", values="macro_F1")
    piv["mean_over_domains"] = piv.mean(axis=1)
    pick = piv["mean_over_domains"].idxmax()
    print(piv.round(3)); print("  English final training set:", pick)
    joblib.dump(models[pick], ROOT / "models" / "sentiment.joblib")
    summary["english"] = {"final_training_set": pick, "matrix": piv.round(4).to_dict(),
                          "old_movie_only_on_tweets": float(X[(X.trained_on == "movie reviews") & (X.tested_on == "tweets")].macro_F1.iloc[0])}
    # three-model comparison on the FINAL English training set, scored on every domain
    trd = sets[pick][0]; rows3 = []
    for name, clf in (("Naive Bayes", ComplementNB(alpha=0.3)), ("Linear SVM", LinearSVC(C=0.3, random_state=SEED)), ("KNN", KNeighborsClassifier(n_neighbors=15, metric="cosine", algorithm="brute"))):
        m = Pipeline([("f", en_feats()), ("m", clf)]).fit(trd.text, trd.label)
        for ten, (_, ted) in dom.items():
            p = m.predict(ted.text); lo, hi = bootstrap_ci(ted.label, p, "macro_f1", n=500)
            rows3.append({"model": name, "tested_on": ten, "macro_F1": f1_score(ted.label, p, average="macro"), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows3).to_csv(R / "sentiment_english_models.csv", index=False)
    # cross-LINGUAL: Hinglish model on English pos/neg (neutral never predicted correctly -> drop)
    en_te = dom["tweets"][1]; hp = bmodel.predict(en_te.text)
    keep = np.array(hp) != "neutral"
    summary["cross_lingual"] = {"hinglish_model_on_english_tweets_accuracy_(neutral_predictions_dropped)": float(accuracy_score(en_te.label[keep], np.array(hp)[keep])),
                                "share_dropped": float(1 - keep.mean())}
    (R / "sentiment_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
