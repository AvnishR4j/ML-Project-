"""Language ID with a reject option ("uncertain"): trade coverage for accuracy.

   python src/final_language.py
The abstention threshold is chosen on cross-validated TRAIN predictions (never on the test sets).
"""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data import ROOT, load_language_v2
from text_norm import fix_spacing

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42; TARGET = 0.99


def feats():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, lowercase=True)


def candidates():
    return {"Naive Bayes": Pipeline([("f", feats()), ("m", MultinomialNB(alpha=0.1))]),
            "Linear SVM (calibrated)": Pipeline([("f", feats()), ("m", CalibratedClassifierCV(LinearSVC(C=1, random_state=SEED), method="sigmoid", cv=3))])}


def risk_coverage(conf, ok):
    order = np.argsort(-conf); ok = ok[order]
    cov = np.arange(1, len(ok) + 1) / len(ok); acc = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    return cov, acc, float(np.mean(1 - acc))          # last = area under risk-coverage (lower is better)


def main():
    tr, tests = load_language_v2()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X = tr.text.map(fix_spacing)
    best = None; rows = []
    for name, pipe in candidates().items():
        P = cross_val_predict(pipe, X, tr.label, cv=cv, method="predict_proba", n_jobs=2)
        classes = np.array(sorted(tr.label.unique())); pred = classes[P.argmax(1)]; conf = P.max(1); ok = (pred == tr.label.values).astype(float)
        _, _, aurc = risk_coverage(conf, ok)
        # threshold: lowest confidence such that accuracy of accepted predictions >= TARGET on CV
        grid = np.linspace(0.3, 0.99, 140); thr = next((t for t in grid if ok[conf >= t].mean() >= TARGET), 0.99)
        rows.append({"model": name, "cv_accuracy": ok.mean(), "AURC": aurc, "threshold": float(thr), "cv_coverage_at_threshold": float((conf >= thr).mean())})
        if best is None or aurc < best[0]:
            best = (aurc, name, pipe, float(thr))
    pd.DataFrame(rows).to_csv(R / "language_selection.csv", index=False)
    _, name, pipe, thr = best
    pipe.fit(X, tr.label); classes = pipe.classes_
    out = []
    for tname, df in tests.items():
        P = pipe.predict_proba(df.text.map(fix_spacing)); pred = classes[P.argmax(1)]; conf = P.max(1); acc_all = float((pred == df.label.values).mean())
        keep = conf >= thr
        out.append({"test_set": tname, "n": len(df), "accuracy_all": acc_all, "coverage": float(keep.mean()),
                    "accuracy_on_accepted": float((pred[keep] == df.label.values[keep]).mean()) if keep.any() else np.nan})
    O = pd.DataFrame(out); O.to_csv(R / "language_selective.csv", index=False)
    # accuracy-vs-length with and without abstention (hard sets)
    hard = pd.concat([tests[k] for k in ("real Hinglish (HingLID test)", "informal English (SMS incl. spam)", "hand-written Hinglish (OOD)")], ignore_index=True)
    P = pipe.predict_proba(hard.text.map(fix_spacing)); pred = classes[P.argmax(1)]; conf = P.max(1)
    L = pd.DataFrame({"ok": pred == hard.label.values, "keep": conf >= thr, "n": hard.text.str.split().str.len()})
    L["words"] = pd.cut(L.n, [0, 3, 6, 10, 20, 1000], labels=["1-3", "4-6", "7-10", "11-20", "21+"])
    g = L.groupby("words", observed=True).apply(lambda d: pd.Series({"accuracy_all": d.ok.mean(), "coverage": d.keep.mean(), "accuracy_on_accepted": d[d.keep].ok.mean(), "n": len(d)}))
    g.to_csv(R / "language_by_length_selective.csv")
    # the two known failure cases
    probes = ["Merci beaucoup pour votre aide, a bientot", "ok", "Hello, how are you today?", "kal milte hain, ok?", "Danke schoen, bis morgen"]
    pr = pipe.predict_proba([fix_spacing(p) for p in probes]); probe_out = [{"text": p, "pred": classes[r.argmax()], "confidence": float(r.max()), "accepted": bool(r.max() >= thr)} for p, r in zip(probes, pr)]
    joblib.dump({"model": pipe, "threshold": thr, "name": name}, ROOT / "models" / "language_final.joblib")
    (R / "language_final_summary.json").write_text(json.dumps({"chosen": name, "threshold": thr, "target_accepted_accuracy": TARGET, "probes": probe_out}, indent=2))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(O.round(4).to_string(index=False)); print(g.round(3)); print(json.dumps(probe_out, indent=1))


if __name__ == "__main__":
    main()
