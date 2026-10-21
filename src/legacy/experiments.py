"""Evaluation experiments: (A) calibration + threshold, (B) spam robustness, (C) error propagation.
Run after train.py:   python experiments.py
"""
import ast, json, warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import ROOT, load_spam
from text_norm import attack_homoglyph, attack_spacing, normalize

warnings.filterwarnings("ignore")
RES = ROOT / "results"
SEED = 42
out = {}


def feats():
    return FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                         ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2))])


def make(name, params):
    clf = {"Naive Bayes": MultinomialNB(), "Linear SVM": LinearSVC(random_state=SEED),
           "KNN": KNeighborsClassifier(metric="cosine", algorithm="brute")}[name]
    return Pipeline([("f", feats()), ("m", clf)]).set_params(**params)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


tr, te, _ = load_spam()
cmp_ = pd.read_csv(RES / "model_comparison.csv")
cmp_ = cmp_[cmp_.task == "Spam detection"].set_index("model")
best_params = {m: ast.literal_eval(cmp_.loc[m, "best_params"]) for m in cmp_.index}
ytr, yte = (tr.label == "spam").astype(int).values, (te.label == "spam").astype(int).values

# ---------------- A. calibration + threshold at a low false-positive rate ----------------
fit, val = train_test_split(tr, test_size=0.25, stratify=tr.label, random_state=SEED)
yfit, yval = (fit.label == "spam").astype(int).values, (val.label == "spam").astype(int).values
cands = {}
nb = make("Naive Bayes", best_params["Naive Bayes"]).fit(fit.text, yfit)
cands["Naive Bayes (raw)"] = nb
svm_cal = CalibratedClassifierCV(make("Linear SVM", best_params["Linear SVM"]), cv=5, method="sigmoid").fit(fit.text, yfit)
cands["Linear SVM (calibrated)"] = svm_cal
nb_cal = CalibratedClassifierCV(make("Naive Bayes", best_params["Naive Bayes"]), cv=5, method="sigmoid").fit(fit.text, yfit)
cands["Naive Bayes (calibrated)"] = nb_cal
rows, curves = [], {}
for name, m in cands.items():
    pv, pt = m.predict_proba(val.text)[:, 1], m.predict_proba(te.text)[:, 1]
    # threshold: highest recall on VALIDATION subject to false-positive rate <= 0.5%
    best_t, best_r = 0.5, -1
    for t in np.unique(np.round(np.linspace(0.02, 0.98, 97), 2)):
        pred = pv >= t
        fpr = (pred & (yval == 0)).sum() / max((yval == 0).sum(), 1)
        rec = (pred & (yval == 1)).sum() / max((yval == 1).sum(), 1)
        if fpr <= 0.005 and rec > best_r:
            best_t, best_r = float(t), rec
    for label, thr in (("default 0.5", 0.5), (f"FPR<=0.5% (t={best_t:.2f})", best_t)):
        pred = (pt >= thr).astype(int)
        rows.append({"model": name, "threshold": label, "ECE": round(ece(yte, pt), 4), "Brier": round(brier_score_loss(yte, pt), 4),
                     "spam_precision": round(precision_score(yte, pred), 3), "spam_recall": round(recall_score(yte, pred), 3),
                     "ham_flagged_as_spam": int(((pred == 1) & (yte == 0)).sum()), "n_ham_test": int((yte == 0).sum())})
    curves[name] = (pt, best_t)
calib = pd.DataFrame(rows)
calib.to_csv(RES / "calibration_threshold.csv", index=False)
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, (pt, _) in curves.items():
    bins = np.linspace(0, 1, 11); idx = np.digitize(pt, bins[1:-1])
    xs = [pt[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
    ys = [yte[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
    ax.plot(xs, ys, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("predicted spam probability"); ax.set_ylabel("observed spam rate")
ax.set_title("Reliability diagram (spam, test set)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(RES / "spam_reliability.png", dpi=130); plt.close(fig)

# deploy the model with the lowest validation Brier score, refit on ALL training data
vb = {n: brier_score_loss(yval, m.predict_proba(val.text)[:, 1]) for n, m in cands.items()}
deploy_name = min(vb, key=vb.get)
base = deploy_name.split(" (")[0]
if "calibrated" in deploy_name:
    final = CalibratedClassifierCV(make(base, best_params[base]), cv=5, method="sigmoid").fit(tr.text, ytr)
else:
    final = make(base, best_params[base]).fit(tr.text, ytr)
t_dep = curves[deploy_name][1]
joblib.dump({"model": final, "threshold": t_dep, "name": deploy_name}, ROOT / "models" / "spam_deploy.joblib")
out["calibration"] = {"deployed": deploy_name, "threshold": t_dep, "val_brier": {k: round(v, 4) for k, v in vb.items()}}

# ---------------- B. robustness to evasion ----------------
spam_te, ham_te = te[te.label == "spam"].text.tolist(), te[te.label == "ham"].text.tolist()
atk = {"clean": lambda s: s, "spacing attack": attack_spacing, "homoglyph attack": attack_homoglyph,
       "both attacks": lambda s: attack_homoglyph(attack_spacing(s))}
rob = []
for mname in ("Naive Bayes", "Linear SVM", "KNN"):
    m = make(mname, best_params[mname]).fit(tr.text, ytr)
    for aname, f in atk.items():
        att = [f(s) for s in spam_te]
        r_raw = float(np.mean(m.predict(att) == 1))
        r_norm = float(np.mean(m.predict([normalize(a) for a in att]) == 1))
        rob.append({"model": mname, "attack": aname, "spam_recall_no_defence": round(r_raw, 3), "spam_recall_with_normalizer": round(r_norm, 3)})
    fp_raw = float(np.mean(m.predict(ham_te) == 1)); fp_norm = float(np.mean(m.predict([normalize(h) for h in ham_te]) == 1))
    rob.append({"model": mname, "attack": "ham false-positive rate (clean text)", "spam_recall_no_defence": round(fp_raw, 4), "spam_recall_with_normalizer": round(fp_norm, 4)})
pd.DataFrame(rob).to_csv(RES / "robustness.csv", index=False)

# ---------------- C. error propagation on the (DRAFT) pipeline test set ----------------
from pipeline import analyze  # noqa: E402
ts = pd.read_csv(ROOT / "data" / "pipeline_testset_DRAFT.csv").fillna("")
recs = []
for _, r in ts.iterrows():
    pred, orc = analyze(r.text), analyze(r.text, force_language=r.true_lang)
    def spam_of(a): return a.get("spam") or "skipped"
    def sent_of(a): return a.get("sentiment") or "skipped"
    exp_spam = r.true_spam if r.true_lang in ("en", "hinglish") else "skipped"
    exp_sent = {"pos": "positive", "neg": "negative"}.get(r.true_sentiment, "skipped") if (r.true_lang == "en" and r.true_spam == "ham") else "skipped"
    recs.append({"text": r.text[:70], "true_lang": r.true_lang, "pred_lang": pred["language"],
                 "lang_ok": pred["language"] == r.true_lang,
                 "spam_pred": spam_of(pred), "spam_oracle": spam_of(orc), "spam_expected": exp_spam,
                 "sent_pred": sent_of(pred), "sent_oracle": sent_of(orc), "sent_expected": exp_sent,
                 "e2e_pred": pred["language"] == r.true_lang and spam_of(pred) == exp_spam and sent_of(pred) == exp_sent,
                 "e2e_oracle": spam_of(orc) == exp_spam and sent_of(orc) == exp_sent})
P = pd.DataFrame(recs); P.to_csv(RES / "pipeline_eval_detail.csv", index=False)
route = P[P.true_lang.isin(["en", "hinglish"])]
summ = {"n": len(P),
        "language_accuracy": round(P.lang_ok.mean(), 3),
        "spam_accuracy_predicted_language": round((route.spam_pred == route.spam_expected).mean(), 3),
        "spam_accuracy_oracle_language": round((route.spam_oracle == route.spam_expected).mean(), 3),
        "end_to_end_exact_match_predicted": round(P.e2e_pred.mean(), 3),
        "end_to_end_exact_match_oracle_language": round(P.e2e_oracle.mean(), 3),
        "messages_where_language_error_changed_outcome": int((P.e2e_pred != P.e2e_oracle).sum())}
en_ham = P[(P.true_lang == "en") & (P.sent_expected != "skipped")]
summ["english_sentiment_accuracy_oracle"] = round((en_ham.sent_oracle == en_ham.sent_expected).mean(), 3)
summ["english_sentiment_n"] = int(len(en_ham))
out["pipeline_eval"] = summ
(RES / "experiments_summary.json").write_text(json.dumps(out, indent=2))
print(calib.to_string()); print(pd.DataFrame(rob).to_string()); print(json.dumps(out, indent=2))
