"""Final English sentiment v4: 3 classes (negative / neutral / positive).   python src/final_sentiment_en3.py
Variant C3 of exp_sentiment_en3.py (chosen on data/eval/sent_en3_dev.csv + TweetEval val). Held-out sets scored once here."""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
from data import ROOT
from exp_sentiment_en3 import build, tweeteval3, dev, tw_va
from exp_sentiment_en_lex import TESTS
warnings.filterwarnings("ignore")
m = build(neu_tw_w=0.5, neu_tpl=2500, neu_tpl_w=0.5)
test = pd.read_csv(ROOT / "data/eval/sent_en3_test.csv"); tw_te = tweeteval3("test")
L = ["negative", "neutral", "positive"]
def sc(d): p = m.predict(d.text); return {"macroF1": round(float(f1_score(d.label, p, average="macro")), 4), "accuracy": round(float(accuracy_score(d.label, p)), 4), "recall(neg,neu,pos)": [round(float(x), 3) for x in recall_score(d.label, p, average=None, labels=L)]}
res = {"customer+neutral TEST (70, scored once)": sc(test), "customer+neutral DEV (89)": sc(dev), "TweetEval 3-class TEST (12,284)": sc(tw_te), "TweetEval 3-class val": sc(tw_va)}
# no-regression check on positive/negative-only sets: how often does the 3-class model keep the right side (neutral counts as wrong)
for k in ("fresh everyday (32)", "negation challenge", "fresh everyday #2 (24)"):
    d = TESTS[k] if k in TESTS else None
    if d is not None: p = m.predict(d.text); res["posneg-only: " + k] = {"accuracy": round(float(accuracy_score(d.label, p)), 4), "predicted_neutral": int((p == "neutral").sum()), "n": len(d)}
pn = test[test.label != "neutral"]; p = m.predict(pn.text); res["customer TEST pos/neg only (40)"] = {"accuracy": round(float(accuracy_score(pn.label, p)), 4), "predicted_neutral": int((p == "neutral").sum())}
# margin cut for the review queue: 15th percentile of (top - second) decision margin on TweetEval val + customer dev
X = pd.concat([tw_va.text, dev.text]); d = np.sort(m.decision_function(X), axis=1); marg = d[:, -1] - d[:, -2]; cut = float(np.percentile(marg, 15))
pred = m.predict(X); y = pd.concat([tw_va.label, dev.label]).values; lo = marg <= cut
res["margin_cut_15pct"] = round(cut, 3); res["accuracy_below_cut"] = round(float((pred[lo] == y[lo]).mean()), 3); res["accuracy_above_cut"] = round(float((pred[~lo] == y[~lo]).mean()), 3)
joblib.dump(m, ROOT / "models" / "sentiment.joblib"); (ROOT / "results" / "sentiment_en_final_v4.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
for t in ["your watch is looking ok", "hey your phone is really good and smooth", "The delivery was quick and the staff were really helpful", "Is the refund processed yet, my card was charged twice", "it is okay", "I love it", "this is bad", "not bad at all", "I ordered a phone yesterday", "the food was not good", "meh"]:
    print(m.predict([t])[0], "|", t)
for n, dd in (("TEST", test),):
    for t, l, q in zip(dd.text, dd.label, m.predict(dd.text)):
        if l != q: print("  wrong", n, "(true", l + ", got", q + "):", t)
