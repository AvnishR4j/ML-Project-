"""Evaluate the full pipeline on 250 real held-out messages with MACHINE-ASSIGNED labels (language, spam, sentiment).
Labels: data/eval/e2e_labelled_machine.csv - written by Claude reading each message, NOT by independent human annotators.
Treat as 'machine-labelled'; two classmates should verify before it is called human-verified.   python src/eval_labelled.py"""
import json
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from data import ROOT
from pipeline import analyze

D = pd.read_csv(ROOT / "data" / "eval" / "e2e_labelled_machine.csv").fillna("")
res = [analyze(t) for t in D.text]
D["pl"] = [r["language"] for r in res]; D["ps"] = [r.get("spam") or "" for r in res]; D["pt"] = [r.get("sentiment", "") for r in res]
ev = D[D.lang != "?"]; ans = ev.pl != "uncertain"
out = {"n": len(D), "n_language_eval": int(len(ev)),
       "language": {"coverage": float(ans.mean()), "accuracy_when_answered": float((ev.pl[ans] == ev.lang[ans]).mean()),
                    "accuracy_all_uncertain_wrong": float((ev.pl == ev.lang).mean())}}
sp = D[D.lang.isin(["en", "hinglish"])]
y = (sp.spam == "spam").astype(int); p = (sp.ps == "spam").astype(int)
out["spam"] = {"n_spam": int(y.sum()), "n_ham": int((1 - y).sum()), "precision": float(precision_score(y, p)), "recall": float(recall_score(y, p)),
               "F1": float(f1_score(y, p)), "genuine_flagged": int(((p == 1) & (y == 0)).sum())}
for lg in ("en", "hinglish"):
    s = sp[sp.lang == lg]; yy = (s.spam == "spam").astype(int); pp = (s.ps == "spam").astype(int)
    out["spam"][lg] = {"n_spam": int(yy.sum()), "recall": float(recall_score(yy, pp)), "genuine_flagged": int(((pp == 1) & (yy == 0)).sum()), "n_ham": int((1 - yy).sum())}
out["sentiment"] = {}
for lg, labs in (("en", ["positive", "negative"]), ("hinglish", ["negative", "neutral", "positive"])):
    s = D[(D.lang == lg) & (D.spam == "ham") & (D.pt != "") & D.sent.isin(labs)]
    out["sentiment"][lg] = {"n": int(len(s)), "accuracy": float(accuracy_score(s.sent, s.pt)), "macro_F1": float(f1_score(s.sent, s.pt, average="macro", labels=labs, zero_division=0))}
(ROOT / "results" / "e2e_labelled.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
D[(D.lang != D.pl) | (D.spam != D.ps)].to_csv("/tmp/e2e_errors.csv", index=False)
