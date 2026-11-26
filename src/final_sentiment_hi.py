"""Hinglish sentiment retrained with a small hand-written set of everyday sentences (data/hinglish_sentiment_seed.csv).
The IIIT-H comments are mostly about films, so everyday feeling words ("dil udaas", "bekar") were rare in training.
Same model family and settings as exp_sentiment.py (chosen by CV there); the IIIT-H test split is unchanged.
Also scores a small fresh set of everyday sentences written separately (not used for training)."""
import json, sys
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score
from data import ROOT
from data2 import load_hinglish_sentiment

REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FRESH = [  # fresh everyday sentences, NOT in the seed file
 ("Arun bhai ka dil bekar hai", "negative"), ("mera dil udaas hai", "negative"), ("aaj bahut bura din tha yaar", "negative"),
 ("yeh app bakwas hai", "negative"), ("main bahut dukhi hu", "negative"), ("mujhe kuch accha nahi lag raha", "negative"),
 ("khana ekdum ganda tha", "negative"), ("tumhari wajah se mera din kharab ho gaya", "negative"),
 ("kal 3 baje milte hain", "neutral"), ("tumhara ghar kahan hai", "neutral"), ("main abhi market mein hu", "neutral"),
 ("yeh train kitne baje chalti hai", "neutral"), ("mera exam kal hai", "neutral"), ("wo office gaya hai", "neutral"),
 ("Arun bhai ka dil bahut accha hai", "positive"), ("aaj main bahut khush hu", "positive"), ("yeh gana ekdum mast hai", "positive"),
 ("tumne bahut accha kaam kiya", "positive"), ("kya badhiya khana tha", "positive"), ("mujhe yeh movie bahut pasand aayi", "positive"),
]
tr, te = load_hinglish_sentiment()
seed = pd.read_csv(ROOT / "data" / "hinglish_sentiment_seed.csv")
old = joblib.load(ROOT / "models" / "hinglish_sentiment.joblib")
fresh = pd.DataFrame(FRESH, columns=["text", "label"])
def score(m, name):
    p = m.predict(te.text); pf = m.predict(fresh.text)
    r = {"model": name, "iiith_test_macroF1": f1_score(te.label, p, average="macro"), "iiith_test_acc": accuracy_score(te.label, p),
         "fresh_acc": accuracy_score(fresh.label, pf), "fresh_n": len(fresh)}
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}); return r, pf
r0, _ = score(old, "before (IIIT-H only)")
new = clone(old).fit(pd.concat([tr.text] + [seed.text] * REPEAT, ignore_index=True), pd.concat([tr.label] + [seed.label] * REPEAT, ignore_index=True))
r1, pf = score(new, f"after (+{len(seed)} seed sentences x{REPEAT})")
for (t, l), p in zip(FRESH, pf): print(f"  {'OK ' if p == l else 'ERR'} {l:8s} -> {p:8s} {t}")
if "--save" in sys.argv:
    joblib.dump(new, ROOT / "models" / "hinglish_sentiment.joblib")
    json.dump({"seed_rows": len(seed), "repeat": REPEAT, "before": r0, "after": r1}, open(ROOT / "results" / "sentiment_hinglish_seed.json", "w"), indent=1, default=float)
    print("saved")
