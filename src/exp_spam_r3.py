"""Spam round 3: H (live) vs I = H + spam_templates3 (conversational Hinglish scams, brand impersonation, emergency asks, misspelled prize/loan; genuine twins).
New blind test data/eval/spam_blind_r3.csv (60 lines written before training I; author-written, so optimistic). gupta and probes_clean are now DEV (templates were written after seeing their misses)."""
import json, warnings, numpy as np, pandas as pd
import exp_spam_r2 as X, exp_spam_indian as E
from sklearn.metrics import roc_auc_score
from data import ROOT
from spam_templates3 import make3
warnings.filterwarnings("ignore")
tpl3 = pd.DataFrame(make3(), columns=["text", "label"]); X.SETS["blind60"] = pd.read_csv(ROOT / "data/eval/spam_blind_r3.csv")
H = [E.ext, E.sms172, E.tpl, X.tpl2, X.pre_tr, X.jun_tr]
if __name__ == "__main__":
    res = []
    for name, extra in (("H (live)", H), ("I H + templates3", H + [tpl3])):
        m, P, o = X.score(name, extra); res.append(o)
    json.dump(res, open(ROOT / "results/spam_r3_variants.json", "w"), indent=1)
