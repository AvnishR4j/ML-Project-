"""Spam round 2.   python src/exp_spam_r2.py
New real data: PRECOG 2011 Indian SMS (Hindi/English, 1,984) and junioralive India SMS (1,973); 75% of each trains, 25% is held out.
Templates: spam_templates2 (genuine bank/UPI/delivery/gov alerts with official links + threat/arrest/job-fee/loan/OLX/gift-card scams + SMS-speak).
Scored on: probes.py from a third-party repo (HINGLISH + MISSPELLED groups never viewed by us = clean test; GENERAL group partly seen = reported separately),
indep83 (examples.json from the same repo; its errors were seen while diagnosing, so it is a dev set), gupta challenge, held-out slices, English SMS F1."""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import exp_spam_indian as E
from data import ROOT
from spam_templates2 import make2
warnings.filterwarnings("ignore")
S = "/tmp/claude-0/-home-claude/93af4ad2-4824-5f04-bd0e-afd45a6faf9c/scratchpad/"
pre = pd.read_csv(ROOT / "data/external/india_precog2011_2k.csv"); jun = pd.read_csv(ROOT / "data/external/india_junioralive_2k.csv")
pre_tr, pre_ho = train_test_split(pre, test_size=0.25, stratify=pre.label, random_state=5); jun_tr, jun_ho = train_test_split(jun, test_size=0.25, stratify=jun.label, random_state=5)
tpl2 = pd.DataFrame(make2(), columns=["text", "label"])
P_all = pd.read_csv(S + "probes_all.csv"); probes_clean = P_all[P_all.grp.isin(["HINGLISH", "MISSPELLED"])]; probes_gen = P_all[P_all.grp == "GENERAL"]
ind = pd.read_csv(ROOT / "data/eval/spam_independent_83.csv"); ind = ind[ind.label.isin(["spam", "ham"])].drop_duplicates("text")
SETS = {"probes_clean52": probes_clean, "probes_general32": probes_gen, "indep83_dev": ind, "gupta": E.gup, "test3": pd.read_csv(ROOT / "data/eval/spam_indian_test3.csv"), "precog_holdout": pre_ho, "junior_holdout": jun_ho}

def score(name, extra, tag=""):
    model, r, P, hi = E.run(name, extra)
    en_thr = float(np.quantile(P(E.val[E.val.label == "ham"].text), 0.995)); out = {"variant": name, "en_thr": round(en_thr, 3), "hi_thr": round(hi, 3), "en_F1_at_en_thr": r["en_F1"]}
    for k, d in SETS.items():
        y = (d.label == "spam").astype(int).values; p = P(d.text)
        for thr_name, thr in (("0.25", 0.25), ("0.40", 0.40)):
            pr = p >= thr; out[f"{k}@{thr_name}"] = f"recall {pr[y == 1].mean():.2f} | false-alarm {pr[y == 0].mean():.2f} ({int(pr[y == 0].sum())}/{int((y == 0).sum())}) | AUC {roc_auc_score(y, p):.3f}"
    print(json.dumps(out, indent=1)); return model, P, out

if __name__ == "__main__":
    res = []
    for name, extra in (("D (live)", [E.ext, E.sms172, E.tpl]), ("F D + real Indian SMS", [E.ext, E.sms172, E.tpl, pre_tr, jun_tr]),
                        ("G D + templates2", [E.ext, E.sms172, E.tpl, tpl2]), ("H everything", [E.ext, E.sms172, E.tpl, tpl2, pre_tr, jun_tr])):
        m, P, o = score(name, extra); res.append(o)
    json.dump(res, open(ROOT / "results/spam_r2_variants.json", "w"), indent=1)
