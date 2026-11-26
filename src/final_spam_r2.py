"""Final spam model v3 = variant H of exp_spam_r2.py (all sources + templates2). Thresholds: English from validation ham 99.5th pct, Hinglish fixed at 0.25."""
import json, warnings, joblib, numpy as np
import exp_spam_r2 as X, exp_spam_indian as E
from data import ROOT
warnings.filterwarnings("ignore")
model, P, o = X.score("H final", [E.ext, E.sms172, E.tpl, X.tpl2, X.pre_tr, X.jun_tr])
joblib.dump({"model": model, "thresholds": {"en": o["en_thr"], "hinglish": 0.25}, "defend": True}, ROOT / "models" / "spam_final.joblib")
json.dump(o, open(ROOT / "results" / "spam_final_v3.json", "w"), indent=1)
