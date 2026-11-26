"""Final spam model v3 = variant I of exp_spam_r3.py (H + templates3). Thresholds: English from validation ham 99.5th pct, Hinglish fixed at 0.25."""
import json, warnings, joblib, numpy as np
import exp_spam_r3 as R3, exp_spam_r2 as X, exp_spam_indian as E
from data import ROOT
warnings.filterwarnings("ignore")
model, P, o = X.score("I final", R3.H + [R3.tpl3])
joblib.dump({"model": model, "thresholds": {"en": o["en_thr"], "hinglish": 0.25}, "defend": True}, ROOT / "models" / "spam_final.joblib")
json.dump(o, open(ROOT / "results" / "spam_final_v4.json", "w"), indent=1)
