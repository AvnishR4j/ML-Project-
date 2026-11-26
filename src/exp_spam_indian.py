"""Spam model with external Indian-style scam data.   python src/exp_spam_indian.py
Training variants add public GitHub data (data/external/README.md). Judged on sets NEVER used for training:
  * gupta7050 challenge (101 unique third-party Hinglish lines)   * data/eval/spam_indian_test.csv (81 lines written by us: subsidy / KYC / OTP / lottery scams + genuine bank/OTP/family money messages)
  * pjbrav new_test_430 (same repo as part of the training data, so partly optimistic)   * English SMS test (regression check)"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from data import ROOT, RAW, load_spam, _read_hinglid
from data2 import load_hinglish_sentiment, load_scam_challenge
from exp_spam import build
from exp_spam_synth import real_genuine_hinglish
from text_norm import normalize
warnings.filterwarnings("ignore")
SEED = 42; R = ROOT / "results"
ext = pd.read_csv(ROOT / "data/external/pjbrav_hinglish_spam.csv"); sms172 = pd.read_csv(ROOT / "data/external/indian_scam_sms_172.csv")
mine = pd.read_csv(ROOT / "data/eval/spam_indian_test.csv"); fresh2 = pd.read_csv(ROOT / "data/eval/spam_indian_test2.csv"); from spam_templates import make; tpl = pd.DataFrame(make(), columns=["text", "label"]); new430 = pd.read_csv(ROOT / "data/external/pjbrav_new_test_430.csv"); gup = load_scam_challenge()["hinglish"]
tr, te, _ = load_spam(); fit, val = train_test_split(tr, test_size=0.2, stratify=tr.label, random_state=SEED)
syn = pd.read_csv(ROOT / "data/synthetic/hinglish_spam_synthetic.csv"); gen = real_genuine_hinglish()
sents = _read_hinglid(RAW / "hinglid" / "validation.txt"); rng = np.random.default_rng(0); idx = rng.choice(len(sents), 4000, replace=False)
real = [" ".join(sents[i]) for i in idx]; calib, held = real[:2000], real[2000:]; ic, _ = load_hinglish_sentiment(); calib += ic.text.tolist()[:1500]

def run(name, extra_sets, ext_w=1.0):
    X = fit.text.tolist() + syn.text.tolist() + gen; y = fit.label.tolist() + syn.label.tolist() + ["ham"] * len(gen)
    for d in extra_sets: X += d.text.tolist(); y += d.label.tolist()
    model = CalibratedClassifierCV(build("+ structural + lexicon"), method="sigmoid", cv=5).fit([normalize(t) for t in X], pd.Series(y).map({"ham": 0, "spam": 1}))
    P = lambda t: model.predict_proba([normalize(s) for s in t])[:, 1]
    en_thr = float(np.quantile(P(val[val.label == "ham"].text), 0.995)); hi_thr = float(np.quantile(P(calib), 0.99))
    r = {"variant": name, "hi_thr": round(hi_thr, 3), "real_genuine_hi_flag_rate": round(float((P(held) >= hi_thr).mean()), 4)}
    for k, d in (("gupta", gup), ("mine", mine), ("fresh2", fresh2), ("pjbrav430", new430)):
        y_ = (d.label == "spam").astype(int).values; p = P(d.text); pr = p >= hi_thr
        r[k] = {"AUC": round(float(roc_auc_score(y_, p)), 3), "spam_recall": round(float(pr[y_ == 1].mean()), 3), "genuine_flagged": f"{int(pr[y_ == 0].sum())}/{int((y_ == 0).sum())}"}
    ye = (te.label == "spam").astype(int).values; pe = P(te.text); r["en_F1"] = round(float(f1_score(ye, pe >= en_thr)), 3)
    print(json.dumps(r)); return model, r, P, hi_thr

if __name__ == "__main__":
    rows = []; keep = {}
    for name, sets in (("A current", []), ("C ext + Indian SMS 172", [ext, sms172]), ("D C + templates (genuine bank/OTP + scams)", [ext, sms172, tpl])):
        m, r, P, thr = run(name, sets); rows.append(r); keep[name] = (m, P, thr)
    json.dump(rows, open(R / "spam_indian_variants.json", "w"), indent=1)
    for name, (m, P, thr) in keep.items():
        print(name, [(t[:34], round(float(P([t])[0]), 2)) for t in ["I got 2 lakh subsidy so I need to give bank details", "I got 2 crore subsidy so I need to give bank details", "Mujhe 2 lakh ki subsidy mili hai, bank details dene hain", "Bhai kal 3 baje milte hain, canteen mein"]], "thr", round(thr, 3))

def show_errors():
    m, r, P, thr = run("D", [ext, sms172, tpl])
    for nm, d in (("fresh2", fresh2), ("gupta", gup)):
        p = P(d.text)
        for t, l, s in zip(d.text, d.label, p):
            if (s >= thr) != (l == "spam"): print(nm, ("MISSED SPAM " if l == "spam" else "FALSE ALARM "), round(float(s), 2), t[:105])
