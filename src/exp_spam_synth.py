"""Does SYNTHETIC Hinglish augmentation help the spam model? Tested only on the independent third-party Hinglish challenge set
(101 unique lines, never used for training or tuning) and the English SMS test.   python src/exp_spam_synth.py"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from data import ROOT, RAW, load_spam, _read_hinglid
from data2 import load_hinglish_sentiment, load_scam_challenge
from exp_spam import build
from text_norm import normalize

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42


def real_genuine_hinglish(n_train=1500):
    tr = [" ".join(s) for s in _read_hinglid(RAW / "hinglid" / "train.txt")]
    rng = np.random.default_rng(1); idx = rng.choice(len(tr), min(n_train, len(tr)), replace=False)
    return [tr[i] for i in idx]


def fit_eval(use_syn, syn_n=None):
    tr, te, _ = load_spam(); ch = load_scam_challenge()["hinglish"]
    fit, val = train_test_split(tr, test_size=0.2, stratify=tr.label, random_state=SEED)
    X, y = fit.text.tolist(), fit.label.tolist()
    if use_syn:
        syn = pd.read_csv(ROOT / "data" / "synthetic" / "hinglish_spam_synthetic.csv")
        if syn_n: syn = pd.concat([syn[syn.label == "spam"].head(syn_n), syn[syn.label == "ham"].head(syn_n)])
        gen = real_genuine_hinglish()
        X += syn.text.tolist() + gen; y += syn.label.tolist() + ["ham"] * len(gen)
    model = CalibratedClassifierCV(build("+ structural + lexicon"), method="sigmoid", cv=5).fit([normalize(t) for t in X], pd.Series(y).map({"ham": 0, "spam": 1}))
    P = lambda t: model.predict_proba([normalize(s) for s in t])[:, 1]
    en_thr = float(np.quantile(P(val[val.label == "ham"].text), 0.995))
    sents = _read_hinglid(RAW / "hinglid" / "validation.txt"); rng = np.random.default_rng(0); idx = rng.choice(len(sents), 4000, replace=False)
    real = [" ".join(sents[i]) for i in idx]; calib, held = real[:2000], real[2000:]
    ic, _ = load_hinglish_sentiment(); calib += ic.text.tolist()[:1500]
    hi_thr = float(np.quantile(P(calib), 0.99))
    yt = (te.label == "spam").astype(int).values; pe = P(te.text); yc = (ch.label == "spam").astype(int).values; pc = P(ch.text)
    return {"syn": use_syn, "syn_per_class": syn_n, "en_F1": f1_score(yt, pe >= en_thr), "en_recall": recall_score(yt, pe >= en_thr),
            "hi_challenge_recall": recall_score(yc, pc >= hi_thr), "hi_challenge_AUC": roc_auc_score(yc, pc),
            "hi_challenge_genuine_flagged": int(((pc >= hi_thr) & (yc == 0)).sum()), "real_genuine_hi_flag_rate": float((P(held) >= hi_thr).mean()), "hi_thr": hi_thr}


if __name__ == "__main__":
    rows = [fit_eval(False), fit_eval(True, 100), fit_eval(True, 300), fit_eval(True)]
    D = pd.DataFrame(rows).round(3); D.to_csv(R / "spam_synthetic_augmentation.csv", index=False); print(D.to_string())
