"""Final spam model: structural cues + scam lexicon + TF-IDF, calibrated, with per-language thresholds.

Threshold method (same for both routes): the score above which at most X% of GENUINE messages of that
language fall.  English: 0.5% of English genuine SMS (validation).  Hinglish: 1% of REAL genuine Hinglish
text (L3Cube tweets + IIIT-H comments) - needs no Hinglish spam labels.
   python src/final_spam.py
"""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from data import ROOT, RAW, load_spam, _read_hinglid
from data2 import load_hinglish_sentiment, load_scam_challenge
from exp_spam import build
from text_norm import normalize, attack_spacing, attack_homoglyph

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42
EN_FPR, HI_FPR = 0.005, 0.01


def proba(model, texts, defend=True):
    return model.predict_proba([normalize(t) if defend else t for t in texts])[:, 1]


def main():
    tr, te, _ = load_spam(); ch = load_scam_challenge()["hinglish"]
    fit, val = train_test_split(tr, test_size=0.2, stratify=tr.label, random_state=SEED)
    base = build("+ structural + lexicon")
    # augmentation: SYNTHETIC Hinglish scams/genuine + real genuine Hinglish tweets (training only; see exp_spam_synth.py)
    from exp_spam_synth import real_genuine_hinglish
    syn = pd.read_csv(ROOT / "data" / "synthetic" / "hinglish_spam_synthetic.csv"); gen = real_genuine_hinglish()
    X = fit.text.tolist() + syn.text.tolist() + gen; y = fit.label.tolist() + syn.label.tolist() + ["ham"] * len(gen)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5).fit([normalize(t) for t in X], pd.Series(y).map({"ham": 0, "spam": 1}))
    # --- thresholds
    en_thr = float(np.quantile(proba(model, val[val.label == "ham"].text), 1 - EN_FPR))
    sents = _read_hinglid(RAW / "hinglid" / "validation.txt")
    rng = np.random.default_rng(0); idx = rng.choice(len(sents), 4000, replace=False)
    real = [" ".join(sents[i]) for i in idx]; calib, held = real[:2000], real[2000:]
    ic, _ = load_hinglish_sentiment(); calib += ic.text.tolist()[:1500]
    hi_thr = float(np.quantile(proba(model, calib), 1 - HI_FPR))
    # --- English test
    yt = (te.label == "spam").astype(int).values; pt = proba(model, te.text)
    pred = (pt >= en_thr).astype(int)
    en = {"precision": precision_score(yt, pred), "recall": recall_score(yt, pred), "F1": f1_score(yt, pred),
          "AUC": roc_auc_score(yt, pt), "genuine_flagged": int(((pred == 1) & (yt == 0)).sum()), "genuine_total": int((yt == 0).sum()),
          "threshold": en_thr}
    # --- Hinglish challenge + held-out real genuine Hinglish
    yc = (ch.label == "spam").astype(int).values; pc = proba(model, ch.text)
    hi = {"recall": recall_score(yc, (pc >= hi_thr).astype(int)), "AUC": roc_auc_score(yc, pc),
          "challenge_genuine_flagged": int(((pc >= hi_thr) & (yc == 0)).sum()), "challenge_genuine_total": int((yc == 0).sum()),
          "real_genuine_flagged_rate": float((proba(model, held) >= hi_thr).mean()), "threshold": hi_thr, "n_spam": int(yc.sum())}
    # --- calibration: reliability curve, expected calibration error, Brier score (English test)
    from sklearn.metrics import brier_score_loss
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(pt, bins) - 1, 0, 9)
    conf = [pt[idx == b].mean() for b in range(10) if (idx == b).any()]; frac = [yt[idx == b].mean() for b in range(10) if (idx == b).any()]
    w = [(idx == b).mean() for b in range(10) if (idx == b).any()]
    ece = float(sum(wi * abs(c - f) for wi, c, f in zip(w, conf, frac)))
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); ax.plot([0, 1], [0, 1], "--", color="grey"); ax.plot(conf, frac, "o-")
    ax.set_xlabel("predicted P(spam)"); ax.set_ylabel("observed spam share"); ax.set_title(f"Reliability (ECE {ece:.3f})"); fig.tight_layout(); fig.savefig(R / "spam_reliability.png", dpi=130); plt.close(fig)
    en["ECE"] = ece; en["Brier"] = float(brier_score_loss(yt, pt))
    # --- evasion robustness on English test spam
    spam_txt = te[te.label == "spam"].text.tolist(); rob = []
    for name, f in (("clean", lambda s: s), ("letters spaced", attack_spacing), ("look-alike letters", attack_homoglyph)):
        att = [f(s) for s in spam_txt]
        rob.append({"attack": name, "recall_no_defence": float((proba(model, att, False) >= en_thr).mean()),
                    "recall_with_cleanup": float((proba(model, att, True) >= en_thr).mean())})
    pd.DataFrame(rob).to_csv(R / "spam_final_robustness.csv", index=False)
    out = {"english_test": en, "hinglish": hi, "robustness": rob, "model": "TF-IDF(word+char) + structural cues + scam lexicon -> LinearSVC -> Platt calibration; trained with synthetic Hinglish augmentation"}
    (R / "spam_final_summary.json").write_text(json.dumps(out, indent=2))
    joblib.dump({"model": model, "thresholds": {"en": en_thr, "hinglish": hi_thr}, "defend": True}, ROOT / "models" / "spam_final.joblib")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
