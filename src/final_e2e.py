"""End-to-end evaluation on a mixed stream of REAL held-out test data (no model ever saw these messages).

   python src/final_e2e.py
Labels per source (what is known vs assumed is stated in the output):
  English SMS test        : language en (known)       spam (known)
  HingLID test windows    : language hinglish (known) spam = genuine (ASSUMED: public tweets)
  IIIT-H test comments    : language hinglish (assumed from corpus) spam = genuine (ASSUMED)  sentiment (known, 3 class)
  English tweets test     : language en (known)       spam = genuine (ASSUMED)                 sentiment (known, 2 class)
  UDHR test chunks        : language known (15 other languages)
  Hinglish / Hindi scam challenge sets (third-party hand-written): language, spam (known)
"""
import json, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from data import ROOT, load_language_v2, load_spam
from data2 import load_hinglish_sentiment, load_english_sentiment, load_scam_challenge
from pipeline import analyze, _models

warnings.filterwarnings("ignore")
R = ROOT / "results"; N_EN_TW = 1000


def build_stream():
    _, tests = load_language_v2(); _, te_sms, _ = load_spam(); _, h_te = load_hinglish_sentiment()
    en_tw = load_english_sentiment()["tweets"][1].sample(N_EN_TW, random_state=1); ch = load_scam_challenge()
    S = []
    add = lambda df, src, lang, spam=None, sent=None: S.extend(
        {"text": r.text, "source": src, "lang": getattr(r, "lang", lang), "spam": getattr(r, "spam", spam), "sent": getattr(r, "sent", sent)} for r in df.itertuples())
    add(te_sms.assign(spam=te_sms.label), "English SMS", "en")
    add(tests["real Hinglish (HingLID test)"], "Hinglish tweets (L3Cube)", "hinglish", "ham")
    add(h_te.assign(sent=h_te.label), "Hinglish comments (IIIT-H)", "hinglish", "ham")
    add(en_tw.assign(sent=en_tw.label), "English tweets", "en", "ham")
    u = tests["udhr (other languages)"]; add(u.assign(lang=u.label), "Other languages (UDHR)", None)
    add(ch["hinglish"].assign(spam=ch["hinglish"].label), "Hinglish scam challenge", "hinglish")
    add(ch["hindi"].assign(spam=ch["hindi"].label, lang="hi"), "Hindi scam challenge", None)
    return pd.DataFrame(S)


def run(df, oracle):
    t0 = time.time()
    res = [analyze(t, force_language=(l if oracle else None)) for t, l in zip(df.text, df.lang)]
    return res, time.time() - t0


def main():
    df = build_stream(); M = _models()
    print("stream size", len(df)); print(df.source.value_counts().to_string())
    pred, secs = run(df, False); orc, _ = run(df, True)
    df["pred_lang"] = [r["language"] for r in pred]; df["pred_spam"] = [r.get("spam") for r in pred]; df["pred_sent"] = [r.get("sentiment") for r in pred]
    df["orc_spam"] = [r.get("spam") for r in orc]; df["orc_sent"] = [r.get("sentiment") for r in orc]
    out = {"n": len(df), "seconds": secs, "messages_per_second": len(df) / secs}

    # ---- language (only sources whose language label is reliable; IIIT-H comments and the Hindi challenge set are code-mixed
    #      with no language ground truth, so they are excluded here but still flow through the rest of the pipeline)
    rel = df[~df.source.isin(["Hinglish comments (IIIT-H)", "Hindi scam challenge"])]
    acc = rel.pred_lang != "uncertain"
    out["language"] = {"n": int(len(rel)), "coverage": float(acc.mean()), "accuracy_on_accepted": float((rel.pred_lang[acc] == rel.lang[acc]).mean()),
                       "accuracy_overall_if_uncertain_counted_wrong": float((rel.pred_lang == rel.lang).mean())}
    out["language_by_source"] = {s: float((g.pred_lang == g.lang).mean()) for s, g in rel.groupby("source")}

    # ---- spam (only for languages the spam model covers: en/hinglish true-language messages)
    sp = df[df.spam.notna() & df.lang.isin(["en", "hinglish"])]
    def spam_stats(d, col):
        y = (d.spam == "spam"); p = (d[col] == "spam")
        tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
        return {"n_spam": int(y.sum()), "n_genuine": int((~y).sum()), "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                "genuine_flagged": fp, "false_alarm_rate": fp / max((~y).sum(), 1)}
    out["spam"] = {"english_predicted_routing": spam_stats(sp[sp.lang == "en"], "pred_spam"), "english_oracle_routing": spam_stats(sp[sp.lang == "en"], "orc_spam"),
                   "hinglish_predicted_routing": spam_stats(sp[sp.lang == "hinglish"], "pred_spam"), "hinglish_oracle_routing": spam_stats(sp[sp.lang == "hinglish"], "orc_spam")}

    # ---- sentiment (messages with a known sentiment label that were not rejected/filtered)
    for name, lang in (("english", "en"), ("hinglish", "hinglish")):
        d = df[df.sent.notna() & (df.lang == lang)]
        row = {}
        for col, tag in (("pred_sent", "predicted_routing"), ("orc_sent", "oracle_routing")):
            keep = d[col].notna()
            row[tag] = {"coverage": float(keep.mean()), "accuracy": float(accuracy_score(d.sent[keep], d[col][keep])), "macro_F1": float(f1_score(d.sent[keep], d[col][keep], average="macro", labels=sorted(d.sent.unique())))}
        out.setdefault("sentiment", {})[name] = row

    df.to_csv(R / "e2e_detail.csv", index=False)
    (R / "e2e_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
