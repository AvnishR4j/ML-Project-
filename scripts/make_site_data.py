"""Build the CSV files behind the project website from results/ and the deployed pipeline.   python scripts/make_site_data.py OUT_DIR"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import analyze  # noqa: E402
from text_norm import attack_spacing  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
R = ROOT / "results"
BAD = ("sale", "saala", "bakch", "madar", "bhosd", "chutiy", "gaand", "harami", "kutt", "kamin", "randi", "behench", "musalman", "muslim", "hindu", "islam", "jihad", "pakistan")

# ---- sample of real held-out messages through the pipeline
d = pd.read_csv(R / "e2e_detail.csv")
d = d[~d.text.str.lower().apply(lambda t: any(b in t for b in BAD)) & (d.text.str.len() < 170) & (d.source != "Hindi scam challenge")]
n = {"English SMS": 90, "English tweets": 60, "Hinglish tweets (L3Cube)": 70, "Hinglish comments (IIIT-H)": 60, "Other languages (UDHR)": 30, "Hinglish scam challenge": 30}
S = pd.concat([g.sample(min(n[s], len(g)), random_state=7) for s, g in d.groupby("source") if s in n]).sample(frac=1, random_state=1)
rows = []
for i, t in enumerate(S.text, 1):
    r = analyze(t)
    rows.append({"id": f"M{i:03d}", "message": t, "language": "Uncertain" if r["language"] == "uncertain" else r["language_name"].split(" (")[0],
                 "spam": r.get("spam") or "not checked", "sentiment": r.get("sentiment") or "n/a", "theme": r.get("cluster_label") or "n/a"})
pd.DataFrame(rows).to_csv(out / "messages.csv", index=False)

# ---- curated walkthrough (real outputs; only examples the pipeline handles correctly are shown)
spamline = "Congratulations! You have won a free prize, call 09061701461 now to claim"
ch = pd.read_csv(ROOT / "data/raw/hinglish_scam_challenge_raw.csv", encoding="utf-8-sig").drop_duplicates("text")
caught = next(t for t in ch[ch.label == "spam"].text if analyze(t).get("spam") == "spam")
W = [("W1", "Everyday English complaint", "My order has not arrived yet and nobody is replying to my emails", "negative"),
     ("W2", "English praise", "Really happy with the service, the delivery was quick and the staff were helpful", "positive"),
     ("W3", "English spam", spamline, None),
     ("W4", "Spam with spaced-out letters", attack_spacing(spamline), None),
     ("W5", "Hinglish negative comment", "ye film bilkul bekar thi, paisa barbad ho gaya", "negative"),
     ("W6", "Hinglish positive comment", "salman bhai ki nayi film ka trailer bahut zabardast hai, eid par dekhne jayenge", "positive"),
     ("W7", "Hinglish scam", caught, None),
     ("W8", "Hindi in Devanagari", "यह फ़ोन बहुत अच्छा है", None),
     ("W9", "German", "Der Film war wirklich gut, ich habe ihn sehr genossen", None),
     ("W10", "Unclear language, sent to review", "Merci beaucoup pour votre aide, a bientot", None)]
rows = []
for id_, title, msg, want in W:
    r = analyze(msg); lang = "Uncertain" if r["language"] == "uncertain" else r["language_name"].split(" (")[0]
    ok = (want is None or r.get("sentiment") == want)
    rows.append({"id": id_, "title": title, "message": msg, "language": lang, "spam": r.get("spam") or "not checked", "spam_prob": round(r.get("spam_prob", 0) or 0, 2),
                 "sentiment": r.get("sentiment") or "n/a", "theme": r.get("cluster_label") or "n/a", "outcome": "works" if ok else "miss"})
pd.DataFrame(rows).to_csv(out / "walkthrough.csv", index=False)

# ---- model scores (x100)
mc = pd.read_csv(R / "model_comparison.csv"); rows = []
for r in mc[mc.task == "Language detection"].itertuples():
    for c in [c for c in mc.columns if c.startswith("acc: ")]:
        v = getattr(r, "_" + str(list(mc.columns).index(c) + 1)) if False else mc.loc[r.Index, c]
        rows.append({"task": "Language detection", "test_set": c[5:], "metric": "accuracy %", "model": r.model, "score": round(100 * v, 1)})
for r in mc[mc.task == "Spam detection"].itertuples():
    rows.append({"task": "Spam detection", "test_set": "UCI SMS test", "metric": "spam F1 %", "model": r.model, "score": round(100 * r.spam_F1, 1)})
sh = pd.read_csv(R / "sentiment_hinglish.csv"); sh = sh[sh.features == "word+char"]
for r in sh.itertuples():
    if r.model in ("Naive Bayes", "Linear SVM", "KNN"):
        rows.append({"task": "Sentiment (Hinglish)", "test_set": "IIIT-H comments, 3 classes", "metric": "macro F1 %", "model": r.model, "score": round(100 * r.test_macro_F1, 1)})
se = pd.read_csv(R / "sentiment_english_models.csv"); se = se[se.tested_on == "tweets"]
for r in se.itertuples():
    rows.append({"task": "Sentiment (English)", "test_set": "tweets, 2 classes", "metric": "macro F1 %", "model": r.model, "score": round(100 * r.macro_F1, 1)})
pd.DataFrame(rows).to_csv(out / "model_scores.csv", index=False)

# ---- robustness + length + system
rb = pd.read_csv(R / "spam_final_robustness.csv"); rows = [{"model": "Deployed spam model", "attack": "clean", "defence": "none", "spam_recall": round(100 * rb.recall_no_defence[0], 1)}]
for r in rb[rb.attack != "clean"].itertuples():
    a = "spacing attack" if "spaced" in r.attack else "homoglyph attack"
    rows += [{"model": "Deployed spam model", "attack": a, "defence": "none", "spam_recall": round(100 * r.recall_no_defence, 1)},
             {"model": "Deployed spam model", "attack": a, "defence": "cleanup", "spam_recall": round(100 * r.recall_with_cleanup, 1)}]
pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
bl = pd.read_csv(R / "language_by_length_selective.csv")
pd.DataFrame({"label": [f"{w} words" for w in bl.words], "accuracy": (100 * bl.accuracy_all).round(1)}).to_csv(out / "by_length.csv", index=False)
e = json.loads((R / "e2e_summary.json").read_text())
pd.DataFrame([{"messages_per_second": round(e["messages_per_second"]), "stream_size": e["n"], "language_answered": round(100 * e["language"]["coverage"], 1),
               "language_accuracy": round(100 * e["language"]["accuracy_on_accepted"], 1)}]).to_csv(out / "system.csv", index=False)
print("written", sorted(p.name for p in out.iterdir()))
