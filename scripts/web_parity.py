"""Parity check: the in-browser engine (web/engine.js) must reproduce the Python pipeline.   python scripts/web_parity.py"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pipeline import analyze

EDGE = ["", "ok", "   ", "नमस्ते दोस्तों, कैसे हो आप सब", "f r e e  p r i z e  now call 09061702893", "Clаim your frеe prіze now http://bit.ly/abc",
        "Bonjour, je voudrais réserver une table", "Hello!!! WIN Rs 50,000 lottery click www.win-now.in", "yaar ye film bahut mast thi, maza aa gaya!!!!",
        "Das ist ein sehr schönes Wetter heute", "你好，今天天气很好", "Привет, как дела?", "soooo gooood movieeee", "KYC update karein abhi: 9876543210 turant",
        "I love this so much 😍😍", "Order kab aayega? 3 din ho gaye", "Meeting kal 5 baje hai, please time par aana", "URGENT! Your account will be blocked. Verify OTP now 100%"]
rows = pd.read_csv(ROOT / "data" / "eval" / "e2e_labelled_machine.csv").text.tolist()
syn = pd.read_csv(ROOT / "data" / "synthetic" / "hinglish_spam_synthetic.csv").sample(80, random_state=1).text.tolist()
neg = pd.read_csv(ROOT / "data" / "eval" / "negation_challenge.csv").text.tolist()
THEME = ["powerbank me battery bohot kam hai", "mera order abhi tak nahi aaya", "kal baarish hone wali hai kya", "UPI payment fail ho gaya paise kat gaye", "blackhole ke andar kya hota hoga",
         "my phone battery dies in two hours", "I need to reset my bank card PIN", "how many calories are in a banana", "remind me to call mom at 6", "what is the capital of mongolia in asia",
         "photosynthesis converts sunlight into chemical energy in plants", "zzzz qqqq xxxx", "kal ka match kaun jeeta", "the movie was boring and way too long"]
_th = pd.read_csv(ROOT / "data" / "hinglish_topics_seed.csv").sample(120, random_state=3).text.tolist()
DEVA = ["यह फ़ोन बहुत अच्छा है", "आपका बैंक अकाउंट बंद हो जाएगा, अभी लिंक क्लिक करें", "Bhai, शादी की shopping कब करने जाएं?", "खराब सर्विस बकवास, पैसे वापस करो", "मुझे यह फिल्म बहुत पसंद आई",
        "बधाई हो! आपने ₹५०,००,००० की लॉटरी जीती है, ओटीपी भेजें", "कल बारिश होने वाली है क्या", "समझना मुश्किल है लेकिन कोशिश करो", "संपर्क करें 9876543210", "I love this, बहुत मस्त!"]
texts = EDGE + DEVA + THEME + _th + rows + syn + neg
exp = [analyze(t) for t in texts]
(ROOT / "web" / "parity_in.json").write_text(json.dumps(texts))
r = subprocess.run(["node", str(ROOT / "web" / "parity.js")], capture_output=True, text=True); got = json.loads(r.stdout)
bad = {"language": 0, "spam": 0, "sentiment": 0, "cluster": 0, "theme_rejected": 0}; maxd = {"conf": 0.0, "spam_prob": 0.0}; shown = 0
for t, e, g in zip(texts, exp, got):
    if e["language"] != g["language"]: bad["language"] += 1
    maxd["conf"] = max(maxd["conf"], abs(e["language_confidence"] - g["language_confidence"]))
    if (e.get("spam") or None) != (g.get("spam") or None): bad["spam"] += 1
    if "spam_prob" in e and "spam_prob" in g: maxd["spam_prob"] = max(maxd["spam_prob"], abs(e["spam_prob"] - g["spam_prob"]))
    if e.get("sentiment") != g.get("sentiment"): bad["sentiment"] += 1
    if e.get("cluster") != g.get("cluster"): bad["cluster"] += 1
    if e.get("cluster_label") != g.get("cluster_label"): bad["label"] = bad.get("label", 0) + 1
    if e.get("transliterated") != g.get("transliterated"): bad["translit"] = bad.get("translit", 0) + 1
    if "sentiment_margin" in e and "sentiment_margin" in g: maxd["margin"] = max(maxd.get("margin", 0.0), abs(e["sentiment_margin"] - g["sentiment_margin"]))
    if bool(e.get("theme_rejected")) != bool(g.get("theme_rejected")): bad["theme_rejected"] += 1
    if shown < 8 and (e["language"] != g["language"] or e.get("spam") != g.get("spam") or e.get("sentiment") != g.get("sentiment") or e.get("cluster") != g.get("cluster") or bool(e.get("theme_rejected")) != bool(g.get("theme_rejected"))):
        shown += 1; print("MISMATCH", repr(t[:70]), {k: e.get(k) for k in ("language", "spam", "sentiment", "cluster")}, {k: g.get(k) for k in ("language", "spam", "sentiment", "cluster")})
print("n", len(texts), "mismatches", bad, "max abs diff", maxd)
