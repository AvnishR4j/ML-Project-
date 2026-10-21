"""Final English sentiment model: tweets + short reviews, negation handling (+ template negation examples).   python src/final_sentiment_en.py
Replaces models/sentiment.joblib; the comparison behind the choice is in results/sentiment_negation.csv."""
import joblib, json, warnings, pandas as pd
from sklearn.metrics import f1_score
from data import ROOT
from data2 import load_english_sentiment
from negation import NegPrep, learn_polarity, augment_negation
from exp_negation import en_model
warnings.filterwarnings("ignore")
dom = load_english_sentiment(); trd = pd.concat([dom["tweets"][0], dom["short reviews"][0]], ignore_index=True)
pol = learn_polarity(trd.text, trd.label, "positive", "negative"); a = augment_negation(pol)
m = en_model(NegPrep(pol)).fit(list(trd.text) + [s for s, _ in a], list(trd.label) + [l for _, l in a])
ch = pd.read_csv(ROOT / "data/eval/negation_challenge.csv"); ch = ch[ch.lang == "en"]
res = {k: float(f1_score(v[1].label, m.predict(v[1].text), average="macro")) for k, v in dom.items()}
p = m.predict(ch.text); res["negation_challenge_acc"] = float((p == ch.label.values).mean()); res["synthetic_examples"] = len(a)
joblib.dump(m, ROOT / "models" / "sentiment.joblib"); (ROOT / "results" / "sentiment_en_final.json").write_text(json.dumps(res, indent=2)); print(res)
