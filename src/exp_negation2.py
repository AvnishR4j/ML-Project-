import warnings, pandas as pd, joblib
from sklearn.metrics import f1_score
from data import ROOT
from data2 import load_english_sentiment, prep
from negation import NegPrep, learn_polarity, augment_negation
from exp_negation import en_model
warnings.filterwarnings("ignore")
dom = load_english_sentiment(); ch = pd.read_csv(ROOT / "data/eval/negation_challenge.csv"); ch = ch[ch.lang == "en"]
def run(name, doms, aug, pp_neg=True):
    trd = pd.concat([dom[d][0] for d in doms], ignore_index=True)
    pol = learn_polarity(trd.text, trd.label, "positive", "negative")
    tx, ty = list(trd.text), list(trd.label)
    if aug:
        a = augment_negation(pol); tx += [s for s, _ in a]; ty += [l for _, l in a]
    m = en_model(NegPrep(pol) if pp_neg else prep).fit(tx, ty)
    r = {k: round(f1_score(v[1].label, m.predict(v[1].text), average="macro"), 3) for k, v in dom.items()}
    p = m.predict(ch.text); r["challenge_F1"] = round(f1_score(ch.label, p, average="macro"), 3); r["challenge_acc"] = round(float((p == ch.label.values).mean()), 3)
    print(name, r); return m
run("A neg, tweets+reviews", ["tweets", "short reviews"], False)
run("B neg + aug, tweets+reviews", ["tweets", "short reviews"], True)
run("C neg, all 3 domains", ["tweets", "short reviews", "movie reviews"], False)
run("D neg + aug, all 3", ["tweets", "short reviews", "movie reviews"], True)
