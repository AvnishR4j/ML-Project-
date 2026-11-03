import warnings, random, pandas as pd
from sklearn.metrics import f1_score
from data import ROOT
from data2 import load_hinglish_sentiment, prep
from negation import NegPrep, learn_polarity
from exp_negation import hi_model
warnings.filterwarnings("ignore")
tr, te = load_hinglish_sentiment(); ch = pd.read_csv(ROOT / "data/eval/negation_challenge.csv"); ch = ch[ch.lang == "hi"]
pol = learn_polarity(tr.text, tr.label, "positive", "negative", top=800, min_df=2)
N = ["film", "gaana", "khana", "phone", "acting", "video", "movie", "song", "service", "scene"]
def aug(k, per=4, seed=0):
    rnd = random.Random(seed); out = []
    for w in [w for w in pol if w.isalpha() and len(w) >= 3][:k]:
        y = pol[w]
        for _ in range(per):
            n = rnd.choice(N); t = rnd.choice([0, 1])
            out.append((f"yeh {n} {w} hai", y) if t == 0 else (f"yeh {n} {w} nahi hai", -y))
    return [(s, "positive" if l > 0 else "negative") for s, l in out]
for name, pp, a in (("baseline", prep, 0), ("neg", NegPrep(pol), 0), ("neg+aug150", NegPrep(pol), 150), ("neg+aug300", NegPrep(pol), 300)):
    tx, ty = list(tr.text), list(tr.label)
    if a: ax = aug(a); tx += [s for s, _ in ax]; ty += [l for _, l in ax]
    m = hi_model(pp).fit(tx, ty); p = m.predict(ch.text)
    print(name, round(f1_score(te.label, m.predict(te.text), average="macro"), 3), "challenge acc", round(float((p == ch.label.values).mean()), 3))
