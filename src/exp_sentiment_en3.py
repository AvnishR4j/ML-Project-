"""English sentiment with a NEUTRAL class (3 classes).   python src/exp_sentiment_en3.py
Why: the 2-class model had to call 'your watch is looking ok' positive. Neutral data: TweetEval neutral tweets (informational/plain tweets) +
neutral templates (ok / fine / average / plain facts). Variants chosen on data/eval/sent_en3_dev.csv + TweetEval val; sent_en3_test.csv scored once in final_sentiment_en3.py."""
import json, random, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from data import ROOT, RAW
from data2 import clean_social
from exp_sentiment_en_customer import customer_templates, sst2, lexicon, lex_sentences, base
from negation import NegPrep, learn_polarity, augment_negation
warnings.filterwarnings("ignore")
LAB = {"0": "negative", "1": "neutral", "2": "positive"}

def tweeteval3(split):
    txt = (RAW / "tweeteval" / f"{split}_text.txt").read_text(encoding="utf-8").splitlines(); lab = (RAW / "tweeteval" / f"{split}_labels.txt").read_text().split()
    df = pd.DataFrame([(clean_social(t), LAB[l]) for t, l in zip(txt, lab)], columns=["text", "label"]); df = df[df.text.str.len() > 0]
    df["key"] = df.text.str.lower().str.replace(r"\W+", "", regex=True); return df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)

NEU_ADJ = ["ok", "okay", "fine", "alright", "average", "decent", "acceptable", "ordinary", "normal", "standard", "regular", "fair", "so-so", "passable", "typical", "moderate"]
NEU_SUBJ = ["the phone", "the watch", "the laptop", "the delivery", "the service", "the food", "the room", "the app", "the website", "the hotel", "the battery", "the camera", "the screen", "the packaging", "the staff", "the class", "the movie", "the product", "the shoes", "the speaker"]
FACT = ["I ordered {x} {t}", "I bought {x} {t}", "I received {x} {t}", "{X} arrived {t}", "{X} is scheduled for {d}", "{X} opens at {h}", "I have been using {x} for {n}", "{X} was delivered {t}", "my appointment is on {d}", "the meeting is set for {d}"]
THINGS = ["a phone", "a laptop", "a watch", "the package", "the parcel", "an order", "a speaker", "a jacket", "a charger", "a keyboard"]
WHO = ["the agent", "the support team", "customer care", "the driver", "the technician", "the shop", "the bank", "the courier", "the hotel", "the teacher", "they"]
VERB = ["replied", "called", "emailed", "messaged", "answered", "responded", "confirmed", "visited"]
OBJ = ["my order number", "my address", "the receipt", "a photo of the item", "my phone number", "the booking id", "some details"]
PLACE = ["the hotel", "the mall", "the airport", "the office", "the clinic", "the store", "college", "the station"]
WHEN = ["tomorrow", "tonight", "on Sunday", "next week", "this evening", "after lunch", "in the morning"]
EVENT = ["{who} {v} {w}", "{who} {v} after two days", "{who} asked for {o}", "{who} asked me to send {o}", "{who} said it will take a few days", "I am going to {p} {w}", "we are going to {p} {w}", "I will visit {p} {w}",
         "a new update is available now", "the update is available now", "an update was released today", "the offer ends {w}", "I have used {x} for {n}", "I have owned {x} for {n}", "I signed up for the app last month", "the order id is in the email", "the plan renews {w}"]
HEDGE = ["it works fine, I guess", "{s} is {a} for the price", "{s} is {a} for daily use", "I have seen better and worse", "nothing exciting, it does the job", "it is fine, nothing exciting", "{s} is {a}, I guess", "{s} does the job", "{s} is {a} but nothing to write home about", "not much to say about {s}"]
def neutral_templates(k=1500, seed=3):
    r = random.Random(seed); out = []
    for _ in range(k):
        t = r.randrange(7); s = r.choice(NEU_SUBJ); a = r.choice(NEU_ADJ); h = r.choice(["", "hey ", "honestly, ", "I think "])
        if t == 0: x = f"{h}{s} is {a}"
        elif t == 1: x = f"{h}{s} was {r.choice(['', 'just ', 'pretty ', 'kind of '])}{a}"
        elif t == 2: x = f"{h}{s} is looking {a}"
        elif t == 3: x = f"{s} is {a}, nothing special"
        elif t == 4: x = f"{s} was {a}, not great, not bad"
        elif t == 5: x = r.choice(["not bad, not great either", "neither good nor bad", "it works, nothing more", "it does the job", "looks {} to me".format(r.choice(NEU_ADJ)), "seems {} to me".format(r.choice(NEU_ADJ)), "it is what it is"])
        elif t == 6 and r.random() < 0.5:
            x = r.choice(EVENT).format(who=r.choice(WHO), v=r.choice(VERB), o=r.choice(OBJ), p=r.choice(PLACE), w=r.choice(WHEN), x=r.choice(THINGS), n=r.choice(["a month", "a year", "two weeks", "six months", "three days"]))
        elif t == 5 and r.random() < 0.5:
            x = r.choice(HEDGE).format(a=r.choice(NEU_ADJ), s=s)
        else:
            x = r.choice(FACT).format(x=r.choice(THINGS), X=s, t=r.choice(["yesterday", "today", "this morning", "last week", "on Tuesday", "on Monday afternoon", "two days ago"]),
                                      d=r.choice(["Friday", "Monday", "Thursday", "tomorrow", "next week"]), h=r.choice(["nine", "ten in the morning", "eight in the evening"]), n=r.choice(["a month", "a year", "two weeks", "six months"]))
        out.append((x, "neutral"))
    return out

dev = pd.read_csv(ROOT / "data/eval/sent_en3_dev.csv"); tw_va = tweeteval3("val")

def build(neu_tw_w=1.0, neu_tpl=0, neu_tpl_w=0.3, char=True, C=1.0, ct_w=0.3, sst=True):
    tw = tweeteval3("train"); tr = pd.concat([base, tw], ignore_index=True)
    pol = learn_polarity(tr.text, tr.label, "positive", "negative"); aug = augment_negation(pol)
    L = lexicon(); parts = [(list(tr.text), list(tr.label), [1.0] * len(tr)), ([s for s, _ in aug], [l for _, l in aug], [1.0] * len(aug))]
    parts[0] = (list(tr.text), list(tr.label), [neu_tw_w if l == "neutral" else 1.0 for l in tr.label])
    lx = lex_sentences(L); ct = customer_templates(L); parts += [([s for s, _ in lx], [l for _, l in lx], [0.1] * len(lx)), ([s for s, _ in ct], [l for _, l in ct], [ct_w] * len(ct))]
    if sst: s2 = list(sst2().itertuples(index=False, name=None)); parts.append(([s for s, _ in s2], [l for _, l in s2], [1.0] * len(s2)))
    if neu_tpl: nt = neutral_templates(neu_tpl); parts.append(([s for s, _ in nt], [l for _, l in nt], [neu_tpl_w] * len(nt)))
    X = sum((p[0] for p in parts), []); y = sum((p[1] for p in parts), []); w = np.array(sum((p[2] for p in parts), []))
    wv = lambda: TfidfVectorizer(preprocessor=NegPrep(pol), ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    f = FeatureUnion([("w", wv()), ("c", TfidfVectorizer(preprocessor=NegPrep(pol), analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3))]) if char else wv()
    m = Pipeline([("f", f), ("m", LinearSVC(C=C, random_state=42))]); m.fit(X, y, m__sample_weight=w); return m

def score(m, d):
    p = m.predict(d.text); r = recall_score(d.label, p, average=None, labels=["negative", "neutral", "positive"])
    return {"macroF1": round(float(f1_score(d.label, p, average="macro")), 4), "rec_neg": round(r[0], 3), "rec_neu": round(r[1], 3), "rec_pos": round(r[2], 3)}

if __name__ == "__main__":
    rows = []
    for name, kw in (("C2 broader neutral templates, tweet-neutral w0.5", dict(neu_tw_w=0.5, neu_tpl=2500)), ("C3 same, tpl w0.5", dict(neu_tw_w=0.5, neu_tpl=2500, neu_tpl_w=0.5)), ("C4 tweet-neutral w0.3, tpl w0.5", dict(neu_tw_w=0.3, neu_tpl=2500, neu_tpl_w=0.5)),):
        m = build(**kw); d = score(m, dev); t = score(m, tw_va); r = {"variant": name, **{"dev_" + k: v for k, v in d.items()}, **{"tweetval_" + k: v for k, v in t.items()}}
        print(json.dumps(r)); rows.append(r)
        print("   ", [m.predict([x])[0] for x in ["your watch is looking ok", "hey your phone is really good and smooth", "the delivery was late and the staff were rude", "I ordered a phone yesterday"]])
    pd.DataFrame(rows).to_csv(ROOT / "results" / "sentiment_en3_variants.csv", index=False)
