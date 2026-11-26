"""English sentiment: does more (and more varied) training data help?  python src/exp_sentiment_en_big.py
Baseline = current training mix (NLTK tweets + short reviews). Variants add the SemEval-2017 TweetEval training tweets.
Every variant is scored on the same held-out sets; the official TweetEval test split is never used to choose anything."""
import json, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data import ROOT
from data2 import load_english_sentiment, load_tweeteval
from negation import NegPrep, learn_polarity, augment_negation
warnings.filterwarnings("ignore")

FRESH = [  # everyday sentences written separately from every training source
 ("I am handsome", "positive"), ("You look beautiful today", "positive"), ("She is such a kind person", "positive"), ("I feel great", "positive"),
 ("What a lovely morning", "positive"), ("He is very smart and funny", "positive"), ("I love this place", "positive"), ("Thank you so much, you are the best", "positive"),
 ("My team won, I am so proud", "positive"), ("This food is delicious", "positive"), ("I am so excited for the trip", "positive"), ("We had a wonderful time",  "positive"),
 ("The staff were friendly and helpful", "positive"), ("That was an amazing performance", "positive"), ("I am proud of you", "positive"), ("Everything went perfectly", "positive"),
 ("I am ugly", "negative"), ("I feel terrible today", "negative"), ("He is rude and lazy", "negative"), ("I hate waiting in traffic", "negative"),
 ("What a horrible day", "negative"), ("This food is disgusting", "negative"), ("I am so tired of this", "negative"), ("The service was awful and slow", "negative"),
 ("I failed the exam and I feel bad", "negative"), ("Nobody cares about me", "negative"), ("This phone is a waste of money", "negative"), ("That was a boring movie", "negative"),
 ("I am very angry with you", "negative"), ("It was the worst trip ever", "negative"), ("I feel lonely and sad", "negative"), ("They were not helpful at all", "negative"),
]
fresh = pd.DataFrame(FRESH, columns=["text", "label"])
dom = load_english_sentiment()
tw_tr, tw_va, tw_te = load_tweeteval("train"), load_tweeteval("val"), load_tweeteval("test")
ch = pd.read_csv(ROOT / "data/eval/negation_challenge.csv"); ch = ch[ch.lang == "en"]
base = pd.concat([dom["tweets"][0], dom["short reviews"][0]], ignore_index=True)
TESTS = {"NLTK tweets": dom["tweets"][1], "short reviews": dom["short reviews"][1], "movie reviews (cross-domain)": dom["movie reviews"][1],
         "TweetEval test (SemEval-2017)": tw_te, "fresh everyday (32)": fresh, "negation challenge": ch[["text", "label"]]}

def build(train, C, cw=None, aug=True):
    pol = learn_polarity(train.text, train.label, "positive", "negative"); a = augment_negation(pol) if aug else []
    m = Pipeline([("f", TfidfVectorizer(preprocessor=NegPrep(pol), ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                  ("m", LinearSVC(C=C, random_state=42, class_weight=cw))])
    return m, list(train.text) + [s for s, _ in a], list(train.label) + [l for _, l in a]

def run(name, train, cw=None):
    best = None
    for C in (0.1, 0.3, 1.0):                      # chosen by cross-validated macro-F1 on TRAINING data only
        m, X, y = build(train, C, cw)
        p = cross_val_predict(m, X, y, cv=3); f = f1_score(y, p, average="macro")
        if best is None or f > best[0]: best = (f, C)
    m, X, y = build(train, best[1], cw); m.fit(X, y)
    r = {"variant": name, "train_n": len(train), "C": best[1], "cv_F1": round(best[0], 4)}
    for k, t in TESTS.items(): r[k] = round(f1_score(t.label, m.predict(t.text), average="macro"), 4)
    print(json.dumps(r)); return m, r

if __name__ == "__main__":
    rows = []
    for name, tr, cw in (("A baseline (NLTK tweets + short reviews)", base, None),
                         ("B + TweetEval train", pd.concat([base, tw_tr], ignore_index=True), None),
                         ("C + TweetEval train, balanced classes", pd.concat([base, tw_tr], ignore_index=True), "balanced")):
        m, r = run(name, tr, cw); rows.append(r)
        print("   ", {t: m.predict([t])[0] for t in ["I am handsome", "I am ugly", "You are handsome and kind"]})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "sentiment_en_bigdata.csv", index=False)
