"""English sentiment: add training sentences generated from the VADER word list (7.5k words rated by humans) so rare words such as
'handsome' are known.  python src/exp_sentiment_en_lex.py     Weight of the generated sentences is chosen on the TweetEval validation split."""
import json, os, warnings
import numpy as np, pandas as pd, vaderSentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data import ROOT
from exp_sentiment_en_big import base, tw_tr, tw_va, TESTS, fresh
from negation import NegPrep, learn_polarity, augment_negation
warnings.filterwarnings("ignore")

FRESH2 = [  # second unseen set, written after the weight was chosen
 ("She is gorgeous", "positive"), ("What a charming little town", "positive"), ("He is brilliant and generous", "positive"), ("I feel confident and strong today", "positive"),
 ("The view was breathtaking", "positive"), ("You are talented and hardworking", "positive"), ("This cake is tasty", "positive"), ("They were polite and welcoming", "positive"),
 ("I am grateful for everything", "positive"), ("It was a cheerful evening", "positive"), ("He looks elegant in that suit", "positive"), ("What a splendid idea", "positive"),
 ("She is arrogant and selfish", "negative"), ("What a dreadful smell", "negative"), ("He is stupid and cruel", "negative"), ("I feel worthless and weak today", "negative"),
 ("The room was filthy", "negative"), ("You are lazy and dishonest", "negative"), ("This soup is bland", "negative"), ("They were impolite and unfriendly", "negative"),
 ("I am miserable about everything", "negative"), ("It was a gloomy evening", "negative"), ("He looks sloppy in that shirt", "negative"), ("What a terrible idea", "negative"),
]
fresh2 = pd.DataFrame(FRESH2, columns=["text", "label"])
TESTS = dict(TESTS); TESTS["fresh everyday #2 (24)"] = fresh2

def lexicon(minabs=1.5):
    p = os.path.join(os.path.dirname(vaderSentiment.__file__), "vader_lexicon.txt"); L = {}
    for l in open(p, encoding="utf-8"):
        a = l.split("\t"); L[a[0]] = float(a[1])
    return {w: v for w, v in L.items() if w.isalpha() and len(w) >= 3 and abs(v) >= minabs}

FRAMES = ["{}", "so {}", "very {}", "I am {}", "he is {}", "she is {}", "this is {}", "it was {}", "you are {}", "what a {} day"]
def lex_sentences(L):
    out = []
    for w, v in sorted(L.items()):
        lab = "positive" if v > 0 else "negative"
        out += [(f.format(w), lab) for f in FRAMES]
    return out

def fit(train, lexs, wt, C=1.0):
    pol = learn_polarity(train.text, train.label, "positive", "negative"); a = augment_negation(pol)
    X = list(train.text) + [s for s, _ in a] + [s for s, _ in lexs]; y = list(train.label) + [l for _, l in a] + [l for _, l in lexs]
    sw = np.array([1.0] * (len(train) + len(a)) + [wt] * len(lexs))
    m = Pipeline([("f", TfidfVectorizer(preprocessor=NegPrep(pol), ngram_range=(1, 2), sublinear_tf=True, min_df=1)), ("m", LinearSVC(C=C, random_state=42))])
    m.fit(X, y, m__sample_weight=sw); return m

if __name__ == "__main__":
    tr = pd.concat([base, tw_tr], ignore_index=True); lexs = lex_sentences(lexicon()); print("lexicon sentences", len(lexs))
    rows = []
    for wt in (0.0, 0.1, 0.25, 0.5, 1.0):
        m = fit(tr, lexs if wt else [], wt or 1.0)
        r = {"lex_weight": wt, "tweeteval_val": round(f1_score(tw_va.label, m.predict(tw_va.text), average="macro"), 4)}
        for k, t in TESTS.items(): r[k] = round(f1_score(t.label, m.predict(t.text), average="macro"), 4)
        r["handsome"] = str(m.predict(["I am handsome"])[0]); rows.append(r); print(json.dumps(r))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "sentiment_en_lexicon_weight.csv", index=False)
