"""English sentiment: close the customer-service domain gap.   python src/exp_sentiment_en_customer.py
Problem found: the deployed model misreads everyday customer wording ('the staff were really helpful', 'quick', 'unhelpful', 'ignored my emails').
Variants (all scored on data/eval/sent_en_customer_dev.csv; choice made on dev + TweetEval val; the customer TEST set is scored once at the end):
  A  current recipe                      B  + SST-2 sentences
  C  + customer-service templates        D  C + SST-2          E  D + char n-grams
Templates use ONLY the VADER word list (human-rated) and the data-learned negation words; labels are never taken from the test sets."""
import json, random, sys, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from data import ROOT
from exp_sentiment_en_lex import TESTS, base, tw_tr, tw_va, lexicon, lex_sentences
from negation import NegPrep, learn_polarity, augment_negation
warnings.filterwarnings("ignore")

dev = pd.read_csv(ROOT / "data/eval/sent_en_customer_dev.csv")
test = pd.read_csv(ROOT / "data/eval/sent_en_customer_test.csv")

def sst2():
    out = []
    for f in ("SST2_train.tsv", "SST2_dev.tsv"):
        for l in (ROOT / "data/raw/extra" / f).read_text(encoding="utf-8").splitlines():
            t, y = l.rsplit("\t", 1); out.append((t.replace(" n't", "n't").replace(" 's", "'s").replace(" ,", ",").replace(" .", "."), "positive" if y == "1" else "negative"))
    return pd.DataFrame(out, columns=["text", "label"])

SUBJ = ["the staff", "the delivery", "the service", "the support team", "the app", "the website", "the product", "the packaging", "the food", "the room",
        "the driver", "the agent", "the teacher", "the refund", "the price", "the quality", "the phone", "the battery", "the screen", "the camera", "the checkout", "the store", "the hotel", "the technician"]
INTENS = ["", "", "really ", "very ", "so ", "quite "]
def customer_templates(L, k=500, seed=1):
    """Customer-service sentences built from VADER-rated words. 'not <word>' flips the label; 'un<word>'-style words are already in VADER."""
    rnd = random.Random(seed); pos = [w for w, v in L.items() if v > 0]; neg = [w for w, v in L.items() if v < 0]
    # extra everyday service adjectives (not in VADER lists) with their polarity - written independently of the test sets
    for w in ("quick", "fast", "prompt", "speedy", "smooth", "helpful", "responsive", "reliable", "polite", "courteous", "punctual", "tidy", "spotless", "affordable", "convenient", "efficient", "professional", "friendly"): pos.append(w)
    for w in ("slow", "late", "unhelpful", "unresponsive", "unreliable", "rude", "careless", "dirty", "filthy", "overpriced", "inconvenient", "inefficient", "unprofessional", "unfriendly", "delayed", "damaged", "broken", "noisy", "laggy", "buggy"): neg.append(w)
    pos = sorted(set(pos)); neg = sorted(set(neg)); out = []
    def adj(y): return rnd.choice(pos if y > 0 else neg)
    for _ in range(k * 6):
        y = rnd.choice([1, -1]); s = rnd.choice(SUBJ); i = rnd.choice(INTENS); t = rnd.randrange(8)
        if t == 0: x = f"{s} was {i}{adj(y)}"
        elif t == 1: x = f"{s} was {i}{adj(y)} and {rnd.choice(SUBJ)} was {i}{adj(y)}"
        elif t == 2: x = f"{s} were {i}{adj(y)} and {adj(y)}"
        elif t == 3: x = f"{s} was not {adj(-y)}"
        elif t == 4: x = f"{rnd.choice(['hey ', 'wow, ', 'honestly, ', ''])}{s} is {i}{adj(y)}"
        elif t == 5: x = f"{adj(y)} {s.replace('the ', '')} and {adj(y)} {rnd.choice(SUBJ).replace('the ', '')}"
        elif t == 6: x = f"{rnd.choice(['I am', 'we are'])} {i}{'happy' if y > 0 else 'unhappy'} with {s}"
        else: x = f"{s} {rnd.choice(['came', 'arrived'])} {rnd.choice(['quickly', 'on time', 'early']) if y > 0 else rnd.choice(['late', 'damaged', 'very slowly'])}"
        out.append((x, "positive" if y > 0 else "negative"))
    return out[: k * 6]

def fit(train, extra, C=1.0, char=False, wts=None):
    pol = learn_polarity(train.text, train.label, "positive", "negative"); a = augment_negation(pol)
    X = list(train.text) + [s for s, _ in a] + [s for s, _ in extra]; y = list(train.label) + [l for _, l in a] + [l for _, l in extra]
    sw = np.array([1.0] * (len(train) + len(a)) + ([w for _, _, w in wts] if wts else []) )
    w = lambda: TfidfVectorizer(preprocessor=NegPrep(pol), ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    f = FeatureUnion([("w", w()), ("c", TfidfVectorizer(preprocessor=NegPrep(pol), analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3))]) if char else w()
    m = Pipeline([("f", f), ("m", LinearSVC(C=C, random_state=42))]); return m, X, y

def run(name, train, groups, C=1.0, char=False):
    """groups: list of (sentences, weight)."""
    extra = [s for g, _ in groups for s in g]; wlist = [(0, 0, wt) for g, wt in groups for _ in g]
    m, X, y = fit(train, extra, C, char, wlist); m.fit(X, y, m__sample_weight=np.array([1.0] * (len(X) - len(extra)) + [w for _, _, w in wlist]))
    sc = lambda d: round(float(f1_score(d.label, m.predict(d.text), average="macro")), 4)
    r = {"variant": name, "C": C, "customer_dev": sc(dev), "tweeteval_val": sc(tw_va), **{k: sc(t) for k, t in TESTS.items() if k != "TweetEval test (SemEval-2017)"}}
    print(json.dumps(r)); return m, r

if __name__ == "__main__":
    tr = pd.concat([base, tw_tr], ignore_index=True); L = lexicon(); lexs = lex_sentences(L); ct = customer_templates(L); s2 = list(sst2().itertuples(index=False, name=None))
    rows = []
    for name, groups, C, char in (("A current", [(lexs, 0.1)], 1.0, False), ("B + SST-2", [(lexs, 0.1), (s2, 1.0)], 1.0, False),
                                  ("C + customer templates w0.1", [(lexs, 0.1), (ct, 0.1)], 1.0, False), ("C2 + customer templates w0.3", [(lexs, 0.1), (ct, 0.3)], 1.0, False),
                                  ("D C2 + SST-2", [(lexs, 0.1), (ct, 0.3), (s2, 1.0)], 1.0, False), ("E D + char n-grams", [(lexs, 0.1), (ct, 0.3), (s2, 1.0)], 1.0, True),
                                  ("F D, C=0.5", [(lexs, 0.1), (ct, 0.3), (s2, 1.0)], 0.5, False)):
        rows.append(run(name, tr, groups, C, char)[1])
    pd.DataFrame(rows).to_csv(ROOT / "results" / "sentiment_en_customer_variants.csv", index=False)
