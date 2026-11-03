"""Negation-aware text preprocessing for the sentiment models (picklable).

Words inside a negation scope (after not / never / n't / nahi ... up to punctuation or 3 words) are rewritten as
`not_<word>`, and also emit a `negflip_pos` / `negflip_neg` token telling the classifier that a word it knows as
positive/negative was negated. Word polarity comes from the training data itself (a first linear model), no hand-made lexicon.
"""
import re
from data2 import squeeze

EN_NEG = {"not", "no", "never", "nothing", "neither", "nor", "without", "cannot", "hardly", "barely", "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "cant", "wont", "wouldnt", "shouldnt", "couldnt", "aint", "nobody"}
HI_NEG = {"nahi", "nahin", "nhi", "nai", "na", "mat", "bilkul nahi", "kabhi nahi"}
_CONTR = [(r"\bcan'?t\b", "can not"), (r"\bwon'?t\b", "will not"), (r"\bain'?t\b", "is not"), (r"n['’]t\b", " not"), (r"\bcannot\b", "can not")]
_TOK = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)
_STOP = {"the", "a", "an", "is", "was", "are", "were", "be", "been", "am", "it", "this", "that", "to", "at", "all", "very", "so", "too", "really", "much", "any", "i", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "hai", "tha", "thi", "the", "ho", "hain", "hota", "hoti", "bahut", "bohot", "ek", "ye", "yeh", "ki", "ka", "ke"}


class NegPrep:
    def __init__(self, pol=None, scope=3, hindi=True):
        self.pol = dict(pol or {}); self.scope = scope; self.hindi = hindi

    def __call__(self, text):
        t = squeeze(text.lower())
        for a, b in _CONTR: t = re.sub(a, b, t)
        toks = _TOK.findall(t); out = []; left = 0
        negs = EN_NEG | (HI_NEG if self.hindi else set())
        for w in toks:
            if not re.match(r"\w", w): left = 0; out.append(w); continue
            if w in negs: left = self.scope; out.append(w); continue
            if left > 0:
                if w in _STOP: out.append(w); continue
                out.append("not_" + w); p = self.pol.get(w, 0)
                if p: out.append("negflip_pos" if p > 0 else "negflip_neg")
                left -= 1
            else: out.append(w)
        return " ".join(out)


def learn_polarity(texts, labels, pos, neg, top=1500, min_df=3):
    """Word polarity (+1/-1) from a plain linear model on the training data (|weight| in the top `top` words)."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    v = TfidfVectorizer(preprocessor=lambda s: squeeze(s.lower()), min_df=min_df, sublinear_tf=True)
    X = v.fit_transform(texts); y = np.array([1 if l == pos else -1 if l == neg else 0 for l in labels]); k = y != 0
    m = LinearSVC(C=0.3).fit(X[k], y[k]); w = m.coef_[0]; terms = v.get_feature_names_out()
    idx = np.argsort(-np.abs(w))[:top]
    return {terms[i]: (1 if w[i] > 0 else -1) for i in idx if terms[i] not in EN_NEG and terms[i] not in _STOP}


_NOUNS = ["food", "movie", "phone", "room", "service", "staff", "place", "product", "trip", "show", "app", "song", "hotel", "book", "game", "day", "view", "class", "teacher", "price", "coffee", "pizza", "film", "delivery", "battery"]


def augment_negation(pol, k=300, seed=0, per_word=4):
    """Template sentences built from the DATA-LEARNED polarity words: 'the X is W', 'the X is not W', 'I do not W the X', 'I W the X'.
    Label follows polarity and flips under 'not'. Training only."""
    import random
    rnd = random.Random(seed); out = []
    words = [w for w in pol if w.isalpha() and len(w) >= 3][:k]
    for w in words:
        y = pol[w]
        for _ in range(per_word):
            n = rnd.choice(_NOUNS); t = rnd.choice([0, 1, 2, 3])
            if t == 0: s, l = f"the {n} is {w}", y
            elif t == 1: s, l = f"the {n} is not {w}", -y
            elif t == 2: s, l = f"i do not {w} the {n}", -y
            else: s, l = f"i {w} the {n}", y
            out.append((s, "positive" if l > 0 else "negative"))
    return out
