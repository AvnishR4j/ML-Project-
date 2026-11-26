"""Word embeddings trained on in-domain text (word2vec) and an IDF-weighted message embedder."""
import re
from collections import Counter
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

SEED = 42
TOK = re.compile(r"[a-z\u0900-\u097f']{2,}")


def tok(s):
    return TOK.findall(s.lower())


class Embedder:
    """IDF-weighted average of word2vec vectors."""
    def __init__(self, sents, dim=100):
        self.w2v = Word2Vec(sents, vector_size=dim, window=5, min_count=3, workers=1, epochs=15, seed=SEED, sg=1)
        df = Counter(w for s in sents for w in set(s)); N = len(sents)
        self.idf = {w: np.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    def __call__(self, texts):
        out = np.zeros((len(texts), self.w2v.vector_size))
        for i, t in enumerate(texts):
            ws = [w for w in tok(t) if w in self.w2v.wv]
            if ws:
                out[i] = np.average([self.w2v.wv[w] for w in ws], axis=0, weights=[self.idf.get(w, 1) for w in ws])
        return normalize(out)




class SIF(Embedder):
    """Smooth-inverse-frequency average of word2vec vectors (Arora et al., 2017) with the first principal component removed.
    weight(w) = a / (a + p(w));  message vector = weighted mean - its projection on the corpus-wide first component, then L2-normalised."""
    def __init__(self, sents, dim=100, a=1e-3):
        super().__init__(sents, dim)
        c = Counter(w for s in sents for w in s); n = sum(c.values()); self.a = a
        self.sif = {w: a / (a + v / n) for w, v in c.items()}; self.pc = None

    def _raw(self, texts):
        out = np.zeros((len(texts), self.w2v.vector_size))
        for i, t in enumerate(texts):
            ws = [w for w in tok(t) if w in self.w2v.wv]
            if ws:
                out[i] = np.average([self.w2v.wv[w] for w in ws], axis=0, weights=[self.sif.get(w, 1.0) for w in ws])
        return out

    def fit_pc(self, texts):
        from sklearn.decomposition import TruncatedSVD
        self.pc = TruncatedSVD(1, random_state=SEED).fit(self._raw(texts)).components_[0]; return self

    def __call__(self, texts):
        X = self._raw(texts)
        if self.pc is not None:
            X = X - np.outer(X @ self.pc, self.pc)
        return normalize(X)

    def known(self, texts):
        """number of words of each text that the embedder knows (0 means the vector is meaningless)"""
        return np.array([sum(w in self.w2v.wv for w in tok(t)) for t in texts])
