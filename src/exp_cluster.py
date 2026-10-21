"""Theme discovery experiment: which representation gives themes that are real, stable and readable?

   python src/exp_cluster.py
Methods:  tfidf+SVD(PCA)+KMeans | NMF topics | word2vec (trained on in-domain text) + KMeans
Metrics:  silhouette | bootstrap stability (ARI between clusterings of overlapping subsamples) |
          topic coherence (NPMI of each cluster's top terms) | NMI against labels the clustering never saw
"""
import json, re, warnings
import joblib, numpy as np, pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import Normalizer, normalize
from data import ROOT, RAW, load_spam, _read_hinglid
from embed import Embedder, tok, TOK
from data2 import load_hinglish_sentiment, load_english_sentiment, clean_social

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42


def corpora():
    h_tr, h_te = load_hinglish_sentiment(); hing = pd.concat([h_tr, h_te], ignore_index=True)
    s_tr, s_te, _ = load_spam(); sms = pd.concat([s_tr, s_te], ignore_index=True)
    sents = _read_hinglid(RAW / "hinglid" / "train.txt")
    rng = np.random.default_rng(0); idx = rng.choice(len(sents), min(60000, len(sents)), replace=False)
    w2v_hing = [tok(" ".join(sents[i])) for i in idx] + [tok(t) for t in hing.text]
    en = load_english_sentiment()
    w2v_en = [tok(t) for t in sms.text] + [tok(t) for t in en["tweets"][0].text[:6000]] + [tok(t) for t in en["short reviews"][0].text[:6000]]
    return {"Hinglish (IIIT-H comments)": (hing, "label", w2v_hing, None),
            "English SMS (genuine + spam)": (sms, "label", w2v_en, "english")}


def hinglish_stopwords(w2v_sents, n=120):
    from collections import Counter
    c = Counter(w for s in w2v_sents for w in s)
    return set(w for w, _ in c.most_common(n))


def top_terms(texts, labels, stop, k_terms=8):
    vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, stop_words=list(stop), min_df=3, max_df=0.4, sublinear_tf=True)
    X = vec.fit_transform(texts); terms = np.array(vec.get_feature_names_out()); out = {}
    for c in sorted(set(labels)):
        m = np.asarray(X[labels == c].mean(0)).ravel() - np.asarray(X[labels != c].mean(0)).ravel()   # distinctive, not just frequent
        out[c] = list(terms[np.argsort(-m)[:k_terms]])
    return out


def coherence(texts, labels, stop):
    docs = [[w for w in tok(t) if w not in stop] for t in texts]
    d = Dictionary(docs)
    tt = [v for v in top_terms(texts, labels, stop).values()]
    tt = [[w for w in t if w in d.token2id] for t in tt]; tt = [t for t in tt if len(t) >= 3]
    if not tt:
        return np.nan
    return float(CoherenceModel(topics=tt, texts=docs, dictionary=d, coherence="c_npmi").get_coherence())


def cluster_fn(method, texts, stop, emb):
    """returns function(index_array, k) -> labels over those indices (fit + assign)."""
    if method == "tfidf+SVD+KMeans":
        vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, stop_words=list(stop), min_df=2, max_df=0.5, sublinear_tf=True)
        Z = Normalizer().fit_transform(TruncatedSVD(50, random_state=SEED).fit_transform(vec.fit_transform(texts)))
    elif method == "word2vec+KMeans":
        Z = emb(texts)
    else:                                  # NMF topics
        vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, stop_words=list(stop), min_df=2, max_df=0.5, sublinear_tf=True)
        X = vec.fit_transform(texts)
        return lambda idx, k: NMF(k, init="nndsvda", random_state=SEED, max_iter=300).fit_transform(X[idx]).argmax(1), None
    return (lambda idx, k: KMeans(k, n_init=5, random_state=SEED).fit_predict(Z[idx])), Z


def main():
    rows = []
    for cname, (df, lab, w2v_sents, kind) in corpora().items():
        texts = df.text.tolist(); y = df[lab].values
        stop = set(ENGLISH_STOP_WORDS) if kind == "english" else hinglish_stopwords(w2v_sents) | set(ENGLISH_STOP_WORDS)
        emb = Embedder(w2v_sents)
        for method in ("tfidf+SVD+KMeans", "NMF topics", "word2vec+KMeans"):
            fn, Z = cluster_fn(method, texts, stop, emb)
            for k in (4, 6, 8, 10, 12):
                full = fn(np.arange(len(texts)), k)
                sil = float(silhouette_score(Z, full)) if Z is not None else np.nan
                rng = np.random.default_rng(SEED); aris = []
                for _ in range(4):
                    a = rng.choice(len(texts), int(0.8 * len(texts)), replace=False); b = rng.choice(len(texts), int(0.8 * len(texts)), replace=False)
                    common = np.intersect1d(a, b); la = dict(zip(a, fn(a, k))); lb = dict(zip(b, fn(b, k)))
                    aris.append(adjusted_rand_score([la[i] for i in common], [lb[i] for i in common]))
                rows.append({"corpus": cname, "method": method, "k": k, "silhouette": sil, "stability_ARI": float(np.mean(aris)),
                             "coherence_NPMI": coherence(texts, full, stop), "NMI_vs_" + lab: normalized_mutual_info_score(y, full),
                             "smallest_cluster_share": float(pd.Series(full).value_counts(normalize=True).min())})
                print(f"{cname[:10]:10s} {method:18s} k={k:2d} sil={sil:.3f} stab={np.mean(aris):.3f} coh={rows[-1]['coherence_NPMI']:.3f} nmi={rows[-1]['NMI_vs_label']:.3f}")
    pd.DataFrame(rows).to_csv(R / "cluster_experiment.csv", index=False)


if __name__ == "__main__":
    main()
