"""Train + save the deployed theme models (word2vec + KMeans), one per language, k=6 (chosen in exp_cluster.py)."""
import json, warnings
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from exp_cluster import corpora, Embedder, hinglish_stopwords, top_terms
from data import ROOT

warnings.filterwarnings("ignore")
R = ROOT / "results"; K = 6; SEED = 42


def main():
    summary = {}
    for cname, (df, lab, w2v_sents, kind) in corpora().items():
        english = kind == "english"
        key = "en" if english else "hinglish"
        stop = set(ENGLISH_STOP_WORDS) if english else hinglish_stopwords(w2v_sents) | set(ENGLISH_STOP_WORDS)
        d = df[df.label == "ham"] if english else df                  # themes are for genuine messages
        texts = d.text.tolist()
        emb = Embedder(w2v_sents); Z = emb(texts)
        km = KMeans(K, n_init=10, random_state=SEED).fit(Z)
        labels = km.labels_; terms = top_terms(texts, labels, stop)
        rows = []
        for c in range(K):
            idx = np.where(labels == c)[0]; dist = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1)
            reps = [texts[i] for i in idx[np.argsort(dist)[:3]]]
            rows.append({"cluster": c, "size": len(idx), "share": len(idx) / len(texts), "top_terms": ", ".join(terms[c][:6]), "examples": " || ".join(reps)})
        T = pd.DataFrame(rows); T.to_csv(R / f"themes_{key}.csv", index=False)
        joblib.dump({"emb": emb, "km": km, "names": {c: ", ".join(terms[c][:4]) for c in range(K)}, "k": K, "stop": stop}, ROOT / "models" / f"theme_{key}.joblib")
        P = PCA(2, random_state=SEED).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(6.5, 5)); ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=6, alpha=.6)
        ax.set_title(f"{cname}: word2vec + K-Means (k={K}), PCA to 2D"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout(); fig.savefig(R / f"themes_{key}.png", dpi=130); plt.close(fig)
        summary[key] = {"n": len(texts), "clusters": T[["cluster", "size", "top_terms"]].to_dict("records")}
        print(cname); print(T[["cluster", "size", "top_terms"]].to_string(index=False))
    (R / "themes_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
