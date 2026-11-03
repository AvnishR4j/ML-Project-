"""Theme discovery v2: choose representation and k by how well unsupervised clusters recover held-out topic labels.

   python src/exp_themes2.py [en|hi]

Clusters are fitted WITHOUT labels on the training pool. Held-out messages (never seen in training) are assigned to the
nearest centre and compared with their topic labels:
   NMI / ARI    agreement between clusters and topics (1 = perfect, 0 = chance)
   purity       share of messages whose cluster's majority topic is their own topic
   stability    ARI between two fits on different 80 % subsamples, measured on the held-out set
   baseline     the previous deployed model (k=6, SMS/IIIT-H only) scored on the same held-out set
"""
import sys, json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import Normalizer, normalize
from collections import Counter
from embed import Embedder, SIF, tok
from data import ROOT
import themes_data as TD

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42


def purity(y, c):
    df = pd.DataFrame({"y": y, "c": c}); return float(df.groupby("c").y.agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))


def metrics(y, lab):
    return {"NMI": normalized_mutual_info_score(y, lab), "ARI": adjusted_rand_score(y, lab), "purity": purity(y, lab)}


def main(which, only=None):
    if which == "en":
        tr, ev, ood = TD.english(); extra = []; weights = None
    else:
        tr, ev, ood, extra = TD.hinglish(); weights = None
    # the small author-written Hinglish set is repeated so that its topics are not drowned out (it is 3 % of the pool otherwise)
    if which == "hi":
        seed_mask = tr.topic.isin(["gadgets", "food", "delivery", "banking", "travel", "study_work", "health", "sports", "social", "weather_daily"])
        tr = pd.concat([tr] + [tr[seed_mask]] * 5, ignore_index=True)
    texts = tr.text.tolist(); ytr = tr.topic.values; w2v_sents = [tok(t) for t in texts] + [tok(t) for t in extra]
    stop = set(w for w, _ in Counter(w for s in w2v_sents for w in s).most_common(100))
    print(which, "train", len(texts), "eval", len(ev))
    reps = {}
    e = Embedder(w2v_sents); reps["w2v-IDF"] = (e, e(texts), e(ev.text.tolist()))
    s = SIF(w2v_sents); s.fit_pc(texts); reps["w2v-SIF"] = (s, s(texts), s(ev.text.tolist()))
    vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, min_df=2, max_df=0.5, sublinear_tf=True)
    svd = TruncatedSVD(100, random_state=SEED); Xt = Normalizer().fit_transform(svd.fit_transform(vec.fit_transform(texts))); Xe = Normalizer().transform(svd.transform(vec.transform(ev.text.tolist())))
    reps["tfidf+SVD"] = (None, Xt, Xe)
    rows = []; yev = ev.topic.values; rng = np.random.default_rng(SEED)
    for name, (_, Zt, Ze) in reps.items():
        if only and name not in only:
            continue
        for k in (6, 8, 10, 12, 14, 16, 20, 24):
            km = KMeans(k, n_init=5, random_state=SEED).fit(Zt); lab = km.predict(Ze)
            aris = []
            for _ in range(3):
                a = rng.choice(len(Zt), int(.8 * len(Zt)), replace=False); b = rng.choice(len(Zt), int(.8 * len(Zt)), replace=False)
                aris.append(adjusted_rand_score(KMeans(k, n_init=3, random_state=1).fit(Zt[a]).predict(Ze), KMeans(k, n_init=3, random_state=2).fit(Zt[b]).predict(Ze)))
            m = metrics(yev, lab); m.update({"rep": name, "k": k, "stability": float(np.mean(aris)), "silhouette": float(silhouette_score(Ze, lab, sample_size=2000, random_state=0)), "smallest_share": float(pd.Series(lab).value_counts(normalize=True).min())})
            rows.append(m); print(f"{name:10s} k={k:2d} NMI={m['NMI']:.3f} ARI={m['ARI']:.3f} pur={m['purity']:.3f} stab={m['stability']:.3f} sil={m['silhouette']:.3f}", flush=True)
    if only:
        print("(partial sweep, selection file not overwritten)"); out = pd.DataFrame(rows); out.to_csv(R / f"themes_v2_selection_{which}_{'_'.join(only)}.csv", index=False); return
    out = pd.DataFrame(rows)[["rep", "k", "NMI", "ARI", "purity", "stability", "silhouette", "smallest_share"]]; out.to_csv(R / f"themes_v2_selection_{which}.csv", index=False)
    # previous deployed model on the same held-out set
    old = joblib.load(ROOT / "models" / ("theme_en.joblib" if which == "en" else "theme_hinglish.joblib"))
    lab = old["km"].predict(old["emb"](ev.text.tolist())); b = metrics(yev, lab); print("PREVIOUS model (k=6):", b)
    (R / f"themes_v2_baseline_{which}.json").write_text(json.dumps({"previous_model_k6": b}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "en", sys.argv[2].split(",") if len(sys.argv) > 2 else None)
