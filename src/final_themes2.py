"""Final theme models v2 (word2vec-SIF + K-Means, one per language, many more topics, with an out-of-topic reject option).

   python src/final_themes2.py en 20      # language, k
   python src/final_themes2.py hi 16

Protocol (no leakage):
  fit set     90 % of the training pool       -> embedder (word2vec), first principal component, K-Means
  calib set   other 10 % of the pool          -> reject threshold = distance below which 95 % of in-topic messages fall
  test set    held-out messages, never seen   -> NMI / ARI / purity, in-topic acceptance
  OOD set     messages on topics the model never saw (CLINC150 'oos'; author-written Hinglish)  -> rejection rate, AUROC
The model is saved with a cluster report (size, top terms, topic mix, nearest messages) so that names can be written by reading it.
"""
import sys, json, warnings
import joblib, numpy as np, pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from embed import SIF, tok
from data import ROOT
import themes_data as TD

warnings.filterwarnings("ignore")
R = ROOT / "results"; SEED = 42
SEED_TOPICS = ["gadgets", "food", "delivery", "banking", "travel", "study_work", "health", "sports", "social", "weather_daily"]


def purity(y, c):
    df = pd.DataFrame({"y": y, "c": c}); return float(df.groupby("c").y.agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))


def dist_to_nearest(km, Z):
    d = ((Z[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1) ** .5; return d.min(1), d.argmin(1)


def top_terms(texts, labels, stop, n=8):
    vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, stop_words=list(stop), min_df=3, max_df=0.4, sublinear_tf=True)
    X = vec.fit_transform(texts); terms = np.array(vec.get_feature_names_out()); out = {}
    for c in sorted(set(labels)):
        m = np.asarray(X[labels == c].mean(0)).ravel() - np.asarray(X[labels != c].mean(0)).ravel(); out[c] = list(terms[np.argsort(-m)[:n]])
    return out


def main(which, K):
    english = which == "en"; key = "en" if english else "hinglish"
    if english:
        tr, ev, ood = TD.english(); extra = []
    else:
        tr, ev, ood, extra = TD.hinglish()
    rng = np.random.default_rng(SEED)
    # fit / calibration split (stratified by topic)
    cal_idx = np.concatenate([rng.choice(np.where(tr.topic.values == t)[0], max(2, int(.1 * (tr.topic.values == t).sum())), replace=False) for t in tr.topic.unique()])
    mask = np.zeros(len(tr), bool); mask[cal_idx] = True
    fit, cal = tr[~mask].reset_index(drop=True), tr[mask].reset_index(drop=True)
    fit_texts = fit.text.tolist()
    if not english:                                             # repeat the small author-written set 5x so it is not drowned out
        seed_rows = fit[fit.topic.isin(SEED_TOPICS)]; fit = pd.concat([fit] + [seed_rows] * 5, ignore_index=True); fit_texts = fit.text.tolist()
    w2v_sents = [tok(t) for t in fit_texts] + [tok(t) for t in extra]
    stop = (set(ENGLISH_STOP_WORDS) if english else set(w for w, _ in Counter(w for s in w2v_sents for w in s).most_common(100)) | set(ENGLISH_STOP_WORDS))
    emb = SIF(w2v_sents).fit_pc(fit_texts); Z = emb(fit_texts)
    km = KMeans(K, n_init=20, random_state=SEED).fit(Z); lab_fit = km.labels_
    # --- held-out test
    Ze = emb(ev.text.tolist()); de, le = dist_to_nearest(km, Ze); yev = ev.topic.values
    res = {"language": key, "k": K, "n_fit": len(fit_texts), "n_calib": len(cal), "n_test": len(ev),
           "NMI": normalized_mutual_info_score(yev, le), "ARI": adjusted_rand_score(yev, le), "purity": purity(yev, le)}
    # --- reject score: three candidates, chosen on a VALIDATION out-of-topic set; reported on a separate TEST out-of-topic set
    if english:
        ood_val, ood_test = ood.iloc[-200:].reset_index(drop=True), ood.iloc[:1000].reset_index(drop=True)          # CLINC oos_val+oos_train | oos_test
    else:
        g = ood.groupby("topic").cumcount(); ood_val, ood_test = ood[g < 3].reset_index(drop=True), ood[g >= 3].reset_index(drop=True)
    PROTO = {m: KMeans(m * K, n_init=1, random_state=SEED).fit(Z).cluster_centers_ for m in (10, 30, 60)}              # many small prototypes = local density
    d_fit, a_fit = dist_to_nearest(km, Z); scale = np.array([np.median(d_fit[a_fit == c]) if (a_fit == c).any() else 1.0 for c in range(K)])
    def scores(texts):
        V = emb(texts); d, a = dist_to_nearest(km, V)
        out = {"centroid": d, "centroid_scaled": d / scale[a]}
        for m, P in PROTO.items():
            out[f"prototype{m}"] = np.concatenate([(((V[i:i + 500, None, :] - P[None]) ** 2).sum(-1)).min(1) ** .5 for i in range(0, len(V), 500)])
        return out, emb.known(texts) > 0, a
    S_cal, k_cal, _ = scores(cal.text.tolist()); S_ev, k_ev, _ = scores(ev.text.tolist()); S_ov, k_ov, _ = scores(ood_val.text.tolist()); S_ot, k_ot, _ = scores(ood_test.text.tolist())
    aur = {m: float(roc_auc_score(np.r_[np.zeros(len(cal)), np.ones(len(ood_val))], np.r_[S_cal[m], S_ov[m]])) for m in S_cal}
    best = max(aur, key=aur.get); res["reject_score_candidates_AUROC_validation"] = aur; res["reject_score_chosen"] = best
    thr = float(np.quantile(S_cal[best][k_cal], 0.95)); res["reject_threshold"] = thr
    acc_in = (S_ev[best] <= thr) & k_ev; rej_out = ~((S_ot[best] <= thr) & k_ot)
    res.update({"in_topic_accepted": float(acc_in.mean()), "out_of_topic_rejected": float(rej_out.mean()), "n_ood_test": len(ood_test), "n_ood_validation": len(ood_val),
                "AUROC_test": {m: float(roc_auc_score(np.r_[np.zeros(len(ev)), np.ones(len(ood_test))], np.r_[S_ev[m], S_ot[m]])) for m in S_ev}})
    if not english:                                                       # author-written everyday topics vs real Hinglish-TOP / film-comment test messages
        sd = ev.topic.isin(SEED_TOPICS).values; res["in_topic_accepted_author_written_only"] = float(acc_in[sd].mean()); res["n_author_written_test"] = int(sd.sum())
    # per-source breakdown on the test set
    per = pd.DataFrame({"topic": yev, "c": le, "acc": acc_in}).groupby("topic").agg(n=("c", "size"), accepted=("acc", "mean"), purity=("c", lambda s: s.value_counts(normalize=True).iloc[0]))
    per["dominant_cluster"] = pd.Series(le).groupby(yev).agg(lambda s: int(s.value_counts().index[0])).reindex(per.index)
    per.to_csv(R / f"themes_v2_pertopic_{which}.csv")
    # --- cluster report (from the fit set; with topic mix)
    terms = top_terms(fit_texts, lab_fit[:len(fit_texts)] if len(lab_fit) >= len(fit_texts) else lab_fit, stop)
    ftexts = fit.text.tolist(); rows = []
    for c in range(K):
        idx = np.where(lab_fit == c)[0]; dd = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1) if len(idx) else np.array([])
        mix = pd.Series(fit.topic.values[idx]).value_counts(normalize=True).head(3)
        rows.append({"cluster": c, "size": len(idx), "top_terms": ", ".join(top_terms_c for top_terms_c in terms.get(c, [])[:6]),
                     "topic_mix": "; ".join(f"{t} {v:.0%}" for t, v in mix.items()), "examples": " || ".join(ftexts[i] for i in idx[np.argsort(dd)[:4]])})
    rep = pd.DataFrame(rows); rep.to_csv(R / f"themes_v2_clusters_{which}.csv", index=False)
    joblib.dump({"protos": PROTO.get(int(best[9:])) if best.startswith("prototype") else None, "reject_score": best, "scale": scale, "emb": emb, "km": km, "names": {c: ", ".join(terms.get(c, [])[:4]) for c in range(K)}, "k": K, "stop": stop, "reject_threshold": thr, "version": 2},
                ROOT / "models" / f"theme_v2_{key}.joblib")
    (R / f"themes_v2_summary_{which}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print(per.round(3).to_string()); print(rep[["cluster", "size", "top_terms", "topic_mix"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
