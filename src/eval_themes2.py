"""Score the previous (v1, k=6) and new (v2) theme models on the SAME held-out topic-labelled messages, and write results/themes_v2_vs_v1.json.
   python src/eval_themes2.py"""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from data import ROOT
import themes_data as TD
warnings.filterwarnings("ignore")


def purity(y, c):
    df = pd.DataFrame({"y": y, "c": c}); return float(df.groupby("c").y.agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))


def main():
    out = {}
    for which, f1, f2 in (("en", "theme_en_v1.joblib", "theme_v2_en.joblib"), ("hinglish", "theme_hinglish_v1.joblib", "theme_v2_hinglish.joblib")):
        ev = TD.english()[1] if which == "en" else TD.hinglish()[1]; y = ev.topic.values; out[which] = {"n_test": len(ev), "n_topics": int(len(set(y)))}
        for tag, f in (("v1_k6", f1), ("v2", f2)):
            M = joblib.load(ROOT / "models" / f); lab = M["km"].predict(M["emb"](ev.text.tolist()))
            out[which][tag] = {"k": int(M["k"]) if "k" in M else int(M["km"].n_clusters), "NMI": normalized_mutual_info_score(y, lab), "ARI": adjusted_rand_score(y, lab), "purity": purity(y, lab)}
    (ROOT / "results" / "themes_v2_vs_v1.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
