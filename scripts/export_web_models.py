"""Export the deployed models to compact JSON so the website can run the pipeline in the browser.
Weights are quantised to int8/int16 (per-row scale); parity with the Python pipeline is checked by scripts/web_parity.py.
   python scripts/export_web_models.py  [out_dir]"""
import base64, json, sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import LANG_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "web" / "models"
OUT.mkdir(parents=True, exist_ok=True)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def vec(v, prep):
    terms = [None] * len(v.vocabulary_)
    for t, i in v.vocabulary_.items():
        terms[i] = t
    return {"an": v.analyzer, "ng": list(v.ngram_range), "prep": prep, "vocab": terms, "idf": b64(v.idf_.astype("<f4"))}


def quant(W, bits):
    """W: (rows, feats) -> scales (rows,), int array flattened FEATURE-MAJOR (feats, rows)."""
    qmax = 127 if bits == 8 else 32767
    sc = np.abs(W).max(axis=1) / qmax; sc[sc == 0] = 1.0
    q = np.rint(W / sc[:, None]).astype("<i1" if bits == 8 else "<i2")
    return sc.astype("<f4"), q.T.copy()


def dump(name, obj):
    p = OUT / f"{name}.json"; p.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False)); print(name, round(p.stat().st_size / 1e6, 2), "MB")


def main():
    M = ROOT / "models"
    # ---- language: char_wb 1-4 tfidf + 3 calibrated LinearSVC folds (17 classes)
    L = joblib.load(M / "language_final.joblib"); pipe = L["model"]; v = pipe.named_steps["f"]; cal = pipe.named_steps["m"]
    W = np.vstack([cc.estimator.coef_ for cc in cal.calibrated_classifiers_])             # (3*17, F)
    sc, q = quant(W, 16)
    dump("language", {"vec": vec(v, "lower"), "classes": [str(c) for c in pipe.classes_], "folds": len(cal.calibrated_classifiers_),
                      "scale": b64(sc), "w": b64(q), "bits": 16,
                      "b": [float(x) for cc in cal.calibrated_classifiers_ for x in cc.estimator.intercept_],
                      "a": [float(c.a_) for cc in cal.calibrated_classifiers_ for c in cc.calibrators],
                      "c": [float(c.b_) for cc in cal.calibrated_classifiers_ for c in cc.calibrators], "threshold": float(L["threshold"])})
    # ---- spam: 5 calibrated pipelines (word 1-2 + char_wb 2-4 tfidf + 9 structural features)
    S = joblib.load(M / "spam_final.joblib"); folds = []
    for cc in S["model"].calibrated_classifiers_:
        est = cc.estimator; tl = dict(est.named_steps["f"].transformer_list); m = est.named_steps["m"]
        coef = m.coef_[0]; sc, q = quant(coef[None, :], 16)
        folds.append({"w": vec(tl["w"], "lower"), "c": vec(tl["c"], "lower"), "scale": float(sc[0]), "coef": b64(q), "b": float(m.intercept_[0]),
                      "a": float(cc.calibrators[0].a_), "cb": float(cc.calibrators[0].b_)})
    dump("spam", {"folds": folds, "thresholds": S["thresholds"], "n_struct": 9})
    # ---- Hinglish sentiment (3 classes) and English sentiment (2 classes)
    for name, fn in (("sent_hi", "hinglish_sentiment.joblib"), ("sent_en", "sentiment.joblib")):
        P = joblib.load(M / fn); steps = dict(P.named_steps); f, m = steps["f"], steps["m"]
        prep = "prep" if getattr(f if hasattr(f, "preprocessor") else f.transformer_list[0][1], "preprocessor", None) is not None else "lower"
        if hasattr(f, "transformer_list"):
            tl = dict(f.transformer_list); vs = {"w": vec(tl["w"], prep), "c": vec(tl["c"], prep)}
        else:
            vs = {"w": vec(f, prep)}
        coef = np.atleast_2d(m.coef_); sc, q = quant(coef, 16)
        dump(name, {"vecs": vs, "classes": [str(c) for c in m.classes_], "scale": b64(sc), "w": b64(q), "b": [float(x) for x in np.atleast_1d(m.intercept_)]})
    # ---- themes
    labels = json.loads((ROOT / "results" / "theme_labels.json").read_text())
    for name, fn, key in (("theme_en", "theme_en.joblib", "en"), ("theme_hi", "theme_hinglish.joblib", "hinglish")):
        T = joblib.load(M / fn); e = T["emb"]; words = list(e.w2v.wv.index_to_key)
        V = np.vstack([e.w2v.wv[w] for w in words]); sc, q = quant(V, 8)
        dump(name, {"words": words, "scale": b64(sc), "vecs": b64(np.rint(V / sc[:, None]).astype("<i1")), "dim": V.shape[1],
                    "idf": b64(np.array([e.idf.get(w, 1.0) for w in words], dtype="<f4")), "centers": b64(T["km"].cluster_centers_.astype("<f4")),
                    "k": int(T["k"]), "names": list(T["names"]), "labels": labels.get(key, {})})
    dump("meta", {"lang_names": LANG_NAMES})


if __name__ == "__main__":
    main()
