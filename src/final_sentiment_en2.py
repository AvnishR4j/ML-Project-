"""Final English sentiment model v3 (customer-service gap closed).   python src/final_sentiment_en2.py
Training: tweets + short reviews + TweetEval train + SST-2 sentences + data-learned negation templates + VADER-word sentences (w 0.1)
          + customer-service templates (w 0.3).   Model: negation-aware word 1-2 + char_wb 3-5 TF-IDF -> LinearSVC (C=1).
Choice of this variant: results/sentiment_en_customer_variants.csv (customer DEV + TweetEval val). Customer TEST and TweetEval test scored once here.
NOTE: SST-2 sentences come from the same Rotten Tomatoes pool as 'short reviews', so the short-review score is inflated by overlap and is not claimed as an improvement."""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from data import ROOT
from exp_sentiment_en_customer import *   # noqa
from exp_sentiment_en_customer import fit, sst2, customer_templates, dev, test
warnings.filterwarnings("ignore")
tr = pd.concat([base, tw_tr], ignore_index=True); L = lexicon(); lexs = lex_sentences(L); ct = customer_templates(L); s2 = list(sst2().itertuples(index=False, name=None))
groups = [(lexs, 0.1), (ct, 0.3), (s2, 1.0)]
extra = [s for g, _ in groups for s in g]; ew = [w for g, w in groups for _ in g]
m, X, y = fit(tr, extra, 1.0, True); n0 = len(X) - len(extra)
m.fit(X, y, m__sample_weight=np.array([1.0] * n0 + ew))
prev = None
try:
    import sys; sys.path.insert(0, str(ROOT / "src")); from exp_sentiment_en_lex import fit as fit_old
    prev = fit_old(tr, lexs, 0.1)
except Exception as e: print("prev", e)
sets = dict(TESTS); sets["customer DEV (59)"] = dev; sets["customer TEST (40, scored once)"] = test
score = lambda mod: {k: round(float(f1_score(t.label, mod.predict(t.text), average="macro")), 4) for k, t in sets.items()}
res = {"before": score(prev), "after": score(m), "customer_extra": {"templates": len(ct), "sst2": len(s2)}}
res["tweeteval_val_F1"] = round(float(f1_score(tw_va.label, m.predict(tw_va.text), average="macro")), 4)
joblib.dump(m, ROOT / "models" / "sentiment.joblib"); (ROOT / "results" / "sentiment_en_final_v3.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
for t in ["hey your phone is really good and smooth", "The delivery was quick and the staff were really helpful", "Is the refund processed yet, my card was charged twice", "I am not happy with the delivery", "not bad at all", "the food was not good"]:
    print(m.predict([t])[0], "|", t)
for n, d in (("DEV", dev), ("TEST", test)):
    for t, l, q in zip(d.text, d.label, m.predict(d.text)):
        if l != q: print("  still wrong", n, "(true", l + "):", t)
