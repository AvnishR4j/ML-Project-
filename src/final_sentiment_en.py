"""Final English sentiment model (replaces the earlier tweets + short-reviews model).   python src/final_sentiment_en.py
Training data: NLTK tweets + short reviews + SemEval-2017 TweetEval training tweets (positive/negative) + template negation examples
+ generated sentences from the VADER word list (weight 0.1, chosen on the TweetEval validation split).
Evidence for each choice: results/sentiment_en_bigdata.csv and results/sentiment_en_lexicon_weight.csv.
Model: negation-aware word 1-2 gram TF-IDF (min_df 2) + LinearSVC (C=1)."""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data import ROOT
from exp_sentiment_en_lex import TESTS, base, tw_tr, tw_va, lexicon, lex_sentences
from negation import NegPrep, learn_polarity, augment_negation
warnings.filterwarnings("ignore")
LEX_WEIGHT, C = 0.1, 1.0
tr = pd.concat([base, tw_tr], ignore_index=True)
pol = learn_polarity(tr.text, tr.label, "positive", "negative"); aug = augment_negation(pol); lexs = lex_sentences(lexicon())
X = list(tr.text) + [s for s, _ in aug] + [s for s, _ in lexs]; y = list(tr.label) + [l for _, l in aug] + [l for _, l in lexs]
sw = np.array([1.0] * (len(tr) + len(aug)) + [LEX_WEIGHT] * len(lexs))
m = Pipeline([("f", TfidfVectorizer(preprocessor=NegPrep(pol), ngram_range=(1, 2), sublinear_tf=True, min_df=2)), ("m", LinearSVC(C=C, random_state=42))])
m.fit(X, y, m__sample_weight=sw)
prev = joblib.load(ROOT / "models" / "sentiment_prev.joblib")
score = lambda mod: {k: round(float(f1_score(t.label, mod.predict(t.text), average="macro")), 4) for k, t in TESTS.items()}
res = {"before": score(prev), "after": score(m), "train_texts": len(tr), "negation_templates": len(aug), "lexicon_sentences": len(lexs), "lexicon_weight": LEX_WEIGHT,
       "tweeteval_val_F1": round(float(f1_score(tw_va.label, m.predict(tw_va.text), average="macro")), 4), "vocab": len(m.named_steps["f"].vocabulary_)}
joblib.dump(m, ROOT / "models" / "sentiment.joblib"); (ROOT / "results" / "sentiment_en_final.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
