"""Rebuild README.md from results/*.csv|json so every number matches the last run.   python scripts/make_readme.py"""
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
res = R / "results"
csv = lambda n: pd.read_csv(res / n)
js = lambda n: json.loads((res / n).read_text())
md = lambda x: x.to_markdown(index=False)

mc = csv("model_comparison.csv")
lang = mc[mc.task == "Language detection"]
lang = lang[["model"] + [c for c in lang.columns if c.startswith("acc: ")]].round(3)
lang.columns = ["model"] + [c[5:] for c in lang.columns[1:]]
lsel = csv("language_selective.csv").round(3)
lenb = csv("language_by_length_selective.csv").round(3)
spam3 = mc[mc.task == "Spam detection"][["model", "accuracy", "spam_precision", "spam_recall", "spam_F1"]].round(3)
sp = js("spam_final_summary.json"); en, hi = sp["english_test"], sp["hinglish"]
tr = csv("spam_transfer.csv").round(3); fs = csv("spam_fewshot.csv").round(3)
rob = csv("spam_final_robustness.csv").round(3)
sh = csv("sentiment_hinglish.csv").drop(columns=["best_params"]).round(3)
sm = js("sentiment_summary.json"); H = sm["hinglish"]
xd = csv("sentiment_english_crossdomain.csv").pivot(index="trained_on", columns="tested_on", values="macro_F1").round(3).reset_index()
ce = csv("cluster_experiment.csv")
cbest = (ce.groupby(["corpus", "method"])[["silhouette", "stability_ARI", "coherence_NPMI"]].mean().round(3).reset_index())
nmi = [c for c in ce.columns if c.startswith("NMI_vs_")][0]
cbest["NMI vs held-out label"] = ce.groupby(["corpus", "method"])[nmi].mean().round(3).values
e2e = js("e2e_summary.json"); L = e2e["language"]
sp_rows = pd.DataFrame([{"setting": k.replace("_", " "), "spam": v["n_spam"], "genuine": v["n_genuine"], "recall": round(v["recall"], 3),
                         "precision": round(v["precision"], 3), "genuine flagged": v["genuine_flagged"]} for k, v in e2e["spam"].items()])
se_rows = pd.DataFrame([{"language": l, "routing": r.replace("_", " "), "answered": round(s["coverage"], 3), "accuracy": round(s["accuracy"], 3), "macro-F1": round(s["macro_F1"], 3)}
                        for l, v in e2e["sentiment"].items() for r, s in v.items()])
th_en = csv("themes_en.csv")[["cluster", "size", "top_terms"]]; th_hi = csv("themes_hinglish.csv")[["cluster", "size", "top_terms"]]

(R / "README.md").write_text(f"""# Multilingual Message Intelligence

Reads a short message in **English, Hindi or Hinglish** (Hindi typed in English letters) and decides
**language -> spam or genuine -> sentiment -> theme**. Built for a company that receives thousands of mixed-language messages and needs them sorted automatically.

```
message -> [1 language ID, with a reject option] -> [2 spam filter, per-language threshold]
        -> [3 sentiment: English | Hinglish]     -> [4 theme: word2vec + K-Means, English | Hinglish]
```

Combines five CodeChef ML project ideas: Language Detection System, Email Spam Classifier, Movie Review Sentiment Analyzer,
Clustering and Dimensionality Reduction, and the KNN / SVM / Naive Bayes comparison (used as the model-selection method on every classification task).

## Headline results (all on held-out data)
| Stage | Result |
|---|---|
| Language ID, real Hinglish tweets | **{lsel.loc[lsel.test_set.str.startswith('real'), 'accuracy_all'].iloc[0]:.1%}** accuracy |
| Spam, English SMS | **F1 {en['F1']:.3f}**, recall {en['recall']:.1%}, {en['genuine_flagged']} of {en['genuine_total']} genuine messages flagged |
| Spam, Hinglish scams | recall **{hi['recall']:.0%}** at about 1% false alarms (the weakest part, see below) |
| Sentiment, Hinglish (3 classes) | macro-F1 **{H['test_macro_F1']:.3f}** (95% CI {H['ci'][0]:.3f} to {H['ci'][1]:.3f}); majority baseline {H['majority_baseline_macroF1']:.3f} |
| Themes | word2vec + K-Means is the most stable representation on both corpora |
| Full pipeline | {e2e['n']:,} real held-out messages, {e2e['messages_per_second']:.0f} messages per second on one CPU core |

## Run it
```
pip install -r requirements.txt
python run_all.py                       # downloads data, trains, evaluates, rebuilds this README (about 15-20 min, CPU only)
streamlit run app/streamlit_app.py      # interactive app
pytest -q                               # data-leakage checks, normaliser, pipeline behaviour
```
Transformer track (needs a free Colab GPU): `notebooks/transformer_finetune_colab.ipynb` (same data and test split as the classical models; not yet run).

## Method and evidence

### 1. Language detection
Character n-gram TF-IDF with Naive Bayes, linear SVM and KNN, scored on separate test sets (a pooled average hides the hard cases):

{md(lang)}

Deployed: **calibrated linear SVM with a reject option** (confidence threshold chosen on cross-validated training predictions; messages below it go to manual review instead of being guessed):

{md(lsel)}

Accuracy by message length (hard sets):

{md(lenb.reset_index())}

### 2. Spam
Three-model comparison on English SMS (UCI, near-duplicates removed before splitting):

{md(spam3)}

Deployed: TF-IDF + language-agnostic structural cues (URL, phone number, money, digits, capitals, exclamation marks) + a small scam-vocabulary list, calibrated.
English test: precision {en['precision']:.3f}, recall {en['recall']:.3f}, F1 {en['F1']:.3f}, AUC {en['AUC']:.3f}, expected calibration error {en['ECE']:.3f}, Brier {en['Brier']:.4f}.
Thresholds use one method for both languages: the score above which at most 0.5% (English) or 1% (Hinglish) of **genuine** messages fall. The Hinglish threshold is set from real genuine Hinglish text, so it needs no Hinglish spam labels.

**Does an English spam filter transfer to Hinglish and Hindi?** (third-party hand-written challenge sets, 101 unique messages each; indicative only)

{md(tr)}

**What a few real Hinglish examples would add** (English + N Hinglish training examples, tested on the rest, 20 splits; optimistic because the lines are repetitive):

{md(fs)}

**Evasion tricks** (letters spaced out, look-alike letters), English spam recall:

{md(rob)}

### 3. Sentiment
**Hinglish**: IIIT-H corpus (real social-media comments, negative / neutral / positive). Model chosen by 5-fold cross-validation on the training set only; the test set is touched once.
Character n-grams beat word features because romanized Hindi has no fixed spelling. Naive Bayes, SVM and logistic regression are statistically tied; KNN is significantly worse (McNemar, `results/sentiment_hinglish_mcnemar.csv`).

{md(sh)}

**English**: the same model trained on one domain breaks on another (macro-F1; rows = training data, columns = test data). The deployed model is trained on tweets + short reviews:

{md(xd)}

### 4. Themes
Representations compared by silhouette, stability under resampling (ARI), topic coherence (NPMI) and agreement with a label the clustering never saw (mean over k = 4 to 12):

{md(cbest)}

Deployed: word2vec trained on in-domain text (IDF-weighted average) + K-Means, k = 6.

English themes (genuine SMS):

{md(th_en)}

Hinglish themes (real comments):

{md(th_hi)}

### 5. End-to-end evaluation
A mixed stream of **{e2e['n']:,} real held-out messages** (no model saw any of them) from seven sources. Language: answered **{L['coverage']:.1%}** of messages, **{L['accuracy_on_accepted']:.1%}** correct when answered.

Spam, with language predicted by the pipeline vs the true language supplied (the gap is the cost of language-ID mistakes):

{md(sp_rows)}

Sentiment, same comparison:

{md(se_rows)}

Genuine-vs-spam labels for tweets and comments are assumed (public social posts treated as genuine). The IIIT-H comments and the Hindi challenge set have no reliable language ground truth, so they are excluded from the language accuracy.

## Data
| Used for | Dataset | Licence / note |
|---|---|---|
| Language | UN Declaration of Human Rights (NLTK), 15 languages | formal text |
| Language, themes | L3Cube-HingLID (Nayak and Joshi, 2022), real romanized Hinglish tweets | CC BY-NC-SA 4.0, academic use, cite arXiv:2204.08398 |
| Language, spam, themes | UCI SMS Spam Collection | English |
| Sentiment (Hinglish) | IIIT-H Hindi-English code-mixed corpus (Prabhu et al., 2016) | 3,876 comments after de-duplication |
| Sentiment (English) | NLTK twitter_samples, sentence_polarity, movie_reviews | tweet labels come from emoticons (removed from the text) |
| Spam transfer test | github.com/gupta7050/multilingual-phishing-detection | 5,000 rows per file but only ~101 unique lines, de-duplicated; used for testing only |

## Limitations (read before quoting numbers)
- **Hinglish spam is the weakest part.** No public dataset of real Hinglish scams exists (the repositories checked either hand-write or template-generate them). About a third of Hinglish scams are caught at about 1% false alarms; the transfer experiment suggests roughly 50 real labelled examples would fix most of it.
- The scam vocabulary list is hand-written domain knowledge and was shaped while looking at the challenge set, so its gain is indicative.
- Hindi in Devanagari is detected but not spam-checked, scored for sentiment or grouped into themes.
- Hinglish sentiment labels are noisy and from one domain (mostly comments on videos and political posts); expect lower scores elsewhere.
- Themes are unsupervised: cluster names are read from top terms and examples, not ground truth.
- The language model covers 17 classes; other languages are forced into the nearest class or rejected.
- No transformer results yet.

## Repository
`src/` pipeline and experiments (`final_*.py` train the deployed models, `exp_*.py` are the comparisons), `app/` Streamlit app, `tests/`, `results/` all outputs,
`notebooks/` Colab transformer track, `docs/` plan and research notes, `legacy` folders hold earlier prototype files kept for traceability.
""")
print("README written")
