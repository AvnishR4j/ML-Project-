"""Rebuild README.md from results/*.csv|json so the numbers always match the last run.   python scripts/make_readme.py"""
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
d = pd.read_csv(R / "results/model_comparison.csv")
lang = d[d.task == "Language detection"][["model"] + [c for c in d.columns if c.startswith("acc: ")]].round(3)
lang.columns = ["model"] + [c[5:] for c in lang.columns[1:]]
oth = d[d.task != "Language detection"][["task", "model", "accuracy", "macro_F1", "spam_F1"]].round(3).fillna("-")
ex = json.load(open(R / "results/experiments_summary.json")); p = ex["pipeline_eval"]
rob = pd.read_csv(R / "results/robustness.csv"); cal = pd.read_csv(R / "results/calibration_threshold.csv")
md = lambda x: x.to_markdown(index=False)
(R / "README.md").write_text(f"""# Multilingual Message Intelligence System (prototype v0.3)

A pipeline that takes a short message and returns its **language -> spam / not spam -> sentiment -> theme cluster**,
with a Naive Bayes / linear SVM / KNN comparison on every task. Built from five CodeChef ML project ideas:
Language Detection, Email Spam Classifier, Movie Review Sentiment Analyzer, Clustering & Dimensionality Reduction,
and the KNN / SVM / Naive Bayes algorithm comparison (Iris project).

## Run it
```
pip install -r requirements.txt
python scripts/download_data.py                 # SMS spam, UN-declaration texts, movie reviews -> data/raw
# real Hinglish data: copy train/validation/test.txt from github.com/l3cube-pune/code-mixed-nlp (folder L3Cube-HingLID)
#   into data/raw/hinglid/   (CC BY-NC-SA 4.0 - academic use; cite Nayak & Joshi 2022, arXiv:2204.08398)
cd src && python train.py && python experiments.py
cd .. && streamlit run app/streamlit_app.py
```
Optional: Hinglish sentiment -> download SemEval-2020 Task 9 (Zenodo 3974927) into `data/raw/semeval/`, run `python src/semeval.py`.
Transformers -> `notebooks/transformer_finetune_colab.ipynb` (needs a Colab GPU; written but not yet run).

## Dashboard
The app has a **Batch dashboard** tab: upload a CSV with a `text` column (or use the demo sample) to see language mix, spam share, themes and a filterable table, with a CSV download. Sentiment counts there are flagged unreliable until a domain-specific model is trained.

## Results
**Language detection** (accuracy per separate test set; training data: UN-declaration pieces for 15 languages, informal English SMS, real Hinglish tweets):

{md(lang)}

**Other tasks**

{md(oth)}

**Spam: calibration and threshold** (test set; threshold picked on validation data for <=0.5% genuine messages flagged)

{md(cal)}

**Spam: robustness to evasion** (spam recall)

{md(rob)}

**End-to-end on a DRAFT {p['n']}-message test set** (written by the AI assistant; replace with real team-labelled data):
language accuracy {p['language_accuracy']:.1%}; end-to-end correct {p['end_to_end_exact_match_predicted']:.1%} with predicted language vs {p['end_to_end_exact_match_oracle_language']:.1%} with the true language.

## Read these numbers carefully
- The 15 non-Hinglish languages come from formal UN text and are easy; the informative language results are real Hinglish, informal English and the hand-written out-of-distribution lines.
- Hinglish spam is currently missed (the spam model is English-only), sentiment is trained on movie reviews, clusters are weak (silhouette ~0.08).
- The Hinglish class absorbs unfamiliar Latin-script text (a French test message was labelled Hinglish).
- The normalizer inverts the exact attacks it is tested against, so its full recovery is by construction.
- No transformer results yet.

## Roadmap
See `docs/RESEARCH_PLAYBOOK.md`.
""")
