# Multilingual Message Intelligence System (prototype v0.3)

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

| model       |   udhr (other languages) |   informal English (SMS incl. spam) |   real Hinglish (HingLID test) |   hand-written Hinglish (OOD) |
|:------------|-------------------------:|------------------------------------:|-------------------------------:|------------------------------:|
| Naive Bayes |                    0.996 |                               0.999 |                          0.958 |                         1     |
| Linear SVM  |                    0.992 |                               0.993 |                          0.965 |                         0.987 |
| KNN         |                    0.988 |                               0.992 |                          0.912 |                         0.962 |

**Other tasks**

| task                | model       |   accuracy |   macro_F1 | spam_F1   |
|:--------------------|:------------|-----------:|-----------:|:----------|
| Spam detection      | Naive Bayes |      0.991 |      0.98  | 0.965     |
| Spam detection      | Linear SVM  |      0.99  |      0.978 | 0.961     |
| Spam detection      | KNN         |      0.985 |      0.966 | 0.94      |
| Sentiment (English) | Naive Bayes |      0.81  |      0.809 | -         |
| Sentiment (English) | Linear SVM  |      0.845 |      0.845 | -         |
| Sentiment (English) | KNN         |      0.71  |      0.709 | -         |

**Spam: calibration and threshold** (test set; threshold picked on validation data for <=0.5% genuine messages flagged)

| model                    | threshold          |    ECE |   Brier |   spam_precision |   spam_recall |   ham_flagged_as_spam |   n_ham_test |
|:-------------------------|:-------------------|-------:|--------:|-----------------:|--------------:|----------------------:|-------------:|
| Naive Bayes (raw)        | default 0.5        | 0.0087 |  0.008  |            0.984 |         0.954 |                     2 |          903 |
| Naive Bayes (raw)        | FPR<=0.5% (t=0.55) | 0.0087 |  0.008  |            0.984 |         0.947 |                     2 |          903 |
| Linear SVM (calibrated)  | default 0.5        | 0.0103 |  0.0075 |            0.992 |         0.931 |                     1 |          903 |
| Linear SVM (calibrated)  | FPR<=0.5% (t=0.27) | 0.0103 |  0.0075 |            0.962 |         0.962 |                     5 |          903 |
| Naive Bayes (calibrated) | default 0.5        | 0.0056 |  0.0078 |            0.984 |         0.954 |                     2 |          903 |
| Naive Bayes (calibrated) | FPR<=0.5% (t=0.41) | 0.0056 |  0.0078 |            0.984 |         0.954 |                     2 |          903 |

**Spam: robustness to evasion** (spam recall)

| model       | attack                               |   spam_recall_no_defence |   spam_recall_with_normalizer |
|:------------|:-------------------------------------|-------------------------:|------------------------------:|
| Naive Bayes | clean                                |                   0.947  |                        0.947  |
| Naive Bayes | spacing attack                       |                   0.908  |                        0.947  |
| Naive Bayes | homoglyph attack                     |                   0.954  |                        0.947  |
| Naive Bayes | both attacks                         |                   0.908  |                        0.947  |
| Naive Bayes | ham false-positive rate (clean text) |                   0.0022 |                        0.0022 |
| Linear SVM  | clean                                |                   0.939  |                        0.939  |
| Linear SVM  | spacing attack                       |                   0.779  |                        0.939  |
| Linear SVM  | homoglyph attack                     |                   0.901  |                        0.939  |
| Linear SVM  | both attacks                         |                   0.779  |                        0.939  |
| Linear SVM  | ham false-positive rate (clean text) |                   0.0022 |                        0.0022 |
| KNN         | clean                                |                   0.893  |                        0.893  |
| KNN         | spacing attack                       |                   0.863  |                        0.893  |
| KNN         | homoglyph attack                     |                   0.863  |                        0.893  |
| KNN         | both attacks                         |                   0.863  |                        0.893  |
| KNN         | ham false-positive rate (clean text) |                   0.0011 |                        0.0011 |

**End-to-end on a DRAFT 47-message test set** (written by the AI assistant; replace with real team-labelled data):
language accuracy 97.9%; end-to-end correct 85.1% with predicted language vs 87.2% with the true language.

## Read these numbers carefully
- The 15 non-Hinglish languages come from formal UN text and are easy; the informative language results are real Hinglish, informal English and the hand-written out-of-distribution lines.
- Hinglish spam is currently missed (the spam model is English-only), sentiment is trained on movie reviews, clusters are weak (silhouette ~0.08).
- The Hinglish class absorbs unfamiliar Latin-script text (a French test message was labelled Hinglish).
- The normalizer inverts the exact attacks it is tested against, so its full recovery is by construction.
- No transformer results yet.

## Roadmap
See `docs/RESEARCH_PLAYBOOK.md`.
