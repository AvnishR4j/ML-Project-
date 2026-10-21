# Multilingual Message Intelligence System (prototype v0.1)

A pipeline that takes a short message and returns its **language -> spam / not spam -> sentiment -> theme cluster**,
with a Naive Bayes / linear SVM / KNN comparison on every task. Built from five CodeChef ML project ideas:
Language Detection, Email Spam Classifier, Movie Review Sentiment Analyzer, Clustering & Dimensionality Reduction,
and the KNN / SVM / Naive Bayes algorithm comparison (Iris project).

## Run it
```
pip install -r requirements.txt
python scripts/download_data.py      # fetches the 3 datasets into data/raw
cd src && python train.py            # trains + compares models (about 1 minute), writes results/
cd .. && streamlit run app/streamlit_app.py
```

## Prototype results (test sets, tuned with 3-fold CV on train only)
| task                | model       |   accuracy |   macro_F1 | spam_F1   |   train_s |   predict_ms_per_msg |
|:--------------------|:------------|-----------:|-----------:|:----------|----------:|---------------------:|
| Language detection  | Naive Bayes |      0.998 |      0.998 | -         |      2.2  |                0.079 |
| Language detection  | Linear SVM  |      1     |      1     | -         |      1.05 |                0.061 |
| Language detection  | KNN         |      0.993 |      0.994 | -         |      1.07 |                0.132 |
| Spam detection      | Naive Bayes |      0.991 |      0.98  | 0.965     |      3.94 |                0.097 |
| Spam detection      | Linear SVM  |      0.99  |      0.978 | 0.961     |      3.65 |                0.103 |
| Spam detection      | KNN         |      0.985 |      0.966 | 0.94      |      4.66 |                0.369 |
| Sentiment (English) | Naive Bayes |      0.81  |      0.809 | -         |      7.94 |                0.62  |
| Sentiment (English) | Linear SVM  |      0.845 |      0.845 | -         |      6.97 |                0.762 |
| Sentiment (English) | KNN         |      0.71  |      0.709 | -         |      7.86 |                0.668 |

Details: language detection = 1243 train / 542 test pieces, 17 classes;
spam = UCI SMS, 403 duplicate messages removed before splitting;
sentiment = 2,000 NLTK movie reviews; clustering = 4516 genuine SMS, SVD(50) + K-Means (k=8).

## Read these numbers carefully
- Language ID uses the UN Declaration of Human Rights (easy, mostly different scripts) -> ~100% means little.
- The **Hinglish** class is 78 hand-written seed sentences from a single author (placeholder, not real data).
- Spam/sentiment/cluster models are English-only; sentiment trained on movie reviews transfers poorly to chat text.
- Clusters are weak (silhouette ~0.07-0.10, no clear elbow).

## Roadmap
See `docs/RESEARCH_PLAYBOOK.md` (datasets, methods, pitfalls, evaluation plan).
