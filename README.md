# Multilingual Message Intelligence

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
| Language ID, real Hinglish tweets | **97.2%** accuracy |
| Spam, English SMS | **F1 0.960**, recall 96.0%, 5 of 900 genuine messages flagged |
| Spam, Hinglish scams | recall **53%** at about 1% false alarms (the weakest part, see below) |
| Hand-labelled set (250 messages, machine-assigned labels) | language 97.5% when answered, spam F1 0.70 (precision 1.00, recall 0.54) |
| Sentiment, Hinglish (3 classes) | macro-F1 **0.694** (95% CI 0.658 to 0.729); majority baseline 0.223 |
| Themes | word2vec + K-Means is the most stable representation on both corpora |
| Full pipeline | 3,921 real held-out messages, 63 messages per second on one CPU core |

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

| model       |   udhr (other languages) |   informal English (SMS incl. spam) |   English tweets (unfiltered; some are not English) |   real Hinglish (HingLID test) |   hand-written Hinglish (OOD) |
|:------------|-------------------------:|------------------------------------:|----------------------------------------------------:|-------------------------------:|------------------------------:|
| Naive Bayes |                    0.998 |                               1     |                                               0.978 |                          0.972 |                         1     |
| Linear SVM  |                    0.988 |                               0.998 |                                               0.975 |                          0.975 |                         1     |
| KNN         |                    0.971 |                               0.982 |                                               0.945 |                          0.968 |                         0.936 |

Deployed: **calibrated linear SVM with a reject option** (confidence threshold chosen on cross-validated training predictions; messages below it go to manual review instead of being guessed):

| test_set                                          |    n |   accuracy_all |   coverage |   accuracy_on_accepted |
|:--------------------------------------------------|-----:|---------------:|-----------:|-----------------------:|
| udhr (other languages)                            |  518 |          0.99  |      0.952 |                  0.998 |
| informal English (SMS incl. spam)                 | 1025 |          0.998 |      0.992 |                  1     |
| English tweets (unfiltered; some are not English) |  400 |          0.972 |      0.942 |                  0.989 |
| real Hinglish (HingLID test)                      |  400 |          0.972 |      0.98  |                  0.98  |
| hand-written Hinglish (OOD)                       |   78 |          1     |      0.987 |                  1     |

Accuracy by message length (hard sets):

|   index | words   |   accuracy_all |   coverage |   accuracy_on_accepted |   n |
|--------:|:--------|---------------:|-----------:|-----------------------:|----:|
|       0 | 1-3     |          0.957 |      0.957 |                  0.978 |  47 |
|       1 | 4-6     |          0.986 |      0.979 |                  0.986 | 288 |
|       2 | 7-10    |          0.982 |      0.976 |                  0.992 | 381 |
|       3 | 11-20   |          1     |      1     |                  1     | 351 |
|       4 | 21+     |          1     |      1     |                  1     | 436 |

### 2. Spam
Three-model comparison on English SMS (UCI, near-duplicates removed before splitting):

| model       |   accuracy |   spam_precision |   spam_recall |   spam_F1 |
|:------------|-----------:|-----------------:|--------------:|----------:|
| Naive Bayes |      0.991 |            0.983 |         0.944 |     0.963 |
| Linear SVM  |      0.99  |            0.983 |         0.936 |     0.959 |
| KNN         |      0.986 |            0.974 |         0.912 |     0.942 |

Deployed: TF-IDF + language-agnostic structural cues (URL, phone number, money, digits, capitals, exclamation marks) + a small scam-vocabulary list, calibrated.
English test: precision 0.960, recall 0.960, F1 0.960, AUC 0.993, expected calibration error 0.007, Brier 0.0101.
Thresholds use one method for both languages: the score above which at most 0.5% (English) or 1% (Hinglish) of **genuine** messages fall. The Hinglish threshold is set from real genuine Hinglish text, so it needs no Hinglish spam labels.

**Synthetic Hinglish augmentation.** No real Hinglish scam dataset exists, so `scripts/make_synthetic_hinglish.py` generates template-based Hinglish scams and everyday messages (clearly synthetic, training only) that are added to the English SMS data together with real genuine Hinglish tweets. The only test is the third-party Hinglish challenge set, which is never trained on. Thresholds are still set from real genuine text.

| training data              |   en_F1 |   hi_challenge_recall |   hi_challenge_AUC |   hi_challenge_genuine_flagged |   real_genuine_hi_flag_rate |
|:---------------------------|--------:|----------------------:|-------------------:|-------------------------------:|----------------------------:|
| English SMS only           |   0.956 |                 0.353 |              0.995 |                              0 |                       0.012 |
| + 100 synthetic per class  |   0.952 |                 0.51  |              0.973 |                              0 |                       0.011 |
| + 300 synthetic per class  |   0.964 |                 0.529 |              0.97  |                              0 |                       0.012 |
| + all synthetic (deployed) |   0.96  |                 0.529 |              0.966 |                              0 |                       0.012 |

Caveat: the synthetic templates and the scam word list were written after the challenge lines had been read, so that set is no longer a fully independent test. The honest conclusion is that augmentation helps and that real labelled messages are still needed.

**Does an English spam filter transfer to Hinglish and Hindi?** (third-party hand-written challenge sets, 101 unique messages each; indicative only)

| variant                   | test_set                     |    n |   AUC |   spam_recall |   spam_F1 |   false_alarm_rate |
|:--------------------------|:-----------------------------|-----:|------:|--------------:|----------:|-------------------:|
| word+char (baseline)      | English SMS test             | 1025 | 0.993 |         0.944 |     0.963 |              0.002 |
| word+char (baseline)      | Hinglish challenge           |  101 | 0.957 |         0.098 |     0.179 |              0     |
| word+char (baseline)      | Hindi (Devanagari) challenge |  101 | 0.938 |         0.235 |     0.381 |              0     |
| char only                 | English SMS test             | 1025 | 0.992 |         0.944 |     0.959 |              0.003 |
| char only                 | Hinglish challenge           |  101 | 0.96  |         0.118 |     0.211 |              0     |
| char only                 | Hindi (Devanagari) challenge |  101 | 0.934 |         0.255 |     0.406 |              0     |
| + structural cues         | English SMS test             | 1025 | 0.996 |         0.928 |     0.955 |              0.002 |
| + structural cues         | Hinglish challenge           |  101 | 0.981 |         0.392 |     0.563 |              0     |
| + structural cues         | Hindi (Devanagari) challenge |  101 | 0.965 |         0.373 |     0.543 |              0     |
| + structural + lexicon    | English SMS test             | 1025 | 0.996 |         0.936 |     0.959 |              0.002 |
| + structural + lexicon    | Hinglish challenge           |  101 | 0.993 |         0.49  |     0.658 |              0     |
| + structural + lexicon    | Hindi (Devanagari) challenge |  101 | 0.988 |         0.588 |     0.741 |              0     |
| structural + lexicon only | English SMS test             | 1025 | 0.985 |         0.912 |     0.901 |              0.016 |
| structural + lexicon only | Hinglish challenge           |  101 | 0.988 |         0.608 |     0.756 |              0     |
| structural + lexicon only | Hindi (Devanagari) challenge |  101 | 0.995 |         0.882 |     0.928 |              0.02  |

**What a few real Hinglish examples would add** (English + N Hinglish training examples, tested on the rest, 20 splits; optimistic because the lines are repetitive):

|   hinglish_training_examples |   AUC_mean |   AUC_std |   spam_F1_mean |   spam_F1_std |
|-----------------------------:|-----------:|----------:|---------------:|--------------:|
|                           10 |      0.996 |     0.004 |          0.772 |         0.055 |
|                           25 |      0.997 |     0.003 |          0.915 |         0.049 |
|                           50 |      0.998 |     0.003 |          0.965 |         0.024 |

**Evasion tricks** (letters spaced out, look-alike letters), English spam recall:

| attack             |   recall_no_defence |   recall_with_cleanup |
|:-------------------|--------------------:|----------------------:|
| clean              |               0.96  |                  0.96 |
| letters spaced     |               0.896 |                  0.96 |
| look-alike letters |               0.928 |                  0.96 |

### 3. Sentiment
**Hinglish**: IIIT-H corpus (real social-media comments, negative / neutral / positive). Model chosen by 5-fold cross-validation on the training set only; the test set is touched once.
Character n-grams beat word features because romanized Hindi has no fixed spelling. Naive Bayes, SVM and logistic regression are statistically tied; KNN is significantly worse (McNemar, `results/sentiment_hinglish_mcnemar.csv`).

| features   | model               |   cv_macro_F1 |   test_macro_F1 |   ci_low |   ci_high |   test_accuracy |   predict_ms_per_msg |
|:-----------|:--------------------|--------------:|----------------:|---------:|----------:|----------------:|---------------------:|
| word       | Naive Bayes         |         0.65  |           0.648 |    0.612 |     0.685 |           0.684 |                0.013 |
| word       | Linear SVM          |         0.67  |           0.635 |    0.596 |     0.672 |           0.671 |                0.017 |
| word       | Logistic Regression |         0.667 |           0.632 |    0.592 |     0.671 |           0.666 |                0.025 |
| word       | KNN                 |         0.595 |           0.581 |    0.541 |     0.62  |           0.633 |                0.064 |
| char       | Naive Bayes         |         0.687 |           0.693 |    0.656 |     0.728 |           0.718 |                0.055 |
| char       | Linear SVM          |         0.689 |           0.694 |    0.658 |     0.73  |           0.722 |                0.072 |
| char       | Logistic Regression |         0.681 |           0.687 |    0.652 |     0.721 |           0.705 |                0.111 |
| char       | KNN                 |         0.634 |           0.649 |    0.606 |     0.687 |           0.704 |                0.206 |
| word+char  | Naive Bayes         |         0.686 |           0.692 |    0.655 |     0.729 |           0.724 |                0.097 |
| word+char  | Linear SVM          |         0.691 |           0.694 |    0.658 |     0.729 |           0.719 |                0.094 |
| word+char  | Logistic Regression |         0.688 |           0.689 |    0.652 |     0.722 |           0.709 |                0.089 |
| word+char  | KNN                 |         0.632 |           0.626 |    0.586 |     0.664 |           0.668 |                0.217 |

**English**: the same model trained on one domain breaks on another (macro-F1; rows = training data, columns = test data). The deployed model is trained on tweets + short reviews:

| trained_on             |   movie reviews |   short reviews |   tweets |
|:-----------------------|----------------:|----------------:|---------:|
| movie reviews          |           0.875 |           0.743 |    0.604 |
| short reviews          |           0.714 |           0.793 |    0.625 |
| tweets                 |           0.562 |           0.576 |    0.759 |
| tweets + short reviews |           0.744 |           0.78  |    0.759 |

### 4. Themes
Representations compared by silhouette, stability under resampling (ARI), topic coherence (NPMI) and agreement with a label the clustering never saw (mean over k = 4 to 12):

| corpus                       | method           |   silhouette |   stability_ARI |   coherence_NPMI |   NMI vs held-out label |
|:-----------------------------|:-----------------|-------------:|----------------:|-----------------:|------------------------:|
| English SMS (genuine + spam) | NMF topics       |      nan     |           0.561 |            0.013 |                   0.133 |
| English SMS (genuine + spam) | tfidf+SVD+KMeans |        0.074 |           0.5   |            0.026 |                   0.214 |
| English SMS (genuine + spam) | word2vec+KMeans  |        0.077 |           0.786 |            0.073 |                   0.252 |
| Hinglish (IIIT-H comments)   | NMF topics       |      nan     |           0.479 |           -0.153 |                   0.042 |
| Hinglish (IIIT-H comments)   | tfidf+SVD+KMeans |        0.082 |           0.556 |           -0.173 |                   0.045 |
| Hinglish (IIIT-H comments)   | word2vec+KMeans  |        0.05  |           0.729 |           -0.069 |                   0.089 |

Deployed: word2vec trained on in-domain text (IDF-weighted average) + K-Means, k = 6.

English themes (genuine SMS):

|   cluster |   size | top_terms                               |
|----------:|-------:|:----------------------------------------|
|         0 |    686 | lor, wat, ok, wan, dun, thk             |
|         1 |   1267 | i'm, yeah, i'll, want, you're, gonna    |
|         2 |    816 | like, life, does, that's, world, people |
|         3 |    768 | send, pls, number, dont, ur, hi         |
|         4 |    763 | good, love, day, dear, great, hope      |
|         5 |    199 | gt, lt, min, decimal, mins, rs          |

Hinglish themes (real comments):

|   cluster |   size | top_terms                                      |
|----------:|-------:|:-----------------------------------------------|
|         0 |    252 | jai, bharat, pm, hind, jay, bihar              |
|         1 |    343 | baj, gaye, gye, jao, jaldi, aao                |
|         2 |   1203 | musalman, sale, tere, teri, mat, muslim        |
|         3 |    311 | love, ur, luv, yes, gud, uuuu                  |
|         4 |   1026 | plz, reply, apka, wait, fan, meri              |
|         5 |    741 | bhaijaan, khan, sallu, bajrangi, jaan, bhaijan |

### 5. End-to-end evaluation
A mixed stream of **3,921 real held-out messages** (no model saw any of them) from seven sources. Language: answered **96.3%** of messages, **99.1%** correct when answered.

Spam, with language predicted by the pipeline vs the true language supplied (the gap is the cost of language-ID mistakes):

| setting                    |   spam |   genuine |   recall |   precision |   genuine flagged |
|:---------------------------|-------:|----------:|---------:|------------:|------------------:|
| english predicted routing  |    125 |      1900 |    0.96  |       0.882 |                16 |
| english oracle routing     |    125 |      1900 |    0.96  |       0.876 |                17 |
| hinglish predicted routing |     51 |      1226 |    0.471 |       0.706 |                10 |
| hinglish oracle routing    |     51 |      1226 |    0.529 |       0.73  |                10 |

Sentiment, same comparison:

| language   | routing           |   answered |   accuracy |   macro-F1 |
|:-----------|:------------------|-----------:|-----------:|-----------:|
| english    | predicted routing |      0.93  |      0.746 |      0.748 |
| english    | oracle routing    |      0.988 |      0.747 |      0.747 |
| hinglish   | predicted routing |      0.912 |      0.682 |      0.659 |
| hinglish   | oracle routing    |      0.988 |      0.717 |      0.693 |

**Labelled set.** 250 real held-out messages (English and Hinglish messages, 11 other languages, scams) were labelled for language, spam and sentiment, and the whole pipeline was scored on them. The labels were assigned by Claude reading each message, not by independent human annotators, so call this set machine-labelled until two people verify it (`data/eval/e2e_labelled_machine.csv`, `src/eval_labelled.py`). One chain message with no clear language was excluded from the language score.

| | Result |
|---|---|
| Language | answered 94.8%, correct 97.5% when answered |
| Spam (English and Hinglish) | precision 1.00, recall 0.54, F1 0.70; 0 genuine messages flagged out of 193 |
| Spam recall by language | English 75% (8 scams), Hinglish 44% (18 scams) |
| Sentiment | English accuracy 71% (n=34), Hinglish accuracy 64% (n=95) |

Genuine-vs-spam labels for the large stream above (tweets and comments) are assumed (public social posts treated as genuine). The IIIT-H comments and the Hindi challenge set have no reliable language ground truth, so they are excluded from the language accuracy.

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
- **Hinglish spam is the weakest part.** No public dataset of real Hinglish scams exists (the repositories checked either hand-write or template-generate them). About half of Hinglish scams are caught at about 1% false alarms (up from a third before the synthetic augmentation); real labelled examples would still help most.
- The scam vocabulary list is hand-written domain knowledge and was shaped while looking at the challenge set, so its gain is indicative.
- Hindi in Devanagari is detected but not spam-checked, scored for sentiment or grouped into themes.
- Hinglish sentiment labels are noisy and from one domain (mostly comments on videos and political posts); expect lower scores elsewhere.
- Themes are unsupervised: cluster names are read from top terms and examples, not ground truth.
- The language model covers 17 classes; other languages are forced into the nearest class or rejected.
- No transformer results yet (the Colab run needs a Google sign-in).
- The 250-message labelled set has machine-assigned labels; independent human verification is still pending.

## Repository
`src/` pipeline and experiments (`final_*.py` train the deployed models, `exp_*.py` are the comparisons), `app/` Streamlit app, `tests/`, `results/` all outputs,
`notebooks/` Colab transformer track, `docs/` plan and research notes, `legacy` folders hold earlier prototype files kept for traceability.
