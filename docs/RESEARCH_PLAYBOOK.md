# Multilingual Message Intelligence System: Research Playbook

Pipeline: message -> language detection -> spam filter -> sentiment -> theme clustering,
with a KNN / SVM / Naive Bayes comparison running across the first three steps.

Everything below comes from sources read on 2026-10-21. Where something is my inference
or I could not verify it, it is marked **(unverified)**.

---

## 1. The five findings that matter most

1. **The 20-language dataset has no romanized Hinglish.** Hindi there is Devanagari, and the text comes from Amazon reviews, XNLI and STSb, which is clean text. A model scoring ~99% on it says little about WhatsApp-style "kal milte hain, ok?" messages. The Hinglish part of language detection needs its own data (section 2).
2. **No public dataset has language + spam + sentiment labels on the same messages.** The end-to-end pipeline therefore needs a small test set that you build and hand-label yourselves (section 7). Say clearly in the report that it is hand-made.
3. **Realistic scores on Hinglish sentiment are ~60-75 weighted F1, not 90%.** SemEval-2020 Task 9 labels have only 55% inter-annotator agreement, and the winning system scored 75.0.
4. **Accuracy is the wrong headline metric for spam.** Report precision, recall and F1 for the spam class.
5. **The translated "multilingual" SMS spam dataset is unreliable.** Translation method is undocumented and some rows are garbled. Train on the English column; use translations only as a labelled stress test.

---

## 2. Module 1: Language detection

**Data**
- `papluca/language-identification` (Hugging Face): 20 languages, 70k train / 10k val / 10k test, balanced (3,500 / 500 / 500 per language). Fields `text`, `labels`. License/citation fields on the card are empty, so cite it by name and source datasets.
- Hinglish (romanized Hindi) data options:
  - L3Cube-HingCorpus: ~52.9M real Hindi-English code-mixed sentences in Roman script, scraped from Twitter. Code and data: github.com/l3cube-pune/code-mixed-nlp. A pretrained `hing-bert-lid` model exists under the `l3cube-pune` namespace on Hugging Face.
  - SemEval-2020 Task 9 Hinglish tweets (20k): word-level Hindi/English/Other tags, but tags came from an automatic tool, so they are noisy.
  - Microsoft LID-tool: word-level Hindi-English tagger. Needs Java, MALLET and Python 3. Useful as a baseline or for tagging, heavy to set up.

**Method plan**
- Features: character n-grams (1-4 or 2-5) with TF-IDF. Character n-grams beat word features for language ID, and a character-level CNN with attention reached 97-99% on noisy tweets in the 2019 short-text study, versus 92-95% for langid.py.
- Models: Naive Bayes, linear SVM, KNN (the comparison), plus fastText and/or XLM-R as the strong reference. A published XLM-R language-detection model reports 99.6% on the papluca test set.
- **Add a 21st class, "Hinglish (romanized Hindi)"**, built from HingCorpus/SemEval text. This is the part that makes the module non-trivial.

**Pitfalls**
- Short messages (<20 characters) are the hard case. Plot accuracy against message length.
- Romanized Hindi vs English is genuinely ambiguous for words like "to", "me", "a", "do". Expect confusion, and show it in a confusion matrix.
- A single-label output cannot represent mixed messages: the dominant language wins. State this limitation.
- Hindi vs Urdu is a classic confusable pair in this dataset; check it.

---

## 3. Module 2: Spam

**Data**
- UCI SMS Spam Collection: 5,574 English messages, ~13% spam. The 2022 SMS-scam study calls it outdated and incomplete. It is easy; do not treat a high score on it as impressive.
- Multilingual SMS Spam (Hugging Face `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`): the same 5.57k messages with 20 translated columns. Method undocumented (looks machine-translated), some rows garbled or padded with repeated text, license listed as "gpl". Use the English column for training.
- Hinglish scam dataset (`bolewara/hinglish-scam-text-dataset`): 3,787 rows, CC-BY-4.0, but only 63 Hinglish rows and those are hand-written examples. Good for a small Hinglish spam test, not for training.
- Hindi spam SMS (IEEE DataPort): 3,894 messages, but needs a subscription and the script is not stated.

**Method plan**
- TF-IDF (word + char n-grams) with Naive Bayes, SVM, KNN, logistic regression. Optionally fine-tune a transformer.
- Metric: spam-class precision / recall / F1, PR curve. Tune the threshold for a **low false-positive rate**, because a real message wrongly deleted is worse than spam let through. Report the chosen threshold and its trade-off.

**Pitfalls**
- **Deduplicate before splitting.** Duplicate SMS in train and test inflate scores (verified in the prototype: 403 of the 5,574 UCI messages are exact duplicate texts).
- Random splits from a single source are optimistic. The study above recommends cross-dataset evaluation: train on UCI, test on the Hinglish scam rows.
- Robustness (an easy extra that few teams do): the 2022 study found character-spacing attacks ("F R E E  p r i z e") evaded most deep models, and homoglyph attacks hurt most models except bag-of-words ones. Test your models on perturbed spam and report the drop.

---

## 4. Module 3: Sentiment

**Data**
- IMDB 50k reviews (English, binary). Use for the English / transformer baseline.
- SemEval-2020 Task 9, Hinglish: 14,000 train / 3,000 val / 3,000 test; classes positive 33%, neutral 38%, negative 29% in train. Metric used: weighted F1. Download via the Zenodo record linked from the task paper.
- Devanagari Hindi sentiment sets exist on Hugging Face (several MuRIL fine-tunes are public), but I did not verify any dataset's size or quality **(unverified)**.

**Reference scores (what "good" looks like)**
| Setup | Score | Source |
|---|---|---|
| IMDB, TF-IDF models (LR, SVC, NB, Ridge) | 0.877-0.899 acc | BERT-vs-traditional study |
| IMDB, BERT | 0.9387 acc | same |
| SemEval Hinglish, fine-tuned mBERT baseline | 65.4 weighted F1 | task paper |
| SemEval Hinglish, 2nd best system (XLM-R) | 72.6 | task paper |
| SemEval Hinglish, winner (XLM-R + adversarial training) | 75.0 | task paper |
| GLUECoS sentiment, Roman: XLM-R vs HingRoBERTa | 61.6 vs 66.4 F1 | HingCorpus paper |

Caveat: the BERT-vs-traditional paper's traditional baselines were mostly AutoML tools, so a carefully tuned TF-IDF + linear SVM may close part of that gap. Test this yourselves; it is a good result either way.

**Method plan**
- Classical: TF-IDF + NB / SVM / KNN / LR.
- Transformer: XLM-R base, and `l3cube-pune/hing-roberta` for romanized Hinglish. SemEval baseline settings that are known to work: AdamW, lr 2e-5, 3 epochs, max length ~56 tokens for tweets.
- **Decide your label scheme up front.** IMDB is binary, SemEval is 3-class. Either train separate heads or collapse SemEval (drop neutral, or merge). Write the decision into the report.

**Pitfalls**
- Domain mismatch: movie reviews vs tweets vs customer messages. A model trained on IMDB will not transfer cleanly. Measure it.
- Cross-lingual transfer (train English, test Hinglish): I did not find a trustworthy number for this pair **(unverified)**, so make it one of your own experiments and report the drop.
- Noisy labels cap the achievable score, so do not over-tune on the test set.

---

## 5. Module 4: Clustering and PCA

**Method plan**
- Represent messages two ways and compare: (a) TF-IDF -> PCA (or TruncatedSVD) -> K-Means, (b) multilingual sentence embeddings -> PCA -> K-Means. Candidate embedding models: `paraphrase-multilingual-MiniLM-L12-v2`, LaBSE. Their quality on romanized Hinglish is **(unverified)**; test it with a few known-similar pairs before committing.
- Choose k with silhouette score and the elbow method; show both.
- Optional upgrade: BERTopic, which can use K-Means in place of HDBSCAN.

**Practical tips from the BERTopic docs**
- Do not remove stop words before computing embeddings; remove them afterwards when building topic words.
- Compute embeddings once, save them to disk, and reuse.
- If you compare topics across languages, use the exact same multilingual embedding model throughout.
- Topic quality is judged mostly qualitatively: read top words and representative messages.

**Pitfalls**
- There is no ground-truth "theme" label, so say so. Validate with (a) silhouette, (b) NMI/ARI against language, spam and sentiment labels as a sanity check, and (c) manual reading of ~10 messages per cluster, with cluster names you assign and justify.
- Clusters may split by language rather than by topic, because that is the strongest signal. Check for this and consider clustering within one language.
- Plot PCA/UMAP 2-D scatter coloured by cluster for the demo.

---

## 6. Model comparison module (KNN / SVM / Naive Bayes)

- Same splits, same features, same metrics across all three tasks. Report accuracy, macro-F1, spam-class F1, training time and prediction time in one table.
- Tune with GridSearchCV or RandomizedSearchCV inside cross-validation (k for KNN, C and kernel for SVM, alpha for NB). Never tune on the test set.
- Expect KNN to be slowest at prediction time on TF-IDF vectors and weakest in high dimensions; show it with a timing plot. This kind of analysis is what the Iris idea is really about.
- Use stratified k-fold and report mean +/- std. Compare two models with a paired test (McNemar) or bootstrap confidence intervals.
- Fix random seeds and log library versions.

---

## 7. End-to-end pipeline evaluation

- **Build a hand-labelled mini test set**: ~200-300 short messages in English, Hindi and Hinglish, each with language, spam/not-spam and sentiment labels, mixing realistic complaints, promotions and scam-style texts. Label it jointly so you can check agreement. This is the only way to evaluate the whole chain.
- **Error propagation experiment**: compare the pipeline using predicted language vs the true language (oracle). The gap is the measurable cost of language-detection mistakes. This is the most distinctive result you can show.
- Calibration: report reliability curve and expected calibration error for the spam and sentiment models; calibrate (Platt / isotonic) before setting thresholds.
- Latency: messages per second for each stage and for the full pipeline.

---

## 8. Compute and tooling

- Classical models run on a laptop. XLM-R / HingRoBERTa fine-tuning needs a GPU: use free Colab or Kaggle notebooks, keep batch size small, max length 56-128, and save checkpoints to Drive.
- Libraries: pandas, scikit-learn, matplotlib/seaborn, `transformers`, `datasets`, `sentence-transformers`, `fasttext` (optional), Streamlit.
- Keep large datasets out of git; put download scripts in the repo instead.

---

## 9. Suggested timeline weights

Data + baselines (25%) -> transformers + Hinglish (25%) -> clustering (10%) -> comparison and tuning (15%) -> pipeline, error propagation, mini test set (15%) -> app, report, demo (10%).

---

## 10. What to say in the report

- A "Limitations" section: public datasets (not real company data), noisy Hinglish labels, hand-made pipeline test set, translated spam data, single-label language output.
- Honest numbers beat inflated ones; the literature scores above show Hinglish sentiment around 60-75 F1.

## Sources
- SemEval-2020 Task 9 overview: https://arxiv.org/pdf/2008.04277
- papluca/language-identification: https://huggingface.co/datasets/papluca/language-identification
- BERT vs traditional text classification: https://arxiv.org/pdf/2005.13012
- L3Cube-HingCorpus and HingBERT: https://arxiv.org/pdf/2204.08398v1
- Empirical analysis of SMS scam detection systems: https://arxiv.org/pdf/2210.10451
- Language ID on massive datasets of short messages: https://arxiv.org/pdf/1910.06748
- Hinglish emotion interpretability study: https://arxiv.org/pdf/2402.03137
- Microsoft LID-tool: https://github.com/microsoft/LID-tool
- Multilingual SMS spam set: https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset
- Hinglish scam text dataset: https://huggingface.co/bolewara/hinglish-scam-text-dataset
- BERTopic tips: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html
