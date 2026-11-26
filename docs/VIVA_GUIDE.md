# Multilingual Message Intelligence: Viva & Demo Guide

Everything you need to explain, defend and demo the project. Read sections 1-4 first (understanding), then 8-10 (questions), then 11 (demo script).

Live demo: https://message-intelligence.vercel.app  |  Code: github.com/AvnishR4j/ML-Project-  (branch `prototype-v0.1`)

---

## 1. The project in 30 seconds

A company gets thousands of short messages in **English, Hindi and Hinglish** (Hindi typed in English letters, e.g. *"khana bohut achha tha"*). Reading them by hand is impossible, so the system reads each message and answers four questions in order:

1. **Which language is it?**
2. **Is it spam / a scam?**
3. **What is the sentiment?** (negative / neutral / positive)
4. **What is it about?** (theme, e.g. cards, delivery, battery)

Anything the models are unsure about is **not guessed**. It is sent to a human review queue. That "know when you don't know" idea is the main design principle.

It runs **entirely in the browser**: the trained models are exported as files, a small JavaScript engine runs them, and no message is sent to any server.

---

## 1b. Purpose: the problem, the users, the motivation

**Problem in one line (say this first):** "Businesses get thousands of mixed English, Hindi and Hinglish messages that no one can read by hand, and our system sorts them automatically by language, spam, sentiment and topic, sending anything it's unsure about to a human."

**Why it is needed**
- **Volume:** banks, e-commerce, telecom and helpdesks receive thousands of messages a day (SMS, WhatsApp, reviews, complaints).
- **Hinglish gap:** many Indian users write "mera order abhi tak nahi aaya". English tools can't read it and Hindi tools can't read Roman script.
- **Fraud:** KYC, subsidy and lottery scam messages are everywhere in India, and customers need warning before they click.

**Who would use it, and for what**
| User | Use |
|---|---|
| Customer support teams | Sort the inbox by sentiment and theme so angry delivery complaints reach a person first |
| Product / brand teams | See what customers talk about and how they feel, without reading every review |
| Banks and apps | Flag scam messages; tell a real bank alert from a fake KYC link |
| Moderators | Filter spam and see which messages need a human look |

**Why it works for them:** it admits uncertainty (unsure messages go to review instead of getting a confident wrong label), it runs in the browser (private, no server cost), and it shows why a message was flagged.

**Why you worked on it (honest framing):** it is a lab project that combines five ML ideas (language detection, spam, sentiment, clustering, and the KNN / SVM / Naive Bayes comparison) into one system that does something useful. Hinglish was chosen because few tools handle it, which makes the problem harder and more worth solving.

**Be ready to add:** it is a working prototype, not a finished product. Hinglish scam detection is the weakest part and needs more real labelled data before a company could rely on it.

---

## 2. Architecture

### 2.1 Pipeline

```
                      message (text)
                           |
                           v
                +---------------------+
                |  Cleaning           |  lowercase, fix spacing, undo tricks like
                |  (normaliser)       |  "f r e e" or "fr33"; Hindi script -> Roman letters
                +---------------------+
                           |
                           v
                +---------------------+      not confident
                | 1. LANGUAGE ID      | -----------------------> "Unclear / other" -> REVIEW
                | char n-grams + SVM  |
                +---------------------+
                           |  English / Hinglish
                           v
                +---------------------+
                | Unclear-text gate   |  too many unknown words (gibberish)
                +---------------------+ -----------------------> "Unclear text" -> REVIEW
                           |
                           v
                +---------------------+
                | 2. SPAM FILTER      |  per-language threshold
                | TF-IDF + cues + SVM |  (English 0.37, Hinglish 0.25)
                +---------------------+
                           |  genuine (spam messages stop here and are flagged)
                           v
                +---------------------+      score near the line / small margin
                | 3. SENTIMENT        | -----------------------> flagged for REVIEW
                | EN model | HI model |
                +---------------------+
                           |
                           v
                +---------------------+      far from every topic
                | 4. THEME            | -----------------------> "No clear theme" -> REVIEW
                | word2vec + K-Means  |
                +---------------------+
                           |
                           v
              result card + "why was this flagged"
```

### 2.2 How it is deployed (training vs running)

```
 TRAINING (Python, offline)                    RUNNING (user's browser)
 --------------------------                    ------------------------
 datasets -> scikit-learn / gensim             page loads  mdl/*.json  (about 18 MB, once)
      |                                              |
      v                                              v
 trained models (.joblib)                      engine.js  (same maths in JavaScript)
      |                                              |
      v                                              v
 scripts/export_web_models.py                  result shown on screen
   - weights shrunk to int16 / int8            nothing leaves the device
   - gzip + base64 into JSON files
      |
      v
 site/mdl/*.json  -> Vercel (static hosting, auto-deployed from GitHub)
```

Why this matters in the viva: the Python models and the JavaScript engine were checked against each other. **Scores matched exactly (0 disagreements)**, so the website behaves like the trained model.

---

## 3. The four stages: what, how, why

### 3.1 Language identification
- **Task:** English, Hinglish, Hindi and other languages (17 classes in total).
- **Features:** character n-grams (1-4 letters) with TF-IDF. Letters work even when spelling is messy, which is exactly the Hinglish problem (no fixed spelling).
- **Model:** linear SVM, **calibrated** so its score behaves like a probability.
- **Reject option:** if confidence is below a threshold (0.73, chosen on cross-validation), the system says "unsure" instead of guessing.
- **Why SVM:** compared Naive Bayes, linear SVM and KNN on five separate test sets. NB and SVM were close; KNN was worse and slower. SVM was kept because calibrated confidence made the reject option reliable.
- **Result:** 97.2% on real Hinglish tweets (400 test messages). Short messages (1-3 words) are the weak spot, around 96%.

### 3.2 Spam / scam detection
- **Features (what the model looks at):**
  1. Word 1-2-grams (TF-IDF)
  2. Character 2-4-grams
  3. Nine structural cues: link present, phone number, money amount, digits, capital letters, exclamation marks, etc.
  4. A small scam-vocabulary list (KYC, OTP, subsidy, lakh, crore, bank details...)
- **Model:** linear SVM, then **Platt calibration** (turns the SVM score into a probability), averaged over 5 cross-validation folds.
- **Two thresholds:** English 0.37, Hinglish 0.25. Hinglish needs a lower bar because real Hinglish scams look less like the English data it learned most from. Lowering the bar catches more scams but flags more genuine messages. That is the trade-off.
- **Training data:** UCI SMS spam, two public Indian SMS collections, three Hinglish/Indian spam sets from GitHub, plus generated messages. See section 5.
- **Generated "genuine twins":** many scams and real messages share words (a bank alert and a bank scam both say *account, bank, Rs*). For each scam type we generated a genuine message with the same words but without the demand for money/OTP, so the model learns **the ask**, not the vocabulary.
- **Anti-evasion:** the cleaner undoes spaced letters (*f r e e*) and look-alike characters (*fr33*). English recall under these attacks: 0.90 and 0.93 without it, 0.96 with it.
- **Result:** English SMS F1 **0.963**. Hinglish scam recall about **0.9** on author-written blind test sets, with some false alarms on genuine bank/OTP alerts (see limitations).

### 3.3 Sentiment
Two separate models, because English and Hinglish behave differently.

| | English | Hinglish |
|---|---|---|
| Classes | negative / neutral / positive | negative / neutral / positive |
| Features | word 1-2-grams + char 3-5-grams, with **negation handling** | word 1-2-grams + char 2-5-grams |
| Model | linear SVM (C = 1) | linear SVM (C = 0.3, balanced class weights) |
| Main data | tweets, short reviews, TweetEval, SST-2 | IIIT-H code-mixed corpus |
| Extra | customer-service and neutral templates | hand-written everyday seed rows + spelling-variant templates |

- **Negation handling:** "not tasty" is rewritten to `not_tasty` so the model learns it as a different word from "tasty". A plain bag-of-words model called "the food is not tasty" positive.
- **Neutral class:** the first version had only positive/negative, so "your watch is looking ok" came out positive. A neutral class was added.
- **Margin flag:** if the top two class scores are close, the message goes to review. (Sentiment margin cut-off: about 0.2.)
- **Transformers were tested too:** 5 pretrained models (BERT, XLM-R, MuRIL, HingBERT, HingRoBERTa) were fine-tuned on Colab. HingRoBERTa reached macro-F1 0.726 versus 0.694 for the classical SVM, only a small gain. The classical model was kept because it runs in a browser with no server, is small and fast, and the gain was within error.

### 3.4 Theme (clustering)
- **Idea:** unsupervised. The system groups messages by meaning, then we name the groups.
- **Method:** each message becomes one vector by averaging **word2vec** word vectors with **SIF weighting** (rare words count more; common words count less). **K-Means** with **20 clusters per language** groups them.
- **Why not TF-IDF + SVD:** measured. word2vec gave NMI about 0.45 versus about 0.15.
- **How it is scored:** clustering has no answer key, so we score on held-out messages that have a topic label the clustering never saw (NMI, ARI, purity).
- **No-clear-theme option:** if a message is far from every cluster, it is not forced into one.
- **Result:** NMI **0.48 (English)** and **0.61 (Hinglish)**, up from about 0.10 for the first 6-cluster version.

---

## 4. The "unclear text" gate
Added after a demo input like `I need t8 w2jde dejhe8 y go` was given a confident sentiment. For English/Hinglish, if at least 2 words and at least half (60% for Hinglish) of the words are unknown to the sentiment vocabulary, the message is labelled **Unclear text** and sent to review. In tests it flagged about 0.1% of English tweets and 0% of customer messages.

---

## 5. What we used

### Tools
| Part | Tool |
|---|---|
| Models | Python, scikit-learn (TfidfVectorizer, LinearSVC, KNN, Naive Bayes, KMeans, CalibratedClassifierCV) |
| Embeddings | gensim word2vec |
| Transformer experiments | Hugging Face on Colab (T4 GPU) |
| Browser engine | plain JavaScript (`site/engine.js`) |
| Hosting | Vercel (static), GitHub |
| Extra | Streamlit app, pytest tests (leakage checks, normaliser, pipeline behaviour) |

### Data (public datasets, about 15)
| Stage | Datasets |
|---|---|
| Language | UDHR (many languages), Hinglish language-ID corpus, UCI SMS (English) |
| Spam | UCI SMS, pjbrav Hinglish spam, Indian scam SMS (172), PRECOG 2011 Indian SMS, India SMS set |
| English sentiment | NLTK tweets, movie-review sentences, TweetEval, SST-2, VADER-word sentences |
| Hinglish sentiment | IIIT-H code-mixed corpus |
| Theme | Banking77, CLINC150, Hinglish TOP |

Also used: hand-written seed rows and **generated template messages** (written by us, training only; never counted as independent evidence).

### Link to the course brief (CodeChef ML project ideas combined)
Language Detection, Email/SMS Spam Classifier, Movie Review Sentiment, Clustering & Dimensionality Reduction, and the KNN / SVM / Naive Bayes comparison (used as the model-selection method in every classification stage).

---

## 6. Results (honest summary)

| Stage | Result | How it was tested |
|---|---|---|
| Language ID | 97.2% on real Hinglish tweets | 400 held-out messages |
| Spam, English | F1 0.963 | UCI SMS test split, near-duplicates removed |
| Spam, real Indian SMS | recall 97% | 25% held out from the PRECOG and India SMS sets |
| Spam, blind test | recall 97%, 1 of 30 genuine flagged | 60 lines we wrote before training (optimistic) |
| Spam, third-party probes | recall 89%, 0 false alarms | Hinglish + misspelled lines from another repo |
| Sentiment, English (3-class) | 87% on 70 unseen everyday messages | author-written (optimistic) |
| Sentiment, English, TweetEval | macro-F1 0.48 | public benchmark; neutral recall only about 20% because that dataset defines "neutral" differently |
| Sentiment, Hinglish | macro-F1 about 0.67 (IIIT-H test); 96% on 45 unseen everyday lines | test set touched once |
| Themes | NMI 0.48 English / 0.61 Hinglish | held-out topic labels |
| Whole pipeline | 63 messages/sec on one CPU core (3,921 real held-out messages) | Python version |

---

## 7. Key terms in one line each

| Term | Meaning |
|---|---|
| TF-IDF | Turns text into numbers; words that are frequent in a message but rare overall get more weight. |
| n-gram | A run of n words (or letters). Letter n-grams survive spelling variation. |
| Linear SVM | Finds the straight boundary that separates classes with the widest gap. |
| Calibration (Platt) | Rescales scores so "0.9" really means about 90% likely. Needed for thresholds. |
| Threshold | The score above which we say "spam". Moving it trades missed scams against false alarms. |
| Reject option | Letting the model say "not sure" instead of guessing. |
| Precision / recall | Precision: of what we flagged, how much was right. Recall: of all real spam, how much we caught. |
| F1 | Balance of precision and recall. |
| Macro-F1 | F1 averaged over classes equally, so a rare class counts as much as a common one. |
| AUC | How well scores rank spam above genuine, regardless of threshold. |
| word2vec / SIF | Word vectors from context; SIF weights rare words more when averaging into a message vector. |
| K-Means | Groups vectors into k clusters around centres. |
| NMI / ARI / purity | Scores of how well clusters match held-out topic labels. |
| Data leakage | Test data influencing training. It makes scores look better than they are. |
| McNemar test | Checks whether two models' accuracy difference is real or noise. |
| Quantisation | Storing weights in 8/16-bit numbers to shrink the file for the browser. |

---

## 8. Why we chose X (design decisions)

| Decision | Reason |
|---|---|
| Classical ML (SVM) over transformers | Runs in the browser with no server, small, fast, easy to explain; transformer gain was small (0.726 vs 0.694). |
| Character n-grams | Romanised Hindi has no fixed spelling ("achha / acha / accha"); letters are robust. |
| Separate English and Hinglish sentiment models | Different vocabulary and different labelled data. |
| Per-language spam thresholds | The same score means different things in each language. |
| Reject options everywhere | A wrong confident answer is worse than "please review". |
| Calibrated probabilities | Thresholds only make sense if scores behave like probabilities. |
| Test set touched once | Prevents choosing a model by peeking at the test data. |
| Generated "genuine twins" | Teaches the ask (send money / OTP), not just scary vocabulary. |
| Run in the browser | Privacy (messages never leave the device), no server cost, instant demo. |

---

## 9. Known limitations (say these before the teacher does)

1. **Hinglish scams are the weakest part.** No large public Hinglish scam dataset exists. About 1 in 10 Hinglish scams is still missed, and some genuine bank/OTP alerts are still flagged.
2. **Several test sets were written by us**, so those scores are optimistic. Independent numbers are the public benchmarks, the held-out slices of public data and the third-party probes.
3. **Hinglish sentiment is about 0.67 macro-F1** on a noisy real-world dataset (labels are subjective). Neutral is the hardest class.
4. **TweetEval neutral recall is low (about 20%)** because that dataset's "neutral" includes news-like text the model treats differently.
5. **Short or ambiguous messages:** a lone "ok" is called positive; "kal ghar se achha khana laya hu" is called neutral (arguable).
6. **Themes:** unsupervised clusters are named by us; a topic not in the training data can still land in a wrong cluster, and the "no clear theme" check is nearly useless in Hinglish.
7. **No sarcasm or irony handling.**
8. **Other Indian languages** (Tamil, Bengali, etc.) are only detected, not analysed.
9. The model sees **text only**: no sender, no link reputation lookup.

---

## 10. Questions a teacher may ask

### How teachers think (and how to answer)
A viva usually probes four things. Know which one the question is aiming at.

| They are checking | Typical trigger words | How to answer |
|---|---|---|
| **Did you build and understand this?** | "explain", "how does", "what happens when" | Walk through the flow in your own words; use the diagram. |
| **Can you justify choices?** | "why this, not that" | Name the alternative, say what you measured, say what you gave up. |
| **Is the result real?** | "accuracy", "overfitting", "test set" | Separate public/held-out evidence from author-written tests. |
| **Do you know the limits?** | "what if", "where does it fail" | State the weakness first and say what you would do next. |

Pattern that always works: **what, why, evidence, limitation**. Never say "100% accurate".

### A. Ownership and understanding

**Q0. What is the use of your project, and who would use it?**
Give the one-line problem from section 1b, then two users (support teams, banks) and one reason it suits them (it admits uncertainty and runs privately).

**Q1. Explain your project in one minute.**
Use section 1, then the pipeline diagram. End with: "and uncertain messages go to a human review queue."

**Q2. What happens when I type a message?**
It is cleaned, then language is detected. If unsure, it is flagged. Otherwise spam is checked with a per-language threshold. Then sentiment (English or Hinglish model) and theme. Any low-confidence step adds a review flag.

**Q3. What is Hinglish and why is it hard?**
Hindi written in English letters. It has no fixed spelling, mixes two languages in one sentence, and has little labelled data. A normal English model fails; a normal Hindi model can't read Roman script.

**Q4. What is TF-IDF?**
A way to turn text into numbers by weighting words that are frequent in a message but rare across all messages. Section 7.

**Q5. How does an SVM work?**
It finds the boundary line (hyperplane) between two classes with the largest margin. For more than two classes it does one-versus-rest.

**Q6. What does calibration do?**
A raw SVM score is not a probability. Platt calibration fits a small sigmoid so a score of 0.9 really means about 90%. We need it because the spam and reject thresholds depend on probabilities.

**Q7. How does word2vec + K-Means find themes?**
Each word has a vector from word2vec. Averaging them (with rare words weighted up) gives one vector per message. K-Means groups nearby vectors into 20 clusters, and we name each cluster from its top words and nearest messages.

**Q8. How did you decide on 20 clusters?**
We tried several values and scored each on held-out topic labels (NMI, ARI, purity). It was chosen by measurement, not guessed.

### B. Justification

**Q9. Why not BERT / a transformer?**
We tried five. The best (HingRoBERTa) got 0.726 against 0.694 for the SVM, a small gain inside the noise. The SVM runs in the browser with no server, is small and fast, and is easy to explain. A transformer would need a server, which breaks the privacy idea.

**Q10. Why separate English and Hinglish sentiment models?**
Different vocabulary, spelling and training data: English has large labelled sets, Hinglish has one noisy corpus plus our seed rows. Separate models let each be tuned (different features and settings) for its own data.

**Q11. Why character n-grams?**
Hinglish spelling varies ("achha / acha / accha"). Letter patterns survive this, whole words don't.

**Q12. Why different spam thresholds per language?**
The English model is trained on much more English data, so its scores are sharper. Hinglish scams score lower, so the same bar would miss most of them. We set each threshold from that language's genuine messages and our tolerance for false alarms.

**Q13. Why did you compare KNN, Naive Bayes and SVM?**
That is the course requirement and also our method of choosing. SVM and NB were close; KNN was worse and slower at prediction time. McNemar tests showed which differences were real.

**Q14. Why a reject option?**
In a company inbox a wrong confident label is worse than "please review". Rejecting trades a bit of coverage for much higher accuracy on what is answered.

### C. Is the result real?

**Q15. What accuracy did you get?**
Give the table from section 6, **by stage**. Never quote one number for everything.

**Q16. Isn't that overfitting?**
We chose models with cross-validation on training data only and touched the test set once. Test data are separate sources where possible, and `pytest` includes leakage checks.

**Q17. Did you test on data you wrote yourself?**
Yes, some small sets. We say so, and treat them as optimistic. Independent evidence is the public benchmarks, the held-out slices of public data, and probes from another repository.

**Q18. Why is Hinglish sentiment only about 0.67?**
Real social-media labels are noisy and subjective. The same corpus tops out around 0.73 even with transformers fine-tuned on a GPU. Neutral is hardest, because "neutral" is a judgement call.

**Q19. The majority-class baseline is 0.22. Why does that matter?**
Because accuracy alone can mislead on imbalanced data. We report macro-F1 and compare to a baseline, so we know the model learned something.

**Q20. Your English TweetEval score is only 0.48. Isn't that bad?**
It is a hard 3-class benchmark with a different definition of neutral. On everyday customer-style messages it scores about 87%. We report both, and we don't claim the high number for the hard set.

### D. Failure cases

**Q21. Where does it fail?**
Section 9. Lead with Hinglish scam recall, then short ambiguous messages, then sarcasm.

**Q22. How would a spammer beat your filter?**
Spaced letters and look-alike characters (handled by the cleaner), new scam wording (needs retraining), images or links without text (can't see them). Retraining on fresh examples is the fix.

**Q23. What if a message is in Tamil or Bengali?**
The language stage detects it but there is no sentiment/theme model, so it goes to review as "other language".

**Q24. What if someone types gibberish?**
The unclear-text gate sends it to review instead of giving a confident label.

**Q25. A genuine bank alert is flagged as spam. Why?**
Bank alerts contain the same words as bank scams (account, Rs, link). We reduced this with genuine-twin training data, but it is a trade-off with catching scams. The review queue is the safety net.

### E. Complexity and contribution

**Q26. What did you add beyond the basic idea?**
Chain of four stages; reject options at each; calibrated per-language thresholds; negation handling; neutral class; anti-evasion cleaning; unclear-text gate; Hinglish data work (templates, genuine twins); browser deployment with exact parity to Python; a review-flag explanation.

**Q27. How did you make a Python model run in a browser?**
Exported the weights, shrank them to 8/16-bit numbers, compressed to JSON, and re-implemented the same maths (vectorising, dot product, sigmoid) in JavaScript. We verified Python and JavaScript scores match exactly.

**Q28. Which part was hardest?**
Hinglish: little labelled data, no spelling standard, and almost no public scam data. We handled it with character n-grams, templates with genuine twins, and honest evaluation sets.

**Q29. What did you personally do vs use from libraries?**
Be ready to answer for your own part. The data cleaning, feature design, thresholds, templates, evaluation design and browser engine were ours; scikit-learn and gensim provide the algorithms.

### F. Curveballs

**Q30. Is it ethical to run this on people's messages?**
Messages never leave the device (browser-only). In a company setting you would need consent and a clear policy. Low-confidence cases go to a human instead of being auto-actioned.

**Q31. Can it scale?**
The Python pipeline did 63 messages/sec on one CPU core. The browser version is per user, so scaling is free. For millions of messages you would run the Python version in a batch job.

**Q32. How would you improve it with more time?**
Labelled real Hinglish scam data (in our experiment, 10 real Hinglish examples gave spam F1 about 0.77 and 50 gave about 0.97, on repetitive data so optimistic), a Hinglish-specific transformer on a server, per-class threshold tuning, and active learning from the review queue.

**Q33. What is the difference between precision and recall for spam?**
Precision: of messages we called spam, what fraction really were. Recall: of all real spam, what fraction we caught. A lower threshold raises recall and lowers precision.

**Q34. What is the difference between supervised and unsupervised here?**
Language, spam and sentiment are supervised (labelled data). Themes are unsupervised (clustering), scored afterwards with held-out topic labels.

**Q35. What is a confusion matrix?**
A table of predicted versus true classes; the diagonal is correct. We have one for language and one for Hinglish sentiment in `results/`.

---

## 11. Demo script (tested on the live engine)

Use these in the order below. Each gives the result shown.

| # | Type this | You should see | Say |
|---|---|---|---|
| 1 | `your phone is really good and smooth` | English, genuine, **positive** | Everyday praise. |
| 2 | `your watch is looking ok` | English, genuine, **neutral** | Neutral class works. |
| 3 | `khana bohut achha tha` | Hinglish, genuine, **positive** | Romanised Hindi handled. |
| 4 | `I got 2 lakh subsidy so I need to give bank details` | **Spam** | Classic subsidy scam. |
| 5 | `Mujhe 2 lakh ki subsidy mili hai, bank details dene hain` | Hinglish, **spam** | Same scam in Hinglish. |
| 6 | `URGENT: aapka account block ho jayega, abhi KYC update karein` | Hinglish, **spam** | KYC scam. |
| 7 | `Aaj raat dinner pe aa jao, mummy ne paneer banaya hai` | Hinglish, genuine, neutral | Normal chat. |
| 8 | `I need t8 w2jde dejhe8 y go` | **Unclear text**, goes to review | Refuses to guess. |

### Avoid these in the demo (known weak spots)
- A lone `ok` (called positive).
- A bank OTP message such as `Your OTP is 482913. Do not share it with anyone` (spam stage says genuine, but sentiment shows negative).
- `kal ghar se achha khana laya hu` (neutral; arguable).
- Anything with sarcasm.

If the teacher types something and it is wrong: **don't argue.** Say "that's the kind of case that goes to the review queue; here's why the model is unsure", and point to the flag.

### Before you go in
- Open the live link once so the models are cached (first load downloads about 18 MB).
- Keep a screenshot of the pipeline diagram ready in case the Wi-Fi fails.
- The repo has a Streamlit version too (`streamlit run app/streamlit_app.py`) as backup.

---

## 12. Things that are easy to forget (add to your answers)

- **Privacy:** everything runs on the user's device.
- **Reproducibility:** `python run_all.py` rebuilds data, models and results; `pytest -q` runs the leakage and behaviour tests.
- **Honest evaluation:** held-out sets, test set touched once, optimistic sets labelled as such.
- **Data licences:** the external GitHub datasets have no stated licence; used for coursework and cited in `data/external/README.md`.
- **Team split:** be ready to say who did which stage (add your own lines here).
- **Future work:** section Q32.
- **Version history:** the system started with two-class English sentiment and a Hinglish scam recall of about 53%. After adding real Indian SMS data and genuine-twin templates it moved to about 90% on blind tests. Being able to tell this story shows you iterated by measurement.
