# Quick Revision: Multilingual Message Intelligence

Read this the night before and again just before the viva. Full detail is in `VIVA_GUIDE.md`.

---

## 1. The whole project in 4 lines

- **Input:** a short message in English, Hindi, or **Hinglish** (Hindi in English letters).
- **Output:** language, spam or genuine, sentiment, theme.
- **Idea:** when a step is not confident, **don't guess; send to human review.**
- **Runs fully in the browser.** No message leaves the device.

```
message -> clean -> LANGUAGE -> (gibberish gate) -> SPAM -> SENTIMENT -> THEME -> result
                      |              |               |          |          |
                   unsure         unclear        near the     small      no clear
                      \_____________\______________ line _____ margin ____/
                                         -> REVIEW QUEUE
```

Memory hook: **L-S-S-T** = Language, Spam, Sentiment, Theme.

---

## 2. Each stage on one card

| Stage | Features | Model | Special trick | Key number |
|---|---|---|---|---|
| **Language** | char 1-4-grams, TF-IDF | calibrated linear SVM, 17 classes | **reject option** (threshold 0.73) | 97.2% real Hinglish tweets |
| **Spam** | word 1-2 + char 2-4 + 9 structural cues + scam word list | linear SVM + Platt calibration, 5 folds | **per-language threshold**: English 0.37, Hinglish 0.25 | English F1 **0.963** |
| **Sentiment EN** | word 1-2 + char 3-5, negation-aware | linear SVM, 3 classes | `not tasty` -> `not_tasty`; neutral class; margin flag | 87% on 70 unseen everyday messages |
| **Sentiment HI** | word 1-2 + char 2-5 | linear SVM (C 0.3, balanced) | seed rows + spelling-variant templates | macro-F1 about **0.67** |
| **Theme** | word2vec averaged with SIF weights | K-Means, 20 clusters per language | **no-clear-theme** option | NMI **0.48** EN / **0.61** HI |

---

## 3. Concepts to know cold (one line each)

- **TF-IDF:** word weight = how often in this message x how rare overall.
- **n-gram:** run of n words or letters. Letter n-grams survive spelling changes (*achha / acha*).
- **Linear SVM:** straight boundary with the widest margin between classes.
- **Calibration (Platt):** turns an SVM score into a real probability, so thresholds make sense.
- **Threshold:** the bar for "spam". Lower = catch more scams, flag more genuine.
- **Reject option:** model may say "not sure".
- **Precision:** of what I flagged, how much was right.  **Recall:** of all real spam, how much I caught.
- **F1:** balance of precision and recall.  **Macro-F1:** F1 averaged equally over classes.
- **AUC:** how well scores rank spam above genuine, any threshold.
- **word2vec + SIF:** word vectors from context; SIF gives rare words more weight when averaging.
- **K-Means:** group vectors around k centres.
- **NMI / ARI / purity:** how well clusters match held-out topic labels (clustering has no answer key).
- **Supervised:** language, spam, sentiment.  **Unsupervised:** theme.
- **Data leakage:** test data leaking into training, which inflates scores.
- **McNemar test:** is the gap between two models real or noise?
- **Quantisation:** 8/16-bit weights to make the browser files small.

---

## 4. Why we chose it (the 8 justifications)

1. **SVM, not transformer:** best transformer 0.726 vs SVM 0.694, a small gain; SVM runs in the browser, is small and explainable.
2. **Character n-grams:** Hinglish has no fixed spelling.
3. **Two sentiment models:** different vocabulary and different data.
4. **Different spam thresholds:** same score means different things per language.
5. **Reject options:** a wrong confident answer is worse than "please review".
6. **Calibration:** thresholds need real probabilities.
7. **Test set touched once:** stops us choosing models by peeking.
8. **Browser deployment:** privacy, no server, instant demo.

---

## 5. Problems we hit, and the fix (great for "what challenges did you face?")

| Problem | Fix |
|---|---|
| "not tasty" read as positive | negation handling (`not_tasty`) |
| "looking ok" read as positive | added a **neutral** class |
| Gibberish got a confident label | **unclear-text gate** |
| Hinglish scams missed (recall about 53% at the start) | real Indian SMS data + templates + lower Hinglish threshold (about 90% on blind tests) |
| Genuine bank alerts flagged as spam | **genuine twins** (same words, no money/OTP demand) |
| Letters spaced out or swapped (`f r e e`, `fr33`) | cleaner undoes it (recall 0.90 -> 0.96) |
| Website used old models | models now load from the same site |
| Everything first looked too good | separated test sets; marked author-written ones as optimistic |

---

## 6. Numbers to remember

| Item | Number |
|---|---|
| Language ID, real Hinglish tweets | **97.2%** |
| Spam English F1 | **0.963** |
| Spam real Indian SMS held out | recall **97%** |
| Spam blind test (written by us) | recall **97%**, 1 of 30 genuine flagged |
| Spam, third-party probes | recall **89%**, 0 false alarms |
| Sentiment EN, 70 unseen everyday | **87%** |
| Sentiment EN, TweetEval (hard) | macro-F1 **0.48** |
| Sentiment Hinglish, IIIT-H | macro-F1 about **0.67** (majority baseline 0.22) |
| Best transformer (HingRoBERTa) | **0.726** |
| Themes NMI | **0.48** EN, **0.61** HI (first version about 0.10) |
| Speed (Python, one CPU core) | **63** messages/sec |
| Browser download | about **18 MB**, once |
| Datasets | about **15** public + our templates |

---

## 7. Weak spots (say them first)

1. Hinglish scams: about 1 in 10 still missed; some genuine bank/OTP alerts flagged.
2. Several test sets written by us, so optimistic.
3. Hinglish sentiment is about 0.67 (noisy labels, neutral hardest).
4. Short or ambiguous text: a lone "ok" is called positive.
5. No sarcasm handling.
6. Other Indian languages only detected, not analysed.
7. Themes: unsupervised, named by us; weak "no clear theme" check in Hinglish.

---

## 8. Rapid-fire answers

- **Why Hinglish is hard?** No fixed spelling, mixes two languages, little labelled data.
- **Overfitting?** Cross-validation on training only; test set touched once; leakage tests.
- **Accuracy?** Never one number. Give it per stage.
- **Why not BERT?** Tried five; small gain; needs a server.
- **How in a browser?** Export weights, shrink, re-implement the maths in JavaScript, scores matched exactly.
- **What if wrong?** Goes to review; the flag says why.
- **Spam trade-off?** Lower threshold catches more scams but flags more genuine.
- **Improve next?** Real labelled Hinglish scam data, Hinglish transformer on a server, learn from the review queue.

---

## 9. Answer formula

**What -> Why -> Evidence -> Limitation.**
Example: "Spam uses an SVM with TF-IDF and structural cues (what), because it is fast and explainable in the browser (why). English F1 is 0.963 on held-out data (evidence). Hinglish scams are still the weakest part (limitation)."

---

## 10. Demo cheat line-up

| Type | Expect |
|---|---|
| `your phone is really good and smooth` | genuine, positive |
| `your watch is looking ok` | genuine, neutral |
| `khana bohut achha tha` | Hinglish, positive |
| `I got 2 lakh subsidy so I need to give bank details` | spam |
| `Mujhe 2 lakh ki subsidy mili hai, bank details dene hain` | Hinglish, spam |
| `I need t8 w2jde dejhe8 y go` | unclear text, review |

**Avoid:** a lone `ok`, an OTP message, sarcasm. Open the site once beforehand so the models are cached.

---

## 11. Last-minute checklist

- [ ] Site loaded once, models cached
- [ ] Repo link: github.com/AvnishR4j/ML-Project-
- [ ] Know your own part of the work
- [ ] Say the limitations before they ask
- [ ] Stay calm: if a demo answer is wrong, say "this goes to review" and show the flag
