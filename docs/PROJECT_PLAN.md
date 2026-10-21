# Project plan: evidence, decisions and what is left

Written 2026-10-21 after a second research pass. Every number below is reproduced by `python run_all.py` and stored in `results/`.

## 1. What I checked before building anything

I tried to download every candidate dataset from the build environment and looked inside each one before deciding.

| Candidate | Finding | Decision |
|---|---|---|
| IIIT-H Hindi-English code-mixed sentiment (Prabhu et al. 2016), 3,879 comments | Real romanized Hinglish, 3 classes, no duplicates. Domain is mostly comments on videos and political posts. | **Used**: Hinglish sentiment |
| SemEval-2020 Task 9 (14k Hinglish tweets) | Hosted on Zenodo, which the build environment cannot reach. | Not used. Optional: download yourself and run `python src/semeval.py` |
| Hinglish / Hindi scam files in `gupta7050/multilingual-phishing-detection` | 5,000 rows per file but **only ~101 unique texts** (each repeated ~50 times). Training on them would leak into test and fake high scores. | **Rejected for training**; the 101 unique lines per language are used as a small independent *test* only |
| `SentMix-3L` | 1,007 rows, mixed scripts (Bangla/Hindi/English), AGPL, meant as a test set. | Not used |
| Other public Hinglish scam repos | Hand-written or LLM/template-generated messages; one lists real Indian scam SMS only as a future TODO. | **No real public Hinglish spam dataset exists.** Treated as a measured transfer problem |
| NLTK `twitter_samples`, `sentence_polarity`, `movie_reviews` | Three English domains. Tweet labels come from emoticons, so the emoticons are stripped from the text. | **Used**: English sentiment and cross-domain study |
| L3Cube-HingLID (real Hinglish tweets) | Already used; now 5,000 training samples, also supplies genuine Hinglish text for the spam threshold. | **Used** |
| Pretrained transformers (XLM-R, HingRoBERTa, MuRIL) | Hugging Face is not reachable from the build environment. | Colab notebook prepared, same data and split as the classical models |

## 2. What was weak in the prototype and what was done about it

| Weakness (prototype) | What I did | Outcome |
|---|---|---|
| Sentiment trained on movie reviews; unreliable on chat | Real Hinglish sentiment model + English model trained on tweets and short reviews; cross-domain study | Hinglish macro-F1 0.69 (baseline 0.22); English movie-only model scores 0.60 on tweets vs 0.76 when trained on tweets |
| Spam model English-only; Hinglish scams missed | Structural cues + scam vocabulary; per-language thresholds set from real genuine text; transfer and few-shot experiments | Hinglish recall 10% -> 35% at about 1% false alarms; English F1 0.956 on a stricter leak-free split (earlier 0.961 and 0.973 were measured on a split containing 7 near-duplicates) |
| Language model called French "Hinglish", "ok" Turkish | Calibrated model with a reject option; English tweets and more Hinglish added to training | Real Hinglish 96.5% -> 97.2%; 1-3 word messages 86.5% -> 95.7%; uncertain cases go to review |
| Weak, unstable clusters | Compared TF-IDF+SVD, NMF and word2vec with four metrics | word2vec stability 0.62-0.92 vs 0.39-0.66 for TF-IDF+SVD |
| End-to-end set was 47 AI-written messages | Mixed stream of ~3,900 real held-out messages from seven sources | Measured cost of language errors; throughput |
| Duplicate/near-duplicate leakage risk | Tests that fail if any train/test text overlaps; found and fixed a real near-duplicate leak (7 SMS; it had inflated spam F1 by about 0.01-0.02) | `pytest` guards it |

## 3. Findings worth stating in the report

1. **Character n-grams beat word features for romanized Hinglish** (about +0.05 macro-F1) because spelling is not standardised ("nahi / nahin / nhi").
2. **A spam filter can rank Hinglish scams well (AUC 0.96) yet catch almost none (recall 10%)**: the English-calibrated threshold is wrong for a new language. Setting the threshold per language from genuine text fixes the calibration problem without any Hinglish spam labels.
3. **Domain shift dominates sentiment**: movie reviews to tweets loses 0.27 macro-F1.
4. **In-domain word2vec gives much more stable themes than TF-IDF + SVD** (the PCA step of the original CodeChef idea).
5. **Language-ID errors cost coverage more than accuracy**: with predicted routing, 7% of English and 8% of Hinglish messages are held back for review (coverage 93% and 92%, against 99% with perfect routing) instead of being mis-routed.
6. **No public dataset has language, spam and sentiment labels on the same message**, so the end-to-end evaluation uses held-out test data from separate sources with labels known per task (assumptions listed in the README).

## 3b. Added after the first plan (2026-10-21, afternoon)
- **Synthetic Hinglish augmentation** (`scripts/make_synthetic_hinglish.py`, `src/exp_spam_synth.py`): Hinglish scam recall at about 1% false alarms rose from 35% to 53%; English F1 unchanged (0.96). Synthetic data is training-only. The challenge set is no longer fully independent because the templates and word list were written after reading it.
- **250-message labelled end-to-end set** (`data/eval/e2e_labelled_machine.csv`, `src/eval_labelled.py`): labels assigned by Claude, not by human annotators. Language 97.5% when answered (94.8% answered), spam precision 1.00 / recall 0.54 (0 of 193 genuine flagged), sentiment 71% English / 64% Hinglish accuracy on small subsets.
- **Colab transformer run**: blocked on a Google sign-in in the browser pane; still to be done.

## 4. Remaining work (ordered by value for the grade)

| # | Task | Why | Who / effort |
|---|---|---|---|
| 1 | Run `notebooks/transformer_finetune_colab.ipynb` on a free Colab GPU, save `transformer_results.csv` into `results/` | The only missing "complex model" result; compares transformers against the 0.694 classical baseline on identical data | You, about 30 min |
| 2 | Collect and label 50-100 **real** Hinglish scam and genuine messages (your own inbox, forwarded scams, friends' permission) | The main data gap; the transfer experiment predicts large gains | You and team, 1-2 hours |
| 3 | Have two classmates check the machine-assigned labels in `data/eval/e2e_labelled_machine.csv` (about 30 minutes), fix any mistakes, rerun `python src/eval_labelled.py` | Turns "machine-labelled" into "human-verified" | Team, 30 min |
| 4 | Optional: download SemEval-2020 Task 9 and run `python src/semeval.py` | A second, larger Hinglish sentiment benchmark | You, 15 min |
| 5 | Write the report (use the README sections as the skeleton) and rehearse the demo | Marks | Team |

## 5. Risks and how they are handled
- **Over-claiming**: every table is generated from result files; limits are listed in the README and app.
- **Test-set reuse**: model choice uses cross-validation on training data only; test sets are scored once per final model.
- **Small challenge sets** (101 messages): reported as indicative, with the repetition caveat.
- **Licence**: L3Cube-HingLID is CC BY-NC-SA (academic use, cite the paper); the IIIT-H and scam files carry no stated licence, so use them for coursework only and cite the sources.
