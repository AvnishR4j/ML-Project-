import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import analyze  # noqa: E402

R = ROOT / "results"
st.set_page_config(page_title="Multilingual Message Intelligence", layout="wide")
st.title("Multilingual Message Intelligence")
st.caption("Reads short English, Hindi and Hinglish messages and decides: language, spam, sentiment and theme.")

tabs = st.tabs(["Try it", "Batch dashboard", "Results", "End-to-end evaluation", "Themes", "Data and limits"])


@st.cache_data(show_spinner=False)
def csv(name):
    return pd.read_csv(R / name)


@st.cache_data(show_spinner=False)
def js(name):
    return json.loads((R / name).read_text())


EXAMPLES = {
    "English complaint": "My order has not arrived yet and nobody is replying to my emails",
    "English spam": "Congratulations! You have won a free prize, call 09061701461 now to claim",
    "Spam, letters spaced": "F R E E  p r i z e  waiting  c l a i m  n o w  call 09061701461",
    "Hinglish complaint": "mera order abhi tak nahi pahuncha, bahut gussa aa raha hai",
    "Hinglish praise": "yaar ye gana bahut accha hai, mujhe bahut pasand aaya",
    "Hinglish scam": "aapka KYC update nahi hua hai, account band ho jayega, abhi link par click karein",
    "German": "Der Film war wirklich gut, ich habe ihn sehr genossen",
    "Hindi (Devanagari)": "यह फ़ोन बहुत अच्छा है",
}

# ------------------------------------------------------------------ Try it
with tabs[0]:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES["Hinglish complaint"]
    cols = st.columns(4)
    for i, (name, txt) in enumerate(EXAMPLES.items()):
        if cols[i % 4].button(name, width="stretch"):
            st.session_state.text = txt
    text = st.text_area("Message", key="text", height=100)
    if text.strip():
        r = analyze(text)
        st.subheader("What the pipeline decided")
        for step, value, note in r["steps"]:
            a, b = st.columns([1, 3])
            a.markdown(f"**{step}**")
            b.markdown(f"`{value}`" + (f"  \n:grey[{note}]" if note else ""))
        if r.get("needs_review"):
            st.info("This message is routed to manual review because the language could not be identified with enough confidence.")


@st.cache_data(show_spinner=False)
def run_batch(texts: tuple):
    rows = []
    for t in texts:
        r = analyze(t)
        rows.append({"message": t, "language": r["language_name"], "spam": r.get("spam") or "not checked",
                     "sentiment": r.get("sentiment") or "n/a",
                     "theme": (r.get("cluster_label") or f"#{r['cluster']}") if "cluster" in r else "n/a",
                     "needs_review": bool(r.get("needs_review"))})
    return pd.DataFrame(rows)


def demo_sample():
    d = csv("e2e_detail.csv")
    return d.sample(500, random_state=3).text.tolist()


# ------------------------------------------------------------------ Batch dashboard
with tabs[1]:
    st.markdown("Upload a CSV with a **text** column, or use a sample of held-out real messages.")
    up = st.file_uploader("CSV with a 'text' column", type="csv")
    use_demo = st.button("Use sample")
    texts = None
    if up is not None:
        d0 = pd.read_csv(up)
        if "text" in d0.columns:
            texts = d0["text"].astype(str).tolist()[:5000]
        else:
            st.error("The CSV needs a column named 'text'.")
    elif use_demo or st.session_state.get("demo_loaded"):
        st.session_state["demo_loaded"] = True
        texts = demo_sample()
    if texts:
        with st.spinner(f"Analysing {len(texts)} messages ..."):
            B = run_batch(tuple(texts))
        checked = B[B.spam != "not checked"]
        k = st.columns(4)
        k[0].metric("Messages", len(B))
        k[1].metric("Flagged as spam", f"{(checked.spam == 'spam').mean():.0%}" if len(checked) else "n/a",
                    help="Of English and Hinglish messages (the languages the spam model covers)")
        k[2].metric("Sent to manual review", f"{B.needs_review.mean():.0%}", help="Language not identified with enough confidence")
        k[3].metric("Languages seen", B.language[~B.language.str.startswith("Uncertain")].nunique())
        c1, c2 = st.columns(2)
        c1.markdown("**Messages by language**"); c1.bar_chart(B[~B.language.str.startswith("Uncertain")].language.value_counts())
        s = B[B.sentiment != "n/a"]
        c2.markdown("**Sentiment of genuine English and Hinglish messages**"); c2.bar_chart(s.sentiment.value_counts())
        st.markdown("**Biggest themes among genuine messages**")
        st.bar_chart(B[B.theme != "n/a"].theme.value_counts().head(10))
        f1, f2 = st.columns(2)
        lang_sel = f1.multiselect("Language", sorted(B.language.unique()))
        spam_sel = f2.multiselect("Spam status", sorted(B.spam.unique()))
        mask = pd.Series(True, index=B.index)
        if lang_sel:
            mask &= B.language.isin(lang_sel)
        if spam_sel:
            mask &= B.spam.isin(spam_sel)
        st.dataframe(B[mask], hide_index=True, width="stretch")
        st.download_button("Download results (CSV)", B.to_csv(index=False).encode("utf-8"), "pipeline_results.csv", "text/csv")

# ------------------------------------------------------------------ Results
with tabs[2]:
    task = st.radio("Task", ["Language detection", "Spam detection", "Sentiment", "Themes", "Transformers"], horizontal=True)
    if task == "Language detection":
        st.markdown("Character n-gram TF-IDF with three classifiers, scored on separate test sets (one pooled number would hide the hard cases).")
        mc = csv("model_comparison.csv"); lang = mc[mc.task == "Language detection"]
        cols_ = ["model"] + [c for c in lang.columns if c.startswith("acc: ")]
        t = lang[cols_].copy(); t.columns = ["model"] + [c[5:] for c in cols_[1:]]
        st.dataframe(t.round(3), hide_index=True, width="stretch")
        st.markdown("**Deployed: calibrated linear SVM with a reject option.** Messages below the confidence threshold are sent to review instead of guessed.")
        st.dataframe(csv("language_selective.csv").round(3), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Accuracy by message length (words)**")
        c1.dataframe(csv("language_by_length_selective.csv").round(3), hide_index=True)
        c2.image(str(R / "language_confusion.png"), caption="Confusion matrix of the baseline model (pooled test sets)")
    elif task == "Transformers":
        st.markdown("Five pretrained encoders fine-tuned on the same Hinglish sentiment split (3 seeds each, Colab T4 GPU), scored on the same 776 test messages. "
                    "Classical reference: linear SVM macro-F1 0.694 (95% CI 0.658 to 0.729).")
        st.dataframe(csv("transformer_results.csv"), hide_index=True, width="stretch")
        st.markdown("The two Hinglish-specific models are slightly above the SVM; general multilingual models are below it; MuRIL did not train reliably at this learning rate. "
                    "The SVM trains in seconds on a CPU, so it stays in the pipeline.")
        st.markdown("**Zero-shot spam transfer** (XLM-R trained on English SMS only, threshold 0.5): good ranking of Hinglish scams, few caught at the default threshold.")
        st.dataframe(csv("transformer_spam_transfer.csv"), hide_index=True, width="stretch")
        st.caption("Scores were copied from the Colab output tables.")
    elif task == "Spam detection":
        j = js("spam_final_summary.json"); e = j["english_test"]; h = j["hinglish"]
        mc = csv("model_comparison.csv")
        st.markdown("**Three-model comparison, English SMS (UCI, duplicates removed)**")
        st.dataframe(mc[mc.task == "Spam detection"][["model", "accuracy", "spam_precision", "spam_recall", "spam_F1"]].round(3), hide_index=True, width="stretch")
        st.markdown("**Deployed model**: " + j["model"])
        m = st.columns(4)
        m[0].metric("English F1", f"{e['F1']:.3f}"); m[1].metric("English recall", f"{e['recall']:.1%}")
        m[2].metric("Genuine flagged", f"{e['genuine_flagged']} of {e['genuine_total']}"); m[3].metric("Hinglish scam recall", f"{h['recall']:.0%}",
                                                                                              help="At about 1% false alarms on real genuine Hinglish text")
        st.markdown("**Does an English spam filter work on Hinglish and Hindi?** (small, third-party hand-written challenge sets, 101 messages each)")
        st.dataframe(csv("spam_transfer.csv").round(3), hide_index=True, width="stretch")
        st.markdown("**What a few real Hinglish examples would add** (trained on English plus N Hinglish examples, tested on the rest, 20 random splits)")
        st.dataframe(csv("spam_fewshot.csv").round(3), hide_index=True)
        st.caption("These messages are repetitive and hand-written, so few-shot numbers are optimistic. Real labelled Hinglish spam is the main data gap.")
        st.markdown("**Evasion tricks (spam recall)**")
        st.dataframe(csv("spam_final_robustness.csv").round(3), hide_index=True)
    elif task == "Sentiment":
        sj = js("sentiment_summary.json")
        st.markdown(f"**Hinglish** (IIIT-H corpus, real comments, 3 classes). Chosen by cross-validation: {' + '.join(sj['hinglish']['chosen'])}. "
                    f"Test macro-F1 **{sj['hinglish']['test_macro_F1']:.3f}** (95% CI {sj['hinglish']['ci'][0]:.3f} to {sj['hinglish']['ci'][1]:.3f}); "
                    f"always predicting the majority class scores {sj['hinglish']['majority_baseline_macroF1']:.3f}. Published SemEval-2020 systems score 65 to 75 (different, noisier dataset).")
        st.dataframe(csv("sentiment_hinglish.csv").drop(columns=["best_params"]).round(3), hide_index=True, width="stretch")
        st.markdown("**English**: the same model trained on one domain breaks on another. Macro-F1, rows = training data, columns = test data.")
        X = csv("sentiment_english_crossdomain.csv").pivot(index="trained_on", columns="tested_on", values="macro_F1").round(3)
        st.dataframe(X, width="stretch")
        st.dataframe(csv("sentiment_english_models.csv").round(3), hide_index=True, width="stretch")
    else:
        st.markdown("Which message representation gives themes that are stable, coherent and meaningful? "
                    "Stability = agreement (ARI) between clusterings of overlapping random subsamples. Coherence = NPMI of each cluster's top terms.")
        ce = csv("cluster_experiment.csv")
        st.dataframe(ce.round(3), hide_index=True, width="stretch")
        st.markdown("word2vec trained on in-domain text is the most stable representation on both corpora, so it is the one deployed (k = 6).")

# ------------------------------------------------------------------ End-to-end
with tabs[3]:
    e = js("e2e_summary.json")
    st.markdown(f"A mixed stream of **{e['n']:,} real held-out messages** (no model saw any of them) from seven sources, run through the full pipeline. "
                f"Throughput: **{e['messages_per_second']:.0f} messages per second** on one CPU core.")
    L = e["language"]
    m = st.columns(3)
    m[0].metric("Language: answered", f"{L['coverage']:.1%}"); m[1].metric("Language: accuracy when answered", f"{L['accuracy_on_accepted']:.1%}")
    m[2].metric("Language: overall (uncertain = wrong)", f"{L['accuracy_overall_if_uncertain_counted_wrong']:.1%}")
    st.markdown("**Spam** (English and Hinglish messages)")
    rows = []
    for k_, v in e["spam"].items():
        rows.append({"setting": k_.replace("_", " "), "spam": v["n_spam"], "genuine": v["n_genuine"], "recall": v["recall"], "precision": v["precision"], "genuine flagged": v["genuine_flagged"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.markdown("**Sentiment: cost of language mistakes.** Predicted routing = the pipeline decides the language; oracle = the true language is supplied.")
    rows = []
    for lang_, v in e["sentiment"].items():
        for route, s in v.items():
            rows.append({"language": lang_, "routing": route.replace("_", " "), "answered": s["coverage"], "accuracy": s["accuracy"], "macro-F1": s["macro_F1"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("Genuine-vs-spam labels for tweets and comments are assumed (public social posts treated as genuine). "
               "The IIIT-H comments and the Hindi challenge set have no reliable language labels, so they are excluded from language accuracy.")
    st.dataframe(csv("e2e_detail.csv").sample(200, random_state=0), hide_index=True, width="stretch")
    st.markdown("**Labelled set: 250 real held-out messages** labelled for language, spam and sentiment. "
                "The labels were assigned by Claude, not by independent human annotators, so treat them as machine-labelled until two people verify them.")
    lab = js("e2e_labelled.json")
    k = st.columns(4)
    k[0].metric("Language: answered", f"{lab['language']['coverage']:.1%}"); k[1].metric("Language: correct when answered", f"{lab['language']['accuracy_when_answered']:.1%}")
    k[2].metric("Spam precision", f"{lab['spam']['precision']:.2f}"); k[3].metric("Spam recall", f"{lab['spam']['recall']:.2f}")
    st.caption(f"{lab['spam']['genuine_flagged']} of {lab['spam']['n_ham']} genuine messages flagged. Spam recall: English {lab['spam']['en']['recall']:.0%} "
               f"({lab['spam']['en']['n_spam']} scams), Hinglish {lab['spam']['hinglish']['recall']:.0%} ({lab['spam']['hinglish']['n_spam']} scams). "
               f"Sentiment accuracy: English {lab['sentiment']['en']['accuracy']:.0%} (n={lab['sentiment']['en']['n']}), Hinglish {lab['sentiment']['hinglish']['accuracy']:.0%} (n={lab['sentiment']['hinglish']['n']}).")

# ------------------------------------------------------------------ Themes
with tabs[4]:
    c1, c2 = st.columns(2)
    for col, key, title in ((c1, "en", "English (genuine SMS)"), (c2, "hinglish", "Hinglish (real comments)")):
        col.subheader(title)
        col.image(str(R / f"themes_{key}.png"))
        t = csv(f"themes_{key}.csv")
        names = js("theme_labels.json").get(key, {}); t.insert(1, "name", t.cluster.astype(str).map(names))
        col.dataframe(t[["cluster", "name", "size", "top_terms"]], hide_index=True, width="stretch")
        with col.expander("Example messages per theme"):
            for r in t.itertuples():
                st.markdown(f"**#{r.cluster}** " + " | ".join(r.examples.split(" || ")))

# ------------------------------------------------------------------ Data and limits
with tabs[5]:
    st.markdown("""
**Data**
| Used for | Dataset | Size | Note |
|---|---|---|---|
| Language | UN Declaration of Human Rights (NLTK), 15 languages | pieces of 25-120 characters | formal text |
| Language | L3Cube-HingLID (Nayak and Joshi, 2022), CC BY-NC-SA 4.0 | 5,000 train / 400 test samples used | real romanized Hinglish tweets |
| Language, spam | UCI SMS Spam Collection | 5,171 unique messages | English |
| Sentiment (Hinglish) | IIIT-H code-mixed corpus (Prabhu et al., 2016) | 3,876 comments | real Hinglish, 3 classes |
| Sentiment (English) | NLTK twitter_samples, sentence_polarity, movie_reviews | 10k / 10.6k / 2k | 3 domains |
| Spam transfer test | third-party hand-written Hinglish and Hindi scam lines | 101 unique each | the source files repeat ~100 lines 50 times each, de-duplicated here |

**Known limits**
- Spam in Hinglish is the weakest part: about half of Hinglish scams are caught at about 1% false alarms (the model also learns from synthetic Hinglish scams). The fix is real labelled Hinglish spam.
- Hindi in Devanagari is detected but not spam-checked, scored for sentiment or grouped into themes.
- Sentiment labels for English tweets come from emoticons (distant supervision); the Hinglish labels are noisy (typical agreement is low for this task).
- Themes are unsupervised: the cluster names were assigned by reading top terms and examples, not ground truth.
- The 250-message labelled set has machine-assigned labels; independent human verification is pending.
- Languages outside the 17 trained classes are not covered; very short text (1-3 words) is the hardest case.
- Transformers (Results tab) were run on Colab and are not part of the deployed pipeline.
""")
