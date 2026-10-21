import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import analyze  # noqa: E402

R = ROOT / "results"
st.set_page_config(page_title="Multilingual Message Intelligence - Prototype", layout="wide")
st.title("Multilingual Message Intelligence System")
st.caption("PROTOTYPE v0.3 - classical models. Language detection -> spam filter -> sentiment -> theme clustering.")

tabs = st.tabs(["Try it", "Batch dashboard", "Model comparison", "Evaluation", "Clusters", "Status & limits"])

EXAMPLES = {
    "English complaint": "My order has not arrived yet and nobody is replying to my emails",
    "English spam": "Congratulations! You have won a free prize, call 09061701461 now to claim",
    "Spam, letters spaced": "F R E E  p r i z e  waiting  c l a i m  n o w  call 09061701461",
    "Hinglish complaint": "mera order abhi tak nahi pahuncha, bahut gussa aa raha hai",
    "Hinglish spam (known miss)": "aapka KYC update nahi hua hai, account band ho jayega, abhi link par click karein",
    "German": "Der Film war wirklich gut, ich habe ihn sehr genossen",
    "Hindi (Devanagari)": "यह फ़ोन बहुत अच्छा है",
}

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
        st.subheader("Pipeline result")
        for step, value, note in r["steps"]:
            a, b = st.columns([1, 3])
            a.markdown(f"**{step}**")
            b.markdown(f"`{value}`" + (f"  \n:grey[{note}]" if note else ""))


@st.cache_data(show_spinner=False)
def run_batch(texts: tuple):
    rows = []
    for t in texts:
        r = analyze(t)
        rows.append({"message": t, "language": r["language_name"], "spam": r.get("spam") or "not checked",
                     "sentiment": r.get("sentiment") or "n/a", "theme": (f"#{r['cluster']}: " + r["cluster_terms"]) if "cluster" in r else "n/a"})
    return pd.DataFrame(rows)


def demo_sample():
    import numpy as np
    frames = [pd.read_csv(ROOT / "data" / "pipeline_testset_DRAFT.csv").text]
    raw = ROOT / "data" / "raw"
    if (raw / "sms.tsv").exists():
        sms = pd.read_csv(raw / "sms.tsv", sep="\t", header=None, names=["l", "text"]).sample(250, random_state=1).text
        frames.append(sms)
    if (raw / "hinglid" / "test.txt").exists():
        from data import _hing_samples
        frames.append(pd.Series(_hing_samples(raw / "hinglid" / "test.txt", 80, np.random.default_rng(1))))
    return pd.concat(frames, ignore_index=True).tolist()


with tabs[1]:
    st.markdown("What a company would see: upload a CSV with a **text** column, or use the demo sample "
                "(draft test messages + English SMS + real Hinglish tweets). Everything runs through the same pipeline.")
    up = st.file_uploader("CSV with a 'text' column", type="csv")
    use_demo = st.button("Use demo sample")
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
        n = len(B); checked = B[B.spam != "not checked"]
        k = st.columns(4)
        k[0].metric("Messages", n)
        k[1].metric("Flagged as spam", f"{(checked.spam == 'spam').mean():.0%}" if len(checked) else "n/a", help="Of messages in English/Hinglish (the only languages the spam model covers)")
        eng = B[B.sentiment.isin(["positive", "negative"])]
        k[2].metric("Negative (English, non-spam) - UNRELIABLE", f"{(eng.sentiment == 'negative').mean():.0%}" if len(eng) else "n/a",
                 help="Sentiment model was trained on movie reviews; casual chat/SMS text is mostly labelled negative. Do not quote this number.")
        k[3].metric("Languages seen", B.language.nunique())
        st.warning("Sentiment numbers are unreliable for now: the model was trained on movie reviews and reads casual messages as negative. "
                   "They become meaningful after domain-specific / Hinglish sentiment training. Language, spam and theme counts are the usable parts.")
        c1, c2 = st.columns(2)
        c1.markdown("**Messages by language**"); c1.bar_chart(B.language.value_counts())
        c2.markdown("**Sentiment (English, non-spam only)**")
        c2.bar_chart(B.sentiment.value_counts().drop(["n/a"], errors="ignore"))
        st.markdown("**Biggest themes among genuine English messages**")
        th = B[B.theme != "n/a"].theme.value_counts().head(8)
        st.bar_chart(th)
        st.markdown("**Filter messages**")
        f1, f2 = st.columns(2)
        lang_sel = f1.multiselect("Language", sorted(B.language.unique()))
        spam_sel = f2.multiselect("Spam status", sorted(B.spam.unique()))
        mask = pd.Series(True, index=B.index)
        if lang_sel:
            mask &= B.language.isin(lang_sel)
        if spam_sel:
            mask &= B.spam.isin(spam_sel)
        V = B[mask]
        st.dataframe(V, hide_index=True, width="stretch")
        st.download_button("Download results (CSV)", B.to_csv(index=False).encode("utf-8"), "pipeline_results.csv", "text/csv")
        st.caption("Demo caveat: Hinglish messages are not checked for sentiment/themes yet, and Hinglish spam is mostly missed (see Status & limits).")

with tabs[2]:
    st.markdown("Same features for all three models on each task; hyper-parameters tuned with 3-fold CV on the training split only.")
    df = pd.read_csv(R / "model_comparison.csv").drop(columns=["best_params", "n_test"])
    st.dataframe(df.round(3), width="stretch", hide_index=True)
    st.markdown("**Language detection** is scored on four separate test sets because one pooled number hides the hard cases: "
                "real Hinglish tweets (L3Cube-HingLID), informal English incl. spam (must not be called Hinglish), "
                "the other 15 languages, and 78 hand-written Hinglish lines the model never trained on.")
    c1, c2 = st.columns(2)
    c1.image(str(R / "language_confusion.png"), caption="Language confusion matrix (pooled test sets)")
    c2.markdown("**Language accuracy by message length (words), hard sets**")
    c2.dataframe(pd.read_csv(R / "language_accuracy_by_length.csv").rename(columns={"mean": "accuracy", "size": "n"}).round(3), hide_index=True)
    c2.caption("Very short fragments are the weak spot; many misses are plainly English words inside tweets labelled Hinglish.")

with tabs[3]:
    ex = json.loads((R / "experiments_summary.json").read_text())
    st.subheader("1. Calibration and spam threshold")
    st.markdown(f"Deployed spam model: **{ex['calibration']['deployed']}**, threshold **{ex['calibration']['threshold']}** "
                "(highest spam recall on validation data with at most 0.5% of genuine messages flagged). "
                "ECE = expected calibration error (lower is better).")
    c1, c2 = st.columns([2, 1])
    c1.dataframe(pd.read_csv(R / "calibration_threshold.csv"), hide_index=True, width="stretch")
    c2.image(str(R / "spam_reliability.png"))
    st.subheader("2. Robustness to evasion tricks (spam recall)")
    st.dataframe(pd.read_csv(R / "robustness.csv"), hide_index=True, width="stretch")
    st.caption("The normalizer undoes letter-spacing and look-alike letters. It was written to invert these exact attacks, so full recovery "
               "is expected; real spammers will use other tricks.")
    st.subheader("3. Error propagation on a small DRAFT end-to-end test set")
    p = ex["pipeline_eval"]
    m = st.columns(4)
    m[0].metric("Messages", p["n"])
    m[1].metric("Language accuracy", f"{p['language_accuracy']:.0%}")
    m[2].metric("End-to-end correct (predicted language)", f"{p['end_to_end_exact_match_predicted']:.0%}")
    m[3].metric("End-to-end correct (true language)", f"{p['end_to_end_exact_match_oracle_language']:.0%}")
    st.markdown("The gap between the last two is the cost of language-detection mistakes flowing into later steps. "
                f"{p['messages_where_language_error_changed_outcome']} of {p['n']} messages were affected.")
    st.warning("This test set is a DRAFT of 47 messages written by the AI assistant, not real data. Replace/extend it with real, team-labelled messages "
               "before quoting any number. Findings so far: all 4 Hinglish spam messages were missed (spam model is English-only); "
               "a French message was absorbed by the Hinglish class; 2 of 6 clearly positive English messages were called negative.")
    st.dataframe(pd.read_csv(R / "pipeline_eval_detail.csv"), hide_index=True, width="stretch")

with tabs[4]:
    summ = json.loads((R / "summary.json").read_text())["clustering"]
    st.markdown(f"Genuine (non-spam) English SMS, {summ['n_messages']} messages -> TF-IDF -> SVD/PCA (50 dims) -> K-Means, k={summ['k_best']}.")
    c1, c2 = st.columns(2)
    c1.image(str(R / "clusters.png"))
    c2.dataframe(pd.read_csv(R / "cluster_summary.csv"), hide_index=True, width="stretch")
    st.warning("Clusters are weak so far (silhouette about 0.07-0.10, rising with k, no elbow). Next step: multilingual sentence embeddings.")

with tabs[5]:
    st.markdown("""
**Done so far**
- Language detection on 15 languages + English + **real romanized Hinglish** (L3Cube-HingLID), NB / SVM / KNN
- Spam detection (UCI SMS, duplicates removed before splitting), probability calibration and a low-false-positive threshold
- Evasion-robustness test (letter spacing, look-alike letters) and a normalizer defence
- English sentiment (movie reviews), PCA + K-Means theme clustering
- End-to-end pipeline, error-propagation experiment on a draft test set, this app

**Honest limits of this version**
- Spam, sentiment and clustering models are English only. Hinglish spam is currently missed (0 of 4 on the draft set).
- Sentiment is trained on movie reviews and transfers poorly to chat/customer text.
- The Hinglish class acts as a catch-all for unfamiliar Latin-script text (a French message was labelled Hinglish); more informal non-English data is needed.
- HingLID labels come from tweets, so very short English-looking fragments are labelled Hinglish by source, not by content.
- The end-to-end test set is a small AI-written draft; the 15 non-English languages come from formal UN text.
- No transformer results yet (notebook ready: `notebooks/transformer_finetune_colab.ipynb`); Hinglish sentiment waits for the SemEval data (`python src/semeval.py`).

**Next**
1. Download SemEval-2020 Task 9 -> Hinglish sentiment (classical now, transformers on Colab)
2. Hinglish spam data (e.g. the Hinglish scam text dataset) and informal French/German/Spanish text
3. Multilingual embeddings for clustering
4. Replace the draft end-to-end set with a real team-labelled set (200-300 messages)
5. Fine-tune XLM-R / HingRoBERTa and compare with the classical baselines
""")
