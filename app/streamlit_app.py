import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import analyze  # noqa: E402

st.set_page_config(page_title="Multilingual Message Intelligence - Prototype", layout="wide")
st.title("Multilingual Message Intelligence System")
st.caption("PROTOTYPE v0.1 - classical models only. Language detection -> spam filter -> sentiment -> theme clustering.")

tab_try, tab_cmp, tab_cl, tab_about = st.tabs(["Try it", "Model comparison", "Clusters", "Status & limits"])

EXAMPLES = {
    "English complaint": "My order has not arrived yet and nobody is replying to my emails",
    "English spam": "Congratulations! You have won a free prize, call 09061701461 now to claim",
    "Hinglish complaint": "mera order abhi tak nahi pahuncha, bahut gussa aa raha hai",
    "Hinglish casual": "bhai kal ka plan kya hai, main late aaunga",
    "German": "Der Film war wirklich gut, ich habe ihn sehr genossen",
    "Hindi (Devanagari)": "यह फ़ोन बहुत अच्छा है",
}

with tab_try:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES["Hinglish complaint"]
    cols = st.columns(len(EXAMPLES))
    for c, (name, txt) in zip(cols, EXAMPLES.items()):
        if c.button(name, width="stretch"):
            st.session_state.text = txt
    text = st.text_area("Message", key="text", height=100)
    if text.strip():
        r = analyze(text)
        st.subheader("Pipeline result")
        for step, value, note in r["steps"]:
            a, b = st.columns([1, 3])
            a.markdown(f"**{step}**")
            b.markdown(f"`{value}`" + (f"  \n:grey[{note}]" if note else ""))

with tab_cmp:
    st.markdown("Same features for all three models on each task; hyper-parameters tuned with 3-fold CV on the training split only.")
    df = pd.read_csv(ROOT / "results" / "model_comparison.csv").drop(columns=["best_params", "n_test"])
    st.dataframe(df.round(3), width="stretch", hide_index=True)
    st.markdown("**Language-detection confusion matrix (test set)**")
    st.image(str(ROOT / "results" / "language_confusion.png"), width=520)

with tab_cl:
    summ = json.loads((ROOT / "results" / "summary.json").read_text())["clustering"]
    st.markdown(f"Genuine (non-spam) English SMS, {summ['n_messages']} messages -> TF-IDF -> SVD/PCA (50 dims) -> K-Means, k={summ['k_best']}.")
    c1, c2 = st.columns([1, 1])
    c1.image(str(ROOT / "results" / "clusters.png"))
    c2.dataframe(pd.read_csv(ROOT / "results" / "cluster_summary.csv"), hide_index=True, width="stretch")
    st.warning("Clusters are weak so far (silhouette about 0.07-0.10 and no clear elbow). Next step: multilingual sentence embeddings.")

with tab_about:
    st.markdown("""
**Done in this prototype**
- Language detection (char n-gram TF-IDF; Naive Bayes / SVM / KNN) on 16 languages + a Hinglish class
- Spam detection on the UCI SMS Spam Collection (duplicates removed before splitting)
- English sentiment on movie reviews
- PCA + K-Means theme clustering of genuine SMS
- End-to-end pipeline + this demo app

**Honest limits of this version**
- Language-detection data (UN Declaration of Human Rights) is easy, so scores near 100% mean little.
- The Hinglish class uses 78 hand-written seed sentences (single author, not real-world data) - a placeholder, not a result.
- Spam and sentiment models are English only; sentiment is trained on movie reviews and does not transfer well to chat messages.
- No transformer models and no Hinglish sentiment yet.

**Next**
1. Replace the Hinglish seed with real data (L3Cube HingCorpus / SemEval-2020 Task 9)
2. Larger language-ID set (20 languages), test on short and mixed messages
3. XLM-R / HingRoBERTa for sentiment; Hinglish sentiment (SemEval)
4. Multilingual embeddings for clustering
5. Hand-labelled end-to-end test set and error-propagation experiment
""")
