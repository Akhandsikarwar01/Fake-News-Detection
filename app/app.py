"""Streamlit web application for Fake News Detection.

Run with:
    streamlit run app/app.py

Expects trained model and TF-IDF vectoriser to be present in the
``models/`` directory.  Train them first with:
    python src/train.py --data data/train.csv --model logistic_regression
"""

import os
import sys

import streamlit as st

# Allow imports from ``src/`` regardless of where the app is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import clean_text, load_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        """
        **Fake News Detector** uses Machine Learning and NLP to classify
        news articles as *fake* or *real*.

        **Tech Stack**
        - Python 3.8+
        - Scikit-learn
        - NLTK
        - Streamlit

        **Dataset**
        [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)

        ---
        *Author: Akhand Sikarwar*
        """
    )

    # Model selector
    available_models = []
    if os.path.isdir(MODELS_DIR):
        available_models = [
            f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib") and "vectorizer" not in f
        ]

    if available_models:
        selected_model_file = st.selectbox("Select model", available_models)
        model_path = os.path.join(MODELS_DIR, selected_model_file)
    else:
        model_path = DEFAULT_MODEL_PATH


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources(model_path: str):
    """Load and cache the model and vectoriser."""
    if not os.path.exists(model_path):
        return None, None
    if not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = load_model(model_path)
    vectorizer = load_model(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_resources(model_path)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.markdown("Paste a news article below and click **Check News** to find out if it's real or fake.")

news_input = st.text_area(
    label="News Article",
    placeholder="Paste the news article text or headline here ...",
    height=200,
)

if st.button("🔍 Check News", use_container_width=True):
    if not news_input.strip():
        st.warning("Please enter some text before clicking 'Check News'.")
    elif model is None or vectorizer is None:
        st.error(
            "Model or vectoriser not found.  "
            "Please train the model first:\n\n"
            "```bash\npython src/train.py --data data/train.csv --model logistic_regression\n```"
        )
    else:
        with st.spinner("Analysing ..."):
            cleaned = clean_text(news_input)
            features = vectorizer.transform([cleaned])
            label = int(model.predict(features)[0])

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence = float(max(proba))
            else:
                confidence = None

        st.markdown("---")

        if label == 1:
            st.error("## 🚨 FAKE NEWS")
            st.markdown("This article is likely **fake**.")
        else:
            st.success("## ✅ REAL NEWS")
            st.markdown("This article appears to be **real**.")

        if confidence is not None:
            st.metric(label="Confidence", value=f"{confidence:.1%}")

        with st.expander("🔎 Preprocessed text"):
            st.write(cleaned if cleaned else "_No tokens remained after preprocessing._")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Fake News Detector · Built with Streamlit and Scikit-learn")
