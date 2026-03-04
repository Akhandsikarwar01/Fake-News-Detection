"""Prediction script for Fake News Detection.

Loads a trained model and TF-IDF vectoriser, applies the same preprocessing
pipeline to the user-supplied text, and outputs a prediction.

Usage:
    python src/predict.py \\
        --text "Scientists discover a new planet in the solar system." \\
        --model models/logistic_regression.joblib \\
        --vectorizer models/tfidf_vectorizer.joblib
"""

import argparse
import os
import sys

# Allow running this script directly from the ``src/`` directory
sys.path.insert(0, os.path.dirname(__file__))

from utils import clean_text, load_model

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict(text: str, model, vectorizer) -> dict:
    """Predict whether the supplied text is fake or real news.

    Args:
        text:       Raw news article text (title, content, or both).
        model:      Fitted scikit-learn estimator.
        vectorizer: Fitted TF-IDF vectoriser.

    Returns:
        Dictionary with keys ``label`` (0 or 1), ``prediction`` (str), and
        ``confidence`` (float, only available for models that support
        ``predict_proba``).
    """
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    label = int(model.predict(features)[0])

    result = {
        "label": label,
        "prediction": "FAKE NEWS 🚨" if label == 1 else "REAL NEWS ✅",
    }

    # Confidence score (if the model supports probability estimates)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        result["confidence"] = float(max(proba))
    else:
        result["confidence"] = None

    return result


MAX_DISPLAY_LENGTH = 120

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fake/real news from text.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="News article text to classify.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/logistic_regression.joblib",
        help="Path to trained model .joblib file (default: models/logistic_regression.joblib)",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="models/tfidf_vectorizer.joblib",
        help="Path to TF-IDF vectoriser .joblib file (default: models/tfidf_vectorizer.joblib)",
    )
    args = parser.parse_args()

    # Validate file paths
    for path, label in [(args.model, "model"), (args.vectorizer, "vectorizer")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found at '{path}'.")
            print("  Train the model first:  python src/train.py --data data/train.csv")
            sys.exit(1)

    model = load_model(args.model)
    vectorizer = load_model(args.vectorizer)

    result = predict(args.text, model, vectorizer)

    print(f"\nInput text  : {args.text[:MAX_DISPLAY_LENGTH]}{'...' if len(args.text) > MAX_DISPLAY_LENGTH else ''}")
    print(f"Prediction  : {result['prediction']}")
    if result["confidence"] is not None:
        print(f"Confidence  : {result['confidence']:.2%}")
