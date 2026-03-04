"""Utility / helper functions shared across the Fake News Detection project."""

import joblib
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK data is available
for _resource in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{_resource}" if _resource == "stopwords" else f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean and normalise a raw text string.

    Steps applied:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation and special characters
    4. Tokenise by whitespace
    5. Remove stopwords
    6. Apply Porter stemming

    Args:
        text: Raw input string.

    Returns:
        Cleaned, stemmed string of tokens joined by spaces.
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove punctuation and non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenise
    tokens = text.split()

    # Remove stopwords and apply stemming
    tokens = [
        _stemmer.stem(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    ]

    return " ".join(tokens)


def load_model(model_path: str):
    """Load a joblib-serialised model from disk.

    Args:
        model_path: Path to the `.joblib` file.

    Returns:
        The deserialised model object.
    """
    return joblib.load(model_path)


def save_model(model, path: str) -> None:
    """Save a model to disk using joblib serialisation.

    Args:
        model: Trained model (or any object) to serialise.
        path:  Destination file path (e.g. ``models/logistic_regression.joblib``).
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")
