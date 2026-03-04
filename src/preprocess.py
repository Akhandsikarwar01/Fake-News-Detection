"""Text preprocessing pipeline for Fake News Detection.

Usage:
    python src/preprocess.py --data data/train.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import clean_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the dataset CSV and return a cleaned DataFrame.

    Missing values in ``author``, ``title``, and ``text`` columns are filled
    with empty strings.  A combined ``content`` feature is created by
    concatenating ``author`` and ``title``.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Preprocessed DataFrame with a ``content`` and ``label`` column.
    """
    df = pd.read_csv(filepath)

    # Fill missing values
    for col in ("author", "title", "text"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Create combined content feature
    df["content"] = df["author"] + " " + df["title"]

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the text-cleaning pipeline to the ``content`` column.

    Args:
        df: DataFrame containing a ``content`` column.

    Returns:
        DataFrame with a cleaned ``content`` column.
    """
    df = df.copy()
    df["content"] = df["content"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# TF-IDF vectorisation
# ---------------------------------------------------------------------------

def build_tfidf_features(
    df: pd.DataFrame,
    max_features: int = TFIDF_MAX_FEATURES,
    save_vectorizer: bool = True,
):
    """Fit a TF-IDF vectoriser and return feature matrix and labels.

    Args:
        df:             Preprocessed DataFrame with ``content`` and ``label`` columns.
        max_features:   Maximum number of TF-IDF features.
        save_vectorizer: If ``True``, serialise the fitted vectoriser to disk.

    Returns:
        Tuple of (X, y, vectorizer) where *X* is a sparse feature matrix and
        *y* is a NumPy label array.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["content"])
    y = df["label"].values

    if save_vectorizer:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print(f"TF-IDF vectorizer saved to {VECTORIZER_PATH}")

    return X, y, vectorizer


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split features and labels into training and test sets.

    Args:
        X:            Feature matrix.
        y:            Label array.
        test_size:    Fraction of the dataset to reserve for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(filepath: str):
    """Execute the full preprocessing pipeline end-to-end.

    Args:
        filepath: Path to the raw CSV dataset.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test, vectorizer)``.
    """
    print(f"Loading dataset from: {filepath}")
    df = load_dataset(filepath)
    print(f"  Dataset shape: {df.shape}")

    print("Cleaning text ...")
    df = preprocess_dataframe(df)

    print("Building TF-IDF features ...")
    X, y, vectorizer = build_tfidf_features(df)
    print(f"  Feature matrix shape: {X.shape}")

    print("Splitting into train/test sets ...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, vectorizer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Fake News dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Path to the training CSV file (default: data/train.csv)",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, vectorizer = run_preprocessing_pipeline(args.data)
    print("Preprocessing complete.")
