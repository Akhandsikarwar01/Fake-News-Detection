"""Model training script for Fake News Detection.

Trains one or more classifiers on the preprocessed dataset and saves each
trained model to the ``models/`` directory.

Usage:
    # Train all models
    python src/train.py --data data/train.csv --model all

    # Train a specific model
    python src/train.py --data data/train.csv --model logistic_regression
    python src/train.py --data data/train.csv --model random_forest
    python src/train.py --data data/train.csv --model decision_tree
    python src/train.py --data data/train.csv --model passive_aggressive
"""

import argparse
import os
import sys

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Allow running this script directly from the ``src/`` directory
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import run_preprocessing_pipeline
from utils import save_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "passive_aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(name: str, model, X_train, y_train, X_test, y_test) -> float:
    """Fit a single classifier and report training accuracy.

    Args:
        name:    Human-readable model name (used for saving).
        model:   Unfitted scikit-learn estimator.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test:  Test feature matrix.
        y_test:  Test labels.

    Returns:
        Training accuracy as a float.
    """
    print(f"\nTraining: {name} ...")
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"  Training Accuracy: {train_acc:.4f}")

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"  Test Accuracy    : {test_acc:.4f}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    save_model(model, model_path)

    return train_acc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection models.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Path to the training CSV file (default: data/train.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + list(MODEL_REGISTRY.keys()),
        help="Model to train (default: all)",
    )
    args = parser.parse_args()

    # Preprocess data
    X_train, X_test, y_train, y_test, _ = run_preprocessing_pipeline(args.data)

    # Select models to train
    models_to_train = (
        MODEL_REGISTRY if args.model == "all" else {args.model: MODEL_REGISTRY[args.model]}
    )

    print(f"\n{'='*50}")
    print(f"Training {len(models_to_train)} model(s) ...")
    print(f"{'='*50}")

    for name, model in models_to_train.items():
        train_model(name, model, X_train, y_train, X_test, y_test)

    print(f"\n{'='*50}")
    print("Training complete.  Models saved to the 'models/' directory.")
    print(f"{'='*50}")
