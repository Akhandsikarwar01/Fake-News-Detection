"""Model evaluation script for Fake News Detection.

Loads a trained model and test data, then computes:
- Accuracy score
- Precision, Recall, F1-score (classification report)
- Confusion matrix (visualised with seaborn and saved as an image)

Usage:
    python src/evaluate.py --data data/train.csv --model models/logistic_regression.joblib
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Allow running this script directly from the ``src/`` directory
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import run_preprocessing_pipeline
from utils import load_model

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    """Compute evaluation metrics and display a confusion matrix.

    Args:
        model:      Fitted scikit-learn estimator.
        X_test:     Test feature matrix.
        y_test:     True test labels.
        model_name: Name used for the saved confusion matrix image.

    Returns:
        Dictionary containing ``accuracy``, ``report``, and ``cm`` keys.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot and save confusion matrix
    _plot_confusion_matrix(cm, model_name)

    return {"accuracy": accuracy, "report": report, "cm": cm}


def _plot_confusion_matrix(cm, model_name: str) -> None:
    """Render the confusion matrix as a seaborn heatmap and save it.

    Args:
        cm:         Confusion matrix array from ``sklearn.metrics.confusion_matrix``.
        model_name: Used to build the output filename.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Fake News Detection model.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Path to the training CSV file (default: data/train.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model .joblib file (e.g. models/logistic_regression.joblib)",
    )
    args = parser.parse_args()

    # Preprocess data to get test split
    _, X_test, _, y_test, _ = run_preprocessing_pipeline(args.data)

    # Load model
    model = load_model(args.model)
    model_name = os.path.splitext(os.path.basename(args.model))[0]

    # Evaluate
    evaluate_model(model, X_test, y_test, model_name)
