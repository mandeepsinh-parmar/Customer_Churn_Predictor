"""
evaluate.py — Model evaluation and comparison utilities.

Provides functions to compute classification metrics (accuracy, precision,
recall, F1, ROC AUC) for a trained model, and to build a side-by-side
comparison DataFrame across multiple models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained classifier on the test set.

    Returns a dictionary of standard binary classification metrics,
    all rounded to 4 decimal places for clean display.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (Churn)

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC AUC": round(roc_auc_score(y_test, y_proba), 4),
    }


def get_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Return the confusion matrix as a 2x2 numpy array."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> str:
    """Return a formatted classification report string."""
    y_pred = model.predict(X_test)
    return classification_report(
        y_test, y_pred, target_names=["Not Churned", "Churned"]
    )


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {model_name: metrics_dict}.

    Sorts by F1 Score (descending) so the best model appears first.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.sort_values("F1 Score", ascending=False)
