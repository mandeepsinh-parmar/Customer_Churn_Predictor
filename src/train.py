"""
train.py — Model training functions.

Provides factory functions for three classifiers:
  - Logistic Regression (linear baseline)
  - Random Forest (bagging ensemble)
  - XGBoost (boosting ensemble)

Each function accepts training data, fits the model, and returns it.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Uses max_iter=1000 to ensure convergence on larger feature sets.
    Best used with scaled features (see preprocess.scale_data).
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Hyperparameters:
      - 200 trees for a good bias-variance trade-off
      - max_depth=10 to prevent overfitting
      - n_jobs=-1 to use all CPU cores
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost (Extreme Gradient Boosting) classifier.

    Hyperparameters:
      - 200 boosting rounds
      - max_depth=6 (default, controls tree complexity)
      - learning_rate=0.1 (shrinkage to prevent overfitting)
      - eval_metric="logloss" for binary classification
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
