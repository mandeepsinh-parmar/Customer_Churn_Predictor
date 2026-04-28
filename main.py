"""
main.py - CLI entry point for the Customer Churn Predictor.
Usage: python main.py
"""

import pandas as pd
import shap

from src.preprocess import load_data, preprocess_data, split_data, scale_data
from src.train import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, compare_models


def main():
    # 1. Load raw data
    print("Loading data...")
    df = load_data("Data/Telco.csv")
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # 2. Preprocess (clean + encode)
    print("Preprocessing...")
    df = preprocess_data(df)
    print(f"  After preprocessing: {df.shape[0]} rows, {df.shape[1]} features.\n")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

    # 4. Scale features (needed for Logistic Regression only)
    X_train_scaled, X_test_scaled, _ = scale_data(X_train, X_test)

    # 5. Train models
    print("Training models...")
    lr = train_logistic_regression(X_train_scaled, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)
    print("  Done.\n")

    # 6. Evaluate and compare
    results = {
        "Logistic Regression": evaluate_model(lr, X_test_scaled, y_test),
        "Random Forest": evaluate_model(rf, X_test, y_test),
        "XGBoost": evaluate_model(xgb, X_test, y_test),
    }
    print("Model Comparison:")
    print(compare_models(results).to_string())

    # 7. SHAP explanations on XGBoost
    print("\nGenerating SHAP explanations...")
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.plots.bar(shap_values)


if __name__ == "__main__":
    main()
