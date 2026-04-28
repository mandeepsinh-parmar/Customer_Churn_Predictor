"""
preprocess.py — Data loading, cleaning, encoding, and splitting utilities.

This module handles the full preprocessing pipeline for the Telco Customer
Churn dataset: loading the raw CSV, converting types, encoding categoricals,
and preparing train/test splits ready for model consumption.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Telco churn CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. "Data/Telco.csv").

    Returns
    -------
    pd.DataFrame
        Raw dataframe as-is from the CSV.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the Telco churn dataset.

    Steps performed:
      1. Drop 'customerID' (not a predictive feature).
      2. Convert 'TotalCharges' from string to numeric, fill blanks with median.
      3. Map binary Yes/No columns to 1/0.
      4. Collapse "No internet service" / "No phone service" into "No" then encode.
      5. One-hot encode any remaining categorical columns.
      6. Convert boolean dummy columns to int for sklearn compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data().

    Returns
    -------
    pd.DataFrame
        Fully numeric dataframe ready for modelling.
    """
    # --- Step 1: Drop customerID (unique identifier, not a feature) ---
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --- Step 2: Fix TotalCharges (stored as string with blanks) ---
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill missing values with the column median
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # --- Step 3: Encode the target variable ---
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- Step 4: Encode simple binary columns ---
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Gender: Male=1, Female=0
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # --- Step 5: Columns with "No internet/phone service" → treat as "No" ---
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # --- Step 6: One-hot encode remaining object columns (Contract, PaymentMethod, etc.) ---
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convert any boolean dummies to int (sklearn needs numeric types)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Safety net: drop rows that still have NaN
    df = df.dropna()

    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataframe into train and test sets.

    Uses stratified splitting to maintain the same churn ratio in both sets.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardize features using StandardScaler (zero mean, unit variance).

    This is important for Logistic Regression which is sensitive to feature
    scales. Tree-based models (RF, XGBoost) don't need this.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
