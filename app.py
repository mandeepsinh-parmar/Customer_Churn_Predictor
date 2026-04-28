"""
app.py - Streamlit web UI for the Customer Churn Predictor.

Provides an interactive dashboard with:
  - Dataset overview and statistics
  - Model training with progress feedback
  - Side-by-side model comparison (metrics table + bar chart)
  - Confusion matrices for each model
  - SHAP feature importance explanations
  - Single-customer churn prediction form

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import shap
import matplotlib.pyplot as plt

from src.preprocess import load_data, preprocess_data, split_data, scale_data
from src.train import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, get_confusion_matrix, compare_models

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS for a cleaner look
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Caching: load and process data once, train models once
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def get_raw_data():
    """Load the raw CSV (cached so it only runs once)."""
    return load_data("Data/Telco.csv")


@st.cache_data
def get_processed_data():
    """Preprocess + split + scale the data (cached)."""
    df = preprocess_data(get_raw_data().copy())
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sc, X_test_sc, scaler = scale_data(X_train, X_test)
    return df, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler


@st.cache_resource
def get_trained_models():
    """Train all three models (cached as a resource so models persist)."""
    _, X_train, _, y_train, _, X_train_sc, _, _ = get_processed_data()
    lr = train_logistic_regression(X_train_sc, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)
    return lr, rf, xgb


# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>📉 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Predict whether a telecom customer will churn using ML models"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("---")
    # Button to trigger training (first click loads everything)
    train_btn = st.button("🚀 Train Models", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "**Models used:**\n"
        "- Logistic Regression\n"
        "- Random Forest\n"
        "- XGBoost\n"
    )
    st.markdown("---")
    st.caption("Built with Streamlit • scikit-learn • XGBoost • SHAP")

# ─────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset", "🏋️ Model Results", "🔍 SHAP Analysis", "🎯 Predict"
])

# =====================================================================
# TAB 1: Dataset Overview
# =====================================================================
with tab1:
    raw_df = get_raw_data()

    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # Quick stats in metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(raw_df):,}")
    col2.metric("Features", raw_df.shape[1] - 1)  # minus target

    # Churn distribution
    churn_counts = raw_df["Churn"].value_counts()
    col3.metric("Churned", f"{churn_counts.get('Yes', 0):,}")
    col4.metric("Retained", f"{churn_counts.get('No', 0):,}")

    st.markdown("---")

    # Churn distribution chart
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Distribution")
        fig_churn = px.pie(
            names=["Not Churned", "Churned"],
            values=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
            color_discrete_sequence=["#36b37e", "#ff5630"],
            hole=0.45,
        )
        fig_churn.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_churn, use_container_width=True)

    with c2:
        st.subheader("Monthly Charges Distribution")
        fig_charges = px.histogram(
            raw_df, x="MonthlyCharges", color="Churn",
            barmode="overlay", nbins=40,
            color_discrete_map={"Yes": "#ff5630", "No": "#36b37e"},
        )
        fig_charges.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_charges, use_container_width=True)

# =====================================================================
# TAB 2: Model Results
# =====================================================================
with tab2:
    if train_btn or "models_trained" in st.session_state:
        # Train (or use cached) models
        with st.spinner("Training models... please wait"):
            lr, rf, xgb = get_trained_models()
            _, _, X_test, _, y_test, _, X_test_sc, _ = get_processed_data()
            st.session_state["models_trained"] = True

        st.success("✅ All models trained successfully!")
        st.markdown("---")

        # Compute metrics
        results = {
            "Logistic Regression": evaluate_model(lr, X_test_sc, y_test),
            "Random Forest": evaluate_model(rf, X_test, y_test),
            "XGBoost": evaluate_model(xgb, X_test, y_test),
        }
        results_df = compare_models(results)

        # ── Metrics table ──
        st.subheader("📋 Metrics Comparison")
        st.dataframe(
            results_df.style.highlight_max(axis=0, color="#c6efce"),
            use_container_width=True,
        )

        # ── Bar chart comparison ──
        st.subheader("📊 Visual Comparison")
        melted = results_df.reset_index().melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig_bar = px.bar(
            melted, x="Metric", y="Score", color="Model",
            barmode="group", text_auto=".3f",
            color_discrete_sequence=["#667eea", "#36b37e", "#ff5630"],
        )
        fig_bar.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Confusion Matrices ──
        st.subheader("🔢 Confusion Matrices")
        cm_cols = st.columns(3)
        model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
        models = [lr, rf, xgb]
        test_Xs = [X_test_sc, X_test, X_test]

        for i, (name, mdl, Xt) in enumerate(zip(model_names, models, test_Xs)):
            cm = get_confusion_matrix(mdl, Xt, y_test)
            with cm_cols[i]:
                st.markdown(f"**{name}**")
                # Build a heatmap with plotly
                fig_cm = ff.create_annotated_heatmap(
                    z=cm[::-1],
                    x=["Not Churned", "Churned"],
                    y=["Churned", "Not Churned"],
                    colorscale="Blues",
                    showscale=False,
                )
                fig_cm.update_layout(height=280, margin=dict(t=30, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("👈 Click **Train Models** in the sidebar to get started.")

# =====================================================================
# TAB 3: SHAP Analysis
# =====================================================================
with tab3:
    if "models_trained" in st.session_state:
        st.subheader("🔍 SHAP Feature Importance (XGBoost)")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) shows how each feature "
            "contributes to the model's predictions."
        )

        _, _, X_test, _, y_test, _, _, _ = get_processed_data()
        _, _, xgb = get_trained_models()

        # Compute SHAP values
        with st.spinner("Computing SHAP values..."):
            explainer = shap.Explainer(xgb)
            shap_values = explainer(X_test)

        # Bar plot — mean absolute SHAP value per feature
        st.markdown("#### Mean Feature Importance")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, ax=ax_bar, show=False)
        st.pyplot(fig_bar)

        # Beeswarm / summary plot
        st.markdown("#### Feature Impact (Beeswarm)")
        fig_sw, ax_sw = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig_sw)
    else:
        st.info("👈 Train the models first to see SHAP analysis.")

# =====================================================================
# TAB 4: Predict for a Single Customer
# =====================================================================
with tab4:
    if "models_trained" in st.session_state:
        st.subheader("🎯 Predict Churn for a Customer")
        st.markdown("Fill in the customer details below and hit **Predict**.")

        _, X_train, _, _, _, _, _, scaler = get_processed_data()
        lr, rf, xgb = get_trained_models()
        feature_names = X_train.columns.tolist()

        # Build a simple input form using the feature names
        with st.form("predict_form"):
            st.markdown("**Adjust feature values:**")
            input_cols = st.columns(3)
            user_input = {}
            for idx, feat in enumerate(feature_names):
                col = input_cols[idx % 3]
                # Use median as default value
                default = float(X_train[feat].median())
                user_input[feat] = col.number_input(
                    feat, value=default, format="%.2f", key=f"input_{feat}"
                )
            submitted = st.form_submit_button(
                "⚡ Predict", use_container_width=True
            )

        if submitted:
            # Build a single-row dataframe matching training features
            input_df = pd.DataFrame([user_input])
            input_scaled = pd.DataFrame(
                scaler.transform(input_df),
                columns=feature_names,
            )

            # Get predictions from all models
            pred_lr = lr.predict_proba(input_scaled)[0][1]
            pred_rf = rf.predict_proba(input_df)[0][1]
            pred_xgb = xgb.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.subheader("Prediction Results")
            p1, p2, p3 = st.columns(3)
            p1.metric("Logistic Regression", f"{pred_lr:.1%}",
                       "Churn" if pred_lr > 0.5 else "No Churn")
            p2.metric("Random Forest", f"{pred_rf:.1%}",
                       "Churn" if pred_rf > 0.5 else "No Churn")
            p3.metric("XGBoost", f"{pred_xgb:.1%}",
                       "Churn" if pred_xgb > 0.5 else "No Churn")
    else:
        st.info("👈 Train the models first to make predictions.")
