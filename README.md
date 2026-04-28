# 📉 Customer Churn Predictor

A machine learning project that predicts customer churn for a telecom company using classification models and provides an interactive Streamlit dashboard for visualization and real-time predictions.

---

## 👤 Author

**Mandeepsinh Parmar**

> This is a mini project developed as part of my internship in the domain of **Artificial Intelligence & Machine Learning**. The project demonstrates a complete end-to-end ML pipeline — from raw data preprocessing to model training, evaluation, explainability (SHAP), and deployment through an interactive web-based UI.

---

## 📌 Problem Statement

Customer churn is a critical problem for telecom companies. Acquiring new customers costs significantly more than retaining existing ones. The goal of this project is to build a predictive model that identifies customers likely to churn, enabling the business to take proactive retention measures.

---

## 🗂️ Project Structure

```
Customer_Churn_Predictor/
│
├── Data/
│   └── Telco.csv                # Telco Customer Churn dataset
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── preprocess.py            # Data loading, cleaning, encoding, splitting
│   ├── train.py                 # Model training (LR, RF, XGBoost)
│   └── evaluate.py              # Metrics computation and model comparison
│
├── app.py                       # Streamlit web dashboard
├── main.py                      # CLI entry point (console-based pipeline)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation (this file)
```

---

## 🧠 Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear baseline classifier; works well with scaled features |
| **Random Forest** | Ensemble of decision trees using bagging; robust to overfitting |
| **XGBoost** | Gradient boosting algorithm; often the top performer on tabular data |

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset (originally from IBM), which contains information about 7,043 telecom customers and whether they churned.

**Key features include:**
- Demographics — gender, senior citizen, partner, dependents
- Account info — tenure, contract type, payment method, monthly/total charges
- Services — phone, internet, online security, streaming, tech support, etc.
- Target — `Churn` (Yes / No)

---

## 🚀 How to Run

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI Pipeline

```bash
python main.py
```

This will train all three models, print a comparison table, and generate SHAP plots.

### 3. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

This opens an interactive web app in your browser with four tabs:

| Tab | What it shows |
|-----|---------------|
| 📊 **Dataset** | Raw data preview, churn distribution pie chart, monthly charges histogram |
| 🏋️ **Model Results** | Metrics comparison table, bar chart, confusion matrices |
| 🔍 **SHAP Analysis** | Feature importance bar plot and beeswarm plot (XGBoost) |
| 🎯 **Predict** | Input form to predict churn probability for a single customer |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8070 | 0.6594 | 0.5642 | 0.6081 | 0.8417 |
| Random Forest | 0.8027 | 0.6644 | 0.5187 | 0.5826 | 0.8432 |
| XGBoost | 0.7921 | 0.6270 | 0.5348 | 0.5772 | 0.8327 |

> Logistic Regression achieves the best F1 Score, while Random Forest leads in ROC AUC — indicating strong overall discriminative ability.

---

## 🔍 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to interpret the XGBoost model's predictions. It reveals which features have the most impact on whether a customer churns:

- **Top factors:** Contract type, tenure, monthly charges, and internet service type
- The SHAP beeswarm plot shows both the direction and magnitude of each feature's effect

---

## 🛠️ Tech Stack

- **Python** — Core programming language
- **pandas / NumPy** — Data manipulation and numerical computing
- **scikit-learn** — Preprocessing, Logistic Regression, Random Forest, metrics
- **XGBoost** — Gradient boosting classifier
- **SHAP** — Model explainability
- **Streamlit** — Interactive web dashboard
- **Plotly / Matplotlib** — Data visualization

---

## 📝 License

This project is developed for academic and internship purposes.

---
