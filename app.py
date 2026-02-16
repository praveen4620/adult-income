import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ==============================
# Page Setup
# ==============================

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification - ML Assignment 2")

st.write("Upload the Adult Income CSV dataset (Test Data Only)")

# ==============================
# Download Sample Test Data
# ==============================

st.subheader("Download Sample Test Data")

try:
    sample_df = pd.read_csv("adult.csv")
    sample_df = sample_df.sample(200, random_state=42)

    st.download_button(
        label="Download Sample Test CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )
except:
    st.info("adult.csv not found in repository.")

# ==============================
# Upload CSV
# ==============================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ==============================
# Model Selection
# ==============================

model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# ==============================
# Train Model Function
# ==============================

def train_model(model_name, X, y):

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", models[model_name])
    ])

    pipe.fit(X, y)
    return pipe

# ==============================
# Main Logic
# ==============================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    y = df["income"]
    X = df.drop("income", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(model_name, X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    # ================= Metrics =================

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ================= Confusion Matrix =================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
