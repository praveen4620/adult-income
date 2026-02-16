import pandas as pd
import numpy as np
import joblib
import os

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
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("adult.csv")

# Clean data
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Encode target
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

X = df.drop("income", axis=1)
y = df["income"]

# ==============================
# Preprocessing
# ==============================

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols) # Added sparse_output=False
])

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Models (Required 6)
# ==============================

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(1),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(10),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

# Create folder
os.makedirs("models", exist_ok=True)

results = []

# ==============================
# Training Loop
# ==============================

for name, model in models.items():

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, acc, auc, prec, rec, f1, mcc])

    joblib.dump(pipe, f"models/{name}.pkl")

# Save results table
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "AUC", "Precision",
    "Recall", "F1", "MCC"
])

results_df.to_csv("model_results.csv", index=False)

print(results_df)