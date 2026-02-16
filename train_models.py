import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Import models from model folder
from model.logistic_regression import get_model as lr_model
from model.decision_tree import get_model as dt_model
from model.knn import get_model as knn_model
from model.naive_bayes import get_model as nb_model
from model.random_forest import get_model as rf_model
from model.xgboost_model import get_model as xgb_model

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("adult.csv")

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

X = df.drop("income", axis=1)
y = df["income"]

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic_Regression": lr_model(),
    "Decision_Tree": dt_model(),
    "KNN": knn_model(),
    "Naive_Bayes": nb_model(),
    "Random_Forest": rf_model(),
    "XGBoost": xgb_model()
}

results = []

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

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "AUC",
    "Precision", "Recall", "F1", "MCC"
])

results_df.to_csv("model_results.csv", index=False)

print(results_df)

# Create the 'model' directory
if not os.path.exists('model'):
    os.makedirs('model')
    print("Created directory: model/")
else:
    print("Directory 'model/' already exists.")

# Create an empty __init__.py file inside 'model' to make it a Python package
with open('model/__init__.py', 'w') as f:
    pass
print("Created model/__init__.py")

# Create model/logistic_regression.py
model_code = """from sklearn.linear_model import LogisticRegression\n\ndef get_model():\n    return LogisticRegression(random_state=42)"""
with open('model/logistic_regression.py', 'w') as f:
    f.write(model_code)
print("Created model/logistic_regression.py")

# Create model/decision_tree.py
model_code = """from sklearn.tree import DecisionTreeClassifier\n\ndef get_model():\n    return DecisionTreeClassifier(random_state=42)"""
with open('model/decision_tree.py', 'w') as f:
    f.write(model_code)
print("Created model/decision_tree.py")

# Create model/knn.py
model_code = """from sklearn.neighbors import KNeighborsClassifier\n\ndef get_model():\n    return KNeighborsClassifier()"""
with open('model/knn.py', 'w') as f:
    f.write(model_code)
print("Created model/knn.py")

# Create model/naive_bayes.py
model_code = """from sklearn.naive_bayes import GaussianNB\n\ndef get_model():\n    return GaussianNB()"""
with open('model/naive_bayes.py', 'w') as f:
    f.write(model_code)
print("Created model/naive_bayes.py")

# Create model/random_forest.py
model_code = """from sklearn.ensemble import RandomForestClassifier\n\ndef get_model():\n    return RandomForestClassifier(random_state=42)"""
with open('model/random_forest.py', 'w') as f:
    f.write(model_code)
print("Created model/random_forest.py")

# Create model/xgboost_model.py
model_code = """import xgboost as xgb\n\ndef get_model():\n    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)"""
with open('model/xgboost_model.py', 'w') as f:
    f.write(model_code)
print("Created model/xgboost_model.py")