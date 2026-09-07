"""Train an XGBoost classifier on the real-world IBM Telco Customer Churn
dataset and export the artifacts (model + feature statistics) consumed by
the serving API.

Run:
    python model/train.py
"""
import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw", "Telco-Customer-Churn.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)
    # Download the dataset
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Downloading the Telco Customer Churn dataset from {DATA_URL} ...")
        pd.read_csv(DATA_URL).to_csv(RAW_CSV_PATH, index=False)
    else:
        print(f"Using cached dataset at {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)

    # TotalCharges loads as a string column because a few rows are blank.
    # errors="coerce" turns those blanks into NaN instead of raising, and
    # fillna(0.0) fills them with 0, blanks belong to brand-new customers 
    # (tenure=0) who haven't been billed yet, not actually-missing data.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Binary label XGBoost needs: True/False -> 1/0 (1 = customer churned).
    df["target"] = (df["Churn"] == "Yes").astype(int)
    for col in CATEGORICAL_FEATURES:
        # required to enable_categorical=True to split
        # these natively instead of needing one-hot encoding.
        df[col] = df[col].astype(str).astype("category")
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype(float)
    return df


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_data()
    X = df[FEATURES]
    y = df["target"]

    print(f"X: {X.head()}, y: {y.head()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC : {auc:.4f}")

    model.save_model(os.path.join(ARTIFACT_DIR, "model.json"))

    feature_stats = {
        "numeric": {
            col: {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std() or 1.0),
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
            }
            for col in NUMERIC_FEATURES
        },
        "categorical": {
            col: sorted(X_train[col].cat.categories.tolist())
            for col in CATEGORICAL_FEATURES
        },
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": {"accuracy": acc, "roc_auc": auc},
    }
    with open(os.path.join(ARTIFACT_DIR, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)

    print(f"Artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
