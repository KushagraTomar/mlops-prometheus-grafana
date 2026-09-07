"""Train an XGBoost classifier on the UCI Adult Census Income dataset and
export the artifacts (model + feature statistics) consumed by the serving API.

Run:
    python model/train.py
"""
import json
import os

import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    print("Fetching the Adult Income dataset from OpenML (cached after first run)...")
    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace("_", "-") for c in df.columns]
    target_col = "class" if "class" in df.columns else "income"
    df = df.rename(columns={target_col: "target"})
    df["target"] = df["target"].astype(str).str.contains(">50K").astype(int)
    for col in CATEGORICAL_FEATURES:
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
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

    # Held-out sample used by the traffic generator to replay realistic requests.
    sample = X_test.copy()
    sample["target"] = y_test.values
    sample.head(500).to_json(
        os.path.join(DATA_DIR, "sample_requests.json"), orient="records"
    )

    print(f"Artifacts written to {ARTIFACT_DIR}")
    print(f"Sample requests written to {DATA_DIR}/sample_requests.json")


if __name__ == "__main__":
    main()
