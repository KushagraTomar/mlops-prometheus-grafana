import json
import os

import pandas as pd
import xgboost as xgb

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "artifacts")


class IncomeModel:
    """Wraps the trained XGBoost booster plus the training-time feature
    statistics needed to build inference rows and compute drift scores."""

    def __init__(self):
        with open(os.path.join(ARTIFACT_DIR, "feature_stats.json")) as f:
            self.stats = json.load(f)

        self.booster = xgb.Booster()
        self.booster.load_model(os.path.join(ARTIFACT_DIR, "model.json"))

        self.numeric_features = self.stats["numeric_features"]
        self.categorical_features = self.stats["categorical_features"]
        self.features = self.stats["features"]

    def _build_frame(self, payload: dict) -> pd.DataFrame:
        row = {}
        for col in self.numeric_features:
            row[col] = [float(payload[col])] if col in payload else [self.stats["numeric"][col]["mean"]]
        for col in self.categorical_features:
            categories = self.stats["categorical"][col]
            value = payload.get(col, categories[0])
            if value not in categories:
                value = categories[0]
            row[col] = pd.Categorical([value], categories=categories)
        df = pd.DataFrame(row)
        return df[self.features]

    def predict(self, payload: dict) -> dict:
        df = self._build_frame(payload)
        dmatrix = xgb.DMatrix(df, enable_categorical=True)
        probability = float(self.booster.predict(dmatrix)[0])
        return {
            "probability": probability,
            "predicted_class": int(probability >= 0.5),
        }

    def drift_scores(self, payload: dict) -> dict:
        """Simple z-score of each numeric feature vs. its training-time
        distribution -- a lightweight, explainable data-drift signal."""
        scores = {}
        for col in self.numeric_features:
            if col in payload:
                s = self.stats["numeric"][col]
                std = s["std"] or 1.0
                scores[col] = (float(payload[col]) - s["mean"]) / std
        return scores
