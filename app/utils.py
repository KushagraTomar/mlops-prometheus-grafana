import json
import os
0
import pandas as pd
import xgboost as xgb

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "artifacts")


class ChurnModel:
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

    def _build_dataframe(self, payload: dict) -> pd.DataFrame:
        """Build a single-row DataFrame from the JSON payload"""
        row = {}
        for col in self.numeric_features:
            row[col] = [float(payload[col])]
        for col in self.categorical_features:
            categories = self.stats["categorical"][col]
            value = payload[col] if payload[col] in categories else categories[0]
            row[col] = pd.Categorical([value], categories=categories)
        df = pd.DataFrame(row)
        return df[self.features]

    def predict(self, payload: dict) -> dict:
        df = self._build_dataframe(payload)
        dmatrix = xgb.DMatrix(df, enable_categorical=True)
        probability = float(self.booster.predict(dmatrix)[0])
        return {
            "probability": probability,
            "predicted_class": int(probability >= 0.5),
        }

    # def drift_scores(self, payload: dict) -> dict:
    #     """Simple z-score of each numeric feature vs. its training-time
    #     distribution -- a lightweight, explainable data-drift signal."""
    #     scores = {}
    #     for col in self.numeric_features:
    #         s = self.stats["numeric"][col]
    #         std = s["std"] or 1.0
    #         scores[col] = (float(payload[col]) - s["mean"]) / std
    #     return scores
