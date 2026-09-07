from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from model_utils import IncomeModel

app = FastAPI(title="Income Prediction Service", version="1.0.0")
model = IncomeModel()

# ---------------------------------------------------------------------------
# Custom, ML-specific Prometheus metrics.
# The instrumentator below already covers generic HTTP metrics (request
# count, latency histogram, response size); these track things that matter
# specifically for a deployed model: what it's predicting, whether incoming
# data still looks like training data, and how it performed at train time.
# ---------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total", "Total number of predictions served, by predicted class",
    ["predicted_class"],
)
PREDICTION_PROBABILITY = Histogram(
    "ml_prediction_probability",
    "Distribution of predicted positive-class probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total", "Total number of failed prediction requests",
)
FEATURE_DRIFT = Gauge(
    "ml_feature_drift_zscore",
    "Z-score of the most recent request's numeric feature vs. its training-time distribution",
    ["feature"],
)
MODEL_INFO = Gauge(
    "ml_model_info", "Static info about the currently loaded model (always 1)",
    ["model_version", "model_type"],
)
MODEL_ACCURACY = Gauge("ml_model_training_accuracy", "Held-out accuracy recorded at training time")
MODEL_AUC = Gauge("ml_model_training_auc", "Held-out ROC AUC recorded at training time")

MODEL_INFO.labels(model_version=app.version, model_type="xgboost").set(1)
MODEL_ACCURACY.set(model.stats["metrics"]["accuracy"])
MODEL_AUC.set(model.stats["metrics"]["roc_auc"])

# Adds request-count / latency-histogram / response-size metrics automatically
# and exposes everything registered with prometheus_client at GET /metrics.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: float
    fnlwgt: float = 200000
    education_num: float = Field(10, alias="education-num")
    capital_gain: float = Field(0, alias="capital-gain")
    capital_loss: float = Field(0, alias="capital-loss")
    hours_per_week: float = Field(40, alias="hours-per-week")
    workclass: str
    education: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field("United-States", alias="native-country")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        row = payload.model_dump(by_alias=True)
        result = model.predict(row)
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(exc))

    PREDICTIONS_TOTAL.labels(predicted_class=str(result["predicted_class"])).inc()
    PREDICTION_PROBABILITY.observe(result["probability"])
    for feature, z in model.drift_scores(row).items():
        FEATURE_DRIFT.labels(feature=feature).set(z)

    return result
