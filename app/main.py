from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from utils import ChurnModel

app = FastAPI(title="Customer Churn Prediction Service", version="1.0.0")
model = ChurnModel()

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
# FEATURE_DRIFT = Gauge(
#     "ml_feature_drift_zscore",
#     "Z-score of the most recent request's numeric feature vs. its training-time distribution",
#     ["feature"],
# )
# MODEL_INFO = Gauge(
#     "ml_model_info", "Static info about the currently loaded model (always 1)",
#     ["model_version", "model_type"],
# )
# MODEL_ACCURACY = Gauge("ml_model_training_accuracy", "Held-out accuracy recorded at training time")
# MODEL_AUC = Gauge("ml_model_training_auc", "Held-out ROC AUC recorded at training time")

# MODEL_INFO.labels(model_version=app.version, model_type="xgboost").set(1)
# MODEL_ACCURACY.set(model.stats["metrics"]["accuracy"])
# MODEL_AUC.set(model.stats["metrics"]["roc_auc"])

# Exposes everything registered with prometheus_client at GET /metrics.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class PredictRequest(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float = 0
    SeniorCitizen: float = 0
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        row = payload.model_dump()
        result = model.predict(row)
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(exc))

    PREDICTIONS_TOTAL.labels(predicted_class=str(result["predicted_class"])).inc()
    PREDICTION_PROBABILITY.observe(result["probability"])
    # for feature, z in model.drift_scores(row).items():
    #     FEATURE_DRIFT.labels(feature=feature).set(z)

    return result
