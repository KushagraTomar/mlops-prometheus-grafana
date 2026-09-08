# MLOps Monitoring Demo: XGBoost + Prometheus + Grafana

A small, end-to-end MLOps project built to *teach* the important concepts of
Prometheus and Grafana, using a real model in the loop rather than a toy
counter. An XGBoost classifier trained on the real-world **IBM Telco
Customer Churn** dataset (predict whether a customer will cancel their
subscription) is served behind FastAPI, instrumented with Prometheus
metrics, scraped by Prometheus, and visualized with a Grafana dashboard
provisioned as code.

## Architecture

```
                 ┌─────────────┐   scrapes /metrics   ┌────────────┐   datasource   ┌─────────┐
 traffic-gen ───►│   FastAPI   │◄──────────────────────│ Prometheus │◄───────────────│ Grafana │
 (simulates      │  + XGBoost  │        every 5s        │  (+ alert  │                │  (dash- │
  real users)    │   model     │                        │   rules)   │                │  board) │
                 └─────────────┘                         └────────────┘                └─────────┘
```

- **model/train.py** — downloads the dataset, trains the XGBoost model, saves
  the model + feature statistics used for drift detection, and saves a sample
  of held-out rows for the traffic generator.
- **app/** — FastAPI service exposing `/predict`, `/health`, and `/metrics`.
- **monitoring/prometheus/** — scrape config + alerting rules.
- **monitoring/grafana/** — datasource + dashboard provisioning (as code).
- **scripts/generate_traffic.py** — replays real held-out rows against the
  API to produce realistic traffic, and can inject drift/errors on demand.

## Quickstart

```bash
# 1. Train the model (downloads the dataset to data/raw/ on first run, ~10s)
pip install -r requirements.txt
python model/train.py

# 2. Build and start the app, Prometheus, and Grafana
docker compose up --build

# 3. (optional) generate realistic traffic including drift + errors
docker compose --profile demo up traffic-generator
# or, from your host:
pip install requests
python scripts/generate_traffic.py --rps 5 --drift
```

Then open:
- App: http://localhost:8000/docs (interactive `/predict` API)
- Metrics (raw): http://localhost:8000/metrics
- Prometheus: http://localhost:9091 (mapped from the container's 9090 to avoid clashing with anything already using 9090/3000 on your host — change the mapping in `docker-compose.yml` if you'd rather use the defaults)
- Grafana: http://localhost:3001 (admin/admin, or anonymous viewer access is enabled) — the "ML Model Monitoring (XGBoost)" dashboard is preloaded

## Prometheus concepts covered

| Concept | Where |
|---|---|
| **Metric types** — Counter, Gauge, Histogram | [app/main.py](app/main.py): `ml_predictions_total` (Counter), `ml_feature_drift_zscore` (Gauge), `ml_prediction_probability` (Histogram) |
| **Instrumenting an app** | `prometheus-fastapi-instrumentator` wires up standard HTTP metrics; `prometheus_client` is used directly for custom ML metrics |
| **Pull-based scraping & static service discovery** | [monitoring/prometheus/prometheus.yml](monitoring/prometheus/prometheus.yml) — `static_configs` targeting `app:8000` |
| **PromQL: `rate()` / `sum()`** | Request rate, error rate, predictions-by-class panels |
| **PromQL: `histogram_quantile()`** | Latency p50/p95/p99 and predicted-probability percentiles, computed from histogram buckets |
| **Labels & cardinality** | `predicted_class`, `feature`, `status`, `method` labels — and why you keep label cardinality bounded (no raw user IDs as labels!) |
| **Alerting rules** | [monitoring/prometheus/alerts.yml](monitoring/prometheus/alerts.yml) — high error rate, high latency, feature drift |
| **Recording-rule-style aggregation** | The error-rate and latency queries show the "aggregate first" pattern alerts rely on |

## Grafana concepts covered

| Concept | Where |
|---|---|
| **Provisioning as code** (no manual clicking) | [monitoring/grafana/provisioning/datasources](monitoring/grafana/provisioning/datasources) and [.../dashboards](monitoring/grafana/provisioning/dashboards) |
| **Datasource config** | `datasource.yml` — Prometheus wired up automatically on container start |
| **Dashboard-as-JSON** | [ml-model-dashboard.json](monitoring/grafana/dashboards/ml-model-dashboard.json) — check this into version control like any other code |
| **Panel types**: timeseries, stat | Latency/rate graphs vs. single-value model-accuracy/AUC stat panels |
| **Thresholds & color coding** | Error-rate and drift panels turn yellow/red past configured thresholds |
| **Auto-refresh** | Dashboard refreshes every 5s to feel "live" |

## What the ML-specific metrics teach

Beyond generic HTTP monitoring, this project demonstrates monitoring
concerns that are specific to a deployed ML model:

- **`ml_predictions_total{predicted_class}`** — is the model's output
  distribution shifting over time (e.g., suddenly predicting one class much
  more often)?
- **`ml_prediction_probability`** (histogram) — is model confidence drifting?
- **`ml_feature_drift_zscore{feature}`** — a simple, explainable drift
  signal: how many standard deviations is an incoming feature value from
  what the model was trained on. Run the traffic generator with `--drift` to
  see `tenure` and `MonthlyCharges` spike (and trigger the
  `FeatureDriftDetected` alert) — e.g. simulating a batch of new,
  high-paying enterprise customers the model never trained on.
- **`ml_model_training_accuracy` / `ml_model_training_auc`** — surfaces
  model quality metrics recorded at training time right alongside live
  serving metrics, so a dashboard viewer has both "is it fast/healthy" and
  "is it any good" in one place.

## Retraining

Re-run `python model/train.py` any time — it re-downloads/re-splits the
data, retrains, and overwrites `model/artifacts/`. Rebuild the app image
(`docker compose up --build app`) to pick up the new model.

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
— 7,043 real telecom customer records (demographics, account info, and
subscribed services) with a binary target: did the customer churn
(cancel) or not. `model/train.py` downloads it once to `data/raw/` and
reuses the cached copy afterwards. It's a good fit here because it's a
well-known, real, freely available tabular dataset with an intuitive
business story — churn prediction is one of the most common real-world
uses of XGBoost — and it has a natural mix of numeric (`tenure`,
`MonthlyCharges`, `TotalCharges`) and categorical (`Contract`,
`InternetService`, `PaymentMethod`, ...) features for reasoning about
drift.

Example request body for `/predict`:

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5,
  "SeniorCitizen": 0,
  "gender": "Female",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```
