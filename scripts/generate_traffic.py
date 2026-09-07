"""Send synthetic, real-world-like traffic to the /predict endpoint.

Replays rows sampled from the held-out test set so Prometheus/Grafana have
realistic request-rate, latency, and prediction-distribution data to show,
and periodically injects out-of-distribution values (drift) or malformed
requests (errors) so the drift and error-rate panels/alerts have something
to react to.

Run (after the app is up):
    python scripts/generate_traffic.py --rps 5 --drift
"""
import argparse
import json
import os
import random
import time

import requests

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_requests.json")


def load_samples():
    with open(DATA_PATH) as f:
        return json.load(f)


def maybe_drift(row: dict, drift_enabled: bool) -> dict:
    if drift_enabled and random.random() < 0.15:
        row = dict(row)
        row["age"] = row.get("age", 40) * random.uniform(2.5, 4.0)
        row["hours-per-week"] = row.get("hours-per-week", 40) * random.uniform(2.0, 3.0)
    return row


def maybe_break(row: dict, error_rate: float) -> dict:
    if random.random() < error_rate:
        row = dict(row)
        row.pop("workclass", None)  # required field -> triggers a validation error
    return row


def main():
    parser = argparse.ArgumentParser(description="Traffic generator for the ML service")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--rps", type=float, default=3.0, help="requests per second")
    parser.add_argument("--drift", action="store_true", help="inject feature drift")
    parser.add_argument("--error-rate", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=0, help="seconds to run, 0 = forever")
    args = parser.parse_args()

    samples = load_samples()
    delay = 1.0 / args.rps if args.rps > 0 else 0.2
    start = time.time()
    sent = 0

    print(
        f"Sending traffic to {args.url} at ~{args.rps} req/s "
        f"(drift={'on' if args.drift else 'off'}, error_rate={args.error_rate})"
    )

    while True:
        row = dict(random.choice(samples))
        row.pop("target", None)
        row = maybe_drift(row, args.drift)
        row = maybe_break(row, args.error_rate)
        try:
            resp = requests.post(args.url, json=row, timeout=5)
            sent += 1
            if sent % 25 == 0:
                print(f"[{sent}] status={resp.status_code}")
        except requests.RequestException as exc:
            print(f"request failed: {exc}")

        if args.duration and (time.time() - start) > args.duration:
            break
        time.sleep(delay)


if __name__ == "__main__":
    main()
