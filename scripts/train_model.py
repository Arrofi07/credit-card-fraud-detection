"""Reproducible entrypoint: raw CSV -> trained model artifact.

Trains the shipped model (class-weighted XGBoost — see src/train.py for why)
and writes:
  - models/fraud_model.joblib   (gitignored artifact, used by the API later)
  - reports/metrics.json        (tracked — test-set metrics for this run)

Usage: python scripts/train_model.py
"""

import json
import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_data, split_data
from src.evaluate import best_threshold_for_f1, evaluate
from src.train import train_xgboost

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_xgboost(X_train, y_train)

    threshold = best_threshold_for_f1(model, X_test, y_test)
    metrics = evaluate(model, X_test, y_test, threshold=threshold)
    metrics["model"] = "xgboost_class_weighted"

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
