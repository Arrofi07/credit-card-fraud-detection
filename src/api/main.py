"""FastAPI service wrapping the trained fraud model.

Run: uvicorn src.api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionResponse, Transaction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
DEFAULT_THRESHOLD = 0.5

_model = None
_threshold = DEFAULT_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _threshold

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python scripts/train_model.py` first."
        )
    _model = joblib.load(MODEL_PATH)

    if METRICS_PATH.exists():
        _threshold = json.loads(METRICS_PATH.read_text()).get(
            "threshold", DEFAULT_THRESHOLD
        )

    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts fraud probability for a credit card transaction "
    "using a class-weighted XGBoost model trained on the Kaggle "
    "mlg-ulb/creditcardfraud dataset.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = pd.DataFrame([transaction.model_dump()])
    proba = float(_model.predict_proba(X)[:, 1][0])

    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= _threshold,
        threshold=_threshold,
    )
