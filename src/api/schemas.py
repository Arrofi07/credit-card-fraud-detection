"""Request/response schemas for the fraud prediction API.

Fields mirror the creditcardfraud dataset columns exactly (Time, V1-V28,
Amount) — V1-V28 are PCA-anonymized and have no real-world meaning.
"""

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since the first transaction in the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 406.0,
                "V1": -2.3122,
                "V2": 1.9519,
                "V3": -1.6098,
                "V4": 3.9979,
                "V5": -0.5222,
                "V6": -1.4265,
                "V7": -2.5373,
                "V8": 1.3916,
                "V9": -2.7700,
                "V10": -2.7722,
                "V11": 3.2021,
                "V12": -2.8999,
                "V13": -0.5952,
                "V14": -4.2892,
                "V15": 0.3897,
                "V16": -1.1407,
                "V17": -2.8301,
                "V18": -0.0168,
                "V19": 0.4166,
                "V20": 0.1270,
                "V21": 0.5172,
                "V22": -0.0350,
                "V23": -0.4652,
                "V24": 0.3202,
                "V25": 0.0445,
                "V26": 0.1778,
                "V27": 0.2611,
                "V28": -0.1433,
                "Amount": 0.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model's predicted probability of fraud, 0-1")
    is_fraud: bool = Field(..., description="fraud_probability >= threshold")
    threshold: float = Field(..., description="Decision threshold applied to fraud_probability")
