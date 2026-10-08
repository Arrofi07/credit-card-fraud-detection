"""Data loading and splitting for the creditcardfraud dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
TARGET_COL = "Class"


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ./scripts/download_data.sh first."
        )
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split. Test set keeps the real-world class
    imbalance intact — it must reflect production conditions."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
