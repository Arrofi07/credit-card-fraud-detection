"""Evaluation metrics for the (severely imbalanced) fraud classifier.

Never report plain accuracy as a success metric on this dataset.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def best_threshold_for_f1(model, X_test, y_test) -> float:
    """Sweep the PR curve and return the threshold that maximizes F1.

    Used only to report an achievable operating point — the model actually
    shipped in Phase 4 should let callers pick their own precision/recall
    tradeoff via the probability score, not hardcode this threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
    return float(thresholds[best_idx])
