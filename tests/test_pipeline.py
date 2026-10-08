import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.evaluate import evaluate
from src.train import train_logistic_baseline, train_xgboost


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        weights=[0.98, 0.02],
        random_state=42,
    )
    split = 1600
    return X[:split], X[split:], y[:split], y[split:]


def test_logistic_baseline_predicts_both_classes(imbalanced_data):
    X_train, X_test, y_train, y_test = imbalanced_data
    model = train_logistic_baseline(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_trains_and_predicts(imbalanced_data):
    X_train, X_test, y_train, y_test = imbalanced_data
    model = train_xgboost(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_expected_keys(imbalanced_data):
    X_train, X_test, y_train, y_test = imbalanced_data
    model = train_logistic_baseline(X_train, y_train)

    metrics = evaluate(model, X_test, y_test, threshold=0.5)

    assert set(metrics) == {
        "threshold",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }
    assert 0 <= metrics["pr_auc"] <= 1
    assert np.array(metrics["confusion_matrix"]).shape == (2, 2)


def test_evaluate_perfect_predictions_scores_one():
    class PerfectModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    y_test = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0], [0.1], [0.9], [1.0]])

    metrics = evaluate(PerfectModel(), X_test, y_test, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
