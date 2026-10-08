"""Model training helpers.

Imbalance strategy: class weighting / scale_pos_weight, not SMOTE. On this
dataset the two approaches score within noise of each other in local
comparisons (see notebooks/02_modeling.ipynb), and class weighting avoids
synthetic minority samples leaking signal across CV folds. Keep this as the
default; revisit only if a documented comparison shows SMOTE clearly wins.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_logistic_baseline(X_train, y_train, random_state: int = 42) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_logistic_smote(X_train, y_train, random_state: int = 42) -> ImbPipeline:
    """Comparison model: SMOTE-resampled training data + plain LogisticRegression."""
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf
