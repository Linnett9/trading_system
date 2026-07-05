from __future__ import annotations

from typing import Any


def _scikit_learn_dependencies() -> tuple[Any, Any]:
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "ML logistic regression requires scikit-learn and its dependencies. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc
    return LogisticRegression, joblib
def _tree_dependencies() -> tuple[Any, Any, Any]:
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "ML tree models require scikit-learn. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return RandomForestClassifier, GradientBoostingClassifier, joblib
