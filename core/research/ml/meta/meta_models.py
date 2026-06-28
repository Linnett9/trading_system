from __future__ import annotations

import ctypes.util
from dataclasses import dataclass
import platform
from typing import Any


@dataclass
class MetaLearnerModel:
    model_type: str
    feature_names: list[str]
    estimator: Any = None
    constant_probability: float | None = None

    def predict_proba(self, features: list[dict[str, float]]) -> list[float]:
        if self.constant_probability is not None:
            return [self.constant_probability for _ in features]
        matrix = _feature_matrix(features, self.feature_names)
        probabilities = self.estimator.predict_proba(matrix)[:, 1].tolist()
        return [float(value) for value in probabilities]


def _fit_meta_model(
    model_type: str,
    features: list[dict[str, float]],
    labels: list[int],
    random_seed: int,
    sklearn_n_jobs: int = 1,
) -> MetaLearnerModel:
    normalized = _normalize_meta_model_type(model_type)
    feature_names = sorted(features[0]) if features else []
    if not features:
        return MetaLearnerModel(normalized, feature_names, constant_probability=0.5)
    if len(set(labels)) < 2:
        probability = sum(labels) / len(labels) if labels else 0.5
        return MetaLearnerModel(
            normalized,
            feature_names,
            constant_probability=float(probability),
        )

    matrix = _feature_matrix(features, feature_names)
    if normalized in {"logistic_regression", "ridge_logistic"}:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=100.0 if normalized == "logistic_regression" else 1.0,
                max_iter=5_000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=random_seed,
            ),
        )
    elif normalized == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=sklearn_n_jobs,
        )
    elif normalized == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        estimator = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=12,
            random_state=random_seed,
        )
    elif normalized == "lightgbm":
        _ensure_lightgbm_runtime_available()
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM meta learner requested but lightgbm is not installed") from exc
        estimator = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_child_samples=12,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            verbose=-1,
            n_jobs=sklearn_n_jobs,
        )
    else:
        raise RuntimeError(f"Unsupported ml.meta_model_type '{model_type}'")

    estimator.fit(matrix, labels)
    return MetaLearnerModel(normalized, feature_names, estimator=estimator)


def _ensure_lightgbm_runtime_available() -> None:
    if platform.system() != "Darwin":
        return
    if ctypes.util.find_library("omp") or ctypes.util.find_library("libomp"):
        return
    raise RuntimeError(
        "LightGBM meta learner requested but macOS libomp is not available. "
        "Install it with 'brew install libomp' or remove lightgbm from "
        "ml.meta_model_types."
    )


def _feature_matrix(
    features: list[dict[str, float]],
    feature_names: list[str],
) -> list[list[float]]:
    return [[float(row.get(name, 0.0)) for name in feature_names] for row in features]


def _normalize_meta_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    aliases = {
        "logistic": "logistic_regression",
        "meta_ensemble_logistic": "logistic_regression",
        "ridge": "ridge_logistic",
        "gbm": "gradient_boosting",
        "light_gradient_boosting": "lightgbm",
    }
    return aliases.get(normalized, normalized)


def _meta_ensemble_name(model_type: str) -> str:
    normalized = _normalize_meta_model_type(model_type)
    if normalized == "logistic_regression":
        return "meta_ensemble_logistic"
    return f"meta_ensemble_{normalized}"


def _meta_model_types(
    ml_config: dict[str, Any],
    selected_meta_model_type: str,
) -> list[str]:
    configured = ml_config.get(
        "meta_model_types",
        [
            "logistic_regression",
            "ridge_logistic",
            "random_forest",
            "gradient_boosting",
            "lightgbm",
        ],
    )
    model_types = [_normalize_meta_model_type(value) for value in configured]
    selected = _normalize_meta_model_type(selected_meta_model_type)
    if selected not in model_types:
        model_types.insert(0, selected)
    return list(dict.fromkeys(model_types))
