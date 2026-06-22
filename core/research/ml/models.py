from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any


class IMLModel(ABC):
    @abstractmethod
    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: list[dict[str, float]]) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def feature_importances(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "IMLModel":
        raise NotImplementedError


class NoOpMLModel(IMLModel):
    """Neutral research-only model used to wire ML safely."""

    model_type = "noop"

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        return None

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [0 for _ in x]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        return [0.5 for _ in x]

    def feature_importances(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model_type": self.model_type}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "NoOpMLModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model payload: {payload}")
        return cls()


class LogisticRegressionMLModel(IMLModel):
    """Deterministic scikit-learn logistic-regression baseline."""

    model_type = "logistic_regression"

    def __init__(
        self,
        random_seed: int = 42,
        max_iterations: int = 1_000,
        l2_penalty: float = 0.01,
        class_weight: str | None = None,
    ):
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        self.l2_penalty = l2_penalty
        self.class_weight = class_weight
        self.feature_names: list[str] = []
        self.model: Any = None

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        if len(x_train) != len(y_train):
            raise ValueError("Features and labels must have the same length")
        if not x_train:
            return

        self.feature_names = sorted(x_train[0])
        if len(set(y_train)) < 2:
            raise ValueError("Logistic regression requires both label classes in training")
        LogisticRegression, _ = _scikit_learn_dependencies()
        self.model = LogisticRegression(
            C=1.0 / self.l2_penalty,
            max_iter=self.max_iterations,
            random_state=self.random_seed,
            solver="lbfgs",
            class_weight=self.class_weight,
        )
        self.model.fit(self._matrix(x_train), y_train)

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(probability >= 0.5) for probability in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        if self.model is None:
            return [0.5 for _ in x]
        return self.model.predict_proba(self._matrix(x))[:, 1].tolist()

    def feature_importances(self) -> dict[str, float]:
        return {
            name: abs(float(coefficient))
            for name, coefficient in zip(self.feature_names, self.model.coef_[0])
        } if self.model is not None else {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _, joblib = _scikit_learn_dependencies()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionMLModel":
        _, joblib = _scikit_learn_dependencies()
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"Unsupported model payload: {type(model).__name__}")
        return model

    def _matrix(self, rows: list[dict[str, float]]) -> list[list[float]]:
        return [
            [float(row[name]) for name in self.feature_names]
            for row in rows
        ]


class TreeClassifierMLModel(IMLModel):
    """Constrained tree baseline for nonlinear research comparisons."""

    def __init__(self, model_type: str, random_seed: int = 42):
        self.model_type = model_type
        self.random_seed = random_seed
        self.feature_names: list[str] = []
        self.model: Any = None

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        if not x_train:
            return
        if len(set(y_train)) < 2:
            raise ValueError("Tree classifier requires both label classes in training")
        RandomForestClassifier, GradientBoostingClassifier, _ = _tree_dependencies()
        self.feature_names = sorted(x_train[0])
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=12,
                class_weight="balanced",
                random_state=self.random_seed,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=12,
                random_state=self.random_seed,
            )
        self.model.fit(self._matrix(x_train), y_train)

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(value >= 0.5) for value in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        if self.model is None:
            return [0.5 for _ in x]
        return self.model.predict_proba(self._matrix(x))[:, 1].tolist()

    def feature_importances(self) -> dict[str, float]:
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, self.model.feature_importances_)
        } if self.model is not None else {}

    def save(self, path: Path) -> None:
        _, _, joblib = _tree_dependencies()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TreeClassifierMLModel":
        _, _, joblib = _tree_dependencies()
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"Unsupported model payload: {type(model).__name__}")
        return model

    def _matrix(self, rows: list[dict[str, float]]) -> list[list[float]]:
        return [[float(row[name]) for name in self.feature_names] for row in rows]


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


def build_ml_model(
    model_type: str,
    random_seed: int = 42,
    class_weight: str | None = None,
) -> IMLModel:
    if model_type in {"noop", "no_op"}:
        return NoOpMLModel()
    if model_type == LogisticRegressionMLModel.model_type:
        return LogisticRegressionMLModel(
            random_seed=random_seed,
            class_weight=class_weight,
        )
    if model_type in {"random_forest", "gradient_boosting"}:
        return TreeClassifierMLModel(model_type=model_type, random_seed=random_seed)
    raise RuntimeError(
        f"Unsupported ml.model_type '{model_type}'. "
        "Available models: gradient_boosting, logistic_regression, noop, random_forest."
    )
