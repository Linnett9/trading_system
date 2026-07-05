from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.models.registry_dependencies import (
    _scikit_learn_dependencies,
    _tree_dependencies,
)
from core.research.ml.models.registry_protocol import IMLModel


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
        if self.model is None:
            return {}
        return {
            name: abs(float(coefficient))
            for name, coefficient in zip(self.feature_names, self.model.coef_[0])
        }

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

    def __init__(self, model_type: str, random_seed: int = 42, n_jobs: int = 1):
        self.model_type = model_type
        self.random_seed = random_seed
        self.n_jobs = int(n_jobs)
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
                n_jobs=self.n_jobs,
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
        if self.model is None:
            return {}
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, self.model.feature_importances_)
        }

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
