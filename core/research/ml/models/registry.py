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
    model_config: dict[str, Any] | None = None,
) -> IMLModel:
    model_type = model_type.strip().lower()

    if model_type in {"noop", "no_op"}:
        return NoOpMLModel()

    if model_type == LogisticRegressionMLModel.model_type:
        return LogisticRegressionMLModel(
            random_seed=random_seed,
            class_weight=class_weight,
        )

    if model_type in {"random_forest", "gradient_boosting"}:
        config = model_config or {}
        return TreeClassifierMLModel(
            model_type=model_type,
            random_seed=random_seed,
            n_jobs=int(config.get("sklearn_n_jobs", 1)),
        )

    if model_type == "transformer":
        from core.research.ml.models.transformer_model import TransformerSequenceMLModel

        config = model_config or {}
        return TransformerSequenceMLModel(
            sequence_length=int(config.get("sequence_length", 63)),
            d_model=int(config.get("transformer_d_model", 32)),
            nhead=int(config.get("transformer_heads", 4)),
            num_layers=int(config.get("transformer_layers", 2)),
            dim_feedforward=int(config.get("transformer_feedforward", 64)),
            dropout=float(config.get("transformer_dropout", 0.10)),
            epochs=int(config.get("transformer_epochs", 20)),
            batch_size=int(config.get("transformer_batch_size", 32)),
            learning_rate=float(config.get("transformer_learning_rate", 0.001)),
            weight_decay=float(config.get("transformer_weight_decay", 0.0001)),
            random_seed=random_seed,
            device=str(config.get("transformer_device", "cpu")),
        )

    if model_type == "patchtst":
        from core.research.ml.models.patchtst_model import PatchTSTSequenceMLModel

        config = model_config or {}
        return PatchTSTSequenceMLModel(
            sequence_length=int(
                config.get("patchtst_sequence_length", config.get("sequence_length", 126))
            ),
            patch_length=int(config.get("patchtst_patch_length", 16)),
            patch_stride=int(config.get("patchtst_patch_stride", 8)),
            d_model=int(config.get("patchtst_d_model", 64)),
            nhead=int(config.get("patchtst_heads", 4)),
            num_layers=int(config.get("patchtst_layers", 2)),
            dim_feedforward=int(config.get("patchtst_feedforward", 128)),
            dropout=float(config.get("patchtst_dropout", 0.10)),
            epochs=int(config.get("patchtst_epochs", 30)),
            batch_size=int(config.get("patchtst_batch_size", 32)),
            learning_rate=float(config.get("patchtst_learning_rate", 0.001)),
            weight_decay=float(config.get("patchtst_weight_decay", 0.0001)),
            random_seed=random_seed,
            device=str(
                config.get("patchtst_device", config.get("transformer_device", "cpu"))
            ),
            pos_weight=config.get("patchtst_pos_weight", "auto"),
        )

    if model_type == "dlinear":
        from core.research.ml.models.dlinear_model import DLinearSequenceMLModel

        config = model_config or {}
        return DLinearSequenceMLModel(
            sequence_length=int(
                config.get("dlinear_sequence_length", config.get("sequence_length", 126))
            ),
            epochs=int(config.get("dlinear_epochs", 50)),
            batch_size=int(config.get("dlinear_batch_size", 32)),
            learning_rate=float(config.get("dlinear_learning_rate", 0.001)),
            weight_decay=float(config.get("dlinear_weight_decay", 0.001)),
            random_seed=random_seed,
            device=str(
                config.get("dlinear_device", config.get("transformer_device", "cpu"))
            ),
            pos_weight=config.get("dlinear_pos_weight", "auto"),
        )

    if model_type == "itransformer":
        from core.research.ml.models.itransformer_model import ITransformerSequenceMLModel

        config = model_config or {}
        return ITransformerSequenceMLModel(
            sequence_length=int(
                config.get("itransformer_sequence_length", config.get("sequence_length", 126))
            ),
            d_model=int(config.get("itransformer_d_model", 64)),
            nhead=int(config.get("itransformer_heads", 4)),
            num_layers=int(config.get("itransformer_layers", 2)),
            dim_feedforward=int(config.get("itransformer_feedforward", 128)),
            dropout=float(config.get("itransformer_dropout", 0.10)),
            epochs=int(config.get("itransformer_epochs", 30)),
            batch_size=int(config.get("itransformer_batch_size", 32)),
            learning_rate=float(config.get("itransformer_learning_rate", 0.001)),
            weight_decay=float(config.get("itransformer_weight_decay", 0.0001)),
            random_seed=random_seed,
            device=str(
                config.get("itransformer_device", config.get("transformer_device", "cpu"))
            ),
            pos_weight=config.get("itransformer_pos_weight", "auto"),
        )

    if model_type == "momentum_transformer":
        from core.research.ml.models.momentum_transformer_model import (
            MomentumTransformerSequenceMLModel,
        )

        config = model_config or {}
        return MomentumTransformerSequenceMLModel(
            sequence_length=int(
                config.get(
                    "momentum_transformer_sequence_length",
                    config.get("sequence_length", 126),
                )
            ),
            d_model=int(config.get("momentum_transformer_d_model", 64)),
            nhead=int(config.get("momentum_transformer_heads", 4)),
            num_layers=int(config.get("momentum_transformer_layers", 2)),
            dim_feedforward=int(config.get("momentum_transformer_feedforward", 128)),
            dropout=float(config.get("momentum_transformer_dropout", 0.10)),
            epochs=int(config.get("momentum_transformer_epochs", 30)),
            batch_size=int(config.get("momentum_transformer_batch_size", 32)),
            learning_rate=float(config.get("momentum_transformer_learning_rate", 0.001)),
            weight_decay=float(config.get("momentum_transformer_weight_decay", 0.0001)),
            random_seed=random_seed,
            device=str(
                config.get(
                    "momentum_transformer_device",
                    config.get("transformer_device", "cpu"),
                )
            ),
            pos_weight=config.get("momentum_transformer_pos_weight", "auto"),
            size_multiplier_floor=float(
                config.get("momentum_transformer_size_multiplier_floor", 0.25)
            ),
            size_multiplier_ceiling=float(
                config.get("momentum_transformer_size_multiplier_ceiling", 1.25)
            ),
        )

    if model_type == "multitask_transformer":
        from core.research.ml.models.multitask_transformer_model import (
            DEFAULT_REGRESSION_TARGETS,
            MultiTaskTransformerSequenceMLModel,
        )

        config = model_config or {}
        regression_targets = list(
            config.get("multitask_regression_targets", DEFAULT_REGRESSION_TARGETS)
        )
        nested_loss = config.get("multitask_loss", {})
        nested_weights = (
            nested_loss.get("regression_weights", {})
            if isinstance(nested_loss, dict)
            else {}
        )
        regression_weights = {
            target: float(
                config.get(
                    f"multitask_{target}_weight",
                    nested_weights.get(target, 0.2),
                )
            )
            for target in regression_targets
        }
        classification_weight = (
            nested_loss.get("classification_weight", 1.0)
            if isinstance(nested_loss, dict)
            else 1.0
        )
        regression_loss = (
            nested_loss.get("regression_loss", config.get("multitask_regression_loss", "huber"))
            if isinstance(nested_loss, dict)
            else config.get("multitask_regression_loss", "huber")
        )
        huber_delta = (
            nested_loss.get("huber_delta", config.get("multitask_huber_delta", 1.0))
            if isinstance(nested_loss, dict)
            else config.get("multitask_huber_delta", 1.0)
        )
        return MultiTaskTransformerSequenceMLModel(
            sequence_length=int(
                config.get(
                    "multitask_transformer_sequence_length",
                    config.get("sequence_length", 63),
                )
            ),
            d_model=int(config.get("multitask_transformer_d_model", 32)),
            nhead=int(config.get("multitask_transformer_heads", 4)),
            num_layers=int(config.get("multitask_transformer_layers", 2)),
            dim_feedforward=int(config.get("multitask_transformer_feedforward", 64)),
            dropout=float(config.get("multitask_transformer_dropout", 0.10)),
            epochs=int(config.get("multitask_transformer_epochs", 20)),
            batch_size=int(config.get("multitask_transformer_batch_size", 32)),
            learning_rate=float(config.get("multitask_transformer_learning_rate", 0.001)),
            weight_decay=float(config.get("multitask_transformer_weight_decay", 0.0001)),
            random_seed=random_seed,
            device=str(
                config.get(
                    "multitask_transformer_device",
                    config.get("transformer_device", "cpu"),
                )
            ),
            regression_targets=regression_targets,
            classification_weight=float(
                config.get("multitask_classification_weight", classification_weight)
            ),
            regression_loss=str(regression_loss),
            huber_delta=float(huber_delta),
            regression_weights=regression_weights,
        )

    if model_type == "market_context_encoder":
        from core.research.ml.models.market_context_encoder_model import (
            MarketContextEncoderMLModel,
        )

        config = model_config or {}
        return MarketContextEncoderMLModel(
            sequence_length=int(
                config.get(
                    "market_context_sequence_length",
                    config.get("sequence_length", 63),
                )
            ),
            hidden_size=int(config.get("market_context_hidden_size", 32)),
            epochs=int(config.get("market_context_epochs", 20)),
            batch_size=int(config.get("market_context_batch_size", 32)),
            learning_rate=float(config.get("market_context_learning_rate", 0.001)),
            weight_decay=float(config.get("market_context_weight_decay", 0.0001)),
            dropout=float(config.get("market_context_dropout", 0.10)),
            random_seed=random_seed,
            device=str(
                config.get("market_context_device", config.get("transformer_device", "cpu"))
            ),
            risk_multiplier_floor=float(
                config.get("market_context_risk_multiplier_floor", 0.25)
            ),
            risk_multiplier_ceiling=float(
                config.get("market_context_risk_multiplier_ceiling", 1.25)
            ),
        )

    if model_type == "news_analysis_transformer":
        from core.research.ml.models.news_analysis_transformer_model import (
            NewsAnalysisTransformerMLModel,
        )

        config = model_config or {}
        return NewsAnalysisTransformerMLModel(
            sequence_length=int(
                config.get(
                    "news_transformer_sequence_length",
                    config.get("sequence_length", 63),
                )
            ),
            d_model=int(config.get("news_transformer_d_model", 32)),
            nhead=int(config.get("news_transformer_heads", 4)),
            num_layers=int(config.get("news_transformer_layers", 1)),
            dim_feedforward=int(config.get("news_transformer_feedforward", 64)),
            dropout=float(config.get("news_transformer_dropout", 0.10)),
            epochs=int(config.get("news_transformer_epochs", 20)),
            batch_size=int(config.get("news_transformer_batch_size", 32)),
            learning_rate=float(config.get("news_transformer_learning_rate", 0.001)),
            weight_decay=float(config.get("news_transformer_weight_decay", 0.0001)),
            random_seed=random_seed,
            device=str(
                config.get("news_transformer_device", config.get("transformer_device", "cpu"))
            ),
        )

    if model_type == "temporal_fusion_transformer":
        from core.research.ml.models.temporal_fusion_transformer_model import (
            DEFAULT_KNOWN_FUTURE_FEATURES,
            TemporalFusionTransformerMLModel,
        )

        config = model_config or {}
        return TemporalFusionTransformerMLModel(
            sequence_length=int(
                config.get("tft_encoder_length", config.get("sequence_length", 64))
            ),
            hidden_size=int(config.get("tft_hidden_size", 64)),
            attention_heads=int(config.get("tft_attention_heads", 4)),
            num_layers=int(config.get("tft_layers", config.get("tft_lstm_layers", 1))),
            dropout=float(config.get("tft_dropout", 0.15)),
            epochs=int(config.get("tft_epochs", 30)),
            batch_size=int(config.get("tft_batch_size", 64)),
            learning_rate=float(config.get("tft_learning_rate", 0.001)),
            weight_decay=float(config.get("tft_weight_decay", 0.0005)),
            random_seed=random_seed,
            device=str(config.get("tft_device", config.get("transformer_device", "cpu"))),
            known_future_features=list(
                config.get("tft_known_future_features", DEFAULT_KNOWN_FUTURE_FEATURES)
            ),
        )

    raise RuntimeError(
        f"Unsupported ml.model_type '{model_type}'. "
        "Available models: dlinear, gradient_boosting, logistic_regression, "
        "itransformer, market_context_encoder, momentum_transformer, "
        "multitask_transformer, news_analysis_transformer, noop, patchtst, "
        "random_forest, temporal_fusion_transformer, transformer."
    )
