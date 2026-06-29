from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.research.framework.registry import ModelRegistry
from core.research.ml.stock_level_benchmark_types import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    SEQUENCE_MODEL_NAMES,
    TABULAR_MODEL_NAMES,
)


@dataclass(frozen=True)
class TabularModelFactory:
    model_name: str
    random_seed: int
    sklearn_n_jobs: int

    def __call__(self) -> Any:
        return _build_tabular_model(
            self.model_name,
            self.random_seed,
            self.sklearn_n_jobs,
        )


@dataclass(frozen=True)
class SequenceModelFactory:
    architecture: str
    sequence_length: int
    epochs: int
    batch_size: int
    random_seed: int
    device: str
    torch_num_threads: int | None

    def __call__(self) -> Any:
        from core.research.ml.stock_level.stock_level_sequence_regressors import (
            SequenceRegressorConfig,
            TorchSequenceReturnRegressor,
        )

        return TorchSequenceReturnRegressor(
            SequenceRegressorConfig(
                architecture=self.architecture,
                sequence_length=self.sequence_length,
                epochs=self.epochs,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                device=self.device,
                torch_num_threads=self.torch_num_threads,
            )
        )


def _model_factories(
    random_seed: int,
    sklearn_n_jobs: int,
) -> dict[str, Callable[[], Any]]:
    return dict(
        stock_ranker_model_registry(
            random_seed=random_seed,
            sklearn_n_jobs=sklearn_n_jobs,
        ).items()
    )


def stock_ranker_model_registry(
    *,
    random_seed: int,
    sklearn_n_jobs: int,
) -> ModelRegistry[Callable[[], Any]]:
    registry: ModelRegistry[Callable[[], Any]] = ModelRegistry()
    for name in TABULAR_MODEL_NAMES:
        registry.register(
            name,
            TabularModelFactory(name, random_seed, sklearn_n_jobs),
            metadata={"family": "tabular_regressor"},
        )
    return registry


def _build_tabular_model(
    model_name: str,
    random_seed: int,
    sklearn_n_jobs: int,
) -> Any:
    try:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Stock-level ranking benchmark requires scikit-learn. "
            "Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc

    def scaled_target(regressor: Any) -> Any:
        return TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )

    if model_name == "ridge":
        return scaled_target(
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        )
    if model_name == "elastic_net":
        return scaled_target(
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.25,
                    max_iter=5_000,
                    random_state=random_seed,
                ),
            )
        )
    if model_name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=20,
                max_features=0.8,
                n_jobs=sklearn_n_jobs,
                random_state=random_seed,
            ),
        )
    if model_name == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=75,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=20,
                loss="huber",
                random_state=random_seed,
            ),
        )
    raise ValueError(f"Unsupported stock-level tabular model: {model_name}")


def _sequence_model_factories(
    *,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    random_seed: int,
    device: str,
    torch_num_threads: int | None,
) -> dict[str, Callable[[], Any]]:
    return {
        name: SequenceModelFactory(
            architecture=name,
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device,
            torch_num_threads=torch_num_threads,
        )
        for name in SEQUENCE_MODEL_NAMES
    }


def _sequence_feature_columns(
    model_name: str,
    news_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[str, ...]:
    if model_name in {"market_context_encoder", "temporal_fusion_transformer"}:
        return (*feature_columns, *CONTEXT_COLUMNS)
    if model_name == "news_analysis_transformer":
        return (*feature_columns, *news_columns)
    return feature_columns
