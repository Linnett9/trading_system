from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from core.research.ml.stock_level_alpha_features import ENGINEERED_FEATURE_COLUMNS
from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.parallel import ParallelTaskExecutor
from core.research.framework.ranking import CrossSectionalRankingEvaluator
from core.research.framework.registry import ModelRegistry
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.framework.walk_forward import (
    ExpandingWindowSpec,
    ExpandingWindowSplitter,
)


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
TARGET_COLUMN = "actual_forward_return_10d"
FEATURE_COLUMNS = (
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
CONTEXT_COLUMNS = (
    "breadth_above_sma_200",
    "spy_realized_volatility_21d",
    "spy_realized_volatility_63d",
    "spy_max_drawdown_63d",
    "spy_max_drawdown_126d",
)
AUXILIARY_TARGET_COLUMNS = (
    "actual_forward_return_5d",
    "actual_future_volatility",
    "actual_future_drawdown",
)
BASELINE_COLUMNS = {
    "momentum_120d": "predicted_momentum_120d",
    "risk_adjusted_momentum": "predicted_risk_adjusted_momentum",
}
TABULAR_MODEL_NAMES = (
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)
SEQUENCE_MODEL_NAMES = (
    "dlinear",
    "patchtst",
    "transformer",
    "itransformer",
    "momentum_transformer",
    "multitask_transformer",
    "market_context_encoder",
    "news_analysis_transformer",
    "temporal_fusion_transformer",
)
MODEL_NAMES = (*TABULAR_MODEL_NAMES, *SEQUENCE_MODEL_NAMES)
PREDICTION_PREFIX = "stock_level_predicted_forward_return_10d_"
ALL_FEATURE_COLUMNS = (*FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)
_MODEL_WORKER_CONTEXT: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class StockLevelModelRankingBenchmarkPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class ModelRunSpec:
    name: str
    kind: str
    factory: Callable[[], Any]
    feature_columns: tuple[str, ...]


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
        from core.research.ml.stock_level_sequence_regressors import (
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


def write_stock_level_model_ranking_benchmark(
    config: dict[str, Any],
) -> StockLevelModelRankingBenchmarkPaths:
    """Train the isolated stock-level benchmark and write research artifacts."""
    settings = StockLevelResearchConfig.from_mapping(config)
    output_dir = settings.output_dir
    source_path = settings.artifact_path
    if not source_path.exists():
        raise FileNotFoundError(f"Stock-level prediction artifact not found: {source_path}")

    logger = ResearchStageLogger("stock_level_alpha_benchmark")
    with logger.stage("loading"):
        rows = CsvRowRepository().read(source_path)
    feature_columns = _available_feature_columns(
        rows,
        include_engineered=settings.include_engineered_features,
    )
    with logger.stage("training_and_evaluation"):
        predictions, payload = build_stock_level_model_ranking_benchmark(
            rows,
            feature_columns=feature_columns,
            source_path=str(source_path),
            min_train_dates=settings.min_train_dates,
            test_window_dates=settings.test_window_dates,
            embargo_dates=settings.embargo_dates,
            random_seed=settings.random_seed,
            sklearn_n_jobs=settings.sklearn_n_jobs,
            model_n_jobs=settings.model_n_jobs,
            include_sequence_models=settings.include_sequence_models,
            sequence_length=settings.sequence_length,
            sequence_epochs=settings.sequence_epochs,
            sequence_batch_size=settings.sequence_batch_size,
            sequence_device=settings.sequence_device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelModelRankingBenchmarkPaths(
        csv_path=output_dir / "stock_level_model_ranking_benchmark.csv",
        json_path=output_dir / "stock_level_model_ranking_benchmark.json",
        markdown_path=output_dir / "stock_level_model_ranking_benchmark.md",
        predictions_path=output_dir / "stock_level_model_oos_predictions.csv",
    )
    with logger.stage("report_generation"):
        writer = ResearchArtifactWriter()
        writer.write_csv(
            paths.csv_path,
            payload["leaderboard"],
            fieldnames=_leaderboard_columns(),
        )
        writer.write_csv(
            paths.predictions_path,
            predictions,
            fieldnames=_prediction_columns(payload["completed_models"]),
        )
        writer.write_json(paths.json_path, payload)
        writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_level_model_ranking_benchmark(
    rows: list[dict[str, Any]],
    *,
    source_path: str | None = None,
    min_train_dates: int = 52,
    test_window_dates: int = 13,
    embargo_dates: int = 2,
    random_seed: int = 42,
    sklearn_n_jobs: int = 1,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    include_sequence_models: bool = True,
    sequence_length: int = 13,
    sequence_epochs: int = 5,
    sequence_batch_size: int = 256,
    sequence_device: str = "cpu",
    sequence_model_factories: dict[str, Callable[[], Any]] | None = None,
    model_n_jobs: int = 1,
    executor_cls: type | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create expanding-window predictions and an OOS ranking leaderboard."""
    _validate_split_settings(min_train_dates, test_window_dates, embargo_dates)
    if model_n_jobs < 1:
        raise ValueError("stock_ranker_model_n_jobs must be at least one")
    prepared_rows, excluded_row_count = _prepare_rows(rows, feature_columns)
    _validate_unique_keys(prepared_rows)
    dates = sorted({row["rebalance_date"] for row in prepared_rows})
    first_test_index = min_train_dates + embargo_dates
    if len(dates) <= first_test_index:
        raise ValueError(
            "Not enough rebalance dates for the requested walk-forward split: "
            f"found {len(dates)}, need more than {first_test_index}"
        )

    effective_sklearn_n_jobs = 1 if model_n_jobs > 1 else sklearn_n_jobs
    effective_torch_num_threads = 1 if model_n_jobs > 1 else None
    tabular_factories = model_factories or _model_factories(
        random_seed, effective_sklearn_n_jobs
    )
    sequence_factories = (
        sequence_model_factories
        if sequence_model_factories is not None
        else _sequence_model_factories(
            sequence_length=sequence_length,
            epochs=sequence_epochs,
            batch_size=sequence_batch_size,
            random_seed=random_seed,
            device=sequence_device,
            torch_num_threads=effective_torch_num_threads,
        )
        if include_sequence_models
        else {}
    )
    if not tabular_factories and not sequence_factories:
        raise ValueError("At least one stock-level model is required")

    news_columns = tuple(
        name
        for name in (rows[0] if rows else {})
        if name.startswith("news_") or "sentiment" in name.lower()
    )
    unavailable_models: list[dict[str, str]] = []
    if "news_analysis_transformer" in sequence_factories and not news_columns:
        sequence_factories = dict(sequence_factories)
        sequence_factories.pop("news_analysis_transformer")
        unavailable_models.append(
            {
                "name": "news_analysis_transformer",
                "status": "unavailable",
                "reason": (
                    "The stock-level input contains no point-in-time symbol-level "
                    "news or sentiment features; synthetic news inputs are forbidden."
                ),
            }
        )

    specs = [
        ModelRunSpec(name, "tabular", factory, feature_columns)
        for name, factory in tabular_factories.items()
    ]
    specs.extend(
        ModelRunSpec(
            name,
            "sequence",
            factory,
            _sequence_feature_columns(name, news_columns, feature_columns),
        )
        for name, factory in sequence_factories.items()
    )
    model_results, model_errors = _execute_model_runs(
        specs,
        prepared_rows=prepared_rows,
        dates=dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
        sequence_length=sequence_length,
        model_n_jobs=model_n_jobs,
        executor_cls=executor_cls or ProcessPoolExecutor,
    )
    unavailable_models.extend(
        {
            "name": spec.name,
            "status": "error",
            "reason": model_errors[spec.name],
        }
        for spec in specs
        if spec.name in model_errors
    )
    folds, predictions = _build_oos_prediction_rows(
        prepared_rows,
        dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
    )
    for model_name, values_by_key in model_results.items():
        column = f"{PREDICTION_PREFIX}{model_name}"
        for row in predictions:
            row[column] = values_by_key[(row["rebalance_date"], row["symbol"])]

    _validate_unique_keys(predictions)
    completed_models = tuple(spec.name for spec in specs if spec.name in model_results)
    leaderboard = _build_leaderboard(predictions, tuple(completed_models))
    full_period_baselines = [
        _evaluate_signal(prepared_rows, name, column, kind="baseline")
        for name, column in BASELINE_COLUMNS.items()
    ]
    best_ml = next((row for row in leaderboard if row["kind"] == "ml_model"), None)
    momentum = next(
        row for row in leaderboard if row["name"] == "momentum_120d"
    )
    comparison = _compare_to_momentum(best_ml, momentum)
    payload = {
        "mode": "stock_level_model_ranking_benchmark_research_only",
        "purpose": (
            "Benchmark simple stock-level return rankers using chronological, "
            "expanding-window out-of-sample predictions."
        ),
        "source_path": source_path,
        "target_column": TARGET_COLUMN,
        "feature_columns": list(feature_columns),
        "requested_models": list(MODEL_NAMES),
        "completed_models": list(completed_models),
        "unavailable_models": unavailable_models,
        "baseline_columns": BASELINE_COLUMNS,
        "input_row_count": len(rows),
        "eligible_row_count": len(prepared_rows),
        "excluded_incomplete_row_count": excluded_row_count,
        "input_date_count": len(dates),
        "input_symbol_count": len({row["symbol"] for row in prepared_rows}),
        "oos_row_count": len(predictions),
        "oos_date_count": len({row["rebalance_date"] for row in predictions}),
        "oos_symbol_count": len({row["symbol"] for row in predictions}),
        "prediction_columns": [
            f"{PREDICTION_PREFIX}{name}" for name in completed_models
        ],
        "parallelism": {
            "strategy": "independent_models",
            "stock_ranker_model_n_jobs": model_n_jobs,
            "effective_model_workers": min(model_n_jobs, len(specs)),
            "requested_sklearn_n_jobs": sklearn_n_jobs,
            "effective_per_model_sklearn_n_jobs": effective_sklearn_n_jobs,
            "effective_per_model_torch_num_threads": effective_torch_num_threads,
            "effective_per_model_native_thread_limit": (
                1 if model_n_jobs > 1 else None
            ),
            "folds_parallelized": False,
            "dates_parallelized": False,
        },
        "walk_forward": {
            "method": "chronological_expanding_window",
            "min_train_dates": min_train_dates,
            "test_window_dates": test_window_dates,
            "embargo_rebalance_dates": embargo_dates,
            "sequence_length": sequence_length,
            "out_of_sample_only": True,
            "all_chronological_guards_passed": all(
                fold["chronological_guard_passed"] for fold in folds
            ),
            "folds": folds,
        },
        "ranking_rule": (
            "Descending mean Spearman IC, then descending top-minus-bottom spread."
        ),
        "ml_beats_momentum_rule": (
            "Best ML model must exceed the OOS-aligned momentum_120d baseline on "
            "both mean Spearman IC and top-minus-bottom spread."
        ),
        "leaderboard": leaderboard,
        "full_period_baselines": full_period_baselines,
        "best_ml_model": best_ml,
        "best_ml_vs_momentum_120d": comparison,
        "ml_beats_momentum_120d": comparison["beats_momentum_120d"],
        **RESEARCH_METADATA,
    }
    return predictions, payload


def _execute_model_runs(
    specs: list[ModelRunSpec],
    *,
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    sequence_length: int,
    model_n_jobs: int,
    executor_cls: type,
) -> tuple[
    dict[str, dict[tuple[str, str], float]],
    dict[str, str],
]:
    arguments = (
        prepared_rows,
        dates,
        first_test_index,
        test_window_dates,
        embargo_dates,
        sequence_length,
        1 if model_n_jobs > 1 else None,
    )
    if model_n_jobs == 1:
        worker = lambda spec: _run_model_walk_forward(spec, *arguments)
        initializer = None
        initargs = ()
    else:
        worker = _run_initialized_model
        initializer = _initialize_model_worker
        initargs = (arguments,)
    execution = ParallelTaskExecutor[ModelRunSpec, dict[tuple[str, str], float]]()
    result = execution.execute(
        specs,
        worker,
        key=lambda spec: spec.name,
        max_workers=model_n_jobs,
        executor_cls=executor_cls,
        initializer=initializer,
        initargs=initargs,
    )
    return result.results, result.errors


def _initialize_model_worker(arguments: tuple[Any, ...]) -> None:
    global _MODEL_WORKER_CONTEXT
    _MODEL_WORKER_CONTEXT = arguments


def _run_initialized_model(
    spec: ModelRunSpec,
) -> dict[tuple[str, str], float]:
    if _MODEL_WORKER_CONTEXT is None:
        raise RuntimeError("Model worker context was not initialized")
    return _run_model_walk_forward(spec, *_MODEL_WORKER_CONTEXT)


def _run_model_walk_forward(
    spec: ModelRunSpec,
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    sequence_length: int,
    native_thread_limit: int | None,
) -> dict[tuple[str, str], float]:
    thread_context: Any = nullcontext()
    if native_thread_limit is not None:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # no native sklearn/BLAS pools in this environment
            pass
        else:
            thread_context = threadpool_limits(limits=native_thread_limit)
    with thread_context:
        return _run_model_walk_forward_unlimited(
            spec,
            prepared_rows,
            dates,
            first_test_index,
            test_window_dates,
            embargo_dates,
            sequence_length,
        )


def _run_model_walk_forward_unlimited(
    spec: ModelRunSpec,
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    sequence_length: int,
) -> dict[tuple[str, str], float]:
    predictions: dict[tuple[str, str], float] = {}
    for _, train_rows, test_rows, _, _, _ in _walk_forward_partitions(
        prepared_rows,
        dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
    ):
        model = spec.factory()
        if spec.kind == "tabular":
            x_train = [
                [row[column] for column in spec.feature_columns]
                for row in train_rows
            ]
            x_test = [
                [row[column] for column in spec.feature_columns]
                for row in test_rows
            ]
            model.fit(x_train, [row[TARGET_COLUMN] for row in train_rows])
            values = [float(value) for value in model.predict(x_test)]
            prediction_rows = test_rows
        else:
            train_sequences, sequence_train_rows = _build_sequences(
                prepared_rows,
                train_rows,
                spec.feature_columns,
                sequence_length,
            )
            test_sequences, prediction_rows = _build_sequences(
                prepared_rows,
                test_rows,
                spec.feature_columns,
                sequence_length,
            )
            if len(prediction_rows) != len(test_rows):
                raise ValueError(
                    f"{spec.name} cannot produce one prediction per OOS row; "
                    "increase min_train_dates or reduce sequence_length"
                )
            auxiliary_targets = (
                [
                    [row[column] for column in AUXILIARY_TARGET_COLUMNS]
                    for row in sequence_train_rows
                ]
                if spec.name == "multitask_transformer"
                else None
            )
            model.fit(
                train_sequences,
                [row[TARGET_COLUMN] for row in sequence_train_rows],
                auxiliary_targets,
            )
            values = [float(value) for value in model.predict(test_sequences)]
        if len(values) != len(prediction_rows):
            raise ValueError(
                f"{spec.name} returned {len(values)} predictions for "
                f"{len(prediction_rows)} OOS rows"
            )
        predictions.update(
            {
                (row["rebalance_date"], row["symbol"]): value
                for row, value in zip(prediction_rows, values)
            }
        )
    return predictions


def _build_oos_prediction_rows(
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    *,
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_id, train_rows, test_rows, train_dates, test_dates, embargoed_dates in (
        _walk_forward_partitions(
            prepared_rows,
            dates,
            first_test_index=first_test_index,
            test_window_dates=test_window_dates,
            embargo_dates=embargo_dates,
        )
    ):
        predictions.extend(_base_prediction_row(row, fold_id) for row in test_rows)
        folds.append(
            {
                "fold_id": fold_id,
                "train_start_date": train_dates[0],
                "train_end_date": train_dates[-1],
                "train_date_count": len(train_dates),
                "train_row_count": len(train_rows),
                "embargoed_dates": embargoed_dates,
                "test_start_date": test_dates[0],
                "test_end_date": test_dates[-1],
                "test_date_count": len(test_dates),
                "test_row_count": len(test_rows),
                "chronological_guard_passed": train_dates[-1] < test_dates[0],
            }
        )
    predictions.sort(key=lambda row: (row["rebalance_date"], row["symbol"]))
    return folds, predictions


def _walk_forward_partitions(
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    *,
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
):
    splitter = ExpandingWindowSplitter(
        ExpandingWindowSpec(
            min_train_dates=first_test_index - embargo_dates,
            test_window_dates=test_window_dates,
            embargo_dates=embargo_dates,
        )
    )
    for fold in splitter.split(prepared_rows, dates=dates):
        yield (
            fold.fold_id,
            list(fold.train_rows),
            list(fold.test_rows),
            list(fold.train_dates),
            list(fold.test_dates),
            list(fold.embargoed_dates),
        )


def _prepare_rows(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[list[dict[str, Any]], int]:
    required = ("rebalance_date", "symbol", TARGET_COLUMN)
    prepared = []
    for source in rows:
        date = str(source.get("rebalance_date", "")).strip()
        symbol = str(source.get("symbol", "")).strip().upper()
        numbers = {column: _number(source.get(column)) for column in required[2:]}
        if not date or not symbol or any(value is None for value in numbers.values()):
            continue
        optional_columns = {
            *CONTEXT_COLUMNS,
            *AUXILIARY_TARGET_COLUMNS,
            *(
                name
                for name in source
                if name.startswith("news_") or "sentiment" in name.lower()
            ),
        }
        prepared.append({
            "rebalance_date": date,
            "symbol": symbol,
            **{column: float(value) for column, value in numbers.items()},
            **{
                column: (
                    float(value)
                    if (value := _number(source.get(column))) is not None
                    else math.nan
                )
                for column in feature_columns
            },
            **{
                column: _number(source.get(column)) or 0.0
                for column in optional_columns
            },
        })
    return prepared, len(rows) - len(prepared)


def _available_feature_columns(
    rows: list[dict[str, Any]],
    *,
    include_engineered: bool,
) -> tuple[str, ...]:
    if not include_engineered:
        return FEATURE_COLUMNS
    available_engineered = tuple(
        column
        for column in ENGINEERED_FEATURE_COLUMNS
        if any(_number(row.get(column)) is not None for row in rows)
    )
    return (*FEATURE_COLUMNS, *available_engineered)


def _base_prediction_row(row: dict[str, Any], fold_id: int) -> dict[str, Any]:
    return {
        "rebalance_date": row["rebalance_date"],
        "symbol": row["symbol"],
        "fold_id": fold_id,
        TARGET_COLUMN: row[TARGET_COLUMN],
        "actual_future_volatility": row.get("actual_future_volatility"),
        "actual_future_drawdown": row.get("actual_future_drawdown"),
        "predicted_momentum_120d": row["predicted_momentum_120d"],
        "predicted_risk_adjusted_momentum": row[
            "predicted_risk_adjusted_momentum"
        ],
    }


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


def _build_sequences(
    all_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    sequence_length: int,
) -> tuple[list[list[list[float]]], list[dict[str, Any]]]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    target_keys = {
        (row["rebalance_date"], row["symbol"]): row for row in target_rows
    }
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_symbol.setdefault(row["symbol"], []).append(row)
    keyed_sequences: list[tuple[tuple[str, str], list[list[float]], dict[str, Any]]] = []
    for symbol, symbol_rows in by_symbol.items():
        ordered = sorted(symbol_rows, key=lambda row: row["rebalance_date"])
        for end_index in range(sequence_length - 1, len(ordered)):
            end_row = ordered[end_index]
            key = (end_row["rebalance_date"], symbol)
            target_row = target_keys.get(key)
            if target_row is None:
                continue
            window = ordered[end_index - sequence_length + 1 : end_index + 1]
            keyed_sequences.append(
                (
                    key,
                    [
                        [float(row.get(column, 0.0)) for column in feature_columns]
                        for row in window
                    ],
                    target_row,
                )
            )
    keyed_sequences.sort(key=lambda item: item[0])
    return (
        [item[1] for item in keyed_sequences],
        [item[2] for item in keyed_sequences],
    )


def _build_leaderboard(
    predictions: list[dict[str, Any]],
    model_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    summaries = [
        _evaluate_signal(
            predictions,
            name,
            f"{PREDICTION_PREFIX}{name}",
            kind="ml_model",
        )
        for name in model_names
    ]
    summaries.extend(
        _evaluate_signal(predictions, name, column, kind="baseline")
        for name, column in BASELINE_COLUMNS.items()
    )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -_sort_value(row["mean_spearman_ic"]),
            -_sort_value(row["top_minus_bottom_spread"]),
            row["name"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _evaluate_signal(
    rows: list[dict[str, Any]],
    name: str,
    signal_column: str,
    *,
    kind: str,
) -> dict[str, Any]:
    return CrossSectionalRankingEvaluator(target_column=TARGET_COLUMN).evaluate(
        rows,
        name=name,
        signal_column=signal_column,
        kind=kind,
    )


def _compare_to_momentum(
    best_ml: dict[str, Any] | None,
    momentum: dict[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "mean_pearson_ic",
        "mean_spearman_ic",
        "top_decile_return",
        "bottom_decile_return",
        "top_minus_bottom_spread",
        "top_decile_hit_rate",
        "risk_adjusted_spread",
        "spread_sharpe",
    )
    deltas = {
        name: (
            None
            if best_ml is None
            or best_ml.get(name) is None
            or momentum.get(name) is None
            else float(best_ml[name]) - float(momentum[name])
        )
        for name in metric_names
    }
    beats = bool(
        best_ml
        and deltas["mean_spearman_ic"] is not None
        and deltas["top_minus_bottom_spread"] is not None
        and deltas["mean_spearman_ic"] > 0.0
        and deltas["top_minus_bottom_spread"] > 0.0
    )
    return {
        "model": best_ml["name"] if best_ml else None,
        "momentum_baseline": momentum["name"],
        "metric_deltas_ml_minus_momentum": deltas,
        "beats_momentum_120d": beats,
    }


def _validate_split_settings(
    min_train_dates: int,
    test_window_dates: int,
    embargo_dates: int,
) -> None:
    if min_train_dates < 1:
        raise ValueError("min_train_dates must be at least one")
    if test_window_dates < 1:
        raise ValueError("test_window_dates must be at least one")
    if embargo_dates < 0:
        raise ValueError("embargo_dates cannot be negative")


def _validate_unique_keys(rows: list[dict[str, Any]]) -> None:
    keys = [(str(row["rebalance_date"]), str(row["symbol"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Stock-level rows must be unique by rebalance_date and symbol")


def _average(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sort_value(value: Any) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else -math.inf


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)


def _leaderboard_columns() -> list[str]:
    return [
        "rank",
        "name",
        "kind",
        "signal_column",
        "mean_pearson_ic",
        "mean_spearman_ic",
        "top_decile_return",
        "bottom_decile_return",
        "top_minus_bottom_spread",
        "top_decile_hit_rate",
        "risk_adjusted_spread",
        "spread_sharpe",
        "date_count",
        "row_count",
    ]


def _prediction_columns(model_names: list[str]) -> list[str]:
    return [
        "rebalance_date",
        "symbol",
        "fold_id",
        TARGET_COLUMN,
        "actual_future_volatility",
        "actual_future_drawdown",
        "predicted_momentum_120d",
        "predicted_risk_adjusted_momentum",
        *(f"{PREDICTION_PREFIX}{name}" for name in model_names),
    ]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)


def _markdown(payload: dict[str, Any]) -> str:
    comparison = payload["best_ml_vs_momentum_120d"]
    lines = [
        "# Stock-Level Alpha Benchmark Suite",
        "",
        NOTICE,
        "",
        f"- Target: `{payload['target_column']}`",
        f"- Eligible input rows: {payload['eligible_row_count']}",
        f"- OOS rows: {payload['oos_row_count']}",
        f"- OOS dates: {payload['oos_date_count']}",
        f"- Completed models: {len(payload['completed_models'])}",
        f"- Unavailable models: {len(payload['unavailable_models'])}",
        "- Split: chronological expanding window with "
        f"{payload['walk_forward']['embargo_rebalance_dates']} embargoed rebalance dates",
        "- Promotion thresholds changed: false",
        "",
        "## OOS Leaderboard",
        "",
        "| Rank | Model / baseline | Kind | Dates | Pearson IC | Spearman IC | Top decile | Bottom decile | Spread | Sharpe | Top hit rate | Risk-adjusted spread |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            "| {rank} | {name} | {kind} | {dates} | {pearson} | {ic} | {top} | {bottom} | {spread} | {sharpe} | {hit} | {risk} |".format(
                rank=row["rank"],
                name=row["name"],
                kind=row["kind"],
                dates=row["date_count"],
                pearson=_fmt(row["mean_pearson_ic"]),
                ic=_fmt(row["mean_spearman_ic"]),
                top=_fmt(row["top_decile_return"]),
                bottom=_fmt(row["bottom_decile_return"]),
                spread=_fmt(row["top_minus_bottom_spread"]),
                sharpe=_fmt(row["spread_sharpe"]),
                hit=_fmt(row["top_decile_hit_rate"]),
                risk=_fmt(row["risk_adjusted_spread"]),
            )
        )
    lines.extend(
        [
            "",
            "## Best ML vs Momentum 120d",
            "",
            f"- Best ML model: {comparison['model']}",
            f"- Beats OOS-aligned momentum_120d: {comparison['beats_momentum_120d']}",
            "- Decision rule: higher mean Spearman IC and higher top-minus-bottom spread.",
            "- Spearman IC delta: "
            f"{_fmt(comparison['metric_deltas_ml_minus_momentum']['mean_spearman_ic'])}",
            "- Spread delta: "
            f"{_fmt(comparison['metric_deltas_ml_minus_momentum']['top_minus_bottom_spread'])}",
            "",
            "## Full-Period Baseline Reference",
            "",
            "| Baseline | Dates | Spearman IC | Spread | Top hit rate | Risk-adjusted spread |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["full_period_baselines"]:
        lines.append(
            "| {name} | {dates} | {ic} | {spread} | {hit} | {risk} |".format(
                name=row["name"],
                dates=row["date_count"],
                ic=_fmt(row["mean_spearman_ic"]),
                spread=_fmt(row["top_minus_bottom_spread"]),
                hit=_fmt(row["top_decile_hit_rate"]),
                risk=_fmt(row["risk_adjusted_spread"]),
            )
        )
    if payload["unavailable_models"]:
        lines.extend(["", "## Unavailable Models", ""])
        for row in payload["unavailable_models"]:
            lines.append(f"- {row['name']}: {row['reason']}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
