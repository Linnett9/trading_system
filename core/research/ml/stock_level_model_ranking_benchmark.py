from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from core.research.ml.stock_level_alpha_features import ENGINEERED_FEATURE_COLUMNS


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
    ml_config = config.get("ml", {})
    output_dir = _output_dir(config)
    source_path = Path(
        ml_config.get(
            "stock_level_prediction_artifacts_path",
            output_dir / "stock_level_prediction_artifacts.csv",
        )
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Stock-level prediction artifact not found: {source_path}")

    rows = _read_csv(source_path)
    feature_columns = _available_feature_columns(
        rows,
        include_engineered=bool(
            ml_config.get("stock_ranker_include_engineered_features", False)
        ),
    )
    predictions, payload = build_stock_level_model_ranking_benchmark(
        rows,
        feature_columns=feature_columns,
        source_path=str(source_path),
        min_train_dates=int(ml_config.get("stock_ranker_min_train_dates", 52)),
        test_window_dates=int(ml_config.get("stock_ranker_test_window_dates", 13)),
        embargo_dates=int(ml_config.get("stock_ranker_embargo_dates", 2)),
        random_seed=int(ml_config.get("random_seed", 42)),
        sklearn_n_jobs=int(ml_config.get("sklearn_n_jobs", 1)),
        model_n_jobs=int(ml_config.get("stock_ranker_model_n_jobs", 1)),
        include_sequence_models=bool(
            ml_config.get("stock_ranker_include_sequence_models", True)
        ),
        sequence_length=int(ml_config.get("stock_ranker_sequence_length", 13)),
        sequence_epochs=int(ml_config.get("stock_ranker_sequence_epochs", 5)),
        sequence_batch_size=int(
            ml_config.get("stock_ranker_sequence_batch_size", 256)
        ),
        sequence_device=str(ml_config.get("stock_ranker_sequence_device", "cpu")),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelModelRankingBenchmarkPaths(
        csv_path=output_dir / "stock_level_model_ranking_benchmark.csv",
        json_path=output_dir / "stock_level_model_ranking_benchmark.json",
        markdown_path=output_dir / "stock_level_model_ranking_benchmark.md",
        predictions_path=output_dir / "stock_level_model_oos_predictions.csv",
    )
    _write_csv(paths.csv_path, payload["leaderboard"], _leaderboard_columns())
    _write_csv(
        paths.predictions_path,
        predictions,
        _prediction_columns(payload["completed_models"]),
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
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
    results: dict[str, dict[tuple[str, str], float]] = {}
    errors: dict[str, str] = {}
    if model_n_jobs == 1:
        for spec in specs:
            try:
                results[spec.name] = _run_model_walk_forward(spec, *arguments)
            except Exception as exc:  # isolated research model boundary
                errors[spec.name] = f"{type(exc).__name__}: {exc}"
        return results, errors

    with executor_cls(
        max_workers=min(model_n_jobs, len(specs)),
        initializer=_initialize_model_worker,
        initargs=(arguments,),
    ) as executor:
        futures = {
            executor.submit(_run_initialized_model, spec): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results[spec.name] = future.result()
            except Exception as exc:  # isolated process boundary
                errors[spec.name] = f"{type(exc).__name__}: {exc}"
    return results, errors


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
    for fold_id, test_start in enumerate(
        range(first_test_index, len(dates), test_window_dates),
        start=1,
    ):
        test_dates = dates[test_start : test_start + test_window_dates]
        train_dates = dates[: test_start - embargo_dates]
        embargoed_dates = dates[test_start - embargo_dates : test_start]
        train_date_set = set(train_dates)
        test_date_set = set(test_dates)
        train_rows = [
            row for row in prepared_rows if row["rebalance_date"] in train_date_set
        ]
        test_rows = [
            row for row in prepared_rows if row["rebalance_date"] in test_date_set
        ]
        yield (
            fold_id,
            train_rows,
            test_rows,
            train_dates,
            test_dates,
            embargoed_dates,
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
    return {
        name: TabularModelFactory(name, random_seed, sklearn_n_jobs)
        for name in TABULAR_MODEL_NAMES
    }


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
    by_date: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        signal = _number(row.get(signal_column))
        target = _number(row.get(TARGET_COLUMN))
        if signal is None or target is None:
            continue
        risk = max(
            abs(_number(row.get("actual_future_drawdown")) or 0.0),
            abs(_number(row.get("actual_future_volatility")) or 0.0),
            1e-6,
        )
        by_date.setdefault(str(row["rebalance_date"]), []).append(
            {"score": signal, "target": target, "risk_target": target / risk}
        )
    date_metrics = [
        _date_metrics(group) for _, group in sorted(by_date.items()) if len(group) >= 2
    ]
    return {
        "rank": None,
        "name": name,
        "kind": kind,
        "signal_column": signal_column,
        "mean_pearson_ic": _average([row["pearson"] for row in date_metrics]),
        "mean_spearman_ic": _average([row["spearman"] for row in date_metrics]),
        "top_decile_return": _average([row["top_decile_return"] for row in date_metrics]),
        "bottom_decile_return": _average(
            [row["bottom_decile_return"] for row in date_metrics]
        ),
        "top_minus_bottom_spread": _average(
            [row["top_minus_bottom_spread"] for row in date_metrics]
        ),
        "top_decile_hit_rate": _average(
            [row["top_decile_hit_rate"] for row in date_metrics]
        ),
        "risk_adjusted_spread": _average(
            [row["risk_adjusted_spread"] for row in date_metrics]
        ),
        "spread_sharpe": _annualized_sharpe(
            [row["top_minus_bottom_spread"] for row in date_metrics]
        ),
        "date_count": len(date_metrics),
        "row_count": sum(row["row_count"] for row in date_metrics),
    }


def _date_metrics(rows: list[dict[str, float]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
    bucket_size = max(1, math.ceil(len(ordered) * 0.10))
    top = ordered[:bucket_size]
    bottom = ordered[-bucket_size:]
    top_return = mean(row["target"] for row in top)
    bottom_return = mean(row["target"] for row in bottom)
    return {
        "row_count": len(ordered),
        "pearson": _pearson(
            [row["score"] for row in ordered],
            [row["target"] for row in ordered],
        ),
        "spearman": _spearman(
            [row["score"] for row in ordered],
            [row["target"] for row in ordered],
        ),
        "top_decile_return": top_return,
        "bottom_decile_return": bottom_return,
        "top_minus_bottom_spread": top_return - bottom_return,
        "top_decile_hit_rate": mean(
            1.0 if row["target"] > 0.0 else 0.0 for row in top
        ),
        "risk_adjusted_spread": (
            mean(row["risk_target"] for row in top)
            - mean(row["risk_target"] for row in bottom)
        ),
    }


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


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_variance * right_variance)
    return numerator / denominator if denominator > 0.0 else None


def _average(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _annualized_sharpe(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if len(finite) < 2:
        return None
    average = mean(finite)
    variance = sum((value - average) ** 2 for value in finite) / (len(finite) - 1)
    return average / math.sqrt(variance) * math.sqrt(52.0) if variance > 0.0 else None


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
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
