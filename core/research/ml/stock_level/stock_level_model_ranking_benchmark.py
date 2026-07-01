from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

from core.research.ml.stock_level_benchmark_types import (
    BASELINE_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    ModelRunSpec,
    NOTICE,
    PREDICTION_PREFIX,
    RESEARCH_METADATA,
    SEQUENCE_MODEL_NAMES,
    TABULAR_MODEL_NAMES,
    TARGET_COLUMN,
    StockLevelModelRankingBenchmarkPaths,
)
from core.research.ml.stock_level_benchmark_models import (
    SequenceModelFactory,
    TabularModelFactory,
    _build_tabular_model,
    _model_factories,
    _sequence_feature_columns,
    _sequence_model_factories,
    stock_ranker_model_registry,
)
from core.research.ml.stock_level_benchmark_execution import (
    _MODEL_WORKER_CONTEXT,
    _build_sequences,
    _execute_model_runs,
    _initialize_model_worker,
    _run_initialized_model,
    _run_model_walk_forward,
    _run_model_walk_forward_unlimited,
    _walk_forward_partitions,
)
from core.research.ml.stock_level_benchmark_data import (
    _available_feature_columns,
    _average,
    _base_prediction_row,
    _build_oos_prediction_rows,
    _number,
    _prepare_rows,
    _validate_split_settings,
    _validate_unique_keys,
)
from core.research.ml.stock_level_benchmark_evaluation import (
    _build_leaderboard,
    _compare_to_momentum,
    _evaluate_signal,
)
from core.research.ml.stock_level_benchmark_reporting import (
    _fmt,
    _leaderboard_columns,
    _markdown,
    _output_dir,
    _prediction_columns,
    _read_csv,
    _write_csv,
)
from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.stock_alpha_news_contract import validate_news_contract
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_model_sets import FULL_SEQUENCE_MODELS, StockAlphaModelSet, resolve_stock_alpha_model_set
from datetime import datetime, timezone
import time


def write_stock_level_model_ranking_benchmark(
    config: dict[str, Any],
) -> StockLevelModelRankingBenchmarkPaths:
    """Train the isolated stock-level benchmark and write research artifacts."""
    settings = StockLevelResearchConfig.from_mapping(config)
    thread_caps = apply_stock_alpha_worker_caps(config)
    started_at = datetime.now(timezone.utc).isoformat(); started = time.perf_counter()
    output_dir = settings.output_dir
    source_path = settings.artifact_path
    if not source_path.exists():
        raise FileNotFoundError(f"Stock-level prediction artifact not found: {source_path}")

    logger = ResearchStageLogger("stock_level_alpha_benchmark")
    with logger.stage("loading"):
        rows = CsvRowRepository().read(source_path)
        rows, run_profile = apply_stock_alpha_run_profile(rows, settings)
    feature_columns = _available_feature_columns(
        rows,
        include_engineered=settings.include_engineered_features,
    )
    with logger.stage("training_and_evaluation"):
        model_set = resolve_stock_alpha_model_set(settings.ranker_model_set, include_sequence_models=settings.include_sequence_models)
        tabular_factories, sequence_factories = _factories_for_model_set(settings, model_set, sklearn_n_jobs=settings.sklearn_n_jobs, torch_num_threads=thread_caps["torch_num_threads"])
        predictions, payload = build_stock_level_model_ranking_benchmark(
            rows,
            target_column=settings.target_column,
            feature_columns=feature_columns,
            source_path=str(source_path),
            config_path=str(config.get("config_path", "config/config.yaml")),
            min_train_dates=settings.min_train_dates,
            test_window_dates=settings.test_window_dates,
            embargo_dates=settings.embargo_dates,
            random_seed=settings.random_seed,
            sklearn_n_jobs=settings.sklearn_n_jobs,
            model_n_jobs=settings.model_n_jobs,
            include_sequence_models=settings.include_sequence_models,
            model_factories={name: factory for name, factory in tabular_factories.items() if name in model_set.included_models},
            sequence_model_factories={name: factory for name, factory in sequence_factories.items() if name in model_set.included_models},
            sequence_length=settings.sequence_length,
            sequence_epochs=settings.sequence_epochs,
            sequence_batch_size=settings.sequence_batch_size,
            sequence_device=settings.sequence_device,
            news_contract_available=validate_news_contract(config, rows).available,
        )
        payload.update(run_profile)
        payload.update(model_set.metadata())
        payload["stock_ranker_model_set"] = settings.ranker_model_set
        payload["requested_models"] = list(model_set.included_models)
        payload.update(stock_alpha_report_metadata(config, output_dir, source_artifact_path=source_path))
        payload.update({"started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "thread_caps": thread_caps})

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


def _factories_for_model_set(
    settings: StockLevelResearchConfig,
    model_set: StockAlphaModelSet,
    *,
    sklearn_n_jobs: int,
    torch_num_threads: int,
) -> tuple[dict[str, Callable[[], Any]], dict[str, Callable[[], Any]]]:
    tabular = {name: factory for name, factory in _model_factories(settings.random_seed, sklearn_n_jobs).items() if name in model_set.included_models}
    if not any(name in FULL_SEQUENCE_MODELS for name in model_set.included_models):
        return tabular, {}
    sequence = _sequence_model_factories(
        sequence_length=settings.sequence_length,
        epochs=settings.sequence_epochs,
        batch_size=settings.sequence_batch_size,
        random_seed=settings.random_seed,
        device=settings.sequence_device,
        torch_num_threads=torch_num_threads,
    )
    return tabular, {name: factory for name, factory in sequence.items() if name in model_set.included_models}


def build_stock_level_model_ranking_benchmark(
    rows: list[dict[str, Any]],
    *,
    source_path: str | None = None,
    config_path: str | None = None,
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
    target_column: str = TARGET_COLUMN,
    news_contract_available: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create expanding-window predictions and an OOS ranking leaderboard."""
    _validate_split_settings(min_train_dates, test_window_dates, embargo_dates)
    if model_n_jobs < 1:
        raise ValueError("stock_ranker_model_n_jobs must be at least one")
    if target_column != TARGET_COLUMN:
        rows = [dict(row, **{TARGET_COLUMN: row.get(target_column, "")}) for row in rows]
    prepared_rows, excluded_row_count = _prepare_rows(rows, feature_columns)
    _validate_unique_keys(prepared_rows)
    dates = sorted({row["rebalance_date"] for row in prepared_rows})
    first_test_index = min_train_dates + embargo_dates
    if len(dates) <= first_test_index:
        raise ValueError(
            "Not enough rebalance dates for the requested walk-forward split: "
            f"found {len(dates)}, need more than {first_test_index}. "
            f"available_rebalance_dates={len(dates)}; "
            f"required_first_test_index={first_test_index}; "
            f"min_train_dates={min_train_dates}; "
            f"test_window_dates={test_window_dates}; "
            f"embargo_dates={embargo_dates}; "
            f"active_config_path={config_path or 'unknown'}; "
            f"source_path={source_path or 'unknown'}. "
            "For dev runs, reduce ml.stock_ranker_min_train_dates, "
            "ml.stock_ranker_test_window_dates, or ml.stock_ranker_embargo_dates, "
            "or use benchmark/full data."
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
    if "news_analysis_transformer" in sequence_factories and (not news_columns or not news_contract_available):
        sequence_factories = dict(sequence_factories)
        sequence_factories.pop("news_analysis_transformer")
        reason = (
            "news_analysis_transformer unavailable: missing valid point-in-time news contract"
            if news_columns
            else (
                "The stock-level input contains no point-in-time symbol-level "
                "news or sentiment features; synthetic news inputs are forbidden."
            )
        )
        unavailable_models.append(
            {
                "name": "news_analysis_transformer",
                "status": "unavailable",
                "reason": reason,
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
    model_results, model_errors, model_timings = _execute_model_runs(
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
        "target_column": target_column,
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
            "requested_workers": model_n_jobs,
            "effective_workers": min(model_n_jobs, len(specs)),
            "nested_sklearn_n_jobs": effective_sklearn_n_jobs,
            "nested_torch_num_threads": effective_torch_num_threads,
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
        "model_timings": model_timings,
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
