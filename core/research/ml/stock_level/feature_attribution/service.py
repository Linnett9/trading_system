from __future__ import annotations

from typing import Any, Callable

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository, JsonRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.feature_attribution.io import (
    _fmt,
    _markdown,
    _output_dir,
    _read_csv,
    _write_csv,
)
from core.research.ml.stock_level.feature_attribution.math import (
    _average,
    _difference,
    _first_number,
    _flat_numbers,
)
from core.research.ml.stock_level.feature_attribution.model import (
    _TabularAttributionFactory,
    _attribute_model,
    _evaluate_signal,
    _extract_model_attribution,
    _matrix,
    _metrics,
    _normalized_magnitudes,
    _permuted_matrix,
    _prediction_rows,
    _reference_metrics,
)
from core.research.ml.stock_level.feature_attribution.types import (
    METRIC_NAMES,
    NOTICE,
    StockLevelFeatureAttributionPaths,
)
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    FEATURE_COLUMNS,
    PREDICTION_PREFIX,
    RESEARCH_METADATA,
    TABULAR_MODEL_NAMES,
    TARGET_COLUMN,
    _prepare_rows,
    _walk_forward_partitions,
)


def write_stock_level_feature_attribution(
    config: dict[str, Any],
) -> StockLevelFeatureAttributionPaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    output_dir = settings.output_dir
    source_path = settings.artifact_path
    benchmark_path = settings.benchmark_path
    predictions_path = settings.oos_predictions_path
    for path in (source_path, benchmark_path, predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"Required stock-level research artifact not found: {path}")

    logger = ResearchStageLogger("stock_level_feature_attribution")
    with logger.stage("loading"):
        rows = CsvRowRepository().read(source_path)
        benchmark = JsonRepository().read(benchmark_path)
        existing_predictions = CsvRowRepository().read(predictions_path)
    with logger.stage("evaluation"):
        payload = build_stock_level_feature_attribution(
            rows,
            benchmark,
            existing_prediction_rows=existing_predictions,
            source_path=str(source_path),
            benchmark_path=str(benchmark_path),
            predictions_path=str(predictions_path),
            random_seed=settings.random_seed,
            sklearn_n_jobs=settings.sklearn_n_jobs,
            permutation_repeats=(settings.attribution_permutation_repeats if settings.run_size == "dev" else settings.permutation_repeats),
            max_models=settings.attribution_max_models if settings.run_size == "dev" else None,
            max_features=settings.attribution_max_features if settings.run_size == "dev" else None,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelFeatureAttributionPaths(
        csv_path=output_dir / "stock_level_feature_attribution.csv",
        json_path=output_dir / "stock_level_feature_attribution.json",
        markdown_path=output_dir / "stock_level_feature_attribution.md",
    )
    with logger.stage("report_generation"):
        _write_csv(paths.csv_path, payload["feature_rows"])
        writer = ResearchArtifactWriter()
        writer.write_json(paths.json_path, payload)
        writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_level_feature_attribution(
    rows: list[dict[str, Any]],
    benchmark: dict[str, Any],
    *,
    existing_prediction_rows: list[dict[str, Any]] | None = None,
    source_path: str | None = None,
    benchmark_path: str | None = None,
    predictions_path: str | None = None,
    random_seed: int = 42,
    sklearn_n_jobs: int = 1,
    permutation_repeats: int = 3,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    max_models: int | None = None,
    max_features: int | None = None,
) -> dict[str, Any]:
    if permutation_repeats < 0:
        raise ValueError("permutation_repeats cannot be negative")
    split = benchmark.get("walk_forward", {})
    min_train_dates = int(split.get("min_train_dates", 52))
    test_window_dates = int(split.get("test_window_dates", 13))
    embargo_dates = int(split.get("embargo_rebalance_dates", 2))
    prepared_rows, excluded_row_count = _prepare_rows(rows)
    dates = sorted({row["rebalance_date"] for row in prepared_rows})
    first_test_index = min_train_dates + embargo_dates
    if len(dates) <= first_test_index:
        raise ValueError("Not enough dates for benchmark-compatible attribution folds")

    completed = set(benchmark.get("completed_models", []))
    model_names = [name for name in TABULAR_MODEL_NAMES if name in completed]
    if max_models is not None:
        model_names = model_names[:max_models]
    feature_columns = FEATURE_COLUMNS[:max_features] if max_features is not None else FEATURE_COLUMNS
    factories = model_factories or {
        name: _TabularAttributionFactory(name, random_seed, sklearn_n_jobs)
        for name in model_names
    }
    model_names = [name for name in model_names if name in factories]
    reference_by_model = {
        row["name"]: row
        for row in benchmark.get("leaderboard", [])
        if row.get("name") in model_names
    }

    model_payloads: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model_index, model_name in enumerate(model_names):
        try:
            model_payloads.append(
                _attribute_model(
                    model_name,
                    factories[model_name],
                    prepared_rows,
                    dates,
                    first_test_index=first_test_index,
                    test_window_dates=test_window_dates,
                    embargo_dates=embargo_dates,
                    reference_metrics=reference_by_model.get(model_name),
                    permutation_repeats=permutation_repeats,
                    random_seed=random_seed + model_index * 100_000,
                    feature_columns=feature_columns,
                )
            )
        except Exception as exc:  # isolated research-model boundary
            errors.append(
                {
                    "model": model_name,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    feature_rows = [
        row
        for model_payload in model_payloads
        for row in model_payload["feature_rows"]
    ]
    existing_prediction_rows = existing_prediction_rows or []
    expected_prediction_columns = {
        f"{PREDICTION_PREFIX}{name}" for name in model_names
    }
    available_prediction_columns = set(existing_prediction_rows[0]) if existing_prediction_rows else set()
    fold_audit = [
        {
            "fold_id": fold_id,
            "train_start_date": train_dates[0],
            "train_end_date": train_dates[-1],
            "test_start_date": test_dates[0],
            "test_end_date": test_dates[-1],
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "embargoed_dates": embargoed_dates,
            "chronological_guard_passed": train_dates[-1] < test_dates[0],
        }
        for fold_id, train_rows, test_rows, train_dates, test_dates, embargoed_dates
        in _walk_forward_partitions(
            prepared_rows,
            dates,
            first_test_index=first_test_index,
            test_window_dates=test_window_dates,
            embargo_dates=embargo_dates,
        )
    ]
    return {
        "mode": "stock_level_feature_attribution_research_only",
        "purpose": (
            "Explain tabular stock-ranker feature contributions and measure "
            "leakage-safe leave-one-feature-out degradation."
        ),
        "source_path": source_path,
        "benchmark_path": benchmark_path,
        "predictions_path": predictions_path,
        "target_column": TARGET_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "models_requested": model_names,
        "models_completed": [row["model"] for row in model_payloads],
        "model_errors": errors,
        "input_row_count": len(rows),
        "eligible_row_count": len(prepared_rows),
        "excluded_incomplete_row_count": excluded_row_count,
        "benchmark_oos_prediction_row_count": len(existing_prediction_rows),
        "benchmark_prediction_columns_present": sorted(
            expected_prediction_columns & available_prediction_columns
        ),
        "walk_forward": {
            "method": "chronological_expanding_window",
            "min_train_dates": min_train_dates,
            "test_window_dates": test_window_dates,
            "embargo_rebalance_dates": embargo_dates,
            "out_of_sample_only": True,
            "all_chronological_guards_passed": all(
                fold["chronological_guard_passed"] for fold in fold_audit
            ),
            "folds": fold_audit,
        },
        "permutation_importance": {
            "enabled": permutation_repeats > 0,
            "repeats_per_fold": permutation_repeats,
            "method": (
                "Within-rebalance-date feature permutation; importance is the "
                "drop in fold mean Spearman IC."
            ),
        },
        "coefficient_interpretation": (
            "Linear coefficients are converted to actual-return units per one "
            "training-standard-deviation change in the feature."
        ),
        "models": model_payloads,
        "feature_rows": feature_rows,
        **RESEARCH_METADATA,
    }
