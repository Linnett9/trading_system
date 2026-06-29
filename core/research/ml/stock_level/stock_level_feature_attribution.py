from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository, JsonRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.ranking import CrossSectionalRankingEvaluator
from core.research.framework.reporting import ResearchArtifactWriter

from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    FEATURE_COLUMNS,
    PREDICTION_PREFIX,
    RESEARCH_METADATA,
    TABULAR_MODEL_NAMES,
    TARGET_COLUMN,
    _base_prediction_row,
    _build_tabular_model,
    _prepare_rows,
    _walk_forward_partitions,
)


NOTICE = "Research only. Trading impact: none. Production validated: false."
METRIC_NAMES = (
    "mean_spearman_ic",
    "top_minus_bottom_spread",
    "top_decile_return",
    "bottom_decile_return",
    "top_decile_hit_rate",
    "risk_adjusted_spread",
    "spread_sharpe",
)


@dataclass(frozen=True)
class StockLevelFeatureAttributionPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


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


@dataclass(frozen=True)
class _TabularAttributionFactory:
    model_name: str
    random_seed: int
    sklearn_n_jobs: int

    def __call__(self) -> Any:
        return _build_tabular_model(
            self.model_name,
            self.random_seed,
            self.sklearn_n_jobs,
        )


def _attribute_model(
    model_name: str,
    factory: Callable[[], Any],
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    *,
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
    reference_metrics: dict[str, Any] | None,
    permutation_repeats: int,
    random_seed: int,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    attribution_by_feature = {
        feature: {
            "coefficients": [],
            "coefficient_magnitudes": [],
            "feature_importances": [],
            "permutation_ic_drops": [],
            "permutation_records": [],
        }
        for feature in feature_columns
    }
    ablation_predictions = {feature: [] for feature in feature_columns}
    fold_count = 0
    for fold_id, train_rows, test_rows, _, _, _ in _walk_forward_partitions(
        prepared_rows,
        dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
    ):
        fold_count += 1
        x_train = _matrix(train_rows, feature_columns)
        y_train = [row[TARGET_COLUMN] for row in train_rows]
        x_test = _matrix(test_rows, feature_columns)
        model = factory()
        model.fit(x_train, y_train)
        full_values = [float(value) for value in model.predict(x_test)]
        full_fold_rows = _prediction_rows(test_rows, full_values, fold_id)
        full_fold_metrics = _metrics(
            _evaluate_signal(full_fold_rows, model_name, "prediction", kind="ml_model")
        )
        extracted = _extract_model_attribution(model, feature_columns)
        for feature in feature_columns:
            coefficient = extracted["coefficients"].get(feature)
            importance = extracted["feature_importances"].get(feature)
            if coefficient is not None:
                attribution_by_feature[feature]["coefficients"].append(coefficient)
                attribution_by_feature[feature]["coefficient_magnitudes"].append(
                    abs(coefficient)
                )
            if importance is not None:
                attribution_by_feature[feature]["feature_importances"].append(
                    importance
                )

        for feature_index, feature in enumerate(feature_columns):
            if permutation_repeats > 0:
                for repeat in range(permutation_repeats):
                    permuted = _permuted_matrix(
                        x_test,
                        test_rows,
                        feature_index,
                        seed=(
                            random_seed
                            + fold_id * 10_000
                            + feature_index * 100
                            + repeat
                        ),
                    )
                    permuted_values = [
                        float(value) for value in model.predict(permuted)
                    ]
                    permuted_metrics = _metrics(
                        _evaluate_signal(
                            _prediction_rows(test_rows, permuted_values, fold_id),
                            model_name,
                            "prediction",
                            kind="ml_model",
                        )
                    )
                    full_ic = full_fold_metrics.get("mean_spearman_ic")
                    permuted_ic = permuted_metrics.get("mean_spearman_ic")
                    if full_ic is not None and permuted_ic is not None:
                        attribution_by_feature[feature][
                            "permutation_ic_drops"
                        ].append(float(full_ic) - float(permuted_ic))
                        attribution_by_feature[feature]["permutation_records"].append(
                            {
                                "fold_id": fold_id,
                                "repeat": repeat + 1,
                                "spearman_ic_drop": float(full_ic)
                                - float(permuted_ic),
                            }
                        )

            remaining = tuple(
                candidate for candidate in FEATURE_COLUMNS if candidate != feature
            )
            ablated_model = factory()
            ablated_model.fit(_matrix(train_rows, remaining), y_train)
            ablated_values = [
                float(value)
                for value in ablated_model.predict(_matrix(test_rows, remaining))
            ]
            ablation_predictions[feature].extend(
                _prediction_rows(test_rows, ablated_values, fold_id)
            )

    full_metrics = _reference_metrics(reference_metrics)
    ablation_metrics = {
        feature: _metrics(
            _evaluate_signal(
                rows,
                f"{model_name}_without_{feature}",
                "prediction",
                kind="ablation",
            )
        )
        for feature, rows in ablation_predictions.items()
    }
    normalized = _normalized_magnitudes(attribution_by_feature)
    feature_rows = []
    for feature in FEATURE_COLUMNS:
        values = attribution_by_feature[feature]
        ablated = ablation_metrics[feature]
        row = {
            "model": model_name,
            "feature": feature,
            "attribution_method": (
                "coefficient"
                if values["coefficients"]
                else "feature_importance"
                if values["feature_importances"]
                else "permutation_only"
            ),
            "coefficient_mean": _average(values["coefficients"]),
            "coefficient_abs_mean": _average(values["coefficient_magnitudes"]),
            "normalized_coefficient_or_importance_magnitude": normalized[feature],
            "feature_importance_mean": _average(values["feature_importances"]),
            "permutation_spearman_ic_drop_mean": _average(
                values["permutation_ic_drops"]
            ),
            "permutation_observation_count": len(values["permutation_ic_drops"]),
            "fold_count": fold_count,
        }
        for metric in METRIC_NAMES:
            row[f"full_{metric}"] = full_metrics.get(metric)
            row[f"ablated_{metric}"] = ablated.get(metric)
            row[f"ablation_delta_{metric}"] = _difference(
                ablated.get(metric),
                full_metrics.get(metric),
            )
        feature_rows.append(row)
    return {
        "model": model_name,
        "fold_count": fold_count,
        "full_metrics": full_metrics,
        "permutation_importance_by_fold": [
            {"feature": feature, **record}
            for feature, values in attribution_by_feature.items()
            for record in values["permutation_records"]
        ],
        "feature_rows": feature_rows,
    }


def _extract_model_attribution(
    model: Any,
    feature_columns: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    estimator = getattr(model, "regressor_", model)
    target_scale = _first_number(getattr(getattr(model, "transformer_", None), "scale_", None)) or 1.0
    if hasattr(estimator, "steps"):
        estimator = estimator.steps[-1][1]
    coefficients = _flat_numbers(getattr(estimator, "coef_", None))
    importances = _flat_numbers(getattr(estimator, "feature_importances_", None))
    return {
        "coefficients": {
            feature: coefficient * target_scale
            for feature, coefficient in zip(feature_columns, coefficients)
        },
        "feature_importances": {
            feature: importance
            for feature, importance in zip(feature_columns, importances)
        },
    }


def _normalized_magnitudes(
    attribution_by_feature: dict[str, dict[str, list[float]]],
) -> dict[str, float | None]:
    raw = {}
    for feature, values in attribution_by_feature.items():
        raw[feature] = (
            _average(values["coefficient_magnitudes"])
            if values["coefficient_magnitudes"]
            else _average(values["feature_importances"])
        )
    total = sum(value for value in raw.values() if value is not None)
    return {
        feature: (value / total if value is not None and total > 0.0 else None)
        for feature, value in raw.items()
    }


def _permuted_matrix(
    matrix: list[list[float]],
    rows: list[dict[str, Any]],
    feature_index: int,
    *,
    seed: int,
) -> list[list[float]]:
    output = [list(row) for row in matrix]
    indices_by_date: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_date.setdefault(row["rebalance_date"], []).append(index)
    generator = random.Random(seed)
    for indices in indices_by_date.values():
        values = [output[index][feature_index] for index in indices]
        generator.shuffle(values)
        for index, value in zip(indices, values):
            output[index][feature_index] = value
    return output


def _prediction_rows(
    source_rows: list[dict[str, Any]],
    predictions: list[float],
    fold_id: int,
) -> list[dict[str, Any]]:
    if len(source_rows) != len(predictions):
        raise ValueError("Prediction count must match OOS row count")
    return [
        {**_base_prediction_row(row, fold_id), "prediction": prediction}
        for row, prediction in zip(source_rows, predictions)
    ]


def _matrix(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> list[list[float]]:
    return [[float(row[column]) for column in columns] for row in rows]


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {name: summary.get(name) for name in METRIC_NAMES}


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


def _reference_metrics(reference: dict[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        return {name: None for name in METRIC_NAMES}
    return {name: reference.get(name) for name in METRIC_NAMES}


def _flat_numbers(value: Any) -> list[float]:
    if value is None:
        return []
    candidate = value.tolist() if hasattr(value, "tolist") else value
    while isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], list):
        candidate = candidate[0]
    if not isinstance(candidate, (list, tuple)):
        candidate = [candidate]
    return [float(item) for item in candidate]


def _first_number(value: Any) -> float | None:
    numbers = _flat_numbers(value)
    return numbers[0] if numbers else None


def _average(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else None


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _output_dir(config: dict[str, Any]) -> Path:
    return StockLevelResearchConfig.from_mapping(config).output_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "feature",
        "attribution_method",
        "coefficient_mean",
        "coefficient_abs_mean",
        "normalized_coefficient_or_importance_magnitude",
        "feature_importance_mean",
        "permutation_spearman_ic_drop_mean",
        "permutation_observation_count",
        "fold_count",
        *(f"full_{metric}" for metric in METRIC_NAMES),
        *(f"ablated_{metric}" for metric in METRIC_NAMES),
        *(f"ablation_delta_{metric}" for metric in METRIC_NAMES),
    ]
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stock-Level Feature Attribution and Ablation",
        "",
        NOTICE,
        "",
        f"- Models completed: {', '.join(payload['models_completed'])}",
        f"- Eligible rows: {payload['eligible_row_count']}",
        f"- Permutation repeats per fold: {payload['permutation_importance']['repeats_per_fold']}",
        "- Promotion thresholds changed: false",
    ]
    for model in payload["models"]:
        rows = model["feature_rows"]
        attribution_ranked = sorted(
            rows,
            key=lambda row: -float(
                row.get("normalized_coefficient_or_importance_magnitude") or 0.0
            ),
        )
        ablation_ranked = sorted(
            rows,
            key=lambda row: float(
                row.get("ablation_delta_mean_spearman_ic") or 0.0
            ),
        )
        lines.extend(
            [
                "",
                f"## {model['model']}",
                "",
                "### Attribution",
                "",
                "| Feature | Coefficient | Normalized magnitude | Tree importance | Permutation IC drop |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in attribution_ranked:
            lines.append(
                "| {feature} | {coef} | {magnitude} | {tree} | {permutation} |".format(
                    feature=row["feature"],
                    coef=_fmt(row.get("coefficient_mean")),
                    magnitude=_fmt(
                        row.get("normalized_coefficient_or_importance_magnitude")
                    ),
                    tree=_fmt(row.get("feature_importance_mean")),
                    permutation=_fmt(
                        row.get("permutation_spearman_ic_drop_mean")
                    ),
                )
            )
        lines.extend(
            [
                "",
                "### Leave-One-Feature-Out Ablation",
                "",
                "| Removed feature | Spearman IC | IC delta | Spread | Spread delta | Hit rate | Risk-adjusted spread | Spread Sharpe |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ablation_ranked:
            lines.append(
                "| {feature} | {ic} | {ic_delta} | {spread} | {spread_delta} | {hit} | {risk} | {sharpe} |".format(
                    feature=row["feature"],
                    ic=_fmt(row.get("ablated_mean_spearman_ic")),
                    ic_delta=_fmt(row.get("ablation_delta_mean_spearman_ic")),
                    spread=_fmt(row.get("ablated_top_minus_bottom_spread")),
                    spread_delta=_fmt(
                        row.get("ablation_delta_top_minus_bottom_spread")
                    ),
                    hit=_fmt(row.get("ablated_top_decile_hit_rate")),
                    risk=_fmt(row.get("ablated_risk_adjusted_spread")),
                    sharpe=_fmt(row.get("ablated_spread_sharpe")),
                )
            )
    if payload["model_errors"]:
        lines.extend(["", "## Model Errors", ""])
        for error in payload["model_errors"]:
            lines.append(f"- {error['model']}: {error['reason']}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
