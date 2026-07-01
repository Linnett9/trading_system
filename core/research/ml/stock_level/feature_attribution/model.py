from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from core.research.framework.ranking import CrossSectionalRankingEvaluator
from core.research.ml.stock_level.feature_attribution.math import (
    _average,
    _difference,
    _first_number,
    _flat_numbers,
)
from core.research.ml.stock_level.feature_attribution.types import METRIC_NAMES
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    FEATURE_COLUMNS,
    PREDICTION_PREFIX,
    TARGET_COLUMN,
    _base_prediction_row,
    _build_tabular_model,
    _walk_forward_partitions,
)


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
