from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


AUXILIARY_TARGETS = {
    "actual_forward_return_5d": "meta_predicted_forward_return_5d",
    "actual_forward_return_10d": "meta_predicted_forward_return_10d",
    "actual_future_volatility": "meta_predicted_future_volatility",
    "actual_future_drawdown": "meta_predicted_future_drawdown",
    "actual_max_adverse_excursion": "meta_predicted_max_adverse_excursion",
    "actual_max_favourable_excursion": "meta_predicted_max_favourable_excursion",
}

AUXILIARY_PREDICTION_COLUMNS = tuple(
    prediction_name.removeprefix("meta_")
    for prediction_name in AUXILIARY_TARGETS.values()
)


@dataclass(frozen=True)
class MetaAuxiliaryResult:
    train_rows: list[dict[str, str]]
    holdout_rows: list[dict[str, str]]
    predictions_path: Path
    metrics_json_path: Path
    metrics_markdown_path: Path
    metrics: dict[str, Any]


@dataclass
class _AuxiliaryRegressor:
    feature_names: list[str]
    estimator: Any = None
    constant: float | None = None

    def predict(self, rows: list[dict[str, str]]) -> list[float]:
        if self.constant is not None:
            return [self.constant for _ in rows]
        return [
            float(value)
            for value in self.estimator.predict(_feature_matrix(rows, self.feature_names))
        ]


def run_meta_auxiliary_ensemble(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    output_dir: Path,
) -> MetaAuxiliaryResult:
    augmented_train = [dict(row) for row in train_rows]
    augmented_holdout = [dict(row) for row in holdout_rows]
    feature_names = _auxiliary_feature_names(train_rows)
    target_metrics: dict[str, dict[str, Any]] = {}

    for actual_name, prediction_name in AUXILIARY_TARGETS.items():
        usable_train = [row for row in train_rows if _finite_value(row.get(actual_name))]
        if not usable_train or not feature_names:
            target_metrics[actual_name] = {
                "available": False,
                "reason": "missing training targets or auxiliary source features",
                "prediction_column": prediction_name,
                "sample_count": 0,
            }
            continue
        model = _fit_regressor(usable_train, actual_name, feature_names)
        train_predictions = _cross_fitted_predictions(
            augmented_train,
            actual_name,
            feature_names,
        )
        holdout_predictions = model.predict(augmented_holdout)
        for row, prediction in zip(augmented_train, train_predictions):
            row[prediction_name] = str(prediction)
        for row, prediction in zip(augmented_holdout, holdout_predictions):
            row[prediction_name] = str(prediction)
        target_metrics[actual_name] = _regression_metrics(
            augmented_holdout,
            actual_name,
            prediction_name,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "meta_auxiliary_predictions.csv"
    metrics_json_path = output_dir / "meta_auxiliary_metrics.json"
    metrics_markdown_path = output_dir / "meta_auxiliary_metrics.md"
    _write_predictions(predictions_path, augmented_holdout)
    metrics = {
        "mode": "meta_auxiliary_ensemble_research_only",
        "feature_columns": feature_names,
        "targets": target_metrics,
        "available_targets": [
            name for name, payload in target_metrics.items() if payload.get("available")
        ],
        "train_prediction_method": "deterministic_three_fold_cross_fit",
        "holdout_prediction_method": "refit_all_out_of_fold_rows_then_predict_holdout",
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_metrics_markdown(metrics_markdown_path, metrics)
    return MetaAuxiliaryResult(
        train_rows=augmented_train,
        holdout_rows=augmented_holdout,
        predictions_path=predictions_path,
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        metrics=metrics,
    )


def namespaced_auxiliary_features(
    model: str,
    prediction: dict[str, str],
) -> dict[str, str]:
    return {
        f"{model}__{name}": str(prediction[name])
        for name in AUXILIARY_PREDICTION_COLUMNS
        if prediction.get(name) not in (None, "")
    }


def actual_auxiliary_values(
    expanded_row: dict[str, str],
    source_rows: list[dict[str, str]],
) -> dict[str, str]:
    values = {}
    for actual_name in AUXILIARY_TARGETS:
        candidates = []
        expanded_value = expanded_row.get(actual_name)
        if expanded_value not in (None, ""):
            candidates.append(str(expanded_value))
        candidates.extend(
            str(row[actual_name])
            for row in source_rows
            if row.get(actual_name) not in (None, "")
        )
        finite = [value for value in candidates if _finite_value(value)]
        if finite:
            values[actual_name] = finite[0]
    return values


def _fit_regressor(
    rows: list[dict[str, str]],
    actual_name: str,
    feature_names: list[str],
) -> _AuxiliaryRegressor:
    targets = [float(row[actual_name]) for row in rows]
    if len(rows) < 2 or max(targets) == min(targets):
        return _AuxiliaryRegressor(feature_names, constant=mean(targets))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(_feature_matrix(rows, feature_names), targets)
    return _AuxiliaryRegressor(feature_names, estimator=estimator)


def _cross_fitted_predictions(
    rows: list[dict[str, str]],
    actual_name: str,
    feature_names: list[str],
) -> list[float]:
    if not rows:
        return []
    fold_count = min(3, len(rows))
    predictions = [0.0] * len(rows)
    all_usable = [row for row in rows if _finite_value(row.get(actual_name))]
    fallback = mean(float(row[actual_name]) for row in all_usable)
    for fold in range(fold_count):
        validation_indexes = [
            index for index in range(len(rows)) if index % fold_count == fold
        ]
        training_rows = [
            row
            for index, row in enumerate(rows)
            if index % fold_count != fold and _finite_value(row.get(actual_name))
        ]
        model = (
            _fit_regressor(training_rows, actual_name, feature_names)
            if training_rows
            else _AuxiliaryRegressor(feature_names, constant=fallback)
        )
        fold_predictions = model.predict([rows[index] for index in validation_indexes])
        for index, prediction in zip(validation_indexes, fold_predictions):
            predictions[index] = prediction
    return predictions


def _auxiliary_feature_names(rows: list[dict[str, str]]) -> list[str]:
    return sorted({
        name
        for row in rows
        for name in row
        if "__predicted_" in name
        or name.endswith("_raw_probability")
        or name.endswith("_calibrated_probability")
    })


def _feature_matrix(
    rows: list[dict[str, str]],
    feature_names: list[str],
) -> list[list[float]]:
    return [
        [
            float(row.get(name, 0.0) or 0.0)
            if _finite_value(row.get(name, 0.0))
            else 0.0
            for name in feature_names
        ]
        for row in rows
    ]


def _regression_metrics(
    rows: list[dict[str, str]],
    actual_name: str,
    prediction_name: str,
) -> dict[str, Any]:
    pairs = [
        (float(row[actual_name]), float(row[prediction_name]))
        for row in rows
        if _finite_value(row.get(actual_name))
        and _finite_value(row.get(prediction_name))
    ]
    if not pairs:
        return {
            "available": True,
            "prediction_column": prediction_name,
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "directional_accuracy": None,
            "residual_quantiles": {},
        }
    actual = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    errors = [estimate - observed for observed, estimate in pairs]
    is_return = actual_name in {
        "actual_forward_return_5d",
        "actual_forward_return_10d",
    }
    return {
        "available": True,
        "prediction_column": prediction_name,
        "sample_count": len(pairs),
        "mae": mean(abs(value) for value in errors),
        "rmse": math.sqrt(mean(value * value for value in errors)),
        "pearson_correlation": _pearson(actual, predicted),
        "spearman_correlation": _pearson(_ranks(actual), _ranks(predicted)),
        "directional_accuracy": (
            mean(
                float((observed >= 0.0) == (estimate >= 0.0))
                for observed, estimate in pairs
            )
            if is_return
            else None
        ),
        "residual_quantiles": {
            "p10": _quantile(errors, 0.10),
            "p50": _quantile(errors, 0.50),
            "p90": _quantile(errors, 0.90),
        },
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[ordered[position][0]] = average_rank
        index = end
    return output


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    prediction_columns = [
        name for name in AUXILIARY_TARGETS.values() if any(name in row for row in rows)
    ]
    fieldnames = [
        "feature_id",
        "rebalance_date",
        "variant_id",
        *AUXILIARY_TARGETS.keys(),
        *prediction_columns,
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{name: row.get(name, "") for name in fieldnames},
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
            })


def _write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Meta Auxiliary Metrics",
        "",
        "|target|samples|mae|rmse|pearson|spearman|directional_accuracy|",
        "|---|---|---|---|---|---|---|",
    ]
    for target, payload in metrics["targets"].items():
        lines.append(
            "|{target}|{samples}|{mae}|{rmse}|{pearson}|{spearman}|{directional}|".format(
                target=target,
                samples=payload.get("sample_count", 0),
                mae=_format_metric(payload.get("mae")),
                rmse=_format_metric(payload.get("rmse")),
                pearson=_format_metric(payload.get("pearson_correlation")),
                spearman=_format_metric(payload.get("spearman_correlation")),
                directional=_format_metric(payload.get("directional_accuracy")),
            )
        )
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
