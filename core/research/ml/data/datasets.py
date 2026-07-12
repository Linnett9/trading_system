from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MULTITASK_AUXILIARY_TARGET_COLUMNS = [
    "forward_return_5d",
    "forward_return_10d",
    "future_volatility",
    "future_drawdown",
    "max_adverse_excursion",
    "max_favourable_excursion",
]
MODEL_INPUT_CONTRACT_VERSION = 2

# Columns whose values are known only after feature_date or are used directly to
# construct a target. They may remain in source rows for labels and diagnostics,
# but must never enter a model predictor matrix.
TARGET_DERIVED_COLUMNS = frozenset({
    "champion_return_next_period",
    "benchmark_return_next_period",
    "champion_excess_return",
    "volatility_adjusted_excess_return",
    "future_volatility",
    "future_drawdown",
    "future_max_drawdown",
    "max_adverse_excursion",
    "max_favourable_excursion",
    "good_period",
    "bad_period",
    "underperforms_spy",
    "drawdown_event",
    "should_reduce_exposure",
})


def forbidden_predictor_columns(columns: set[str] | list[str]) -> list[str]:
    return sorted(
        name
        for name in columns
        if name in TARGET_DERIVED_COLUMNS
        or name.startswith("future_")
        or name.startswith("forward_")
        or name.endswith("_next_period")
    )


def dataset_leakage_audit(dataset: "MLDataset") -> dict[str, Any]:
    predictor_columns = {
        name for features in dataset.features for name in features
    }
    forbidden = forbidden_predictor_columns(predictor_columns)
    temporal_order_passed = all(
        feature_date < label_start <= label_end
        for feature_date, label_start, label_end in zip(
            dataset.feature_dates,
            dataset.label_start_dates,
            dataset.label_end_dates,
        )
    )
    return {
        "temporal_order_passed": temporal_order_passed,
        "forbidden_predictor_columns": forbidden,
        "target_derived_predictors_absent": not forbidden,
        "leakage_check_passed": temporal_order_passed and not forbidden,
    }


@dataclass(frozen=True)
class MLDataset:
    features: list[dict[str, float]]
    labels: list[int]
    feature_dates: list[str]
    label_start_dates: list[str]
    label_end_dates: list[str]
    feature_ids: list[str] = field(default_factory=list)
    metadata: list[dict[str, str]] = field(default_factory=list)
    auxiliary_targets: list[dict[str, float | None]] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return min(len(self.features), len(self.labels))

    @property
    def feature_count(self) -> int:
        if not self.features:
            return 0
        return len(self.features[0])


def build_dataset(
    feature_rows: list[dict[str, float | str]],
    label_rows: list[dict[str, Any]],
    label_name: str = "risk_regime",
) -> MLDataset:
    labels_by_key = {
        str(row.get("feature_id", row["feature_date"])): row
        for row in label_rows
    }
    samples: list[dict[str, Any]] = []

    for feature_row in feature_rows:
        feature_date = str(feature_row["feature_date"])
        feature_key = str(feature_row.get("feature_id", feature_date))
        label_row = labels_by_key.get(feature_key)
        if label_row is None:
            continue

        label_start = str(label_row["label_start_date"])
        label_end = str(label_row["label_end_date"])
        if not feature_date < label_start <= label_end:
            raise ValueError("Dataset contains a label that leaks future information")

        predictor_values = _numeric_feature_values(feature_row)
        forbidden = forbidden_predictor_columns(list(predictor_values))
        if forbidden:
            raise ValueError(
                "Dataset predictor matrix contains target-derived columns: "
                + ", ".join(forbidden)
            )
        samples.append({
            "features": predictor_values,
            "label": int(label_row[label_name]),
            "feature_date": feature_date,
            "label_start_date": label_start,
            "label_end_date": label_end,
            "feature_id": feature_key,
            "metadata": {
                "rebalance_date": str(feature_row.get("rebalance_date", feature_date)),
                "feature_timestamp": str(feature_row.get("feature_timestamp", feature_date)),
                "decision_timestamp": str(feature_row.get("decision_timestamp", feature_date)),
                "label_available_timestamp": str(
                    label_row.get("label_available_timestamp", label_end)
                ),
                "variant_id": str(feature_row.get("variant_id", "")),
                "symbol": str(feature_row.get("symbol", "")),
                "selected_symbols": str(feature_row.get("selected_symbols", "")),
                "variant_universe": str(feature_row.get("variant_universe", "")),
                "variant_rebalance_frequency": str(
                    feature_row.get("variant_rebalance_frequency", "")
                ),
                "variant_weighting": str(feature_row.get("variant_weighting", "")),
            },
            "auxiliary_targets": _auxiliary_target_values(label_row),
        })

    samples.sort(key=lambda item: (item["feature_date"], item["feature_id"]))

    return MLDataset(
        features=[sample["features"] for sample in samples],
        labels=[sample["label"] for sample in samples],
        feature_dates=[sample["feature_date"] for sample in samples],
        label_start_dates=[sample["label_start_date"] for sample in samples],
        label_end_dates=[sample["label_end_date"] for sample in samples],
        feature_ids=[sample["feature_id"] for sample in samples],
        metadata=[sample["metadata"] for sample in samples],
        auxiliary_targets=[sample["auxiliary_targets"] for sample in samples],
    )


def _auxiliary_target_values(row: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in MULTITASK_AUXILIARY_TARGET_COLUMNS:
        raw_value = row.get(name)
        if raw_value is None or raw_value == "":
            values[name] = None
            continue
        try:
            values[name] = float(raw_value)
        except (TypeError, ValueError):
            values[name] = None
    return values


def _numeric_feature_values(row: dict[str, float | str]) -> dict[str, float]:
    ignored = {
        "feature_date",
        "feature_id",
        "label_start_date",
        "label_end_date",
        "label_available_timestamp",
        "feature_timestamp",
        "decision_timestamp",
        "rebalance_date",
        "outcome_end_date",
        "selected_symbols",
        "symbol",
        "regime_label",
        "variant_id",
        "variant_rebalance_frequency",
        "variant_weighting",
        "variant_universe",
        *TARGET_DERIVED_COLUMNS,
        *MULTITASK_AUXILIARY_TARGET_COLUMNS,
    }
    features = {}
    for name, value in row.items():
        if name in ignored or forbidden_predictor_columns([name]):
            continue
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            continue
    return features


def write_dataset(path: Path, dataset: MLDataset, label_name: str = "risk_regime") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    leading_fields = [
        "feature_date",
        "label_start_date",
        "label_end_date",
        label_name,
    ]
    fieldnames = [
        *leading_fields,
        *sorted({
            fieldname
            for features in dataset.features
            for fieldname in features
            if fieldname not in leading_fields
        }),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, features in enumerate(dataset.features):
            row = {
                "feature_date": dataset.feature_dates[index],
                "label_start_date": dataset.label_start_dates[index],
                "label_end_date": dataset.label_end_dates[index],
                label_name: dataset.labels[index],
                **features,
            }
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})
