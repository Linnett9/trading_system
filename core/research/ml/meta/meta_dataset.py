from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.artifacts.artifact_validator import validate_prediction_artifacts
from core.research.ml.meta.meta_auxiliary import (
    actual_auxiliary_values,
    namespaced_auxiliary_features,
)
from core.research.ml.meta.meta_io import _read_csv


def build_meta_dataset_rows(
    expanded_rows: list[dict[str, str]],
    source_predictions: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = []
    missing_counts = {model: 0 for model in source_predictions}
    auxiliary_prediction_columns_by_model: dict[str, list[str]] = {}
    namespaced_auxiliary_columns_by_model: dict[str, list[str]] = {}
    ignored_leakage_columns_by_model: dict[str, list[str]] = {}
    duplicate_feature_ids = len(expanded_rows) - len({row["feature_id"] for row in expanded_rows})
    expanded_by_id = {row["feature_id"]: row for row in expanded_rows}
    all_feature_ids = sorted(
        set.intersection(
            set(expanded_by_id),
            *(set(predictions) for predictions in source_predictions.values()),
        )
    )
    for feature_id in all_feature_ids:
        expanded = expanded_by_id[feature_id]
        split_values = {
            predictions[feature_id].get("split", "")
            for predictions in source_predictions.values()
        }
        split = "holdout" if "holdout" in split_values else "out_of_fold"
        row = {
            "feature_id": feature_id,
            "rebalance_date": expanded.get("rebalance_date", expanded.get("feature_date", "")),
            "variant_id": expanded.get("variant_id", ""),
            "split": split,
            "fold": next(iter(source_predictions.values()))[feature_id].get("fold", ""),
            "actual_label": next(iter(source_predictions.values()))[feature_id]["actual_label"],
            "label_start_date": expanded.get("label_start_date", ""),
            "label_end_date": expanded.get(
                "label_end_date",
                expanded.get("outcome_end_date", ""),
            ),
        }
        row.update(actual_auxiliary_values(
            expanded,
            [predictions[feature_id] for predictions in source_predictions.values()],
        ))
        for model, predictions in source_predictions.items():
            prediction = predictions.get(feature_id)
            if prediction is None:
                missing_counts[model] += 1
                continue
            row[f"{model}_raw_probability"] = prediction["raw_probability"]
            row[f"{model}_calibrated_probability"] = (
                prediction.get("calibrated_probability") or prediction["raw_probability"]
            )
            auxiliary_features, ignored_columns = _source_prediction_feature_values(
                model,
                prediction,
            )
            auxiliary_prediction_columns_by_model.setdefault(model, [])
            ignored_leakage_columns_by_model.setdefault(model, [])
            for name, value in auxiliary_features.items():
                row[name] = value
                if name not in auxiliary_prediction_columns_by_model[model]:
                    auxiliary_prediction_columns_by_model[model].append(name)
            namespaced_features = namespaced_auxiliary_features(model, prediction)
            namespaced_auxiliary_columns_by_model.setdefault(model, [])
            for name, value in namespaced_features.items():
                row[name] = value
                if name not in namespaced_auxiliary_columns_by_model[model]:
                    namespaced_auxiliary_columns_by_model[model].append(name)
            for name in ignored_columns:
                if name not in ignored_leakage_columns_by_model[model]:
                    ignored_leakage_columns_by_model[model].append(name)
        for name in (
            "variant_top_n",
            "variant_universe_symbol_count",
            "breadth_above_sma_200",
            "spy_realized_volatility_21d",
            "spy_max_drawdown_63d",
            "recent_champion_excess_return",
            "replacements",
            "champion_return_next_period",
        ):
            row[name] = expanded.get(name, "0")
        rows.append(row)
    audit = {
        "row_count": len(rows),
        "feature_count": len(_feature_values(rows[0])) if rows else 0,
        "date_range": [rows[0]["rebalance_date"], rows[-1]["rebalance_date"]] if rows else None,
        "class_balance": _class_balance(rows),
        "source_model_count": len(source_predictions),
        "source_dataset_hash": _single_source_dataset_hash(source_predictions),
        "source_dataset_row_counts_by_model": _source_prediction_field_values(
            source_predictions,
            "source_dataset_row_count",
        ),
        "source_artifact_generated_at_by_model": _source_prediction_field_values(
            source_predictions,
            "artifact_generated_at",
        ),
        "missing_prediction_counts_by_model": missing_counts,
        "auxiliary_prediction_columns_by_model": {
            model: sorted(columns)
            for model, columns in auxiliary_prediction_columns_by_model.items()
        },
        "namespaced_auxiliary_prediction_columns_by_model": {
            model: sorted(columns)
            for model, columns in namespaced_auxiliary_columns_by_model.items()
        },
        "ignored_leakage_columns_by_model": {
            model: sorted(columns)
            for model, columns in ignored_leakage_columns_by_model.items()
        },
        "duplicate_feature_id_count": duplicate_feature_ids,
        "same_date_leakage_check": _same_date_leakage_check(rows),
        "meta_training_uses_in_sample_base_predictions": False,
    }
    return rows, audit


def _load_source_predictions(source_dirs: list[Path]) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    sources = {}
    warnings = []
    dataset_hashes: dict[str, str] = {}
    for source_dir in source_dirs:
        path = source_dir / "prediction_artifacts.csv"
        if not path.exists():
            raise RuntimeError(
                "Missing prediction artifact CSV: "
                f"{path}. Rerun ml-research for {source_dir}."
            )
        metadata_path = source_dir / "prediction_artifacts.json"
        validation_result = validate_prediction_artifacts(path, metadata_path)
        if validation_result.legacy_warnings:
            raise RuntimeError(
                "Prediction artifact is legacy or schema-incomplete: "
                f"{source_dir}: {', '.join(validation_result.legacy_warnings)}. "
                "Rerun ml-research so prediction_artifacts.csv/json are regenerated."
            )
        metadata = _read_prediction_artifact_metadata(metadata_path)
        dataset_hash = str(metadata.get("dataset_hash") or metadata.get("data_hash") or "")
        if not dataset_hash:
            raise RuntimeError(
                "Prediction artifact metadata is missing dataset_hash: "
                f"{metadata_path}. Rerun ml-research for {source_dir}."
            )
        rows = _read_csv(path)
        if not rows:
            continue
        model_type = rows[0]["model_type"]
        row_hashes = {
            row.get("dataset_hash", "")
            for row in rows
            if row.get("split") in {"out_of_fold", "holdout"}
        }
        if "" in row_hashes:
            raise RuntimeError(
                "Prediction artifact CSV is missing dataset_hash values: "
                f"{path}. Rerun ml-research so prediction_artifacts.csv is regenerated."
            )
        if row_hashes and row_hashes != {dataset_hash}:
            raise RuntimeError(
                "Prediction artifact CSV dataset_hash does not match metadata "
                f"for {source_dir}: csv={sorted(row_hashes)} metadata={dataset_hash}"
            )
        dataset_hashes[model_type] = dataset_hash
        sources[model_type] = {
            row["feature_id"]: {
                **row,
                "dataset_hash": dataset_hash,
                "source_dataset_row_count": str(
                    metadata.get("source_dataset_row_count", row.get("source_dataset_row_count", ""))
                ),
                "artifact_generated_at": str(
                    metadata.get("generated_at", row.get("generated_at", ""))
                ),
                "artifact_train_sample_count": str(
                    metadata.get("train_sample_count", row.get("train_sample_count", ""))
                ),
                "artifact_test_sample_count": str(
                    metadata.get("test_sample_count", row.get("test_sample_count", ""))
                ),
            }
            for row in rows
            if row.get("split") in {"out_of_fold", "holdout"}
        }
    _validate_source_dataset_hashes(dataset_hashes)
    return sources, warnings


def _read_prediction_artifact_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing prediction artifact metadata: {path}. "
            "Rerun ml-research so prediction_artifacts.json is regenerated."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_source_dataset_hashes(dataset_hashes: dict[str, str]) -> None:
    unique_hashes = {value for value in dataset_hashes.values() if value}
    if len(unique_hashes) > 1:
        details = ", ".join(
            f"{model}={dataset_hash}"
            for model, dataset_hash in sorted(dataset_hashes.items())
        )
        raise RuntimeError(
            "Meta ensemble prediction artifacts were generated from different "
            f"dataset hashes. Rerun ml-research for all source models. {details}"
        )


def _single_source_dataset_hash(
    source_predictions: dict[str, dict[str, dict[str, str]]],
) -> str | None:
    hashes = {
        row.get("dataset_hash", "")
        for predictions in source_predictions.values()
        for row in predictions.values()
        if row.get("dataset_hash")
    }
    return next(iter(hashes)) if len(hashes) == 1 else None


def _source_prediction_field_values(
    source_predictions: dict[str, dict[str, dict[str, str]]],
    field_name: str,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for model, predictions in source_predictions.items():
        for row in predictions.values():
            value = row.get(field_name)
            if value:
                values[model] = value
                break
    return values


def _feature_values(row: dict[str, str]) -> dict[str, float]:
    ignored = {"feature_id", "rebalance_date", "variant_id", "split", "fold", "actual_label"}
    values = {}
    for name, value in row.items():
        if name in ignored:
            continue
        if "__predicted_" in name or name.startswith("meta_predicted_"):
            continue
        if _is_leakage_column(name):
            continue
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _source_prediction_feature_values(
    model: str,
    prediction: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    ignored_columns: list[str] = []
    for name, value in prediction.items():
        if name.startswith("predicted_"):
            if _is_allowed_source_prediction_feature(name):
                values[f"{model}_{name}"] = value
            else:
                ignored_columns.append(name)
            continue
        if name.startswith("actual_") or _is_leakage_column(name):
            ignored_columns.append(name)
    return values, ignored_columns


def _is_allowed_source_prediction_feature(name: str) -> bool:
    return (
        name.startswith("predicted_")
        and not _is_leakage_column(name)
        and name not in {"predicted_class", "predicted_label"}
    )


def _is_leakage_column(name: str) -> bool:
    normalized = name.lower()
    if normalized.startswith("actual_"):
        return True
    if normalized.startswith("predicted_") or "_predicted_" in normalized:
        return False
    return any(
        token in normalized
        for token in (
            "future_",
            "forward_return_",
            "max_adverse_excursion",
            "max_favourable_excursion",
            "label_start",
            "label_end",
        )
    ) and not normalized.startswith("predicted_")


def _same_date_leakage_check(rows: list[dict[str, str]]) -> dict[str, Any]:
    split_by_date: dict[str, set[str]] = {}
    for row in rows:
        split_by_date.setdefault(row["rebalance_date"], set()).add(row["split"])
    leaked = sorted(date for date, splits in split_by_date.items() if len(splits) > 1)
    return {"passed": not leaked, "leaked_dates": leaked}


def _class_balance(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    positives = sum(int(row["actual_label"]) for row in rows)
    total = len(rows)
    return {
        "positive": positives,
        "negative": total - positives,
        "positive_rate": positives / total if total else None,
    }
