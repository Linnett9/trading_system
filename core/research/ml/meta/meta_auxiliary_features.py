from __future__ import annotations

from typing import Any

from core.research.ml.meta.meta_auxiliary_math import _finite_value
from core.research.ml.meta.meta_auxiliary_types import AUXILIARY_PREDICTION_COLUMNS, AUXILIARY_TARGETS


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
