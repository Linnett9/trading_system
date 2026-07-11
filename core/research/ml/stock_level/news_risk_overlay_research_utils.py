from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.research.ml.stock_level.news_risk_overlay import DECISION_TIMESTAMP_COLUMNS, TIMESTAMP_COLUMNS

LABEL_SOURCE_COLUMNS = (
    "actual_max_adverse_excursion",
    "forward_max_adverse_excursion",
    "max_adverse_excursion",
    "actual_forward_return_20d",
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
    "stop_hit_before_target",
)
RETURN_COLUMNS = (
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
)
PRICE_SCORE_COLUMNS = (
    "stock_level_predicted_forward_return_10d_elastic_net",
    "stock_level_predicted_forward_return_10d_gradient_boosting",
    "stock_level_predicted_forward_return_10d_random_forest",
    "stock_level_predicted_forward_return_10d_ridge",
    "predicted_forward_return_10d",
    "predicted_momentum_120d",
    "predicted_risk_adjusted_momentum",
)
EXCLUDED_FEATURE_PREFIXES = ("actual_", "forward_", "news_risk_", "price_only_", "price_plus_news_")
EXCLUDED_FEATURE_COLUMNS = {
    "symbol",
    "sector",
    "fold_id",
    "source",
    "source_feature_id",
    "source_model_type",
    "source_split",
    "source_dataset_hash",
    "true_stock_level_row",
    "decision_timestamp",
    "rebalance_date",
    "feature_date",
    "date",
    "news_feature_timestamp",
    "news_coverage_status",
    "news_missing_coverage",
    "news_has_coverage_30d",
    "news_news_has_coverage_30d",
}



def _configured_first(value: Any, defaults: Iterable[str]) -> list[str]:
    return [str(value), *defaults] if value else list(defaults)


def _value_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _empty_score_direction_report() -> dict[str, Any]:
    return {
        "uses_out_of_sample_predictions_only": True,
        "candidate_count": 0,
        "spearman_news_score_vs_future_return": 0.0,
        "correlation_news_score_vs_maximum_adverse_excursion": 0.0,
        "correlation_news_score_vs_maximum_favourable_excursion": 0.0,
        "monotonicity": {"direction": "unavailable", "violations": 0},
        "confidence_intervals": {},
        "answers": {
            "higher_score_predicts_lower_return": False,
            "higher_score_predicts_deeper_temporary_drawdown": False,
            "higher_score_predicts_greater_movement_both_directions": False,
            "relationship_supports_inversion": False,
        },
    }


def _first_numeric(row: Mapping[str, Any], columns: Iterable[str]) -> float | None:
    for column in columns:
        value = _number(row.get(column))
        if value is not None:
            return value
    return None


def _favourable_excursion(row: Mapping[str, Any]) -> float | None:
    for column in (
        "actual_max_favourable_excursion",
        "forward_max_favourable_excursion",
        "max_favourable_excursion",
        "actual_max_favorable_excursion",
        "forward_max_favorable_excursion",
        "max_favorable_excursion",
    ):
        value = _number(row.get(column))
        if value is not None:
            return value
    forward = _first_numeric(row, RETURN_COLUMNS)
    return max(forward, 0.0) if forward is not None else None


def _row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("decision_timestamp", row.get("rebalance_date", "")))[:19],
            str(row.get("symbol", "")).upper(),
            str(row.get("price_plus_news_risk_probability", "")),
        ]
    )


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _timestamp(row: Mapping[str, Any]) -> datetime:
    for column in ("decision_timestamp", *DECISION_TIMESTAMP_COLUMNS):
        value = row.get(column)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    raise ValueError("row missing decision timestamp")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] | None = None,
) -> None:
    if not rows:
        if not empty_fields:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(empty_fields)).writeheader()
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _check_output_disk_space(output_dir: Path, ml: Mapping[str, Any]) -> None:
    estimated_bytes = int(ml.get("stock_alpha_news_risk_overlay_estimated_output_bytes", 10_000_000))
    minimum_free_bytes = int(
        ml.get(
            "stock_alpha_news_risk_overlay_min_free_bytes",
            max(25_000_000, estimated_bytes * 2),
        )
    )
    target = output_dir if output_dir.exists() else output_dir.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    free_bytes = shutil.disk_usage(target).free
    if free_bytes < minimum_free_bytes:
        raise ValueError(
            "insufficient disk space for stock-alpha news risk overlay outputs: "
            f"free_bytes={free_bytes}, required_free_bytes={minimum_free_bytes}, "
            f"estimated_output_bytes={estimated_bytes}. "
            "No files were deleted automatically. Safe cleanup options: remove or move "
            "old untracked research-results outputs, old generated reports under reports/ml, "
            "or external local caches after reviewing them."
        )


def _limited_rows(rows: list[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def _limited_audit_details(payload: Mapping[str, Any], limit: int) -> dict[str, Any]:
    output = dict(payload)
    details = output.get("max_news_timestamp_by_decision")
    if isinstance(details, list) and limit > 0:
        output["max_news_timestamp_by_decision_total_count"] = len(details)
        output["max_news_timestamp_by_decision"] = details[:limit]
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else None


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = [name for name in globals() if not name.startswith("__")]
