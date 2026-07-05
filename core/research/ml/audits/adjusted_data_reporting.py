from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from core.research.ml.audits.adjusted_data_types import NOTICE


def _comparison_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", []) or []
    suspicious_rows = [
        row for row in rows
        if row.get("split_like_distortion")
        or row.get("anomaly_survives_adjustment")
        or row.get("raw_suspicious_jump")
        or row.get("adjusted_suspicious_jump")
    ]
    compact = dict(payload)
    compact["rows"] = suspicious_rows[:500]
    compact["row_storage_note"] = (
        "Full side-by-side rows are written to adjusted_data_comparison.csv; "
        "JSON keeps suspicious rows only to avoid duplicating the large CSV."
    )
    compact["suspicious_json_row_count"] = len(compact["rows"])
    compact["full_csv_row_count"] = len(rows)
    return compact


def _write_comparison_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "symbol",
        "date",
        "raw_close",
        "adjusted_close",
        "adjustment_ratio",
        "raw_daily_return",
        "adjusted_daily_return",
        "raw_split_like_factor",
        "adjustment_ratio_split_like_factor",
        "split_like_distortion",
        "anomaly_survives_adjustment",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("rows", []) or []:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_replay_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "candidate_name",
        "available",
        "adjusted_canonical_return",
        "coverage_valid_adjusted_canonical_return",
        "adjusted_benchmark_relative_pass",
        "adjusted_price_return_positive",
        "adjusted_price_replay_verdict",
        "adjusted_coverage_ratio",
        "missing_adjusted_symbols",
        "missing_symbols",
        "raw_fallback_symbols",
        "empty_selection_with_positive_exposure_count",
        "affected_dates",
        "empty_selection_resolution",
        "invalid_period_count",
        "invalid_adjusted_period_count",
        "valid_period_count",
        "valid_adjusted_period_count",
        "valid_adjusted_independent_period_count",
        "minimum_adjusted_independent_periods",
        "minimum_adjusted_independent_periods_pass",
        "adjusted_full_symbol_coverage",
        "fail_closed_reason",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("candidates", {}).values():
            writer.writerow({
                **{name: row.get(name) for name in fieldnames},
                "missing_adjusted_symbols": json.dumps(
                    row.get("missing_adjusted_symbols", [])
                ),
                "missing_symbols": json.dumps(row.get("missing_symbols", [])),
                "raw_fallback_symbols": json.dumps(
                    row.get("raw_fallback_symbols", [])
                ),
                "affected_dates": json.dumps(row.get("affected_dates", [])),
            })


def _comparison_markdown(payload: dict[str, Any]) -> str:
    source = payload.get("adjusted_source", {})
    lines = [
        "# Adjusted Data Comparison",
        "",
        NOTICE,
        "",
        f"Adjusted source status: {source.get('available_status')}",
        f"Split-like distortions: {payload.get('split_like_distortion_count', 0)}",
        "",
        "|symbol|raw rows|adjusted rows|comparable rows|distortions|anomalies survive|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("symbols", []):
        lines.append(
            "|{symbol}|{raw}|{adjusted}|{comparable}|{distortions}|{survive}|".format(
                symbol=row.get("symbol"),
                raw=row.get("raw_row_count"),
                adjusted=row.get("adjusted_row_count"),
                comparable=row.get("comparable_row_count"),
                distortions=row.get("split_like_distortion_count"),
                survive=row.get("anomaly_survives_adjustment_count"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _replay_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Adjusted Price Replay",
        "",
        NOTICE,
        "",
        f"Adjusted source available: {payload.get('adjusted_source_available')}",
        "",
        "|candidate|adjusted return|coverage|valid periods|invalid periods|empty-selection count|resolution|fail-closed reason|verdict|",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload.get("candidates", {}).values():
        lines.append(
            "|{name}|{ret}|{coverage}|{valid}|{invalid}|{empty}|{resolution}|{reason}|{verdict}|".format(
                name=row.get("candidate_name"),
                ret=_fmt(row.get("adjusted_canonical_return")),
                coverage=_fmt(row.get("adjusted_coverage_ratio")),
                valid=row.get("valid_adjusted_independent_period_count"),
                invalid=row.get("invalid_adjusted_period_count"),
                empty=row.get("empty_selection_with_positive_exposure_count"),
                resolution=row.get("empty_selection_resolution"),
                reason=row.get("fail_closed_reason"),
                verdict=row.get("adjusted_price_replay_verdict"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _number_for_format(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    number = _number_for_format(value)
    return "" if number is None else f"{number:.6f}"
