from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
TARGET_PERIODS = (36, 60)


@dataclass(frozen=True)
class HistoricalCoverageAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_historical_coverage_audit(
    config: dict[str, Any],
) -> HistoricalCoverageAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        Path(str(path))
        for path in config.get("ml", {}).get("source_prediction_dirs", []) or []
    ]
    payload = build_historical_coverage_audit(
        raw_price_ranges=_load_raw_price_ranges(
            Path(
                str(
                    config.get("ml", {}).get(
                        "stooq_parquet_dir",
                        "data/processed/stooq_parquet",
                    )
                )
            )
        ),
        adjusted_price_ranges=_load_adjusted_price_ranges(
            Path(
                str(
                    config.get("ml", {})
                    .get("adjusted_data_source", {})
                    .get("adjusted_data_dir", "data/reference/adjusted_prices")
                )
            )
        ),
        source_prediction_ranges=[
            _prediction_artifact_range(path / "prediction_artifacts.csv")
            for path in source_dirs
        ],
        meta_prediction_range=_prediction_artifact_range(
            output_dir / "meta_auxiliary_predictions.csv"
        ),
        canonical_replay=_read_json(output_dir / "canonical_continuous_equity_replay.json"),
        adjusted_price_replay=_read_json(output_dir / "adjusted_price_replay.json"),
        config=_audit_config(config),
    )
    paths = HistoricalCoverageAuditPaths(
        csv_path=output_dir / "historical_coverage_audit.csv",
        json_path=output_dir / "historical_coverage_audit.json",
        markdown_path=output_dir / "historical_coverage_audit.md",
    )
    _write_csv(paths.csv_path, payload)
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_historical_coverage_audit(
    *,
    raw_price_ranges: list[dict[str, Any]],
    adjusted_price_ranges: list[dict[str, Any]],
    source_prediction_ranges: list[dict[str, Any]],
    meta_prediction_range: dict[str, Any],
    canonical_replay: dict[str, Any],
    adjusted_price_replay: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    minimum = int(config.get("min_independent_periods", 36))
    label_window_days = _median_label_window_days(canonical_replay)
    raw = _aggregate_ranges(raw_price_ranges, "raw_stooq_parquet")
    adjusted = _aggregate_ranges(adjusted_price_ranges, "adjusted_reference_csv")
    source_predictions = _aggregate_ranges(
        [row for row in source_prediction_ranges if row.get("available")],
        "source_prediction_artifacts",
    )
    meta_predictions = _range_summary(meta_prediction_range, "meta_auxiliary_predictions")
    canonical = _canonical_summary(canonical_replay)
    adjusted_replay = _adjusted_replay_summary(adjusted_price_replay)
    possible = _possible_non_overlap_windows(canonical_replay)
    needed = {
        str(target): _required_history(
            latest=canonical.get("latest_canonical_replay_date")
            or meta_predictions.get("latest_date")
            or source_predictions.get("latest_date"),
            target_count=target,
            label_window_days=label_window_days,
        )
        for target in TARGET_PERIODS
    }
    bottleneck = _bottleneck(
        raw=raw,
        adjusted=adjusted,
        source_predictions=source_predictions,
        meta_predictions=meta_predictions,
        canonical=canonical,
        minimum=minimum,
    )
    recommendations = _recommendations(
        bottleneck=bottleneck,
        raw=raw,
        adjusted=adjusted,
        source_predictions=source_predictions,
        meta_predictions=meta_predictions,
        needed=needed,
    )
    return {
        "mode": "historical_coverage_audit_research_only",
        "minimum_independent_periods": minimum,
        "target_independent_periods": list(TARGET_PERIODS),
        "label_window_days_median": label_window_days,
        "raw_prices": raw,
        "adjusted_prices": adjusted,
        "source_prediction_artifacts": source_predictions,
        "meta_prediction_artifacts": meta_predictions,
        "canonical_replay": canonical,
        "adjusted_replay": adjusted_replay,
        "possible_leakage_safe_non_overlap_windows": possible,
        "history_required_for_targets": needed,
        "historical_bottleneck": bottleneck,
        "blockers": _blockers(
            bottleneck=bottleneck,
            canonical=canonical,
            adjusted_replay=adjusted_replay,
            minimum=minimum,
        ),
        "recommendations": recommendations,
        "full_model_rerun_required": recommendations["full_model_rerun_required"],
        "overnight_command_if_rerun_justified": (
            "python3.10 main.py --mode ml-research-batch "
            "--config configs/research/regime_transformer_meta_ensemble_v1.yaml "
            "--profile benchmark"
        ),
        "rows": _rows(raw, adjusted, source_predictions, meta_predictions, canonical),
        **RESEARCH_METADATA,
    }


def _load_raw_price_ranges(data_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(data_dir.glob("*.parquet")):
        summary = _parquet_date_range(path)
        if summary:
            output.append(summary)
    return output


def _load_adjusted_price_ranges(data_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == "manifest.json":
            continue
        summary = _csv_date_range(path, date_columns=("date", "Date", "timestamp"))
        if summary:
            summary["symbol"] = path.stem.upper()
            output.append(summary)
    return output


def _prediction_artifact_range(path: Path) -> dict[str, Any]:
    summary = _csv_date_range(
        path,
        date_columns=("rebalance_date", "prediction_date", "date", "feature_date"),
    )
    if not summary:
        return {
            "path": str(path),
            "available": False,
            "row_count": 0,
            "unique_rebalance_dates": 0,
        }
    summary["name"] = path.parent.name
    return summary


def _parquet_date_range(path: Path) -> dict[str, Any] | None:
    try:
        import pandas as pd
    except ImportError:
        return {
            "path": str(path),
            "symbol": path.stem.upper(),
            "available": False,
            "row_count": 0,
            "error": "pandas_not_available_for_parquet_scan",
        }
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:
        return {
            "path": str(path),
            "symbol": path.stem.upper(),
            "available": False,
            "row_count": 0,
            "error": str(exc),
        }
    if frame.empty:
        return None
    dates = [str(value)[:10] for value in frame["timestamp"].dropna().tolist()]
    if not dates:
        return None
    return {
        "path": str(path),
        "symbol": path.stem.upper(),
        "available": True,
        "row_count": len(dates),
        "earliest_date": min(dates),
        "latest_date": max(dates),
    }


def _csv_date_range(path: Path, *, date_columns: tuple[str, ...]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row_count = 0
    dates = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            value = next((row.get(name) for name in date_columns if row.get(name)), None)
            if value:
                dates.append(str(value)[:10])
    if not dates:
        return {
            "path": str(path),
            "available": True,
            "row_count": row_count,
            "unique_rebalance_dates": 0,
        }
    return {
        "path": str(path),
        "available": True,
        "row_count": row_count,
        "earliest_date": min(dates),
        "latest_date": max(dates),
        "unique_rebalance_dates": len(set(dates)),
    }


def _aggregate_ranges(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    available = [row for row in rows if row.get("available") and row.get("earliest_date")]
    return {
        "name": name,
        "available": bool(available),
        "item_count": len(rows),
        "available_item_count": len(available),
        "earliest_date": min((row["earliest_date"] for row in available), default=None),
        "latest_date": max((row["latest_date"] for row in available), default=None),
        "latest_common_start_date": max(
            (row["earliest_date"] for row in available),
            default=None,
        ),
        "earliest_common_end_date": min(
            (row["latest_date"] for row in available),
            default=None,
        ),
        "row_count": sum(int(row.get("row_count") or 0) for row in rows),
        "unique_rebalance_dates": max(
            (int(row.get("unique_rebalance_dates") or 0) for row in rows),
            default=0,
        ),
        "items": rows,
    }


def _range_summary(row: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "available": bool(row.get("available")),
        "path": row.get("path"),
        "row_count": int(row.get("row_count") or 0),
        "earliest_date": row.get("earliest_date"),
        "latest_date": row.get("latest_date"),
        "unique_rebalance_dates": int(row.get("unique_rebalance_dates") or 0),
    }


def _canonical_summary(canonical_replay: dict[str, Any]) -> dict[str, Any]:
    candidates = canonical_replay.get("candidates", {}) or {}
    exact = candidates.get("exact_champion_replay", {}) or {}
    optimizer = candidates.get("selected_bayesian_optimizer_diagnostic_policy", {}) or {}
    all_rows = [
        row for candidate in (exact, optimizer)
        for row in candidate.get("rows", []) or []
        if isinstance(row, dict) and row.get("rebalance_date")
    ]
    return {
        "earliest_canonical_replay_date": min(
            (str(row.get("rebalance_date")) for row in all_rows),
            default=None,
        ),
        "latest_canonical_replay_date": max(
            (str(row.get("rebalance_date")) for row in all_rows),
            default=None,
        ),
        "rebalance_date_count": len({
            str(row.get("rebalance_date")) for row in all_rows
        }),
        "raw_independent_periods_exact": int(
            (exact.get("canonical_continuous_equity", {}) or {}).get("row_count")
            or 0
        ),
        "raw_independent_periods_optimizer": int(
            (optimizer.get("canonical_continuous_equity", {}) or {}).get("row_count")
            or 0
        ),
        "diagnostic_periods_exact": int(
            (exact.get("diagnostic_period_grid", {}) or {}).get("row_count") or 0
        ),
        "diagnostic_periods_optimizer": int(
            (optimizer.get("diagnostic_period_grid", {}) or {}).get("row_count") or 0
        ),
    }


def _adjusted_replay_summary(adjusted_price_replay: dict[str, Any]) -> dict[str, Any]:
    candidates = adjusted_price_replay.get("candidates", {}) or {}
    return {
        name: {
            "valid_adjusted_independent_period_count": int(
                (row or {}).get("valid_adjusted_independent_period_count") or 0
            ),
            "valid_adjusted_period_count": int(
                (row or {}).get("valid_adjusted_period_count") or 0
            ),
            "invalid_adjusted_period_count": int(
                (row or {}).get("invalid_adjusted_period_count") or 0
            ),
            "fail_closed_reason": (row or {}).get("fail_closed_reason"),
        }
        for name, row in candidates.items()
        if isinstance(row, dict)
    }


def _possible_non_overlap_windows(canonical_replay: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, candidate in (canonical_replay.get("candidates", {}) or {}).items():
        rows = [
            row for row in candidate.get("rows", []) or []
            if isinstance(row, dict) and not row.get("exclusion_reason")
        ]
        output[name] = _non_overlap_count(rows)
    return output


def _non_overlap_count(rows: list[dict[str, Any]]) -> int:
    previous_end = None
    count = 0
    for row in sorted(rows, key=lambda item: str(item.get("rebalance_date", ""))):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None:
            continue
        if previous_end is None or start >= previous_end:
            count += 1
            previous_end = end
    return count


def _median_label_window_days(canonical_replay: dict[str, Any]) -> int:
    windows = []
    for candidate in (canonical_replay.get("candidates", {}) or {}).values():
        for row in candidate.get("rows", []) or []:
            start = _date(row.get("rebalance_date"))
            end = _date(row.get("outcome_end_date"))
            if start and end and end >= start:
                windows.append((end - start).days)
    if not windows:
        return 60
    windows = sorted(windows)
    return int(windows[len(windows) // 2])


def _required_history(
    *,
    latest: str | None,
    target_count: int,
    label_window_days: int,
) -> dict[str, Any]:
    latest_date = _date(latest)
    days = max(0, (target_count - 1) * max(1, label_window_days))
    required_start = latest_date - timedelta(days=days) if latest_date else None
    return {
        "target_independent_periods": target_count,
        "estimated_required_calendar_days": days,
        "estimated_required_start_date": (
            required_start.date().isoformat() if required_start else None
        ),
    }


def _bottleneck(
    *,
    raw: dict[str, Any],
    adjusted: dict[str, Any],
    source_predictions: dict[str, Any],
    meta_predictions: dict[str, Any],
    canonical: dict[str, Any],
    minimum: int,
) -> dict[str, Any]:
    raw_count = int(canonical.get("raw_independent_periods_exact") or 0)
    reasons = []
    if raw_count < minimum:
        reasons.append("too_few_canonical_independent_periods")
    if _date(raw.get("earliest_date")) and _date(source_predictions.get("earliest_date")):
        if _date(raw["earliest_date"]) < _date(source_predictions["earliest_date"]):
            reasons.append("prediction_artifacts_start_after_price_history")
    if _date(source_predictions.get("earliest_date")) and _date(meta_predictions.get("earliest_date")):
        if _date(source_predictions["earliest_date"]) < _date(meta_predictions["earliest_date"]):
            reasons.append("meta_or_canonical_artifacts_start_after_source_predictions")
    if not adjusted.get("available"):
        reasons.append("adjusted_prices_missing")
    limiting_layer = "canonical_or_meta_artifacts"
    if reasons == ["too_few_canonical_independent_periods"]:
        limiting_layer = "rebalance_cadence_or_label_window"
    if "meta_or_canonical_artifacts_start_after_source_predictions" in reasons:
        limiting_layer = "meta_or_canonical_artifacts"
    elif "prediction_artifacts_start_after_price_history" in reasons:
        limiting_layer = "prediction_artifacts"
    if not raw.get("available") or not adjusted.get("available"):
        limiting_layer = "price_data"
    return {
        "limiting_layer": limiting_layer,
        "reasons": reasons,
        "current_raw_independent_periods": raw_count,
        "minimum_independent_periods": minimum,
    }


def _blockers(
    *,
    bottleneck: dict[str, Any],
    canonical: dict[str, Any],
    adjusted_replay: dict[str, Any],
    minimum: int,
) -> list[str]:
    blockers = list(bottleneck.get("reasons", []))
    exact_adjusted = adjusted_replay.get("exact_champion_replay", {})
    if int(exact_adjusted.get("valid_adjusted_independent_period_count") or 0) < minimum:
        blockers.append("too_few_valid_adjusted_independent_periods")
    if int(canonical.get("diagnostic_periods_exact") or 0) > int(
        canonical.get("raw_independent_periods_exact") or 0
    ):
        blockers.append("overlapping_label_windows_reduce_independent_count")
    return sorted(set(blockers))


def _recommendations(
    *,
    bottleneck: dict[str, Any],
    raw: dict[str, Any],
    adjusted: dict[str, Any],
    source_predictions: dict[str, Any],
    meta_predictions: dict[str, Any],
    needed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full_rerun = bottleneck.get("limiting_layer") in {
        "prediction_artifacts",
        "meta_or_canonical_artifacts",
    }
    return {
        "extend_benchmark_start_date": (
            "raw_adjusted_data_support_earlier_history"
            if raw.get("earliest_date") and adjusted.get("earliest_date")
            else "price_data_unavailable"
        ),
        "regenerate_base_artifacts": bool(full_rerun),
        "regenerate_reason": bottleneck.get("limiting_layer"),
        "minimum_history_needed": needed,
        "full_model_rerun_required": bool(full_rerun),
        "price_data_supports_earlier_than_predictions": bool(
            _date(raw.get("earliest_date"))
            and _date(source_predictions.get("earliest_date"))
            and _date(raw["earliest_date"]) < _date(source_predictions["earliest_date"])
        ),
        "source_predictions_support_earlier_than_meta": bool(
            _date(source_predictions.get("earliest_date"))
            and _date(meta_predictions.get("earliest_date"))
            and _date(source_predictions["earliest_date"])
            < _date(meta_predictions["earliest_date"])
        ),
    }


def _rows(
    raw: dict[str, Any],
    adjusted: dict[str, Any],
    source_predictions: dict[str, Any],
    meta_predictions: dict[str, Any],
    canonical: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _row("raw_stooq_parquet", raw),
        _row("adjusted_yahoo_reference", adjusted),
        _row("source_prediction_artifacts", source_predictions),
        _row("meta_auxiliary_predictions", meta_predictions),
        {
            "layer": "canonical_replay",
            "earliest_date": canonical.get("earliest_canonical_replay_date"),
            "latest_date": canonical.get("latest_canonical_replay_date"),
            "rebalance_date_count": canonical.get("rebalance_date_count"),
            "independent_period_count": canonical.get("raw_independent_periods_exact"),
        },
    ]


def _row(layer: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": layer,
        "earliest_date": summary.get("earliest_date"),
        "latest_date": summary.get("latest_date"),
        "rebalance_date_count": summary.get("unique_rebalance_dates"),
        "independent_period_count": None,
    }


def _audit_config(config: dict[str, Any]) -> dict[str, Any]:
    validation = dict(
        config.get("ml", {}).get("benchmark_relative_validation", {}) or {}
    )
    return validation


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            "reports/ml/regime_transformer_meta_ensemble_v1",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "layer",
        "earliest_date",
        "latest_date",
        "rebalance_date_count",
        "independent_period_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload.get("rows", []))


def _markdown(payload: dict[str, Any]) -> str:
    bottleneck = payload.get("historical_bottleneck", {})
    lines = [
        "# Historical Coverage Audit",
        "",
        NOTICE,
        "",
        f"Current bottleneck: {bottleneck.get('limiting_layer')}",
        f"Minimum independent periods: {payload.get('minimum_independent_periods')}",
        f"Full model rerun required: {payload.get('full_model_rerun_required')}",
        "",
        "|layer|earliest|latest|rebalance dates|independent periods|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "|{layer}|{earliest}|{latest}|{rebalance}|{independent}|".format(
                layer=row.get("layer"),
                earliest=row.get("earliest_date"),
                latest=row.get("latest_date"),
                rebalance=row.get("rebalance_date_count"),
                independent=row.get("independent_period_count"),
            )
        )
    lines.extend([
        "",
        "## Blockers",
        "",
        *[f"- {item}" for item in payload.get("blockers", [])],
        "",
        "## Overnight Command",
        "",
        f"`{payload.get('overnight_command_if_rerun_justified')}`",
        "",
    ])
    return "\n".join(lines)


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
