from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.interfaces.data_feed import IDataFeed
from core.research.ml.audits.benchmark_relative_validation import (
    _load_required_closes as _load_benchmark_required_closes,
)
from core.research.ml.audits.benchmark_relative_validation import (
    build_benchmark_relative_validation,
)
from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
DEFAULT_INSPECT_SYMBOLS = ("AMSC", "AXTI", "LEU", "LUMN", "MRVL", "MU")
COMMON_SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0)
REPORT_CANDIDATES = (
    "exact_champion_replay",
    "selected_bayesian_optimizer_diagnostic_policy",
    "spy_buy_and_hold",
    "qqq_buy_and_hold",
    "equal_weight_selected_universe",
)


@dataclass(frozen=True)
class DataAdjustmentAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class CleanDataReplayPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class IndependentPeriodValidationPaths:
    json_path: Path
    markdown_path: Path


def write_data_adjustment_audit(
    config: dict[str, Any],
) -> DataAdjustmentAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    audit_config = _audit_config(config)
    symbols = _symbols_to_audit(canonical, champion_audit, audit_config)
    symbol_rows = _load_stooq_price_rows_by_symbol(
        Path(str(audit_config["stooq_parquet_dir"])),
        symbols,
    )
    payload = build_data_adjustment_audit(
        symbol_rows_by_symbol=symbol_rows,
        canonical_replay=canonical,
        champion_audit=champion_audit,
        audit_config=audit_config,
    )
    paths = DataAdjustmentAuditPaths(
        csv_path=output_dir / "data_adjustment_audit.csv",
        json_path=output_dir / "data_adjustment_audit.json",
        markdown_path=output_dir / "data_adjustment_audit.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_adjustment_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_adjustment_markdown(payload), encoding="utf-8")
    return paths


def write_clean_data_replay(
    config: dict[str, Any],
    data_feed: IDataFeed,
) -> CleanDataReplayPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    selected_optimizer = _read_json(output_dir / "selected_optimizer_exposure_path.json")
    adjustment_audit = _read_json(output_dir / "data_adjustment_audit.json")
    closes = _load_benchmark_required_closes(config, data_feed, canonical)
    payload = build_clean_data_replay(
        canonical_replay=canonical,
        champion_audit=champion_audit,
        selected_optimizer=selected_optimizer,
        adjustment_audit=adjustment_audit,
        closes_by_symbol=closes,
        validation_config=_validation_config(config),
    )
    paths = CleanDataReplayPaths(
        csv_path=output_dir / "clean_data_replay.csv",
        json_path=output_dir / "clean_data_replay.json",
        markdown_path=output_dir / "clean_data_replay.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_clean_replay_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_clean_replay_markdown(payload), encoding="utf-8")
    return paths


def write_independent_period_validation(
    config: dict[str, Any],
) -> IndependentPeriodValidationPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    payload = build_independent_period_validation(
        canonical_replay=canonical,
        validation_config=_validation_config(config),
    )
    paths = IndependentPeriodValidationPaths(
        json_path=output_dir / "independent_period_validation.json",
        markdown_path=output_dir / "independent_period_validation.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(
        _independent_period_markdown(payload),
        encoding="utf-8",
    )
    return paths


def detect_split_like_jumps(
    symbol: str,
    rows: list[dict[str, Any]],
    *,
    suspicious_daily_return_abs: float = 0.50,
    impossible_daily_return_abs: float = 4.0,
    split_ratio_tolerance: float = 0.08,
) -> list[dict[str, Any]]:
    normalized = _normalized_price_rows(rows)
    events: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in normalized:
        close = _number(row.get("close"))
        if close is None or close <= 0.0:
            events.append({
                "symbol": symbol.upper(),
                "date": row.get("date"),
                "previous_date": previous.get("date") if previous else None,
                "previous_close": previous.get("close") if previous else None,
                "close": close,
                "daily_return": None,
                "price_ratio": None,
                "event_type": "impossible_ohlcv",
                "split_like_factor": None,
                "severity": "impossible",
                **RESEARCH_METADATA,
            })
            previous = row if close is not None and close > 0.0 else previous
            continue
        if previous is None:
            previous = row
            continue
        previous_close = _number(previous.get("close"))
        if previous_close is None or previous_close <= 0.0:
            previous = row
            continue
        ratio = close / previous_close
        daily_return = ratio - 1.0
        split_factor = _split_like_factor(ratio, split_ratio_tolerance)
        is_suspicious = abs(daily_return) >= suspicious_daily_return_abs
        is_impossible = abs(daily_return) >= impossible_daily_return_abs
        if split_factor is not None or is_suspicious or is_impossible:
            events.append({
                "symbol": symbol.upper(),
                "date": row.get("date"),
                "previous_date": previous.get("date"),
                "previous_close": previous_close,
                "close": close,
                "daily_return": daily_return,
                "price_ratio": ratio,
                "event_type": (
                    "impossible_daily_jump"
                    if is_impossible
                    else "split_like_jump"
                    if split_factor is not None
                    else "suspicious_daily_jump"
                ),
                "split_like_factor": split_factor,
                "severity": (
                    "impossible"
                    if is_impossible
                    else "suspicious"
                ),
                **RESEARCH_METADATA,
            })
        previous = row
    return events


def build_data_adjustment_audit(
    *,
    symbol_rows_by_symbol: dict[str, list[dict[str, Any]]],
    canonical_replay: dict[str, Any],
    champion_audit: dict[str, Any],
    audit_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _normalize_audit_config(audit_config or {})
    symbol_reports = []
    suspicious_rows = []
    for symbol in sorted(symbol_rows_by_symbol):
        rows = symbol_rows_by_symbol.get(symbol, [])
        report = _symbol_adjustment_report(symbol, rows, config)
        symbol_reports.append(report)
        suspicious_rows.extend(report["suspicious_rows"])
    period_anomalies = _period_anomaly_rows(champion_audit, config)
    suspicious_rebalance_dates = sorted({
        *(
            str(row.get("rebalance_date"))
            for row in period_anomalies
            if row.get("rebalance_date")
        ),
        *_suspicious_rebalance_dates_from_daily_rows(
            canonical_replay,
            suspicious_rows,
        ),
    })
    dependencies = _candidate_suspicious_dependencies(
        canonical_replay,
        suspicious_rows=suspicious_rows,
        period_anomalies=period_anomalies,
    )
    adjusted_status = _overall_adjusted_status(symbol_reports)
    acceptable = _adjusted_status_acceptable(adjusted_status, config)
    return {
        "mode": "stooq_data_adjustment_audit_research_only",
        "data_source": "local Stooq parquet research data",
        "data_path": config["stooq_parquet_dir"],
        "inspect_symbols": sorted(symbol_rows_by_symbol),
        "required_inspection_symbols": list(DEFAULT_INSPECT_SYMBOLS),
        "price_column_used": "close",
        "adjusted_price_status": adjusted_status,
        "adjusted_status": adjusted_status,
        "promotion_gate": {
            "adjusted_price_status_acceptable": acceptable,
            "acceptable_statuses": sorted(config["acceptable_adjusted_price_statuses"]),
            "allow_unknown_adjusted_price_status": config[
                "allow_unknown_adjusted_price_status"
            ],
        },
        "thresholds": {
            "suspicious_daily_return_abs": config["suspicious_daily_return_abs"],
            "impossible_daily_return_abs": config["impossible_daily_return_abs"],
            "large_symbol_period_return_abs": config[
                "large_symbol_period_return_abs"
            ],
            "split_ratio_tolerance": config["split_ratio_tolerance"],
        },
        "symbol_count": len(symbol_reports),
        "suspicious_row_count": len(suspicious_rows),
        "period_anomaly_count": len(period_anomalies),
        "suspicious_rebalance_dates": suspicious_rebalance_dates,
        "suspicious_symbols": sorted({
            str(row.get("symbol"))
            for row in [*suspicious_rows, *period_anomalies]
            if row.get("symbol")
        }),
        "candidate_dependencies": dependencies,
        "symbols": symbol_reports,
        "suspicious_rows": suspicious_rows,
        "period_anomalies": period_anomalies,
        "red_flags": _adjustment_red_flags(
            adjusted_status,
            acceptable,
            suspicious_rows,
            period_anomalies,
            dependencies,
        ),
        **RESEARCH_METADATA,
    }


def build_independent_period_validation(
    *,
    canonical_replay: dict[str, Any],
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config or {}
    minimum = int(config.get("min_independent_periods", 36))
    candidate_rows = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        if not isinstance(candidate, dict):
            continue
        canonical_summary = candidate.get("canonical_continuous_equity", {})
        diagnostic_summary = candidate.get("diagnostic_period_grid", {})
        row_count = int(canonical_summary.get("row_count") or 0)
        candidate_rows[name] = {
            "candidate_name": name,
            "independent_period_count": row_count,
            "diagnostic_period_grid_count": int(
                diagnostic_summary.get("row_count") or 0
            ),
            "period_grid_only_rows": max(
                0,
                int(diagnostic_summary.get("row_count") or 0) - row_count,
            ),
            "start_date": canonical_summary.get("start_date"),
            "end_date": canonical_summary.get("end_date"),
            "last_outcome_end_date": canonical_summary.get("last_outcome_end_date"),
            "passes_minimum": row_count >= minimum,
            **RESEARCH_METADATA,
        }
    exact_count = candidate_rows.get("exact_champion_replay", {}).get(
        "independent_period_count"
    )
    count = int(exact_count if exact_count is not None else min(
        (
            row["independent_period_count"]
            for row in candidate_rows.values()
        ),
        default=0,
    ))
    passed = count >= minimum
    return {
        "mode": "independent_period_validation_research_only",
        "independent_period_definition": (
            "canonical non-overlapping rebalance windows from the exact champion "
            "schedule; overlapping diagnostic period-grid rows do not count"
        ),
        "independent_canonical_period_count": count,
        "minimum_independent_periods": minimum,
        "gate": {
            "name": "minimum_independent_periods",
            "passed": passed,
            "actual": count,
            "minimum": minimum,
        },
        "candidate_periods": candidate_rows,
        "red_flags": [] if passed else ["too_few_independent_periods"],
        **RESEARCH_METADATA,
    }


def build_clean_data_replay(
    *,
    canonical_replay: dict[str, Any],
    champion_audit: dict[str, Any],
    selected_optimizer: dict[str, Any],
    adjustment_audit: dict[str, Any],
    closes_by_symbol: dict[str, dict[str, float]],
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    excluded_dates = set(adjustment_audit.get("suspicious_rebalance_dates", []) or [])
    clean_canonical = build_canonical_replay(
        selected_optimizer=selected_optimizer,
        champion_audit=champion_audit,
        excluded_dates=excluded_dates,
    )
    config = validation_config or {}
    raw_validation = build_benchmark_relative_validation(
        canonical_replay=canonical_replay,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=closes_by_symbol,
        validation_config=config,
    )
    clean_validation = build_benchmark_relative_validation(
        canonical_replay=clean_canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=closes_by_symbol,
        validation_config=config,
    )
    raw_by_name = _validation_candidates_by_name(raw_validation)
    clean_by_name = _validation_candidates_by_name(clean_validation)
    candidates = {}
    for name in REPORT_CANDIDATES:
        raw = raw_by_name.get(name, {})
        clean = clean_by_name.get(name, {})
        raw_return = _number(raw.get("canonical_non_overlap_return"))
        clean_return = _number(clean.get("canonical_non_overlap_return"))
        positive = clean_return is not None and clean_return > float(
            config.get("clean_data_min_return", 0.0)
        )
        relative = bool(clean.get("benchmark_relative_pass", False))
        candidates[name] = {
            "candidate_name": name,
            "available": bool(clean.get("available", False)),
            "raw_canonical_return": raw_return,
            "clean_canonical_return": clean_return,
            "return_delta_clean_vs_raw": (
                clean_return - raw_return
                if clean_return is not None and raw_return is not None
                else None
            ),
            "raw_benchmark_relative_pass": raw.get("benchmark_relative_pass"),
            "clean_benchmark_relative_pass": clean.get("benchmark_relative_pass"),
            "clean_data_return_positive": positive,
            "clean_data_benchmark_relative": relative,
            "clean_data_verdict": (
                "pass" if positive and relative else "blocked"
            ),
            "raw_failed_gates": raw.get("failed_gates", []),
            "clean_failed_gates": clean.get("failed_gates", []),
            "excluded_period_count": _excluded_period_count(
                clean_canonical,
                name,
            ),
            "remaining_period_count": clean.get("canonical_period_count"),
            **RESEARCH_METADATA,
        }
    passing = [
        name for name, row in candidates.items()
        if row["clean_data_verdict"] == "pass"
    ]
    return {
        "mode": "clean_data_replay_research_only",
        "clean_data_definition": (
            "canonical replay after excluding rebalance windows that include "
            "suspicious split-like daily rows or large symbol period anomalies"
        ),
        "excluded_rebalance_dates": sorted(excluded_dates),
        "excluded_rebalance_date_count": len(excluded_dates),
        "adjusted_price_status": adjustment_audit.get("adjusted_price_status"),
        "raw_validation": _validation_summary(raw_validation),
        "clean_validation": _validation_summary(clean_validation),
        "candidates": candidates,
        "promotion_candidates": passing,
        "any_candidate_passes": bool(passing),
        "clean_canonical_replay": clean_canonical,
        "red_flags": [] if passing else ["no_candidate_passes_clean_data_replay"],
        **RESEARCH_METADATA,
    }


def _symbol_adjustment_report(
    symbol: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalized_price_rows(rows)
    suspicious = detect_split_like_jumps(
        symbol,
        normalized,
        suspicious_daily_return_abs=config["suspicious_daily_return_abs"],
        impossible_daily_return_abs=config["impossible_daily_return_abs"],
        split_ratio_tolerance=config["split_ratio_tolerance"],
    )
    raw_adjusted = _raw_adjusted_comparison(normalized)
    adjusted_status = _symbol_adjusted_status(normalized, suspicious, raw_adjusted)
    return {
        "symbol": symbol.upper(),
        "row_count": len(normalized),
        "first_date": normalized[0]["date"] if normalized else None,
        "last_date": normalized[-1]["date"] if normalized else None,
        "columns_present": sorted({
            column
            for row in rows
            for column in row
        }),
        "adjusted_status": adjusted_status,
        "raw_adjusted_comparison": raw_adjusted,
        "suspicious_row_count": len(suspicious),
        "suspicious_rows": suspicious,
        **RESEARCH_METADATA,
    }


def _raw_adjusted_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        row for row in rows
        if _number(row.get("raw_close")) is not None
        and _number(row.get("adjusted_close")) is not None
    ]
    if not comparable:
        return {
            "available": False,
            "reason": "raw and adjusted close columns were not both present",
            "row_count": 0,
        }
    close_matches_raw = 0
    close_matches_adjusted = 0
    raw_adjusted_differ = 0
    for row in comparable:
        close = _number(row.get("close"))
        raw = _number(row.get("raw_close"))
        adjusted = _number(row.get("adjusted_close"))
        if raw is None or adjusted is None:
            continue
        if not _numbers_close(raw, adjusted):
            raw_adjusted_differ += 1
        if close is not None and _numbers_close(close, raw):
            close_matches_raw += 1
        if close is not None and _numbers_close(close, adjusted):
            close_matches_adjusted += 1
    return {
        "available": True,
        "row_count": len(comparable),
        "raw_adjusted_differ_count": raw_adjusted_differ,
        "close_matches_raw_count": close_matches_raw,
        "close_matches_adjusted_count": close_matches_adjusted,
    }


def _symbol_adjusted_status(
    rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    if not rows:
        return "unknown_missing_data"
    if comparison.get("available"):
        if int(comparison.get("raw_adjusted_differ_count") or 0) == 0:
            return "raw_adjusted_identical"
        raw_matches = int(comparison.get("close_matches_raw_count") or 0)
        adjusted_matches = int(comparison.get("close_matches_adjusted_count") or 0)
        if adjusted_matches > raw_matches:
            return "known_adjusted"
        if raw_matches > adjusted_matches:
            return "known_unadjusted"
        return "unknown_close_column_mismatch"
    if any(row.get("event_type") == "split_like_jump" for row in suspicious_rows):
        return "appears_unadjusted"
    return "unknown_no_adjusted_column"


def _overall_adjusted_status(symbol_reports: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("adjusted_status")) for row in symbol_reports}
    if not statuses:
        return "unknown_no_symbols"
    if statuses & {"known_unadjusted", "appears_unadjusted"}:
        return "appears_unadjusted"
    if statuses <= {"known_adjusted", "raw_adjusted_identical"}:
        return "known_adjusted"
    if "known_adjusted" in statuses and not any(status.startswith("unknown") for status in statuses):
        return "appears_adjusted"
    return "unknown"


def _adjusted_status_acceptable(
    status: str,
    config: dict[str, Any],
) -> bool:
    if status in config["acceptable_adjusted_price_statuses"]:
        return True
    return bool(config["allow_unknown_adjusted_price_status"]) and status.startswith(
        "unknown"
    )


def _period_anomaly_rows(
    champion_audit: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(config["large_symbol_period_return_abs"])
    rows = []
    for period in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []:
        if not isinstance(period, dict):
            continue
        for anomaly in period.get("symbol_return_anomalies", []) or []:
            symbol_return = _number(anomaly.get("return"))
            if symbol_return is None or abs(symbol_return) < threshold:
                continue
            rows.append({
                "symbol": str(anomaly.get("symbol", "")).upper(),
                "rebalance_date": period.get("rebalance_date"),
                "outcome_end_date": period.get("outcome_end_date"),
                "start_date": anomaly.get("start_date") or period.get("rebalance_date"),
                "end_date": anomaly.get("end_date") or period.get("outcome_end_date"),
                "start_close": _number(anomaly.get("start_close")),
                "end_close": _number(anomaly.get("end_close")),
                "period_return": symbol_return,
                "event_type": "large_symbol_period_return",
                "severity": "suspicious",
                **RESEARCH_METADATA,
            })
    return rows


def _candidate_suspicious_dependencies(
    canonical_replay: dict[str, Any],
    *,
    suspicious_rows: list[dict[str, Any]],
    period_anomalies: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        rows = [
            row for row in candidate.get("rows", []) or []
            if isinstance(row, dict)
        ]
        dependencies = []
        for row in rows:
            dependencies.extend(_daily_row_dependencies(row, suspicious_rows))
            dependencies.extend(_period_anomaly_dependencies(row, period_anomalies))
        unique = _unique_dependency_rows(dependencies)
        output[str(name)] = {
            "candidate_name": str(name),
            "suspicious_dependency_count": len(unique),
            "suspicious_rebalance_dates": sorted({
                str(row.get("rebalance_date"))
                for row in unique
                if row.get("rebalance_date")
            }),
            "suspicious_symbols": sorted({
                str(row.get("symbol"))
                for row in unique
                if row.get("symbol")
            }),
            "dependencies": unique[:100],
            **RESEARCH_METADATA,
        }
    return output


def _daily_row_dependencies(
    period: dict[str, Any],
    suspicious_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = _date(period.get("rebalance_date"))
    end = _date(period.get("outcome_end_date")) or start
    symbols = {str(symbol).upper() for symbol in period.get("selected_symbols", [])}
    if start is None or end is None or not symbols:
        return []
    output = []
    for event in suspicious_rows:
        event_date = _date(event.get("date"))
        symbol = str(event.get("symbol", "")).upper()
        if symbol not in symbols or event_date is None:
            continue
        if start <= event_date <= end:
            output.append({
                "dependency_type": "daily_price_event",
                "rebalance_date": period.get("rebalance_date"),
                "outcome_end_date": period.get("outcome_end_date"),
                "symbol": symbol,
                "event_date": event.get("date"),
                "event_type": event.get("event_type"),
                "daily_return": event.get("daily_return"),
            })
    return output


def _period_anomaly_dependencies(
    period: dict[str, Any],
    period_anomalies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    period_date = str(period.get("rebalance_date", ""))
    symbols = {str(symbol).upper() for symbol in period.get("selected_symbols", [])}
    output = []
    for anomaly in period_anomalies:
        symbol = str(anomaly.get("symbol", "")).upper()
        if anomaly.get("rebalance_date") == period_date and symbol in symbols:
            output.append({
                "dependency_type": "large_symbol_period_return",
                "rebalance_date": period_date,
                "outcome_end_date": period.get("outcome_end_date"),
                "symbol": symbol,
                "event_date": anomaly.get("end_date"),
                "event_type": anomaly.get("event_type"),
                "period_return": anomaly.get("period_return"),
            })
    return output


def _unique_dependency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (
            row.get("dependency_type"),
            row.get("rebalance_date"),
            row.get("symbol"),
            row.get("event_date"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _suspicious_rebalance_dates_from_daily_rows(
    canonical_replay: dict[str, Any],
    suspicious_rows: list[dict[str, Any]],
) -> set[str]:
    dates = set()
    for candidate in canonical_replay.get("candidates", {}).values():
        for row in candidate.get("rows", []) or []:
            if _daily_row_dependencies(row, suspicious_rows):
                dates.add(str(row.get("rebalance_date")))
    return dates


def _validation_candidates_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("candidate_name")): row
        for row in payload.get("candidates", []) or []
        if isinstance(row, dict)
    }


def _validation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_returns": payload.get("benchmark_returns", {}),
        "promotion_candidates": payload.get("promotion_candidates", []),
        "any_candidate_passes": payload.get("any_candidate_passes", False),
    }


def _excluded_period_count(clean_canonical: dict[str, Any], candidate_name: str) -> int:
    candidate = clean_canonical.get("candidates", {}).get(candidate_name, {})
    return sum(
        1 for row in candidate.get("rows", []) or []
        if row.get("exclusion_reason")
    )


def _adjustment_red_flags(
    adjusted_status: str,
    acceptable: bool,
    suspicious_rows: list[dict[str, Any]],
    period_anomalies: list[dict[str, Any]],
    dependencies: dict[str, dict[str, Any]],
) -> list[str]:
    flags = []
    if not acceptable:
        flags.append("adjusted_price_status_not_acceptable")
    if adjusted_status.startswith("unknown"):
        flags.append("adjusted_price_status_unknown")
    if suspicious_rows:
        flags.append("suspicious_daily_price_rows_present")
    if period_anomalies:
        flags.append("large_symbol_period_anomalies_present")
    if any(
        int(row.get("suspicious_dependency_count") or 0) > 0
        for row in dependencies.values()
    ):
        flags.append("candidate_depends_on_suspicious_rows")
    return sorted(set(flags))


def _symbols_to_audit(
    canonical_replay: dict[str, Any],
    champion_audit: dict[str, Any],
    audit_config: dict[str, Any],
) -> list[str]:
    symbols = {str(symbol).upper() for symbol in audit_config["inspect_symbols"]}
    for candidate in canonical_replay.get("candidates", {}).values():
        for row in candidate.get("rows", []) or []:
            symbols.update(
                str(symbol).upper()
                for symbol in row.get("selected_symbols", []) or []
            )
    for row in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []:
        symbols.update(
            str(symbol).upper()
            for symbol in row.get("selected_symbols", []) or []
        )
        for anomaly in row.get("symbol_return_anomalies", []) or []:
            if anomaly.get("symbol"):
                symbols.add(str(anomaly["symbol"]).upper())
    return sorted(symbol for symbol in symbols if symbol)


def _normalized_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        close = _first_number(row, "close", "Close", "<CLOSE>")
        normalized.append({
            "date": _date_string(_first_present(row, "timestamp", "date", "Date", "<DATE>")),
            "close": close,
            "raw_close": _first_number(
                row,
                "raw_close",
                "RawClose",
                "raw_Close",
                "unadjusted_close",
                "UnadjustedClose",
            ),
            "adjusted_close": _first_number(
                row,
                "adjusted_close",
                "adj_close",
                "Adj Close",
                "AdjClose",
                "adjusted",
            ),
            "open": _first_number(row, "open", "Open", "<OPEN>"),
            "high": _first_number(row, "high", "High", "<HIGH>"),
            "low": _first_number(row, "low", "Low", "<LOW>"),
            "volume": _first_number(row, "volume", "Volume", "<VOL>"),
            **row,
        })
    return sorted(
        [row for row in normalized if row.get("date")],
        key=lambda row: str(row["date"]),
    )


def _split_like_factor(ratio: float, tolerance: float) -> float | None:
    if ratio <= 0.0:
        return None
    for factor in COMMON_SPLIT_FACTORS:
        inverse = 1.0 / factor
        if abs(ratio - factor) / factor <= tolerance:
            return factor
        if abs(ratio - inverse) / inverse <= tolerance:
            return factor
    return None


def _load_stooq_price_rows_by_symbol(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        symbol: _load_stooq_price_rows(data_dir / f"{symbol}.parquet")
        for symbol in symbols
    }


def _load_stooq_price_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Stooq adjustment audit requires pyarrow to read local parquet data"
        ) from exc
    table = pq.read_table(path)
    columns = table.to_pydict()
    names = list(columns)
    row_count = len(columns[names[0]]) if names else 0
    rows = []
    for index in range(row_count):
        rows.append({name: columns[name][index] for name in names})
    return rows


def _normalize_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    validation = dict(config)
    acceptable = validation.get("acceptable_adjusted_price_statuses")
    if acceptable is None:
        acceptable = {
            "known_adjusted",
            "appears_adjusted",
            "raw_adjusted_identical",
        }
    return {
        "stooq_parquet_dir": str(
            validation.get("stooq_parquet_dir", "data/processed/stooq_parquet")
        ),
        "inspect_symbols": [
            str(symbol).upper()
            for symbol in validation.get("inspect_symbols", DEFAULT_INSPECT_SYMBOLS)
        ],
        "suspicious_daily_return_abs": float(
            validation.get("suspicious_daily_return_abs", 0.50)
        ),
        "impossible_daily_return_abs": float(
            validation.get("impossible_daily_return_abs", 4.0)
        ),
        "large_symbol_period_return_abs": float(
            validation.get("large_symbol_period_return_abs", 1.0)
        ),
        "split_ratio_tolerance": float(validation.get("split_ratio_tolerance", 0.08)),
        "acceptable_adjusted_price_statuses": {
            str(status) for status in acceptable
        },
        "allow_unknown_adjusted_price_status": bool(
            validation.get("allow_unknown_adjusted_price_status", False)
        ),
    }


def _audit_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    audit = dict(ml_config.get("data_adjustment_audit", {}) or {})
    validation = dict(ml_config.get("benchmark_relative_validation", {}) or {})
    audit.setdefault(
        "stooq_parquet_dir",
        ml_config.get("stooq_parquet_dir", "data/processed/stooq_parquet"),
    )
    for key in (
        "acceptable_adjusted_price_statuses",
        "allow_unknown_adjusted_price_status",
    ):
        if key in validation:
            audit[key] = validation[key]
    return _normalize_audit_config(audit)


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml", {}).get("benchmark_relative_validation", {}) or {})


def _output_dir(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_adjustment_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "symbol",
        "date",
        "previous_date",
        "event_type",
        "severity",
        "daily_return",
        "price_ratio",
        "split_like_factor",
        "adjusted_status",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in payload.get("symbols", []):
            rows = report.get("suspicious_rows", []) or [{
                "symbol": report.get("symbol"),
                "adjusted_status": report.get("adjusted_status"),
            }]
            for row in rows:
                writer.writerow({
                    **{name: row.get(name) for name in fieldnames},
                    "adjusted_status": report.get("adjusted_status"),
                    **RESEARCH_METADATA,
                })


def _write_clean_replay_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "candidate_name",
        "available",
        "raw_canonical_return",
        "clean_canonical_return",
        "return_delta_clean_vs_raw",
        "raw_benchmark_relative_pass",
        "clean_benchmark_relative_pass",
        "clean_data_return_positive",
        "clean_data_benchmark_relative",
        "clean_data_verdict",
        "excluded_period_count",
        "remaining_period_count",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("candidates", {}).values():
            writer.writerow({name: row.get(name) for name in fieldnames})


def _adjustment_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Data Adjustment Audit",
        "",
        NOTICE,
        "",
        f"Adjusted price status: {payload.get('adjusted_price_status')}",
        f"Suspicious rows: {payload.get('suspicious_row_count', 0)}",
        f"Suspicious rebalance dates: {len(payload.get('suspicious_rebalance_dates', []))}",
        "",
        "|symbol|status|rows|suspicious rows|first date|last date|",
        "|---|---|---:|---:|---|---|",
    ]
    for row in payload.get("symbols", []):
        lines.append(
            "|{symbol}|{status}|{rows}|{suspicious}|{first}|{last}|".format(
                symbol=row.get("symbol"),
                status=row.get("adjusted_status"),
                rows=row.get("row_count"),
                suspicious=row.get("suspicious_row_count"),
                first=row.get("first_date") or "",
                last=row.get("last_date") or "",
            )
        )
    lines.extend(["", "## Red Flags", ""])
    lines.extend(f"- {flag}" for flag in payload.get("red_flags", []))
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _clean_replay_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Clean Data Replay",
        "",
        NOTICE,
        "",
        f"Excluded rebalance dates: {payload.get('excluded_rebalance_date_count', 0)}",
        "",
        "|candidate|raw return|clean return|delta|clean benchmark-relative|verdict|",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload.get("candidates", {}).values():
        lines.append(
            "|{name}|{raw}|{clean}|{delta}|{relative}|{verdict}|".format(
                name=row.get("candidate_name"),
                raw=_fmt(row.get("raw_canonical_return")),
                clean=_fmt(row.get("clean_canonical_return")),
                delta=_fmt(row.get("return_delta_clean_vs_raw")),
                relative=row.get("clean_benchmark_relative_pass"),
                verdict=row.get("clean_data_verdict"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _independent_period_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Independent Period Validation",
        "",
        NOTICE,
        "",
        "Independent canonical periods: "
        f"{payload.get('independent_canonical_period_count', 0)}",
        "Minimum required: "
        f"{payload.get('minimum_independent_periods', 0)}",
        f"Gate passed: {payload.get('gate', {}).get('passed', False)}",
        "",
        "|candidate|independent periods|diagnostic rows|period-grid only rows|passes|",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload.get("candidate_periods", {}).values():
        lines.append(
            "|{name}|{count}|{diagnostic}|{extra}|{passes}|".format(
                name=row.get("candidate_name"),
                count=row.get("independent_period_count"),
                diagnostic=row.get("diagnostic_period_grid_count"),
                extra=row.get("period_grid_only_rows"),
                passes=row.get("passes_minimum"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _date_string(value: Any) -> str | None:
    parsed = _date(value)
    return parsed.date().isoformat() if parsed is not None else None


def _first_present(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in {None, ""}:
            return row[name]
    return None


def _first_number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return False
    return math.isclose(left_number, right_number, rel_tol=tolerance, abs_tol=tolerance)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
