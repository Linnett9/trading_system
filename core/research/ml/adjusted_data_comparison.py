from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.research.ml.benchmark_relative_validation import (
    build_benchmark_relative_validation,
)
from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)
from infrastructure.data.adjusted_price_csv_data_feed import (
    AdjustedPricePoint,
    LocalAdjustedPriceCsvDataFeed,
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
class AdjustedDataComparisonPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class AdjustedPriceReplayPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_adjusted_data_comparison(
    config: dict[str, Any],
) -> AdjustedDataComparisonPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    comparison_config = _comparison_config(config)
    symbols = _symbols_to_compare(canonical, comparison_config)
    raw_rows = _load_raw_stooq_rows_by_symbol(
        Path(str(comparison_config["stooq_parquet_dir"])),
        symbols,
    )
    adjusted_rows = _load_adjusted_rows_by_symbol(comparison_config, symbols)
    payload = build_adjusted_data_comparison(
        raw_rows_by_symbol=raw_rows,
        adjusted_rows_by_symbol=adjusted_rows,
        canonical_replay=canonical,
        comparison_config=comparison_config,
    )
    paths = AdjustedDataComparisonPaths(
        csv_path=output_dir / "adjusted_data_comparison.csv",
        json_path=output_dir / "adjusted_data_comparison.json",
        markdown_path=output_dir / "adjusted_data_comparison.md",
    )
    paths.json_path.write_text(
        json.dumps(_comparison_json_payload(payload), indent=2),
        encoding="utf-8",
    )
    _write_comparison_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_comparison_markdown(payload), encoding="utf-8")
    return paths


def write_adjusted_price_replay(
    config: dict[str, Any],
) -> AdjustedPriceReplayPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    selected_optimizer = _read_json(output_dir / "selected_optimizer_exposure_path.json")
    comparison = _read_json(output_dir / "adjusted_data_comparison.json")
    comparison_config = _comparison_config(config)
    symbols = _symbols_to_compare(canonical, comparison_config)
    raw_rows = _load_raw_stooq_rows_by_symbol(
        Path(str(comparison_config["stooq_parquet_dir"])),
        symbols,
    )
    adjusted_rows = _load_adjusted_rows_by_symbol(comparison_config, symbols)
    payload = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion_audit,
        selected_optimizer=selected_optimizer,
        adjusted_comparison=comparison,
        raw_closes_by_symbol={
            symbol: _raw_close_by_date(rows)
            for symbol, rows in raw_rows.items()
        },
        adjusted_closes_by_symbol={
            symbol: _adjusted_close_by_date(rows)
            for symbol, rows in adjusted_rows.items()
        },
        validation_config=_validation_config(config),
    )
    paths = AdjustedPriceReplayPaths(
        csv_path=output_dir / "adjusted_price_replay.csv",
        json_path=output_dir / "adjusted_price_replay.json",
        markdown_path=output_dir / "adjusted_price_replay.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_replay_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_replay_markdown(payload), encoding="utf-8")
    return paths


def build_adjusted_data_comparison(
    *,
    raw_rows_by_symbol: dict[str, list[dict[str, Any]]],
    adjusted_rows_by_symbol: dict[str, list[AdjustedPricePoint | dict[str, Any]]],
    canonical_replay: dict[str, Any],
    comparison_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _normalize_comparison_config(comparison_config or {})
    rows = []
    symbol_reports = []
    for symbol in sorted(raw_rows_by_symbol):
        raw_by_date = _raw_close_by_date(raw_rows_by_symbol.get(symbol, []))
        adjusted_by_date = _adjusted_close_by_date(
            adjusted_rows_by_symbol.get(symbol, [])
        )
        symbol_rows = _comparison_rows_for_symbol(
            symbol,
            raw_by_date,
            adjusted_by_date,
            config,
        )
        rows.extend(symbol_rows)
        symbol_reports.append(_symbol_report(symbol, raw_by_date, adjusted_by_date, symbol_rows))
    dependencies = _candidate_distortion_dependencies(canonical_replay, rows)
    required = set(config["inspect_symbols"])
    required_available = [
        row for row in symbol_reports
        if row["symbol"] in required and row["adjusted_source_available"]
    ]
    source_status = (
        "available"
        if len(required_available) == len(required)
        else "partial"
        if required_available
        else "missing"
    )
    acceptable = source_status == "available"
    anomaly_survival = _anomaly_survival_by_symbol(symbol_reports, required)
    return {
        "mode": "adjusted_data_comparison_research_only",
        "raw_source": {
            "name": "stooq_parquet_close",
            "path": config["stooq_parquet_dir"],
            "preserved_separately": True,
        },
        "adjusted_source": {
            "name": config["adjusted_source_name"],
            "data_dir": config["adjusted_data_dir"],
            "combined_path": config.get("adjusted_combined_path"),
            "available_status": source_status,
            "acceptable": acceptable,
        },
        "promotion_gate": {
            "adjusted_source_available_and_acceptable": acceptable,
        },
        "inspect_symbols": list(config["inspect_symbols"]),
        "symbol_count": len(symbol_reports),
        "comparison_row_count": len(rows),
        "split_like_distortion_count": sum(
            bool(row.get("split_like_distortion")) for row in rows
        ),
        "candidate_dependencies": dependencies,
        "anomaly_survival_by_symbol": anomaly_survival,
        "symbols": symbol_reports,
        "rows": rows,
        "red_flags": _comparison_red_flags(source_status, rows, dependencies),
        **RESEARCH_METADATA,
    }


def build_adjusted_price_replay(
    *,
    canonical_replay: dict[str, Any],
    champion_audit: dict[str, Any],
    selected_optimizer: dict[str, Any],
    adjusted_comparison: dict[str, Any],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    raw_closes_by_symbol: dict[str, dict[str, float]] | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config or {}
    replay_config = _adjusted_replay_config(config)
    raw_closes_by_symbol = raw_closes_by_symbol or {}
    source = adjusted_comparison.get("adjusted_source", {})
    source_available = bool(source.get("acceptable", False))
    coverage_by_candidate: dict[str, dict[str, Any]] = {}
    if source_available:
        adjusted_champion, champion_coverage = _adjusted_champion_audit(
            champion_audit,
            adjusted_closes_by_symbol,
            raw_closes_by_symbol,
            replay_config,
        )
        adjusted_optimizer, optimizer_coverage = _adjusted_selected_optimizer(
            selected_optimizer,
            champion_audit,
            adjusted_champion,
            adjusted_closes_by_symbol,
            raw_closes_by_symbol,
            replay_config,
        )
        adjusted_canonical = build_canonical_replay(
            selected_optimizer=adjusted_optimizer,
            champion_audit=adjusted_champion,
        )
        coverage_by_candidate = {
            "exact_champion_replay": champion_coverage,
            "selected_bayesian_optimizer_diagnostic_policy": optimizer_coverage,
        }
        _attach_adjusted_independent_counts(coverage_by_candidate, adjusted_canonical)
        validation = build_benchmark_relative_validation(
            canonical_replay=adjusted_canonical,
            anomaly_report={"flagged_rebalance_dates": []},
            closes_by_symbol=adjusted_closes_by_symbol,
            validation_config=config,
        )
    else:
        adjusted_canonical = {}
        validation = build_benchmark_relative_validation(
            canonical_replay=canonical_replay,
            anomaly_report={"flagged_rebalance_dates": []},
            closes_by_symbol=adjusted_closes_by_symbol,
            validation_config=config,
        )
    by_name = {
        row.get("candidate_name"): row
        for row in validation.get("candidates", []) or []
        if isinstance(row, dict)
    }
    candidates = {}
    for name in REPORT_CANDIDATES:
        row = by_name.get(name, {})
        coverage = coverage_by_candidate.get(name, _empty_coverage_summary(name))
        full_coverage = bool(coverage.get("adjusted_full_symbol_coverage", True))
        periods_valid = int(coverage.get("invalid_period_count") or 0) == 0
        coverage_ok = _candidate_coverage_ok(coverage, replay_config)
        independent_ok = _valid_adjusted_independent_periods_ok(coverage, replay_config)
        adjusted_return_candidate = (
            _number(row.get("canonical_non_overlap_return"))
            if source_available
            else None
        )
        adjusted_return = (
            adjusted_return_candidate
            if coverage_ok and independent_ok
            else None
        )
        positive = adjusted_return is not None and adjusted_return > float(
            config.get("clean_data_min_return", 0.0)
        )
        benchmark_relative = bool(row.get("benchmark_relative_pass", False))
        failed_gates = list(row.get("failed_gates", []))
        if not full_coverage:
            failed_gates.append("adjusted_full_symbol_coverage")
        if not periods_valid:
            failed_gates.append("adjusted_replay_valid_periods")
        if int(coverage.get("empty_selection_with_positive_exposure_count") or 0) > 0:
            failed_gates.append("empty_selection_with_positive_exposure")
        if not independent_ok:
            failed_gates.append("minimum_adjusted_independent_periods")
        candidates[name] = {
            "candidate_name": name,
            "available": (
                bool(row.get("available", False))
                and source_available
                and coverage_ok
                and independent_ok
            ),
            "adjusted_canonical_return": adjusted_return,
            "coverage_valid_adjusted_canonical_return": adjusted_return_candidate,
            "adjusted_benchmark_relative_pass": benchmark_relative,
            "adjusted_price_return_positive": positive,
            "adjusted_price_replay_verdict": (
                "pass"
                if (
                    source_available
                    and coverage_ok
                    and full_coverage
                    and independent_ok
                    and positive
                    and benchmark_relative
                )
                else "blocked"
            ),
            "failed_gates": sorted(set(failed_gates)),
            "adjusted_coverage_ratio": coverage.get("adjusted_coverage_ratio"),
            "missing_adjusted_symbols": coverage.get("missing_adjusted_symbols", []),
            "missing_symbols": coverage.get("missing_adjusted_symbols", []),
            "raw_fallback_symbols": coverage.get("raw_fallback_symbols", []),
            "empty_selection_with_positive_exposure_count": coverage.get(
                "empty_selection_with_positive_exposure_count",
                0,
            ),
            "affected_dates": coverage.get(
                "empty_selection_with_positive_exposure_dates",
                [],
            ),
            "empty_selection_with_positive_exposure_dates": coverage.get(
                "empty_selection_with_positive_exposure_dates",
                [],
            ),
            "empty_selection_resolution": coverage.get(
                "empty_selection_resolution",
                "unchanged",
            ),
            "invalid_period_count": coverage.get("invalid_period_count", 0),
            "invalid_adjusted_period_count": coverage.get(
                "invalid_adjusted_period_count",
                0,
            ),
            "valid_period_count": coverage.get("valid_period_count", 0),
            "valid_adjusted_period_count": coverage.get(
                "valid_adjusted_period_count",
                0,
            ),
            "valid_adjusted_independent_period_count": coverage.get(
                "valid_adjusted_independent_period_count",
                0,
            ),
            "minimum_adjusted_independent_periods": replay_config[
                "min_independent_periods"
            ],
            "minimum_adjusted_independent_periods_pass": independent_ok,
            "adjusted_full_symbol_coverage": full_coverage,
            "adjusted_replay_valid_periods": periods_valid,
            "fail_closed_reason": _fail_closed_reason(
                coverage,
                coverage_ok=coverage_ok,
                independent_ok=independent_ok,
            ),
            "coverage": coverage,
            **RESEARCH_METADATA,
        }
    passing = [
        name for name, row in candidates.items()
        if row["adjusted_price_replay_verdict"] == "pass"
    ]
    return {
        "mode": "adjusted_price_replay_research_only",
        "replay_semantics": (
            "canonical selected-symbol windows recomputed from adjusted close "
            "only when every selected symbol has adjusted start/end prices; "
            "raw Stooq OHLCV remains unchanged"
        ),
        "coverage_rules": replay_config,
        "adjusted_source_available": source_available,
        "adjusted_source_status": source.get("available_status", "missing"),
        "anomaly_survival_by_symbol": adjusted_comparison.get(
            "anomaly_survival_by_symbol",
            {},
        ),
        "adjusted_validation": validation,
        "adjusted_canonical_replay": adjusted_canonical,
        "candidates": candidates,
        "promotion_candidates": passing,
        "any_candidate_passes": bool(passing),
        "red_flags": _adjusted_replay_red_flags(candidates, passing),
        **RESEARCH_METADATA,
    }


def detect_split_like_adjustment_ratio(
    previous_ratio: float | None,
    current_ratio: float | None,
    *,
    tolerance: float = 0.08,
) -> float | None:
    if previous_ratio is None or current_ratio is None:
        return None
    if previous_ratio <= 0.0 or current_ratio <= 0.0:
        return None
    ratio_change = current_ratio / previous_ratio
    return _split_like_factor(ratio_change, tolerance)


def _comparison_rows_for_symbol(
    symbol: str,
    raw_by_date: dict[str, float],
    adjusted_by_date: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    previous: dict[str, Any] | None = None
    for day in sorted(set(raw_by_date) | set(adjusted_by_date)):
        raw_close = raw_by_date.get(day)
        adjusted_close = adjusted_by_date.get(day)
        adjustment_ratio = (
            adjusted_close / raw_close
            if raw_close is not None and adjusted_close is not None and raw_close > 0
            else None
        )
        raw_daily = _daily_return(previous, "raw_close", raw_close)
        adjusted_daily = _daily_return(previous, "adjusted_close", adjusted_close)
        ratio_factor = detect_split_like_adjustment_ratio(
            _number((previous or {}).get("adjustment_ratio")),
            adjustment_ratio,
            tolerance=config["split_ratio_tolerance"],
        )
        raw_factor = _split_like_return_factor(
            raw_daily,
            config["split_ratio_tolerance"],
        )
        adjusted_factor = _split_like_return_factor(
            adjusted_daily,
            config["split_ratio_tolerance"],
        )
        raw_suspicious = _is_suspicious_return(
            raw_daily,
            config["suspicious_daily_return_abs"],
        )
        adjusted_suspicious = _is_suspicious_return(
            adjusted_daily,
            config["suspicious_daily_return_abs"],
        )
        split_like_distortion = bool(
            raw_suspicious
            and not adjusted_suspicious
            and (ratio_factor is not None or raw_factor is not None)
        )
        row = {
            "symbol": symbol.upper(),
            "date": day,
            "raw_close": raw_close,
            "adjusted_close": adjusted_close,
            "adjustment_ratio": adjustment_ratio,
            "raw_daily_return": raw_daily,
            "adjusted_daily_return": adjusted_daily,
            "raw_split_like_factor": raw_factor,
            "adjusted_split_like_factor": adjusted_factor,
            "adjustment_ratio_split_like_factor": ratio_factor,
            "raw_suspicious_jump": raw_suspicious,
            "adjusted_suspicious_jump": adjusted_suspicious,
            "split_like_distortion": split_like_distortion,
            "anomaly_survives_adjustment": bool(
                raw_suspicious and (adjusted_suspicious or adjusted_close is None)
            ),
            **RESEARCH_METADATA,
        }
        rows.append(row)
        previous = row
    return rows


def _symbol_report(
    symbol: str,
    raw_by_date: dict[str, float],
    adjusted_by_date: dict[str, float],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    comparable = [
        row for row in rows
        if row.get("raw_close") is not None and row.get("adjusted_close") is not None
    ]
    return {
        "symbol": symbol.upper(),
        "raw_source_available": bool(raw_by_date),
        "adjusted_source_available": bool(adjusted_by_date),
        "raw_row_count": len(raw_by_date),
        "adjusted_row_count": len(adjusted_by_date),
        "comparable_row_count": len(comparable),
        "first_comparable_date": comparable[0]["date"] if comparable else None,
        "last_comparable_date": comparable[-1]["date"] if comparable else None,
        "split_like_distortion_count": sum(
            bool(row.get("split_like_distortion")) for row in rows
        ),
        "anomaly_survives_adjustment_count": sum(
            bool(row.get("anomaly_survives_adjustment")) for row in rows
        ),
        **RESEARCH_METADATA,
    }


def _adjusted_champion_audit(
    champion_audit: dict[str, Any],
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    period_rows = []
    coverage_rows = []
    for row in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []:
        if not isinstance(row, dict):
            continue
        coverage = _period_adjusted_coverage(
            row,
            adjusted_closes,
            raw_closes,
            config,
        )
        coverage_rows.append(coverage)
        if not coverage["valid_adjusted_period"]:
            continue
        adjusted_return = _weighted_period_return(
            row,
            adjusted_closes,
            raw_closes,
            config,
        )
        if adjusted_return is None:
            continue
        period_rows.append({
            **row,
            "period_return": adjusted_return * float(row.get("exposure_target") or 1.0),
            "adjusted_symbol_period_return": adjusted_return,
            "symbol_return_anomalies": [],
        })
    return (
        {
            **champion_audit,
            "exact_champion_replay": {
                **champion_audit.get("exact_champion_replay", {}),
                "period_rows": period_rows,
            },
        },
        _coverage_summary("exact_champion_replay", coverage_rows, config),
    )


def _adjusted_selected_optimizer(
    selected_optimizer: dict[str, Any],
    champion_audit: dict[str, Any],
    adjusted_champion: dict[str, Any],
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_period_by_date = {
        str(row.get("rebalance_date")): row
        for row in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []
        if isinstance(row, dict)
    }
    adjusted_period_by_date = {
        str(row.get("rebalance_date")): row
        for row in adjusted_champion.get("exact_champion_replay", {}).get("period_rows", []) or []
    }
    rows = []
    coverage_rows = []
    for row in selected_optimizer.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        raw_period = raw_period_by_date.get(str(row.get("rebalance_date")))
        if not raw_period:
            continue
        exposure = _number(row.get("exposure"))
        coverage_period = {
            **raw_period,
            "exposure": exposure,
            "selected_symbols": raw_period.get("selected_symbols", []) or [],
            "target_weights": raw_period.get("target_weights", {}) or {},
        }
        coverage = _period_adjusted_coverage(
            coverage_period,
            adjusted_closes,
            raw_closes,
            config,
        )
        coverage_rows.append(coverage)
        if not coverage["valid_adjusted_period"]:
            continue
        period = adjusted_period_by_date.get(str(row.get("rebalance_date")))
        if not period:
            continue
        adjusted_return = _weighted_period_return(
            period,
            adjusted_closes,
            raw_closes,
            config,
        )
        if adjusted_return is None or exposure is None:
            continue
        cost = _number(row.get("cost")) or 0.0
        rows.append({
            **row,
            "period_return": adjusted_return,
            "net_return": adjusted_return * exposure - cost,
        })
    return (
        {**selected_optimizer, "rows": rows},
        _coverage_summary(
            "selected_bayesian_optimizer_diagnostic_policy",
            coverage_rows,
            config,
        ),
    )


def _weighted_period_return(
    row: dict[str, Any],
    closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]] | None = None,
    config: dict[str, Any] | None = None,
) -> float | None:
    raw_closes = raw_closes or {}
    config = config or _adjusted_replay_config({})
    symbols = [str(symbol).upper() for symbol in row.get("selected_symbols", [])]
    if not symbols:
        return None
    raw_weights = {
        str(symbol).upper(): _number(weight)
        for symbol, weight in (row.get("target_weights", {}) or {}).items()
    }
    returns = {}
    for symbol in symbols:
        start = _replay_close(
            closes,
            raw_closes,
            symbol,
            str(row.get("rebalance_date", "")),
            config,
        )
        end = _replay_close(
            closes,
            raw_closes,
            symbol,
            str(row.get("outcome_end_date", "")),
            config,
        )
        if start is None or end is None or start <= 0:
            return None
        returns[symbol] = (end / start) - 1.0
    if len(returns) != len(symbols):
        return None
    weights = {
        symbol: raw_weights.get(symbol)
        for symbol in returns
        if raw_weights.get(symbol) is not None
    }
    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0.0:
        weights = {symbol: 1.0 / len(returns) for symbol in returns}
    else:
        weights = {
            symbol: float(weight) / total_weight
            for symbol, weight in weights.items()
        }
    return sum(returns[symbol] * weights[symbol] for symbol in weights)


def _period_adjusted_coverage(
    row: dict[str, Any],
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    symbols = [str(symbol).upper() for symbol in row.get("selected_symbols", [])]
    start_date = str(row.get("rebalance_date", ""))
    end_date = str(row.get("outcome_end_date", ""))
    exposure = _number(row.get("exposure"))
    empty_selection_with_positive_exposure = (
        exposure is not None and exposure > 0.0 and not symbols
    )
    missing = []
    covered = []
    raw_fallback = []
    unresolved = []
    for symbol in symbols:
        values = adjusted_closes.get(symbol, {})
        start = values.get(start_date)
        end = values.get(end_date)
        if start is None or end is None or start <= 0:
            missing.append(symbol)
            if _raw_fallback_available(raw_closes, symbol, start_date, end_date):
                raw_fallback.append(symbol)
            else:
                unresolved.append(symbol)
        else:
            covered.append(symbol)
    coverage_ratio = len(covered) / len(symbols) if symbols else 0.0
    required = float(config["required_adjusted_coverage_ratio"])
    valid = _period_valid_under_policy(
        symbols=symbols,
        missing_symbols=missing,
        unresolved_symbols=unresolved,
        coverage_ratio=coverage_ratio,
        required_coverage_ratio=required,
        config=config,
        empty_selection_with_positive_exposure=empty_selection_with_positive_exposure,
    )
    return {
        "rebalance_date": start_date,
        "outcome_end_date": end_date,
        "exposure": exposure,
        "selected_symbols": symbols,
        "selected_symbol_count": len(symbols),
        "covered_adjusted_symbol_count": len(covered),
        "missing_adjusted_symbols": missing,
        "raw_fallback_symbols": raw_fallback if config["allow_raw_fallback"] else [],
        "unresolved_missing_symbols": unresolved,
        "adjusted_coverage_ratio": coverage_ratio,
        "required_adjusted_coverage_ratio": required,
        "missing_symbol_policy": config["missing_symbol_policy"],
        "valid_adjusted_period": valid,
        "empty_selection_with_positive_exposure": (
            empty_selection_with_positive_exposure
        ),
        "empty_selection_resolution": (
            "invalidated" if empty_selection_with_positive_exposure else None
        ),
        "fail_closed_reason": None
        if valid
        else _period_fail_closed_reason(
            missing,
            unresolved,
            symbols,
            config,
            empty_selection_with_positive_exposure=(
                empty_selection_with_positive_exposure
            ),
        ),
        **RESEARCH_METADATA,
    }


def _coverage_summary(
    candidate_name: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    total_symbols = sum(int(row.get("selected_symbol_count") or 0) for row in rows)
    covered_symbols = sum(
        int(row.get("covered_adjusted_symbol_count") or 0) for row in rows
    )
    invalid = [row for row in rows if not row.get("valid_adjusted_period")]
    valid = [row for row in rows if row.get("valid_adjusted_period")]
    missing_periods = [
        row for row in rows
        if row.get("missing_adjusted_symbols")
    ]
    missing_symbols = sorted({
        symbol
        for row in missing_periods
        for symbol in row.get("missing_adjusted_symbols", []) or []
    })
    empty_selection_periods = [
        row for row in rows
        if row.get("empty_selection_with_positive_exposure")
    ]
    raw_fallback_symbols = sorted({
        symbol
        for row in rows
        for symbol in row.get("raw_fallback_symbols", []) or []
    })
    coverage_ratio = covered_symbols / total_symbols if total_symbols else 0.0
    full_adjusted_coverage = not missing_periods
    return {
        "candidate_name": candidate_name,
        "missing_symbol_policy": config["missing_symbol_policy"],
        "require_full_adjusted_coverage": bool(
            config["require_full_adjusted_coverage"]
        ),
        "allow_raw_fallback": bool(config["allow_raw_fallback"]),
        "required_adjusted_coverage_ratio": float(
            config["required_adjusted_coverage_ratio"]
        ),
        "adjusted_coverage_ratio": coverage_ratio,
        "adjusted_full_symbol_coverage": full_adjusted_coverage,
        "raw_fallback_symbols": raw_fallback_symbols,
        "missing_adjusted_symbols": missing_symbols,
        "empty_selection_with_positive_exposure_count": len(
            empty_selection_periods
        ),
        "empty_selection_with_positive_exposure_dates": [
            row.get("rebalance_date") for row in empty_selection_periods
        ],
        "empty_selection_resolution": (
            "invalidated" if empty_selection_periods else "unchanged"
        ),
        "invalid_period_count": len(invalid),
        "invalid_adjusted_period_count": len(invalid),
        "valid_period_count": len(valid),
        "valid_adjusted_period_count": len(valid),
        "valid_adjusted_independent_period_count": 0,
        "fail_closed_reason": _coverage_fail_closed_reason(invalid),
        "periods": rows,
        **RESEARCH_METADATA,
    }


def _coverage_fail_closed_reason(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    reasons = sorted({
        str(row.get("fail_closed_reason"))
        for row in rows
        if row.get("fail_closed_reason")
    })
    if reasons == ["empty_selection_with_positive_exposure"]:
        return "empty_selection_with_positive_exposure"
    if "empty_selection_with_positive_exposure" in reasons:
        return "multiple_fail_closed_reasons"
    return "missing_adjusted_prices_for_selected_symbols"


def _period_fail_closed_reason(
    missing_symbols: list[str],
    unresolved_symbols: list[str],
    symbols: list[str],
    config: dict[str, Any],
    *,
    empty_selection_with_positive_exposure: bool = False,
) -> str:
    if empty_selection_with_positive_exposure:
        return "empty_selection_with_positive_exposure"
    if not symbols:
        return "no_selected_symbols"
    if config["missing_symbol_policy"] == "skip_period" and missing_symbols:
        return "missing_adjusted_prices_skip_period"
    if missing_symbols and config["missing_symbol_policy"] == "fail_closed":
        return "missing_adjusted_prices_raw_fallback_disabled"
    if unresolved_symbols:
        return "missing_adjusted_and_raw_fallback_prices"
    if missing_symbols:
        return "missing_adjusted_prices"
    return "adjusted_coverage_below_required_ratio"


def _period_valid_under_policy(
    *,
    symbols: list[str],
    missing_symbols: list[str],
    unresolved_symbols: list[str],
    coverage_ratio: float,
    required_coverage_ratio: float,
    config: dict[str, Any],
    empty_selection_with_positive_exposure: bool = False,
) -> bool:
    if empty_selection_with_positive_exposure:
        return False
    if not symbols:
        return False
    policy = config["missing_symbol_policy"]
    if policy == "fallback_raw":
        return not unresolved_symbols
    if policy == "skip_period":
        return not missing_symbols and coverage_ratio >= required_coverage_ratio
    return not missing_symbols and coverage_ratio >= required_coverage_ratio


def _raw_fallback_available(
    raw_closes: dict[str, dict[str, float]],
    symbol: str,
    start_date: str,
    end_date: str,
) -> bool:
    start = raw_closes.get(symbol, {}).get(start_date)
    end = raw_closes.get(symbol, {}).get(end_date)
    return start is not None and end is not None and start > 0


def _replay_close(
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    symbol: str,
    day: str,
    config: dict[str, Any],
) -> float | None:
    adjusted = adjusted_closes.get(symbol, {}).get(day)
    if adjusted is not None and adjusted > 0:
        return adjusted
    if config["missing_symbol_policy"] != "fallback_raw":
        return None
    raw = raw_closes.get(symbol, {}).get(day)
    if raw is not None and raw > 0:
        return raw
    return None


def _attach_adjusted_independent_counts(
    coverage_by_candidate: dict[str, dict[str, Any]],
    adjusted_canonical: dict[str, Any],
) -> None:
    for name, coverage in coverage_by_candidate.items():
        rows = (
            adjusted_canonical.get("candidates", {})
            .get(name, {})
            .get("rows", [])
            or []
        )
        coverage["valid_adjusted_independent_period_count"] = sum(
            bool(row.get("included_in_canonical")) for row in rows
        )


def _empty_coverage_summary(candidate_name: str) -> dict[str, Any]:
    return {
        "candidate_name": candidate_name,
        "missing_symbol_policy": "fail_closed",
        "require_full_adjusted_coverage": False,
        "allow_raw_fallback": False,
        "required_adjusted_coverage_ratio": 1.0,
        "adjusted_coverage_ratio": 1.0,
        "adjusted_full_symbol_coverage": True,
        "raw_fallback_symbols": [],
        "missing_adjusted_symbols": [],
        "invalid_period_count": 0,
        "invalid_adjusted_period_count": 0,
        "valid_period_count": 0,
        "valid_adjusted_period_count": 0,
        "valid_adjusted_independent_period_count": 0,
        "fail_closed_reason": None,
        "periods": [],
        **RESEARCH_METADATA,
    }


def _valid_adjusted_independent_periods_ok(
    coverage: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    if not coverage.get("periods"):
        return True
    actual = int(coverage.get("valid_adjusted_independent_period_count") or 0)
    return actual >= int(config["min_independent_periods"])


def _fail_closed_reason(
    coverage: dict[str, Any],
    *,
    coverage_ok: bool,
    independent_ok: bool,
) -> str | None:
    if not coverage_ok:
        return str(
            coverage.get("fail_closed_reason")
            or "missing_adjusted_prices_for_selected_symbols"
        )
    if not independent_ok:
        return "valid_adjusted_independent_periods_below_minimum"
    return None


def _candidate_coverage_ok(
    coverage: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    if int(coverage.get("empty_selection_with_positive_exposure_count") or 0) > 0:
        return False
    policy = config["missing_symbol_policy"]
    if policy == "fail_closed":
        return bool(coverage.get("adjusted_full_symbol_coverage", True))
    if policy == "fallback_raw":
        return int(coverage.get("invalid_period_count") or 0) == 0
    if policy == "skip_period":
        return True
    return False


def _adjusted_replay_config(config: dict[str, Any]) -> dict[str, Any]:
    replay = dict(config.get("adjusted_replay", {}) or {})
    policy = str(
        replay.get(
            "missing_symbol_policy",
            config.get("missing_symbol_policy", "fail_closed"),
        )
    )
    if policy not in {"fail_closed", "fallback_raw", "skip_period"}:
        raise ValueError(f"Unsupported adjusted replay missing_symbol_policy: {policy}")
    require_full = bool(
        replay.get(
            "require_full_adjusted_coverage",
            config.get("require_full_adjusted_coverage", True),
        )
    )
    if policy == "fail_closed":
        require_full = True
    return {
        "missing_symbol_policy": policy,
        "require_full_adjusted_coverage": require_full,
        "allow_raw_fallback": policy == "fallback_raw",
        "required_adjusted_coverage_ratio": (
            1.0
            if require_full
            else float(
                replay.get(
                    "required_adjusted_coverage_ratio",
                    config.get("required_adjusted_coverage_ratio", 1.0),
                )
            )
        ),
        "min_independent_periods": int(config.get("min_independent_periods", 36)),
    }


def _adjusted_replay_red_flags(
    candidates: dict[str, dict[str, Any]],
    passing: list[str],
) -> list[str]:
    flags = []
    if not passing:
        flags.append("no_candidate_passes_adjusted_price_replay")
    if any(row.get("missing_adjusted_symbols") for row in candidates.values()):
        flags.append("adjusted_replay_missing_selected_symbol_coverage")
    elif any(
        int(row.get("invalid_adjusted_period_count") or 0) > 0
        for row in candidates.values()
    ):
        flags.append("adjusted_replay_invalid_candidate_periods")
    if any(
        row.get("fail_closed_reason")
        == "valid_adjusted_independent_periods_below_minimum"
        for row in candidates.values()
    ):
        flags.append("adjusted_replay_too_few_valid_independent_periods")
    return sorted(set(flags))


def _candidate_distortion_dependencies(
    canonical_replay: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    distortions = [
        row for row in comparison_rows
        if row.get("split_like_distortion")
    ]
    output = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        dependencies = []
        for row in candidate.get("rows", []) or []:
            dependencies.extend(_row_distortion_dependencies(row, distortions))
        unique = _unique_dependencies(dependencies)
        output[str(name)] = {
            "candidate_name": str(name),
            "raw_adjusted_distortion_dependency_count": len(unique),
            "distortion_rebalance_dates": sorted({
                str(row.get("rebalance_date"))
                for row in unique
                if row.get("rebalance_date")
            }),
            "distortion_symbols": sorted({
                str(row.get("symbol"))
                for row in unique
                if row.get("symbol")
            }),
            "dependencies": unique[:100],
            **RESEARCH_METADATA,
        }
    return output


def _row_distortion_dependencies(
    period: dict[str, Any],
    distortions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = _date(period.get("rebalance_date"))
    end = _date(period.get("outcome_end_date")) or start
    symbols = {str(symbol).upper() for symbol in period.get("selected_symbols", [])}
    if start is None or end is None:
        return []
    rows = []
    for distortion in distortions:
        event_date = _date(distortion.get("date"))
        symbol = str(distortion.get("symbol", "")).upper()
        if event_date is None or symbol not in symbols:
            continue
        if start <= event_date <= end:
            rows.append({
                "rebalance_date": period.get("rebalance_date"),
                "outcome_end_date": period.get("outcome_end_date"),
                "symbol": symbol,
                "event_date": distortion.get("date"),
                "raw_daily_return": distortion.get("raw_daily_return"),
                "adjusted_daily_return": distortion.get("adjusted_daily_return"),
                "adjustment_ratio": distortion.get("adjustment_ratio"),
            })
    return rows


def _anomaly_survival_by_symbol(
    symbol_reports: list[dict[str, Any]],
    required_symbols: set[str],
) -> dict[str, dict[str, Any]]:
    by_symbol = {row["symbol"]: row for row in symbol_reports}
    output = {}
    for symbol in sorted(required_symbols):
        row = by_symbol.get(symbol, {})
        adjusted_available = bool(row.get("adjusted_source_available", False))
        survives = (
            not adjusted_available
            or int(row.get("anomaly_survives_adjustment_count") or 0) > 0
        )
        output[symbol] = {
            "symbol": symbol,
            "adjusted_source_available": adjusted_available,
            "raw_adjusted_distortion_count": int(
                row.get("split_like_distortion_count") or 0
            ),
            "anomaly_survives_adjustment_count": int(
                row.get("anomaly_survives_adjustment_count") or 0
            ),
            "anomaly_survives_adjusted_comparison": survives,
        }
    return output


def _comparison_red_flags(
    source_status: str,
    rows: list[dict[str, Any]],
    dependencies: dict[str, dict[str, Any]],
) -> list[str]:
    flags = []
    if source_status != "available":
        flags.append("adjusted_source_missing_or_partial")
    if any(row.get("split_like_distortion") for row in rows):
        flags.append("raw_adjusted_split_like_distortions_present")
    if any(
        int(row.get("raw_adjusted_distortion_dependency_count") or 0) > 0
        for row in dependencies.values()
    ):
        flags.append("candidate_depends_on_raw_adjusted_distortion")
    return sorted(set(flags))


def _adjusted_closes_from_comparison(
    comparison: dict[str, Any],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in comparison.get("rows", []) or []:
        adjusted = _number(row.get("adjusted_close"))
        if adjusted is None:
            continue
        symbol = str(row.get("symbol", "")).upper()
        day = str(row.get("date", ""))
        if not symbol or not day:
            continue
        output.setdefault(symbol, {})[day] = adjusted
    return output


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


def _symbols_to_compare(
    canonical_replay: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    symbols = {str(symbol).upper() for symbol in config["inspect_symbols"]}
    symbols.update({"SPY", "QQQ"})
    for candidate in canonical_replay.get("candidates", {}).values():
        for row in candidate.get("rows", []) or []:
            symbols.update(
                str(symbol).upper()
                for symbol in row.get("selected_symbols", []) or []
            )
    return sorted(symbol for symbol in symbols if symbol)


def _load_raw_stooq_rows_by_symbol(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        symbol: _load_raw_stooq_rows(data_dir / f"{symbol}.parquet")
        for symbol in symbols
    }


def _load_raw_stooq_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Adjusted data comparison requires pyarrow to read raw Stooq parquet"
        ) from exc
    table = pq.read_table(path, columns=["timestamp", "close"])
    columns = table.to_pydict()
    return [
        {"date": _date_string(timestamp), "close": close}
        for timestamp, close in zip(columns["timestamp"], columns["close"])
    ]


def _load_adjusted_rows_by_symbol(
    config: dict[str, Any],
    symbols: list[str],
) -> dict[str, list[AdjustedPricePoint]]:
    feed = LocalAdjustedPriceCsvDataFeed(
        str(config["adjusted_data_dir"]),
        combined_path=config.get("adjusted_combined_path"),
    )
    return {symbol: feed.get_adjusted_prices(symbol) for symbol in symbols}


def _raw_close_by_date(rows: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for row in rows:
        day = _date_string(row.get("timestamp") or row.get("date"))
        close = _number(row.get("close") or row.get("Close"))
        if day and close is not None and close > 0:
            output[day] = close
    return output


def _adjusted_close_by_date(
    rows: list[AdjustedPricePoint | dict[str, Any]],
) -> dict[str, float]:
    output = {}
    for row in rows:
        if isinstance(row, AdjustedPricePoint):
            output[row.timestamp.date().isoformat()] = row.adjusted_close
            continue
        day = _date_string(row.get("timestamp") or row.get("date"))
        close = _number(
            row.get("adjusted_close")
            or row.get("adj_close")
            or row.get("Adj Close")
            or row.get("close")
        )
        if day and close is not None and close > 0:
            output[day] = close
    return output


def _daily_return(
    previous: dict[str, Any] | None,
    key: str,
    current: float | None,
) -> float | None:
    prior = _number((previous or {}).get(key))
    if prior is None or current is None or prior <= 0:
        return None
    return current / prior - 1.0


def _split_like_return_factor(
    daily_return: float | None,
    tolerance: float,
) -> float | None:
    if daily_return is None:
        return None
    return _split_like_factor(1.0 + daily_return, tolerance)


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


def _is_suspicious_return(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value) >= threshold


def _unique_dependencies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (row.get("rebalance_date"), row.get("symbol"), row.get("event_date"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _normalize_comparison_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "stooq_parquet_dir": str(
            config.get("stooq_parquet_dir", "data/processed/stooq_parquet")
        ),
        "adjusted_source_name": str(
            config.get("adjusted_source_name", "local_adjusted_price_csv")
        ),
        "adjusted_data_dir": str(
            config.get("adjusted_data_dir", "data/reference/adjusted_prices")
        ),
        "adjusted_combined_path": config.get("adjusted_combined_path"),
        "inspect_symbols": [
            str(symbol).upper()
            for symbol in config.get("inspect_symbols", DEFAULT_INSPECT_SYMBOLS)
        ],
        "suspicious_daily_return_abs": float(
            config.get("suspicious_daily_return_abs", 0.50)
        ),
        "split_ratio_tolerance": float(config.get("split_ratio_tolerance", 0.08)),
    }


def _comparison_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    source = dict(ml_config.get("adjusted_data_source", {}) or {})
    audit = dict(ml_config.get("data_adjustment_audit", {}) or {})
    source.setdefault(
        "stooq_parquet_dir",
        ml_config.get("stooq_parquet_dir", "data/processed/stooq_parquet"),
    )
    for key in ("inspect_symbols", "suspicious_daily_return_abs", "split_ratio_tolerance"):
        if key in audit:
            source[key] = audit[key]
    return _normalize_comparison_config(source)


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    validation = dict(ml_config.get("benchmark_relative_validation", {}) or {})
    validation["adjusted_replay"] = dict(
        ml_config.get("adjusted_replay", {}) or {}
    )
    return validation


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
    return parsed.date().isoformat() if parsed else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
