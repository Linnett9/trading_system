from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.research.ml.adjusted_data_comparison import (
    NOTICE,
    RESEARCH_METADATA,
    _adjusted_close_by_date,
    _comparison_config,
    _load_adjusted_rows_by_symbol,
    _load_raw_stooq_rows_by_symbol,
    _number,
    _output_dir,
    _raw_close_by_date,
    _read_json,
    _symbols_to_compare,
    detect_split_like_adjustment_ratio,
)


REPORT_CANDIDATES = (
    "exact_champion_replay",
    "selected_bayesian_optimizer_diagnostic_policy",
)


@dataclass(frozen=True)
class AdjustedReplayAlignmentAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_adjusted_replay_alignment_audit(
    config: dict[str, Any],
) -> AdjustedReplayAlignmentAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    adjusted_replay = _read_json(output_dir / "adjusted_price_replay.json")
    comparison_config = _comparison_config(config)
    alignment_config = _alignment_config(config)
    symbols = _symbols_to_compare(canonical, comparison_config)
    raw_rows = _load_raw_stooq_rows_by_symbol(
        Path(str(comparison_config["stooq_parquet_dir"])),
        symbols,
    )
    adjusted_rows = _load_adjusted_rows_by_symbol(comparison_config, symbols)
    payload = build_adjusted_replay_alignment_audit(
        canonical_replay=canonical,
        adjusted_price_replay=adjusted_replay,
        raw_closes_by_symbol={
            symbol: _raw_close_by_date(rows)
            for symbol, rows in raw_rows.items()
        },
        adjusted_closes_by_symbol={
            symbol: _adjusted_close_by_date(rows)
            for symbol, rows in adjusted_rows.items()
        },
        audit_config=alignment_config,
    )
    paths = AdjustedReplayAlignmentAuditPaths(
        csv_path=output_dir / "adjusted_replay_alignment_audit.csv",
        json_path=output_dir / "adjusted_replay_alignment_audit.json",
        markdown_path=output_dir / "adjusted_replay_alignment_audit.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_adjusted_replay_alignment_audit(
    *,
    canonical_replay: dict[str, Any],
    adjusted_price_replay: dict[str, Any],
    raw_closes_by_symbol: dict[str, dict[str, float]],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    audit_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _normalize_audit_config(audit_config or {})
    adjusted_canonical = adjusted_price_replay.get("adjusted_canonical_replay", {})
    replay_candidates = adjusted_price_replay.get("candidates", {}) or {}
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in REPORT_CANDIDATES:
        candidate_rows = _candidate_alignment_rows(
            candidate,
            canonical_replay,
            adjusted_canonical,
            raw_closes_by_symbol,
            adjusted_closes_by_symbol,
            config,
        )
        rows.extend(candidate_rows)
        summaries[candidate] = _candidate_summary(
            candidate,
            candidate_rows,
            replay_candidates.get(candidate, {})
            if isinstance(replay_candidates, dict)
            else {},
        )
    alignment = _alignment_summary(rows, summaries)
    return {
        "mode": "adjusted_replay_alignment_audit_research_only",
        "replay_semantics": (
            "raw canonical replay rows compared against adjusted canonical replay "
            "rows using the same rebalance windows, symbols, exposures, and "
            "non-overlap flags"
        ),
        "audit_config": config,
        "candidate_summaries": summaries,
        "alignment": alignment,
        "biggest_return_deltas": _top_rows(
            rows,
            "return_delta",
            limit=int(config["top_delta_rows"]),
        ),
        "biggest_candidate_net_return_deltas": _top_rows(
            rows,
            "candidate_net_return_delta",
            limit=int(config["top_delta_rows"]),
        ),
        "rows": rows,
        "red_flags": _red_flags(alignment),
        **RESEARCH_METADATA,
    }


def _candidate_alignment_rows(
    candidate: str,
    canonical_replay: dict[str, Any],
    adjusted_canonical: dict[str, Any],
    raw_closes_by_symbol: dict[str, dict[str, float]],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_rows = _rows_by_date(canonical_replay, candidate)
    adjusted_rows = _rows_by_date(adjusted_canonical, candidate)
    output = []
    for rebalance_date in sorted(set(raw_rows) | set(adjusted_rows)):
        raw_row = raw_rows.get(rebalance_date)
        adjusted_row = adjusted_rows.get(rebalance_date)
        raw_symbols = _symbols(raw_row)
        adjusted_symbols = _symbols(adjusted_row)
        symbols = sorted(raw_symbols | adjusted_symbols)
        if not symbols:
            symbols = [""]
        for symbol in symbols:
            output.append(
                _alignment_row(
                    candidate,
                    rebalance_date,
                    symbol,
                    raw_row,
                    adjusted_row,
                    raw_symbols,
                    adjusted_symbols,
                    raw_closes_by_symbol,
                    adjusted_closes_by_symbol,
                    config,
                )
            )
    return output


def _alignment_row(
    candidate: str,
    rebalance_date: str,
    symbol: str,
    raw_row: dict[str, Any] | None,
    adjusted_row: dict[str, Any] | None,
    raw_symbols: set[str],
    adjusted_symbols: set[str],
    raw_closes_by_symbol: dict[str, dict[str, float]],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    raw_start_date = str((raw_row or adjusted_row or {}).get("rebalance_date") or "")
    raw_end_date = str((raw_row or {}).get("outcome_end_date") or "")
    adjusted_start_date = str(
        (adjusted_row or raw_row or {}).get("rebalance_date") or ""
    )
    adjusted_end_date = str((adjusted_row or {}).get("outcome_end_date") or "")
    raw_start = _close(raw_closes_by_symbol, symbol, raw_start_date)
    raw_end = _close(raw_closes_by_symbol, symbol, raw_end_date)
    adjusted_start = _close(adjusted_closes_by_symbol, symbol, adjusted_start_date)
    adjusted_end = _close(adjusted_closes_by_symbol, symbol, adjusted_end_date)
    raw_return = _period_return(raw_start, raw_end)
    adjusted_return = _period_return(adjusted_start, adjusted_end)
    return_delta = _delta(adjusted_return, raw_return)
    ratio_start = _ratio(adjusted_start, raw_start)
    ratio_end = _ratio(adjusted_end, raw_end)
    ratio_change = _ratio(ratio_end, ratio_start)
    expected_adjusted = _expected_adjusted_return(raw_return, ratio_start, ratio_end)
    explained_by_ratio = (
        adjusted_return is not None
        and expected_adjusted is not None
        and abs(adjusted_return - expected_adjusted) <= config["numeric_tolerance"]
    )
    split_like_factor = detect_split_like_adjustment_ratio(
        ratio_start,
        ratio_end,
        tolerance=config["split_ratio_tolerance"],
    )
    missing_adjusted = bool(
        symbol
        and (
            adjusted_start is None
            or adjusted_end is None
            or adjusted_row is None
        )
    )
    date_misalignment = raw_row is None or adjusted_row is None
    symbol_mismatch = raw_symbols != adjusted_symbols
    exposure_mismatch = _mismatch(
        _number((raw_row or {}).get("exposure")),
        _number((adjusted_row or {}).get("exposure")),
        tolerance=config["numeric_tolerance"],
    )
    label_window_mismatch = (
        raw_row is not None
        and adjusted_row is not None
        and str(raw_row.get("outcome_end_date")) != str(adjusted_row.get("outcome_end_date"))
    )
    non_overlap_mismatch = (
        raw_row is not None
        and adjusted_row is not None
        and bool(raw_row.get("included_in_canonical"))
        != bool(adjusted_row.get("included_in_canonical"))
    )
    large_delta = (
        return_delta is not None
        and abs(return_delta) >= config["return_delta_abs_threshold"]
    )
    ratio_jump = bool(
        split_like_factor is not None
        or (
            ratio_change is not None
            and abs(ratio_change - 1.0) >= config["adjustment_ratio_jump_abs_threshold"]
        )
    )
    candidate_net_return_delta = _delta(
        _number((adjusted_row or {}).get("net_return")),
        _number((raw_row or {}).get("net_return")),
    )
    large_candidate_net_delta = (
        candidate_net_return_delta is not None
        and abs(candidate_net_return_delta)
        >= config["candidate_net_return_delta_abs_threshold"]
    )
    unexplained_delta = bool(
        large_delta
        and not missing_adjusted
        and not date_misalignment
        and not explained_by_ratio
    )
    return {
        "candidate": candidate,
        "rebalance_date": rebalance_date,
        "outcome_end_date": (raw_row or {}).get("outcome_end_date"),
        "adjusted_outcome_end_date": (adjusted_row or {}).get("outcome_end_date"),
        "symbol": symbol,
        "raw_return": raw_return,
        "adjusted_return": adjusted_return,
        "return_delta": return_delta,
        "raw_close_start": raw_start,
        "raw_close_end": raw_end,
        "adjusted_close_start": adjusted_start,
        "adjusted_close_end": adjusted_end,
        "adjustment_ratio_start": ratio_start,
        "adjustment_ratio_end": ratio_end,
        "adjustment_ratio_change": ratio_change,
        "adjustment_ratio_split_like_factor": split_like_factor,
        "expected_adjusted_return_from_ratio": expected_adjusted,
        "adjusted_return_matches_ratio": explained_by_ratio,
        "exposure": (raw_row or {}).get("exposure"),
        "adjusted_exposure": (adjusted_row or {}).get("exposure"),
        "raw_candidate_net_return": (raw_row or {}).get("net_return"),
        "adjusted_candidate_net_return": (adjusted_row or {}).get("net_return"),
        "candidate_net_return_delta": candidate_net_return_delta,
        "included_in_canonical_replay": bool(
            (raw_row or {}).get("included_in_canonical", False)
        ),
        "adjusted_included_in_canonical_replay": bool(
            (adjusted_row or {}).get("included_in_canonical", False)
        ),
        "missing_adjusted_prices": missing_adjusted,
        "date_misalignment": date_misalignment,
        "symbol_mismatch": symbol_mismatch,
        "exposure_mismatch": exposure_mismatch,
        "label_window_mismatch": label_window_mismatch,
        "non_overlap_mismatch": non_overlap_mismatch,
        "return_delta_above_threshold": large_delta,
        "candidate_net_return_delta_above_threshold": large_candidate_net_delta,
        "adjustment_ratio_jump": ratio_jump,
        "unexplained_adjusted_delta": unexplained_delta,
        **RESEARCH_METADATA,
    }


def _candidate_summary(
    candidate: str,
    rows: list[dict[str, Any]],
    replay_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "row_count": len(rows),
        "adjusted_coverage_ratio": replay_row.get("adjusted_coverage_ratio"),
        "adjusted_full_symbol_coverage": replay_row.get(
            "adjusted_full_symbol_coverage",
        ),
        "missing_adjusted_symbols": replay_row.get("missing_adjusted_symbols", []),
        "empty_selection_with_positive_exposure_count": int(
            replay_row.get("empty_selection_with_positive_exposure_count") or 0
        ),
        "empty_selection_with_positive_exposure_dates": replay_row.get(
            "empty_selection_with_positive_exposure_dates",
            replay_row.get("affected_dates", []),
        ),
        "empty_selection_resolution": replay_row.get(
            "empty_selection_resolution",
            "unchanged",
        ),
        "invalid_adjusted_period_count": int(
            replay_row.get("invalid_adjusted_period_count") or 0
        ),
        "valid_adjusted_period_count": int(
            replay_row.get("valid_adjusted_period_count") or 0
        ),
        "valid_adjusted_independent_period_count": int(
            replay_row.get("valid_adjusted_independent_period_count") or 0
        ),
        "fail_closed_reason": replay_row.get("fail_closed_reason"),
        "missing_adjusted_price_row_count": _count(rows, "missing_adjusted_prices"),
        "date_misalignment_row_count": _count(rows, "date_misalignment"),
        "symbol_mismatch_row_count": _count(rows, "symbol_mismatch"),
        "exposure_mismatch_row_count": _count(rows, "exposure_mismatch"),
        "label_window_mismatch_row_count": _count(rows, "label_window_mismatch"),
        "non_overlap_mismatch_row_count": _count(rows, "non_overlap_mismatch"),
        "large_return_delta_row_count": _count(rows, "return_delta_above_threshold"),
        "large_candidate_net_return_delta_row_count": _count(
            rows,
            "candidate_net_return_delta_above_threshold",
        ),
        "adjustment_ratio_jump_row_count": _count(rows, "adjustment_ratio_jump"),
        "unexplained_adjusted_delta_row_count": _count(
            rows,
            "unexplained_adjusted_delta",
        ),
        "max_abs_return_delta": _max_abs(rows, "return_delta"),
        "max_abs_candidate_net_return_delta": _max_abs(
            rows,
            "candidate_net_return_delta",
        ),
        **RESEARCH_METADATA,
    }


def _alignment_summary(
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "same_rebalance_dates": not any(row["date_misalignment"] for row in rows),
        "same_selected_symbols": not any(row["symbol_mismatch"] for row in rows),
        "same_exposure_path": not any(row["exposure_mismatch"] for row in rows),
        "same_label_windows": not any(row["label_window_mismatch"] for row in rows),
        "same_non_overlap_periods": not any(row["non_overlap_mismatch"] for row in rows),
        "adjusted_coverage_complete": not any(
            int(summary.get("invalid_adjusted_period_count") or 0) > 0
            or summary.get("adjusted_full_symbol_coverage") is False
            for summary in summaries.values()
        ),
        "no_missing_adjusted_prices": not any(
            row["missing_adjusted_prices"] for row in rows
        ),
        "no_unexplained_large_return_deltas": not any(
            row["unexplained_adjusted_delta"] for row in rows
        ),
    }
    aligned = all(checks.values())
    return {
        "aligned_correctly": aligned,
        "checks": checks,
        "explanation_verdict": _explanation_verdict(checks, rows),
        "candidate_count": len(summaries),
        "row_count": len(rows),
        "missing_adjusted_price_row_count": _count(rows, "missing_adjusted_prices"),
        "date_misalignment_row_count": _count(rows, "date_misalignment"),
        "symbol_mismatch_row_count": _count(rows, "symbol_mismatch"),
        "exposure_mismatch_row_count": _count(rows, "exposure_mismatch"),
        "label_window_mismatch_row_count": _count(rows, "label_window_mismatch"),
        "non_overlap_mismatch_row_count": _count(rows, "non_overlap_mismatch"),
        "invalid_adjusted_period_count": sum(
            int(summary.get("invalid_adjusted_period_count") or 0)
            for summary in summaries.values()
        ),
        "valid_adjusted_period_count": sum(
            int(summary.get("valid_adjusted_period_count") or 0)
            for summary in summaries.values()
        ),
        "valid_adjusted_independent_period_count": sum(
            int(summary.get("valid_adjusted_independent_period_count") or 0)
            for summary in summaries.values()
        ),
        "large_return_delta_row_count": _count(rows, "return_delta_above_threshold"),
        "large_candidate_net_return_delta_row_count": _count(
            rows,
            "candidate_net_return_delta_above_threshold",
        ),
        "adjustment_ratio_jump_row_count": _count(rows, "adjustment_ratio_jump"),
        "unexplained_adjusted_delta_row_count": _count(
            rows,
            "unexplained_adjusted_delta",
        ),
        **RESEARCH_METADATA,
    }


def _explanation_verdict(
    checks: dict[str, bool],
    rows: list[dict[str, Any]],
) -> str:
    if not checks["no_missing_adjusted_prices"]:
        return "not_aligned_missing_adjusted_prices"
    if not checks["adjusted_coverage_complete"]:
        return "not_aligned_adjusted_coverage_failure"
    structural = [
        "same_rebalance_dates",
        "same_selected_symbols",
        "same_exposure_path",
        "same_label_windows",
        "same_non_overlap_periods",
    ]
    if not all(checks[name] for name in structural):
        return "not_aligned_replay_path_mismatch"
    if not checks["no_unexplained_large_return_deltas"]:
        return "not_aligned_unexplained_return_delta"
    if any(row["return_delta_above_threshold"] for row in rows):
        return "aligned_large_deltas_explained_by_adjustment_ratios"
    return "aligned_no_material_adjusted_delta"


def _red_flags(alignment: dict[str, Any]) -> list[str]:
    checks = alignment.get("checks", {})
    flags = []
    if not checks.get("same_rebalance_dates", False):
        flags.append("date_misalignment")
    if not checks.get("same_selected_symbols", False):
        flags.append("symbol_mismatch")
    if not checks.get("same_exposure_path", False):
        flags.append("exposure_mismatch")
    if not checks.get("same_label_windows", False):
        flags.append("label_window_mismatch")
    if not checks.get("same_non_overlap_periods", False):
        flags.append("non_overlap_mismatch")
    if not checks.get("adjusted_coverage_complete", False):
        flags.append("adjusted_coverage_failure")
    if not checks.get("no_missing_adjusted_prices", False):
        flags.append("missing_adjusted_prices")
    if not checks.get("no_unexplained_large_return_deltas", False):
        flags.append("unexplained_adjusted_return_delta")
    return flags


def _rows_by_date(
    replay: dict[str, Any],
    candidate: str,
) -> dict[str, dict[str, Any]]:
    rows = replay.get("candidates", {}).get(candidate, {}).get("rows", []) or []
    return {
        str(row.get("rebalance_date")): row
        for row in rows
        if isinstance(row, dict) and row.get("rebalance_date")
    }


def _symbols(row: dict[str, Any] | None) -> set[str]:
    return {
        str(symbol).upper()
        for symbol in (row or {}).get("selected_symbols", []) or []
        if str(symbol)
    }


def _close(
    closes_by_symbol: dict[str, dict[str, float]],
    symbol: str,
    day: str,
) -> float | None:
    if not symbol or not day:
        return None
    return _number(closes_by_symbol.get(symbol.upper(), {}).get(day))


def _period_return(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _expected_adjusted_return(
    raw_return: float | None,
    ratio_start: float | None,
    ratio_end: float | None,
) -> float | None:
    ratio_change = _ratio(ratio_end, ratio_start)
    if raw_return is None or ratio_change is None:
        return None
    return (1.0 + raw_return) * ratio_change - 1.0


def _mismatch(
    first: float | None,
    second: float | None,
    *,
    tolerance: float,
) -> bool:
    if first is None and second is None:
        return False
    if first is None or second is None:
        return True
    return abs(first - second) > tolerance


def _count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(row.get(field)) for row in rows)


def _max_abs(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        abs(float(value))
        for value in (row.get(field) for row in rows)
        if _number(value) is not None
    ]
    return max(values, default=None)


def _top_rows(
    rows: list[dict[str, Any]],
    field: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = [
        row for row in rows
        if _number(row.get(field)) is not None
    ]
    ranked.sort(key=lambda row: abs(float(row[field])), reverse=True)
    return ranked[:limit]


def _alignment_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    return _normalize_audit_config(
        ml_config.get("adjusted_replay_alignment_audit", {}) or {}
    )


def _normalize_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_delta_abs_threshold": float(
            config.get("return_delta_abs_threshold", 0.05)
        ),
        "adjustment_ratio_jump_abs_threshold": float(
            config.get("adjustment_ratio_jump_abs_threshold", 0.02)
        ),
        "candidate_net_return_delta_abs_threshold": float(
            config.get("candidate_net_return_delta_abs_threshold", 0.05)
        ),
        "split_ratio_tolerance": float(config.get("split_ratio_tolerance", 0.08)),
        "numeric_tolerance": float(config.get("numeric_tolerance", 1e-10)),
        "top_delta_rows": int(config.get("top_delta_rows", 25)),
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "candidate",
        "rebalance_date",
        "outcome_end_date",
        "adjusted_outcome_end_date",
        "symbol",
        "raw_return",
        "adjusted_return",
        "return_delta",
        "raw_close_start",
        "raw_close_end",
        "adjusted_close_start",
        "adjusted_close_end",
        "adjustment_ratio_start",
        "adjustment_ratio_end",
        "adjustment_ratio_change",
        "adjustment_ratio_split_like_factor",
        "expected_adjusted_return_from_ratio",
        "adjusted_return_matches_ratio",
        "exposure",
        "adjusted_exposure",
        "raw_candidate_net_return",
        "adjusted_candidate_net_return",
        "candidate_net_return_delta",
        "included_in_canonical_replay",
        "adjusted_included_in_canonical_replay",
        "missing_adjusted_prices",
        "date_misalignment",
        "symbol_mismatch",
        "exposure_mismatch",
        "label_window_mismatch",
        "non_overlap_mismatch",
        "return_delta_above_threshold",
        "candidate_net_return_delta_above_threshold",
        "adjustment_ratio_jump",
        "unexplained_adjusted_delta",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("rows", []) or []:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _markdown(payload: dict[str, Any]) -> str:
    alignment = payload.get("alignment", {})
    lines = [
        "# Adjusted Replay Alignment Audit",
        "",
        NOTICE,
        "",
        f"Aligned correctly: {alignment.get('aligned_correctly')}",
        f"Explanation verdict: {alignment.get('explanation_verdict')}",
        f"Missing adjusted price rows: {alignment.get('missing_adjusted_price_row_count')}",
        f"Invalid adjusted periods: {alignment.get('invalid_adjusted_period_count')}",
        f"Valid adjusted periods: {alignment.get('valid_adjusted_period_count')}",
        "Valid adjusted independent periods: "
        f"{alignment.get('valid_adjusted_independent_period_count')}",
        f"Date misalignment rows: {alignment.get('date_misalignment_row_count')}",
        f"Symbol mismatch rows: {alignment.get('symbol_mismatch_row_count')}",
        f"Large return-delta rows: {alignment.get('large_return_delta_row_count')}",
        "Large candidate net-return delta rows: "
        f"{alignment.get('large_candidate_net_return_delta_row_count')}",
        f"Adjustment-ratio jump rows: {alignment.get('adjustment_ratio_jump_row_count')}",
        "",
        "|candidate|rows|coverage|valid periods|invalid periods|missing adjusted|date mismatch|symbol mismatch|large delta|max abs delta|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("candidate_summaries", {}).values():
        lines.append(
            "|{candidate}|{rows}|{coverage}|{valid}|{invalid}|{missing}|{dates}|{symbols}|{large}|{delta}|".format(
                candidate=row.get("candidate"),
                rows=row.get("row_count"),
                coverage=_fmt(row.get("adjusted_coverage_ratio")),
                valid=row.get("valid_adjusted_period_count"),
                invalid=row.get("invalid_adjusted_period_count"),
                missing=row.get("missing_adjusted_price_row_count"),
                dates=row.get("date_misalignment_row_count"),
                symbols=row.get("symbol_mismatch_row_count"),
                large=row.get("large_return_delta_row_count"),
                delta=_fmt(row.get("max_abs_return_delta")),
            )
        )
    lines.extend([
        "",
        "## Biggest Return Deltas",
        "",
        "|candidate|rebalance|symbol|raw return|adjusted return|delta|ratio start|ratio end|missing adjusted|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("biggest_return_deltas", [])[:10]:
        lines.append(
            "|{candidate}|{date}|{symbol}|{raw}|{adjusted}|{delta}|{rs}|{re}|{missing}|".format(
                candidate=row.get("candidate"),
                date=row.get("rebalance_date"),
                symbol=row.get("symbol"),
                raw=_fmt(row.get("raw_return")),
                adjusted=_fmt(row.get("adjusted_return")),
                delta=_fmt(row.get("return_delta")),
                rs=_fmt(row.get("adjustment_ratio_start")),
                re=_fmt(row.get("adjustment_ratio_end")),
                missing=row.get("missing_adjusted_prices"),
            )
        )
    lines.extend([
        "",
        "## Biggest Candidate Net-Return Deltas",
        "",
        "|candidate|rebalance|symbol|raw net|adjusted net|delta|missing adjusted|",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in payload.get("biggest_candidate_net_return_deltas", [])[:10]:
        lines.append(
            "|{candidate}|{date}|{symbol}|{raw}|{adjusted}|{delta}|{missing}|".format(
                candidate=row.get("candidate"),
                date=row.get("rebalance_date"),
                symbol=row.get("symbol"),
                raw=_fmt(row.get("raw_candidate_net_return")),
                adjusted=_fmt(row.get("adjusted_candidate_net_return")),
                delta=_fmt(row.get("candidate_net_return_delta")),
                missing=row.get("missing_adjusted_prices"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
