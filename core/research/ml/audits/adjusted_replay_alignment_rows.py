from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_comparison import (
    RESEARCH_METADATA,
    _number,
    detect_split_like_adjustment_ratio,
)
from core.research.ml.audits.adjusted_replay_alignment_math import (
    _close,
    _delta,
    _expected_adjusted_return,
    _mismatch,
    _period_return,
    _ratio,
)


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
