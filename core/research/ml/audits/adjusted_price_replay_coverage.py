from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_loading import _number
from core.research.ml.audits.adjusted_data_types import RESEARCH_METADATA
from core.research.ml.audits.adjusted_price_replay_prices import _raw_fallback_available


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
