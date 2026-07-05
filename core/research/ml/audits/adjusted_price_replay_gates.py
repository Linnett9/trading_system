from __future__ import annotations

from typing import Any


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
