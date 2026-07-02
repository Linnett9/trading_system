from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_loading import _number
from core.research.ml.audits.adjusted_data_types import REPORT_CANDIDATES, RESEARCH_METADATA
from core.research.ml.audits.benchmark_relative_validation import build_benchmark_relative_validation
from core.research.ml.replay.canonical_continuous_equity_replay import build_canonical_replay
from core.research.ml.audits.adjusted_price_replay_config import _adjusted_replay_config
from core.research.ml.audits.adjusted_price_replay_prices import _raw_fallback_available, _replay_close, _weighted_period_return
from core.research.ml.audits.adjusted_price_replay_coverage import _attach_adjusted_independent_counts, _coverage_fail_closed_reason, _coverage_summary, _empty_coverage_summary, _period_adjusted_coverage, _period_fail_closed_reason, _period_valid_under_policy
from core.research.ml.audits.adjusted_price_replay_candidates import _adjusted_champion_audit, _adjusted_selected_optimizer
from core.research.ml.audits.adjusted_price_replay_gates import _adjusted_replay_red_flags, _candidate_coverage_ok, _fail_closed_reason, _valid_adjusted_independent_periods_ok


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
