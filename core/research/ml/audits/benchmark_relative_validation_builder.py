from __future__ import annotations

from typing import Any

from core.research.ml.audits.benchmark_relative_validation_types import COST_STRESS_BPS, RESEARCH_METADATA
from core.research.ml.audits.benchmark_relative_validation_baselines import _canonical_candidate, _canonical_schedule, _market_baseline, _selected_universe_baseline
from core.research.ml.audits.benchmark_relative_validation_scoring import _merge_existing_concentration, _score_candidate
from core.research.ml.audits.benchmark_relative_validation_gates import _apply_gates
from core.research.ml.audits.benchmark_relative_validation_math import _number, _return


def build_benchmark_relative_validation(
    *,
    canonical_replay: dict[str, Any],
    anomaly_report: dict[str, Any],
    concentration_report: dict[str, Any] | None = None,
    closes_by_symbol: dict[str, dict[str, float]],
    validation_config: dict[str, Any] | None = None,
    external_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config or {}
    external_reports = external_reports or {}
    schedule = _canonical_schedule(canonical_replay)
    flagged_dates = set(anomaly_report.get("flagged_rebalance_dates", []) or [])
    candidates = [
        _market_baseline("spy_buy_and_hold", schedule, closes_by_symbol, "SPY"),
        _market_baseline("qqq_buy_and_hold", schedule, closes_by_symbol, "QQQ"),
        _selected_universe_baseline(
            "equal_weight_selected_universe",
            schedule,
            closes_by_symbol,
            equal_weight=True,
        ),
        _selected_universe_baseline(
            "always_full_champion_universe",
            schedule,
            closes_by_symbol,
            equal_weight=False,
        ),
        _canonical_candidate(canonical_replay, "exact_champion_replay"),
        _canonical_candidate(
            canonical_replay,
            "selected_bayesian_optimizer_diagnostic_policy",
        ),
    ]
    scored = [
        _merge_existing_concentration(
            _score_candidate(candidate, flagged_dates),
            (concentration_report or {}).get("candidates", {}).get(
                candidate["candidate_name"],
                {},
            ),
        )
        for candidate in candidates
    ]
    by_name = {row["candidate_name"]: row for row in scored}
    benchmark_returns = {
        "spy": _return(by_name.get("spy_buy_and_hold")),
        "qqq": _return(by_name.get("qqq_buy_and_hold")),
        "equal_weight": _return(by_name.get("equal_weight_selected_universe")),
    }
    spy_drawdown = _number(
        by_name.get("spy_buy_and_hold", {}).get("max_drawdown")
    )
    gated = [
        _apply_gates(
            row,
            benchmark_returns=benchmark_returns,
            spy_drawdown=spy_drawdown,
            config=config,
            external_reports=external_reports,
        )
        for row in scored
    ]
    passing = [
        row["candidate_name"]
        for row in gated
        if row.get("promotion_candidate_status") == "pass"
    ]
    return {
        "mode": "benchmark_relative_tradability_validation_research_only",
        "canonical_alignment": (
            "all candidates use exact champion canonical non-overlapping windows"
        ),
        "cost_stress_semantics": (
            "incremental cost = estimated one-way effective-weight turnover * bps"
        ),
        "cost_stress_bps": list(COST_STRESS_BPS),
        "benchmark_returns": benchmark_returns,
        "gate_config": {
            "max_anomaly_dependency_ratio": float(
                config.get("max_anomaly_dependency_ratio", 0.25)
            ),
            "max_top_5_date_profit_share": float(
                config.get("max_top_5_date_profit_share", 0.50)
            ),
            "max_drawdown_worse_than_spy": float(
                config.get("max_drawdown_worse_than_spy", 0.05)
            ),
            "min_independent_periods": int(
                config.get("min_independent_periods", 36)
            ),
            "acceptable_adjusted_price_statuses": list(
                config.get(
                    "acceptable_adjusted_price_statuses",
                    [
                        "known_adjusted",
                        "appears_adjusted",
                        "raw_adjusted_identical",
                    ],
                )
            ),
            "allow_unknown_adjusted_price_status": bool(
                config.get("allow_unknown_adjusted_price_status", False)
            ),
        },
        "external_gate_reports_available": {
            name: bool(report) for name, report in external_reports.items()
        },
        "candidates": gated,
        "promotion_candidates": passing,
        "any_candidate_passes": bool(passing),
        **RESEARCH_METADATA,
    }
