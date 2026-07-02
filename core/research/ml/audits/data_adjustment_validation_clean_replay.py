from __future__ import annotations

from typing import Any

from core.research.ml.audits.benchmark_relative_validation import (
    build_benchmark_relative_validation,
)
from core.research.ml.audits.data_adjustment_validation_types import (
    REPORT_CANDIDATES,
    RESEARCH_METADATA,
)
from core.research.ml.audits.data_adjustment_validation_utils import _number
from core.research.ml.replay.canonical_continuous_equity_replay import (
    build_canonical_replay,
)


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
