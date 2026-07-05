from __future__ import annotations

from typing import Any

from core.research.ml.audits.benchmark_relative_validation_math import _number


def _apply_gates(
    row: dict[str, Any],
    *,
    benchmark_returns: dict[str, float | None],
    spy_drawdown: float | None,
    config: dict[str, Any],
    external_reports: dict[str, Any],
) -> dict[str, Any]:
    if not row.get("available"):
        return {
            **row,
            "benchmark_relative_pass": False,
            "tradability_validation_pass": False,
            "promotion_candidate_status": "unavailable",
            "failed_gates": ["candidate_unavailable"],
        }
    candidate_return = float(row["canonical_non_overlap_return"])
    excess = {
        name: (
            candidate_return - benchmark
            if benchmark is not None else None
        )
        for name, benchmark in benchmark_returns.items()
    }
    gates = {
        "anomaly_dependency": float(row["anomaly_dependency_ratio"])
        <= float(config.get("max_anomaly_dependency_ratio", 0.25)),
        "top_5_date_concentration": (
            _number(row.get("top_5_date_profit_share")) is not None
            and float(row["top_5_date_profit_share"])
            <= float(config.get("max_top_5_date_profit_share", 0.50))
        ),
        "positive_excess_vs_spy": excess["spy"] is not None and excess["spy"] > 0.0,
        "positive_excess_vs_qqq": excess["qqq"] is not None and excess["qqq"] > 0.0,
        "positive_excess_vs_equal_weight": (
            excess["equal_weight"] is not None and excess["equal_weight"] > 0.0
        ),
        "survives_25bps": float(row["cost_stressed_return_25bps"]) > 0.0,
        "survives_50bps": float(row["cost_stressed_return_50bps"]) > 0.0,
        "drawdown_not_materially_worse_than_spy": (
            spy_drawdown is not None
            and float(row["max_drawdown"])
            <= spy_drawdown + float(config.get("max_drawdown_worse_than_spy", 0.05))
        ),
    }
    external_context = _external_promotion_gate_context(
        str(row["candidate_name"]),
        external_reports,
        config,
    )
    gates.update(external_context.get("gates", {}))
    benchmark_pass = all(
        gates[name]
        for name in (
            "positive_excess_vs_spy",
            "positive_excess_vs_qqq",
            "positive_excess_vs_equal_weight",
        )
    )
    tradability_pass = all(
        value for name, value in gates.items() if not name.startswith("positive_excess")
    )
    return {
        **row,
        "excess_return_vs_spy": excess["spy"],
        "excess_return_vs_qqq": excess["qqq"],
        "excess_return_vs_equal_weight": excess["equal_weight"],
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "external_gate_context": external_context.get("context", {}),
        "benchmark_relative_pass": benchmark_pass,
        "tradability_validation_pass": tradability_pass,
        "promotion_candidate_status": (
            "pass" if benchmark_pass and tradability_pass else "blocked"
        ),
    }

def _external_promotion_gate_context(
    candidate_name: str,
    external_reports: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    context: dict[str, Any] = {}
    adjustment = external_reports.get("data_adjustment_audit") or {}
    if adjustment:
        status = str(
            adjustment.get("adjusted_price_status")
            or adjustment.get("adjusted_status")
            or "unknown"
        )
        dependencies = (
            adjustment.get("candidate_dependencies", {}).get(candidate_name, {})
            if isinstance(adjustment.get("candidate_dependencies"), dict)
            else {}
        )
        dependency_count = int(dependencies.get("suspicious_dependency_count") or 0)
        gates["adjusted_price_status"] = _adjusted_price_status_acceptable(
            status,
            adjustment,
            config,
        )
        gates["no_suspicious_split_like_rows"] = dependency_count == 0
        context["data_adjustment_audit"] = {
            "adjusted_price_status": status,
            "suspicious_dependency_count": dependency_count,
            "suspicious_rebalance_dates": dependencies.get(
                "suspicious_rebalance_dates",
                [],
            ),
        }
    independent = external_reports.get("independent_period_validation") or {}
    if independent:
        gate = independent.get("gate", {})
        actual = int(
            gate.get(
                "actual",
                independent.get("independent_canonical_period_count", 0),
            )
            or 0
        )
        minimum = int(
            gate.get(
                "minimum",
                independent.get(
                    "minimum_independent_periods",
                    config.get("min_independent_periods", 36),
                ),
            )
            or 0
        )
        gates["minimum_independent_periods"] = bool(
            gate.get("passed", actual >= minimum)
        )
        context["independent_period_validation"] = {
            "actual": actual,
            "minimum": minimum,
        }
    clean = external_reports.get("clean_data_replay") or {}
    if clean:
        clean_candidates = clean.get("candidates", {})
        clean_row = (
            clean_candidates.get(candidate_name, {})
            if isinstance(clean_candidates, dict)
            else {}
        )
        gates["clean_data_return_positive"] = bool(
            clean_row.get("clean_data_return_positive", False)
        )
        gates["clean_data_benchmark_relative"] = bool(
            clean_row.get("clean_data_benchmark_relative", False)
        )
        context["clean_data_replay"] = {
            "clean_canonical_return": clean_row.get("clean_canonical_return"),
            "clean_data_verdict": clean_row.get("clean_data_verdict"),
        }
    adjusted_comparison = external_reports.get("adjusted_data_comparison") or {}
    if adjusted_comparison:
        source = adjusted_comparison.get("adjusted_source", {})
        dependencies = (
            adjusted_comparison.get("candidate_dependencies", {}).get(
                candidate_name,
                {},
            )
            if isinstance(adjusted_comparison.get("candidate_dependencies"), dict)
            else {}
        )
        dependency_count = int(
            dependencies.get("raw_adjusted_distortion_dependency_count") or 0
        )
        gates["adjusted_source_available"] = bool(source.get("acceptable", False))
        gates["no_raw_adjusted_split_like_distortion"] = dependency_count == 0
        context["adjusted_data_comparison"] = {
            "adjusted_source_status": source.get("available_status"),
            "raw_adjusted_distortion_dependency_count": dependency_count,
            "distortion_rebalance_dates": dependencies.get(
                "distortion_rebalance_dates",
                [],
            ),
        }
    adjusted_replay = external_reports.get("adjusted_price_replay") or {}
    if adjusted_replay:
        replay_candidates = adjusted_replay.get("candidates", {})
        replay_row = (
            replay_candidates.get(candidate_name, {})
            if isinstance(replay_candidates, dict)
            else {}
        )
        context["adjusted_price_replay"] = {
            "adjusted_source_available": adjusted_replay.get(
                "adjusted_source_available"
            ),
            "adjusted_canonical_return": replay_row.get(
                "adjusted_canonical_return"
            ),
            "adjusted_price_replay_verdict": replay_row.get(
                "adjusted_price_replay_verdict"
            ),
            "adjusted_coverage_ratio": replay_row.get("adjusted_coverage_ratio"),
            "missing_adjusted_symbols": replay_row.get("missing_adjusted_symbols"),
            "invalid_adjusted_period_count": replay_row.get(
                "invalid_adjusted_period_count"
            ),
            "fail_closed_reason": replay_row.get("fail_closed_reason"),
        }
        gates["adjusted_replay_full_symbol_coverage"] = bool(
            replay_row.get("adjusted_full_symbol_coverage", True)
        )
        gates["minimum_adjusted_independent_periods"] = bool(
            replay_row.get("minimum_adjusted_independent_periods_pass", True)
        )
    alignment_audit = external_reports.get("adjusted_replay_alignment_audit") or {}
    if alignment_audit:
        alignment = alignment_audit.get("alignment", {})
        gates["adjusted_replay_alignment"] = bool(
            alignment.get("aligned_correctly", False)
        )
        context["adjusted_replay_alignment_audit"] = {
            "aligned_correctly": alignment.get("aligned_correctly"),
            "explanation_verdict": alignment.get("explanation_verdict"),
            "missing_adjusted_price_row_count": alignment.get(
                "missing_adjusted_price_row_count"
            ),
            "invalid_adjusted_period_count": alignment.get(
                "invalid_adjusted_period_count"
            ),
        }
    return {"gates": gates, "context": context}

def _adjusted_price_status_acceptable(
    status: str,
    adjustment_report: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    gate = adjustment_report.get("promotion_gate", {})
    if "adjusted_price_status_acceptable" in gate:
        return bool(gate["adjusted_price_status_acceptable"])
    acceptable = set(
        str(value)
        for value in config.get(
            "acceptable_adjusted_price_statuses",
            ["known_adjusted", "appears_adjusted", "raw_adjusted_identical"],
        )
    )
    if status in acceptable:
        return True
    return bool(config.get("allow_unknown_adjusted_price_status", False)) and (
        status.startswith("unknown")
    )
