from __future__ import annotations

import inspect

from core.research.ml.audits import independent_period_expansion_audit
from core.research.ml.audits.independent_period_expansion_audit import (
    build_independent_period_expansion_audit,
)


def test_expansion_does_not_create_overlapping_label_windows():
    payload = build_independent_period_expansion_audit(
        adjusted_price_replay=_adjusted_replay(),
        canonical_replay=_canonical_replay(),
        adjusted_closes_by_symbol=_benchmark_closes(),
        validation_config={"min_independent_periods": 3},
        expansion_config={
            "settings": [
                {
                    "name": "monthly",
                    "spacing": "monthly",
                    "minimum_gap_days": 0,
                    "enforce_non_overlap": True,
                }
            ]
        },
    )

    row = payload["rows"][0]

    assert row["leakage_safe"] is True
    assert row["independent_period_count"] == 2
    assert row["selected_rebalance_dates"] == ["2024-01-01", "2024-03-01"]
    assert row["skipped_overlap_dates"] == ["2024-02-01"]


def test_no_selected_symbols_periods_are_reported_as_expected_no_position():
    payload = build_independent_period_expansion_audit(
        adjusted_price_replay=_adjusted_replay(),
        canonical_replay=_canonical_replay(),
        validation_config={"min_independent_periods": 3},
    )

    rows = payload["no_selected_symbols"]
    exact = [row for row in rows if row["candidate"] == "exact_champion_replay"]

    assert len(exact) == 1
    assert exact[0]["rebalance_date"] == "2024-04-01"
    assert exact[0]["why_no_symbols"] == (
        "source replay row has zero exposure and empty target weights"
    )
    assert exact[0]["expected_no_position"] is True
    assert exact[0]["replay_bug_suspected"] is False
    assert payload["no_selected_symbol_summary"]["verdict"] == (
        "expected_no_position_periods"
    )


def test_expansion_report_preserves_promotion_gates():
    payload = build_independent_period_expansion_audit(
        adjusted_price_replay=_adjusted_replay(),
        canonical_replay=_canonical_replay(),
        adjusted_replay_alignment_audit={
            "alignment": {"aligned_correctly": False},
        },
        adjusted_closes_by_symbol=_benchmark_closes(),
        validation_config={"min_independent_periods": 36},
    )

    assert payload["minimum_independent_periods"] == 36
    assert payload["promotion_thresholds_changed"] is False
    assert all(row["promotion_gate_status"] == "blocked" for row in payload["rows"])
    assert any(
        "minimum_adjusted_independent_periods" in row["failed_gates"]
        for row in payload["rows"]
    )


def test_independent_period_expansion_audit_has_no_operational_imports():
    source = inspect.getsource(independent_period_expansion_audit)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def test_top_level_independent_period_expansion_audit_reexports_helpers():
    from core.research.ml import independent_period_expansion_audit as top_level

    assert (
        top_level.build_independent_period_expansion_audit
        is build_independent_period_expansion_audit
    )
    assert top_level.REPORT_CANDIDATES == independent_period_expansion_audit.REPORT_CANDIDATES
    assert top_level._compound([0.10, -0.05]) == 0.04499999999999993
    assert top_level._normalized_expansion_config({})["settings"][0]["name"] == (
        "current_strict_non_overlap"
    )


def _adjusted_replay() -> dict:
    rows = [
        _row("2024-01-01", "2024-02-15", 0.10, included=True),
        _row("2024-02-01", "2024-03-15", 0.05),
        _row("2024-03-01", "2024-04-15", -0.02),
    ]
    coverage_periods = [
        {
            "rebalance_date": "2024-01-01",
            "outcome_end_date": "2024-02-15",
            "selected_symbols": ["AAA"],
            "selected_symbol_count": 1,
            "covered_adjusted_symbol_count": 1,
            "missing_adjusted_symbols": [],
            "adjusted_coverage_ratio": 1.0,
            "valid_adjusted_period": True,
            "fail_closed_reason": None,
        },
        {
            "rebalance_date": "2024-04-01",
            "outcome_end_date": "2024-05-15",
            "selected_symbols": [],
            "selected_symbol_count": 0,
            "covered_adjusted_symbol_count": 0,
            "missing_adjusted_symbols": [],
            "adjusted_coverage_ratio": 0.0,
            "valid_adjusted_period": False,
            "fail_closed_reason": "no_selected_symbols",
        },
    ]
    return {
        "adjusted_canonical_replay": {
            "candidates": {
                "exact_champion_replay": {"rows": rows},
                "selected_bayesian_optimizer_diagnostic_policy": {"rows": rows},
            }
        },
        "candidates": {
            "exact_champion_replay": _summary(coverage_periods),
            "selected_bayesian_optimizer_diagnostic_policy": _summary(
                coverage_periods
            ),
        },
    }


def _canonical_replay() -> dict:
    raw_rows = [
        _row("2024-01-01", "2024-02-15", 0.10, included=True),
        _row("2024-04-01", "2024-05-15", 0.0, exposure=0.0, symbols=[]),
    ]
    return {
        "candidates": {
            "exact_champion_replay": {"rows": raw_rows},
            "selected_bayesian_optimizer_diagnostic_policy": {"rows": raw_rows},
        }
    }


def _row(
    start: str,
    end: str,
    value: float,
    *,
    included: bool = False,
    exposure: float = 1.0,
    symbols: list[str] | None = None,
) -> dict:
    return {
        "rebalance_date": start,
        "outcome_end_date": end,
        "net_return": value,
        "period_return": value,
        "included_in_canonical": included,
        "exclusion_reason": None,
        "exposure": exposure,
        "selected_symbols": ["AAA"] if symbols is None else symbols,
    }


def _summary(periods: list[dict]) -> dict:
    return {
        "adjusted_coverage_ratio": 1.0,
        "adjusted_full_symbol_coverage": True,
        "valid_adjusted_period_count": 3,
        "invalid_adjusted_period_count": 1,
        "valid_adjusted_independent_period_count": 1,
        "coverage": {"periods": periods},
    }


def _benchmark_closes() -> dict:
    return {
        "SPY": {
            "2024-01-01": 100.0,
            "2024-02-01": 100.0,
            "2024-03-01": 100.0,
            "2024-02-15": 101.0,
            "2024-03-15": 101.0,
            "2024-04-15": 101.0,
        }
    }
