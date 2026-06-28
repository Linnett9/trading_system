from __future__ import annotations

import inspect

from core.research.ml import benchmark_relative_validation
from core.research.ml.benchmark_relative_validation import (
    build_benchmark_relative_validation,
)


def test_benchmark_relative_calculations_are_deterministic():
    canonical, closes = _fixtures()
    anomaly = {"flagged_rebalance_dates": ["2024-01-01"]}

    first = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report=anomaly,
        closes_by_symbol=closes,
    )
    second = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report=anomaly,
        closes_by_symbol=closes,
    )

    assert first == second
    assert [row["candidate_name"] for row in first["candidates"]] == [
        "spy_buy_and_hold",
        "qqq_buy_and_hold",
        "equal_weight_selected_universe",
        "always_full_champion_universe",
        "exact_champion_replay",
        "selected_bayesian_optimizer_diagnostic_policy",
    ]
    assert all(
        f"cost_stressed_return_{bps}bps" in first["candidates"][0]
        for bps in (5, 10, 25, 50, 100)
    )


def test_failing_concentration_gate_blocks_promotion():
    canonical, closes = _fixtures(exact_returns=[0.50, 0.01, 0.01, 0.01, 0.01, 0.01])

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=closes,
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["top_5_date_profit_share"] > 0.50
    assert exact["gates"]["top_5_date_concentration"] is False
    assert exact["promotion_candidate_status"] == "blocked"


def test_failing_anomaly_gate_blocks_promotion():
    canonical, closes = _fixtures(exact_returns=[0.40, 0.01, 0.01, 0.01, 0.01, 0.01])

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": ["2024-01-01"]},
        closes_by_symbol=closes,
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["anomaly_dependency_ratio"] > 0.25
    assert exact["gates"]["anomaly_dependency"] is False
    assert exact["promotion_candidate_status"] == "blocked"


def test_high_raw_return_with_poor_benchmark_excess_fails():
    canonical, closes = _fixtures(exact_returns=[0.02] * 6)

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=closes,
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["canonical_non_overlap_return"] > 0.0
    assert exact["excess_return_vs_spy"] < 0.0
    assert exact["excess_return_vs_qqq"] < 0.0
    assert exact["benchmark_relative_pass"] is False
    assert exact["promotion_candidate_status"] == "blocked"


def test_existing_canonical_concentration_audit_is_reused():
    canonical, closes = _fixtures()
    concentration = {
        "candidates": {
            "exact_champion_replay": {
                "profit_concentration": {
                    "top_1_date_positive_return_share": 0.80,
                    "top_5_date_positive_return_share": 0.95,
                    "top_1_symbol_contribution_share": 0.60,
                },
                "scenarios": [
                    {
                        "scenario_name": "remove_anomaly_dates",
                        "summary": {"total_return": 0.10},
                    }
                ],
            }
        }
    }

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        concentration_report=concentration,
        closes_by_symbol=closes,
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["anomaly_adjusted_return"] == 0.10
    assert exact["top_1_date_profit_share"] == 0.80
    assert exact["top_5_date_profit_share"] == 0.95
    assert exact["top_1_symbol_profit_share"] == 0.60


def test_false_adjusted_alignment_audit_blocks_promotion():
    canonical, closes = _fixtures()

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=closes,
        validation_config={
            "max_top_5_date_profit_share": 1.0,
            "max_drawdown_worse_than_spy": 1.0,
        },
        external_reports={
            "adjusted_replay_alignment_audit": {
                "alignment": {
                    "aligned_correctly": False,
                    "explanation_verdict": "not_aligned_missing_adjusted_prices",
                    "missing_adjusted_price_row_count": 2,
                    "invalid_adjusted_period_count": 1,
                }
            }
        },
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["gates"]["adjusted_replay_alignment"] is False
    assert "adjusted_replay_alignment" in exact["failed_gates"]
    assert exact["promotion_candidate_status"] == "blocked"


def test_benchmark_validation_has_no_operational_imports_or_references():
    source = inspect.getsource(benchmark_relative_validation)

    assert "broker" not in source
    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "order_execution" not in source


def _fixtures(
    *,
    exact_returns: list[float] | None = None,
) -> tuple[dict, dict[str, dict[str, float]]]:
    exact_returns = exact_returns or [0.08] * 6
    starts = [f"2024-01-{1 + index * 2:02d}" for index in range(6)]
    ends = [f"2024-01-{2 + index * 2:02d}" for index in range(6)]
    exact_rows = [
        _canonical_row(start, end, value, ["AAA", "BBB"])
        for start, end, value in zip(starts, ends, exact_returns)
    ]
    optimizer_rows = [
        _canonical_row(start, end, 0.06, ["AAA", "BBB"], exposure=0.8)
        for start, end in zip(starts, ends)
    ]
    canonical = {
        "candidates": {
            "exact_champion_replay": {"rows": exact_rows},
            "selected_bayesian_optimizer_diagnostic_policy": {
                "rows": optimizer_rows
            },
        }
    }
    closes: dict[str, dict[str, float]] = {}
    for symbol, period_return in (
        ("SPY", 0.05),
        ("QQQ", 0.04),
        ("AAA", 0.03),
        ("BBB", 0.03),
    ):
        values = {}
        for start, end in zip(starts, ends):
            values[start] = 100.0
            values[end] = 100.0 * (1.0 + period_return)
        closes[symbol] = values
    return canonical, closes


def _canonical_row(
    start: str,
    end: str,
    period_return: float,
    symbols: list[str],
    *,
    exposure: float = 1.0,
) -> dict:
    return {
        "rebalance_date": start,
        "outcome_end_date": end,
        "included_in_canonical": True,
        "exclusion_reason": None,
        "period_return": period_return,
        "net_return": period_return,
        "exposure": exposure,
        "selected_symbols": symbols,
        "target_weights": {symbol: 1.0 / len(symbols) for symbol in symbols},
    }


def _candidate(payload: dict, name: str) -> dict:
    return next(
        row for row in payload["candidates"]
        if row["candidate_name"] == name
    )
