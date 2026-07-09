from __future__ import annotations

import pytest

from core.research.ml.stock_level.bar_cadence_portfolio_replay import (
    build_bar_cadence_portfolio_replay,
)


def test_bar_cadence_replay_scores_every_bar_and_executes_next_bar():
    result = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        cost_bps=10,
        slippage_bps=5,
    )

    assert result.payload["scoring_cadence"] == "every_completed_5m_bar"
    assert result.payload["decision_cadence"] == "every_1_5m_bar"
    assert result.payload["retraining_cadence"] == "external"
    assert result.payload["execution_timing"] == "next_eligible_bar_open"
    assert len(result.scores) == 10

    first = result.periods[0]
    assert first["decision_timestamp"] == "2024-01-01T09:30:00+00:00"
    assert first["execution_timestamp"] == "2024-01-01T09:35:00+00:00"
    assert first["exit_timestamp"] == "2024-01-01T09:40:00+00:00"
    assert first["gross_return"] == pytest.approx(0.10)
    assert first["transaction_cost_fraction"] == pytest.approx(0.0015)
    assert first["net_return"] == pytest.approx(0.0985)
    assert result.summary["total_return"] == result.summary["net_total_return"]
    assert result.summary["total_return_semantics"] == "net_total_return"


def test_bar_cadence_replay_emits_buy_sell_hold_decisions():
    result = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        cost_bps=0,
        slippage_bps=0,
    )

    decisions_by_time = {}
    for row in result.decisions:
        decisions_by_time.setdefault(row["decision_timestamp"], {})[row["symbol"]] = row["action"]

    assert decisions_by_time["2024-01-01T09:30:00+00:00"] == {"AAA": "BUY"}
    assert decisions_by_time["2024-01-01T09:35:00+00:00"] == {"AAA": "HOLD"}
    assert decisions_by_time["2024-01-01T09:40:00+00:00"] == {
        "AAA": "SELL",
        "BBB": "BUY",
    }
    assert result.summary["action_counts"]["BUY"] == 2
    assert result.summary["action_counts"]["SELL"] == 1
    assert result.summary["action_counts"]["HOLD"] == 1


def test_decision_cadence_can_differ_from_scoring_cadence():
    result = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        decision_frequency_bars=2,
    )

    assert result.payload["score_count"] == 10
    assert result.payload["decision_cadence"] == "every_2_5m_bar"
    assert [row["decision_timestamp"] for row in result.periods] == [
        "2024-01-01T09:30:00+00:00"
    ]
    assert result.periods[0]["execution_timestamp"] == "2024-01-01T09:35:00+00:00"
    assert result.periods[0]["exit_timestamp"] == "2024-01-01T09:45:00+00:00"


def test_parallel_configuration_keeps_stateful_replay_chronological():
    sequential = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        max_workers=1,
    )
    parallel = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        max_workers=4,
    )

    assert parallel.periods == sequential.periods
    assert parallel.decisions == sequential.decisions
    assert parallel.summary == sequential.summary
    assert parallel.payload["parallelism"]["requested_workers"] == 4
    assert parallel.payload["parallelism"]["effective_workers"] == 1
    assert parallel.payload["parallelism"]["stateful_execution_workers"] == 1
    assert parallel.payload["parallelism"]["independent_unit_workers"] == 4
    assert parallel.payload["parallelism"]["full_dataset_copy_per_worker"] is False


def test_bar_cadence_replay_compounds_gross_and_net_equity_after_costs():
    rows = [
        {
            "timestamp": "2024-01-01T09:30:00+00:00",
            "symbol": "AAA",
            "timeframe": "1Day",
            "open": 100.0,
            "close": 100.0,
            "score": 1.0,
        },
        {
            "timestamp": "2024-01-02T09:30:00+00:00",
            "symbol": "AAA",
            "timeframe": "1Day",
            "open": 110.0,
            "close": 110.0,
            "score": 1.0,
        },
        {
            "timestamp": "2024-01-03T09:30:00+00:00",
            "symbol": "AAA",
            "timeframe": "1Day",
            "open": 121.0,
            "close": 121.0,
            "score": 1.0,
        },
    ]

    result = build_bar_cadence_portfolio_replay(
        rows,
        signal_column="score",
        timeframe="1Day",
        top_n=1,
        max_position_weight=1.0,
        cost_bps=10,
        slippage_bps=0,
        starting_equity=1000.0,
    )

    assert len(result.periods) == 1
    period = result.periods[0]
    assert period["gross_return"] == pytest.approx(0.10)
    assert period["transaction_cost_fraction"] == pytest.approx(0.001)
    assert period["transaction_cost_amount"] == pytest.approx(1.0)
    assert period["gross_equity"] == pytest.approx(1100.0)
    assert period["net_equity"] == pytest.approx(1099.0)
    assert result.summary["ending_gross_equity"] == pytest.approx(1100.0)
    assert result.summary["ending_net_equity"] == pytest.approx(1099.0)
    assert result.summary["gross_total_return"] == pytest.approx(0.10)
    assert result.summary["net_total_return"] == pytest.approx(0.099)
    assert result.summary["total_return"] == pytest.approx(result.summary["ending_net_equity"] / 1000.0 - 1.0)
    assert result.summary["total_transaction_cost_amount"] == pytest.approx(1.0)
    assert result.summary["transaction_cost_fraction_of_starting_equity"] == pytest.approx(0.001)
    assert result.summary["return_drag_attributable_to_costs"] == pytest.approx(0.001)


def test_bar_cadence_replay_zero_cost_invariant():
    zero_cost = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        cost_bps=0,
        slippage_bps=0,
    )
    positive_cost = build_bar_cadence_portfolio_replay(
        _rows(),
        signal_column="score",
        timeframe="5m",
        top_n=1,
        max_position_weight=1.0,
        cost_bps=10,
        slippage_bps=5,
    )

    assert zero_cost.summary["ending_gross_equity"] == pytest.approx(zero_cost.summary["ending_net_equity"])
    assert zero_cost.summary["gross_total_return"] == pytest.approx(zero_cost.summary["net_total_return"])
    assert zero_cost.summary["cost_drag"] == pytest.approx(0.0)
    assert positive_cost.summary["ending_net_equity"] < positive_cost.summary["ending_gross_equity"]
    assert positive_cost.summary["cost_drag"] > 0.0


@pytest.mark.parametrize(
    ("timeframe", "factor", "min_periods"),
    [("1Day", 252.0, 20), ("1h", 1638.0, 65), ("5m", 19656.0, 390)],
)
def test_bar_cadence_replay_uses_timeframe_annualization(timeframe, factor, min_periods):
    rows = []
    for timestamp, open_price in [
        ("2024-01-01T09:30:00+00:00", 100.0),
        ("2024-01-01T09:35:00+00:00", 101.0),
        ("2024-01-01T09:40:00+00:00", 102.0),
    ]:
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "AAA",
                "timeframe": timeframe,
                "open": open_price,
                "close": open_price,
                "score": 1.0,
            }
        )

    result = build_bar_cadence_portfolio_replay(
        rows,
        signal_column="score",
        timeframe=timeframe,
        top_n=1,
        max_position_weight=1.0,
    )

    assert result.summary["annualization_factor"] == pytest.approx(factor)
    assert result.summary["annualization_min_cagr_periods"] == min_periods
    assert result.summary["cagr"] is None


def _rows():
    bars = [
        ("2024-01-01T09:30:00+00:00", 100.0, 100.0, 0.90, 0.10),
        ("2024-01-01T09:35:00+00:00", 100.0, 20.0, 0.80, 0.20),
        ("2024-01-01T09:40:00+00:00", 110.0, 22.0, 0.10, 0.90),
        ("2024-01-01T09:45:00+00:00", 100.0, 24.0, 0.20, 0.80),
        ("2024-01-01T09:50:00+00:00", 100.0, 25.0, 0.30, 0.70),
    ]
    rows = []
    for timestamp, aaa_open, bbb_open, aaa_score, bbb_score in bars:
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "AAA",
                "timeframe": "5m",
                "open": aaa_open,
                "close": aaa_open,
                "score": aaa_score,
            }
        )
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "BBB",
                "timeframe": "5m",
                "open": bbb_open,
                "close": bbb_open,
                "score": bbb_score,
            }
        )
    return rows
