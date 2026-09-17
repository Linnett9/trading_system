from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.research.ml.ds24.cost_aware_policy_engine import (
    ResizeExecutionMode,
    DailyFeeAccumulator,
    CostContract,
    EvidenceState,
    FeeSchedule,
    InsufficientCashForFees,
    MarketQuote,
    MarkPrice,
    PolicyContract,
    PortfolioLedger,
    Position,
    RankSignal,
    RetentionMode,
    TradeSide,
    apply_weight_no_trade_band,
    rank_weights,
    select_symbols,
    validate_policy_admission,
)


def test_alpaca_fee_types_round_separately_after_daily_account_aggregation() -> None:
    schedule = FeeSchedule(
        authority="ALPACA_2026_09_01",
        retrieval_date="2026-09-17",
        source="https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf",
        sec_sell_notional_rate=0.0000206,
        taf_sell_per_share=0.000195,
        taf_trade_cap_usd=9.79,
        cat_per_share=0.000003,
    )
    fees = DailyFeeAccumulator(schedule)
    first = fees.record(TradeSide.BUY, 1.0, 100.0, TS)
    second = fees.record(TradeSide.BUY, 1.0, 100.0, TS + timedelta(minutes=5))
    assert (first, second) == pytest.approx((0.01, 0.0))

    sale = fees.record(TradeSide.SELL, 1.0, 100.0, TS + timedelta(minutes=10))
    assert sale == pytest.approx(0.02)  # SEC and TAF round independently; CAT already paid.
    assert fees.total_paid == pytest.approx(0.03)


def test_alpaca_taf_cap_applies_per_trade_before_daily_rounding() -> None:
    schedule = FeeSchedule(
        authority="ALPACA_2026_09_01",
        retrieval_date="2026-09-17",
        source="https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf",
        taf_sell_per_share=0.000195,
        taf_trade_cap_usd=9.79,
    )
    fees = DailyFeeAccumulator(schedule)
    assert fees.record(TradeSide.SELL, 100_000.0, 1.0, TS) == pytest.approx(9.79)
    assert fees.record(TradeSide.SELL, 100_000.0, 1.0, TS + timedelta(minutes=5)) == pytest.approx(9.79)
    assert fees.total_paid == pytest.approx(19.58)


def test_alpaca_daily_fee_bucket_resets_on_next_new_york_session() -> None:
    fees = DailyFeeAccumulator(FeeSchedule(
        authority="fixture", retrieval_date="2026-09-17", source="synthetic",
        cat_per_share=0.000003,
    ))
    assert fees.record(TradeSide.BUY, 1.0, 100.0, TS) == pytest.approx(0.01)
    assert fees.record(TradeSide.BUY, 1.0, 100.0, TS + timedelta(days=1)) == pytest.approx(0.01)
    assert fees.total_paid == pytest.approx(0.02)


def test_fee_preview_does_not_charge_cash_until_trade_is_recorded() -> None:
    fees = DailyFeeAccumulator(FeeSchedule(
        authority="fixture", retrieval_date="2026-09-17", source="synthetic",
        cat_per_share=0.000003,
    ))
    assert fees.preview(TradeSide.BUY, 1.0, 100.0, TS) == pytest.approx(0.01)
    assert fees.preview(TradeSide.BUY, 1.0, 100.0, TS) == pytest.approx(0.01)
    assert fees.total_paid == 0.0
    assert fees.record(TradeSide.BUY, 1.0, 100.0, TS) == pytest.approx(0.01)


def test_ledger_applies_daily_aggregated_fee_once_across_two_buys() -> None:
    ledger = PortfolioLedger("two_buys")
    schedule = FeeSchedule(
        authority="fixture", retrieval_date="2026-09-17", source="synthetic",
        cat_per_share=0.000003,
    )
    result = ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("A", 1, 1.0), RankSignal("B", 2, 0.5)],
        policy=policy(entry_top_n=2, exit_rank=2),
        cost=CostContract(schedule, 0.0, "SYNTHETIC", timedelta(0)),
        quotes={"A": quote("A", 100.0, 100.0), "B": quote("B", 100.0, 100.0)},
        marks={"A": mark("A", 100.0), "B": mark("B", 100.0)},
    )
    assert len(ledger.trades) == 2
    assert ledger.explicit_fees == pytest.approx(0.01)
    assert sum(trade.explicit_fee for trade in ledger.trades) == pytest.approx(0.01)
    assert result.cash + result.position_value == pytest.approx(result.equity)


def test_fee_contract_rejects_malformed_trade_amounts() -> None:
    schedule = FeeSchedule(authority="fixture", retrieval_date="2026-09-17", source="synthetic")
    with pytest.raises(ValueError, match="finite"):
        schedule.raw_components(TradeSide.BUY, float("nan"), 100.0)
    with pytest.raises(ValueError, match="finite"):
        schedule.raw_components(TradeSide.BUY, 1.0, float("inf"))
    with pytest.raises(ValueError, match="trade side"):
        schedule.raw_components("HOLD", 1.0, 100.0)


def test_exit_hysteresis_never_exceeds_simultaneous_holding_capacity() -> None:
    held = {
        "OLD_A": Position("OLD_A", 1.0, 100.0, TS, last_rank=25),
        "OLD_B": Position("OLD_B", 1.0, 100.0, TS, last_rank=26),
    }
    signals = [
        RankSignal("NEW_A", 1, 1.0), RankSignal("NEW_B", 2, 0.9),
        RankSignal("OLD_A", 25, 0.2), RankSignal("OLD_B", 26, 0.1),
    ]
    selected = select_symbols(signals, held, TS + timedelta(minutes=60),
                              policy(entry_top_n=2, exit_rank=30))
    assert selected == ("NEW_A", "NEW_B")
    assert len(selected) == 2


def test_r3_wide_exit_band_does_not_authorize_more_than_k_holdings() -> None:
    held = {
        f"OLD_{index}": Position(f"OLD_{index}", 1.0, 100.0, TS, last_rank=10 + index)
        for index in range(1, 6)
    }
    signals = [
        RankSignal("NEW_TOP", 1, 2.0),
        *[RankSignal(symbol, 10 + index, 1.0 - index / 100.0)
          for index, symbol in enumerate(held, start=1)],
    ]

    selected = select_symbols(
        signals,
        held,
        TS + timedelta(minutes=60),
        policy(
            entry_top_n=5,
            exit_rank=20,
            max_holdings=5,
            retention_mode=RetentionMode.PURE_RETENTION,
        ),
    )

    assert set(selected) == set(held)
    assert len(selected) == 5


def test_r3_challenger_override_replaces_weakest_incumbent_when_margin_passes() -> None:
    held = {
        "OLD_STRONG": Position("OLD_STRONG", 1.0, 100.0, TS, last_rank=8),
        "OLD_WEAK": Position("OLD_WEAK", 1.0, 100.0, TS, last_rank=18),
    }
    signals = [
        RankSignal("NEW", 4, 2.0),
        RankSignal("OLD_STRONG", 8, 1.0),
        RankSignal("OLD_WEAK", 18, 0.5),
    ]

    selected = select_symbols(
        signals,
        held,
        TS + timedelta(minutes=60),
        policy(
            entry_top_n=5,
            exit_rank=20,
            max_holdings=2,
            retention_mode=RetentionMode.CHALLENGER_OVERRIDE,
            minimum_rank_improvement=10,
        ),
    )

    assert selected == ("NEW", "OLD_STRONG")


def test_r3_challenger_override_respects_minimum_rank_improvement() -> None:
    held = {"OLD": Position("OLD", 1.0, 100.0, TS, last_rank=11)}
    signals = [RankSignal("NEW", 4, 2.0), RankSignal("OLD", 11, 1.0)]

    selected = select_symbols(
        signals,
        held,
        TS + timedelta(minutes=60),
        policy(
            entry_top_n=5,
            exit_rank=20,
            max_holdings=1,
            retention_mode=RetentionMode.CHALLENGER_OVERRIDE,
            minimum_rank_improvement=10,
        ),
    )

    assert selected == ("OLD",)


def test_protected_hold_cannot_be_displaced_when_capacity_is_full() -> None:
    held = {
        "OLD_A": Position("OLD_A", 1.0, 100.0, TS, last_rank=25),
        "OLD_B": Position("OLD_B", 1.0, 100.0, TS, last_rank=26),
    }
    signals = [
        RankSignal("NEW", 1, 1.0), RankSignal("OLD_A", 25, 0.2),
        RankSignal("OLD_B", 26, 0.1),
    ]
    selected = select_symbols(signals, held, TS + timedelta(minutes=15),
                              policy(entry_top_n=2, exit_rank=30,
                                     minimum_hold=timedelta(minutes=30)))
    assert set(selected) == {"OLD_A", "OLD_B"}


def test_missing_one_buy_preserves_other_fill_and_valued_account() -> None:
    ledger = PortfolioLedger("partial_execution")
    snapshot = ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("A", 1, 1.0), RankSignal("B", 2, 0.9)],
        policy=policy(entry_top_n=2, exit_rank=2),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"A": quote("A", 100.0, 100.0)},
        marks={"A": mark("A", 100.0), "B": mark("B", 100.0)},
    )
    assert len(ledger.trades) == 1
    assert "A" in ledger.positions and "B" not in ledger.positions
    assert snapshot.evidence_state is EvidenceState.MISSING_EXECUTION_QUOTE
    assert snapshot.unresolved_symbols == ("B",)
    assert snapshot.cash + snapshot.position_value == pytest.approx(snapshot.equity)
    assert snapshot.equity > snapshot.cash


def test_missing_new_target_mark_leaves_target_cash_idle() -> None:
    ledger = PortfolioLedger("partial_mark")
    snapshot = ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("A", 1, 1.0), RankSignal("B", 2, 0.9)],
        policy=policy(entry_top_n=2, exit_rank=2),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"A": quote("A", 100.0, 100.0)},
        marks={"A": mark("A", 100.0)},
    )
    assert "A" in ledger.positions and "B" not in ledger.positions
    assert snapshot.evidence_state is EvidenceState.MISSING_MARK
    assert snapshot.unresolved_symbols == ("B",)
    assert snapshot.cash + snapshot.position_value == pytest.approx(snapshot.equity)


TS = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)


def fee_schedule() -> FeeSchedule:
    return FeeSchedule(
        authority="ALPACA_TEST_FEE_SCHEDULE",
        retrieval_date="2026-09-17",
        source="synthetic unit fixture",
        sec_sell_notional_rate=0.0000229,
        taf_sell_per_share=0.000166,
        taf_trade_cap_usd=8.30,
        cat_per_share=0.000022,
    )


def cost(extra_slippage_bps: float = 2.5) -> CostContract:
    return CostContract(
        fee_schedule=fee_schedule(),
        extra_slippage_bps_per_side=extra_slippage_bps,
        quote_source="SYNTHETIC_SIP",
        stale_after=timedelta(minutes=1),
    )


def policy(**overrides: object) -> PolicyContract:
    values = {
        "policy_id": "fixture",
        "entry_top_n": 2,
        "exit_rank": 3,
        "weighting_mode": "equal_weight_control",
    }
    values.update(overrides)
    return PolicyContract(**values)


def quote(symbol: str, bid: float, ask: float, ts: datetime = TS) -> MarketQuote:
    return MarketQuote(symbol=symbol, timestamp=ts, bid=bid, ask=ask)


def mark(symbol: str, price: float, ts: datetime = TS) -> MarkPrice:
    return MarkPrice(symbol=symbol, timestamp=ts, price=price)


def signals() -> list[RankSignal]:
    return [
        RankSignal("AAA", 1, 0.9),
        RankSignal("BBB", 2, 0.8),
        RankSignal("CCC", 3, 0.7),
    ]


def test_cash_plus_positions_equals_equity_after_buy() -> None:
    ledger = PortfolioLedger("elastic_net")

    snap = ledger.rebalance(
        timestamp=TS,
        signals=signals(),
        policy=policy(),
        cost=cost(),
        quotes={"AAA": quote("AAA", 99.90, 100.10), "BBB": quote("BBB", 49.95, 50.05)},
        marks={"AAA": mark("AAA", 100.00), "BBB": mark("BBB", 50.00)},
    )

    assert snap.evidence_state is EvidenceState.COMPLETE
    assert snap.cash + snap.position_value == pytest.approx(snap.equity)


def test_buying_reduces_cash_and_charges_buy_side_costs_once() -> None:
    ledger = PortfolioLedger("elastic_net")

    ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=10.0),
        quotes={"AAA": quote("AAA", 99.00, 101.00)},
        marks={"AAA": mark("AAA", 100.00)},
    )

    trade = ledger.trades[0]
    assert trade.side is TradeSide.BUY
    assert ledger.cash < ledger.starting_equity
    assert trade.execution_price > trade.reference_price
    assert trade.spread_cost > 0
    assert trade.extra_slippage_cost > 0
    assert ledger.explicit_fees == pytest.approx(sum(t.explicit_fee for t in ledger.trades))


def test_selling_increases_cash_and_deducts_explicit_fees_once() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=10.0, average_cost=90.0, opened_at=TS - timedelta(hours=1), last_rank=1)
    ledger.cash = 4_000.0

    ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("AAA", 10, 0.1)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": quote("AAA", 99.00, 101.00)},
        marks={"AAA": mark("AAA", 100.00)},
    )

    trade = ledger.trades[0]
    assert trade.side is TradeSide.SELL
    assert ledger.cash == pytest.approx(4_000.0 + trade.trade_notional - trade.explicit_fee)
    assert ledger.explicit_fees == pytest.approx(trade.explicit_fee)
    assert ledger.realized_pnl == pytest.approx(10.0 * (99.00 - 90.0) - trade.explicit_fee)


def test_spread_and_slippage_are_charged_in_the_correct_direction() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=100.0),
        quotes={"AAA": quote("AAA", 98.00, 102.00)},
        marks={"AAA": mark("AAA", 100.00)},
    )
    buy = ledger.trades[0]
    ledger.rebalance(
        timestamp=TS + timedelta(minutes=5),
        signals=[RankSignal("AAA", 5, 0.0)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=100.0),
        quotes={"AAA": quote("AAA", 98.00, 102.00, TS + timedelta(minutes=5))},
        marks={"AAA": mark("AAA", 100.00, TS + timedelta(minutes=5))},
    )
    sell = ledger.trades[-1]

    assert buy.execution_price > buy.reference_price
    assert sell.execution_price < sell.reference_price
    assert buy.spread_cost > 0
    assert sell.spread_cost > 0


def test_existing_holding_is_not_sold_and_rebought_when_target_is_unchanged() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=50.0, average_cost=100.0, opened_at=TS, last_rank=1)
    ledger.cash = 0.0

    ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": quote("AAA", 100.00, 100.00)},
        marks={"AAA": mark("AAA", 100.00)},
    )

    assert len(ledger.trades) == 0


def test_same_symbol_across_virtual_sleeves_uses_a_common_mark_and_net_delta() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=50.0, average_cost=100.0, opened_at=TS - timedelta(minutes=30), last_rank=1)
    ledger.cash = 0.0

    ledger.rebalance(
        timestamp=TS,
        signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": quote("AAA", 100.00, 100.00)},
        marks={"AAA": mark("AAA", 100.00)},
    )

    assert len(ledger.trades) == 0
    assert ledger.snapshot(TS, {"AAA": mark("AAA", 100.00)}).equity == pytest.approx(5_000.0)


def test_duplicate_decisions_append_snapshots_without_dropping_capital() -> None:
    ledger = PortfolioLedger("elastic_net")
    kwargs = {
        "timestamp": TS,
        "signals": [RankSignal("AAA", 1, 1.0)],
        "policy": policy(entry_top_n=1, exit_rank=1),
        "cost": cost(extra_slippage_bps=0.0),
        "quotes": {"AAA": quote("AAA", 100.00, 100.00)},
        "marks": {"AAA": mark("AAA", 100.00)},
    }

    ledger.rebalance(**kwargs)
    ledger.rebalance(**kwargs)

    assert len(ledger.snapshots) == 2
    assert ledger.snapshots[-1].equity == pytest.approx(ledger.snapshots[0].equity)


def test_missing_marks_leave_unresolved_exposure() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=10.0, average_cost=100.0, opened_at=TS, last_rank=1)

    snap = ledger.snapshot(TS + timedelta(minutes=5), {})

    assert snap.evidence_state is EvidenceState.MISSING_MARK
    assert snap.unresolved_symbols == ("AAA",)


def test_rebalance_with_missing_open_position_mark_fails_closed() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=10.0, average_cost=100.0, opened_at=TS, last_rank=1)

    snap = ledger.rebalance(
        timestamp=TS + timedelta(minutes=5),
        signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": quote("AAA", 99.00, 101.00, TS + timedelta(minutes=5))},
        marks={},
    )

    assert snap.evidence_state is EvidenceState.MISSING_MARK
    assert snap.unresolved_symbols == ("AAA",)
    assert len(ledger.trades) == 0


def test_missing_final_mark_does_not_move_requested_exit_earlier() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=10.0, average_cost=100.0, opened_at=TS, last_rank=1)

    snap = ledger.snapshot(TS + timedelta(minutes=60), {})

    assert snap.timestamp == TS + timedelta(minutes=60)
    assert snap.evidence_state is EvidenceState.MISSING_MARK


def test_partial_exit_sells_available_name_then_retries_only_unresolved_name() -> None:
    ledger = PortfolioLedger("xendcg")
    ledger.positions["AAA"] = Position("AAA", 10.0, 100.0, TS, 1)
    ledger.positions["BBB"] = Position("BBB", 10.0, 100.0, TS, 2)
    ledger.cash = 3000.0
    hold_policy = policy(entry_top_n=2, exit_rank=2, minimum_hold=timedelta(minutes=60))
    zero_cost = cost(extra_slippage_bps=0.0)
    exit_ts = TS + timedelta(minutes=60)

    first = ledger.liquidate_available(
        timestamp=exit_ts, policy=hold_policy, cost=zero_cost,
        quotes={"AAA": quote("AAA", 101.0, 101.0, exit_ts)},
        marks={"AAA": mark("AAA", 101.0, exit_ts)},
    )

    assert first.evidence_state is EvidenceState.MISSING_MARK
    assert first.unresolved_symbols == ("BBB",)
    assert "AAA" not in ledger.positions
    assert "BBB" in ledger.positions
    assert len(ledger.trades) == 1

    retry_ts = exit_ts + timedelta(minutes=5)
    final = ledger.liquidate_available(
        timestamp=retry_ts, policy=hold_policy, cost=zero_cost,
        quotes={"BBB": quote("BBB", 99.0, 99.0, retry_ts)},
        marks={"BBB": mark("BBB", 99.0, retry_ts)},
    )

    assert final.evidence_state is EvidenceState.COMPLETE
    assert final.equity == pytest.approx(5000.0 - ledger.explicit_fees)
    assert ledger.positions == {}
    assert len(ledger.trades) == 2


def test_cash_constrained_zero_buy_does_not_create_a_position_or_trade() -> None:
    ledger = PortfolioLedger("cash_constrained")
    ledger.cash = 0.005

    snapshot = ledger.rebalance(
        timestamp=TS, signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1), cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": quote("AAA", 100.0, 100.0)},
        marks={"AAA": mark("AAA", 100.0)},
    )

    assert snapshot.evidence_state is EvidenceState.MISSING_EXECUTION_QUOTE
    assert ledger.positions == {}
    assert ledger.trades == []


def test_trade_records_quote_level_execution_source() -> None:
    ledger = PortfolioLedger("source_attribution")
    ledger.rebalance(
        timestamp=TS, signals=[RankSignal("AAA", 1, 1.0)],
        policy=policy(entry_top_n=1, exit_rank=1), cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": MarketQuote("AAA", TS, 100.0, 100.0,
                                   source="ALPACA_HISTORICAL_SIP")},
        marks={"AAA": mark("AAA", 100.0)},
    )
    assert ledger.trades[0].execution_source == "ALPACA_HISTORICAL_SIP"


def test_liquidation_clears_legacy_zero_quantity_position() -> None:
    ledger = PortfolioLedger("legacy_zero")
    ledger.positions["AAA"] = Position("AAA", 0.0, 100.0, TS, 1)

    snapshot = ledger.liquidate_available(
        timestamp=TS + timedelta(minutes=60), policy=policy(entry_top_n=1, exit_rank=1),
        cost=cost(extra_slippage_bps=0.0), quotes={}, marks={},
    )

    assert snapshot.evidence_state is EvidenceState.COMPLETE
    assert ledger.positions == {}


def test_fee_larger_than_tiny_sale_proceeds_never_borrows_or_mutates_fee_state() -> None:
    ledger = PortfolioLedger("no_borrowing")
    ledger.cash = 0.0
    ledger.positions["AAA"] = Position("AAA", 0.001, 0.01, TS, 1)
    hold_policy = policy(entry_top_n=1, exit_rank=1)
    fee_cost = cost(extra_slippage_bps=0.0)

    with pytest.raises(InsufficientCashForFees, match="without borrowing"):
        ledger.liquidate_available(
            timestamp=TS + timedelta(minutes=60), policy=hold_policy, cost=fee_cost,
            quotes={"AAA": MarketQuote("AAA", TS + timedelta(minutes=60), 0.01, 0.01)},
            marks={"AAA": MarkPrice("AAA", TS + timedelta(minutes=60), 0.01)},
        )

    assert ledger.cash == 0.0
    assert ledger.positions["AAA"].quantity == 0.001
    assert ledger.explicit_fees == 0.0
    assert ledger._fee_accumulator is not None and ledger._fee_accumulator.total_paid == 0.0


def test_resume_checkpoints_reject_incompatible_cost_or_policy_configurations() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.bind_resume_contracts(policy(), cost())

    with pytest.raises(ValueError, match="policy"):
        ledger.bind_resume_contracts(policy(exit_rank=4), cost())
    cost_ledger = PortfolioLedger("elastic_net")
    cost_ledger.bind_resume_contracts(policy(), cost())
    with pytest.raises(ValueError, match="cost"):
        cost_ledger.bind_resume_contracts(policy(), cost(5.0))


def test_resumed_run_equals_uninterrupted_run() -> None:
    uninterrupted = PortfolioLedger("elastic_net")
    resumed = PortfolioLedger("elastic_net")
    kwargs = {
        "timestamp": TS,
        "signals": [RankSignal("AAA", 1, 1.0)],
        "policy": policy(entry_top_n=1, exit_rank=1),
        "cost": cost(extra_slippage_bps=0.0),
        "quotes": {"AAA": quote("AAA", 100.00, 100.00)},
        "marks": {"AAA": mark("AAA", 100.00)},
    }

    uninterrupted.rebalance(**kwargs)
    resumed.bind_resume_contracts(kwargs["policy"], kwargs["cost"])
    resumed.rebalance(**kwargs)

    assert resumed.cash == pytest.approx(uninterrupted.cash)
    assert resumed.positions["AAA"].quantity == pytest.approx(uninterrupted.positions["AAA"].quantity)


def test_future_completeness_filtering_is_rejected_for_policy_admission() -> None:
    with pytest.raises(ValueError, match="future-completeness"):
        validate_policy_admission(
            candidate_symbols=["AAA"],
            available_symbols=["AAA"],
            future_complete_filter_used=True,
        )


def test_model_accounts_do_not_cross_or_net_positions() -> None:
    elastic = PortfolioLedger("elastic_net")
    huber = PortfolioLedger("huber")
    elastic.positions["AAA"] = Position("AAA", quantity=10.0, average_cost=100.0, opened_at=TS, last_rank=1)
    huber.positions["AAA"] = Position("AAA", quantity=3.0, average_cost=100.0, opened_at=TS, last_rank=1)

    elastic.snapshot(TS, {"AAA": mark("AAA", 100.00)})
    huber.snapshot(TS, {"AAA": mark("AAA", 100.00)})

    assert elastic.account_id != huber.account_id
    assert elastic.positions["AAA"].quantity == 10.0
    assert huber.positions["AAA"].quantity == 3.0


def test_sqrt_inverse_rank_weights_sum_to_one_and_decrease() -> None:
    selected = ["AAA", "BBB", "CCC"]

    weights = rank_weights(signals(), selected, "sqrt_inverse_rank")

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["AAA"] > weights["BBB"] > weights["CCC"]


def test_capped_inverse_rank_requires_registered_cap_and_limits_top_weight() -> None:
    selected = ["AAA", "BBB", "CCC"]

    with pytest.raises(ValueError, match="rank_weight_cap"):
        rank_weights(signals(), selected, "capped_inverse_rank")
    weights = rank_weights(signals(), selected, "capped_inverse_rank", rank_weight_cap=0.50)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["AAA"] == pytest.approx(0.50)
    assert weights["BBB"] > weights["CCC"]


def test_weight_no_trade_band_can_hold_or_trade_to_boundary() -> None:
    current = {"AAA": 0.10}
    target = {"AAA": 0.106}

    held = apply_weight_no_trade_band(
        target_weights=target,
        current_weights=current,
        retained_symbols={"AAA"},
        band_bps=100,
        mode=ResizeExecutionMode.FULL_TARGET,
    )
    boundary = apply_weight_no_trade_band(
        target_weights={"AAA": 0.12},
        current_weights={"AAA": 0.09},
        retained_symbols={"AAA"},
        band_bps=100,
        mode=ResizeExecutionMode.TRADE_TO_BAND_BOUNDARY,
    )

    assert held["AAA"] == pytest.approx(0.10)
    assert boundary["AAA"] == pytest.approx(0.11)


def test_rank_weighting_uses_only_rank_not_realized_future_returns() -> None:
    selected = ["AAA", "BBB"]

    base = rank_weights(signals(), selected, "sqrt_inverse_rank")
    same_ranks_with_changed_scores = rank_weights(
        [RankSignal("AAA", 1, -100.0), RankSignal("BBB", 2, 1000.0)],
        selected,
        "sqrt_inverse_rank",
    )

    assert same_ranks_with_changed_scores == base


def test_minimum_hold_blocks_rank_exit_until_protected_interval_expires() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["AAA"] = Position("AAA", quantity=50.0, average_cost=100.0, opened_at=TS, last_rank=1)
    ledger.cash = 0.0

    ledger.rebalance(
        timestamp=TS + timedelta(minutes=30),
        signals=[RankSignal("AAA", 99, 0.0)],
        policy=policy(entry_top_n=1, exit_rank=1, minimum_hold=timedelta(minutes=60)),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"AAA": quote("AAA", 99.00, 101.00, TS + timedelta(minutes=30))},
        marks={"AAA": mark("AAA", 100.00, TS + timedelta(minutes=30))},
    )

    assert len(ledger.trades) == 0
    assert ledger.positions["AAA"].quantity == 50.0


def test_replacement_buffer_requires_deterministic_rank_improvement() -> None:
    ledger = PortfolioLedger("elastic_net")
    ledger.positions["OLD"] = Position("OLD", quantity=50.0, average_cost=100.0, opened_at=TS, last_rank=27)
    ledger.cash = 0.0

    ledger.rebalance(
        timestamp=TS + timedelta(minutes=70),
        signals=[RankSignal("NEW", 19, 1.0), RankSignal("OLD", 27, 0.5)],
        policy=policy(entry_top_n=20, exit_rank=30, minimum_rank_improvement=10),
        cost=cost(extra_slippage_bps=0.0),
        quotes={"OLD": quote("OLD", 100.0, 100.0, TS + timedelta(minutes=70)), "NEW": quote("NEW", 100.0, 100.0, TS + timedelta(minutes=70))},
        marks={"OLD": mark("OLD", 100.0, TS + timedelta(minutes=70)), "NEW": mark("NEW", 100.0, TS + timedelta(minutes=70))},
    )

    assert "OLD" in ledger.positions
    assert "NEW" not in ledger.positions
