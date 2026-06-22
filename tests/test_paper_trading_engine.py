import json

import pytest

from core.paper.paper_trading_engine import PaperTradingEngine


def test_no_order_fill_preserves_decision_equity(tmp_path):
    engine = PaperTradingEngine(report_dir=tmp_path, starting_cash=500)
    decision_path = tmp_path / "2026-06-01_decision.json"
    decision_path.write_text(
        json.dumps({
            "timestamp": "2026-06-01T04:00:00+00:00",
            "cash": 162.5,
            "equity": 500.0,
            "orders": [],
        }),
        encoding="utf-8",
    )
    engine.state_path.write_text(
        json.dumps({
            "cash": 162.5,
            "positions": {"AMAT": 0.13},
            "filled_decision_paths": [],
        }),
        encoding="utf-8",
    )

    fill_record = engine.fill_latest_decision(decision_path)

    assert fill_record["status"] == "no_orders"
    assert fill_record["equity_after"] == 500.0


def test_fractional_order_quantity_uses_precision():
    engine = PaperTradingEngine(
        supports_fractional=True,
        quantity_precision=3,
        order_type="limit",
        limit_offset_bps=10,
    )

    orders = engine._orders(
        target_weights={"AMAT": 1.0},
        exposure_target=1.0,
        positions={},
        prices_by_symbol={"AMAT": 300},
        equity=1000,
        selected_symbols=["AMAT"],
        scores={"AMAT": 1.0},
        model_context={"selection_mode": "ranked", "top_n": 1},
        rebalance_threshold=0,
    )

    assert orders[0].quantity_delta == 3.333
    assert orders[0].order_type == "LIMIT"
    assert orders[0].limit_price == pytest.approx(300.3)


def test_whole_share_order_sizing_skips_subshare_buy():
    engine = PaperTradingEngine(
        supports_fractional=False,
        min_trade_value=1,
    )

    orders = engine._orders(
        target_weights={"AMAT": 0.10},
        exposure_target=1.0,
        positions={},
        prices_by_symbol={"AMAT": 300},
        equity=1000,
        selected_symbols=["AMAT"],
        scores={"AMAT": 1.0},
        model_context={"selection_mode": "ranked", "top_n": 1},
        rebalance_threshold=0,
    )

    assert orders == []


def test_whole_share_sell_does_not_exceed_available_whole_quantity():
    engine = PaperTradingEngine(
        supports_fractional=False,
        min_trade_value=1,
    )

    orders = engine._orders(
        target_weights={"AMAT": 0.0},
        exposure_target=1.0,
        positions={"AMAT": 1.5},
        prices_by_symbol={"AMAT": 300},
        equity=1000,
        selected_symbols=[],
        scores={},
        model_context={"selection_mode": "ranked", "top_n": 1},
        rebalance_threshold=0,
    )

    assert orders[0].quantity_delta == -1.0
    assert orders[0].dollar_delta == -300
