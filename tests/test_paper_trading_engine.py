from datetime import datetime
from types import SimpleNamespace

from core.paper.paper_trading_engine import PaperTradingEngine


def test_paper_trading_engine_creates_initial_buy_decision(tmp_path):
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="risk-on",
        risk_on=True,
        exposure_target=0.80,
        target_weights={"AAPL": 0.50, "SPY": 0.50},
        symbols=["AAPL", "SPY"],
        scores={"AAPL": 0.12, "SPY": 0.08},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "all_positive",
            "ranking_score_mode": "average_momentum",
            "top_n": 5,
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )

    decision = engine.create_decision(
        result,
        prices_by_symbol={"AAPL": 100, "SPY": 50},
    )

    assert decision.equity == 500
    assert len(decision.orders) == 2
    assert {order.symbol for order in decision.orders} == {"AAPL", "SPY"}
    assert sum(order.dollar_delta for order in decision.orders) == 400
    assert decision.model_context["selection_mode"] == "all_positive"
    assert "positive momentum" in decision.orders[0].reason
    assert decision.report_path.exists()
    assert decision.state_path.exists()


def test_paper_trading_engine_sells_symbols_no_longer_selected(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        '{"cash": 100, "positions": {"AAPL": 2, "MSFT": 1}}',
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="risk-on",
        risk_on=True,
        exposure_target=0.50,
        target_weights={"AAPL": 1.0},
        symbols=["AAPL"],
        scores={"AAPL": 0.10},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "ranked",
            "ranking_score_mode": "enhanced",
            "top_n": 1,
            "min_selection_score": 0.03,
            "max_selected_assets": 1,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )

    decision = engine.create_decision(
        result,
        prices_by_symbol={"AAPL": 100, "MSFT": 50},
    )

    msft_order = [
        order for order in decision.orders if order.symbol == "MSFT"
    ][0]
    assert decision.equity == 350
    assert msft_order.side == "SELL"
    assert msft_order.reason == "no longer selected by current model"


def test_paper_trading_engine_explains_skipped_assets(tmp_path):
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="risk-on",
        risk_on=True,
        exposure_target=1.0,
        target_weights={"AAPL": 1.0},
        symbols=["AAPL"],
        scores={"AAPL": 0.12, "TSLA": 0.002},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "all_positive",
            "ranking_score_mode": "average_momentum",
            "top_n": 5,
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )

    decision = engine.create_decision(
        result,
        prices_by_symbol={"AAPL": 100, "TSLA": 50},
    )

    skipped = decision.model_context["skipped_assets"]
    assert skipped[0]["symbol"] == "TSLA"
    assert "below min_selection_score" in skipped[0]["reason"]


def test_paper_trading_engine_fills_latest_decision_and_updates_state(tmp_path):
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="risk-on",
        risk_on=True,
        exposure_target=1.0,
        target_weights={"AAPL": 1.0},
        symbols=["AAPL"],
        scores={"AAPL": 0.12},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "all_positive",
            "ranking_score_mode": "average_momentum",
            "top_n": 5,
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )
    engine.create_decision(result, prices_by_symbol={"AAPL": 100})

    fill_record = engine.fill_latest_decision()
    status = engine.status()

    assert fill_record["fills"][0]["symbol"] == "AAPL"
    assert status["cash"] == 0
    assert status["positions"]["AAPL"] == 5
    assert len(status["fills"]) == 1
    assert len(status["equity_history"]) == 1
    assert len(status["filled_decision_paths"]) == 1
    assert fill_record["status"] == "filled"


def test_paper_trading_engine_does_not_fill_same_decision_twice(tmp_path):
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="risk-on",
        risk_on=True,
        exposure_target=1.0,
        target_weights={"AAPL": 1.0},
        symbols=["AAPL"],
        scores={"AAPL": 0.12},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "all_positive",
            "ranking_score_mode": "average_momentum",
            "top_n": 5,
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )
    engine.create_decision(result, prices_by_symbol={"AAPL": 100})

    first_fill = engine.fill_latest_decision()
    second_fill = engine.fill_latest_decision()
    status = engine.status()

    assert first_fill["already_filled"] is False
    assert first_fill["status"] == "filled"
    assert second_fill["already_filled"] is True
    assert second_fill["status"] == "already_filled"
    assert second_fill["fills"] == []
    assert status["cash"] == 0
    assert status["positions"]["AAPL"] == 5
    assert len(status["fills"]) == 1


def test_paper_trading_engine_no_order_decision_does_not_create_fill(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        '{"cash": 500, "positions": {}, "fills": []}',
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="cash",
        risk_on=False,
        exposure_target=0.0,
        target_weights={},
        symbols=[],
        scores={},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "all_positive",
            "ranking_score_mode": "average_momentum",
            "top_n": 5,
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )
    engine.create_decision(result, prices_by_symbol={})

    fill_record = engine.fill_latest_decision()
    status = engine.status()

    assert fill_record["status"] == "no_orders"
    assert fill_record["no_orders"] is True
    assert status["cash"] == 500
    assert status["positions"] == {}
    assert len(status["fills"]) == 0


def test_paper_trading_engine_rebalance_threshold_blocks_small_drifts(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        '{"cash": 0, "positions": {"AAPL": 4.95, "SPY": 5.05}}',
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=1000,
        min_trade_value=1,
        rebalance_threshold=0.02,
    )
    selection = SimpleNamespace(
        timestamp=datetime(2026, 6, 16),
        regime_label="risk-on",
        risk_on=True,
        exposure_target=1.0,
        target_weights={"AAPL": 0.50, "SPY": 0.50},
        symbols=["AAPL", "SPY"],
        scores={"AAPL": 0.12, "SPY": 0.08},
    )
    result = SimpleNamespace(
        selections=[selection],
        config={
            "selection_mode": "all_positive",
            "ranking_score_mode": "average_momentum",
            "top_n": 5,
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "momentum_periods": [63, 126],
            "weighting": "equal",
        },
    )

    decision = engine.create_decision(
        result,
        prices_by_symbol={"AAPL": 100, "SPY": 100},
    )

    assert decision.orders == []


def test_paper_trading_engine_can_fill_specific_decision_file(tmp_path):
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )
    decision_path = tmp_path / "custom_decision.json"
    decision_path.write_text(
        """
        {
          "timestamp": "2026-06-16T00:00:00",
          "orders": [
            {
              "symbol": "MSFT",
              "side": "BUY",
              "quantity_delta": 2,
              "dollar_delta": 200,
              "price": 100,
              "reason": "test fill"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    fill_record = engine.fill_latest_decision(decision_path)

    assert fill_record["positions_after"]["MSFT"] == 2


def test_paper_trading_engine_status_can_mark_to_market_positions(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        '{"starting_cash": 500, "cash": 100, "positions": {"AAPL": 2}}',
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )

    status = engine.status(prices_by_symbol={"AAPL": 125})

    assert status["starting_cash"] == 500
    assert status["cash"] == 100
    assert status["mark_to_market_equity"] == 350
    assert status["prices_used"]["AAPL"] == 125


def test_paper_trading_engine_repairs_old_empty_fill_records(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        """
        {
          "starting_cash": 500,
          "cash": 0,
          "positions": {"AAPL": 5},
          "fills": [
            {"fills": [{"symbol": "AAPL"}], "decision_timestamp": "2026-06-01T00:00:00"},
            {"fills": [], "decision_timestamp": "2026-06-01T00:00:00"}
          ],
          "equity_history": [
            {"timestamp": "2026-06-01T00:00:00", "equity": 0, "cash": 0}
          ]
        }
        """,
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )

    repair = engine.repair_state(prices_by_symbol={"AAPL": 100})
    status = engine.status(prices_by_symbol={"AAPL": 100})

    assert repair["removed_empty_fills"] == 1
    assert status["mark_to_market_equity"] == 500
    assert len(status["fills"]) == 1
    assert len(status["equity_history"]) == 1


def test_paper_trading_engine_reset_state(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        '{"cash": 0, "positions": {"AAPL": 5}, "fills": [{"fills": []}]}',
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
        min_trade_value=1,
    )

    reset = engine.reset_state()
    status = engine.status()

    assert reset["cash"] == 500
    assert status["cash"] == 500
    assert status["positions"] == {}
    assert status["fills"] == []
