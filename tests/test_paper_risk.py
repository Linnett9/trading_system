from types import SimpleNamespace

from core.risk.paper_risk import (
    model_kill_switch_checks,
    post_trade_risk_checks,
    portfolio_kill_switch_checks,
    pre_trade_risk_checks,
    risk_blocks_submission,
    risk_status,
)


def test_pre_trade_risk_checks_pass_for_small_clean_order():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        data_quality={},
        target_weights={"AAPL": 0.20},
        exposure_target=1.0,
        equity=1000,
        cash=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                dollar_delta=200,
            ),
        ],
    )

    checks = pre_trade_risk_checks(decision, {
        "risk": {"paper": {"max_position_weight": 0.30}},
        "portfolio": {"cash_buffer_percent": 0.02},
        "broker": {"min_order_notional": 1.0},
    })

    assert risk_status(checks) == "PASS"


def test_pre_trade_risk_checks_block_unpriced_legacy_holdings():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False}, data_quality={},
        target_weights={"AAPL": 0.2}, exposure_target=1.0,
        equity=1_000, cash=1_000, current_positions={"LEGACY": 2.0},
        orders=[SimpleNamespace(symbol="AAPL", dollar_delta=200)],
    )

    checks = pre_trade_risk_checks(decision, {
        "risk": {"paper": {"max_position_weight": 0.30}},
        "portfolio": {"cash_buffer_percent": 0.02},
        "broker": {"min_order_notional": 1.0},
    })

    assert any(check.reason == "unpriced_current_holdings" for check in checks)
    assert risk_blocks_submission(checks)


def test_pre_trade_risk_checks_block_large_position():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        data_quality={},
        target_weights={"AAPL": 0.40},
        exposure_target=1.0,
        equity=1000,
        cash=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                dollar_delta=400,
            ),
        ],
    )

    checks = pre_trade_risk_checks(decision, {
        "risk": {"paper": {"max_position_weight": 0.30}},
        "portfolio": {"cash_buffer_percent": 0.02},
        "broker": {"min_order_notional": 1.0},
    })

    assert risk_status(checks) == "ERROR"
    assert risk_blocks_submission(checks)


def test_pre_trade_risk_checks_block_bad_data_for_selected_symbol():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        data_quality={
            "issues_by_symbol": {
                "AAPL": [
                    {
                        "reason": "zero_or_negative_prices",
                        "severity": "ERROR",
                        "details": {"count": 1},
                    },
                ],
            },
        },
        selected_symbols=["AAPL"],
        target_weights={"AAPL": 0.20},
        exposure_target=1.0,
        equity=1000,
        cash=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                dollar_delta=200,
            ),
        ],
    )

    checks = pre_trade_risk_checks(decision, {
        "risk": {"paper": {"max_position_weight": 0.30}},
        "portfolio": {"cash_buffer_percent": 0.02},
        "broker": {"min_order_notional": 1.0},
    })

    assert risk_status(checks) == "ERROR"
    assert risk_blocks_submission(checks)
    assert any(
        check.reason == "data_quality_zero_or_negative_prices"
        for check in checks
    )


def test_pre_trade_risk_blocks_broker_unsupported_limit_order():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        data_quality={},
        target_weights={"AAPL": 0.20},
        exposure_target=1.0,
        equity=1000,
        cash=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                dollar_delta=200,
                quantity_delta=2,
                order_type="LIMIT",
            ),
        ],
    )

    checks = pre_trade_risk_checks(decision, {
        "paper_trading": {"execution_adapter": "broker"},
        "risk": {"paper": {"max_position_weight": 0.30}},
        "portfolio": {"cash_buffer_percent": 0.02},
        "broker": {
            "min_order_notional": 1.0,
            "supports_limit_orders": False,
        },
    })

    assert risk_status(checks) == "ERROR"
    assert any(
        check.reason == "broker_limit_orders_unsupported"
        for check in checks
    )


def test_pre_trade_risk_blocks_broker_fractional_quantity():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        data_quality={},
        target_weights={"AAPL": 0.20},
        exposure_target=1.0,
        equity=1000,
        cash=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                dollar_delta=150,
                quantity_delta=1.5,
                order_type="MARKET",
            ),
        ],
    )

    checks = pre_trade_risk_checks(decision, {
        "paper_trading": {"execution_adapter": "broker"},
        "risk": {"paper": {"max_position_weight": 0.30}},
        "portfolio": {"cash_buffer_percent": 0.02},
        "broker": {
            "min_order_notional": 1.0,
            "supports_fractional": False,
        },
    })

    assert risk_status(checks) == "ERROR"
    assert any(
        check.reason == "broker_fractional_quantity_unsupported"
        for check in checks
    )


def test_portfolio_kill_switch_blocks_daily_loss():
    checks = portfolio_kill_switch_checks(
        current_equity=950,
        equity_history=[{"equity": 1000}],
        config={
            "risk": {
                "kill_switch": {
                    "enabled": True,
                    "max_daily_loss": 0.03,
                },
            },
        },
    )

    assert risk_status(checks) == "CRITICAL"
    assert checks[0].reason == "portfolio_daily_loss_kill_switch"


def test_portfolio_kill_switch_blocks_drawdown_from_peak():
    checks = portfolio_kill_switch_checks(
        current_equity=890,
        equity_history=[{"equity": 1000}, {"equity": 1100}],
        config={
            "risk": {
                "kill_switch": {
                    "enabled": True,
                    "max_drawdown_from_paper_start": 0.10,
                },
            },
        },
    )

    assert risk_status(checks) == "CRITICAL"
    assert any(
        check.reason == "portfolio_drawdown_kill_switch"
        for check in checks
    )


def test_model_kill_switch_blocks_candidate_config_hash_drift():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        model_context={"strategy": "dual_momentum"},
    )

    checks = model_kill_switch_checks(
        decision=decision,
        config={
            "risk": {
                "model_kill_switch": {
                    "enabled": True,
                    "expected_candidate_config_hash": "expected",
                },
            },
        },
        reproducibility={
            "candidate_config_hash": "actual",
            "candidate_config_path": "configs/paper/example.yaml",
        },
    )

    assert risk_status(checks) == "CRITICAL"
    assert checks[0].reason == "candidate_config_hash_drift"


def test_model_kill_switch_blocks_missing_model_context():
    decision = SimpleNamespace(
        data_freshness={"is_stale": False},
        model_context={},
    )

    checks = model_kill_switch_checks(
        decision=decision,
        config={
            "risk": {
                "model_kill_switch": {
                    "enabled": True,
                    "require_model_context": True,
                },
            },
        },
    )

    assert risk_status(checks) == "CRITICAL"
    assert checks[0].reason == "model_signal_unavailable"


def test_post_trade_risk_checks_pass_when_fill_matches_target():
    decision = SimpleNamespace(
        target_weights={"AAPL": 0.20},
        exposure_target=1.0,
        equity=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                price=100,
            ),
        ],
    )
    fill_record = {
        "fills": [{"symbol": "AAPL"}],
        "cash_after": 800,
        "equity_after": 1000,
        "positions_after": {"AAPL": 2},
    }

    checks = post_trade_risk_checks(decision, fill_record, {
        "risk": {"paper": {"post_trade_drift_tolerance": 0.005}},
        "portfolio": {"cash_buffer_percent": 0.02},
    })

    assert risk_status(checks) == "PASS"
    assert not risk_blocks_submission(checks)


def test_post_trade_risk_checks_flag_unexpected_position():
    decision = SimpleNamespace(
        target_weights={"AAPL": 0.20},
        exposure_target=1.0,
        equity=1000,
        orders=[
            SimpleNamespace(
                symbol="AAPL",
                price=100,
            ),
        ],
    )
    fill_record = {
        "fills": [{"symbol": "AAPL"}],
        "cash_after": 700,
        "equity_after": 1000,
        "positions_after": {"AAPL": 2, "MSFT": 1},
    }

    checks = post_trade_risk_checks(decision, fill_record, {
        "risk": {"paper": {"post_trade_drift_tolerance": 0.005}},
        "portfolio": {"cash_buffer_percent": 0.02},
    })

    assert risk_status(checks) == "ERROR"
    assert risk_blocks_submission(checks)
