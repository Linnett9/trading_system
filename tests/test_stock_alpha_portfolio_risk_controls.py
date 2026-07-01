from __future__ import annotations

from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import (
    _risk_control_metadata,
    _winners,
    build_policy_grid,
)


def test_risk_control_grid_produces_expected_policy_count():
    config = {
        "ml": {
            "stock_alpha_run_size": "dev",
            "stock_portfolio_policy_sweep_max_configs_dev": 1000,
            "stock_portfolio_policy_sweep_signals": ["ml_signal"],
            "stock_portfolio_policy_sweep_top_n_values": [10],
            "stock_portfolio_policy_sweep_max_position_weights": [0.02, 0.03],
            "stock_portfolio_policy_sweep_cost_bps_values": [5],
            "stock_portfolio_policy_sweep_slippage_bps_values": [5],
            "stock_portfolio_policy_sweep_turnover_caps": [None, 0.20],
            "stock_portfolio_policy_sweep_volatility_targets": [None, 0.10],
            "stock_portfolio_policy_sweep_gross_exposure_values": [1.0, 0.5],
            "stock_portfolio_policy_sweep_signal_thresholds": ["none", "positive"],
        }
    }
    grid = build_policy_grid(config, ["ml_signal"])
    assert len(grid) == 5 * 6 * 1 * 2 * 1 * 1 * 2 * 2 * 2 * 2 * 1
    assert {row["gross_exposure"] for row in grid} == {1.0, 0.5}
    assert {row["signal_threshold"] for row in grid} == {"none", "positive"}


def test_risk_control_metadata_reports_grid_dimensions():
    config = {"ml": {"stock_portfolio_policy_sweep_turnover_caps": [None, 0.2]}}
    grid = [
        {"max_position_weight_limit": 0.02, "gross_exposure": 1.0, "signal_threshold": "none"},
        {"max_position_weight_limit": 0.03, "gross_exposure": 0.5, "signal_threshold": "positive"},
    ]
    metadata = _risk_control_metadata(config, grid)
    assert metadata["active"] is True
    assert metadata["gross_exposure_values"] == [0.5, 1.0]
    assert metadata["market_regime_filter"]["available"] is False


def test_best_under_drawdown_limit_selects_lower_drawdown_policy():
    high_return = {
        "status": "completed",
        "kind": "ml_model",
        "signal_column": "ml",
        "net_return": 1.0,
        "total_return": 1.0,
        "sharpe": 0.5,
        "calmar_ratio": 1.0,
        "max_drawdown": -0.6,
        "average_turnover": 0.5,
        "cost_drag": 0.2,
    }
    controlled = {
        **high_return,
        "net_return": 0.6,
        "total_return": 0.6,
        "sharpe": 1.2,
        "max_drawdown": -0.1,
        "average_turnover": 0.2,
        "cost_drag": 0.05,
    }
    momentum = {
        **controlled,
        "kind": "baseline",
        "signal_column": "predicted_momentum_120d",
        "net_return": 0.3,
        "max_drawdown": -0.2,
        "sharpe": 0.8,
    }
    winners = _winners([high_return, controlled, momentum])
    assert winners["best_by_net_return_after_costs"] is high_return
    assert winners["best_under_drawdown_limit"] is controlled
    assert winners["best_ml_policy_with_lower_drawdown_than_momentum"] is controlled
    assert winners["best_ml_policy_with_higher_sharpe_than_momentum"] is controlled
