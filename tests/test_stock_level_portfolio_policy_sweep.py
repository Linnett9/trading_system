import inspect

from core.research.ml.stock_level import stock_level_portfolio_policy_sweep as sweep
from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import build_policy_grid, build_sizing_weights, _negative_return_warnings, _winners


def test_policy_grid_is_deterministic_and_dev_capped():
    config = {"ml": {"stock_alpha_run_size": "dev", "stock_portfolio_policy_sweep_max_configs_dev": 7, "stock_portfolio_policy_sweep_signals": ["signal"]}}
    first = build_policy_grid(config, ["signal"])
    second = build_policy_grid(config, ["signal"])
    assert first == second
    assert len(first) == 7
    assert [row["config_id"] for row in first] == sorted(row["config_id"] for row in first)


def test_profile_specific_grid_caps():
    for run_size, limit in (("benchmark", 3), ("full", 5)):
        config = {"ml": {"stock_alpha_run_size": run_size, f"stock_portfolio_policy_sweep_max_configs_{run_size}": limit, "stock_portfolio_policy_sweep_signals": ["signal"]}}
        assert len(build_policy_grid(config, ["signal"])) == limit


def test_sizing_methods_are_valid_and_respect_cap():
    rows = [{"symbol": symbol, "signal": score, "actual_future_volatility": vol} for symbol, score, vol in (("A", 4, .1), ("B", 3, .2), ("C", 2, .3), ("D", 1, .4))]
    for method in ("equal_weight", "score_weighted", "rank_weighted", "softmax_score_weighted", "inverse_volatility_weighted", "score_times_inverse_volatility"):
        weights = build_sizing_weights(rows, "signal", method, 1.0, 0.4)
        assert weights
        assert max(abs(value) for value in weights.values()) <= 0.4
        assert all(value >= 0 for value in weights.values())


def test_sweep_has_no_operational_imports_and_keeps_gates():
    source = inspect.getsource(sweep)
    assert all(token not in source for token in ("core.interfaces.broker", "core.paper", "core.entities.order", "paper_trading"))
    assert sweep.GUARDRAILS["promotion_thresholds_changed"] is False


def test_dev_grid_includes_momentum_baseline_when_available():
    config = {"ml": {"stock_alpha_run_size": "dev", "stock_portfolio_policy_sweep_max_configs_dev": 8, "stock_portfolio_policy_sweep_signals": ["ml_signal"]}}
    grid = build_policy_grid(config, ["ml_signal", "predicted_momentum_120d"])
    assert "predicted_momentum_120d" in {row["signal_column"] for row in grid}


def test_winners_populate_baseline_and_numeric_ml_delta():
    rows = [{"status": "completed", "kind": "ml_model", "signal_column": "ml", "net_return": -0.1, "total_return": -0.1, "sharpe": -1, "calmar_ratio": -1, "max_drawdown": -0.2, "average_turnover": .2}, {"status": "completed", "kind": "baseline", "signal_column": "predicted_momentum_120d", "net_return": -0.2, "total_return": -0.2, "sharpe": -2, "calmar_ratio": -2, "max_drawdown": -0.3, "average_turnover": .1}]
    winners = _winners(rows)
    assert winners["best_baseline_policy"] is rows[1]
    assert winners["best_ml_vs_momentum_120d"]["net_return_delta"] == 0.1
    assert _negative_return_warnings(rows) == {"all_candidate_net_returns_negative": True, "best_return_is_negative": True}
