from types import SimpleNamespace

from main import (
    apply_runtime_overrides,
    dual_momentum_candidate_configs,
    limited_grid,
)


def test_limited_grid_keeps_first_n_values():
    grid = {
        "ema_fast_period": [20, 50, 100],
        "target_exposure": [0.20, 0.40],
        "constant": "value",
    }

    assert limited_grid(grid, 1) == {
        "ema_fast_period": [20],
        "target_exposure": [0.20],
        "constant": "value",
    }


def test_fast_mode_limits_symbols_years_strategies_and_grid():
    config = {
        "backtest": {
            "symbols": ["AAPL", "MSFT", "SPY"],
            "years": 5,
            "warmup_bars": 200,
        },
        "research": {
            "fast_mode": {
                "symbols": ["AAPL"],
                "years": 2,
                "strategies": ["trend_pullback"],
                "max_grid_values_per_parameter": 1,
                "stage_one_max_combinations": 5,
                "walk_forward_folds": [{"train_start": "2024-01-01"}],
            },
            "stage_one_max_combinations": 80,
            "parameter_grid": {
                "ema_fast_period": [20, 50],
            },
            "strategy_comparison": [
                {
                    "name": "trend_pullback",
                    "parameter_grid": {
                        "pullback_tolerance": [0.01, 0.02],
                    },
                },
                {
                    "name": "buy_and_hold",
                    "parameter_grid": {
                        "target_exposure": [1.0],
                    },
                },
            ],
        },
    }
    args = SimpleNamespace(
        fast=True,
        symbols=None,
        years=None,
        strategies=None,
        grid_values=None,
        universe="default",
    )

    updated = apply_runtime_overrides(config, args)

    assert updated["backtest"]["symbols"] == ["AAPL"]
    assert updated["backtest"]["years"] == 2
    assert updated["research"]["stage_one_max_combinations"] == 5
    assert updated["research"]["parameter_grid"]["ema_fast_period"] == [20]
    assert updated["research"]["strategy_comparison"] == [
        {
            "name": "trend_pullback",
            "parameter_grid": {"pullback_tolerance": [0.01]},
        }
    ]


def test_runtime_overrides_can_select_etf_universe():
    config = {
        "backtest": {
            "symbols": ["AAPL", "SPY"],
            "years": 5,
        },
        "research": {
            "dual_momentum": {
                "symbols": ["AAPL", "SPY"],
                "etf_symbols": ["XLK", "XLF", "SPY"],
            },
            "strategy_comparison": [],
            "parameter_grid": {},
        },
    }
    args = SimpleNamespace(
        fast=False,
        symbols=None,
        years=None,
        strategies=None,
        grid_values=None,
        universe="etf",
    )

    updated = apply_runtime_overrides(config, args)

    assert updated["backtest"]["symbols"] == ["XLK", "XLF", "SPY"]
    assert updated["research"]["dual_momentum"]["symbols"] == [
        "XLK",
        "XLF",
        "SPY",
    ]


def test_dual_momentum_candidate_configs_expand_grid():
    dual_config = {
        "top_n": 3,
        "momentum_periods": [126, 252],
        "rebalance_frequency": "monthly",
        "use_asset_trend_filter": True,
        "target_volatility": None,
        "max_drawdown_guard": None,
        "min_breadth_percent": 0,
        "experiment_grid": {
            "top_n": [3, 5],
            "rebalance_frequency": ["monthly"],
            "momentum_periods": [[63, 126]],
            "use_asset_trend_filter": [True, False],
            "target_volatility": [None],
            "max_drawdown_guard": [0.2],
            "min_breadth_percent": [0.5],
        },
    }

    candidates = list(dual_momentum_candidate_configs(dual_config))

    assert len(candidates) == 4
    assert candidates[0]["momentum_periods"] == [63, 126]
    assert candidates[0]["min_breadth_percent"] == 0.5
    assert {candidate["top_n"] for candidate in candidates} == {3, 5}
    assert {
        candidate["use_asset_trend_filter"]
        for candidate in candidates
    } == {True, False}
