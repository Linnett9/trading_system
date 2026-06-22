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


def test_runtime_overrides_can_select_stooq_test_universe():
    config = {
        "backtest": {"symbols": ["AAPL"], "years": 5},
        "research": {
            "stooq_test_symbols": ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"],
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
        universe="stooq_test",
    )

    updated = apply_runtime_overrides(config, args)

    assert updated["backtest"]["symbols"] == [
        "AAPL", "MSFT", "NVDA", "SPY", "QQQ",
    ]


def test_ml_research_defaults_to_its_longer_history_window():
    config = {
        "backtest": {"symbols": ["AAPL", "SPY"], "years": 5},
        "ml": {
            "research_years": 10,
            "historical_data_provider": "stooq",
        },
        "research": {"strategy_comparison": [], "parameter_grid": {}},
    }
    args = SimpleNamespace(
        mode="ml-research",
        fast=False,
        symbols=None,
        years=None,
        strategies=None,
        grid_values=None,
        universe="default",
    )

    updated = apply_runtime_overrides(config, args)

    assert updated["backtest"]["years"] == 10
    assert updated["backtest"]["provider"] == "stooq"


def test_stooq_30_uses_only_stocks_for_the_champion_and_imports_benchmarks():
    config = {
        "backtest": {"symbols": ["AAPL"], "years": 5},
        "ml": {},
        "research": {
            "stooq_30_symbols": ["AAPL", "MSFT", "SPY", "QQQ"],
            "stooq_30_investable_symbols": ["AAPL", "MSFT"],
            "strategy_comparison": [],
            "parameter_grid": {},
        },
    }
    args = SimpleNamespace(
        mode="import-stooq-bulk",
        fast=False,
        symbols=None,
        years=None,
        strategies=None,
        grid_values=None,
        universe="stooq_30",
    )

    updated = apply_runtime_overrides(config, args)

    assert updated["backtest"]["symbols"] == ["AAPL", "MSFT"]
    assert updated["research"]["stooq_import_symbols"] == [
        "AAPL", "MSFT", "SPY", "QQQ",
    ]


def test_champion_robustness_uses_configured_local_ml_provider():
    config = {
        "backtest": {"symbols": ["AAPL"], "years": 5},
        "ml": {
            "research_years": 10,
            "historical_data_provider": "stooq_parquet",
            "stooq_parquet_dir": "data/processed/stooq_parquet",
        },
        "research": {"strategy_comparison": [], "parameter_grid": {}},
    }
    args = SimpleNamespace(mode="champion-robustness", fast=False, symbols=None, years=None, strategies=None, grid_values=None, universe="default")

    updated = apply_runtime_overrides(config, args)

    assert updated["backtest"]["provider"] == "stooq_parquet"
    assert updated["backtest"]["data_dir"] == "data/processed/stooq_parquet"
    assert updated["backtest"]["years"] == 10


def test_paper_dry_run_uses_configured_local_ml_provider():
    config = {"backtest": {"symbols": ["AAPL"], "years": 5}, "ml": {"historical_data_provider": "stooq_parquet", "stooq_parquet_dir": "data/processed/stooq_parquet"}, "research": {"strategy_comparison": [], "parameter_grid": {}}}
    args = SimpleNamespace(mode="paper-dry-run", fast=False, symbols=None, years=None, strategies=None, grid_values=None, universe="stooq_30")
    updated = apply_runtime_overrides(config, args)
    assert updated["backtest"]["provider"] == "stooq_parquet"


def test_ml_smoke_test_allows_short_history_only_in_separate_output_dir():
    config = {
        "backtest": {"symbols": ["AAPL", "SPY"], "years": 5},
        "ml": {
            "research_years": 10,
            "smoke_test_minimum_history_years": 5,
            "historical_data_provider": "alpaca",
            "smoke_test_output_dir": "reports/ml/smoke_test",
            "smoke_test_cache_dir": "cache/ml/smoke_test",
        },
        "research": {"strategy_comparison": [], "parameter_grid": {}},
    }
    args = SimpleNamespace(
        mode="ml-smoke-test",
        fast=False,
        symbols=None,
        years=None,
        strategies=None,
        grid_values=None,
        universe="default",
    )

    updated = apply_runtime_overrides(config, args)

    assert updated["ml"]["minimum_history_years"] == 5
    assert updated["ml"]["allow_short_history_for_smoke_test"] is True
    assert updated["ml"]["research_label"] == "SMOKE_TEST_NOT_PRODUCTION_VALIDATED"
    assert updated["ml"]["output_dir"] == "reports/ml/smoke_test"
    assert updated["cache"]["ml_dir"] == "cache/ml/smoke_test"


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
