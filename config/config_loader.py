import yaml


DEFAULT_CONFIG = {
    "strategy": {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    },
    "position_sizing": {
        "mode": "fixed_fractional",
        "target_exposure": 0.20,
        "max_exposure": 0.20,
    },
    "research": {
        "optimization_metric": "composite",
        "report_top_n": 10,
        "parallel_workers": 1,
        "parallel_mode": "thread",
        "two_stage_enabled": True,
        "two_stage_top_n": 5,
        "stage_one_max_combinations": 80,
        "early_stop_max_drawdown": 0.30,
        "early_stop_equity_floor_pct": 0.70,
        "optimizer_min_closed_trades": 1,
        "min_closed_trades": 20,
        "min_profit_factor": 1.1,
        "min_sharpe": 0,
        "max_drawdown": 0.20,
        "min_time_in_market": 0.02,
        "max_time_in_market": 0.95,
        "require_positive_excess": True,
        "require_sharpe_edge": True,
        "parameter_grid": {
            "ema_fast_period": [20, 40, 50, 60, 100],
            "ema_slow_period": [50, 180, 200, 220, 300],
            "atr_stop_multiplier": [1.0, 2.0, 3.0],
            "trailing_atr_multiplier": [2.0, 3.0],
            "atr_take_profit_multiplier": [2.0, 3.0, 4.0],
            "target_exposure": [0.10, 0.20, 0.30, 0.40, 0.60],
        },
        "walk_forward_folds": [
            {
                "train_start": "2021-01-01",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31",
            },
            {
                "train_start": "2022-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2025-12-31",
            },
            {
                "train_start": "2023-01-01",
                "train_end": "2025-12-31",
                "test_start": "2026-01-01",
                "test_end": "2026-12-31",
            },
        ],
        "strategy_comparison": [
            {
                "name": "ema_crossover",
                "parameter_grid": {
                    "ema_fast_period": [20, 40, 50, 60],
                    "ema_slow_period": [50, 180, 200, 220],
                    "atr_stop_multiplier": [1.0, 2.0],
                    "atr_take_profit_multiplier": [3.0, 4.0],
                    "target_exposure": [0.10, 0.20, 0.30, 0.40, 0.60],
                },
            },
            {
                "name": "ema_rsi_filter",
                "parameter_grid": {
                    "ema_fast_period": [20, 40],
                    "ema_slow_period": [50, 180],
                    "rsi_entry": [50, 55],
                    "rsi_exit": [40, 45],
                    "atr_stop_multiplier": [1.0, 2.0],
                    "atr_take_profit_multiplier": [3.0],
                    "target_exposure": [0.10, 0.20, 0.30, 0.40, 0.60],
                },
            },
            {
                "name": "rsi_mean_reversion",
                "parameter_grid": {
                    "rsi_oversold": [30, 35, 40, 45],
                    "rsi_exit_level": [50, 55, 60, 65],
                    "atr_stop_multiplier": [1.0, 2.0],
                    "atr_take_profit_multiplier": [2.0, 3.0],
                    "target_exposure": [0.10, 0.20, 0.30, 0.40, 0.60],
                },
            },
            {
                "name": "donchian_breakout",
                "parameter_grid": {
                    "donchian_lookback": [5, 10, 20, 55],
                    "atr_stop_multiplier": [2.0, 3.0],
                    "atr_take_profit_multiplier": [3.0, 4.0],
                    "target_exposure": [0.10, 0.20, 0.30, 0.40, 0.60],
                },
            },
            {
                "name": "trend_pullback",
                "parameter_grid": {
                    "pullback_fast_period": [20, 50],
                    "pullback_tolerance": [0.01, 0.02, 0.04],
                    "pullback_exit_extension": [0.04, 0.08],
                    "use_regime_filter": [True, False],
                    "atr_stop_multiplier": [1.0, 2.0],
                    "atr_take_profit_multiplier": [2.0, 3.0],
                    "target_exposure": [0.20, 0.40, 0.60],
                },
            },
            {
                "name": "buy_and_hold",
                "parameter_grid": {
                    "target_exposure": [1.0],
                    "position_max_exposure": [1.0],
                },
            },
        ],
    },
    "reports": {
        "backtest_dir": "reports/backtests",
        "walk_forward_dir": "reports/walk_forward",
        "summary_dir": "reports/summary",
    },
    "cache": {
        "enabled": True,
        "data_dir": "cache/data",
        "results_dir": "cache/results",
    },
}


def merge_defaults(defaults, values):
    merged = defaults.copy()

    for key, value in (values or {}).items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        loaded = yaml.safe_load(f)

    return merge_defaults(DEFAULT_CONFIG, loaded)
