from config.config_defaults_research_relative_strength import RELATIVE_STRENGTH_DEFAULTS
from config.config_defaults_research_dual_momentum import DUAL_MOMENTUM_DEFAULTS
from config.config_defaults_research_multi_strategy import MULTI_STRATEGY_DEFAULTS
from config.config_defaults_research_classic import (
    FAST_MODE_DEFAULTS,
    PARAMETER_GRID_DEFAULTS,
    STRATEGY_COMPARISON_DEFAULTS,
    WALK_FORWARD_FOLDS_DEFAULTS,
)


RESEARCH_DEFAULTS = {
    "stooq_test_symbols": ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"],
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
    "relative_strength": RELATIVE_STRENGTH_DEFAULTS,
    "dual_momentum": DUAL_MOMENTUM_DEFAULTS,
    "multi_strategy": MULTI_STRATEGY_DEFAULTS,
    "fast_mode": FAST_MODE_DEFAULTS,
    "parameter_grid": PARAMETER_GRID_DEFAULTS,
    "walk_forward_folds": WALK_FORWARD_FOLDS_DEFAULTS,
    "strategy_comparison": STRATEGY_COMPARISON_DEFAULTS,
}
