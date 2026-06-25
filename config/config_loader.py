import os

import yaml

from core.entities.trading_mode import TradingMode


DEFAULT_CONFIG = {
    "paper_candidate_id": "",
    "backtest": {
        "provider": "alpaca",
        "data_feed": "iex",
        "data_adjustment": "all",
        "historical_bar_limit": 10_000,
    },
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
    "paper_trading": {
        "enabled": True,
        "report_dir": "reports/paper",
        "min_trade_value": 1.0,
        "auto_fill": False,
        "refuse_stale_data": True,
        "max_data_age_days": 3,
        "paper_candidate_id": "",
        "fill_log_path": "data/paper/fills.csv",
        "event_log_path": "reports/paper/events.jsonl",
        "dashboard_path": "reports/paper/dashboard.csv",
        "execution_adapter": "local_ledger",
        "inspect_broker_state": False,
        "broker_state_source": "local",
        "sync_broker_state_before_decision": False,
    },
    "trading": {
        "mode": "paper",
        "live_enabled": False,
    },
    "broker": {
        "adapter": "fake",
        "supports_fractional": True,
        "supports_market_orders": True,
        "supports_limit_orders": True,
        "quantity_precision": 6,
        "min_order_notional": 1.0,
        "asset_class": "equity",
        "trading_hours": "regular",
        "partial_fill_ratio": 1.0,
        "reject_symbols": [],
        "open_orders": [],
        "cash_tolerance": 1.0,
        "position_tolerance": 1e-6,
        "cash_reconciliation": "account",
        "sleeve_cash": None,
    },
    "portfolio": {
        "cash_buffer_percent": 0.02,
    },
    "execution": {
        "order_type": "market",
        "limit_offset_bps": 10.0,
        "assumed_slippage_bps": 5.0,
        "commission_bps": 1.0,
    },
    "alerts": {
        "enabled": True,
        "service": "console",
    },
    "ml": {
        "enabled": False,
        "mode": "research",
        "research_years": 10,
        "minimum_history_years": 9,
        "history_coverage_tolerance_days": 10,
        "smoke_test_minimum_history_years": 5,
        "allow_short_history_for_smoke_test": False,
        "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
        "smoke_test_output_dir": "reports/ml/smoke_test",
        "smoke_test_cache_dir": "cache/ml/smoke_test",
        "historical_data_provider": "stooq_parquet",
        "stooq_csv_dir": "data/raw/stooq",
        "stooq_bulk_extracted_dir": "data/raw/stooq_bulk/extracted",
        "stooq_bulk_zip_path": "data/raw/stooq_bulk/us_daily_ascii.zip",
        "stooq_parquet_dir": "data/processed/stooq_parquet",
        "sector_reference_path": "data/reference/sector_by_symbol.json",
        "sector_by_symbol": {},
        "calibration_bin_count": 10,
        "rolling_base_rate_lookback_samples": 252,
        "ranking_quantile_count": 5,
        "prediction_target": "champion_success",
        "model_type": "logistic_regression",
        "feature_set": "price_regime_v1",
        "label_type": "champion_success",
        "train_start": None,
        "train_end": None,
        "test_start": None,
        "test_end": None,
        "prediction_horizon": 42,
        "label_horizon_days": 42,
        "drawdown_risk_threshold": 0.08,
        "decision_threshold": 0.50,
        "class_weight_balanced": True,
        "comparison_models": ["logistic_regression", "random_forest", "gradient_boosting"],
        "shadow_model_type": "gradient_boosting",
        "shadow_thresholds": [0.10, 0.15, 0.20, 0.25],
        "shadow_reduced_exposures": [0.70, 0.80, 0.90],
        "shadow_transaction_cost_bps": 5.0,
        "shadow_holdout_threshold": 0.20,
        "shadow_holdout_reduced_exposure": 0.70,
        "include_champion_state_features": True,
        "test_fraction": 0.20,
        "walk_forward_folds": 3,
        "random_seed": 42,
        "output_dir": "reports/ml",
        "benchmark_symbols": ["SPY", "QQQ"],
        "feature_lookback_days": 252,
    },
    "risk": {
        "kill_switch": {
            "enabled": True,
            "max_daily_loss": 0.03,
            "max_weekly_loss": 0.07,
            "max_drawdown_from_paper_start": 0.10,
        },
        "model_kill_switch": {
            "enabled": True,
            "block_stale_data": True,
            "require_model_context": True,
            "expected_candidate_config_hash": "",
        },
        "paper": {
            "max_position_weight": 0.30,
            "max_gross_exposure": 1.0,
            "max_single_order_notional": 0.50,
            "max_turnover": 1.0,
            "max_orders": 10,
            "post_trade_drift_tolerance": 0.005,
            "min_lookback_bars": 252,
            "max_latest_gap_percent": 0.40,
        },
    },
    "research": {
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
        "relative_strength": {
            "symbols": [
                "AAPL",
                "MSFT",
                "NVDA",
                "META",
                "GOOGL",
                "AMZN",
                "TSLA",
                "SPY",
                "QQQ",
                "BIL",
                "SHY",
                "IEF",
                "TLT",
                "GLD",
            ],
            "top_n": 2,
            "momentum_periods": [63, 126],
            "sma_period": 200,
            "rebalance_frequency": "monthly",
            "target_exposure": 1.0,
            "benchmark_symbol": "SPY",
            "transaction_cost_bps": 2.0,
            "experiment_grid": {
                "top_n": [1, 2, 3],
                "rebalance_frequency": ["monthly", "weekly"],
                "momentum_periods": [[21, 63], [63, 126], [126, 252]],
            },
        },
        "dual_momentum": {
            "symbols": [
                "AAPL",
                "MSFT",
                "NVDA",
                "META",
                "GOOGL",
                "AMZN",
                "TSLA",
                "SPY",
                "QQQ",
            ],
            "stock_symbols": [
                "AAPL",
                "MSFT",
                "NVDA",
                "META",
                "GOOGL",
                "AMZN",
                "TSLA",
                "AVGO",
                "AMD",
                "CRM",
                "ORCL",
                "ADBE",
                "NFLX",
                "COST",
                "JPM",
                "V",
                "MA",
                "HD",
                "UNH",
                "LLY",
                "XOM",
                "CVX",
                "CAT",
                "DE",
                "GE",
                "NOW",
                "PANW",
                "WMT",
                "PG",
                "KO",
                "PEP",
                "MCD",
                "NKE",
                "DIS",
                "CMCSA",
                "TMO",
                "ABT",
                "MRK",
                "PFE",
                "ABBV",
                "JNJ",
                "BAC",
                "GS",
                "MS",
                "BLK",
                "SCHW",
                "TXN",
                "QCOM",
                "INTU",
                "IBM",
                "AMAT",
                "LRCX",
                "MU",
                "HON",
                "UPS",
                "LOW",
                "RTX",
                "LMT",
                "BA",
                "COP",
                "SLB",
                "EOG",
                "NEE",
                "DUK",
                "SO",
                "PLD",
                "AMT",
                "SPY",
                "QQQ",
            ],
            "etf_symbols": [
                "XLK",
                "XLY",
                "XLF",
                "XLI",
                "XLV",
                "XLE",
                "XLU",
                "XLP",
                "XLB",
                "VNQ",
                "TLT",
                "GLD",
                "SPY",
                "QQQ",
            ],
            "top_n": 5,
            "momentum_periods": [63, 126],
            "regime_symbol": "SPY",
            "regime_sma_period": 200,
            "use_asset_trend_filter": True,
            "asset_sma_period": 200,
            "selection_mode": "all_positive",
            "min_selection_score": 0.03,
            "max_selected_assets": 5,
            "weighting": "inverse_volatility",
            "max_position_weight": 0.35,
            "weight_volatility_lookback": 63,
            "rebalance_frequency": "monthly",
            "target_exposure": 1.0,
            "target_volatility": None,
            "volatility_lookback": 63,
            "max_drawdown_guard": 0.20,
            "drawdown_guard_cooldown": 1,
            "strict_drawdown_kill_switch": True,
            "risk_off_symbols": [],
            "risk_off_top_n": 1,
            "risk_off_momentum_periods": [63, 126],
            "risk_regime_mode": "scaled",
            "mixed_risk_exposure": 0.75,
            "risk_off_risk_exposure": 0.25,
            "fast_reentry_enabled": True,
            "fast_reentry_symbols": [],
            "fast_reentry_sma_period": 100,
            "fast_reentry_momentum_period": 63,
            "fast_reentry_breadth_percent": 0.60,
            "fallback_symbols": ["SPY", "QQQ"],
            "fallback_allocation": 0.0,
            "fallback_min_risk_assets": 3,
            "fallback_momentum_periods": [63, 126],
            "decay_exit_enabled": False,
            "decay_momentum_period": 63,
            "rank_drop_exit_top_n": None,
            "chop_filter_enabled": True,
            "chop_lookback": 63,
            "min_chop_momentum": 0.02,
            "chop_risk_exposure": 0.50,
            "quality_filter_enabled": False,
            "quality_momentum_period": 21,
            "quality_sma_period": 50,
            "quality_require_momentum_improving": False,
            "cooldown_enabled": False,
            "cooldown_periods": 2,
            "cooldown_loss_threshold": -0.03,
            "rank_hysteresis_enabled": False,
            "rank_hysteresis_margin": 2,
            "rank_hysteresis_max_rank": None,
            "max_rebalance_replacements": None,
            "replacement_score_gap": 0.0,
            "rebalance_min_trade_weight": 0.0,
            "leadership_filter_enabled": False,
            "leadership_symbol": "SPY",
            "leadership_momentum_periods": [21, 63],
            "benchmark_sleeve_symbols": ["SPY", "QQQ"],
            "benchmark_sleeve_allocation": 0.0,
            "benchmark_sleeve_momentum_periods": [63],
            "benchmark_sleeve_top_n": 1,
            "benchmark_participation_filter_enabled": False,
            "benchmark_participation_period": 63,
            "benchmark_participation_min_return": 0.03,
            "benchmark_participation_max_selected_excess": 0.0,
            "sector_map": {},
            "max_sector_weight": None,
            "ranking_score_mode": "average_momentum",
            "enhanced_momentum_periods": [21, 63, 126],
            "enhanced_momentum_weights": [0.20, 0.35, 0.45],
            "relative_strength_symbol": "SPY",
            "relative_strength_periods": [21, 63],
            "relative_strength_weight": 0.25,
            "volatility_penalty_weight": 0.05,
            "ranking_volatility_lookback": 63,
            "min_breadth_percent": 0.60,
            "benchmark_symbol": "SPY",
            "transaction_cost_bps": 2.0,
            "experiment_grid": {
                "top_n": [5],
                "min_selection_score": [0.03],
                "max_selected_assets": [5],
                "rebalance_frequency": ["monthly"],
                "momentum_periods": [[63, 126]],
                "use_asset_trend_filter": [True],
                "selection_mode": ["all_positive"],
                "weighting": ["equal", "inverse_volatility"],
                "max_position_weight": [0.25, 0.35],
                "min_breadth_percent": [0.60],
                "target_volatility": [None],
                "max_drawdown_guard": [0.20],
                "strict_drawdown_kill_switch": [True],
                "mixed_risk_exposure": [0.75],
                "risk_off_risk_exposure": [0.25],
                "fallback_allocation": [0.0],
                "decay_exit_enabled": [False],
                "rank_drop_exit_top_n": [None],
                "chop_filter_enabled": [True],
                "chop_risk_exposure": [0.50, 0.60],
                "quality_filter_enabled": [False],
                "quality_require_momentum_improving": [False],
                "cooldown_enabled": [False],
                "rank_hysteresis_enabled": [False],
                "rank_hysteresis_margin": [2],
                "rank_hysteresis_max_rank": [None],
                "max_rebalance_replacements": [None],
                "replacement_score_gap": [0.0],
                "rebalance_min_trade_weight": [0.0],
                "leadership_filter_enabled": [False, True],
                "benchmark_sleeve_allocation": [0.0, 0.10, 0.15, 0.20, 0.25],
                "benchmark_participation_filter_enabled": [False],
                "max_sector_weight": [None],
                "ranking_score_mode": ["average_momentum", "enhanced"],
                "transaction_cost_bps": [2.0],
                "commission_bps": [0.0],
                "slippage_bps": [0.0],
                "spread_cost_bps": [0.0],
            },
            "risk_regime_experiments": [
                {
                    "name": "baseline_inverse_vol",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "binary",
                        "risk_off_risk_exposure": 0.0,
                        "fast_reentry_enabled": False,
                    },
                },
                {
                    "name": "cash_risk_off",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "binary",
                        "risk_off_risk_exposure": 0.0,
                        "fast_reentry_enabled": False,
                    },
                },
                {
                    "name": "defensive_assets",
                    "overrides": {
                        "risk_off_symbols": [
                            "BIL",
                            "SHY",
                            "IEF",
                            "TLT",
                            "GLD",
                        ],
                        "risk_regime_mode": "binary",
                        "risk_off_risk_exposure": 0.0,
                        "fast_reentry_enabled": False,
                    },
                },
                {
                    "name": "scaled_exposure",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": False,
                    },
                },
                {
                    "name": "fast_reentry",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "binary",
                        "fast_reentry_enabled": True,
                    },
                },
                {
                    "name": "scaled_plus_fast_reentry",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.50,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                    },
                },
                {
                    "name": "scaled_fast_reentry_75",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                    },
                },
                {
                    "name": "scaled_fast_reentry_fallback",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "fallback_symbols": ["SPY", "QQQ"],
                        "fallback_allocation": 0.25,
                        "fallback_min_risk_assets": 3,
                    },
                },
                {
                    "name": "scaled_fast_reentry_decay",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "decay_exit_enabled": True,
                        "decay_momentum_period": 63,
                        "rank_drop_exit_top_n": 7,
                    },
                },
                {
                    "name": "scaled_fast_reentry_chop",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_lookback": 63,
                        "min_chop_momentum": 0.02,
                        "chop_risk_exposure": 0.50,
                    },
                },
                {
                    "name": "ranked_top3_enhanced_chop",
                    "overrides": {
                        "selection_mode": "ranked",
                        "top_n": 3,
                        "min_selection_score": 0.03,
                        "max_selected_assets": 3,
                        "ranking_score_mode": "enhanced",
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_risk_exposure": 0.50,
                    },
                },
                {
                    "name": "ranked_top5_enhanced_chop",
                    "overrides": {
                        "selection_mode": "ranked",
                        "top_n": 5,
                        "min_selection_score": 0.03,
                        "max_selected_assets": 5,
                        "ranking_score_mode": "enhanced",
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_risk_exposure": 0.50,
                    },
                },
                {
                    "name": "ranked_top7_enhanced_chop",
                    "overrides": {
                        "selection_mode": "ranked",
                        "top_n": 7,
                        "min_selection_score": 0.03,
                        "max_selected_assets": 7,
                        "ranking_score_mode": "enhanced",
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_risk_exposure": 0.50,
                    },
                },
                {
                    "name": "scaled_chop_60",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_risk_exposure": 0.60,
                    },
                },
                {
                    "name": "scaled_chop_70",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_risk_exposure": 0.70,
                    },
                },
                {
                    "name": "scaled_chop_75",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "chop_risk_exposure": 0.75,
                    },
                },
                {
                    "name": "fast_reentry_50",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "fast_reentry_sma_period": 50,
                        "chop_filter_enabled": True,
                    },
                },
                {
                    "name": "fast_reentry_spy_qqq_50",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "fast_reentry_symbols": ["SPY", "QQQ"],
                        "fast_reentry_sma_period": 50,
                        "chop_filter_enabled": True,
                    },
                },
                {
                    "name": "chop_benchmark_sleeve_20",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "benchmark_sleeve_symbols": ["SPY", "QQQ"],
                        "benchmark_sleeve_allocation": 0.20,
                    },
                },
                {
                    "name": "chop_leadership_filter",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "leadership_filter_enabled": True,
                        "leadership_symbol": "SPY",
                        "leadership_momentum_periods": [21, 63],
                    },
                },
                {
                    "name": "scaled_reentry_chop_quality",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "quality_filter_enabled": True,
                        "quality_require_momentum_improving": True,
                        "cooldown_enabled": False,
                    },
                },
                {
                    "name": "scaled_reentry_chop_quality_cooldown",
                    "overrides": {
                        "risk_off_symbols": [],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "chop_filter_enabled": True,
                        "quality_filter_enabled": True,
                        "quality_require_momentum_improving": True,
                        "cooldown_enabled": True,
                        "cooldown_periods": 2,
                        "cooldown_loss_threshold": -0.03,
                    },
                },
            ],
        },
        "multi_strategy": {
            "symbols": [
                "AAPL",
                "MSFT",
                "NVDA",
                "META",
                "GOOGL",
                "AMZN",
                "TSLA",
                "SPY",
                "QQQ",
                "BIL",
                "SHY",
                "IEF",
                "TLT",
                "GLD",
            ],
            "benchmark_symbol": "SPY",
            "warmup_days": 500,
            "sleeves": [
                {
                    "name": "dual_momentum",
                    "enabled": True,
                    "weight": 0.70,
                    "parameters": {
                        "top_n": 5,
                        "momentum_periods": [63, 126],
                        "regime_symbol": "SPY",
                        "regime_sma_period": 200,
                        "use_asset_trend_filter": True,
                        "asset_sma_period": 200,
                        "selection_mode": "all_positive",
                        "weighting": "equal",
                        "max_position_weight": 0.25,
                        "rebalance_frequency": "monthly",
                        "target_exposure": 1.0,
                        "target_volatility": 0.20,
                        "volatility_lookback": 63,
                        "max_drawdown_guard": 0.20,
                        "drawdown_guard_cooldown": 1,
                        "strict_drawdown_kill_switch": True,
                        "risk_off_symbols": [],
                        "risk_off_top_n": 1,
                        "risk_off_momentum_periods": [63, 126],
                        "risk_regime_mode": "scaled",
                        "mixed_risk_exposure": 0.75,
                        "risk_off_risk_exposure": 0.25,
                        "fast_reentry_enabled": True,
                        "fast_reentry_symbols": [],
                        "fast_reentry_sma_period": 100,
                        "fast_reentry_momentum_period": 63,
                        "fast_reentry_breadth_percent": 0.60,
                        "fallback_symbols": ["SPY", "QQQ"],
                        "fallback_allocation": 0.0,
                        "fallback_min_risk_assets": 3,
                        "fallback_momentum_periods": [63, 126],
                        "decay_exit_enabled": False,
                        "decay_momentum_period": 63,
                        "rank_drop_exit_top_n": None,
                        "chop_filter_enabled": True,
                        "chop_lookback": 63,
                        "min_chop_momentum": 0.02,
                        "chop_risk_exposure": 0.50,
                        "quality_filter_enabled": False,
                        "quality_momentum_period": 21,
                        "quality_sma_period": 50,
                        "quality_require_momentum_improving": False,
                        "cooldown_enabled": False,
                        "cooldown_periods": 2,
                        "cooldown_loss_threshold": -0.03,
                        "leadership_filter_enabled": False,
                        "leadership_symbol": "SPY",
                        "leadership_momentum_periods": [21, 63],
                        "benchmark_sleeve_symbols": ["SPY", "QQQ"],
                        "benchmark_sleeve_allocation": 0.0,
                        "benchmark_sleeve_momentum_periods": [63],
                        "benchmark_sleeve_top_n": 1,
                        "min_breadth_percent": 0.60,
                        "benchmark_symbol": "SPY",
                        "transaction_cost_bps": 2.0,
                    },
                },
                {
                    "name": "relative_strength",
                    "enabled": True,
                    "weight": 0.30,
                    "parameters": {
                        "symbols": [
                            "AAPL",
                            "MSFT",
                            "NVDA",
                            "META",
                            "GOOGL",
                            "AMZN",
                            "TSLA",
                            "SPY",
                            "QQQ",
                        ],
                        "top_n": 3,
                        "momentum_periods": [63, 126],
                        "sma_period": 200,
                        "rebalance_frequency": "monthly",
                        "target_exposure": 1.0,
                        "benchmark_symbol": "SPY",
                        "transaction_cost_bps": 2.0,
                    },
                },
            ],
            "experiment_grid": {
                "sleeve_weights": [
                    [1.0, 0.0],
                    [0.7, 0.3],
                    [0.5, 0.5],
                    [0.3, 0.7],
                    [0.0, 1.0],
                ],
            },
        },
        "fast_mode": {
            "symbols": ["AAPL", "SPY"],
            "relative_strength_symbols": ["AAPL", "SPY"],
            "dual_momentum_symbols": ["AAPL", "SPY"],
            "multi_strategy_symbols": ["AAPL", "SPY"],
            "years": 2,
            "strategies": [
                "trend_pullback",
                "ema_rsi_pullback",
                "ensemble_vote",
                "buy_and_hold",
            ],
            "max_grid_values_per_parameter": 1,
            "stage_one_max_combinations": 12,
            "two_stage_top_n": 2,
            "min_closed_trades": 5,
            "optimizer_min_closed_trades": 1,
            "walk_forward_folds": [
                {
                    "train_start": "2024-07-01",
                    "train_end": "2025-06-30",
                    "test_start": "2025-07-01",
                    "test_end": "2026-06-15",
                },
            ],
        },
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
                "name": "ensemble_vote",
                "parameter_grid": {
                    "ensemble_min_buy_votes": [2, 3],
                    "ensemble_min_sell_votes": [1, 2],
                    "rsi_entry": [50, 55],
                    "rsi_exit": [40, 45],
                    "rsi_pullback": [40, 45],
                    "pullback_tolerance": [0.02, 0.04],
                    "use_regime_filter": [True],
                    "use_breakout_vote": [True, False],
                    "atr_stop_multiplier": [1.5, 2.0],
                    "trailing_atr_multiplier": [2.0, 3.0],
                    "atr_take_profit_multiplier": [6.0, None],
                    "target_exposure": [0.40],
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
        "paper_dir": "reports/paper",
        "ml_dir": "reports/ml",
    },
    "cache": {
        "enabled": True,
        "data_dir": "cache/data",
        "results_dir": "cache/results",
        "ml_dir": "cache/ml",
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


def load_config(path="config/config.yaml", overlay_project_config=False):
    default_path = "config/config.yaml"

    if path == default_path or overlay_project_config:
        with open(default_path, "r") as f:
            base_loaded = yaml.safe_load(f)

        config = merge_defaults(DEFAULT_CONFIG, base_loaded)
    else:
        config = DEFAULT_CONFIG

    if path != default_path:
        with open(path, "r") as f:
            override_loaded = yaml.safe_load(f)

        config = merge_defaults(config, override_loaded)

    _apply_environment_credentials(config)
    validate_config(config)
    return config


def _apply_environment_credentials(config):
    """Keep provider credentials out of tracked configuration files."""
    alpaca_config = config.setdefault("alpaca", {})
    api_key = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
    )
    secret_key = (
        os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET")
        or os.environ.get("APCA_API_SECRET_KEY")
    )
    if api_key:
        alpaca_config["api_key"] = api_key
    if secret_key:
        alpaca_config["secret_key"] = secret_key


def validate_config(config):
    trading_config = config.get("trading", {})
    mode = trading_config.get("mode", "paper")
    broker_config = config.get("broker", {})
    paper_config = config.get("paper_trading", {})
    execution_config = config.get("execution", {})
    ml_config = config.get("ml", {})

    try:
        trading_mode = TradingMode(mode)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in TradingMode)
        raise RuntimeError(f"Unsupported trading mode '{mode}'. Use one of: {allowed}") from exc

    if (
        trading_mode == TradingMode.LIVE
        and not trading_config.get("live_enabled", False)
    ):
        raise RuntimeError(
            "Live trading is disabled. Set trading.live_enabled=true "
            "only after broker, risk, and monitoring checks are ready."
        )

    broker_adapter = broker_config.get("adapter", "fake")
    allowed_brokers = {"fake", "alpaca"}
    if broker_adapter not in allowed_brokers:
        allowed = ", ".join(sorted(allowed_brokers))
        raise RuntimeError(
            f"Unsupported broker.adapter '{broker_adapter}'. Use one of: {allowed}"
        )

    execution_adapter = paper_config.get("execution_adapter", "local_ledger")
    allowed_execution_adapters = {"local_ledger", "broker"}
    if execution_adapter not in allowed_execution_adapters:
        allowed = ", ".join(sorted(allowed_execution_adapters))
        raise RuntimeError(
            "Unsupported paper_trading.execution_adapter "
            f"'{execution_adapter}'. Use one of: {allowed}"
        )

    broker_state_source = str(
        paper_config.get("broker_state_source", "local")
    ).lower()
    allowed_broker_state_sources = {"local", "broker"}
    if broker_state_source not in allowed_broker_state_sources:
        allowed = ", ".join(sorted(allowed_broker_state_sources))
        raise RuntimeError(
            "Unsupported paper_trading.broker_state_source "
            f"'{broker_state_source}'. Use one of: {allowed}"
        )

    cash_reconciliation = str(
        broker_config.get("cash_reconciliation", "account")
    ).lower()
    allowed_cash_reconciliation = {"account", "sleeve", "off"}
    if cash_reconciliation not in allowed_cash_reconciliation:
        allowed = ", ".join(sorted(allowed_cash_reconciliation))
        raise RuntimeError(
            "Unsupported broker.cash_reconciliation "
            f"'{cash_reconciliation}'. Use one of: {allowed}"
        )

    if broker_adapter == "alpaca":
        missing = [
            name for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
            if not config.get("alpaca", {}).get(
                "api_key" if name == "ALPACA_API_KEY" else "secret_key"
            )
        ]
        if missing:
            raise RuntimeError(
                "Alpaca broker selected or Alpaca data provider selected; required environment variables "
                f"are missing: {', '.join(missing)}. "
                "Accepted aliases include APCA_API_KEY_ID, ALPACA_SECRET, and APCA_API_SECRET_KEY."
            )

    order_type = str(execution_config.get("order_type", "market")).lower()
    if order_type not in {"market", "limit"}:
        raise RuntimeError(
            "Unsupported execution.order_type "
            f"'{order_type}'. Use one of: limit, market"
        )

    quantity_precision = int(broker_config.get("quantity_precision", 6))
    if quantity_precision < 0:
        raise RuntimeError("broker.quantity_precision must be >= 0")

    ml_mode = str(ml_config.get("mode", "research"))
    if ml_mode not in {"research", "shadow"}:
        raise RuntimeError(
            f"Unsupported ml.mode '{ml_mode}'. Use one of: research, shadow"
        )

    ml_model_type = str(ml_config.get("model_type", "noop"))
    if ml_model_type not in {
        "noop", "logistic_regression", "random_forest", "gradient_boosting",
    }:
        raise RuntimeError(
            f"Unsupported ml.model_type '{ml_model_type}'. "
            "Available models: gradient_boosting, logistic_regression, noop, random_forest."
        )

    random_seed = int(ml_config.get("random_seed", 42))
    if random_seed < 0:
        raise RuntimeError("ml.random_seed must be >= 0")

    label_horizon_days = int(
        ml_config.get("label_horizon_days", ml_config.get("prediction_horizon", 42))
    )
    if label_horizon_days <= 0:
        raise RuntimeError("ml.label_horizon_days must be greater than zero")

    label_type = str(ml_config.get("label_type", "champion_success"))
    if label_type not in {"risk_regime", "drawdown_risk", "champion_success"}:
        raise RuntimeError(
            "Unsupported ml.label_type "
            f"'{label_type}'. Use one of: champion_success, drawdown_risk, risk_regime"
        )

    drawdown_risk_threshold = float(
        ml_config.get("drawdown_risk_threshold", 0.08)
    )
    if not 0 < drawdown_risk_threshold < 1:
        raise RuntimeError("ml.drawdown_risk_threshold must be between zero and one")

    decision_threshold = float(ml_config.get("decision_threshold", 0.50))
    if not 0 < decision_threshold < 1:
        raise RuntimeError("ml.decision_threshold must be between zero and one")

    test_fraction = float(ml_config.get("test_fraction", 0.20))
    if not 0 < test_fraction < 1:
        raise RuntimeError("ml.test_fraction must be between zero and one")

    walk_forward_folds = int(ml_config.get("walk_forward_folds", 3))
    if walk_forward_folds < 1:
        raise RuntimeError("ml.walk_forward_folds must be at least one")

    calibration_bin_count = int(ml_config.get("calibration_bin_count", 10))
    if calibration_bin_count < 2:
        raise RuntimeError("ml.calibration_bin_count must be at least two")

    rolling_base_rate_lookback_samples = int(
        ml_config.get("rolling_base_rate_lookback_samples", 252)
    )
    if rolling_base_rate_lookback_samples < 1:
        raise RuntimeError(
            "ml.rolling_base_rate_lookback_samples must be at least one"
        )

    ranking_quantile_count = int(ml_config.get("ranking_quantile_count", 5))
    if ranking_quantile_count < 2:
        raise RuntimeError("ml.ranking_quantile_count must be at least two")

    if execution_adapter == "broker":
        supports_market_orders = broker_config.get("supports_market_orders", True)
        supports_limit_orders = broker_config.get("supports_limit_orders", True)
        if order_type == "market" and not supports_market_orders:
            raise RuntimeError(
                "execution.order_type=market is incompatible with "
                "broker.supports_market_orders=false"
            )
        if order_type == "limit" and not supports_limit_orders:
            raise RuntimeError(
                "execution.order_type=limit is incompatible with "
                "broker.supports_limit_orders=false"
            )

    return config
