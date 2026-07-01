PAPER_CANDIDATE_ID_DEFAULT = ""

BACKTEST_DEFAULTS = {
    "provider": "alpaca",
    "data_feed": "iex",
    "data_adjustment": "all",
    "historical_bar_limit": 10_000,
}

STRATEGY_DEFAULTS = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
}

POSITION_SIZING_DEFAULTS = {
    "mode": "fixed_fractional",
    "target_exposure": 0.20,
    "max_exposure": 0.20,
}

PAPER_TRADING_DEFAULTS = {
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
}

TRADING_DEFAULTS = {
    "mode": "paper",
    "live_enabled": False,
}

BROKER_DEFAULTS = {
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
}

PORTFOLIO_DEFAULTS = {
    "cash_buffer_percent": 0.02,
}

EXECUTION_DEFAULTS = {
    "order_type": "market",
    "limit_offset_bps": 10.0,
    "assumed_slippage_bps": 5.0,
    "commission_bps": 1.0,
}

ALERTS_DEFAULTS = {
    "enabled": True,
    "service": "console",
}

RISK_DEFAULTS = {
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
}

REPORTS_DEFAULTS = {
    "backtest_dir": "reports/backtests",
    "walk_forward_dir": "reports/walk_forward",
    "summary_dir": "reports/summary",
    "paper_dir": "reports/paper",
    "ml_dir": "reports/ml",
}

CACHE_DEFAULTS = {
    "enabled": True,
    "data_dir": "cache/data",
    "results_dir": "cache/results",
    "ml_dir": "cache/ml",
}
