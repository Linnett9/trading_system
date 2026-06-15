from config.config_loader import load_config


def test_load_config_adds_missing_research_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join([
            "alpaca:",
            '  api_key: "key"',
            '  secret_key: "secret"',
            "backtest:",
            "  symbols:",
            "    - AAPL",
            '  timeframe: "1Day"',
            "  years: 5",
            "  starting_equity: 10000",
            "  warmup_bars: 200",
            "strategy:",
            '  name: "ema_crossover"',
            "  ema_fast_period: 50",
            "  ema_slow_period: 200",
            "risk:",
            '  manager: "atr"',
            "  max_risk_per_trade: 0.0025",
            "  max_exposure: 0.02",
            "  atr_multiplier: 2.0",
            "  atr_stop_multiplier: 2.0",
            "  atr_take_profit_multiplier: 3.0",
            "execution:",
            "  spread_bps: 2.0",
            "  slippage_bps: 1.0",
            "reports:",
            '  backtest_dir: "reports/backtests"',
        ]),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config["research"]["optimization_metric"] == "composite"
    assert config["research"]["report_top_n"] == 10
    assert config["research"]["parallel_mode"] == "thread"
    assert config["research"]["two_stage_enabled"]
    assert config["research"]["optimizer_min_closed_trades"] == 1
    assert config["research"]["min_closed_trades"] == 20
    assert config["research"]["min_profit_factor"] == 1.1
    assert config["research"]["require_positive_excess"]
    assert "parameter_grid" in config["research"]
    assert "walk_forward_folds" in config["research"]
    assert config["strategy"]["rsi_period"] == 14
    assert config["position_sizing"]["mode"] == "fixed_fractional"
    assert config["position_sizing"]["target_exposure"] == 0.20
    dual_momentum = config["research"]["dual_momentum"]
    assert dual_momentum["risk_regime_mode"] == "scaled"
    assert dual_momentum["risk_off_risk_exposure"] == 0.25
    assert dual_momentum["fast_reentry_enabled"]
    assert dual_momentum["min_breadth_percent"] == 0.60
    assert dual_momentum["risk_off_symbols"] == []
    defensive = [
        experiment
        for experiment in dual_momentum["risk_regime_experiments"]
        if experiment["name"] == "defensive_assets"
    ][0]
    assert defensive["overrides"]["risk_off_symbols"] == [
        "BIL",
        "SHY",
        "IEF",
        "TLT",
        "GLD",
    ]
    assert config["reports"]["walk_forward_dir"] == "reports/walk_forward"
    assert config["reports"]["summary_dir"] == "reports/summary"
    assert config["cache"]["enabled"]
