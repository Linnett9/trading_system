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
    assert config["paper_trading"]["report_dir"] == "reports/paper"
    assert config["paper_trading"]["min_trade_value"] == 1.0
    dual_momentum = config["research"]["dual_momentum"]
    assert dual_momentum["risk_regime_mode"] == "scaled"
    assert dual_momentum["mixed_risk_exposure"] == 0.75
    assert dual_momentum["risk_off_risk_exposure"] == 0.25
    assert dual_momentum["fast_reentry_enabled"]
    assert dual_momentum["min_breadth_percent"] == 0.60
    assert dual_momentum["min_selection_score"] == 0.03
    assert dual_momentum["max_selected_assets"] == 5
    assert dual_momentum["risk_off_symbols"] == []
    assert dual_momentum["fallback_symbols"] == ["SPY", "QQQ"]
    assert dual_momentum["fallback_allocation"] == 0.0
    assert dual_momentum["fallback_min_risk_assets"] == 3
    assert not dual_momentum["decay_exit_enabled"]
    assert dual_momentum["chop_filter_enabled"]
    assert not dual_momentum["quality_filter_enabled"]
    assert not dual_momentum["quality_require_momentum_improving"]
    assert not dual_momentum["cooldown_enabled"]
    assert not dual_momentum["leadership_filter_enabled"]
    assert dual_momentum["benchmark_sleeve_allocation"] == 0.0
    assert dual_momentum["benchmark_sleeve_symbols"] == ["SPY", "QQQ"]
    assert dual_momentum["ranking_score_mode"] == "average_momentum"
    assert dual_momentum["enhanced_momentum_periods"] == [21, 63, 126]
    assert dual_momentum["enhanced_momentum_weights"] == [0.20, 0.35, 0.45]
    assert dual_momentum["relative_strength_symbol"] == "SPY"
    assert dual_momentum["relative_strength_periods"] == [21, 63]
    assert dual_momentum["relative_strength_weight"] == 0.25
    assert dual_momentum["volatility_penalty_weight"] == 0.05
    assert dual_momentum["ranking_volatility_lookback"] == 63
    assert "enhanced" in dual_momentum["experiment_grid"]["ranking_score_mode"]
    assert dual_momentum["experiment_grid"]["chop_risk_exposure"] == [
        0.50,
        0.60,
    ]
    assert dual_momentum["experiment_grid"]["benchmark_sleeve_allocation"] == [
        0.0,
        0.10,
        0.15,
        0.20,
        0.25,
    ]
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
    enhanced = [
        experiment
        for experiment in dual_momentum["risk_regime_experiments"]
        if experiment["name"] == "ranked_top5_enhanced_chop"
    ][0]
    assert enhanced["overrides"]["selection_mode"] == "ranked"
    assert enhanced["overrides"]["ranking_score_mode"] == "enhanced"
    assert config["reports"]["walk_forward_dir"] == "reports/walk_forward"
    assert config["reports"]["summary_dir"] == "reports/summary"
    assert config["reports"]["paper_dir"] == "reports/paper"
    assert config["cache"]["enabled"]
