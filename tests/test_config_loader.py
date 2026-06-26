import pytest

from config.config_loader import load_config, validate_config


def test_environment_alpaca_credentials_override_file_values(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "environment-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "environment-secret")

    config = load_config()

    assert config["alpaca"] == {
        "api_key": "environment-key",
        "secret_key": "environment-secret",
    }


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
    assert config["ml"]["enabled"] is False
    assert config["ml"]["mode"] == "research"
    assert config["ml"]["model_type"] == "logistic_regression"
    assert config["ml"]["random_seed"] == 42
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


def test_validate_config_blocks_live_mode_without_enable_flag():
    with pytest.raises(RuntimeError):
        validate_config({
            "trading": {
                "mode": "live",
                "live_enabled": False,
            },
        })


def test_validate_config_rejects_unknown_broker_adapter():
    with pytest.raises(RuntimeError, match="Unsupported broker.adapter"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "unknown"},
        })


def test_validate_config_rejects_unknown_paper_execution_adapter():
    with pytest.raises(RuntimeError, match="Unsupported paper_trading.execution_adapter"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "paper_trading": {"execution_adapter": "telepathy"},
            "broker": {"adapter": "fake"},
        })


def test_validate_config_requires_alpaca_environment(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Alpaca broker selected"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "alpaca"},
        })


def test_validate_config_rejects_unknown_order_type():
    with pytest.raises(RuntimeError, match="Unsupported execution.order_type"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "execution": {"order_type": "iceberg"},
        })


def test_validate_config_rejects_negative_quantity_precision():
    with pytest.raises(RuntimeError, match="broker.quantity_precision"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake", "quantity_precision": -1},
        })


def test_validate_config_rejects_unsupported_broker_order_type():
    with pytest.raises(RuntimeError, match="broker.supports_limit_orders=false"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "paper_trading": {"execution_adapter": "broker"},
            "broker": {
                "adapter": "fake",
                "supports_limit_orders": False,
            },
            "execution": {"order_type": "limit"},
        })


def test_validate_config_rejects_unknown_ml_mode():
    with pytest.raises(RuntimeError, match="Unsupported ml.mode"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {"mode": "production"},
        })


def test_validate_config_rejects_unknown_ml_model_type():
    with pytest.raises(RuntimeError, match="Unsupported ml.model_type"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {"model_type": "crystal_ball"},
        })


def test_patchtst_champion_config_runs_only_patchtst_model():
    config = load_config(
        "configs/research/patchtst_champion_success.yaml",
        overlay_project_config=True,
    )

    assert config["ml"]["model_type"] == "patchtst"
    assert config["ml"]["shadow_model_type"] == "patchtst"
    assert config["ml"]["comparison_models"] == ["patchtst"]
    assert config["ml"]["overlay_comparison_models"] == ["patchtst"]


def test_itransformer_research_config_validates():
    config = load_config(
        "configs/research/itransformer_should_reduce_exposure.yaml",
        overlay_project_config=True,
    )

    assert config["ml"]["model_type"] == "itransformer"
    assert config["ml"]["shadow_model_type"] == "itransformer"
    assert config["ml"]["comparison_models"] == ["itransformer"]
    assert config["ml"]["overlay_comparison_models"] == ["itransformer"]


def test_validate_config_rejects_invalid_itransformer_head_dimensions():
    with pytest.raises(RuntimeError, match="itransformer_d_model"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "itransformer",
                "itransformer_d_model": 10,
                "itransformer_heads": 4,
            },
        })


def test_momentum_transformer_research_config_validates():
    config = load_config(
        "configs/research/momentum_transformer_should_reduce_exposure.yaml",
        overlay_project_config=True,
    )

    assert config["ml"]["model_type"] == "momentum_transformer"
    assert config["ml"]["shadow_model_type"] == "momentum_transformer"
    assert config["ml"]["comparison_models"] == ["momentum_transformer"]
    assert config["ml"]["overlay_comparison_models"] == ["momentum_transformer"]


def test_validate_config_rejects_invalid_momentum_transformer_head_dimensions():
    with pytest.raises(RuntimeError, match="momentum_transformer_d_model"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "momentum_transformer",
                "momentum_transformer_d_model": 10,
                "momentum_transformer_heads": 4,
            },
        })


def test_multitask_transformer_research_config_validates():
    config = load_config(
        "configs/research/multitask_transformer_should_reduce_exposure.yaml",
        overlay_project_config=True,
    )

    assert config["ml"]["model_type"] == "multitask_transformer"
    assert config["ml"]["shadow_model_type"] == "multitask_transformer"
    assert config["ml"]["comparison_models"] == ["multitask_transformer"]
    assert config["ml"]["overlay_comparison_models"] == ["multitask_transformer"]
    assert config["ml"]["multitask_enabled"]
    assert "forward_return_5d" in config["ml"]["multitask_regression_targets"]


def test_validate_config_rejects_invalid_multitask_transformer_head_dimensions():
    with pytest.raises(RuntimeError, match="multitask_transformer_d_model"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "multitask_transformer",
                "multitask_transformer_d_model": 10,
                "multitask_transformer_heads": 4,
            },
        })


def test_validate_config_rejects_unknown_multitask_target():
    with pytest.raises(RuntimeError, match="multitask_regression_targets"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "multitask_transformer",
                "multitask_regression_targets": ["tomorrow_close"],
            },
        })


def test_market_context_encoder_research_config_validates():
    config = load_config(
        "configs/research/market_context_encoder_should_reduce_exposure.yaml",
        overlay_project_config=True,
    )

    assert config["ml"]["model_type"] == "market_context_encoder"
    assert config["ml"]["shadow_model_type"] == "market_context_encoder"
    assert config["ml"]["comparison_models"] == ["market_context_encoder"]
    assert config["ml"]["overlay_comparison_models"] == ["market_context_encoder"]


def test_validate_config_rejects_invalid_market_context_multiplier_bounds():
    with pytest.raises(RuntimeError, match="market_context_risk_multiplier_ceiling"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "market_context_encoder",
                "market_context_risk_multiplier_floor": 1.25,
                "market_context_risk_multiplier_ceiling": 0.25,
            },
        })


def test_news_analysis_transformer_research_config_validates():
    config = load_config(
        "configs/research/news_analysis_transformer_should_reduce_exposure.yaml",
        overlay_project_config=True,
    )

    assert config["ml"]["model_type"] == "news_analysis_transformer"
    assert config["ml"]["shadow_model_type"] == "news_analysis_transformer"
    assert config["ml"]["comparison_models"] == ["news_analysis_transformer"]
    assert config["ml"]["overlay_comparison_models"] == ["news_analysis_transformer"]
    assert config["ml"]["sentiment_lookback_windows"] == [1, 5, 10, 21]


def test_validate_config_rejects_invalid_news_transformer_head_dimensions():
    with pytest.raises(RuntimeError, match="news_transformer_d_model"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "news_analysis_transformer",
                "news_transformer_d_model": 10,
                "news_transformer_heads": 4,
            },
        })


def test_validate_config_rejects_invalid_momentum_transformer_multiplier_bounds():
    with pytest.raises(RuntimeError, match="momentum_transformer_size_multiplier_ceiling"):
        validate_config({
            "trading": {"mode": "paper", "live_enabled": False},
            "broker": {"adapter": "fake"},
            "ml": {
                "model_type": "momentum_transformer",
                "momentum_transformer_size_multiplier_floor": 1.25,
                "momentum_transformer_size_multiplier_ceiling": 0.25,
            },
        })


def test_load_config_accepts_ml_inventory_and_universe_defaults():
    config = load_config()

    assert config["ml"]["parquet_dir"] == "data/processed/stooq_parquet"
    assert config["ml"]["inventory_output_dir"] == "reports/ml"
    assert config["ml"]["universe_output_dir"] == "data/reference/universes"
    assert config["ml"]["min_history_years"] == 9
    assert config["ml"]["max_latest_gap_days"] == 14
    assert config["ml"]["min_average_dollar_volume_252d"] == 50_000_000


def test_load_config_merges_override_file_with_project_config(tmp_path):
    override_path = tmp_path / "paper.yaml"
    override_path.write_text(
        "\n".join([
            'paper_candidate_id: "candidate_a"',
            "paper_trading:",
            '  paper_candidate_id: "candidate_a"',
            "trading:",
            '  mode: "paper"',
            "research:",
            "  dual_momentum:",
            '    champion_id: "candidate_a"',
        ]),
        encoding="utf-8",
    )

    config = load_config(str(override_path), overlay_project_config=True)

    assert config["paper_candidate_id"] == "candidate_a"
    assert config["paper_trading"]["paper_candidate_id"] == "candidate_a"
    assert config["research"]["dual_momentum"]["champion_id"] == "candidate_a"
    assert "alpaca" in config
    assert "backtest" in config
