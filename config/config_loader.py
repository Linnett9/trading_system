import yaml

from config.config_defaults import DEFAULT_CONFIG
from config.config_environment import _apply_environment_credentials
from core.entities.trading_mode import TradingMode


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
        "noop",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "transformer",
        "patchtst",
        "dlinear",
        "itransformer",
        "market_context_encoder",
        "momentum_transformer",
        "multitask_transformer",
        "news_analysis_transformer",
        "temporal_fusion_transformer",
        "meta_ensemble",
    }:
        raise RuntimeError(
            f"Unsupported ml.model_type '{ml_model_type}'. "
            "Available models: dlinear, gradient_boosting, logistic_regression, "
            "itransformer, market_context_encoder, meta_ensemble, momentum_transformer, "
            "multitask_transformer, news_analysis_transformer, noop, patchtst, "
            "random_forest, temporal_fusion_transformer, transformer."
        )

    allowed_meta_model_types = {
        "logistic",
        "logistic_regression",
        "ridge",
        "ridge_logistic",
        "random_forest",
        "gradient_boosting",
        "gbm",
        "lightgbm",
    }
    meta_model_type = str(ml_config.get("meta_model_type", "logistic_regression"))
    if meta_model_type not in allowed_meta_model_types:
        raise RuntimeError(
            f"Unsupported ml.meta_model_type '{meta_model_type}'. "
            "Available meta learners: logistic_regression, ridge_logistic, "
            "random_forest, gradient_boosting, lightgbm."
        )
    for value in ml_config.get("meta_model_types", []):
        if str(value) not in allowed_meta_model_types:
            raise RuntimeError(
                f"Unsupported ml.meta_model_types value '{value}'. "
                "Available meta learners: logistic_regression, ridge_logistic, "
                "random_forest, gradient_boosting, lightgbm."
            )

    sequence_length = int(ml_config.get("sequence_length", 63))
    if sequence_length < 2:
        raise RuntimeError("ml.sequence_length must be at least 2")

    for runtime_key in (
        "num_workers",
        "model_threads",
        "torch_num_threads",
        "sklearn_n_jobs",
        "feature_workers",
    ):
        if int(ml_config.get(runtime_key, 1)) < 1:
            raise RuntimeError(f"ml.{runtime_key} must be at least one")

    transformer_d_model = int(ml_config.get("transformer_d_model", 32))
    transformer_heads = int(ml_config.get("transformer_heads", 4))
    if transformer_heads < 1:
        raise RuntimeError("ml.transformer_heads must be at least one")
    if transformer_d_model % transformer_heads != 0:
        raise RuntimeError(
            "ml.transformer_d_model must be divisible by ml.transformer_heads"
        )

    dlinear_sequence_length = int(
        ml_config.get("dlinear_sequence_length", ml_config.get("sequence_length", 126))
    )
    if dlinear_sequence_length < 2:
        raise RuntimeError("ml.dlinear_sequence_length must be at least 2")

    patchtst_sequence_length = int(
        ml_config.get("patchtst_sequence_length", ml_config.get("sequence_length", 126))
    )
    if patchtst_sequence_length < 2:
        raise RuntimeError("ml.patchtst_sequence_length must be at least 2")

    patchtst_patch_length = int(ml_config.get("patchtst_patch_length", 16))
    patchtst_patch_stride = int(ml_config.get("patchtst_patch_stride", 8))
    patchtst_d_model = int(ml_config.get("patchtst_d_model", 64))
    patchtst_heads = int(ml_config.get("patchtst_heads", 4))

    if patchtst_patch_length < 1:
        raise RuntimeError("ml.patchtst_patch_length must be at least 1")

    if patchtst_patch_stride < 1:
        raise RuntimeError("ml.patchtst_patch_stride must be at least 1")

    if patchtst_patch_length > patchtst_sequence_length:
        raise RuntimeError(
            "ml.patchtst_patch_length must be <= ml.patchtst_sequence_length"
        )

    if patchtst_heads < 1:
        raise RuntimeError("ml.patchtst_heads must be at least one")

    if patchtst_d_model % patchtst_heads != 0:
        raise RuntimeError(
            "ml.patchtst_d_model must be divisible by ml.patchtst_heads"
        )

    itransformer_sequence_length = int(
        ml_config.get("itransformer_sequence_length", ml_config.get("sequence_length", 126))
    )
    if itransformer_sequence_length < 2:
        raise RuntimeError("ml.itransformer_sequence_length must be at least 2")

    itransformer_d_model = int(ml_config.get("itransformer_d_model", 64))
    itransformer_heads = int(ml_config.get("itransformer_heads", 4))
    if itransformer_heads < 1:
        raise RuntimeError("ml.itransformer_heads must be at least one")
    if itransformer_d_model % itransformer_heads != 0:
        raise RuntimeError(
            "ml.itransformer_d_model must be divisible by ml.itransformer_heads"
        )

    momentum_transformer_sequence_length = int(
        ml_config.get(
            "momentum_transformer_sequence_length",
            ml_config.get("sequence_length", 126),
        )
    )
    if momentum_transformer_sequence_length < 2:
        raise RuntimeError("ml.momentum_transformer_sequence_length must be at least 2")

    momentum_transformer_d_model = int(
        ml_config.get("momentum_transformer_d_model", 64)
    )
    momentum_transformer_heads = int(
        ml_config.get("momentum_transformer_heads", 4)
    )
    if momentum_transformer_heads < 1:
        raise RuntimeError("ml.momentum_transformer_heads must be at least one")
    if momentum_transformer_d_model % momentum_transformer_heads != 0:
        raise RuntimeError(
            "ml.momentum_transformer_d_model must be divisible by "
            "ml.momentum_transformer_heads"
        )
    momentum_size_floor = float(
        ml_config.get("momentum_transformer_size_multiplier_floor", 0.25)
    )
    momentum_size_ceiling = float(
        ml_config.get("momentum_transformer_size_multiplier_ceiling", 1.25)
    )
    if momentum_size_floor <= 0:
        raise RuntimeError("ml.momentum_transformer_size_multiplier_floor must be positive")
    if momentum_size_ceiling < momentum_size_floor:
        raise RuntimeError(
            "ml.momentum_transformer_size_multiplier_ceiling must be >= "
            "ml.momentum_transformer_size_multiplier_floor"
        )

    multitask_sequence_length = int(
        ml_config.get(
            "multitask_transformer_sequence_length",
            ml_config.get("sequence_length", 63),
        )
    )
    if multitask_sequence_length < 2:
        raise RuntimeError("ml.multitask_transformer_sequence_length must be at least 2")

    multitask_d_model = int(ml_config.get("multitask_transformer_d_model", 32))
    multitask_heads = int(ml_config.get("multitask_transformer_heads", 4))
    if multitask_heads < 1:
        raise RuntimeError("ml.multitask_transformer_heads must be at least one")
    if multitask_d_model % multitask_heads != 0:
        raise RuntimeError(
            "ml.multitask_transformer_d_model must be divisible by "
            "ml.multitask_transformer_heads"
        )

    allowed_multitask_targets = {
        "forward_return_5d",
        "forward_return_10d",
        "future_volatility",
        "future_drawdown",
        "max_adverse_excursion",
        "max_favourable_excursion",
    }
    multitask_primary_target = str(
        ml_config.get("multitask_primary_target", "should_reduce_exposure")
    )
    if multitask_primary_target != "should_reduce_exposure":
        raise RuntimeError(
            "ml.multitask_primary_target must be should_reduce_exposure"
        )
    multitask_targets = ml_config.get("multitask_regression_targets", [])
    if not isinstance(multitask_targets, list):
        raise RuntimeError("ml.multitask_regression_targets must be a list")
    for target in multitask_targets:
        if str(target) not in allowed_multitask_targets:
            raise RuntimeError(
                f"Unsupported ml.multitask_regression_targets value '{target}'. "
                "Allowed targets: forward_return_5d, forward_return_10d, "
                "future_volatility, future_drawdown, max_adverse_excursion, "
                "max_favourable_excursion."
            )
    multitask_classification_weight = float(
        ml_config.get("multitask_classification_weight", 1.0)
    )
    if multitask_classification_weight <= 0:
        raise RuntimeError("ml.multitask_classification_weight must be positive")
    multitask_regression_loss = str(
        ml_config.get("multitask_regression_loss", "huber")
    )
    if multitask_regression_loss not in {"huber", "mse"}:
        raise RuntimeError("ml.multitask_regression_loss must be one of: huber, mse")
    if float(ml_config.get("multitask_huber_delta", 1.0)) <= 0:
        raise RuntimeError("ml.multitask_huber_delta must be positive")
    for target in allowed_multitask_targets:
        weight_key = f"multitask_{target}_weight"
        if float(ml_config.get(weight_key, 0.0)) < 0:
            raise RuntimeError(f"ml.{weight_key} must be >= 0")

    market_context_sequence_length = int(
        ml_config.get("market_context_sequence_length", ml_config.get("sequence_length", 63))
    )
    if market_context_sequence_length < 2:
        raise RuntimeError("ml.market_context_sequence_length must be at least 2")
    if int(ml_config.get("market_context_hidden_size", 32)) < 4:
        raise RuntimeError("ml.market_context_hidden_size must be at least 4")
    market_context_floor = float(
        ml_config.get("market_context_risk_multiplier_floor", 0.25)
    )
    market_context_ceiling = float(
        ml_config.get("market_context_risk_multiplier_ceiling", 1.25)
    )
    if market_context_floor <= 0:
        raise RuntimeError("ml.market_context_risk_multiplier_floor must be positive")
    if market_context_ceiling < market_context_floor:
        raise RuntimeError(
            "ml.market_context_risk_multiplier_ceiling must be >= "
            "ml.market_context_risk_multiplier_floor"
        )

    news_sequence_length = int(
        ml_config.get("news_transformer_sequence_length", ml_config.get("sequence_length", 63))
    )
    if news_sequence_length < 2:
        raise RuntimeError("ml.news_transformer_sequence_length must be at least 2")
    news_d_model = int(ml_config.get("news_transformer_d_model", 32))
    news_heads = int(ml_config.get("news_transformer_heads", 4))
    if news_heads < 1:
        raise RuntimeError("ml.news_transformer_heads must be at least one")
    if news_d_model % news_heads != 0:
        raise RuntimeError(
            "ml.news_transformer_d_model must be divisible by ml.news_transformer_heads"
        )
    if int(ml_config.get("sentiment_min_reliability_tier", 2)) < 1:
        raise RuntimeError("ml.sentiment_min_reliability_tier must be at least 1")
    sentiment_windows = ml_config.get("sentiment_lookback_windows", [1, 5, 10, 21])
    if not isinstance(sentiment_windows, list) or not sentiment_windows:
        raise RuntimeError("ml.sentiment_lookback_windows must be a non-empty list")
    if any(int(window) <= 0 for window in sentiment_windows):
        raise RuntimeError("ml.sentiment_lookback_windows values must be positive")

    tft_encoder_length = int(ml_config.get("tft_encoder_length", ml_config.get("sequence_length", 64)))
    if tft_encoder_length < 2:
        raise RuntimeError("ml.tft_encoder_length must be at least 2")
    tft_hidden_size = int(ml_config.get("tft_hidden_size", 64))
    tft_attention_heads = int(ml_config.get("tft_attention_heads", 4))
    if tft_attention_heads < 1:
        raise RuntimeError("ml.tft_attention_heads must be at least one")
    if tft_hidden_size % tft_attention_heads != 0:
        raise RuntimeError(
            "ml.tft_hidden_size must be divisible by ml.tft_attention_heads"
        )
    tft_horizons = ml_config.get("tft_prediction_horizons", [5, 10])
    if not isinstance(tft_horizons, list) or not tft_horizons:
        raise RuntimeError("ml.tft_prediction_horizons must be a non-empty list")
    if any(int(horizon) <= 0 for horizon in tft_horizons):
        raise RuntimeError("ml.tft_prediction_horizons values must be positive")
    allowed_known_future_features = {
        "day_of_week",
        "month",
        "quarter",
        "is_month_end",
        "is_quarter_end",
        "rebalance_frequency",
        "days_until_next_rebalance",
        "days_since_last_rebalance",
    }
    known_future_features = ml_config.get("tft_known_future_features", [])
    if not isinstance(known_future_features, list):
        raise RuntimeError("ml.tft_known_future_features must be a list")
    for feature in known_future_features:
        if str(feature) not in allowed_known_future_features:
            raise RuntimeError(
                f"Unsupported ml.tft_known_future_features value '{feature}'."
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
    if label_type not in {
        "risk_regime",
        "drawdown_risk",
        "champion_success",
        "should_reduce_exposure",
    }:
        raise RuntimeError(
            "Unsupported ml.label_type "
            f"'{label_type}'. Use one of: champion_success, drawdown_risk, "
            "risk_regime, should_reduce_exposure"
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
