from copy import deepcopy

def dual_momentum_risk_regime_configs(dual_config):
    grid = dual_config.get("risk_regime_experiments", [])

    if not grid:
        grid = [
            {
                "name": "baseline_inverse_vol",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "defensive_assets",
                "overrides": {
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "cash_risk_off",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
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
        ]

    neutral_optional_modules = {
        "fallback_allocation": 0.0,
        "decay_exit_enabled": False,
        "rank_drop_exit_top_n": None,
        "chop_filter_enabled": False,
        "quality_filter_enabled": False,
        "quality_require_momentum_improving": False,
        "cooldown_enabled": False,
        "leadership_filter_enabled": False,
        "benchmark_sleeve_allocation": 0.0,
        "replacement_score_gap": 0.0,
    }

    for item in grid:
        candidate = deepcopy(dual_config)
        candidate.update(neutral_optional_modules)
        candidate.update(item.get("overrides", {}))
        candidate["experiment_name"] = item["name"]

        yield {
            "name": item["name"],
            "config": candidate,
        }
