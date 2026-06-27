from __future__ import annotations

from pathlib import Path

import yaml

from application.services.research_profiles import (
    apply_research_profile,
    load_research_profile,
)


EXPECTED_META_SOURCE_DIRS = [
    "reports/ml/logistic_regression_should_reduce_exposure",
    "reports/ml/random_forest_should_reduce_exposure",
    "reports/ml/gradient_boosting_should_reduce_exposure",
    "reports/ml/dlinear_should_reduce_exposure",
    "reports/ml/patchtst_should_reduce_exposure",
    "reports/ml/transformer_should_reduce_exposure",
    "reports/ml/itransformer_should_reduce_exposure",
    "reports/ml/momentum_transformer_should_reduce_exposure",
    "reports/ml/multitask_transformer_should_reduce_exposure",
    "reports/ml/market_context_encoder_should_reduce_exposure",
    "reports/ml/news_analysis_transformer_should_reduce_exposure",
    "reports/ml/tft_should_reduce_exposure",
]


def test_research_profile_loading():
    profile = load_research_profile("development")

    assert profile["name"] == "development"
    assert profile["universe"] == "current_32"
    assert profile["cache_dir"] == "cache/ml/development"
    assert profile["report_dir"] == "reports/ml/development"


def test_research_profile_isolates_cache_and_reports():
    config = {
        "backtest": {"years": 10},
        "cache": {"ml_dir": "cache/ml"},
        "reports": {"ml_dir": "reports/ml"},
        "research": {"dual_momentum": {}},
        "ml": {
            "research_years": 10,
            "output_dir": "reports/ml/dlinear_should_reduce_exposure",
            "expanded_rebalance_dataset": {},
        },
    }

    development = apply_research_profile(config, "development")
    benchmark = apply_research_profile(config, "benchmark")

    assert development["cache"]["ml_dir"] == "cache/ml/development"
    assert benchmark["cache"]["ml_dir"] == "cache/ml/benchmark"
    assert development["ml"]["output_dir"] == (
        "reports/ml/development/dlinear_should_reduce_exposure"
    )
    assert benchmark["ml"]["output_dir"] == (
        "reports/ml/benchmark/dlinear_should_reduce_exposure"
    )
    assert development["ml"]["expanded_rebalance_dataset_path"] != (
        benchmark["ml"]["expanded_rebalance_dataset_path"]
    )


def test_research_profile_sets_universe_and_years():
    config = {
        "backtest": {"years": 3},
        "cache": {"ml_dir": "cache/ml"},
        "reports": {"ml_dir": "reports/ml"},
        "research": {"dual_momentum": {}},
        "ml": {"expanded_rebalance_dataset": {}},
    }

    profiled = apply_research_profile(config, "benchmark")

    assert profiled["backtest"]["years"] == 10
    assert profiled["ml"]["research_years"] == 10
    assert profiled["ml"]["expanded_rebalance_dataset"]["universe_paths"] == [
        "data/reference/universes/us_liquid_500.yaml"
    ]
    assert profiled["ml"]["expanded_rebalance_dataset"]["max_symbols"] == 500
    assert profiled["research"]["dual_momentum"]["universe_path"] == (
        "data/reference/universes/us_liquid_500.yaml"
    )


def test_research_profile_rewrites_meta_ensemble_paths():
    config = {
        "backtest": {"years": 10},
        "cache": {"ml_dir": "cache/ml"},
        "reports": {"ml_dir": "reports/ml"},
        "research": {"dual_momentum": {}},
        "ml": {
            "model_type": "meta_ensemble",
            "output_dir": "reports/ml/regime_transformer_meta_ensemble_v1",
            "meta_dataset_path": "cache/ml/meta_ensemble_dataset.csv",
            "expanded_rebalance_dataset_path": "cache/ml/expanded_rebalance_dataset.csv",
            "source_prediction_dirs": [
                "reports/ml/dlinear_should_reduce_exposure",
                "reports/ml/patchtst_should_reduce_exposure",
            ],
        },
    }

    profiled = apply_research_profile(config, "development")

    assert profiled["ml"]["output_dir"] == (
        "reports/ml/development/regime_transformer_meta_ensemble_v1"
    )
    assert profiled["ml"]["meta_dataset_path"] == (
        "cache/ml/development/meta_ensemble_dataset.csv"
    )
    assert profiled["ml"]["expanded_rebalance_dataset_path"] == (
        "cache/ml/development/expanded_rebalance_dataset.csv"
    )
    assert profiled["ml"]["source_prediction_dirs"] == [
        "reports/ml/development/dlinear_should_reduce_exposure",
        "reports/ml/development/patchtst_should_reduce_exposure",
    ]


def test_development_profile_rewrites_all_meta_ensemble_source_dirs():
    config = yaml.safe_load(
        Path("configs/research/regime_transformer_meta_ensemble_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config.update({
        "backtest": {"years": 10},
        "cache": {"ml_dir": "cache/ml"},
        "reports": {"ml_dir": "reports/ml"},
        "research": {"dual_momentum": {}},
    })

    profiled = apply_research_profile(config, "development")

    assert config["ml"]["source_prediction_dirs"] == EXPECTED_META_SOURCE_DIRS
    assert profiled["ml"]["source_prediction_dirs"] == [
        f"reports/ml/development/{Path(source_dir).name}"
        for source_dir in EXPECTED_META_SOURCE_DIRS
    ]
    assert all(
        source_dir.startswith("reports/ml/development/")
        for source_dir in profiled["ml"]["source_prediction_dirs"]
    )
    assert all(
        "champion" not in source_dir
        for source_dir in profiled["ml"]["source_prediction_dirs"]
    )


def test_research_profile_sets_batch_runtime_options():
    config = {
        "backtest": {"years": 10},
        "cache": {"ml_dir": "cache/ml"},
        "reports": {"ml_dir": "reports/ml"},
        "research": {"dual_momentum": {}},
        "ml": {},
        "ml_research_batch": {
            "config_paths": [],
            "max_workers": 1,
            "model_threads": 1,
        },
    }

    profiled = apply_research_profile(config, "benchmark")

    assert profiled["ml_research_batch"]["max_workers"] == 4
    assert profiled["ml_research_batch"]["model_threads"] == 2
    assert profiled["ml_research_batch"]["profile"] == "benchmark"


def test_research_profiles_apply_ml_parallelism_values():
    config = {
        "backtest": {"years": 10},
        "cache": {"ml_dir": "cache/ml"},
        "reports": {"ml_dir": "reports/ml"},
        "research": {"dual_momentum": {}},
        "ml": {"expanded_rebalance_dataset": {}},
        "ml_research_batch": {"config_paths": [], "max_workers": 1, "model_threads": 1},
    }

    development = apply_research_profile(config, "development")
    benchmark = apply_research_profile(config, "benchmark")

    assert development["ml"]["num_workers"] == 2
    assert development["ml"]["model_threads"] == 4
    assert development["ml"]["torch_num_threads"] == 4
    assert development["ml"]["sklearn_n_jobs"] == 4
    assert development["ml"]["feature_workers"] == 1
    assert benchmark["ml"]["num_workers"] == 4
    assert benchmark["ml"]["model_threads"] == 2
    assert benchmark["ml"]["torch_num_threads"] == 2
    assert benchmark["ml"]["sklearn_n_jobs"] == 2
    assert benchmark["ml"]["feature_workers"] == 1
