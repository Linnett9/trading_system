import json
from datetime import datetime, timedelta

import pytest

from core.entities.candle import Candle
from core.research.dual_momentum.portfolio import DualMomentumPortfolioBacktester
from core.research.dual_momentum.score_providers import (
    HybridScoreProvider,
    OOSArtifactScoreProvider,
    ScoreCandidate,
    rank_normalize_scores,
)
from core.research.dual_momentum.stock_ml_comparison import (
    stock_ml_comparison_config,
    write_stock_ml_dual_momentum_comparison,
)


def candles(symbol, prices):
    start = datetime(2025, 1, 1)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000,
        )
        for index, price in enumerate(prices)
    ]


def write_artifact(path, rows):
    columns = [
        "rebalance_date",
        "symbol",
        "fold_id",
        "actual_forward_return_10d",
        "predicted_momentum_120d",
        "predicted_risk_adjusted_momentum",
        "stock_level_predicted_forward_return_10d_elastic_net",
        "stock_level_predicted_forward_return_10d_random_forest",
        "stock_level_predicted_forward_return_10d_gradient_boosting",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def tiny_artifact(tmp_path):
    return write_artifact(
        tmp_path / "oos.csv",
        [
            {
                "rebalance_date": "2025-01-03",
                "symbol": "AAA",
                "fold_id": "fold_1",
                "actual_forward_return_10d": "0.01",
                "predicted_momentum_120d": "0.1",
                "predicted_risk_adjusted_momentum": "0.2",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.2",
                "stock_level_predicted_forward_return_10d_random_forest": "0.1",
                "stock_level_predicted_forward_return_10d_gradient_boosting": "0.3",
            },
            {
                "rebalance_date": "2025-01-03",
                "symbol": "BBB",
                "fold_id": "fold_1",
                "actual_forward_return_10d": "0.02",
                "predicted_momentum_120d": "0.2",
                "predicted_risk_adjusted_momentum": "0.1",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.9",
                "stock_level_predicted_forward_return_10d_random_forest": "0.8",
                "stock_level_predicted_forward_return_10d_gradient_boosting": "0.7",
            },
            {
                "rebalance_date": "2025-01-04",
                "symbol": "AAA",
                "fold_id": "fold_2",
                "actual_forward_return_10d": "0.03",
                "predicted_momentum_120d": "0.3",
                "predicted_risk_adjusted_momentum": "0.2",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.7",
                "stock_level_predicted_forward_return_10d_random_forest": "0.9",
                "stock_level_predicted_forward_return_10d_gradient_boosting": "0.8",
            },
            {
                "rebalance_date": "2025-01-04",
                "symbol": "BBB",
                "fold_id": "fold_2",
                "actual_forward_return_10d": "0.01",
                "predicted_momentum_120d": "0.1",
                "predicted_risk_adjusted_momentum": "0.2",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.2",
                "stock_level_predicted_forward_return_10d_random_forest": "0.1",
                "stock_level_predicted_forward_return_10d_gradient_boosting": "0.3",
            },
        ],
    )


def wide_artifact(tmp_path, symbol_count=12):
    rows = []
    for index in range(symbol_count):
        symbol = f"S{index:02d}"
        score = index + 1
        rows.append(
            {
                "rebalance_date": "2025-01-03",
                "symbol": symbol,
                "fold_id": "fold_1",
                "actual_forward_return_10d": f"0.{index + 1:02d}",
                "stock_level_predicted_forward_return_10d_elastic_net": score,
                "stock_level_predicted_forward_return_10d_random_forest": score,
                "stock_level_predicted_forward_return_10d_gradient_boosting": score,
            }
        )
    return write_artifact(tmp_path / "wide_oos.csv", rows)


def wide_candles(symbol_count=12):
    return {
        f"S{index:02d}": candles(
            f"S{index:02d}",
            [10 + index, 11 + index, 12 + index, 13 + index],
        )
        for index in range(symbol_count)
    }


def test_default_dual_momentum_behavior_unchanged():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[1],
        regime_sma_period=2,
        use_asset_trend_filter=False,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAA": candles("AAA", [10, 11, 13, 15]),
        "BBB": candles("BBB", [10, 10.5, 11, 11.5]),
        "SPY": candles("SPY", [10, 11, 12, 13]),
    })

    assert result.selections[0].symbols == ["AAA"]
    assert result.config["score_provider"] == "dual_momentum"


def test_oos_artifact_score_lookup_by_date_and_symbol(tmp_path):
    provider = OOSArtifactScoreProvider(
        tiny_artifact(tmp_path),
        "stock_level_predicted_forward_return_10d_elastic_net",
        rank_normalize=False,
    )

    scores = provider.score_candidates(
        datetime(2025, 1, 3),
        [
            ScoreCandidate("AAA", datetime(2025, 1, 3), 0.1),
            ScoreCandidate("BBB", datetime(2025, 1, 3), 0.2),
        ],
    )

    assert scores == {"AAA": 0.2, "BBB": 0.9}
    assert provider.fold_id(datetime(2025, 1, 3), "AAA") == "fold_1"


def test_duplicate_key_rejected(tmp_path):
    path = write_artifact(
        tmp_path / "dupe.csv",
        [
            {
                "rebalance_date": "2025-01-03",
                "symbol": "AAA",
                "fold_id": "fold_1",
                "actual_forward_return_10d": "0.01",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.2",
            },
            {
                "rebalance_date": "2025-01-03",
                "symbol": "AAA",
                "fold_id": "fold_1",
                "actual_forward_return_10d": "0.01",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.3",
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate OOS score row"):
        OOSArtifactScoreProvider(
            path,
            "stock_level_predicted_forward_return_10d_elastic_net",
        )


def test_missing_score_does_not_forward_fill_or_future_lookup(tmp_path):
    provider = OOSArtifactScoreProvider(
        tiny_artifact(tmp_path),
        "stock_level_predicted_forward_return_10d_elastic_net",
    )

    assert provider.score_candidates(
        datetime(2025, 1, 5),
        [ScoreCandidate("AAA", datetime(2025, 1, 5), 0.1)],
    ) == {}
    assert provider.missing_symbols(datetime(2025, 1, 5), ["AAA"]) == ["AAA"]


def test_non_finite_score_rejected(tmp_path):
    path = write_artifact(
        tmp_path / "bad.csv",
        [
            {
                "rebalance_date": "2025-01-03",
                "symbol": "AAA",
                "fold_id": "fold_1",
                "actual_forward_return_10d": "0.01",
                "stock_level_predicted_forward_return_10d_elastic_net": "nan",
            }
        ],
    )

    with pytest.raises(ValueError, match="Non-finite score"):
        OOSArtifactScoreProvider(
            path,
            "stock_level_predicted_forward_return_10d_elastic_net",
        )


def test_wrong_or_missing_signal_column_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing required"):
        OOSArtifactScoreProvider(tiny_artifact(tmp_path), "missing_signal")


def test_target_column_cannot_be_score_source(tmp_path):
    with pytest.raises(ValueError, match="target-like"):
        OOSArtifactScoreProvider(
            tiny_artifact(tmp_path),
            "actual_forward_return_10d",
        )


def test_ml_only_ranking_is_deterministic(tmp_path):
    provider = OOSArtifactScoreProvider(
        tiny_artifact(tmp_path),
        "stock_level_predicted_forward_return_10d_elastic_net",
    )
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[1],
        regime_sma_period=2,
        use_asset_trend_filter=False,
        transaction_cost_bps=0,
        score_provider=provider,
        rebalance_dates={datetime(2025, 1, 3).date()},
    )

    result = tester.run({
        "AAA": candles("AAA", [10, 10.5, 11]),
        "BBB": candles("BBB", [10, 11, 12]),
        "SPY": candles("SPY", [10, 11, 12]),
    })

    assert result.selections[0].symbols == ["BBB"]
    assert result.selections[0].scores["BBB"] == 1.0


def test_hybrid_rank_normalization_and_weighting(tmp_path):
    provider = OOSArtifactScoreProvider(
        tiny_artifact(tmp_path),
        "stock_level_predicted_forward_return_10d_elastic_net",
    )
    hybrid = HybridScoreProvider(
        provider,
        momentum_weight=0.5,
        artifact_weight=0.5,
    )

    scores = hybrid.score_candidates(
        datetime(2025, 1, 3),
        [
            ScoreCandidate("AAA", datetime(2025, 1, 3), 1.0),
            ScoreCandidate("BBB", datetime(2025, 1, 3), 0.0),
        ],
    )

    assert scores == {"AAA": 0.75, "BBB": 0.75}
    assert rank_normalize_scores({"AAA": 2.0, "BBB": 4.0}) == {
        "AAA": 0.5,
        "BBB": 1.0,
    }


def test_fair_shared_universe_restriction(tmp_path):
    artifact = tiny_artifact(tmp_path)
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(artifact),
                "output_dir": str(tmp_path / "out"),
            }
        },
    }
    dual_config = {
        "symbols": ["AAA", "BBB", "CCC", "SPY"],
        "top_n": 1,
        "momentum_periods": [1],
        "regime_sma_period": 2,
        "use_asset_trend_filter": False,
        "transaction_cost_bps": 0,
    }

    paths = write_stock_ml_dual_momentum_comparison(
        config,
        dual_config,
        {
            "AAA": candles("AAA", [10, 10.5, 11, 12]),
            "BBB": candles("BBB", [10, 11, 12, 13]),
            "CCC": candles("CCC", [10, 20, 30, 40]),
            "SPY": candles("SPY", [10, 11, 12, 13]),
        },
        strategy_names=["dual_momentum", "elastic_net_oos"],
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["fairness_controls"]["shared_symbol_count"] == 2
    assert "CCC" not in payload["fairness_controls"]["shared_symbols"]
    assert "CCC" in payload["fairness_controls"]["universe_diagnostics"]["missing_score_symbols"]


def test_broad_universe_diagnostics_and_top_n_sweep(tmp_path):
    artifact = tiny_artifact(tmp_path)
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(artifact),
                "output_dir": str(tmp_path / "out"),
                "mode": "broad_research_universe",
                "top_n_values": [1, 2],
            }
        },
    }
    paths = write_stock_ml_dual_momentum_comparison(
        config,
        {
            "symbols": ["AAA"],
            "top_n": 1,
            "momentum_periods": [1],
            "regime_sma_period": 2,
            "use_asset_trend_filter": False,
            "transaction_cost_bps": 0,
            "regime_symbol": "S00",
            "benchmark_symbol": "S00",
        },
        {
            "AAA": candles("AAA", [10, 10.5, 11, 12]),
            "BBB": candles("BBB", [10, 11, 12, 13]),
            "SPY": candles("SPY", [10, 11, 12, 13]),
        },
        strategy_names=["elastic_net_oos"],
    )

    payload = paths.json_path.read_text(encoding="utf-8")
    assert '"mode": "broad_research_universe"' in payload
    assert '"requested_symbol_count": 2' in payload
    assert '"market_data_symbol_count": 3' in payload
    assert '"top_n": 1' in payload
    assert '"top_n": 2' in payload


def test_top_n_sweep_changes_effective_selection_limit_and_holdings(tmp_path):
    artifact = wide_artifact(tmp_path, symbol_count=12)
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(artifact),
                "output_dir": str(tmp_path / "out"),
                "mode": "broad_research_universe",
                "top_n_values": [5, 10],
            }
        },
    }

    paths = write_stock_ml_dual_momentum_comparison(
        config,
        {
            "symbols": [],
            "top_n": 5,
            "max_selected_assets": 5,
            "momentum_periods": [1],
            "regime_sma_period": 2,
            "use_asset_trend_filter": False,
            "transaction_cost_bps": 0,
            "regime_symbol": "S00",
            "benchmark_symbol": "S00",
        },
        wide_candles(symbol_count=12),
        strategy_names=["elastic_net_oos"],
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    top_5 = payload["results"]["elastic_net_oos|top_n_5"]
    top_10 = payload["results"]["elastic_net_oos|top_n_10"]

    assert top_5["requested_top_n"] == 5
    assert top_5["effective_top_n"] == 5
    assert top_5["average_holding_count"] == 5
    assert top_5["config"]["max_selected_assets"] == 5
    assert top_10["requested_top_n"] == 10
    assert top_10["effective_top_n"] == 10
    assert top_10["average_holding_count"] == 10
    assert top_10["config"]["max_selected_assets"] == 10
    assert set(top_5["selections"][0]["symbols"]) != set(
        top_10["selections"][0]["symbols"]
    )


def test_top_n_sweep_can_preserve_explicit_max_selected_assets_cap(tmp_path):
    artifact = wide_artifact(tmp_path, symbol_count=12)
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(artifact),
                "output_dir": str(tmp_path / "out"),
                "mode": "broad_research_universe",
                "top_n_values": [10],
                "top_n_sweep_max_selected_assets_policy": (
                    "preserve_dual_config"
                ),
            }
        },
    }

    paths = write_stock_ml_dual_momentum_comparison(
        config,
        {
            "symbols": [],
            "top_n": 10,
            "max_selected_assets": 5,
            "momentum_periods": [1],
            "regime_sma_period": 2,
            "use_asset_trend_filter": False,
            "transaction_cost_bps": 0,
            "regime_symbol": "S00",
            "benchmark_symbol": "S00",
        },
        wide_candles(symbol_count=12),
        strategy_names=["elastic_net_oos"],
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    row = payload["results"]["elastic_net_oos|top_n_10"]

    assert row["requested_top_n"] == 10
    assert row["effective_top_n"] == 5
    assert row["average_holding_count"] == 5
    assert row["max_holding_count"] == 5
    assert row["config"]["max_selected_assets"] == 5


def test_arbitrary_model_score_column_selection_and_ensemble(tmp_path):
    artifact = tiny_artifact(tmp_path)
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(artifact),
                "output_dir": str(tmp_path / "out"),
                "models": ["elastic_net", "gradient_boosting"],
                "ensembles": {
                    "best_tabular_ensemble": {
                        "members": {
                            "elastic_net": 0.25,
                            "gradient_boosting": 0.75,
                        }
                    }
                },
            }
        },
    }

    paths = write_stock_ml_dual_momentum_comparison(
        config,
        {
            "symbols": ["AAA", "BBB", "SPY"],
            "top_n": 1,
            "momentum_periods": [1],
            "regime_sma_period": 2,
            "use_asset_trend_filter": False,
            "transaction_cost_bps": 0,
        },
        {
            "AAA": candles("AAA", [10, 10.5, 11, 12]),
            "BBB": candles("BBB", [10, 11, 12, 13]),
            "SPY": candles("SPY", [10, 11, 12, 13]),
        },
        strategy_names=["gradient_boosting_oos", "best_tabular_ensemble"],
    )

    payload = paths.json_path.read_text(encoding="utf-8")
    assert "stock_level_predicted_forward_return_10d_gradient_boosting" in payload
    assert "best_tabular_ensemble" in payload


def test_missing_configured_score_column_fails_clearly(tmp_path):
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(tiny_artifact(tmp_path)),
                "output_dir": str(tmp_path / "out"),
                "models": ["patchtst"],
            }
        }
    }

    with pytest.raises(ValueError, match="Missing required OOS score artifact columns"):
        write_stock_ml_dual_momentum_comparison(
            config,
            {
                "symbols": ["AAA", "BBB", "SPY"],
                "top_n": 1,
                "momentum_periods": [1],
                "regime_sma_period": 2,
                "use_asset_trend_filter": False,
            },
            {
                "AAA": candles("AAA", [10, 10.5, 11, 12]),
                "BBB": candles("BBB", [10, 11, 12, 13]),
                "SPY": candles("SPY", [10, 11, 12, 13]),
            },
        )


def test_comparison_config_preserves_default_operational_semantics(tmp_path):
    config = {
        "backtest": {"starting_equity": 500},
        "ml": {
            "stock_ml_dual_momentum_comparison": {
                "artifact_path": str(tiny_artifact(tmp_path)),
                "output_dir": str(tmp_path / "out"),
            }
        }
    }

    resolved = stock_ml_comparison_config(config)

    assert resolved["mode"] == "matched_operational_universe"
    assert resolved["models"] == ["elastic_net", "random_forest"]
    assert resolved["top_n_values"] is None

    paths = write_stock_ml_dual_momentum_comparison(
        config,
        {
            "symbols": ["AAA", "BBB"],
            "top_n": 2,
            "max_selected_assets": 1,
            "momentum_periods": [1],
            "regime_sma_period": 2,
            "use_asset_trend_filter": False,
            "transaction_cost_bps": 0,
            "regime_symbol": "AAA",
            "benchmark_symbol": "AAA",
        },
        {
            "AAA": candles("AAA", [10, 10.5, 11, 12]),
            "BBB": candles("BBB", [10, 11, 12, 13]),
        },
        strategy_names=["elastic_net_oos"],
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    row = payload["results"]["elastic_net_oos|top_n_2"]
    assert row["effective_top_n"] == 1
    assert row["average_holding_count"] == 1


def test_downstream_hysteresis_remains_shared_with_ml_scores(tmp_path):
    provider = OOSArtifactScoreProvider(
        tiny_artifact(tmp_path),
        "stock_level_predicted_forward_return_10d_elastic_net",
    )
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[1],
        regime_sma_period=2,
        use_asset_trend_filter=False,
        rank_hysteresis_enabled=True,
        rank_hysteresis_margin=1,
        transaction_cost_bps=0,
        score_provider=provider,
        rebalance_dates={
            datetime(2025, 1, 3).date(),
            datetime(2025, 1, 4).date(),
        },
    )

    result = tester.run({
        "AAA": candles("AAA", [10, 10.5, 11, 12]),
        "BBB": candles("BBB", [10, 11, 12, 13]),
        "SPY": candles("SPY", [10, 11, 12, 13]),
    })

    assert result.selections[0].symbols == ["BBB"]
    assert result.selections[1].symbols == ["BBB"]
