import csv
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.entities.candle import Candle
from core.research.ml.drawdown_review import build_drawdown_event_review
from core.research.ml.config import MLExperimentConfig
from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.features import HistoricalFeatureBuilder, add_champion_state_features
from core.research.ml.labels import DrawdownRiskLabelBuilder, RiskRegimeLabelBuilder
from core.research.ml.models import LogisticRegressionMLModel, NoOpMLModel
from core.research.ml.rule_overlay import (
    run_drawdown_risk_diagnostics,
    run_rule_exposure_study,
)
from core.research.ml.validation import chronological_holdout, rolling_walk_forward
from core.research.ml.datasets import MLDataset
from core.research.ml.history_coverage import assess_history_coverage
from core.research.ml.rebalance_dataset import build_champion_rebalance_rows
from core.research.ml.calibration import build_probability_calibration
from core.research.ml.sector_reference import load_sector_by_symbol
from core.research.ml.diagnostics import (
    build_ranking_diagnostics,
    probability_summary,
    rolling_base_rate_probabilities,
)


def test_ml_experiment_config_uses_defaults(tmp_path):
    config = {
        "reports": {"ml_dir": str(tmp_path)},
        "ml": {},
    }

    experiment_config = MLExperimentConfig.from_config(config)

    assert experiment_config.model_type == "logistic_regression"
    assert experiment_config.label_type == "champion_success"
    assert experiment_config.prediction_horizon == 42
    assert experiment_config.label_horizon_days == 42
    assert experiment_config.random_seed == 42
    assert experiment_config.output_dir == str(tmp_path)


def test_noop_ml_model_returns_neutral_probabilities():
    model = NoOpMLModel()
    rows = [{"return_1m": 0.01}, {"return_1m": -0.02}]

    assert model.predict(rows) == [0, 0]
    assert model.predict_proba(rows) == [0.5, 0.5]


def test_ml_experiment_runner_writes_research_artifacts(tmp_path):
    runner = MLExperimentRunner({
        "reports": {"ml_dir": str(tmp_path)},
        "cache": {"ml_dir": str(tmp_path / "cache")},
        "ml": {
            "enabled": False,
            "mode": "research",
            "model_type": "noop",
            "random_seed": 42,
        },
    })

    result = runner.run()

    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert result.feature_importance_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.metadata_path.exists()
    assert result.model_path.exists()
    assert result.features_path.exists()
    assert result.feature_summary_path.exists()
    assert result.labels_path.exists()
    assert result.dataset_path.exists()
    assert result.dataset_audit_path.exists()
    assert result.walk_forward_metrics_path.exists()
    assert result.threshold_sweep_path.exists()
    assert result.model_comparison_path.exists()
    assert result.shadow_overlay_path.exists()
    assert result.holdout_shadow_overlay_path.exists()
    assert result.rebalance_dataset_path.exists()
    assert result.rebalance_dataset_audit_path.exists()
    assert result.drawdown_event_review_path.exists()
    assert result.rule_exposure_study_path.exists()
    assert result.probability_calibration_path.exists()
    assert result.walk_forward_probability_calibration_path.exists()
    assert result.baseline_model_comparison_path.exists()
    assert result.ranking_diagnostics_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    feature_summary = json.loads(
        result.feature_summary_path.read_text(encoding="utf-8")
    )

    assert metrics["test_sample_count"] == 0
    assert metrics["note"].startswith("Research-only")
    assert metadata["research_only"] is True
    assert metadata["random_seed"] == 42
    assert feature_summary["correlation_matrix"] == {}

    with result.predictions_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_probability_calibration_reports_reliability_and_brier_skill():
    calibration = build_probability_calibration(
        [0, 0, 1, 1],
        [0.10, 0.20, 0.80, 0.90],
        bin_count=2,
    )

    assert calibration["brier_score"] == pytest.approx(0.025)
    assert calibration["brier_skill_score"] == pytest.approx(0.9)
    assert calibration["expected_calibration_error"] == pytest.approx(0.15)
    assert calibration["bins"][0]["observed_positive_rate"] == 0.0
    assert calibration["bins"][1]["observed_positive_rate"] == 1.0


def test_rolling_base_rate_uses_only_matured_labels():
    probabilities = rolling_base_rate_probabilities(
        train_labels=[0, 1],
        train_label_end_dates=["2024-01-10", "2024-01-20"],
        test_feature_dates=["2024-02-01", "2024-02-15"],
        test_labels=[1, 0],
        test_label_end_dates=["2024-02-10", "2024-02-20"],
        lookback_samples=10,
    )

    assert probabilities == [0.5, pytest.approx(2 / 3)]


def test_probability_summary_and_ranking_diagnostics_measure_ordering():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    outcomes = [
        {"strategy_return": -0.02, "excess_spy_return": -0.03, "drawdown_event": 1.0},
        {"strategy_return": -0.01, "excess_spy_return": -0.01, "drawdown_event": 0.0},
        {"strategy_return": 0.02, "excess_spy_return": 0.01, "drawdown_event": 0.0},
        {"strategy_return": 0.04, "excess_spy_return": 0.03, "drawdown_event": 0.0},
    ]

    summary = probability_summary(labels, probabilities, decision_threshold=0.5)
    ranking = build_ranking_diagnostics(labels, probabilities, outcomes, quantile_count=2)

    assert summary["roc_auc"] == 1.0
    assert summary["positive_prediction_rate"] == 0.5
    assert ranking["top_minus_bottom"]["success_rate"] == 1.0
    assert ranking["top_minus_bottom"]["drawdown_frequency"] == -0.5


def test_sector_reference_merges_overrides_and_normalizes_symbols(tmp_path):
    reference_path = tmp_path / "sectors.json"
    reference_path.write_text(
        json.dumps({"aapl": "Technology", "SPY": "Broad Market ETF"}),
        encoding="utf-8",
    )

    sectors = load_sector_by_symbol(
        str(reference_path),
        inline_mapping={"MSFT": "Technology", "AAPL": "Legacy sector"},
    )

    assert sectors == {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "SPY": "Broad Market ETF",
    }


def test_historical_feature_builder_uses_only_prices_available_on_feature_date():
    candles_by_symbol = {
        symbol: _candles(symbol, 300, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    builder = HistoricalFeatureBuilder()

    initial = builder.build(candles_by_symbol)
    original_row = initial.rows[-1]

    candles_by_symbol["SPY"].append(
        Candle(
            symbol="SPY",
            timestamp=datetime(2025, 1, 1),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
        )
    )
    candles_by_symbol["QQQ"].append(
        Candle(
            symbol="QQQ",
            timestamp=datetime(2025, 1, 1),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
        )
    )
    candles_by_symbol["AAPL"].append(
        Candle(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
        )
    )

    extended = builder.build(candles_by_symbol)

    assert initial.dropped_rows == 252
    assert original_row == extended.rows[-2]
    assert original_row["feature_date"] == "2024-10-26"
    assert "spy_return_12m_ex_latest_month" in original_row
    assert "breadth_above_sma_200" in original_row


def test_ml_experiment_runner_caches_historical_features(tmp_path):
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(tmp_path / "cache")},
            "reports": {"ml_dir": str(tmp_path / "reports")},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "noop",
                "output_dir": str(tmp_path / "reports"),
                "comparison_models": ["noop"],
                "shadow_model_type": "noop",
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    with result.features_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(result.feature_summary_path.read_text(encoding="utf-8"))
    audit = json.loads(result.dataset_audit_path.read_text(encoding="utf-8"))

    assert len(rows) == 148
    assert summary["row_count"] == 148
    assert summary["dropped_rows_insufficient_lookback"] == 252
    assert rows[0]["feature_date"] == "2024-09-09"
    assert audit["sample_count"] == 106
    assert audit["feature_count"] == 33
    assert audit["leakage_check_passed"] is True


def test_strict_ml_research_rejects_short_history(tmp_path):
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {"years": 2, "timeframe": "1Day"},
            "cache": {"enabled": False, "ml_dir": str(tmp_path / "cache")},
            "reports": {"ml_dir": str(tmp_path / "reports")},
            "ml": {
                "model_type": "noop",
                "output_dir": str(tmp_path / "reports"),
                "minimum_history_years": 2,
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    with pytest.raises(RuntimeError, match="historical coverage"):
        runner.run()


def test_smoke_test_metadata_is_not_production_validated(tmp_path):
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    report_dir = tmp_path / "smoke_test"
    runner = MLExperimentRunner(
        {
            "backtest": {"years": 2, "timeframe": "1Day"},
            "cache": {"enabled": False, "ml_dir": str(tmp_path / "cache")},
            "reports": {"ml_dir": str(report_dir)},
            "ml": {
                "model_type": "noop",
                "output_dir": str(report_dir),
                "minimum_history_years": 2,
                "allow_short_history_for_smoke_test": True,
                "research_label": "SMOKE_TEST_NOT_PRODUCTION_VALIDATED",
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["research_label"] == "SMOKE_TEST_NOT_PRODUCTION_VALIDATED"
    assert metadata["production_validated"] is False
    assert "warning" in metadata


def test_risk_regime_labels_start_strictly_after_feature_dates():
    candles = _candles("SPY", 8, 100.0)
    feature_rows = [
        {"feature_date": candles[index].timestamp.date().isoformat()}
        for index in range(4)
    ]

    result = RiskRegimeLabelBuilder(horizon_days=3).build(feature_rows, candles)

    assert len(result.rows) == 4
    assert all(
        row["feature_date"] < row["label_start_date"] <= row["label_end_date"]
        for row in result.rows
    )
    assert all(row["risk_regime"] == 1 for row in result.rows)


def test_drawdown_risk_label_captures_future_peak_to_trough_loss():
    prices = [100.0, 102.0, 96.0, 90.0, 95.0, 97.0]
    candles = [
        Candle(
            symbol="SPY",
            timestamp=datetime(2024, 1, 1) + timedelta(days=index),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1_000.0,
        )
        for index, price in enumerate(prices)
    ]
    feature_rows = [{"feature_date": "2024-01-01"}]

    result = DrawdownRiskLabelBuilder(horizon_days=3, threshold=0.08).build(
        feature_rows,
        candles,
    )

    assert result.label_name == "drawdown_risk"
    assert result.rows[0]["drawdown_risk"] == 1
    assert result.rows[0]["future_max_drawdown"] == -0.11764705882352944


def test_champion_state_features_use_latest_available_selection_only():
    rows = [
        {"feature_date": "2024-01-01", "market_feature": 1.0},
        {"feature_date": "2024-01-02", "market_feature": 1.0},
        {"feature_date": "2024-01-03", "market_feature": 1.0},
    ]
    selections = [
        SimpleNamespace(
            timestamp=datetime(2024, 1, 2),
            symbols=["AAPL"],
            scores={"AAPL": 0.4},
            exposure_target=0.8,
            target_weights={"AAPL": 1.0},
            risk_on=True,
            breadth_passes=True,
            drawdown_guard_active=False,
            chop_filter_active=False,
        ),
    ]

    enriched = add_champion_state_features(rows, selections)

    assert enriched[0]["champion_exposure"] == 0.0
    assert enriched[1]["champion_exposure"] == 0.8
    assert enriched[2]["champion_holding_count"] == 1.0
    assert enriched[2]["champion_last_rebalance_turnover"] == 1.0


def test_logistic_regression_model_learns_and_persists(tmp_path):
    features = [
        {"momentum": -2.0},
        {"momentum": -1.0},
        {"momentum": 1.0},
        {"momentum": 2.0},
    ]
    model = LogisticRegressionMLModel(max_iterations=200)
    model.fit(features, [0, 0, 1, 1])

    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = LogisticRegressionMLModel.load(path)

    assert loaded.predict(features) == [0, 0, 1, 1]
    assert loaded.feature_importances()["momentum"] > 0


def test_chronological_holdout_purges_overlapping_label_horizons():
    dataset = MLDataset(
        features=[{"value": float(index)} for index in range(5)],
        labels=[0, 1, 0, 1, 1],
        feature_dates=[
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ],
        label_start_dates=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ],
        label_end_dates=[
            "2024-01-02",
            "2024-01-05",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
        ],
    )

    split = chronological_holdout(dataset, test_fraction=0.40)

    assert split.test_start_date == "2024-01-04"
    assert split.purged_train_samples == 2
    assert split.train.feature_dates == ["2024-01-01"]
    assert split.test.feature_dates == ["2024-01-04", "2024-01-05"]


def test_rolling_walk_forward_uses_disjoint_future_test_windows():
    dataset = MLDataset(
        features=[{"value": float(index)} for index in range(15)],
        labels=[index % 2 for index in range(15)],
        feature_dates=[f"2024-01-{index:02d}" for index in range(1, 16)],
        label_start_dates=[f"2024-01-{index:02d}" for index in range(2, 17)],
        label_end_dates=[f"2024-01-{index:02d}" for index in range(3, 18)],
    )

    folds = rolling_walk_forward(dataset, fold_count=3)

    assert len(folds) == 3
    assert folds[0].split.test.feature_dates == ["2024-01-07", "2024-01-08", "2024-01-09"]
    assert folds[-1].split.test.feature_dates == ["2024-01-13", "2024-01-14", "2024-01-15"]
    assert all(
        all(end < fold.split.test_start_date for end in fold.split.train.label_end_dates)
        for fold in folds
    )


def test_drawdown_risk_diagnostics_measure_rare_stress_conditions():
    rows = [_stress_row(index) for index in range(15)]
    rows[10].update({
        "spy_distance_sma_200": "-0.02",
        "breadth_change_since_last_rebalance": "-0.15",
        "spy_volatility_ratio_21d_63d": "1.20",
        "recent_champion_excess_return_2_rebalances": "-0.03",
        "spy_max_drawdown_63d": "-0.12",
        "drawdown_event": "1",
    })

    full_sample = run_rule_exposure_study(rows, transaction_cost_bps=5.0)
    diagnostics = run_drawdown_risk_diagnostics(rows)
    by_name = {
        result["condition"]: result
        for result in diagnostics["conditions"]
    }

    assert all(result["rule"] != "multi_signal_stress_ladder" for result in full_sample)
    assert by_name["market_stress_with_champion_weakness"]["matched_rebalances"] == 1
    assert by_name["market_stress_with_champion_weakness"]["drawdown_event_rate"] == 1.0
    assert diagnostics["baseline_drawdown_event_rate"] == 1 / 15


def test_drawdown_event_review_compares_event_and_normal_cohorts():
    event = _review_row("2024-01-01", drawdown_event=1)
    normal = _review_row("2024-02-01", drawdown_event=0)

    review = build_drawdown_event_review([event, normal])

    assert review["drawdown_event_count"] == 1
    assert review["event_cases"][0]["selected_symbols"] == "AAPL,MSFT"
    assert round(
        review["event_vs_non_event_means"]["largest_weight"]["difference"],
        4,
    ) == 0.1
    assert review["chronological_event_rate"][0]["drawdown_event_rate"] == 1.0


def test_history_coverage_rejects_a_short_common_range():
    candles = _candles("AAPL", 20, 100.0)

    coverage = assess_history_coverage(
        {"AAPL": candles, "SPY": candles},
        required_years=1,
        tolerance_days=10,
    )

    assert coverage["coverage_sufficient"] is False
    assert coverage["available_calendar_days"] == 19


def test_history_coverage_accepts_a_nine_year_range():
    candles = [
        Candle(
            symbol="AAPL",
            timestamp=datetime(2016, 1, 1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1_000.0,
        ),
        Candle(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1_000.0,
        ),
    ]

    coverage = assess_history_coverage({"AAPL": candles}, required_years=9)

    assert coverage["coverage_sufficient"] is True


def test_rebalance_dataset_adds_selection_risk_features():
    candles = _correlated_candles("AAPL", 70)
    candles_by_symbol = {
        "AAPL": candles,
        "MSFT": _correlated_candles("MSFT", 70),
        "SPY": _correlated_candles("SPY", 70),
    }
    dates = [candle.timestamp for candle in candles]
    selections = [
        SimpleNamespace(
            timestamp=dates[64],
            symbols=["AAPL", "MSFT"],
            scores={"AAPL": 0.4, "MSFT": 0.4},
            exposure_target=0.8,
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            risk_on=True,
            breadth_passes=True,
            drawdown_guard_active=False,
            chop_filter_active=False,
            regime_label="risk-on",
        ),
    ]
    feature_rows = [{
        "feature_date": dates[64].date().isoformat(),
        "spy_distance_sma_200": 0.01,
        "spy_realized_volatility_21d": 0.1,
        "spy_realized_volatility_63d": 0.1,
        "spy_max_drawdown_63d": -0.02,
        "spy_max_drawdown_126d": -0.03,
        "breadth_above_sma_200": 0.7,
    }]
    equity_curve = [
        SimpleNamespace(timestamp=timestamp, equity=100.0 + index)
        for index, timestamp in enumerate(dates)
    ]

    rows = build_champion_rebalance_rows(
        feature_rows,
        selections,
        equity_curve,
        candles_by_symbol["SPY"],
        horizon_days=2,
        candles_by_symbol=candles_by_symbol,
        sector_by_symbol={"AAPL": "technology", "MSFT": "technology"},
    )

    assert rows[0]["selection_weight_herfindahl"] == 0.5
    assert rows[0]["selection_average_pairwise_correlation_63d"] == 1.0
    assert rows[0]["selection_sector_concentration"] == 1.0


def _stress_row(index: int) -> dict[str, str]:
    return {
        "exposure_target": "0.85",
        "champion_return_next_period": "0.01",
        "spy_distance_sma_200": "0.02",
        "breadth_above_sma_200": "0.60",
        "spy_realized_volatility_21d": "0.15",
        "breadth_change_since_last_rebalance": "0.00",
        "spy_volatility_ratio_21d_63d": "1.00",
        "recent_champion_excess_return_2_rebalances": "0.01",
        "spy_max_drawdown_63d": "-0.02",
        "drawdown_event": "0",
        "rebalance_date": f"2024-{index + 1:02d}-01",
    }


def _review_row(rebalance_date: str, drawdown_event: int) -> dict[str, float | str]:
    return {
        "rebalance_date": rebalance_date,
        "outcome_end_date": "2024-03-01",
        "selected_symbols": "AAPL,MSFT",
        "regime_label": "risk-on",
        "selection_count": 2.0,
        "exposure_target": 0.85,
        "cash_weight": 0.15,
        "average_rank_score": 0.2,
        "largest_weight": 0.3 if drawdown_event else 0.2,
        "replacements": 1.0,
        "spy_distance_sma_200": 0.01,
        "spy_realized_volatility_21d": 0.15,
        "spy_volatility_ratio_21d_63d": 1.1,
        "spy_max_drawdown_63d": -0.05,
        "breadth_above_sma_200": 0.6,
        "breadth_change_since_last_rebalance": -0.02,
        "recent_champion_return_2_rebalances": 0.01,
        "recent_champion_excess_return_2_rebalances": -0.01,
        "champion_return_next_period": -0.04,
        "benchmark_return_next_period": -0.02,
        "champion_excess_return": -0.02,
        "future_max_drawdown": -0.12,
        "drawdown_event": drawdown_event,
    }


def _correlated_candles(symbol: str, count: int) -> list[Candle]:
    start = datetime(2024, 1, 1)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.0 + index,
            volume=1_000.0,
        )
        for index in range(count)
    ]


def _candles(symbol: str, count: int, start_price: float) -> list[Candle]:
    start = datetime(2024, 1, 1)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=start_price + index,
            high=start_price + index,
            low=start_price + index,
            close=start_price + index,
            volume=1_000.0,
        )
        for index in range(count)
    ]


class _StaticFeed:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]]):
        self.candles_by_symbol = candles_by_symbol

    def get_historical_bars(self, symbol, timeframe, start, end):
        return self.candles_by_symbol[symbol]
