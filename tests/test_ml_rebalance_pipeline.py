from __future__ import annotations

import csv
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.research.ml.artifacts import MLFeatureCache
from core.research.ml.config import MLExperimentConfig
from core.research.ml.features.features import MLFeatureBuildResult
from core.research.ml.pipelines import MLRebalancePipeline


@pytest.fixture(autouse=True)
def _restore_rebalance_dataset_module():
    previous = sys.modules.get("core.research.ml.data.rebalance_dataset")
    yield
    if previous is None:
        sys.modules.pop("core.research.ml.data.rebalance_dataset", None)
    else:
        sys.modules["core.research.ml.data.rebalance_dataset"] = previous


def test_rebalance_pipeline_reads_existing_expanded_dataset(tmp_path):
    cache_path = tmp_path / "expanded.csv"
    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_date", "alpha"])
        writer.writeheader()
        writer.writerow({"feature_date": "2024-01-01", "alpha": "1.0"})
        writer.writerow({"feature_date": "2024-01-02", "alpha": "2.0"})
    pipeline = _pipeline({
        "label_type": "should_reduce_exposure",
        "read_existing_expanded_rebalance_dataset": True,
        "expanded_rebalance_dataset_path": str(cache_path),
    })

    result = pipeline.build_expanded_rebalance_features(
        MLFeatureBuildResult(rows=[], dropped_rows=7, date_range=None),
        {},
    )

    assert result.rows == [
        {"feature_date": "2024-01-01", "alpha": "1.0"},
        {"feature_date": "2024-01-02", "alpha": "2.0"},
    ]
    assert result.dropped_rows == 7
    assert result.date_range == ("2024-01-01", "2024-01-02")


def test_rebalance_pipeline_preserves_expanded_cache_key_shape():
    pipeline = _pipeline({
        "label_type": "should_reduce_exposure",
        "label_horizon_days": 5,
        "benchmark_symbols": ["SPY", "QQQ"],
        "sector_by_symbol": {"SPY": "ETF"},
    })
    feature_result = MLFeatureBuildResult(
        rows=[{"feature_date": "2024-01-01", "alpha": 1.0}],
        dropped_rows=0,
        date_range=("2024-01-01", "2024-01-01"),
    )
    candles_by_symbol = {"SPY": [_candle("2024-01-01", 100.0)]}

    key = pipeline.expanded_rebalance_cache_key(feature_result, candles_by_symbol)

    assert key == MLFeatureCache.hash_payload({
        "cache_version": 1,
        "cache_type": "expanded_rebalance_dataset",
        "label_type": "should_reduce_exposure",
        "label_horizon_days": 5,
        "expanded_rebalance_dataset": {},
        "benchmark_symbols": ["SPY", "QQQ"],
        "sector_reference_path": None,
        "sector_by_symbol": {"SPY": "ETF"},
        "feature_rows_hash": MLFeatureCache.rows_hash(feature_result.rows),
        "feature_row_count": 1,
        "feature_date_range": ("2024-01-01", "2024-01-01"),
        "history": MLFeatureCache.candles_cache_summary(candles_by_symbol),
    })


def test_rebalance_pipeline_writes_should_reduce_dataset_audit_and_rule_study(tmp_path):
    _install_yaml_stub()
    pipeline = _pipeline({"label_type": "should_reduce_exposure"})
    rows = [_rebalance_row("2024-01-01", should_reduce=1)]

    returned = pipeline.write_rebalance_dataset(
        tmp_path / "rebalance.csv",
        tmp_path / "rebalance_audit.json",
        rows,
        {},
        tmp_path / "rule_study.json",
    )

    assert returned == rows
    with (tmp_path / "rebalance.csv").open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["should_reduce_exposure"] == "1"

    audit = json.loads((tmp_path / "rebalance_audit.json").read_text())
    assert audit == {
        "row_count": 1,
        "should_reduce_exposure_rate": 1.0,
        "drawdown_event_rate": 0.0,
        "underperforms_spy_rate": 1.0,
        "source": "expanded_rebalance_dataset",
        "research_only": True,
        "trading_impact": "none",
    }
    rule_study = json.loads((tmp_path / "rule_study.json").read_text())
    assert rule_study["mode"] == "expanded_rebalance_rule_based_research_only"
    assert rule_study["rules"][0]["transaction_cost_bps"] == 5.0
    assert rule_study["trading_impact"] == "none"


def test_rebalance_pipeline_writes_champion_audit_schema_for_empty_rows(tmp_path):
    _install_yaml_stub()
    pipeline = _pipeline({
        "label_type": "champion_success",
        "research_years": 9,
        "minimum_history_years": 4,
    }, backtest={"years": 3})

    returned = pipeline.write_rebalance_dataset(
        tmp_path / "rebalance.csv",
        tmp_path / "rebalance_audit.json",
        [_rebalance_row("2024-01-01", should_reduce=0)],
        {"SPY": []},
        tmp_path / "rule_study.json",
    )

    assert returned == []
    audit = json.loads((tmp_path / "rebalance_audit.json").read_text())
    assert audit == {
        "row_count": 0,
        "good_period_rate": None,
        "bad_period_rate": None,
        "underperforms_spy_rate": None,
        "drawdown_event_rate": None,
        "history_years": 3,
        "recommended_generalization_years": 9,
        "minimum_history_years": 4,
        "sector_reference_path": None,
        "research_only": True,
    }
    rule_study = json.loads((tmp_path / "rule_study.json").read_text())
    assert rule_study["mode"] == "rule_based_research_only"
    assert rule_study["rules"][0]["transaction_cost_bps"] == 5.0
    assert rule_study["trading_impact"] == "none"


def test_rebalance_pipeline_loads_inline_sector_mapping():
    pipeline = _pipeline({"sector_by_symbol": {"spy": "ETF"}})

    assert pipeline.sector_by_symbol() == {"SPY": "ETF"}
    assert MLRebalancePipeline.row_rate([{"flag": 1}, {"flag": 0}], "flag") == 0.5


def _pipeline(
    ml_config: dict,
    *,
    backtest: dict | None = None,
) -> MLRebalancePipeline:
    config = {"ml": {"model_type": "noop", **ml_config}, "backtest": backtest or {}}
    return MLRebalancePipeline(
        config,
        MLExperimentConfig.from_config(config),
        champion_equity_curve=[],
        champion_selections=[],
    )


def _rebalance_row(feature_date: str, *, should_reduce: int) -> dict[str, float | str]:
    return {
        "feature_date": feature_date,
        "rebalance_date": feature_date,
        "should_reduce_exposure": should_reduce,
        "drawdown_event": 0,
        "underperforms_spy": 1,
        "spy_distance_sma_200": -0.1,
        "breadth_above_sma_200": 0.4,
        "spy_realized_volatility_21d": 0.23,
        "spy_max_drawdown_63d": -0.05,
        "breadth_change_since_last_rebalance": -0.01,
        "spy_volatility_ratio_21d_63d": 1.0,
        "recent_champion_excess_return_2_rebalances": 0.0,
        "champion_return_next_period": 0.02,
        "exposure_target": 1.0,
    }


def _candle(date_text: str, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=SimpleNamespace(date=lambda: SimpleNamespace(isoformat=lambda: date_text)),
        close=close,
    )


def _install_yaml_stub() -> None:
    sys.modules.setdefault("yaml", SimpleNamespace(safe_load=lambda _: {}))
    module = ModuleType("core.research.ml.data.rebalance_dataset")
    module.write_rebalance_dataset = _write_rebalance_dataset
    module.build_champion_rebalance_rows = lambda *args, **kwargs: []
    sys.modules["core.research.ml.data.rebalance_dataset"] = module


def _write_rebalance_dataset(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["rebalance_date"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
