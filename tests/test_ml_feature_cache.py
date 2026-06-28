from __future__ import annotations

import json
from datetime import datetime

from core.entities.candle import Candle
from core.research.ml.feature_cache import MLFeatureCache
from core.research.ml.features import MLFeatureBuildResult


def test_feature_cache_writes_and_loads_historical_feature_rows(tmp_path):
    path = tmp_path / "features.csv"
    cache = MLFeatureCache({"ml": {"cache_feature_rows": True}})
    feature_result = MLFeatureBuildResult(
        rows=[
            {"feature_date": "2024-01-01", "return_1m": 0.1},
            {"feature_date": "2024-01-02", "return_1m": 0.2},
        ],
        dropped_rows=7,
        date_range=("2024-01-01", "2024-01-02"),
    )

    cache.write_feature_rows(path, feature_result, "cache-key")
    loaded = cache.load_feature_rows(path, "cache-key")

    assert loaded is not None
    assert loaded.rows == [
        {"feature_date": "2024-01-01", "return_1m": "0.1"},
        {"feature_date": "2024-01-02", "return_1m": "0.2"},
    ]
    assert loaded.dropped_rows == 7
    assert loaded.date_range == ("2024-01-01", "2024-01-02")

    metadata = json.loads(
        MLFeatureCache.metadata_path(path).read_text(encoding="utf-8")
    )
    assert metadata["cache_type"] == "historical_feature_rows"
    assert metadata["cache_key"] == "cache-key"
    assert metadata["row_count"] == 2
    assert metadata["date_range"] == ["2024-01-01", "2024-01-02"]
    assert metadata["dropped_rows_insufficient_lookback"] == 7
    assert metadata["research_only"] is True
    assert metadata["trading_impact"] == "none"
    assert metadata["config_hash"]
    assert metadata["generated_at"]


def test_feature_cache_respects_cache_toggle_and_cache_key(tmp_path):
    path = tmp_path / "features.csv"
    feature_result = MLFeatureBuildResult(
        rows=[{"feature_date": "2024-01-01"}],
        dropped_rows=0,
        date_range=("2024-01-01", "2024-01-01"),
    )

    disabled = MLFeatureCache({"ml": {"cache_feature_rows": False}})
    disabled.write_feature_rows(path, feature_result, "cache-key")

    assert not path.exists()

    enabled = MLFeatureCache({"ml": {"cache_feature_rows": True}})
    enabled.write_feature_rows(path, feature_result, "cache-key")

    assert enabled.load_feature_rows(path, "different-key") is None


def test_feature_cache_key_uses_ordered_history_summary():
    cache = MLFeatureCache({"ml": {"cache_feature_rows": True}})
    first_key = cache.feature_cache_key(
        ["spy", "aapl"],
        ("qqq", "spy"),
        252,
        {
            "SPY": [
                _candle("SPY", "2024-01-02", 102.0),
                _candle("SPY", "2024-01-01", 100.0),
            ]
        },
    )
    second_key = cache.feature_cache_key(
        ["AAPL", "SPY"],
        ("QQQ", "SPY"),
        252,
        {
            "SPY": [
                _candle("SPY", "2024-01-01", 100.0),
                _candle("SPY", "2024-01-02", 102.0),
            ]
        },
    )

    assert first_key == second_key


def _candle(symbol: str, date_value: str, close: float) -> Candle:
    timestamp = datetime.fromisoformat(f"{date_value}T00:00:00")
    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000,
    )
