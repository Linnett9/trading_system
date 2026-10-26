from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.research.iex_market_data import (
    IexPartitionRequest,
    RAW_BAR_COLUMNS,
    canonical_raw_bar_frame,
    plan_iex_partitions,
    validate_raw_bar_partition,
)
from infrastructure.data.iex_market_data_store import IexMarketDataStore
from infrastructure.data.iex_research_artifact_store import IexResearchArtifactStore


UTC = timezone.utc


def _row(*, symbol: str = "AAPL", minute: int = 30) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": datetime(2021, 1, 4, 14, minute, tzinfo=UTC),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1234.0,
        "trade_count": 42,
        "vwap": 100.5,
        "provider": "alpaca",
    }


def test_partition_plan_is_stable_symbol_year_and_daily() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    end = datetime(2023, 4, 1, tzinfo=UTC)

    five = plan_iex_partitions(("SPY", "AAPL", "AAPL"), start=start, end=end, timeframe="5m")
    daily = plan_iex_partitions(("SPY", "AAPL"), start=start, end=end, timeframe="1Day")

    assert [(row.symbol, row.year) for row in five] == [
        ("AAPL", 2021),
        ("AAPL", 2022),
        ("AAPL", 2023),
        ("SPY", 2021),
        ("SPY", 2022),
        ("SPY", 2023),
    ]
    assert [(row.symbol, row.year) for row in daily] == [("AAPL", None), ("SPY", None)]
    assert five[0].partition_id == five[0].partition_id


def test_canonical_raw_projection_preserves_price_precision_and_types() -> None:
    frame = canonical_raw_bar_frame([_row()])

    assert tuple(frame.columns) == RAW_BAR_COLUMNS
    assert str(frame["open"].dtype) == "float64"
    assert str(frame["volume"].dtype) == "int64"
    assert str(frame["trade_count"].dtype) == "Int64"
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_raw_partition_validation_fails_closed_on_duplicate_and_off_grid() -> None:
    request = IexPartitionRequest(
        symbol="AAPL",
        timeframe="5m",
        start=datetime(2021, 1, 1, tzinfo=UTC),
        end=datetime(2022, 1, 1, tzinfo=UTC),
        year=2021,
    )
    duplicate = canonical_raw_bar_frame([_row(), _row()])
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_bar_partition(duplicate, request)

    off_grid = canonical_raw_bar_frame([_row(minute=31)])
    quality = validate_raw_bar_partition(off_grid, request)
    assert quality["off_grid_count"] == 1


def test_store_publication_is_hash_validated_and_resumable(tmp_path) -> None:
    request = IexPartitionRequest(
        symbol="AAPL",
        timeframe="5m",
        start=datetime(2021, 1, 1, tzinfo=UTC),
        end=datetime(2022, 1, 1, tzinfo=UTC),
        year=2021,
    )
    frame = canonical_raw_bar_frame([_row()])
    quality = validate_raw_bar_partition(frame, request)
    store = IexMarketDataStore(tmp_path / "iex")

    published = store.publish(request, frame, quality=quality, source={"feed": "IEX"})
    resumed = store.completed_manifest(request)

    assert resumed == published
    assert pd.read_parquet(store.parquet_path(request)).shape == (1, len(RAW_BAR_COLUMNS))

    store.parquet_path(request).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.completed_manifest(request)


def test_research_artifact_store_validates_identity_and_hash(tmp_path) -> None:
    store = IexResearchArtifactStore(tmp_path / "iex")
    relative = Path("features/v1/5m/symbol=AAPL/year=2021")
    identity = {"dataset": "features", "symbol": "AAPL", "year": 2021}
    frame = pd.DataFrame({"symbol": ["AAPL", "AAPL"], "value": [1.0, 2.0]})

    published = store.publish(relative, frame, identity=identity, lineage={}, quality={})

    assert published["row_count"] == 2
    assert store.completed(relative, identity) == published
    with pytest.raises(ValueError, match="identity mismatch"):
        store.completed(relative, {**identity, "year": 2022})
