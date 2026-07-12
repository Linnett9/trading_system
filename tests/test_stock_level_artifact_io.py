from __future__ import annotations

import csv
from datetime import datetime, timezone

import pytest

from core.research.ml.stock_level.overnight_stock_alpha_validation import _valid_output
from core.research.ml.stock_level.stock_level_artifact_io import (
    read_stock_level_artifact,
    schema_fingerprint,
    write_stock_level_artifact,
)


def test_stock_level_artifact_writes_zstd_parquet_and_inspection_sample(tmp_path):
    path = tmp_path / "stock_level_prediction_artifacts.parquet"
    sample_path = tmp_path / "stock_level_prediction_artifacts_sample.csv"
    rows = [
        {
            "rebalance_date": "2024-01-02",
            "symbol": "AAA",
            "decision_timestamp": "2024-01-02T14:30:00Z",
            "score": 1.25,
            "rank": 1,
            "tradable": True,
            "actual_forward_return_10d": "0.10",
        },
        {
            "rebalance_date": "2024-01-03",
            "symbol": "BBB",
            "decision_timestamp": "2024-01-03T14:30:00+00:00",
            "score": None,
            "rank": 2,
            "tradable": False,
            "actual_forward_return_10d": "",
        },
    ]
    fieldnames = list(rows[0])

    identity = write_stock_level_artifact(
        path,
        rows,
        fieldnames=fieldnames,
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
        inspection_sample_path=sample_path,
    )

    assert path.exists()
    assert sample_path.exists()
    assert identity["artifact_format"] == "parquet"
    assert identity["compression"] == "zstd"
    assert identity["compression_codecs"] == ["ZSTD"]
    assert identity["stable_column_order"] == fieldnames
    assert identity["row_count"] == 2
    assert identity["symbol_count"] == 2
    assert identity["realized_target_count"] == 1
    assert identity["unrealized_boundary_count"] == 1
    assert identity["duplicate_symbol_decision_keys"] == 0
    assert identity["inspection_sample"]["inspection_only"] is True

    loaded = read_stock_level_artifact(path, required_columns={"rebalance_date", "symbol"})
    assert loaded[0]["symbol"] == "AAA"
    assert loaded[0]["decision_timestamp"] == datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    assert loaded[1]["score"] is None
    assert loaded[1]["tradable"] is False


def test_stock_level_artifact_rejects_csv_unless_fallback_is_explicit(tmp_path):
    path = tmp_path / "stock_level_prediction_artifacts.csv"
    path.write_text("rebalance_date,symbol\n2024-01-02,AAA\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to read non-canonical"):
        read_stock_level_artifact(path)

    assert read_stock_level_artifact(path, allow_csv_fallback=True) == [
        {"rebalance_date": "2024-01-02", "symbol": "AAA"}
    ]


def test_stock_level_artifact_fails_closed_on_schema_or_corrupt_parquet(tmp_path):
    path = tmp_path / "stock_level_prediction_artifacts.parquet"
    write_stock_level_artifact(
        path,
        [{"rebalance_date": "2024-01-02", "symbol": "AAA"}],
        fieldnames=["rebalance_date", "symbol"],
        config={"ml": {"stock_level_artifact_format": "parquet"}},
    )
    expected = schema_fingerprint(["symbol", "rebalance_date"], None)

    with pytest.raises(ValueError, match="Schema fingerprint mismatch"):
        read_stock_level_artifact(path, expected_schema_fingerprint=expected)

    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not a parquet file")
    with pytest.raises(ValueError, match="Could not read Parquet artifact"):
        read_stock_level_artifact(corrupt)


def test_valid_output_accepts_parquet_required_columns_and_rejects_missing(tmp_path):
    path = tmp_path / "stock_level_prediction_artifacts.parquet"
    write_stock_level_artifact(
        path,
        [],
        fieldnames=["rebalance_date", "symbol"],
        config={"ml": {"stock_level_artifact_format": "parquet"}},
    )

    assert _valid_output(path, {"rebalance_date", "symbol"}) is True
    assert _valid_output(path, {"rebalance_date", "missing_column"}) is False


def test_parquet_rows_match_legacy_csv_serialization(tmp_path):
    rows = [
        {"rebalance_date": "2024-01-02", "symbol": "AAA", "score": "1.5"},
        {"rebalance_date": "2024-01-03", "symbol": "BBB", "score": ""},
    ]
    fieldnames = ["rebalance_date", "symbol", "score"]
    parquet_path = tmp_path / "stock_level_prediction_artifacts.parquet"
    csv_path = tmp_path / "legacy.csv"

    write_stock_level_artifact(
        parquet_path,
        rows,
        fieldnames=fieldnames,
        config={"ml": {"stock_level_artifact_format": "parquet"}},
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    parquet_rows = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in read_stock_level_artifact(parquet_path)
    ]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert parquet_rows == csv_rows
