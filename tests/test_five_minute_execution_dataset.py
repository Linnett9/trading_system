from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from core.research.ml.stock_level.five_minute_execution_dataset import (
    STATUS_PASSED,
    Ticket10AConfig,
    validate_temporal_safety,
    write_ticket_10a_execution_dataset_probe,
)
from infrastructure.data.market_sessions import EASTERN


def test_ticket_10a_probe_writes_artifacts_and_temporal_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, sessions=("2026-06-24", "2026-06-25", "2026-06-26"))
    _write_source(config)

    result = write_ticket_10a_execution_dataset_probe(config, code_commit="testhead")

    assert result["status"] == STATUS_PASSED
    assert result["dataset_path"].exists()
    assert result["manifest_path"].exists()
    assert result["feature_contract_path"].exists()
    assert result["target_contract_path"].exists()
    assert result["temporal_audit_path"].exists()
    assert result["summary_path"].exists()
    validation = result["validation"]
    assert validation["duplicate_symbol_decision_keys"] == 0
    assert validation["future_feature_violations"] == 0
    assert validation["target_timestamp_violations"] == 0
    assert validation["symbols"] == ["AAPL", "MSFT", "SPY"]
    rows = result["rows"]
    assert rows
    assert all(row["source_bar_end_timestamp"] <= row["decision_timestamp_utc"] for row in rows)
    assert rows[0]["decision_timestamp_exchange"].endswith("-04:00")
    assert json.loads(result["manifest_path"].read_text(encoding="utf-8"))["code_commit"] == "testhead"
    table = pq.read_table(result["dataset_path"])
    assert table.num_rows == len(rows)


def test_future_bar_change_does_not_change_earlier_features(tmp_path: Path) -> None:
    config = _config(tmp_path, sessions=("2026-06-24", "2026-06-25"))
    rows = _source_rows(config.sessions)
    _write_source(config, rows=rows)
    first = write_ticket_10a_execution_dataset_probe(config)
    key = ("AAPL", "2026-06-24T13:55:00+00:00")
    before = _row_by_key(first["rows"], key)

    for row in rows:
        if row["symbol"] == "AAPL" and row["timestamp"] == "2026-06-24T14:30:00+00:00":
            row["close"] = 999.0
            row["high"] = 1000.0
    _write_source(config, rows=rows)
    second = write_ticket_10a_execution_dataset_probe(config)
    after = _row_by_key(second["rows"], key)

    checked = [
        "return_5m",
        "return_15m",
        "session_to_date_return",
        "relative_volume_same_time_of_day",
        "SPY_return_5m",
        "momentum_rank",
    ]
    assert {column: before[column] for column in checked} == {column: after[column] for column in checked}


def test_future_session_does_not_change_prior_same_time_normalization(tmp_path: Path) -> None:
    all_sessions = ("2026-06-24", "2026-06-25", "2026-06-26")
    source_config = _config(tmp_path, sessions=all_sessions)
    _write_source(source_config, rows=_source_rows(all_sessions))
    two_day = _config(tmp_path, sessions=("2026-06-24", "2026-06-25"))
    three_day = _config(tmp_path, sessions=all_sessions)

    first = write_ticket_10a_execution_dataset_probe(two_day)
    second = write_ticket_10a_execution_dataset_probe(three_day)
    key = ("MSFT", "2026-06-25T13:35:00+00:00")

    assert _row_by_key(first["rows"], key)["relative_volume_same_time_of_day"] == _row_by_key(second["rows"], key)["relative_volume_same_time_of_day"]
    assert _row_by_key(first["rows"], key)["volume_zscore_same_time_of_day"] == _row_by_key(second["rows"], key)["volume_zscore_same_time_of_day"]


def test_temporal_audit_detects_future_selector_and_portfolio_state(tmp_path: Path) -> None:
    config = _config(tmp_path, sessions=("2026-06-24",))
    _write_source(config)
    result = write_ticket_10a_execution_dataset_probe(config)
    row = dict(result["rows"][0])
    row["selector_as_of_timestamp"] = "2026-06-25T13:30:00+00:00"
    row["portfolio_state_as_of_timestamp"] = "2026-06-25T13:30:00+00:00"

    audit = validate_temporal_safety([row])

    assert audit["selector_future_violations"] == 1
    assert audit["portfolio_future_violations"] == 1


def test_target_columns_are_not_predictors(tmp_path: Path) -> None:
    config = _config(tmp_path, sessions=("2026-06-24",))
    _write_source(config)

    result = write_ticket_10a_execution_dataset_probe(config)
    feature_contract = json.loads(result["feature_contract_path"].read_text(encoding="utf-8"))
    target_contract = json.loads(result["target_contract_path"].read_text(encoding="utf-8"))

    assert not (set(feature_contract["predictor_columns"]) & set(target_contract["target_columns"]))


def _config(tmp_path: Path, *, sessions: tuple[str, ...]) -> Ticket10AConfig:
    parquet_root = tmp_path / "parquet"
    raw_root = tmp_path / "raw"
    source = parquet_root / "iex" / "5m" / "SPY-AAPL-MSFT" / "20260624T133000Z_20260626T150000Z" / "bars.parquet"
    return Ticket10AConfig(
        source_files=(source,),
        raw_root=raw_root,
        parquet_root=parquet_root,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        sessions=sessions,
    )


def _write_source(config: Ticket10AConfig, *, rows: list[dict] | None = None) -> None:
    source = config.source_files[0]
    source.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or _source_rows(config.sessions)
    pq.write_table(pa.Table.from_pylist(rows), source)
    raw_dir = config.raw_root / source.relative_to(config.parquet_root).parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "provider": "alpaca",
        "feed": "iex",
        "timeframe_requested": "5m",
        "native_timeframe": "5Min",
        "symbol_batch": ["SPY", "AAPL", "MSFT"],
        "row_count": len(rows),
        "adjustment_mode": "all",
        "completion_state": "completed",
    }
    tombstone = {
        "validation_result": "passed",
        "parquet_path": str(source),
        "parquet_row_count": len(rows),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw_dir / "parquet_conversion.json").write_text(json.dumps(tombstone), encoding="utf-8")


def _source_rows(sessions: tuple[str, ...]) -> list[dict]:
    rows = []
    for session_index, session in enumerate(sessions):
        day = date.fromisoformat(session)
        current = datetime.combine(day, time(9, 30), tzinfo=EASTERN)
        for bar_index in range(15):
            for symbol_index, symbol in enumerate(("SPY", "AAPL", "MSFT")):
                base = 100.0 + symbol_index * 50 + session_index * 2 + bar_index * 0.1
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": current.astimezone(timezone.utc).isoformat(),
                        "open": base,
                        "high": base + 0.3,
                        "low": base - 0.2,
                        "close": base + 0.05,
                        "volume": 1000.0 + symbol_index * 100 + session_index * 20 + bar_index * 5,
                        "trade_count": 10,
                        "vwap": base + 0.02,
                        "provider": "alpaca",
                        "feed": "iex",
                        "collection_timestamp": "2026-07-01T00:00:00+00:00",
                        "requested_timeframe": "5m",
                        "native_timeframe": "5Min",
                        "adjustment_mode": "all",
                        "extended_hours": False,
                        "session_policy": "regular_session_default",
                        "session_type": None,
                        "raw_chunk_identifier": "test-chunk",
                        "normalizer_version": "historical_bar_provider_v1",
                    }
                )
            current += timedelta(minutes=5)
    return rows


def _row_by_key(rows: list[dict], key: tuple[str, str]) -> dict:
    symbol, decision = key
    for row in rows:
        if row["symbol"] == symbol and row["decision_timestamp_utc"] == decision:
            return row
    raise AssertionError(f"missing row {key}")

