from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.five_minute_target_dataset import (
    DEFAULT_TARGET_IDS,
    ELIGIBLE_DECISION_ROW,
    EXCLUDED_DUPLICATE_SOURCE_BAR,
    EXCLUDED_INVALID_PRICE,
    EXCLUDED_NON_REGULAR_SESSION,
    FiveMinuteTargetDatasetError,
    build_five_minute_target_dataset,
    read_five_minute_source_slice,
)


def test_source_partition_resolution_filters_before_python_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "AAA", [*_bars("AAA", date(2024, 1, 2)), *_bars("AAA", date(2024, 1, 9))])

    result = read_five_minute_source_slice(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
    )

    assert result.filters_applied is True
    assert result.whole_archive_scanned is False
    assert {row["session_date"] for row in result.rows} == {"2024-01-02"}


def test_complete_paths_all_four_targets_and_manifest_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = [
        *_bars("AAA", date(2024, 1, 2), start_price=100.0),
        *_bars("AAA", date(2024, 1, 3), start_price=300.0),
    ]
    _write_source(source, "AAA", rows)

    result = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=DEFAULT_TARGET_IDS,
        output_root=tmp_path / "out",
        source_cutoff="2024-01-03T14:35:00Z",
        training_cutoff="2024-01-03T14:35:00Z",
    )

    assert set(row["target_id"] for row in result.target_rows) == set(DEFAULT_TARGET_IDS)
    assert result.dataset_manifest["permitted_use"] == "FIXTURE_OR_MECHANICAL_VALIDATION_ONLY"
    assert result.dataset_manifest["model_training_performed"] is False
    assert (tmp_path / "out" / "target_rows.parquet").exists()
    assert (tmp_path / "out" / "dataset_manifest.json").exists()


def test_decision_row_exclusions_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = _bars("AAA", date(2024, 1, 2), duplicate_first=True, include_extended=True)
    rows[5]["close"] = -1.0
    _write_source(source, "AAA", rows)

    result = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "out",
        source_cutoff="2024-01-02T21:00:00Z",
    )
    counts = result.dataset_manifest["decision_classification_counts"]

    assert counts[EXCLUDED_DUPLICATE_SOURCE_BAR] == 2
    assert counts[EXCLUDED_INVALID_PRICE] == 1
    assert counts[EXCLUDED_NON_REGULAR_SESSION] == 1
    assert counts[ELIGIBLE_DECISION_ROW] > 0


def test_missing_intermediate_bar_and_near_close_are_classified(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = _bars("AAA", date(2024, 1, 2), missing_finals={"2024-01-02T15:05:00Z"})
    _write_source(source, "AAA", rows)

    result = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "out",
        source_cutoff="2024-01-02T21:00:00Z",
    )
    classes = {row["target_resolution_classification"] for row in result.target_rows}

    assert "MISSING_SOURCE_BAR" in classes
    assert "SESSION_BOUNDARY_CONFLICT" in classes


def test_early_close_uses_calendar_close(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "AAA", _bars("AAA", date(2024, 7, 3), close_hour_utc=17, start_price=100.0))

    result = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-07-03",
        end_date="2024-07-03",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "out",
        source_cutoff="2024-07-03T17:00:00Z",
    )

    assert any(row["target_end_timestamp"] == "2024-07-03T17:00:00Z" for row in result.target_rows)
    assert any(row["target_resolution_classification"] == "SESSION_BOUNDARY_CONFLICT" for row in result.target_rows)


def test_next_open_weekend_holiday_and_dst(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = [
        *_bars("AAA", date(2024, 1, 5), start_price=100.0),
        *_bars("AAA", date(2024, 1, 8), start_price=200.0),
        *_bars("BBB", date(2024, 1, 12), start_price=100.0),
        *_bars("BBB", date(2024, 1, 16), start_price=210.0),
        *_bars("CCC", date(2024, 3, 8), start_price=100.0),
        *_bars("CCC", date(2024, 3, 11), open_hour_utc=13, start_price=220.0),
    ]
    for symbol in ("AAA", "BBB", "CCC"):
        _write_source(source, symbol, [row for row in rows if row["canonical_symbol"] == symbol])

    result = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA", "BBB", "CCC"],
        start_date="2024-01-05",
        end_date="2024-03-08",
        target_ids=["forward_return_next_open__decision_5m"],
        output_root=tmp_path / "out",
        source_cutoff="2024-03-11T13:35:00Z",
        allow_large_build=True,
    )
    ends = {row["target_end_timestamp"] for row in result.target_rows if row["target_resolution_classification"] == "MATURED_VALID"}

    assert "2024-01-08T14:30:00Z" in ends
    assert "2024-01-16T14:30:00Z" in ends
    assert "2024-03-11T13:30:00Z" in ends


def test_represented_halt_right_censored_and_not_yet_mature(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "AAA", [*_bars("AAA", date(2024, 1, 2)), *_bars("AAA", date(2024, 1, 3), start_price=200.0)])

    halted = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "halted",
        source_cutoff="2024-01-02T21:00:00Z",
        market_halts=({"start": "2024-01-02T15:00:00Z", "end": "2024-01-02T15:20:00Z"},),
    )
    censored = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "censored",
        source_cutoff="2024-01-02T14:45:00Z",
    )
    immature = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_next_open__decision_5m"],
        output_root=tmp_path / "immature",
        source_cutoff="2024-01-03T14:30:00Z",
    )

    assert "HALT_AFFECTED" in {row["target_resolution_classification"] for row in halted.target_rows}
    assert "RIGHT_CENSORED" in {row["target_resolution_classification"] for row in censored.target_rows}
    assert "NOT_YET_MATURE" in {row["target_resolution_classification"] for row in immature.target_rows}


def test_training_cutoff_controls_trainability(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "AAA", _bars("AAA", date(2024, 1, 2)))

    before = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "before",
        source_cutoff="2024-01-02T21:00:00Z",
        training_cutoff="2024-01-02T14:45:00Z",
    )
    after = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "after",
        source_cutoff="2024-01-02T21:00:00Z",
        training_cutoff="2024-01-02T21:00:00Z",
    )

    assert before.dataset_manifest["trainability_counts"]["trainable"] < after.dataset_manifest["trainability_counts"]["trainable"]


def test_deterministic_rebuild_atomic_publication_and_safety_guard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, "AAA", _bars("AAA", date(2024, 1, 2)))

    first = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "first",
        source_cutoff="2024-01-02T21:00:00Z",
    )
    second = build_five_minute_target_dataset(
        source_root=source,
        symbols=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        target_ids=["forward_return_60m__decision_5m"],
        output_root=tmp_path / "second",
        source_cutoff="2024-01-02T21:00:00Z",
    )

    assert first.content_hash == second.content_hash
    assert first.schema_hash == second.schema_hash
    with pytest.raises(FiveMinuteTargetDatasetError, match="already exists"):
        build_five_minute_target_dataset(
            source_root=source,
            symbols=["AAA"],
            start_date="2024-01-02",
            end_date="2024-01-02",
            target_ids=["forward_return_60m__decision_5m"],
            output_root=tmp_path / "first",
            source_cutoff="2024-01-02T21:00:00Z",
        )
    with pytest.raises(FiveMinuteTargetDatasetError, match="max_workers=1"):
        build_five_minute_target_dataset(
            source_root=source,
            symbols=["AAA"],
            start_date="2024-01-02",
            end_date="2024-01-02",
            target_ids=["forward_return_60m__decision_5m"],
            output_root=tmp_path / "blocked",
            source_cutoff="2024-01-02T21:00:00Z",
            protected_process_active=True,
            max_workers=2,
        )


def _bars(
    symbol: str,
    session: date,
    *,
    start_price: float = 100.0,
    missing_finals: set[str] | None = None,
    duplicate_first: bool = False,
    include_extended: bool = False,
    open_hour_utc: int = 14,
    close_hour_utc: int = 21,
) -> list[dict[str, object]]:
    missing = missing_finals or set()
    open_time = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=open_hour_utc, minutes=30)
    close_time = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=close_hour_utc)
    rows: list[dict[str, object]] = []
    if include_extended:
        rows.append(_row(symbol, session, open_time - timedelta(minutes=30), start_price - 1, "pre_market"))
    current = open_time
    index = 0
    while current < close_time:
        final = current + timedelta(minutes=5)
        if _fmt(final) not in missing:
            row = _row(symbol, session, current, start_price + index, "rth")
            rows.append(row)
            if duplicate_first and index == 0:
                duplicate = dict(row)
                duplicate["source_row_hash"] = str(duplicate["source_row_hash"]) + "-dup"
                rows.append(duplicate)
        current += timedelta(minutes=5)
        index += 1
    return rows


def _row(symbol: str, session: date, timestamp: datetime, price: float, session_type: str) -> dict[str, object]:
    ts = _fmt(timestamp)
    return {
        "asset_id": symbol,
        "canonical_symbol": symbol,
        "provider_symbol": symbol,
        "timestamp_utc": ts,
        "session_date": session.isoformat(),
        "session_type": session_type,
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price + 0.25,
        "volume": 1000.0,
        "trade_count": 10,
        "vwap": price + 0.1,
        "provider": "synthetic",
        "feed": "fixture",
        "timeframe": "5m",
        "adjustment_policy": "raw",
        "raw_chunk_id": f"{symbol}-{session.isoformat()}",
        "source_row_hash": f"{symbol}-{ts}",
        "dataset_version": "fixture_v1",
    }


def _write_source(root: Path, symbol: str, rows: list[dict[str, object]]) -> None:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(str(row["session_date"])[:4]), []).append(row)
    for year, year_rows in grouped.items():
        path = root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(year_rows), path, compression="zstd")


def _fmt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

