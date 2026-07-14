from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from infrastructure.data.alpaca_daily_preflight import (
    DEFAULT_DAILY_ARCHIVE_ROOT,
    classify_universe,
    estimate_request_count,
    latest_completed_session_semantic_date,
    reconcile_smoke_rows,
    recommended_overlap_start,
    run_preflight,
    run_reconcile_only,
    write_daily_archive,
)
from infrastructure.data.historical_bar_providers import _alpaca_timeframe


def test_1day_timeframe_maps_to_alpaca_native_daily():
    assert _alpaca_timeframe("1Day") == "1Day"


def test_latest_completed_session_semantics_skip_weekends():
    assert latest_completed_session_semantic_date(datetime(2026, 7, 13)) == "2026-07-10"


def test_request_count_estimate_uses_symbol_batches_and_date_windows():
    assert estimate_request_count(514, "2026-03-27", "2026-07-10", 50, 31) == 44


def test_overlap_start_uses_60_prior_spy_sessions(tmp_path):
    stooq = tmp_path / "stooq"
    _write_stooq(stooq / "SPY.parquet", "SPY", [datetime(2026, 1, 1) + timedelta(days=i) for i in range(80)])

    start = recommended_overlap_start(
        [{"symbol": "SPY", "latest_session": "2026-03-21"}],
        anchor="2026-03-21",
        stooq_root=stooq,
    )

    assert start == "2026-01-20"


def test_classification_preserves_brk_alpaca_mapping():
    asset = _asset("asset_brkb", "BRK-B")
    alias = _alias("asset_brkb", "BRK.B", "configured_provider_map")

    rows = classify_universe([asset], [alias], stooq_rows=[])

    assert rows[0]["canonical_symbol"] == "BRK-B"
    assert rows[0]["alpaca_provider_symbol"] == "BRK.B"
    assert rows[0]["alpaca_mapping_status"] == "mapped"


def test_missing_alias_is_not_guessed_but_reports_provider_symbol_candidate():
    rows = classify_universe([_asset("asset_x", "XYZ")], [], stooq_rows=[])

    assert rows[0]["alpaca_mapping_status"] == "missing"
    assert rows[0]["alpaca_provider_symbol"] == "XYZ"


def test_daily_archive_path_is_separate_from_stooq_and_5m():
    text = str(DEFAULT_DAILY_ARCHIVE_ROOT).replace("\\", "/")
    assert text == "data/processed/alpaca/symbol_bars/sip/1d"
    assert "stooq" not in text
    assert "5m" not in text


def test_daily_archive_writer_partitions_with_required_fields(tmp_path):
    _stooq, assets, aliases = _fixture_inputs(tmp_path)
    root = tmp_path / "archive"

    report = write_daily_archive(
        [
            {
                "symbol": "BRK-B",
                "provider_symbol": "BRK.B",
                "timestamp": datetime(2026, 1, 2),
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
                "volume": 100,
                "trade_count": 10,
                "vwap": 1.5,
                "feed": "sip",
                "adjustment_mode": "all",
                "raw_chunk_identifier": "chunk",
            }
        ],
        archive_root=root,
        asset_registry=assets,
        alias_registry=aliases,
        dataset_version="test_daily",
    )

    path = root / "symbol=BRK-B" / "year=2026" / "bars.parquet"
    row = pq.read_table(path).to_pylist()[0]
    assert report["written_rows"] == 1
    assert row["asset_id"] == "asset_BRK-B"
    assert row["canonical_symbol"] == "BRK-B"
    assert row["provider_symbol"] == "BRK.B"
    assert row["session_date"] == "2026-01-02"
    assert row["timeframe"] == "1Day"
    assert row["adjustment_policy"] == "all"
    assert row["dataset_version"] == "test_daily"


def test_reconciliation_classifies_missing_live_alpaca_rows():
    rows, summary = reconcile_smoke_rows(
        [
            {"symbol": "AAPL", "session_date": "2026-01-02", "close": 10.0, "volume": 100},
            {"symbol": "AAPL", "session_date": "2026-01-05", "close": 11.0, "volume": 120},
        ]
    )

    assert summary["canonical_source_selected"] is False
    assert summary["classification"] == "blocked_until_live_smoke"
    assert summary["stooq_only_rows"] == 2
    assert rows[1]["stooq_return"] == pytest.approx(0.1)


def test_reconciliation_flags_adjustment_discrepancy():
    rows, summary = reconcile_smoke_rows(
        [{"symbol": "AAPL", "session_date": "2026-01-02", "close": 10.0, "volume": 100}],
        [{"symbol": "AAPL", "session_date": "2026-01-02", "close": 10.5, "volume": 100}],
    )

    assert rows[0]["possible_adjustment_discrepancy"] == "true"
    assert summary["possible_adjustment_discrepancy_count"] == 1


def test_preflight_writes_requested_reports(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    report_root = tmp_path / "reports"

    payload = run_preflight(
        report_root=report_root,
        stooq_root=stooq,
        asset_registry=assets,
        alias_registry=aliases,
        smoke_output_root=tmp_path / "smoke",
        daily_archive_root=tmp_path / "daily_archive",
        smoke_symbols=("AAPL", "SPY", "BRK-B", "MSFT"),
    )

    assert payload["direct_daily_supported"] is True
    for name in [
        "source_freshness.csv",
        "universe_classification.csv",
        "smoke_collection_report.json",
        "smoke_coverage.csv",
        "smoke_reconciliation.csv",
        "smoke_reconciliation.json",
        "production_plan.json",
    ]:
        assert (report_root / name).exists()


def test_preflight_dry_run_writes_nothing(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    report_root = tmp_path / "reports"

    run_preflight(
        report_root=report_root,
        stooq_root=stooq,
        asset_registry=assets,
        alias_registry=aliases,
        smoke_output_root=tmp_path / "smoke",
        daily_archive_root=tmp_path / "daily_archive",
        dry_run=True,
    )

    assert not report_root.exists()


def test_preflight_rejects_rate_limit_above_180(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)

    with pytest.raises(ValueError, match="<= 180"):
        run_preflight(
            report_root=tmp_path / "reports",
            stooq_root=stooq,
            asset_registry=assets,
            alias_registry=aliases,
            requests_per_minute=181,
        )


def test_production_plan_records_manual_command_and_no_canonical_write(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    report_root = tmp_path / "reports"
    run_preflight(report_root=report_root, stooq_root=stooq, asset_registry=assets, alias_registry=aliases)

    plan = json.loads((report_root / "production_plan.json").read_text(encoding="utf-8"))
    assert plan["canonical_market_data_writes_enabled"] is False
    assert "ml-historical-bar-backfill-collect" in plan["manual_command"]


def test_universe_report_counts_valid_alpaca_mappings(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    payload = run_preflight(
        report_root=tmp_path / "reports",
        stooq_root=stooq,
        asset_registry=assets,
        alias_registry=aliases,
        dry_run=True,
    )

    assert payload["valid_alpaca_mappings"] == 5
    assert payload["universe_rows"] == 5


def test_smoke_coverage_reports_alias_and_stooq_endpoint(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    report_root = tmp_path / "reports"
    run_preflight(report_root=report_root, stooq_root=stooq, asset_registry=assets, alias_registry=aliases)

    with (report_root / "smoke_coverage.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    brk = next(row for row in rows if row["canonical_symbol"] == "BRK-B")
    assert brk["alpaca_provider_symbol"] == "BRK.B"
    assert brk["mapping_status"] == "mapped"


def test_preflight_reads_smoke_archive_for_reconciliation(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    archive = tmp_path / "archive"
    report_root = tmp_path / "reports"
    write_daily_archive(
        [
            {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 4), "open": 1, "high": 1, "low": 1, "close": 2.1, "volume": 200, "feed": "sip", "adjustment_mode": "all"},
            {"symbol": "SPY", "timestamp": datetime(2026, 4, 1, 4), "open": 1, "high": 1, "low": 1, "close": 3.0, "volume": 300, "feed": "sip", "adjustment_mode": "all"},
        ],
        archive_root=archive,
        asset_registry=assets,
        alias_registry=aliases,
        dataset_version="smoke",
    )

    run_preflight(
        report_root=report_root,
        stooq_root=stooq,
        asset_registry=assets,
        alias_registry=aliases,
        smoke_archive_root=archive,
        smoke_symbols=("AAPL", "SPY"),
    )

    summary = json.loads((report_root / "smoke_reconciliation.json").read_text(encoding="utf-8"))
    assert summary["classification"] == "live_smoke_reconciled"
    assert summary["matched_rows"] == 1
    assert summary["alpaca_only_rows"] == 1
    assert summary["relative_close_difference_count"] == 1


def test_reconcile_only_reads_256_row_archive_and_replaces_stale_reports(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    archive = tmp_path / "archive"
    report_root = tmp_path / "reports"
    stale = report_root / "smoke_reconciliation.json"
    report_root.mkdir(parents=True)
    stale.write_text(json.dumps({"classification": "blocked_until_live_smoke", "matched_rows": 0}), encoding="utf-8")
    rows = []
    for symbol in ["AAPL", "SPY", "BRK-B", "ABCB"]:
        for index in range(64):
            rows.append(
                {
                    "symbol": symbol,
                    "provider_symbol": "BRK.B" if symbol == "BRK-B" else symbol,
                    "timestamp": datetime(2026, 1, 1, 4) + timedelta(days=index),
                    "open": 1.0,
                    "high": 2.0,
                    "low": 1.0,
                    "close": float(index + 1),
                    "volume": float(index + 100),
                    "feed": "sip",
                    "adjustment_mode": "all",
                }
            )
    write_daily_archive(rows, archive_root=archive, asset_registry=assets, alias_registry=aliases, dataset_version="overlap")

    payload = run_reconcile_only(
        alpaca_archive_root=archive,
        stooq_root=stooq,
        report_root=report_root,
        symbols=["AAPL", "SPY", "BRK-B", "ABCB"],
        start="2026-01-01",
        end="2026-03-05",
    )

    summary = payload["reconciliation"]
    assert summary["alpaca_archive_row_count"] == 256
    assert summary["matched_rows"] == 256
    assert summary["classification"] == "live_smoke_reconciled"
    assert summary["provider_compatibility_decision"] in {"PROVIDER_COMPATIBILITY_ACCEPTABLE", "PROVIDER_COMPATIBILITY_REVIEW_REQUIRED"}
    replaced = json.loads(stale.read_text(encoding="utf-8"))
    assert replaced["alpaca_archive_row_count"] == 256
    with (report_root / "smoke_reconciliation.csv").open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == 256
    assert all(row["classification"] for row in output_rows)
    assert next(row for row in output_rows if row["symbol"] == "BRK-B")["alpaca_close"]
    assert json.loads((report_root / "smoke_collection_report.json").read_text(encoding="utf-8"))["api_requests_attempted"] == 0


def test_reconcile_only_classifies_missing_and_differences(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    archive = tmp_path / "archive"
    write_daily_archive(
        [
            {"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 4), "open": 1, "high": 1, "low": 1, "close": 2.1, "volume": 100, "feed": "sip", "adjustment_mode": "all"},
            {"symbol": "AAPL", "timestamp": datetime(2026, 4, 1, 4), "open": 1, "high": 1, "low": 1, "close": 3.0, "volume": 100, "feed": "sip", "adjustment_mode": "all"},
        ],
        archive_root=archive,
        asset_registry=assets,
        alias_registry=aliases,
        dataset_version="overlap",
    )

    payload = run_reconcile_only(
        alpaca_archive_root=archive,
        stooq_root=stooq,
        report_root=tmp_path / "reports",
        symbols=["AAPL"],
        start="2026-01-01",
        end="2026-04-01",
    )

    counts = payload["reconciliation"]["rows_by_classification"]
    assert counts["ALPACA_ONLY"] == 1
    assert payload["reconciliation"]["matched_rows"] >= 1
    assert payload["reconciliation"]["relative_close_difference_count"] >= 1


def test_reconcile_only_dry_run_writes_nothing(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    archive = tmp_path / "archive"
    write_daily_archive(
        [{"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 4), "open": 1, "high": 1, "low": 1, "close": 2.0, "volume": 100, "feed": "sip", "adjustment_mode": "all"}],
        archive_root=archive,
        asset_registry=assets,
        alias_registry=aliases,
    )
    report_root = tmp_path / "reports"

    run_reconcile_only(
        alpaca_archive_root=archive,
        stooq_root=stooq,
        report_root=report_root,
        symbols=["AAPL"],
        start="2026-01-02",
        end="2026-01-02",
        dry_run=True,
    )

    assert not report_root.exists()


def test_reconcile_only_is_idempotent(tmp_path):
    stooq, assets, aliases = _fixture_inputs(tmp_path)
    archive = tmp_path / "archive"
    write_daily_archive(
        [{"symbol": "AAPL", "timestamp": datetime(2026, 1, 2, 4), "open": 1, "high": 1, "low": 1, "close": 2.0, "volume": 100, "feed": "sip", "adjustment_mode": "all"}],
        archive_root=archive,
        asset_registry=assets,
        alias_registry=aliases,
    )
    report_root = tmp_path / "reports"
    kwargs = dict(alpaca_archive_root=archive, stooq_root=stooq, report_root=report_root, symbols=["AAPL"], start="2026-01-02", end="2026-01-02")

    run_reconcile_only(**kwargs)
    before = (report_root / "smoke_reconciliation.json").read_text(encoding="utf-8")
    run_reconcile_only(**kwargs)
    after = (report_root / "smoke_reconciliation.json").read_text(encoding="utf-8")

    assert before == after


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stooq = tmp_path / "stooq"
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(80)]
    for symbol in ["AAPL", "SPY", "BRK-B", "MSFT", "ABCB"]:
        _write_stooq(stooq / f"{symbol}.parquet", symbol, dates)
    assets = tmp_path / "assets.csv"
    aliases = tmp_path / "aliases.csv"
    _write_csv(
        assets,
        ["asset_id", "canonical_symbol", "security_name", "security_type", "share_class", "exchange", "currency", "country", "cik", "sector", "industry", "valid_from", "valid_to", "is_active", "collection_universe_514", "registry_version"],
        [
            {"asset_id": f"asset_{symbol}", "canonical_symbol": symbol, "security_type": "UNKNOWN", "currency": "USD", "country": "US", "valid_from": "1900-01-01", "is_active": "true", "collection_universe_514": "true", "registry_version": "test"}
            for symbol in ["AAPL", "SPY", "BRK-B", "MSFT", "ABCB"]
        ],
    )
    _write_csv(
        aliases,
        ["asset_id", "provider", "provider_symbol", "valid_from", "valid_to", "is_primary", "mapping_reason", "source", "registry_version"],
        [
            {"asset_id": "asset_AAPL", "provider": "alpaca", "provider_symbol": "AAPL", "valid_from": "1900-01-01", "is_primary": "true", "mapping_reason": "identity", "source": "test", "registry_version": "test"},
            {"asset_id": "asset_SPY", "provider": "alpaca", "provider_symbol": "SPY", "valid_from": "1900-01-01", "is_primary": "true", "mapping_reason": "identity", "source": "test", "registry_version": "test"},
            {"asset_id": "asset_BRK-B", "provider": "alpaca", "provider_symbol": "BRK.B", "valid_from": "1900-01-01", "is_primary": "true", "mapping_reason": "configured_provider_map", "source": "test", "registry_version": "test"},
            {"asset_id": "asset_MSFT", "provider": "alpaca", "provider_symbol": "MSFT", "valid_from": "1900-01-01", "is_primary": "true", "mapping_reason": "identity", "source": "test", "registry_version": "test"},
            {"asset_id": "asset_ABCB", "provider": "alpaca", "provider_symbol": "ABCB", "valid_from": "1900-01-01", "is_primary": "true", "mapping_reason": "identity", "source": "test", "registry_version": "test"},
        ],
    )
    return stooq, assets, aliases


def _write_stooq(path: Path, symbol: str, timestamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {"symbol": symbol, "timestamp": value, "close": float(index + 1), "volume": float(index + 100), "source": "stooq_bulk"}
            for index, value in enumerate(timestamps)
        ]
    )
    pq.write_table(table, path)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _asset(asset_id: str, symbol: str):
    return type(
        "Asset",
        (),
        {
            "asset_id": asset_id,
            "canonical_symbol": symbol,
            "security_type": "UNKNOWN",
            "collection_universe_514": True,
            "is_active": True,
        },
    )()


def _alias(asset_id: str, provider_symbol: str, reason: str):
    return type(
        "Alias",
        (),
        {
            "asset_id": asset_id,
            "provider": "alpaca",
            "provider_symbol": provider_symbol,
            "is_primary": True,
            "mapping_reason": reason,
        },
    )()
