from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from core.research.ml.reference.canonical_assets import (
    build_registry_from_universe,
    write_registry_outputs,
)
from core.research.ml.reference.daily_stock_spine import (
    _canonical_timestamp_text,
    _columnar_rows,
    _temporal_order_valid,
    verify_and_register,
)


def _row(symbol: str, date: str, *, horizon: int = 10) -> dict:
    return {
        "rebalance_date": date,
        "symbol": symbol,
        "decision_timestamp": f"{date}T20:05:00Z",
        "feature_data_cutoff_timestamp": f"{date}T19:00:00Z",
        "target_start_timestamp": date,
        "label_start_timestamp": "2024-02-01T21:00:00Z",
        "label_end_timestamp": "2024-02-15T21:00:00Z",
        "label_available_timestamp": "2024-02-15T22:00:00Z",
        "target_horizon_trading_days": horizon,
        "actual_forward_return_10d": "0.1",
        "actual_benchmark_return_10d": "0.01",
        "actual_market_residual_return_10d": "0.09",
        "source_dataset_hash": "source",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
    }


def _env(tmp_path: Path, base_rows: list[dict], enriched_rows: list[dict]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    symbols = sorted({row["symbol"] for row in base_rows + enriched_rows})
    universe = tmp_path / "universe.txt"
    universe.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    assets, aliases, _ = build_registry_from_universe(universe)
    registry, alias_path = tmp_path / "registry.csv", tmp_path / "aliases.csv"
    write_registry_outputs(
        assets, aliases, asset_output=registry, alias_output=alias_path,
        parquet_output=None,
    )
    base = tmp_path / "base.parquet"
    enriched = tmp_path / "enriched.parquet"
    pq.write_table(pa.Table.from_pylist(base_rows), base, row_group_size=2)
    pq.write_table(pa.Table.from_pylist(enriched_rows), enriched, row_group_size=2)
    return {"base": base, "enriched": enriched, "registry": registry, "aliases": alias_path}


def _verify(env, tmp_path: Path):
    return verify_and_register(
        base_artifact=env["base"], enriched_artifact=env["enriched"],
        registry=env["registry"], aliases=env["aliases"],
        report_root=tmp_path / "reports", verify_only=True, dry_run=True,
        stream_batch_size=2,
    )


def test_identical_population_different_physical_order_passes_alignment(tmp_path):
    rows = [_row("AAPL", "2024-01-02"), _row("MSFT", "2024-01-03")]
    result = _verify(_env(tmp_path, rows, list(reversed(rows))), tmp_path)
    assert result["alignment"]["same_alignment_key_set"] is True
    assert result["alignment"]["common_key_count"] == 2
    assert "row_alignment_failures" not in result["blockers"]


def test_date_major_versus_symbol_major_ordering_passes(tmp_path):
    base = [
        _row("AAPL", "2024-01-02"), _row("MSFT", "2024-01-02"),
        _row("AAPL", "2024-01-03"), _row("MSFT", "2024-01-03"),
    ]
    enriched = [
        _row("AAPL", "2024-01-02"), _row("AAPL", "2024-01-03"),
        _row("MSFT", "2024-01-02"), _row("MSFT", "2024-01-03"),
    ]
    result = _verify(_env(tmp_path, base, enriched), tmp_path)
    assert result["alignment"]["same_alignment_key_set"] is True
    assert result["base_artifact"]["row_population_checksum"] == result["enriched_artifact"]["row_population_checksum"]


def test_missing_base_or_enriched_key_blocks(tmp_path):
    rows = [_row("AAPL", "2024-01-02"), _row("MSFT", "2024-01-03")]
    base_missing = _verify(_env(tmp_path / "base-missing", rows[:1], rows), tmp_path / "base-missing")
    assert base_missing["alignment"]["enriched_only_count"] == 1
    assert "row_alignment_failures" in base_missing["blockers"]
    enriched_missing = _verify(_env(tmp_path / "enriched-missing", rows, rows[:1]), tmp_path / "enriched-missing")
    assert enriched_missing["alignment"]["base_only_count"] == 1
    assert "row_alignment_failures" in enriched_missing["blockers"]


def test_duplicate_keys_block_and_enrichment_only_columns_do_not_change_key(tmp_path):
    duplicate = [_row("AAPL", "2024-01-02"), _row("AAPL", "2024-01-02")]
    duplicated = _verify(_env(tmp_path / "dup", duplicate, duplicate), tmp_path / "dup")
    assert duplicated["alignment"]["duplicate_base_count"] == 1
    assert "row_alignment_failures" in duplicated["blockers"]

    base = [_row("AAPL", "2024-01-02")]
    enriched = [{**base[0], "momentum_250d": 0.42}]
    enriched_only = _verify(_env(tmp_path / "feature", base, enriched), tmp_path / "feature")
    assert enriched_only["alignment"]["same_alignment_key_set"] is True
    assert "row_alignment_failures" not in enriched_only["blockers"]


def test_different_target_identity_changes_alignment_key(tmp_path):
    base = [_row("AAPL", "2024-01-02", horizon=10)]
    enriched = [_row("AAPL", "2024-01-02", horizon=5)]
    result = _verify(_env(tmp_path, base, enriched), tmp_path)
    assert result["alignment"]["base_only_count"] == 1
    assert result["alignment"]["enriched_only_count"] == 1
    assert "row_alignment_failures" in result["blockers"]


def test_arrow_and_string_utc_forms_normalize_identically(tmp_path):
    value = datetime(2024, 1, 2, 20, 5, tzinfo=timezone.utc)
    assert _canonical_timestamp_text(value) == "2024-01-02T20:05:00Z"
    assert _canonical_timestamp_text("2024-01-02 20:05:00+00:00") == "2024-01-02T20:05:00Z"
    assert _canonical_timestamp_text("2024-01-02T20:05:00.000000Z") == "2024-01-02T20:05:00Z"
    batch = pa.RecordBatch.from_arrays([pa.array([value], type=pa.timestamp("us", tz="UTC"))], ["decision_timestamp"])
    assert list(_columnar_rows(batch))[0]["decision_timestamp"] == "2024-01-02T20:05:00Z"


def test_temporal_contract_price_anchor_and_future_label_rules():
    assert _temporal_order_valid("2024-01-02T20:05:00Z", "2024-01-02", "price_anchor")
    assert _temporal_order_valid("2024-01-02T20:05:00Z", "2024-01-03T21:00:00+00:00", "strict_instant")
    assert not _temporal_order_valid("2024-01-02T20:05:00Z", "2024-01-02T00:00:00Z", "strict_instant")
    assert not _temporal_order_valid("2024-01-04T21:00:00Z", "2024-01-03T21:00:00Z", "instant")
    assert not _temporal_order_valid("2024-01-04T21:00:00Z", "2024-01-03T22:00:00Z", "instant")


def test_missing_future_observation_evidence_blocks(tmp_path):
    row = _row("AAPL", "2024-01-02")
    row.pop("label_start_timestamp")
    result = _verify(_env(tmp_path, [row], [row]), tmp_path)
    assert "missing_future_label_observation" in result["blockers"]
