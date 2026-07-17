from __future__ import annotations

import inspect

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.reference.canonical_assets import build_registry_from_universe, write_registry_outputs
from core.research.ml.reference.daily_stock_spine import verify_and_register
from core.research.ml.stock_level.stock_level_artifact_io import iter_stock_level_artifact_batches


def _row(index, *, symbol=None, date=None):
    decision_date = date if date is not None else f"2024-01-{index + 2:02d}"
    return {
        "rebalance_date": decision_date,
        "symbol": symbol if symbol is not None else f"S{index:02d}",
        "decision_timestamp": f"{decision_date}T21:00:00Z" if decision_date else None,
        "feature_data_cutoff_timestamp": f"{decision_date}T20:00:00Z" if decision_date else None,
        "target_start_timestamp": f"{decision_date}T22:00:00Z" if decision_date else None,
        "label_start_timestamp": "2024-01-20T21:00:00Z" if decision_date else None,
        "label_end_timestamp": "2024-02-01T21:00:00Z",
        "label_available_timestamp": "2024-02-01T22:00:00Z",
        "target_horizon_trading_days": 10,
        "actual_forward_return_10d": str(index / 100),
        "actual_benchmark_return_10d": "0.01",
        "actual_market_residual_return_10d": "0.0",
        "unused_feature": float(index),
    }


def _write(path, rows, *, row_group_size=2):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)
    return path


def _environment(tmp_path, rows, enriched_rows=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    symbols = sorted({row["symbol"] for row in rows if row.get("symbol")})
    universe = tmp_path / "universe.txt"
    universe.write_text("\n".join(symbols) + "\n")
    assets, aliases, _ = build_registry_from_universe(universe)
    registry, alias_path = tmp_path / "registry.csv", tmp_path / "aliases.csv"
    write_registry_outputs(assets, aliases, asset_output=registry, alias_output=alias_path, parquet_output=None)
    return {
        "base": _write(tmp_path / "base.parquet", rows),
        "enriched": _write(tmp_path / "enriched.parquet", enriched_rows or rows),
        "registry": registry,
        "aliases": alias_path,
    }


def _verify(env, tmp_path, **kwargs):
    return verify_and_register(
        base_artifact=env["base"], enriched_artifact=env["enriched"],
        registry=env["registry"], aliases=env["aliases"],
        report_root=tmp_path / "reports", verify_only=True, **kwargs,
    )


def test_projected_columns_and_multiple_bounded_batches(tmp_path):
    path = _write(tmp_path / "rows.parquet", [_row(index) for index in range(5)])
    batches = list(iter_stock_level_artifact_batches(
        path, required_columns=["rebalance_date", "symbol"], batch_size=2,
    ))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert set(batches[0][0]) == {"rebalance_date", "symbol"}


def test_missing_column_empty_artifact_and_invalid_batch_size(tmp_path):
    path = _write(tmp_path / "rows.parquet", [_row(0)])
    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_stock_level_artifact_batches(path, required_columns=["missing"], batch_size=1))
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_stock_level_artifact_batches(path, required_columns=["symbol"], batch_size=0))
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"rebalance_date": pa.array([], type=pa.string()), "symbol": pa.array([], type=pa.string())}), empty)
    env = _environment(tmp_path / "empty-env", [_row(0)])
    env["base"] = empty
    env["enriched"] = empty
    assert "invalid_row:empty_artifact" in _verify(env, tmp_path / "empty-report")["blockers"]


def test_duplicate_missing_date_and_missing_symbol_detection(tmp_path):
    duplicate = [_row(0, symbol="S00"), _row(0, symbol="S00")]
    env = _environment(tmp_path / "duplicate", duplicate)
    assert "duplicate_economic_rows" in _verify(env, tmp_path / "duplicate")["blockers"]
    rows = [_row(0), _row(1)]
    rows[0]["rebalance_date"] = None
    rows[0]["decision_timestamp"] = None
    env = _environment(tmp_path / "missing-date", rows)
    assert "invalid_row:missing_date" in _verify(env, tmp_path / "missing-date")["blockers"]
    rows = [_row(0), _row(1)]
    rows[0]["symbol"] = None
    env = _environment(tmp_path / "missing-symbol", rows)
    assert "invalid_row:missing_symbol" in _verify(env, tmp_path / "missing-symbol")["blockers"]


def test_batch_size_does_not_change_logical_result_or_population_checksum(tmp_path):
    rows = [_row(index) for index in range(8)]
    env = _environment(tmp_path, rows)
    small = _verify(env, tmp_path, stream_batch_size=2, dry_run=True)
    large = _verify(env, tmp_path, stream_batch_size=20, dry_run=True)
    assert small["spine_dataset_id"] == large["spine_dataset_id"]
    assert small["base_artifact"]["row_population_checksum"] == large["base_artifact"]["row_population_checksum"]
    assert small["streaming_diagnostics"]["maximum_batch_row_count"] == 2
    assert large["streaming_diagnostics"]["maximum_batch_row_count"] == 8


def test_streaming_identity_matches_legacy_small_fixture(tmp_path):
    rows = [_row(index) for index in range(4)]
    env = _environment(tmp_path, rows)
    legacy = verify_and_register(
        base_artifact=env["base"], enriched_artifact=env["enriched"],
        registry=env["registry"], aliases=env["aliases"], dry_run=True,
    )
    streamed = _verify(env, tmp_path, dry_run=True, stream_batch_size=2)
    assert streamed["spine_dataset_id"] == legacy["spine_dataset_id"]
    assert streamed["price_feature_dataset_id"] == legacy["price_feature_dataset_id"]
    owner = __import__("core.research.ml.reference.daily_stock_spine", fromlist=["_augment_rows"])
    legacy_rows = owner._augment_rows(
        pq.read_table(env["base"]).to_pylist(), streamed["symbol_resolution"], streamed["lineage"],
    )
    assert streamed["base_artifact"]["row_population_checksum"] == owner._row_id_checksum(legacy_rows)


def test_population_match_mismatch_diagnostics_and_cleanup(tmp_path):
    rows = [_row(index) for index in range(4)]
    env = _environment(tmp_path / "match", rows)
    matched = _verify(env, tmp_path / "match", stream_temp_root=tmp_path / "temp")
    assert matched["alignment"]["same_row_id_set"]
    assert matched["streaming_diagnostics"]["temporary_resource"] == "cleaned_temporary_sqlite"
    assert not list((tmp_path / "temp").iterdir())
    env = _environment(tmp_path / "mismatch", rows, rows[:-1])
    mismatched = _verify(env, tmp_path / "mismatch")
    assert mismatched["alignment"]["base_only_count"] == 1
    assert "row_alignment_failures" in mismatched["blockers"]


def test_migrated_verify_only_path_has_no_whole_table_to_pylist(tmp_path):
    rows = [_row(index) for index in range(3)]
    env = _environment(tmp_path, rows)
    result = _verify(env, tmp_path, stream_batch_size=1)
    diagnostics = result["streaming_diagnostics"]
    assert diagnostics["whole_table_to_pylist_used"] is False
    assert diagnostics["projected_column_count"]["base"] < result["base_artifact"]["column_count"]
    source = inspect.getsource(__import__(
        "core.research.ml.reference.daily_stock_spine", fromlist=["_streaming_preflight"]
    )._streaming_preflight)
    assert "read_stock_level_artifact(" not in source
    assert "pq.read_table(" not in source


def test_controlled_stream_read_failure_is_blocking(monkeypatch, tmp_path):
    import core.research.ml.reference.daily_stock_spine as owner

    rows = [_row(index) for index in range(2)]
    env = _environment(tmp_path, rows)

    def fail(*args, **kwargs):
        raise ValueError("synthetic stream failure")
        yield

    monkeypatch.setattr(owner, "_iter_projected_batches", fail)
    result = _verify(env, tmp_path)
    assert result["status"] == "BLOCKED"
    assert any(blocker.startswith("stream_read_failure:") for blocker in result["blockers"])


@pytest.mark.parametrize("workers", [1, 3, 6])
def test_one_scan_per_artifact_and_worker_determinism(tmp_path, workers):
    rows = [_row(index % 8, symbol=f"S{index % 4:02d}", date=f"2024-01-{index % 8 + 2:02d}") for index in range(32)]
    env = _environment(tmp_path / f"w{workers}", rows)
    result = _verify(
        env,
        tmp_path / f"w{workers}",
        dry_run=True,
        stream_batch_size=5,
        max_workers=workers,
    )
    diagnostics = result["streaming_diagnostics"]
    assert diagnostics["source_scan_counts"] == {"base": 1, "enriched": 1}
    assert diagnostics["batches_processed"]["base_symbol_first_pass"] == 0
    assert diagnostics["worker_count"] == workers
    assert diagnostics["sqlite_insert_count"] == 64
    assert diagnostics["peak_working_set_memory_bytes"] > 0
    assert result["logical_output_checksum"]


def test_worker_counts_preserve_logical_identity_and_heartbeat(tmp_path, capsys):
    rows = [_row(index) for index in range(8)]
    env = _environment(tmp_path, rows)
    results = [
        _verify(
            env,
            tmp_path,
            dry_run=True,
            stream_batch_size=2,
            max_workers=workers,
            heartbeat_seconds=0,
        )
        for workers in (1, 3, 6)
    ]
    assert len({row["logical_output_checksum"] for row in results}) == 1
    assert len({row["spine_dataset_id"] for row in results}) == 1
    assert len({row["base_artifact"]["row_population_checksum"] for row in results}) == 1
    assert "daily_spine_preflight_heartbeat" in capsys.readouterr().out


def test_streaming_owner_does_not_materialize_complete_arrow_batches():
    source = inspect.getsource(__import__(
        "core.research.ml.reference.daily_stock_spine", fromlist=["_streaming_preflight"]
    )._streaming_preflight)
    assert ".to_pylist(" not in source
    assert "iter_stock_level_artifact_batches(" not in source


def test_malformed_timestamp_and_nonfinite_value_fail_closed(tmp_path):
    malformed = [_row(0)]
    malformed[0]["decision_timestamp"] = "not-a-timestamp"
    env = _environment(tmp_path / "malformed", malformed)
    result = _verify(env, tmp_path / "malformed")
    assert result["status"] == "BLOCKED"
    assert "temporal_violations" in result["blockers"]
    assert result["temporal_validation"]["violation_count"] >= 1

    nonfinite = [_row(0)]
    nonfinite[0]["actual_forward_return_10d"] = float("inf")
    env = _environment(tmp_path / "nonfinite", nonfinite)
    result = _verify(env, tmp_path / "nonfinite")
    assert result["status"] == "BLOCKED"
    assert "invalid_row:nonfinite_required_value" in result["blockers"]


def test_memory_limit_reduces_concurrency_instead_of_failing(tmp_path):
    rows = [_row(index) for index in range(4)]
    env = _environment(tmp_path, rows)
    result = _verify(
        env, tmp_path, dry_run=True, max_workers=6, memory_limit_mb=128,
    )
    diagnostics = result["streaming_diagnostics"]
    assert diagnostics["requested_worker_count"] == 6
    assert diagnostics["worker_count"] == 1
    assert diagnostics["serial_fallback"] is True
