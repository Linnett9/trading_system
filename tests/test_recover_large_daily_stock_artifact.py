from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.recover_large_daily_stock_artifact import (
    FINAL_BASE_NAME,
    audit_partitions_for_run,
    finalize_base_from_partitions,
    generate_enriched_artifact,
    inspect_partition,
    run_recovery,
)
from core.research.ml.stock_level.stock_level_artifact_io import file_sha256, read_stock_level_artifact


def test_valid_partitions_are_reused(tmp_path):
    env = _env(tmp_path)
    result = run_recovery(config_path=env["config"], run_dir=env["run"], audit_partitions=True, dry_run=True)
    assert result["audit"]["valid_partition_count"] == 2
    assert result["audit"]["recovery_decision"] == "ALL PARTITIONS REUSABLE"


def test_missing_partitions_are_detected(tmp_path):
    env = _env(tmp_path, missing=("MSFT",))
    result = run_recovery(config_path=env["config"], run_dir=env["run"], audit_partitions=True, dry_run=True)
    assert result["audit"]["missing_partition_count"] == 1


def test_empty_partitions_are_rejected(tmp_path):
    env = _env(tmp_path)
    (env["partition_root"] / "MSFT.json").write_text("", encoding="utf-8")
    row = inspect_partition(env["partition_root"] / "MSFT.json", symbol="MSFT")
    assert row["status"] == "EMPTY"


def test_unreadable_partitions_are_reported(tmp_path):
    env = _env(tmp_path)
    (env["partition_root"] / "MSFT.json").write_text("{not-json", encoding="utf-8")
    row = inspect_partition(env["partition_root"] / "MSFT.json", symbol="MSFT")
    assert row["status"] == "UNREADABLE"


def test_schema_mismatches_are_detected(tmp_path):
    env = _env(tmp_path)
    _write_partition(env["partition_root"] / "MSFT.json", "MSFT", extra={"extra_col": 1})
    result = run_recovery(config_path=env["config"], run_dir=env["run"], audit_partitions=True, dry_run=True)
    assert result["audit"]["status_counts"]["SCHEMA_MISMATCH"] == 1


def test_duplicate_economic_rows_are_detected(tmp_path):
    env = _env(tmp_path)
    _write_partition(env["partition_root"] / "MSFT.json", "MSFT", duplicate=True)
    row = inspect_partition(env["partition_root"] / "MSFT.json", symbol="MSFT")
    assert row["status"] == "DUPLICATE_ROWS"


def test_only_invalid_missing_symbols_are_recompute_candidates(tmp_path):
    env = _env(tmp_path, missing=("MSFT",))
    result = run_recovery(config_path=env["config"], run_dir=env["run"], audit_partitions=True, dry_run=True)
    assert result["audit"]["symbols_to_recompute"] == [{"symbol": "MSFT", "reason": "MISSING"}]


def test_full_recomputation_is_not_triggered_by_default(tmp_path):
    env = _env(tmp_path)
    result = run_recovery(config_path=env["config"], run_dir=env["run"], audit_partitions=True, dry_run=True)
    assert result["full_recomputation_triggered"] is False


def test_finalization_can_run_without_symbol_redispatch(tmp_path):
    env = _env(tmp_path)
    result = run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    assert result["finalization"]["status"] == "COMPLETED"
    assert (env["run"] / FINAL_BASE_NAME).exists()


def test_enriched_only_requires_valid_base_artifact(tmp_path):
    env = _env(tmp_path)
    result = run_recovery(config_path=env["config"], run_dir=env["run"], generate_enriched_only=True)
    assert result["enriched"]["status"] == "BLOCKED"


def test_final_base_write_is_atomic(tmp_path):
    env = _env(tmp_path)
    run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    assert (env["run"] / FINAL_BASE_NAME).exists()
    assert not list(env["run"].glob(f".{FINAL_BASE_NAME}.*.tmp"))


def test_failed_temporary_write_does_not_create_final_path(tmp_path, monkeypatch):
    env = _env(tmp_path)
    import scripts.recover_large_daily_stock_artifact as rec

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rec, "_write_recovered_parquet", fail)
    inventory = audit_partitions_for_run(partition_root=env["partition_root"], expected_symbols=["AAPL", "MSFT"])
    with pytest.raises(RuntimeError):
        finalize_base_from_partitions(config=_config_payload(env["run"], env["universe"]), run_dir=env["run"], inventory=inventory, report_root=tmp_path / "reports")
    assert not (env["run"] / FINAL_BASE_NAME).exists()


def test_existing_valid_final_artifact_is_not_overwritten_accidentally(tmp_path):
    env = _env(tmp_path)
    run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    before = file_sha256(env["run"] / FINAL_BASE_NAME)
    second = run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    assert second["finalization"]["status"] == "EXISTING_VALID"
    assert file_sha256(env["run"] / FINAL_BASE_NAME) == before


def test_status_is_not_marked_complete_before_validation(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _write_manifest(env["run"], stock_status="interrupted")
    import scripts.recover_large_daily_stock_artifact as rec

    monkeypatch.setattr(rec, "_validate_written_artifact", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(RuntimeError):
        run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    manifest = json.loads((env["run"] / "stock_alpha_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"][0]["status"] == "interrupted"


def test_recovery_provenance_is_appended_not_destructive(tmp_path):
    env = _env(tmp_path)
    _write_manifest(env["run"], stock_status="interrupted")
    run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    manifest = json.loads((env["run"] / "stock_alpha_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["original_interruption_preserved"] is True
    assert manifest["recovery_history"]


def test_dry_run_writes_nothing(tmp_path):
    env = _env(tmp_path)
    report_root = tmp_path / "reports"
    run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True, report_root=report_root, dry_run=True)
    assert not (env["run"] / FINAL_BASE_NAME).exists()
    assert not report_root.exists()


def test_explicit_run_directory_is_required(tmp_path):
    env = _env(tmp_path)
    with pytest.raises(ValueError):
        run_recovery(config_path=env["config"], run_dir=None)  # type: ignore[arg-type]


def test_no_legacy_artifact_fallback_occurs(tmp_path):
    env = _env(tmp_path, missing=("AAPL", "MSFT"))
    legacy = tmp_path / "stock_level_prediction_artifacts.parquet"
    legacy.write_text("legacy", encoding="utf-8")
    result = run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True, dry_run=True)
    assert result["finalization"]["partitions_to_recompute"] == 2


def test_base_enriched_targets_remain_identical(tmp_path):
    env = _env(tmp_path)
    run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    # Create a simple aligned enriched artifact to exercise existing validation path.
    rows = read_stock_level_artifact(env["run"] / FINAL_BASE_NAME)
    from core.research.ml.stock_level.stock_level_artifact_io import write_stock_level_artifact

    write_stock_level_artifact(env["run"] / "stock_level_prediction_artifacts_enriched.parquet", rows, fieldnames=list(rows[0]), config=_config_payload(env["run"], env["universe"]))
    result = generate_enriched_artifact(config=_config_payload(env["run"], env["universe"]), run_dir=env["run"], report_root=tmp_path / "reports")
    assert result["alignment"]["aligned"] is True


def test_repeated_recovery_runs_are_idempotent(tmp_path):
    env = _env(tmp_path)
    first = run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    second = run_recovery(config_path=env["config"], run_dir=env["run"], finalize_from_partitions=True)
    assert first["finalization"]["identity"]["sha256"] == second["finalization"]["identity"]["sha256"]


def _env(tmp_path: Path, *, missing=()):
    run = tmp_path / "run"
    partition_root = run / "stock_artifact_symbol_partitions"
    partition_root.mkdir(parents=True)
    universe = tmp_path / "universe.yaml"
    universe.write_text("symbols:\n- AAPL\n- MSFT\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps(_config_payload(run, universe)), encoding="utf-8")
    for symbol in ("AAPL", "MSFT"):
        if symbol not in missing:
            _write_partition(partition_root / f"{symbol}.json", symbol)
    return {"run": run, "partition_root": partition_root, "universe": universe, "config": config}


def _config_payload(run: Path, universe: Path):
    return {
        "ml": {
            "output_dir": str(run),
            "stock_level_artifact_format": "parquet",
            "stock_level_parquet_compression": "zstd",
            "stock_alpha_artifact_universe_paths": [str(universe)],
            "stooq_parquet_dir": str(run / "prices"),
            "stock_alpha_feature_n_jobs": 1,
        }
    }


def _write_partition(path: Path, symbol: str, *, extra=None, duplicate=False):
    rows = [_row(symbol, "2024-01-02"), _row(symbol, "2024-01-03")]
    if duplicate:
        rows[1] = dict(rows[0])
    if extra:
        rows = [row | extra for row in rows]
    payload = {
        "schema_version": "stock_level_symbol_partition_v1",
        "symbol": symbol,
        "row_count": len(rows),
        "expected_date_count": 2,
        "rows_sha256": "test",
        "rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(symbol: str, date: str):
    return {
        "rebalance_date": date,
        "symbol": symbol,
        "decision_timestamp": f"{date}T20:05:00Z",
        "feature_data_cutoff_timestamp": f"{date}T20:00:00Z",
        "actual_forward_return_10d": 0.1,
        "actual_benchmark_return_10d": 0.01,
        "actual_market_residual_return_10d": 0.09,
        "actual_rank_normalized_forward_return_10d": "",
        "actual_top_decile_label_10d": "",
        "target_horizon_trading_days": 10,
        "target_provenance_contract_version": "stock_level_target_provenance_v1",
        "benchmark_symbol": "SPY",
        "target_status": "realized",
    }


def _write_manifest(run: Path, *, stock_status: str):
    payload = {
        "stages": [
            {"name": "stock_artifact", "status": stock_status, "output_paths": {}},
            {"name": "alpha_features", "status": "pending", "output_paths": {}},
        ]
    }
    (run / "stock_alpha_run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
