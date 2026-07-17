from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from core.research.ml.reference.canonical_assets import build_registry_from_universe, write_registry_outputs
from core.research.ml.reference.daily_stock_spine import verify_and_register
from core.research.ml.stock_level.stock_level_artifact_io import file_sha256


def test_explicit_source_path_takes_priority(tmp_path):
    env = _env(tmp_path)
    other_base = _write_artifact(tmp_path / "other_base.parquet", [_row("MSFT")])
    manifest = _manifest(tmp_path, base=other_base, enriched=env["enriched"], stock_status="completed", alpha_status="completed")
    result = verify_and_register(
        base_artifact=env["base"],
        enriched_artifact=env["enriched"],
        registry=env["registry"],
        aliases=env["aliases"],
        expected_run_manifest=manifest,
        dry_run=True,
        output_root=tmp_path / "out",
    )
    assert result["selected_sources"]["base_artifact"] == str(env["base"])
    assert result["selected_sources"]["base_source"] == "explicit"


def test_no_legacy_fallback_occurs(tmp_path):
    env = _env(tmp_path)
    result = verify_and_register(registry=env["registry"], aliases=env["aliases"], dry_run=True, output_root=tmp_path / "out")
    assert "base_artifact_path_not_supplied" in result["blockers"]
    assert "enriched_artifact_path_not_supplied" in result["blockers"]


def test_missing_base_artifact_produces_blocked(tmp_path):
    env = _env(tmp_path)
    result = _verify(env, base=tmp_path / "missing.parquet", dry_run=True)
    assert result["status"] == "BLOCKED"
    assert any(str(item).startswith("missing_base_artifact") for item in result["blockers"])


def test_missing_enriched_artifact_produces_blocked(tmp_path):
    env = _env(tmp_path)
    result = _verify(env, enriched=tmp_path / "missing.parquet", dry_run=True)
    assert result["status"] == "BLOCKED"
    assert any(str(item).startswith("missing_enriched_artifact") for item in result["blockers"])


def test_incomplete_run_status_produces_blocked(tmp_path):
    env = _env(tmp_path)
    manifest = _manifest(tmp_path, base=env["base"], enriched=env["enriched"], stock_status="interrupted", alpha_status="pending")
    result = _verify(env, manifest=manifest, dry_run=True)
    assert result["status"] == "BLOCKED"
    assert any("stock_artifact_not_completed" in blocker for blocker in result["blockers"])


def test_all_source_symbols_must_resolve(tmp_path):
    env = _env(tmp_path, symbols=("AAPL",))
    bad = _write_artifact(tmp_path / "bad.parquet", [_row("ZZZZ")])
    result = _verify(env, base=bad, enriched=bad, dry_run=True)
    assert "unresolved_symbols" in result["blockers"]


def test_ambiguous_symbols_block_registration(tmp_path):
    env = _env(tmp_path)
    with env["aliases"].open("a", encoding="utf-8") as handle:
        handle.write("asset_other,canonical,AAPL,1900-01-01,,true,test,test,test\n")
    result = _verify(env, dry_run=True)
    assert "ambiguous_symbols" in result["blockers"]


def test_row_ids_are_independent_of_row_ordering(tmp_path):
    env = _env(tmp_path, base_rows=[_row("AAPL"), _row("MSFT")], enriched_rows=[_row("AAPL"), _row("MSFT")], symbols=("AAPL", "MSFT"))
    left = _verify(env, dry_run=True)
    env2 = _env(tmp_path / "b", base_rows=[_row("MSFT"), _row("AAPL")], enriched_rows=[_row("MSFT"), _row("AAPL")], symbols=("AAPL", "MSFT"))
    right = _verify(env2, dry_run=True)
    assert left["spine_dataset_id"] == right["spine_dataset_id"]


def test_base_and_enriched_rows_can_be_in_different_physical_orders(tmp_path):
    env = _env(tmp_path, base_rows=[_row("AAPL"), _row("MSFT")], enriched_rows=[_row("MSFT"), _row("AAPL")], symbols=("AAPL", "MSFT"))
    result = _verify(env, dry_run=True)
    assert result["alignment"]["same_row_id_set"] is True


def test_base_only_rows_are_detected(tmp_path):
    env = _env(tmp_path, base_rows=[_row("AAPL"), _row("MSFT")], enriched_rows=[_row("AAPL")], symbols=("AAPL", "MSFT"))
    result = _verify(env, dry_run=True)
    assert result["alignment"]["base_only_count"] == 1


def test_enriched_only_rows_are_detected(tmp_path):
    env = _env(tmp_path, base_rows=[_row("AAPL")], enriched_rows=[_row("AAPL"), _row("MSFT")], symbols=("AAPL", "MSFT"))
    result = _verify(env, dry_run=True)
    assert result["alignment"]["enriched_only_count"] == 1


def test_duplicate_economic_rows_are_rejected(tmp_path):
    env = _env(tmp_path, base_rows=[_row("AAPL"), _row("AAPL")], enriched_rows=[_row("AAPL"), _row("AAPL")])
    result = _verify(env, dry_run=True)
    assert "duplicate_economic_rows" in result["blockers"]


def test_target_mismatches_are_detected(tmp_path):
    env = _env(tmp_path, enriched_rows=[_row("AAPL", target="0.2")])
    result = _verify(env, dry_run=True)
    assert result["target_alignment"]["target_mismatch_count"] == 1


def test_benchmark_mismatches_are_detected(tmp_path):
    env = _env(tmp_path, enriched_rows=[_row("AAPL", benchmark="0.2")])
    result = _verify(env, dry_run=True)
    assert result["target_alignment"]["benchmark_mismatch_count"] == 1


def test_temporal_violations_are_detected(tmp_path):
    env = _env(tmp_path, base_rows=[_row("AAPL", feature_cutoff="2024-01-03T21:00:00Z")])
    result = _verify(env, dry_run=True)
    assert result["temporal_validation"]["violation_count"] == 1


def test_unknown_enriched_columns_are_reported(tmp_path):
    env = _env(tmp_path, enriched_rows=[_row("AAPL") | {"mystery_feature": "1"}])
    result = _verify(env, dry_run=True)
    assert "mystery_feature" in result["unknown_columns"]


def test_dry_run_writes_nothing(tmp_path):
    env = _env(tmp_path)
    out = tmp_path / "out"
    report = tmp_path / "reports"
    verify_and_register(base_artifact=env["base"], enriched_artifact=env["enriched"], registry=env["registry"], aliases=env["aliases"], output_root=out, report_root=report, dry_run=True)
    assert not out.exists()
    assert not report.exists()


def test_verify_only_does_not_materialize_spine(tmp_path):
    env = _env(tmp_path)
    out = tmp_path / "out"
    result = verify_and_register(base_artifact=env["base"], enriched_artifact=env["enriched"], registry=env["registry"], aliases=env["aliases"], output_root=out, report_root=tmp_path / "reports", verify_only=True)
    assert result["status"] == "READY"
    assert not out.exists()
    assert (tmp_path / "reports").exists()


def test_original_source_checksums_remain_unchanged(tmp_path):
    env = _env(tmp_path)
    before = (file_sha256(env["base"]), file_sha256(env["enriched"]))
    _verify(env, dry_run=False)
    assert (file_sha256(env["base"]), file_sha256(env["enriched"])) == before


def test_successful_registration_is_deterministic_across_repeated_runs(tmp_path):
    env = _env(tmp_path)
    first = _verify(env, dry_run=False)
    second = _verify(env, dry_run=False)
    assert first["spine_dataset_id"] == second["spine_dataset_id"]
    assert first["price_feature_dataset_id"] == second["price_feature_dataset_id"]


def test_parent_gate_rejects_missing_archive_manifest(tmp_path):
    env = _env(tmp_path)
    registry_manifest = _registry_manifest(tmp_path, env)
    result = verify_and_register(base_artifact=env["base"], enriched_artifact=env["enriched"], registry=env["registry"], aliases=env["aliases"], registry_manifest=registry_manifest, daily_archive_manifest=tmp_path / "missing.json", dry_run=True)
    assert "daily_archive_manifest_missing" in result["blockers"]


def test_parent_gate_rejects_legacy_archive_and_wrong_registry_parent(tmp_path):
    env = _env(tmp_path); registry_manifest = _registry_manifest(tmp_path, env)
    archive = tmp_path / "archive.json"; archive.write_text(json.dumps({"status": "COMPLETE", "row_count": 10, "symbol_count": 514, "dataset_root": "legacy/path"}))
    config = tmp_path / "config.yaml"; config.write_text("ml:\n  historical_data_provider: canonical_daily_v2\n  stooq_parquet_dir: authoritative/path\n")
    result = verify_and_register(base_artifact=env["base"], enriched_artifact=env["enriched"], registry=env["registry"], aliases=env["aliases"], registry_manifest=registry_manifest, daily_archive_manifest=archive, expected_config=config, dry_run=True)
    assert "daily_archive_source_mismatch" in result["blockers"]
    payload = json.loads(registry_manifest.read_text()); payload["registry_content_checksum"] = "wrong"; registry_manifest.write_text(json.dumps(payload))
    result = verify_and_register(base_artifact=env["base"], enriched_artifact=env["enriched"], registry=env["registry"], aliases=env["aliases"], registry_manifest=registry_manifest, daily_archive_manifest=archive, expected_config=config, dry_run=True)
    assert "registry_manifest_source_mismatch" in result["blockers"]


def _env(tmp_path: Path, *, base_rows=None, enriched_rows=None, symbols=("AAPL",)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    universe = tmp_path / "universe.txt"
    universe.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    assets, aliases, _ = build_registry_from_universe(universe)
    registry = tmp_path / "registry.csv"
    alias_path = tmp_path / "aliases.csv"
    write_registry_outputs(assets, aliases, asset_output=registry, alias_output=alias_path, parquet_output=None)
    base = _write_artifact(tmp_path / "base.parquet", base_rows or [_row("AAPL")])
    enriched = _write_artifact(tmp_path / "enriched.parquet", enriched_rows or [_row("AAPL") | {"momentum_250d": "0.1"}])
    return {"base": base, "enriched": enriched, "registry": registry, "aliases": alias_path}


def _registry_manifest(tmp_path, env):
    path = tmp_path / "registry_manifest.json"
    path.write_text(json.dumps({"status": "READY", "validation_status": "VERIFIED", "dataset_id": "registry", "symbol_registry_version": "v1", "registry_path": str(env["registry"]), "registry_content_checksum": file_sha256(env["registry"])}))
    return path


def _verify(env, *, base=None, enriched=None, manifest=None, dry_run=True):
    return verify_and_register(
        base_artifact=base or env["base"],
        enriched_artifact=enriched or env["enriched"],
        registry=env["registry"],
        aliases=env["aliases"],
        expected_run_manifest=manifest,
        output_root=env["base"].parent / "out",
        feature_output_root=env["base"].parent / "features",
        report_root=env["base"].parent / "reports",
        dry_run=dry_run,
    )


def _write_artifact(path: Path, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return path


def _row(symbol: str, *, target="0.1", benchmark="0.01", feature_cutoff="2024-01-02T20:00:00Z"):
    return {
        "rebalance_date": "2024-01-02",
        "symbol": symbol,
        "decision_timestamp": "2024-01-02T21:00:00Z",
        "feature_data_cutoff_timestamp": feature_cutoff,
        "target_start_timestamp": "2024-01-03T21:00:00Z",
        "label_start_timestamp": "2024-01-03T21:00:00Z",
        "label_end_timestamp": "2024-01-16T21:00:00Z",
        "label_available_timestamp": "2024-01-16T22:00:00Z",
        "target_horizon_trading_days": 10,
        "actual_forward_return_10d": target,
        "actual_benchmark_return_10d": benchmark,
        "actual_market_residual_return_10d": "0.09",
        "benchmark_symbol": "AAPL",
        "source_dataset_hash": "source1",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
    }


def _manifest(tmp_path: Path, *, base: Path, enriched: Path, stock_status: str, alpha_status: str):
    path = tmp_path / "manifest.json"
    payload = {
        "stages": [
            {"name": "stock_artifact", "status": stock_status, "output_paths": {"parquet_path": str(base)}},
            {"name": "alpha_features", "status": alpha_status, "output_paths": {"enriched_parquet_path": str(enriched)}},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
