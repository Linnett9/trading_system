from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.artifact_lineage import build_artifact_link
from core.research.ml.dataset_build_manifest import (
    PERMITTED_DIAGNOSTIC,
    PERMITTED_PROMOTION,
    STATUS_CURRENT,
    STATUS_LEGACY_NO_MANIFEST,
    STATUS_STALE,
    dataset_manifest_path,
    manifest_hash,
    write_manifest,
)
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_dataset_lineage import logical_manifest_checksum
from core.research.ml.stock_level.frozen_selector_lineage_guard import (
    check_frozen_selector_dataset_lineage,
    enforce_frozen_selector_dataset_lineage,
)
from core.research.ml.stock_level.selector_dataset import (
    BASELINE_CONTRACT_VERSION,
    DETERMINISTIC_SIGNAL_COLUMNS,
    SELECTOR_DATASET_CONTRACT_VERSION,
    SELECTOR_DATASET_MANIFEST_VERSION,
    build_frozen_selector_dataset,
    frozen_selector_configuration_hash,
    frozen_selector_dataset_build_manifest,
    frozen_selector_target_contract_version,
)


def test_current_frozen_dataset_combined_report_is_promotion_eligible(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)

    first = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
    )
    second = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="PROMOTION_GRADE",
    )

    assert first == second
    assert first["generic_dataset_status"] == STATUS_CURRENT
    assert first["selector_lineage_status"] == "VERIFIED"
    assert first["artifact_lineage_status"] == "NOT_APPLICABLE"
    assert first["permitted_use"] == PERMITTED_PROMOTION
    assert first["promotion_eligible"] is True
    assert first["blocking_reasons"] == []


def test_changed_price_parent_blocks_promotion_without_rebuild(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)
    source = tmp_path / "source_rows.parquet"
    before = _mtimes(root, source)
    source.write_bytes(b"changed-price-parent")

    report = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
    )

    assert report["generic_dataset_status"] == STATUS_STALE
    assert "SOURCE_PATH_HASH_CHANGED" in report["blocking_reasons"]
    assert report["promotion_eligible"] is False
    assert report["changed_parents"]
    assert report["dataset_rebuilt"] is False
    assert report["dataset_modified"] is False
    assert _mtimes(root, source, include_missing=True) == before | {source: source.stat().st_mtime_ns}


@pytest.mark.parametrize(
    ("expected_key", "reason"),
    [
        ("target_contract_version", "TARGET_CONTRACT_CHANGED"),
        ("universe_authority_version", "UNIVERSE_AUTHORITY_CHANGED"),
        ("identity_authority_version", "IDENTITY_AUTHORITY_CHANGED"),
        ("feature_code_version", "FEATURE_CODE_CHANGED"),
    ],
)
def test_changed_selector_parent_or_code_identity_blocks_promotion(
    tmp_path: Path,
    expected_key: str,
    reason: str,
):
    root = _write_guard_dataset(tmp_path)

    report = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
        expected_parents={expected_key: "changed"},
    )

    assert report["generic_dataset_status"] == STATUS_STALE
    assert reason in report["blocking_reasons"]
    assert report["promotion_eligible"] is False


def test_dirty_tree_source_is_unverified_and_not_promotion_grade(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)
    generic_path = dataset_manifest_path(root / "rows.parquet")
    payload = json.loads(generic_path.read_text(encoding="utf-8"))
    payload["source_control"]["dirty_worktree"] = True
    payload["dirty_tree"] = True
    payload["manifest_hash"] = manifest_hash(payload)
    write_manifest(generic_path, payload)

    report = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
    )

    assert report["generic_dataset_status"] == "UNVERIFIED"
    assert report["permitted_use"] == PERMITTED_DIAGNOSTIC
    assert "DIRTY_TREE_BUILD" in report["blocking_reasons"]
    assert report["promotion_eligible"] is False


def test_missing_generic_manifest_is_diagnostic_only_and_promotion_fails_closed(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)
    dataset_manifest_path(root / "rows.parquet").unlink()

    diagnostic = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="diagnostic",
    )
    promotion = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
    )

    assert diagnostic["generic_dataset_status"] == STATUS_LEGACY_NO_MANIFEST
    assert diagnostic["permitted_use"] == PERMITTED_DIAGNOSTIC
    assert diagnostic["use_authorized"] is True
    assert diagnostic["diagnostic_label_required"] is True
    assert promotion["use_authorized"] is False
    assert promotion["promotion_eligible"] is False


def test_existing_selector_lineage_failure_blocks_combined_report(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["publication_status"] = "incomplete"
    payload["logical_checksum"] = logical_manifest_checksum(payload)
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _refresh_generic_output_hash(root, manifest_path)

    report = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
    )

    assert report["generic_dataset_status"] == STATUS_CURRENT
    assert report["selector_lineage_status"] == "UNVERIFIED"
    assert "SELECTOR_DATASET_LINEAGE_UNVERIFIED" in report["blocking_reasons"]
    assert report["promotion_eligible"] is False


def test_existing_artifact_lineage_failure_blocks_combined_report(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)
    artifact_path = tmp_path / "artifact.json"
    link = build_artifact_link(
        artifact_kind="BOUNDED_SELECTOR_PREDICTION",
        artifact_id="selector-2024-01-02",
        artifact_checksum="A" * 64,
        experiment_run_id="run",
        canonical_model_or_policy_id="ridge",
        model_or_policy_entry_hash="B" * 64,
        dataset_id="dataset",
        dataset_checksum="C" * 64,
        row_population_hash="D" * 64,
        feature_schema_hash="E" * 64,
        target_contract_hash="F" * 64,
        decision_start="2024-01-02T20:05:00+00:00",
        decision_end="2024-01-02T20:05:00+00:00",
        training_cutoff="2024-01-01T20:05:00+00:00",
        maximum_label_available_timestamp="2024-01-02T20:05:00+00:00",
        strict_oos_claim=True,
        strict_oos_evidence={
            "prediction_quality_passed": False,
            "row_population_verified": True,
        },
        completion_status="complete",
    )
    artifact_path.write_text(json.dumps({"artifact_link": link}), encoding="utf-8")

    report = check_frozen_selector_dataset_lineage(
        dataset_root=root,
        intended_use="promotion-grade",
        artifact_manifest_path=artifact_path,
    )

    assert report["artifact_lineage_status"] == "CONFLICTING_EVIDENCE"
    assert "PREDICTION_QUALITY_FAILED" in report["blocking_reasons"]
    assert report["promotion_eligible"] is False


def test_guard_fails_closed_before_bounded_promotion_training(tmp_path: Path):
    root = _write_guard_dataset(tmp_path)
    dataset_manifest_path(root / "rows.parquet").unlink()

    with pytest.raises(RuntimeError, match="DATASET_MANIFEST_MISSING"):
        enforce_frozen_selector_dataset_lineage(
            dataset_root=root,
            intended_use="promotion-grade",
        )


def test_producer_emits_generic_dataset_build_manifest(tmp_path: Path):
    paths = _build_real_frozen_dataset(tmp_path)
    generic_path = dataset_manifest_path(paths.rows)
    generic = json.loads(generic_path.read_text(encoding="utf-8"))

    assert generic["manifest_schema_version"] == "dataset_build_manifest_v1"
    assert generic["dataset_type"] == "frozen_selector_dataset"
    assert generic["canonical_price_authority_version"]
    assert generic["universe_authority_version"]
    assert generic["identity_authority_version"]
    assert generic["target_contract_version"]
    assert generic["feature_code_version"]
    assert generic["label_code_version"]
    assert generic["configuration_hash"]
    assert generic["row_count"] == 2
    assert generic["key_count"] == 2
    assert generic["symbol_entity_count"] == 2
    assert generic["earliest_decision_timestamp"]
    assert generic["latest_knowledge_cutoff"]
    assert generic["output_hashes"][0]["path"] == str(paths.rows)
    assert generic["parent_artifact_ids"]


def _write_guard_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    rows = _selector_rows("2024-01-02")
    scores = [
        {
            "row_id": row["row_id"],
            "asset_id": row["asset_id"],
            "decision_timestamp": row["decision_timestamp"],
            "baseline_contract_version": BASELINE_CONTRACT_VERSION,
            **{name: float(index + 1) for name in DETERMINISTIC_SIGNAL_COLUMNS},
        }
        for index, row in enumerate(rows)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "rows.parquet")
    pq.write_table(pa.Table.from_pylist(scores), root / "baseline_scores.parquet")
    (root / "feature_schema.json").write_text(json.dumps({"features": list(DETERMINISTIC_SIGNAL_COLUMNS)}), encoding="utf-8")
    (root / "target_schema.json").write_text(json.dumps({"target": "actual_forward_return_10d"}), encoding="utf-8")
    (root / "candidate_schema.json").write_text(json.dumps({"candidates": ["ridge"]}), encoding="utf-8")
    (root / "quality_report.json").write_text(json.dumps({"row_count": len(rows)}), encoding="utf-8")
    (root / "checksums.json").write_text(json.dumps({}), encoding="utf-8")
    source = tmp_path / "source_rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    source_digest = _sha(source)
    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts",
        "forward_return_10d",
        role="selector",
    )
    target_version = frozen_selector_target_contract_version(
        target.canonical_id,
        target.entry.entry_hash,
        "stock_level_target_provenance_v2",
    )
    parents = {
        "daily_spine_identity": "spine",
        "daily_spine_version": "spine-v1",
        "daily_spine_checksum": "A" * 64,
        "daily_feature_identity": "features",
        "daily_feature_version": "features-v1",
        "daily_feature_checksum": "B" * 64,
        "symbol_registry_identity": "registry",
        "symbol_registry_version": "registry-v1",
        "symbol_registry_checksum": "C" * 64,
        "source_price_artifact_identities": ["prices"],
        "point_in_time_feature_store_identities": ["features"],
    }
    config_hash = frozen_selector_configuration_hash(
        source_digest=source_digest,
        config_hash=None,
        parents=parents,
        selected_symbols=["AAA", "BBB"],
        selected_dates=["2024-01-02"],
        copy_source_rows=True,
    )
    checksums = {
        "rows.parquet": _sha(root / "rows.parquet"),
        "baseline_scores.parquet": _sha(root / "baseline_scores.parquet"),
        "feature_schema.json": _sha(root / "feature_schema.json"),
        "target_schema.json": _sha(root / "target_schema.json"),
        "candidate_schema.json": _sha(root / "candidate_schema.json"),
    }
    manifest = {
        "manifest_schema_version": SELECTOR_DATASET_MANIFEST_VERSION,
        "dataset_id": SELECTOR_DATASET_CONTRACT_VERSION + "_bounded",
        "dataset_path": str(root / "rows.parquet"),
        "dataset_checksum": checksums["rows.parquet"],
        "source_path": str(source),
        "source_sha256": source_digest,
        "row_population_checksum": canonical_hash(sorted(row["row_id"] for row in rows)),
        "row_count": len(rows),
        "symbol_count": 2,
        "daily_stock_spine_identity": parents["daily_spine_identity"],
        "daily_stock_spine_version": parents["daily_spine_version"],
        "daily_stock_spine_checksum": parents["daily_spine_checksum"],
        "daily_feature_store_identity": parents["daily_feature_identity"],
        "daily_feature_store_version": parents["daily_feature_version"],
        "daily_feature_store_checksum": parents["daily_feature_checksum"],
        "symbol_registry_identity": parents["symbol_registry_identity"],
        "symbol_registry_version": parents["symbol_registry_version"],
        "symbol_registry_checksum": parents["symbol_registry_checksum"],
        "source_price_artifact_identities": parents["source_price_artifact_identities"],
        "point_in_time_feature_store_identities": parents["point_in_time_feature_store_identities"],
        "target_contract": target.canonical_id,
        "target_contract_checksum": target.entry.entry_hash,
        "economic_target_id": target.canonical_id,
        "target_registry_entry_checksum": target.entry.entry_hash,
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "target_contract_version": target_version,
        "feature_schema_checksum": checksums["feature_schema.json"],
        "target_schema_checksum": checksums["target_schema.json"],
        "builder_run_identity": "builder",
        "git_commit": "abc",
        "dataset_build_configuration_hash": config_hash,
        "checksums": checksums,
        "frozen_preflight": {"status": "READY", "blockers": []},
        "publication_status": "complete",
        "validation_status": "VERIFIED",
    }
    manifest["logical_checksum"] = logical_manifest_checksum(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    parent_paths = _write_parent_manifests(tmp_path, parents)
    generic = frozen_selector_dataset_build_manifest(
        selector_manifest=manifest,
        source_path=source,
        output_root=root,
        final_root=root,
        rows=rows,
        parents=parents,
        source_manifest_paths=parent_paths,
        target_contract_version=target_version,
        target_registry_entry_checksum=target.entry.entry_hash,
        source_control={
            "contract_version": "source_worktree_provenance_v1",
            "git_commit": "abc",
            "dirty_worktree": False,
        },
    )
    write_manifest(dataset_manifest_path(root / "rows.parquet"), generic)
    return root


def _build_real_frozen_dataset(tmp_path: Path):
    source = tmp_path / "source.parquet"
    market_root = tmp_path / "market"
    decision_day = "2024-06-01"
    rows = [
        {key: value for key, value in row.items() if key != "row_id"}
        for row in _selector_rows(decision_day)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    for symbol in ("AAA", "BBB"):
        symbol_root = market_root / f"symbol={symbol}"
        symbol_root.mkdir(parents=True)
        bars = []
        start = date(2024, 1, 1)
        for index in range(160):
            bars.append({
                "session_date": (start + timedelta(days=index)).isoformat(),
                "model_close": 100.0 + index,
                "raw_volume": 1_000_000 + index,
            })
        pq.write_table(pa.Table.from_pylist(bars), symbol_root / "bars.parquet")
    parents = _write_real_parent_bundle(tmp_path, source, rows)
    base_artifact = tmp_path / "base.parquet"
    pq.write_table(pa.Table.from_pylist(rows), base_artifact)
    base_manifest = tmp_path / "base_manifest.json"
    enriched_manifest = tmp_path / "enriched_manifest.json"
    base_payload = _ready_selector_parent_manifest(
        daily_spine_logical_checksum=parents["daily_spine_logical_checksum"],
    )
    base_payload["dataset_id"] = "base"
    enriched_payload = dict(base_payload)
    enriched_payload["dataset_id"] = "enriched"
    base_manifest.write_text(json.dumps(base_payload), encoding="utf-8")
    enriched_manifest.write_text(json.dumps(enriched_payload), encoding="utf-8")
    return build_frozen_selector_dataset(
        source,
        market_root,
        tmp_path / "out",
        decision_dates=[decision_day],
        daily_spine_manifest_path=parents["spine_manifest_path"],
        daily_feature_manifest_path=parents["feature_manifest_path"],
        symbol_registry_manifest_path=parents["registry_manifest_path"],
        base_artifact_path=base_artifact,
        base_manifest_path=base_manifest,
        enriched_manifest_path=enriched_manifest,
        source_control={
            "contract_version": "source_worktree_provenance_v1",
            "git_commit": "abc",
            "dirty_worktree": False,
        },
    )


def _selector_rows(day: str) -> list[dict[str, object]]:
    rows = []
    for symbol in ("AAA", "BBB"):
        rows.append({
            "row_id": f"{day}-{symbol}",
            "asset_id": symbol.lower(),
            "canonical_symbol": symbol,
            "symbol": symbol,
            "rebalance_date": day,
            "decision_session_date": day,
            "decision_timestamp": f"{day}T20:05:00+00:00",
            "feature_data_cutoff_timestamp": f"{day}T20:00:00+00:00",
            "selector_eligible": True,
            "economic_target_id": "forward_return_10d",
            "target_provenance_contract_version": "stock_level_target_provenance_v2",
            "target_status": "realized",
            "target_is_trainable": True,
            "target_is_mature": True,
            "target_is_realised": True,
            "target_resolution_classification": "MATURED_VALID",
            "actual_forward_return_10d": 0.01,
            "actual_benchmark_return_10d": 0.005,
            "actual_market_residual_return_10d": 0.005,
            "target_start_timestamp": f"{day}T20:05:00+00:00",
            "label_start_timestamp": f"{day}T20:05:00+00:00",
            "label_end_timestamp": f"{day}T20:05:00+00:00",
            "label_available_timestamp": f"{day}T20:05:00+00:00",
            "benchmark_target_start_timestamp": f"{day}T20:05:00+00:00",
            "benchmark_label_start_timestamp": f"{day}T20:05:00+00:00",
            "benchmark_label_end_timestamp": f"{day}T20:05:00+00:00",
            "benchmark_label_available_timestamp": f"{day}T20:05:00+00:00",
        })
    return rows


def _write_parent_manifests(tmp_path: Path, parents: dict[str, object]) -> tuple[Path, Path, Path]:
    paths = []
    for name, dataset_id, checksum in (
        ("spine", parents["daily_spine_identity"], parents["daily_spine_checksum"]),
        ("features", parents["daily_feature_identity"], parents["daily_feature_checksum"]),
        ("registry", parents["symbol_registry_identity"], parents["symbol_registry_checksum"]),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"dataset_id": dataset_id, "checksum": checksum}), encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _write_real_parent_bundle(tmp_path: Path, source: Path, rows: list[dict[str, object]]) -> dict[str, Path | str]:
    spine_file = tmp_path / "spine.parquet"
    pq.write_table(
        pa.Table.from_pylist([
            {"asset_id": row["asset_id"], "session_date": row["decision_session_date"]}
            for row in rows
        ]),
        spine_file,
    )
    registry_csv = tmp_path / "registry.csv"
    registry_csv.write_text("asset_id,canonical_symbol\naaa,AAA\nbbb,BBB\n", encoding="utf-8")
    daily_spine_logical_checksum = "S" * 64
    spine = {
        "status": "READY",
        "dataset_type": "canonical_daily_stock_spine",
        "dataset_id": "spine",
        "schema_version": "spine-v1",
        "spine_artifact_path": str(spine_file),
        "spine_artifact_checksum": _sha(spine_file),
        "logical_checksum": daily_spine_logical_checksum,
        "source_price_artifact_identities": ["prices"],
    }
    feature = {
        "status": "READY",
        "dataset_type": "daily_price_features",
        "dataset_id": "features",
        "schema_version": "features-v1",
        "source_dataset_ids": ["spine"],
        "source_checksums": {str(source): _sha(source)},
    }
    registry = {
        "status": "READY",
        "dataset_type": "canonical_asset_registry_audit",
        "dataset_id": "registry",
        "symbol_registry_version": "registry-v1",
        "registry_path": str(registry_csv),
        "registry_content_checksum": _sha(registry_csv),
    }
    spine_path = tmp_path / "spine_manifest.json"
    feature_path = tmp_path / "feature_manifest.json"
    registry_path = tmp_path / "registry_manifest.json"
    spine_path.write_text(json.dumps(spine), encoding="utf-8")
    feature_path.write_text(json.dumps(feature), encoding="utf-8")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return {
        "spine_manifest_path": spine_path,
        "feature_manifest_path": feature_path,
        "registry_manifest_path": registry_path,
        "daily_spine_logical_checksum": daily_spine_logical_checksum,
    }


def _ready_selector_parent_manifest(*, daily_spine_logical_checksum: str) -> dict[str, object]:
    return {
        "status": "READY",
        "canonical_daily_dataset_version": "canonical-v1",
        "canonical_daily_logical_checksum": "C" * 64,
        "asset_registry_version": "registry-v1",
        "asset_registry_checksum": "R" * 64,
        "daily_spine_identity": "spine",
        "daily_spine_logical_checksum": daily_spine_logical_checksum,
        "calendar_version": "calendar-v1",
        "decision_timing_contract": "decision-v1",
        "configuration_hash": "config",
        "git_commit": "abc",
        "builder_contract_version": "builder-v1",
        "feature_schema_version": "features-v1",
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
    }


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _mtimes(root: Path, source: Path, *, include_missing: bool = False) -> dict[Path, int]:
    paths = [
        root / "rows.parquet",
        root / "baseline_scores.parquet",
        root / "manifest.json",
        dataset_manifest_path(root / "rows.parquet"),
        source,
    ]
    return {
        path: path.stat().st_mtime_ns
        for path in paths
        if include_missing or path.exists()
    }


def _refresh_generic_output_hash(root: Path, changed_output: Path) -> None:
    generic_path = dataset_manifest_path(root / "rows.parquet")
    generic = json.loads(generic_path.read_text(encoding="utf-8"))
    for row in generic["output_hashes"]:
        if row["path"] == str(changed_output):
            row["sha256"] = _sha(changed_output)
            row["size_bytes"] = changed_output.stat().st_size
    generic["manifest_hash"] = manifest_hash(generic)
    write_manifest(generic_path, generic)
