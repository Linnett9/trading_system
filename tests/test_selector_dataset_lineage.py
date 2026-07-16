import hashlib
import json

import pytest

from core.research.ml.selector_dataset_lineage import (
    assess_lineage_repair, logical_manifest_checksum,
    publish_metadata_republication, verify_dataset_lineage_manifest,
)
from core.research.ml.stock_level.selector_dataset import (
    _dataset_population_identity, _validate_parent_manifests,
    _validate_rows_against_parents, canonical_dataset_run_identity,
)


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _parents(tmp_path, source, *, spine_status="READY", registry_hash=True):
    registry_csv = tmp_path / "registry.csv"; registry_csv.write_text("asset_id,canonical_symbol\na,A\nb,B\n")
    pa = pytest.importorskip("pyarrow"); pq = pytest.importorskip("pyarrow.parquet")
    spine_file = tmp_path / "spine.parquet"; pq.write_table(pa.Table.from_pylist([{"asset_id": "a", "session_date": "2024-01-01"}, {"asset_id": "b", "session_date": "2024-01-01"}]), spine_file)
    spine = {"status": spine_status, "dataset_type": "canonical_daily_stock_spine", "dataset_id": "spine", "schema_version": "spine-v1", "spine_artifact_path": str(spine_file), "spine_artifact_checksum": _sha(spine_file), "source_price_artifact_identities": ["prices"], "point_in_time_feature_store_identities": ["features"]}
    feature = {"status": "READY", "dataset_type": "daily_price_features", "dataset_id": "features", "schema_version": "features-v1", "source_dataset_ids": ["spine"], "source_checksums": {str(source): _sha(source)}}
    registry = {"status": "READY", "dataset_type": "canonical_asset_registry_audit", "dataset_id": "registry", "symbol_registry_version": "symbols-v1", "registry_path": str(registry_csv), "registry_content_checksum": _sha(registry_csv) if registry_hash else "stale"}
    sp = tmp_path / "spine.json"; fp = tmp_path / "features.json"; rp = tmp_path / "symbols.json"; sp.write_text(json.dumps(spine)); fp.write_text(json.dumps(feature)); rp.write_text(json.dumps(registry))
    return sp, fp, rp, spine, registry


def test_missing_unknown_and_stale_parent_identities_fail_closed(tmp_path):
    source = tmp_path / "source"; source.write_bytes(b"source")
    sp, fp, rp, _, _ = _parents(tmp_path, source, spine_status="BLOCKED")
    with pytest.raises(ValueError, match="spine"): _validate_parent_manifests(source, sp, fp, rp)
    sp, fp, rp, _, _ = _parents(tmp_path, source, registry_hash=False)
    with pytest.raises(ValueError, match="checksum"): _validate_parent_manifests(source, sp, fp, rp)
    with pytest.raises(ValueError, match="required"):
        from core.research.ml.stock_level.selector_dataset import build_frozen_selector_dataset
        build_frozen_selector_dataset(source, tmp_path, tmp_path / "out")


def test_filename_only_identity_and_ambiguous_parent_are_rejected(tmp_path):
    root = tmp_path / "dataset"; root.mkdir(); rows = root / "rows.parquet"; rows.write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps({"checksums": {"rows.parquet": _sha(rows)}, "source_path": "source", "source_sha256": "x"}))
    fake = tmp_path / "canonical_daily_stock_spine-looks-valid.json"; fake.write_text("{}")
    registry = tmp_path / "symbols.json"; registry.write_text("{}")
    result = assess_lineage_repair(dataset_root=root, daily_spine_manifest=fake, symbol_registry_manifest=registry)
    assert result["classification"] == "AUTHORITATIVE_DATASET_REBUILD_REQUIRED"
    assert "AUTHORITATIVE_DAILY_SPINE_NOT_READY" in result["blocking_reasons"]


def test_safe_metadata_republication_is_atomic_and_old_manifest_immutable(tmp_path):
    root = tmp_path / "dataset"; root.mkdir(); rows = root / "rows.parquet"; rows.write_bytes(b"dataset")
    source = tmp_path / "source"; source.write_bytes(b"source")
    old = root / "manifest.json"; old.write_text(json.dumps({"source_path": str(source), "source_sha256": _sha(source), "checksums": {"rows.parquet": _sha(rows)}}))
    _, _, _, spine, registry = _parents(tmp_path, source)
    spine.update({"source_artifact_path": str(source), "source_artifact_checksum": _sha(source)})
    before = old.read_bytes(); new = root / "manifests" / "v2.json"
    assert publish_metadata_republication(old_manifest=old, new_manifest=new, spine=spine, registry=registry)["status"] == "published"
    assert old.read_bytes() == before and json.loads(new.read_text())["dataset_bytes_changed"] is False
    assert publish_metadata_republication(old_manifest=old, new_manifest=new, spine=spine, registry=registry)["status"] == "skipped_identical"


def test_unsafe_republication_rejects_changed_bytes_and_parent(tmp_path):
    root = tmp_path / "dataset"; root.mkdir(); rows = root / "rows.parquet"; rows.write_bytes(b"changed")
    source = tmp_path / "source"; source.write_bytes(b"source")
    old = root / "manifest.json"; old.write_text(json.dumps({"source_path": str(source), "source_sha256": _sha(source), "checksums": {"rows.parquet": "old"}}))
    _, _, _, spine, registry = _parents(tmp_path, source)
    spine.update({"source_artifact_path": str(source), "source_artifact_checksum": _sha(source)})
    with pytest.raises(ValueError, match="bytes changed"): publish_metadata_republication(old_manifest=old, new_manifest=root / "v2", spine=spine, registry=registry)


def test_population_validation_rejects_duplicate_rows_missing_spine_row_and_unresolved_symbol(tmp_path):
    pa = pytest.importorskip("pyarrow"); pq = pytest.importorskip("pyarrow.parquet")
    rows = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([{"row_id": "1", "asset_id": "a", "canonical_symbol": "A", "decision_session_date": "2024-01-01"}, {"row_id": "2", "asset_id": "a", "canonical_symbol": "A", "decision_session_date": "2024-01-01"}]), rows)
    with pytest.raises(ValueError, match="Duplicate"): _dataset_population_identity(rows)
    pq.write_table(pa.Table.from_pylist([{"row_id": "1", "asset_id": "unknown", "canonical_symbol": "X", "decision_session_date": "2024-01-01"}]), rows)
    registry = tmp_path / "r.csv"; registry.write_text("asset_id,canonical_symbol\na,A\n")
    spine = tmp_path / "spine.parquet"; pq.write_table(pa.Table.from_pylist([{"asset_id": "a", "session_date": "2024-01-01"}]), spine)
    with pytest.raises(ValueError, match="Unresolved"): _validate_rows_against_parents(rows, {"registry_path": registry, "spine_path": spine})


def test_population_validation_rejects_noncanonical_order(tmp_path):
    pa = pytest.importorskip("pyarrow"); pq = pytest.importorskip("pyarrow.parquet")
    rows = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"row_id": "b", "asset_id": "b", "canonical_symbol": "B", "decision_session_date": "2024-01-01"},
        {"row_id": "a", "asset_id": "a", "canonical_symbol": "A", "decision_session_date": "2024-01-01"},
    ]), rows)
    with pytest.raises(ValueError, match="Noncanonical"):
        _dataset_population_identity(rows)


def test_deterministic_refreeze_identity_is_independent_of_timestamps(tmp_path):
    parents = {"daily_spine_identity": "s", "symbol_registry_identity": "r"}
    assert canonical_dataset_run_identity("d", "c", parents) == canonical_dataset_run_identity("d", "c", parents)
    payload = {"dataset_id": "d", "creation_timestamp": "first"}
    first = logical_manifest_checksum(payload)
    payload["creation_timestamp"] = "second"
    assert logical_manifest_checksum(payload) == first


def test_metadata_only_verification_and_incomplete_publication_rejection(tmp_path):
    from core.research.ml.registries import RegistryResolver, load_registry_bundle

    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", "forward_return_10d", role="selector"
    )
    root = tmp_path / "dataset"; root.mkdir()
    for name, content in (
        ("rows.parquet", b"rows"),
        ("feature_schema.json", b"features"),
        ("target_schema.json", b"targets"),
    ):
        (root / name).write_bytes(content)
    checksums = {name: _sha(root / name) for name in (
        "rows.parquet", "feature_schema.json", "target_schema.json"
    )}
    payload = {
        "manifest_schema_version": "authoritative_frozen_selector_dataset_v2",
        "dataset_id": "dataset", "symbol_registry_identity": "registry",
        "symbol_registry_checksum": "R", "daily_stock_spine_identity": "spine",
        "daily_stock_spine_checksum": "S", "daily_feature_store_identity": "features",
        "daily_feature_store_checksum": "F", "target_contract": "forward_return_10d",
        "target_contract_checksum": target.entry.entry_hash, "row_population_checksum": "P",
        "feature_schema_checksum": checksums["feature_schema.json"],
        "target_schema_checksum": checksums["target_schema.json"],
        "builder_run_identity": "B", "git_commit": "G", "checksums": checksums,
        "publication_status": "complete", "validation_status": "VERIFIED",
        "creation_timestamp": "first",
    }
    payload["logical_checksum"] = logical_manifest_checksum(payload)
    manifest = root / "manifest.json"; manifest.write_text(json.dumps(payload))
    assert verify_dataset_lineage_manifest(manifest)["status"] == "VERIFIED"
    payload["publication_status"] = "incomplete"
    payload["logical_checksum"] = logical_manifest_checksum(payload)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Incomplete atomic"):
        verify_dataset_lineage_manifest(manifest)


def test_metadata_verification_rejects_malformed_logical_and_target_mismatch(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    manifest = root / "manifest.json"; manifest.write_text("{")
    with pytest.raises(ValueError, match="Malformed"):
        verify_dataset_lineage_manifest(manifest)

    for name in ("rows.parquet", "feature_schema.json", "target_schema.json"):
        (root / name).write_bytes(name.encode())
    checksums = {name: _sha(root / name) for name in (
        "rows.parquet", "feature_schema.json", "target_schema.json"
    )}
    payload = {
        "manifest_schema_version": "authoritative_frozen_selector_dataset_v2",
        "dataset_id": "dataset", "symbol_registry_identity": "registry",
        "symbol_registry_checksum": "R", "daily_stock_spine_identity": "spine",
        "daily_stock_spine_checksum": "S", "daily_feature_store_identity": "features",
        "daily_feature_store_checksum": "F", "target_contract": "forward_return_10d",
        "target_contract_checksum": "wrong", "row_population_checksum": "P",
        "feature_schema_checksum": checksums["feature_schema.json"],
        "target_schema_checksum": checksums["target_schema.json"],
        "builder_run_identity": "B", "git_commit": "G", "checksums": checksums,
        "publication_status": "complete", "validation_status": "VERIFIED",
    }
    payload["logical_checksum"] = logical_manifest_checksum(payload)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="target-contract mismatch"):
        verify_dataset_lineage_manifest(manifest)
    payload["logical_checksum"] = "wrong"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="logical checksum mismatch"):
        verify_dataset_lineage_manifest(manifest)
