from __future__ import annotations

import json

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from core.research.ml.reference.daily_stock_spine import (
    _temporal_order_valid,
    verify_and_register,
)
from core.research.ml.reference.daily_stock_spine_certification import (
    build_certification_identity,
    certification_path,
    load_ready_certification,
    publish_ready_certification,
)
from scripts.benchmark_daily_spine_preflight import _fixture


def _verify(fixture, certification_root, **updates):
    values = {
        "base_artifact": fixture["base"],
        "enriched_artifact": fixture["enriched"],
        "registry": fixture["registry"],
        "aliases": fixture["aliases"],
        "registry_manifest": fixture["registry_manifest"],
        "daily_archive_manifest": fixture["archive_manifest"],
        "expected_config": fixture["config"],
        "report_root": fixture["root"] / "reports",
        "verify_only": True,
        "stream_batch_size": 32,
        "max_workers": 3,
        "certification_root": certification_root,
        "selector_run_id": "A",
    }
    values.update(updates)
    return verify_and_register(**values)


def test_ready_full_validation_publishes_and_reuses_certification(tmp_path):
    fixture = _fixture(tmp_path / "fixture", 100)
    root = tmp_path / "certifications"
    full = _verify(fixture, root)
    assert full["status"] == "READY"
    assert full["certification_cache_hit"] is False
    assert full["certification_source"]
    cached = _verify(fixture, root, dry_run=True)
    assert cached["status"] == "READY"
    assert cached["certification_cache_hit"] is True
    assert cached["certification_id"] == full["certification_id"]
    assert cached["logical_output_checksum"] == full["logical_output_checksum"]
    assert cached["streaming_diagnostics"]["source_scan_counts"] == {"base": 0, "enriched": 0}
    assert cached["streaming_diagnostics"]["rows_scanned"] == 0
    assert cached["reuse_validation_elapsed_seconds"] >= 0


@pytest.mark.parametrize(
    "parent",
    ["base", "enriched", "registry", "aliases", "archive_manifest", "config"],
)
def test_parent_change_invalidates_certification_identity(tmp_path, parent):
    fixture = _fixture(tmp_path / "fixture", 20)
    manifest = json.loads(fixture["registry_manifest"].read_text())
    before = build_certification_identity(
        base_path=fixture["base"], enriched_path=fixture["enriched"],
        registry_path=fixture["registry"],
        registry_content_hash=manifest.get("dataset_id", ""),
        aliases_path=fixture["aliases"], archive_manifest=fixture["archive_manifest"],
        expected_config=fixture["config"],
    )
    path = fixture[parent]
    if path.suffix == ".parquet":
        table = pq.read_table(path).replace_schema_metadata({b"changed": b"1"})
        pq.write_table(table, path, row_group_size=10)
    elif parent == "archive_manifest":
        payload = json.loads(path.read_text())
        payload["dataset_logical_partition_hash"] = "changed-archive"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif parent == "config":
        path.write_text(path.read_text() + "  certification_test_value: changed\n", encoding="utf-8")
    else:
        path.write_text(path.read_text() + "# changed\n", encoding="utf-8")
    after = build_certification_identity(
        base_path=fixture["base"], enriched_path=fixture["enriched"],
        registry_path=fixture["registry"],
        registry_content_hash=manifest.get("dataset_id", ""),
        aliases_path=fixture["aliases"], archive_manifest=fixture["archive_manifest"],
        expected_config=fixture["config"],
    )
    assert after["certification_id"] != before["certification_id"]


def test_corrupt_certification_falls_back_to_full_validation(tmp_path):
    fixture = _fixture(tmp_path / "fixture", 30)
    root = tmp_path / "certifications"
    full = _verify(fixture, root)
    path = certification_path(root, full["certification_id"])
    path.write_text("{corrupt", encoding="utf-8")
    repeated = _verify(fixture, root, dry_run=True)
    assert repeated["status"] == "READY"
    assert repeated["certification_cache_hit"] is False
    assert repeated["certification_miss_reason"] == "corrupt"
    assert repeated["streaming_diagnostics"]["source_scan_counts"] == {"base": 1, "enriched": 1}


def test_blocked_result_cannot_be_certified(tmp_path):
    with pytest.raises(ValueError, match="Only READY"):
        publish_ready_certification(
            tmp_path, {"certification_id": "A" * 64, "parents": {}},
            {"status": "BLOCKED"},
        )


def test_registry_content_hash_and_validator_version_invalidate_identity(tmp_path, monkeypatch):
    import core.research.ml.reference.daily_stock_spine_certification as owner

    fixture = _fixture(tmp_path / "fixture", 20)
    values = {
        "base_path": fixture["base"], "enriched_path": fixture["enriched"],
        "registry_path": fixture["registry"],
        "aliases_path": fixture["aliases"],
        "archive_manifest": fixture["archive_manifest"],
        "expected_config": fixture["config"],
    }
    first = build_certification_identity(**values, registry_content_hash="A" * 64)
    changed_registry = build_certification_identity(**values, registry_content_hash="B" * 64)
    assert changed_registry["certification_id"] != first["certification_id"]
    monkeypatch.setattr(owner, "VALIDATOR_VERSION", "changed-validator")
    changed_validator = build_certification_identity(**values, registry_content_hash="A" * 64)
    assert changed_validator["certification_id"] != first["certification_id"]


def test_incomplete_and_blocked_certifications_are_not_reused(tmp_path):
    identity = {"certification_id": "A" * 64, "parents": {}}
    path = certification_path(tmp_path, identity["certification_id"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "READY"}), encoding="utf-8")
    assert load_ready_certification(tmp_path, identity)[0] is None


def test_cross_run_manifest_binding_reuses_stable_certification(tmp_path):
    fixture = _fixture(tmp_path / "fixture", 40)
    root = tmp_path / "certifications"
    full = _verify(fixture, root)
    manifest_b = fixture["root"] / "run=B" / "registry_manifest.json"
    manifest_b.parent.mkdir()
    payload = json.loads(fixture["registry_manifest"].read_text())
    payload["run_id"] = "B"
    manifest_b.write_text(json.dumps(payload), encoding="utf-8")

    cached = _verify(
        fixture, root, registry_manifest=manifest_b, selector_run_id="B", dry_run=True,
    )
    assert cached["status"] == "READY"
    assert cached["certification_cache_hit"] is True
    assert cached["certification_id"] == full["certification_id"]
    assert cached["registry_run_binding"]["manifest_run_id"] == "B"
    assert cached["streaming_diagnostics"]["source_scan_counts"] == {"base": 0, "enriched": 0}
    assert cached["streaming_diagnostics"]["rows_scanned"] == 0


def test_run_binding_rejects_wrong_stage_2_manifest(tmp_path):
    fixture = _fixture(tmp_path / "fixture", 20)
    result = _verify(fixture, tmp_path / "certifications", selector_run_id="B")
    assert result["status"] == "BLOCKED"
    assert "registry_run_binding_run_id_mismatch" in result["blockers"]


def test_label_onset_before_decision_remains_a_temporal_blocker(tmp_path):
    fixture = _fixture(tmp_path / "fixture", 20)
    for path in (fixture["base"], fixture["enriched"]):
        rows = pq.read_table(path).to_pylist()
        rows[0]["label_start_timestamp"] = rows[0]["target_start_timestamp"]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=10)
    result = _verify(fixture, tmp_path / "certifications", dry_run=True)
    assert result["status"] == "BLOCKED"
    assert "temporal_violations" in result["blockers"]
    assert any(
        violation["right"] == "label_start_timestamp"
        and violation["semantics"] == "strict_instant"
        for violation in result["temporal_validation"]["violations"]
    )


@pytest.mark.parametrize(
    ("decision", "label_start"),
    [
        ("2024-03-08T21:05:00Z", "2024-03-11T13:30:00-04:00"),  # DST weekend
        ("2024-07-03T17:05:00Z", "2024-07-05T13:30:00Z"),  # early close + holiday
        ("2024-11-01T20:05:00-04:00", "2024-11-04T09:30:00-05:00"),  # DST offset
    ],
)
def test_timezone_aware_next_tradable_session_label_onsets_are_strictly_future(
    decision, label_start,
):
    assert _temporal_order_valid(decision, label_start, "strict_instant")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("2024-01-02T20:05:00", "2024-01-03T14:30:00Z"),
        ("malformed", "2024-01-03T14:30:00Z"),
        ("2024-01-02T20:05:00Z", "not-a-timestamp"),
    ],
)
def test_naive_or_malformed_temporal_events_are_rejected(left, right):
    assert not _temporal_order_valid(left, right, "strict_instant")
