from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.services.ml_lineage_commands import run_dataset_lineage_check
from core.research.ml.dataset_build_manifest import (
    PERMITTED_DIAGNOSTIC,
    PERMITTED_PROMOTION,
    PERMITTED_RESEARCH,
    STATUS_CURRENT,
    STATUS_LEGACY_NO_MANIFEST,
    STATUS_MISSING_PARENT,
    STATUS_STALE,
    STATUS_UNVERIFIED,
    build_dataset_build_manifest,
    check_dataset_lineage,
    dataset_manifest_path,
    file_sha256,
    manifest_hash,
    serialize_manifest,
    write_manifest,
)


def test_current_manifest_serializes_and_checks_deterministically(tmp_path: Path):
    dataset, manifest_path, parent, manifest = _write_current_manifest(tmp_path)
    before = {
        dataset: dataset.stat().st_mtime_ns,
        parent: parent.stat().st_mtime_ns,
        manifest_path: manifest_path.stat().st_mtime_ns,
    }

    first = check_dataset_lineage(
        dataset_path=dataset,
        expected=_expectation(manifest),
        intended_use="promotion-grade",
    )
    second = check_dataset_lineage(
        dataset_path=dataset,
        expected=_expectation(manifest),
        intended_use="PROMOTION_GRADE",
    )

    assert first == second
    assert first["status"] == STATUS_CURRENT
    assert first["permitted_use"] == PERMITTED_PROMOTION
    assert first["use_authorized"] is True
    assert first["dataset_rebuilt"] is False
    assert first["dataset_modified"] is False
    assert first["source_modified"] is False
    assert file_sha256(dataset) == file_sha256(dataset)
    assert serialize_manifest(manifest) == serialize_manifest(json.loads(manifest_path.read_text()))
    assert {path: path.stat().st_mtime_ns for path in before} == before


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("source_hash", "SOURCE_PATH_HASH_CHANGED"),
        ("feature_code", "FEATURE_CODE_CHANGED"),
        ("label_code", "LABEL_CODE_CHANGED"),
        ("universe_authority", "UNIVERSE_AUTHORITY_CHANGED"),
        ("identity_authority", "IDENTITY_AUTHORITY_CHANGED"),
        ("schema", "SCHEMA_VERSION_CHANGED"),
        ("configuration", "CONFIGURATION_CHANGED"),
    ],
)
def test_parent_or_contract_changes_make_dataset_stale(
    tmp_path: Path,
    mutation: str,
    reason: str,
):
    dataset, _, parent, manifest = _write_current_manifest(tmp_path)
    expected = _expectation(manifest)
    if mutation == "source_hash":
        parent.write_text("parent_id,value\nsource,changed\n", encoding="utf-8")
    elif mutation == "feature_code":
        expected["feature_code_version"] = "feature-code-v2"
    elif mutation == "label_code":
        expected["label_code_version"] = "label-code-v2"
    elif mutation == "universe_authority":
        expected["universe_authority_version"] = "universe-v2"
    elif mutation == "identity_authority":
        expected["identity_authority_version"] = "identity-v2"
    elif mutation == "schema":
        expected["schema_version"] = "schema-v2"
    elif mutation == "configuration":
        expected["configuration_hash"] = "config-v2"

    result = check_dataset_lineage(
        dataset_path=dataset,
        expected=expected,
        intended_use="research",
    )

    assert result["status"] == STATUS_STALE
    assert reason in result["reasons"]
    assert result["use_authorized"] is False


def test_authority_and_label_changes_are_independent(tmp_path: Path):
    dataset, _, _, manifest = _write_current_manifest(tmp_path)
    expected = _expectation(manifest)
    expected["corporate_action_authority_version"] = "corp-v2"
    expected["market_calendar_authority_version"] = "calendar-v2"
    expected["label_code_version"] = "label-code-v2"

    result = check_dataset_lineage(dataset_path=dataset, expected=expected)

    assert result["status"] == STATUS_STALE
    assert "CORPORATE_ACTION_AUTHORITY_CHANGED" in result["reasons"]
    assert "MARKET_CALENDAR_AUTHORITY_CHANGED" in result["reasons"]
    assert "LABEL_CODE_CHANGED" in result["reasons"]


def test_missing_parent_and_conflicting_parent_are_distinct(tmp_path: Path):
    dataset, manifest_path, parent, manifest = _write_current_manifest(tmp_path)
    parent.unlink()

    missing = check_dataset_lineage(dataset_path=dataset, expected=_expectation(manifest))

    assert missing["status"] == STATUS_MISSING_PARENT
    assert "SOURCE_PATH_MISSING" in missing["reasons"]
    assert missing["missing_parents"]

    conflict_manifest = dict(manifest)
    conflict_manifest["source_manifest_hashes"] = [
        {"dataset_id": "parent-a", "sha256": "A" * 64},
        {"dataset_id": "parent-a", "sha256": "B" * 64},
    ]
    conflict_manifest["manifest_hash"] = manifest_hash(conflict_manifest)
    write_manifest(manifest_path, conflict_manifest)

    conflict = check_dataset_lineage(dataset_path=dataset, expected=_expectation(conflict_manifest))

    assert conflict["status"] == "CONFLICTING_PARENT"
    assert "CONFLICTING_PARENT" in conflict["reasons"]


def test_legacy_dataset_fails_closed_for_promotion_but_allows_diagnostic(tmp_path: Path):
    dataset = tmp_path / "legacy.csv"
    _write_csv(dataset, [{"feature_id": "a", "feature_date": "2024-01-01"}])

    promotion = check_dataset_lineage(dataset_path=dataset, intended_use="promotion-grade")
    diagnostic = check_dataset_lineage(dataset_path=dataset, intended_use="diagnostic")

    assert promotion["status"] == STATUS_LEGACY_NO_MANIFEST
    assert promotion["permitted_use"] == PERMITTED_DIAGNOSTIC
    assert promotion["use_authorized"] is False
    assert diagnostic["status"] == STATUS_LEGACY_NO_MANIFEST
    assert diagnostic["permitted_use"] == PERMITTED_DIAGNOSTIC
    assert diagnostic["use_authorized"] is True


def test_dirty_tree_build_is_unverified_and_not_promotion_grade(tmp_path: Path):
    dataset, _, _, manifest = _write_current_manifest(
        tmp_path,
        source_control={
            "contract_version": "source_worktree_provenance_v1",
            "git_commit": "abc",
            "dirty_worktree": True,
        },
    )

    research = check_dataset_lineage(
        dataset_path=dataset,
        expected=_expectation(manifest),
        intended_use="research",
    )
    promotion = check_dataset_lineage(
        dataset_path=dataset,
        expected=_expectation(manifest),
        intended_use="promotion-grade",
    )

    assert research["status"] == STATUS_UNVERIFIED
    assert research["permitted_use"] == PERMITTED_RESEARCH
    assert research["use_authorized"] is True
    assert "DIRTY_TREE_BUILD" in promotion["reasons"]
    assert promotion["use_authorized"] is False


def test_duplicate_key_change_is_detected_without_rebuild(tmp_path: Path):
    dataset, _, _, manifest = _write_current_manifest(tmp_path)
    before_manifest = dataset_manifest_path(dataset).stat().st_mtime_ns
    _write_csv(dataset, [
        _row("a", "AAA", "2024-01-01"),
        _row("a", "AAA", "2024-01-02"),
    ])

    result = check_dataset_lineage(dataset_path=dataset, expected=_expectation(manifest))

    assert result["status"] == STATUS_STALE
    assert "DUPLICATE_KEY_COUNT_CHANGED" in result["reasons"]
    assert "OUTPUT_HASH_CHANGED" in result["reasons"]
    assert dataset_manifest_path(dataset).stat().st_mtime_ns == before_manifest


def test_cli_writes_deterministic_dataset_lineage_output(tmp_path: Path):
    dataset, _, _, manifest = _write_current_manifest(tmp_path)
    output = tmp_path / "lineage.json"
    args = SimpleNamespace(
        dataset_path=str(dataset),
        dataset_manifest=None,
        expected_producer=None,
        expected_dataset_id=manifest["dataset_id"],
        expected_dataset_type=manifest["dataset_type"],
        expected_schema_version=manifest["schema_version"],
        expected_producer_command=manifest["producer_command"],
        expected_producer_module=manifest["producer_module"],
        expected_config_hash=manifest["configuration_hash"],
        expected_feature_code_version=manifest["feature_code_version"],
        expected_label_code_version=manifest["label_code_version"],
        universe_authority_version=manifest["universe_authority_version"],
        identity_authority_version=manifest["identity_authority_version"],
        corporate_action_authority_version=manifest["corporate_action_authority_version"],
        market_calendar_authority_version=manifest["market_calendar_authority_version"],
        intended_use="promotion-grade",
        lineage_output=str(output),
        verification_output=None,
    )

    result = run_dataset_lineage_check({}, args)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert result == written
    assert result["status"] == STATUS_CURRENT
    assert output.with_suffix(".md").exists()


def test_cli_exits_nonzero_for_promotion_grade_stale_input(tmp_path: Path):
    dataset, _, parent, manifest = _write_current_manifest(tmp_path)
    parent.write_text("parent_id,value\nsource,changed\n", encoding="utf-8")
    args = SimpleNamespace(
        dataset_path=str(dataset),
        dataset_manifest=None,
        expected_producer=None,
        expected_dataset_id=manifest["dataset_id"],
        expected_dataset_type=manifest["dataset_type"],
        expected_schema_version=manifest["schema_version"],
        expected_producer_command=manifest["producer_command"],
        expected_producer_module=manifest["producer_module"],
        expected_config_hash=manifest["configuration_hash"],
        expected_feature_code_version=manifest["feature_code_version"],
        expected_label_code_version=manifest["label_code_version"],
        universe_authority_version=manifest["universe_authority_version"],
        identity_authority_version=manifest["identity_authority_version"],
        corporate_action_authority_version=manifest["corporate_action_authority_version"],
        market_calendar_authority_version=manifest["market_calendar_authority_version"],
        intended_use="promotion-grade",
        lineage_output=None,
        verification_output=None,
    )

    with pytest.raises(SystemExit) as exc:
        run_dataset_lineage_check({}, args)

    assert exc.value.code == 2


def _write_current_manifest(
    tmp_path: Path,
    *,
    source_control: dict | None = None,
) -> tuple[Path, Path, Path, dict]:
    parent = tmp_path / "parent.csv"
    dataset = tmp_path / "dataset.csv"
    rows = [
        _row("a", "AAA", "2024-01-01"),
        _row("b", "BBB", "2024-01-02"),
    ]
    _write_csv(parent, [{"parent_id": "source", "value": "1"}])
    _write_csv(dataset, rows)
    manifest = build_dataset_build_manifest(
        dataset_id="synthetic-dataset",
        dataset_type="synthetic_dataset",
        schema_version="schema-v1",
        producer_command="ml-test-producer",
        producer_module="tests.test_dataset_build_manifest:producer",
        output_paths=(dataset,),
        source_paths=(parent,),
        source_dataset_ids=("source-dataset",),
        source_content_hashes={"feature_rows": "A" * 64},
        universe_authority_version="universe-v1",
        identity_authority_version="identity-v1",
        corporate_action_authority_version="corp-v1",
        market_calendar_authority_version="calendar-v1",
        feature_code_version="feature-code-v1",
        label_code_version="label-code-v1",
        configuration_hash_value="config-v1",
        random_seed=7,
        rows=rows,
        key_fields=("feature_id",),
        build_timestamp="2026-01-01T00:00:00+00:00",
        source_control=source_control
        or {
            "contract_version": "source_worktree_provenance_v1",
            "git_commit": "abc",
            "dirty_worktree": False,
        },
    )
    path = dataset_manifest_path(dataset)
    write_manifest(path, manifest)
    return dataset, path, parent, manifest


def _row(feature_id: str, symbol: str, day: str) -> dict[str, str]:
    return {
        "feature_id": feature_id,
        "symbol": symbol,
        "feature_date": day,
        "label_end_date": "2024-01-10",
        "selected_symbols": symbol,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["feature_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _expectation(manifest: dict) -> dict:
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_type": manifest["dataset_type"],
        "schema_version": manifest["schema_version"],
        "producer_command": manifest["producer_command"],
        "producer_module": manifest["producer_module"],
        "universe_authority_version": manifest["universe_authority_version"],
        "identity_authority_version": manifest["identity_authority_version"],
        "corporate_action_authority_version": manifest["corporate_action_authority_version"],
        "market_calendar_authority_version": manifest["market_calendar_authority_version"],
        "feature_code_version": manifest["feature_code_version"],
        "label_code_version": manifest["label_code_version"],
        "configuration_hash": manifest["configuration_hash"],
        "random_seed": manifest["random_seed"],
        "source_content_hashes": manifest["source_content_hashes"],
    }
