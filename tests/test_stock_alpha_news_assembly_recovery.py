from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.stock_alpha_news_assembly_recovery import (
    ASSEMBLY_NAME,
    AssemblyRecoveryRequest,
    audit_assembly_recovery,
    discover_recovery_candidates,
)

HEADER = (
    "provider,provider_article_id,symbol,published_at_utc,headline\n"
)


def _request(tmp_path, roots, target, **overrides):
    values = dict(
        search_roots=tuple(str(root) for root in roots),
        target_checksum=target,
        expected_filename_patterns=(
            ASSEMBLY_NAME, "*historical*corpus*assembly*",
        ),
        expected_provider_scope=("Alpaca", "Benzinga"),
        minimum_size_bytes=1,
        maximum_size_bytes=1024 * 1024,
        expected_row_count=2,
        expected_unique_article_count=2,
        expected_symbol_count=2,
        expected_published_min="2020-01-01T00:00:00Z",
        expected_published_max="2021-01-01T00:00:00Z",
        maximum_depth=4,
        maximum_candidates=50,
        maximum_metadata_bytes=1024 * 1024,
        maximum_full_hashes=10,
        output_root=str(tmp_path / "out"),
    )
    values.update(overrides)
    return AssemblyRecoveryRequest(**values)


def _assembly(root, name=ASSEMBLY_NAME, *, sidecar=True, providers=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(
        HEADER
        + "Alpaca,a1,AAPL,2020-01-01T00:00:00Z,x\n"
        + "Benzinga,b1,MSFT,2021-01-01T00:00:00Z,y\n"
    )
    if sidecar:
        metadata = {
            "schema_version": "alpaca_benzinga_historical_backfill_v1",
            "assembly_checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": 2,
            "unique_provider_article_count": 2,
            "symbol_count": 2,
            "min_published_at_utc": "2020-01-01T00:00:00Z",
            "max_published_at_utc": "2021-01-01T00:00:00Z",
            "providers": providers or ["Alpaca", "Benzinga"],
        }
        path.with_suffix(".json").write_text(json.dumps(metadata))
    return path


def test_explicit_roots_and_bounds_required(tmp_path):
    with pytest.raises(ValueError):
        _request(tmp_path, (), "a" * 64)
    root = tmp_path / "root"
    exact = _assembly(root / "deep" / "deeper")
    request = _request(
        tmp_path, (root, tmp_path / "missing"),
        hashlib.sha256(exact.read_bytes()).hexdigest(),
        maximum_depth=1, maximum_candidates=1,
    )
    candidates, roots = discover_recovery_candidates(request)
    assert candidates == []
    assert roots[1]["available"] is False


def test_exact_checksum_match_emits_1f_selection_without_mutation(tmp_path):
    root = tmp_path / "external"
    exact = _assembly(root)
    checksum = hashlib.sha256(exact.read_bytes()).hexdigest()
    before = (exact.stat().st_size, exact.stat().st_mtime_ns)
    result = audit_assembly_recovery(_request(tmp_path, (root,), checksum))
    after = (exact.stat().st_size, exact.stat().st_mtime_ns)
    selection = json.loads(
        (tmp_path / "out" / "corpus_evidence_selection.json").read_text()
    )
    assert result["target_checksum_found"]
    assert result["full_hash_count"] == 1
    assert selection["selection"]["HISTORICAL_SOURCE_ASSEMBLY"] == str(
        exact.resolve()
    )
    assert before == after
    assert not result["external_files_mutated"]


def test_checksum_mismatch_with_semantic_identity_fails_closed(tmp_path):
    root = tmp_path / "external"
    _assembly(root)
    result = audit_assembly_recovery(
        _request(tmp_path, (root,), "0" * 64)
    )
    inventory = json.loads(
        (tmp_path / "out" / "candidate_inventory.json").read_text()
    )["candidates"]
    match = next(row for row in inventory if row["filename"] == ASSEMBLY_NAME)
    assert match["classification"] == "ASSEMBLY_IDENTITY_MATCH_WITHOUT_BYTE_MATCH"
    assert result["status"] == "BLOCKED"
    assert not (tmp_path / "out" / "corpus_evidence_selection.json").exists()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("stock_alpha_news_canonical_corpus.csv",
         "CANONICAL_CORPUS_NOT_ASSEMBLY"),
        ("stock_alpha_news_alpaca_raw_provider_export.csv",
         "RAW_PROVIDER_EXPORT"),
        ("stock_alpha_news_historical_corpus_assembly_smoke.csv",
         "SMOKE_OR_DEVELOPMENT_ARTIFACT"),
        ("stock_alpha_news_historical_corpus_assembly.partial.csv",
         "PARTIAL_OR_TEMPORARY_ARTIFACT"),
    ],
)
def test_nonassembly_artifacts_are_rejected(tmp_path, name, expected):
    root = tmp_path / "external"
    path = _assembly(root, name, sidecar=False)
    request = _request(tmp_path, (root,), "0" * 64)
    candidates, _ = discover_recovery_candidates(request)
    row = next(item for item in candidates if item["path"] == str(path.resolve()))
    assert row["classification"] == expected
    assert not row["preliminary_hash_eligible"]


def test_full_hash_limit_and_content_preferred_over_mtime(tmp_path):
    root = tmp_path / "external"
    first = _assembly(root / "one")
    second = _assembly(root / "two")
    checksum = hashlib.sha256(second.read_bytes()).hexdigest()
    first.touch()
    result = audit_assembly_recovery(_request(
        tmp_path, (root,), checksum, maximum_full_hashes=1,
    ))
    assert result["full_hash_count"] == 1
    assert any(
        row["code"] == "FULL_HASH_LIMIT_REACHED"
        for row in result["warnings"]
    )


def test_provider_scope_is_required_for_preliminary_hash(tmp_path):
    root = tmp_path / "external"
    path = _assembly(root, providers=["Benzinga"])
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata["schema_version"] = "historical_backfill_v1"
    sidecar.write_text(json.dumps(metadata))
    request = _request(
        tmp_path, (root,), hashlib.sha256(path.read_bytes()).hexdigest()
    )
    candidates, _ = discover_recovery_candidates(request)
    row = next(item for item in candidates if item["filename"] == ASSEMBLY_NAME)
    assert "EXPECTED_PROVIDER_SCOPE_MISSING" in row["reason_codes"]
    assert not row["preliminary_hash_eligible"]


def test_complete_shard_inventory_can_be_rebuild_ready(tmp_path):
    root = tmp_path / "external"
    shard = _assembly(root, "historical_news_partition_001.csv", sidecar=False)
    metadata = {
        "schema_version": "alpaca_benzinga_historical_backfill_v1",
        "providers": ["Alpaca", "Benzinga"],
        "complete_partition_count": 1,
        "incomplete_partition_count": 0,
        "ordering_policy": "partition,symbol,published_at_utc",
    }
    (root / "historical_backfill_manifest.json").write_text(
        json.dumps(metadata)
    )
    result = audit_assembly_recovery(
        _request(tmp_path, (root,), "0" * 64)
    )
    assert result["rebuild_readiness"] == "ASSEMBLY_REBUILD_READY"
    assert shard.exists()


def test_missing_inputs_require_external_restoration(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    result = audit_assembly_recovery(
        _request(tmp_path, (root,), "0" * 64), strict=True
    )
    assert result["rebuild_readiness"] == "EXTERNAL_RESTORATION_REQUIRED"
    assert result["strict_failure"]
    assert result["network_access_performed"] is False
    assert result["model_activation_performed"] is False
    assert result["lease_acquired"] is False


def test_request_and_report_identity_are_deterministic(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    first = _request(tmp_path, (root,), "0" * 64)
    second = _request(tmp_path, (root,), "0" * 64)
    assert first.identity == second.identity
    audit_assembly_recovery(first)
    payload = json.loads(
        (tmp_path / "out" / "recovery_request.json").read_text()
    )
    assert payload["request_identity"] == first.identity
