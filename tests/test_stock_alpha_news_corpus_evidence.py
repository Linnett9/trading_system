from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.stock_alpha_news_corpus_evidence import (
    CorpusEvidenceRequest,
    discover_external_evidence,
    resolve_corpus_evidence,
)


def _json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request(tmp_path, root):
    config = tmp_path / "config.yaml"
    config.write_text(
        "ml:\n  stock_alpha_news_canonical_corpus:\n"
        "    write_enabled: false\n  production_validated: false\n"
    )
    return CorpusEvidenceRequest(
        external_roots=(str(root),),
        expected_provider_scope=("Alpaca", "Benzinga"),
        runtime_config_path=str(config),
        canonical_output_root=str(tmp_path / "canonical-output"),
        shared_run_root=str(tmp_path / "runs"),
        resource_ledger_path=str(tmp_path / "ledger.json"),
        run_registry_path=str(tmp_path / "registry.json"),
        readiness_output_root=str(tmp_path / "readiness"),
        source_git_commit="abc123",
        source_git_branch="feature/test",
    )


def _assembly(root, name="assembly", providers=("Alpaca", "Benzinga")):
    stem = name if "assembly" in name else f"{name}_assembly"
    csv_path = root / f"{stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("provider,symbol,published_at_utc\n")
    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = _json(root / f"{stem}.json", {
        "schema_version": "alpaca_benzinga_historical_backfill_v1",
        "assembly_csv_path": csv_path.name,
        "assembly_checksum": checksum,
        "providers": list(providers),
        "row_count": 100,
        "min_published_at_utc": "2020-01-01T00:00:00Z",
        "max_published_at_utc": "2026-01-01T00:00:00Z",
    })
    return csv_path, metadata


def _canonical(root, *, production=True, include_data=True, audit=True):
    manifest = {
        "schema_version": "stock_alpha_news.historical_canonical_corpus.v2",
        "canonical_corpus_identity": "corpus-id",
        "canonical_corpus_checksum": "corpus-checksum",
        "canonical_rows_logical_checksum": "rows-checksum",
        "canonical_schema_checksum": "schema-checksum",
        "logical_manifest_checksum": "manifest-checksum",
        "source_assembly_checksum": "source-checksum",
        "canonical_row_count": 100,
        "source_metadata": {
            "production_validated": production,
            "providers": ["Alpaca", "Benzinga"],
        },
    }
    _json(root / "stock_alpha_news_canonical_corpus_manifest.json", manifest)
    if include_data:
        (root / "stock_alpha_news_canonical_corpus.csv").write_text("header\n")
    if audit:
        _json(root / "stock_alpha_news_canonical_corpus_audit.json", {"ok": True})
    (root / "stock_alpha_news_canonical_corpus_summary.md").write_text("summary")


def test_bounded_discovery_and_external_workspace_unchanged(tmp_path):
    root = tmp_path / "external"
    _assembly(root)
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }
    candidates, unavailable = discover_external_evidence(
        _request(tmp_path, root), max_depth=2, max_candidates=1,
        max_metadata_bytes=1024,
    )
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }
    assert len(candidates) == 1
    assert unavailable == []
    assert before == after


@pytest.mark.parametrize(
    ("production", "include_data", "audit", "expected"),
    [
        (True, True, True, "EXISTING_CANONICAL_CORPUS_READY"),
        (True, False, True, "EXISTING_CANONICAL_CORPUS_INCOMPLETE"),
        (True, True, False, "EXISTING_CANONICAL_CORPUS_INCOMPLETE"),
        (False, True, True, "EXISTING_CANONICAL_CORPUS_INCOMPLETE"),
    ],
)
def test_canonical_bundle_requires_complete_validated_sidecars(
    tmp_path, production, include_data, audit, expected
):
    root = tmp_path / "external"
    root.mkdir()
    _canonical(
        root, production=production, include_data=include_data, audit=audit
    )
    result = resolve_corpus_evidence(
        _request(tmp_path, root), output_root=tmp_path / "out"
    )
    assert result["canonical_status"] == expected


def test_large_canonical_csv_without_manifest_is_not_ready(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    (root / "stock_alpha_news_canonical_corpus.csv").write_text("header\n")
    result = resolve_corpus_evidence(
        _request(tmp_path, root), output_root=tmp_path / "out"
    )
    assert result["canonical_status"] == "NO_CANONICAL_CORPUS_FOUND"


def test_exact_assembly_approval_and_plan_only_are_nonexecuting(tmp_path):
    root = tmp_path / "external"
    csv_path, _ = _assembly(root)
    request = _request(tmp_path, root)
    result = resolve_corpus_evidence(
        request, output_root=tmp_path / "out",
        selection={"HISTORICAL_SOURCE_ASSEMBLY": str(csv_path)},
        approve_selection=True, emit_materialisation_request=True,
        run_plan_only=True,
        repository_root=Path(__file__).resolve().parents[1],
    )
    approved = json.loads(
        (tmp_path / "out" / "canonical_materialisation_request.json").read_text()
    )
    assert approved["execution_authorized"] is False
    assert result["plan_only_result"]["status"] == "PLAN_VALIDATED_NOT_EXECUTED"
    assert result["lease_acquired"] is False
    assert not (tmp_path / "ledger.json").exists()
    assert not (tmp_path / "registry.json").exists()


def test_approval_is_required_and_identity_is_deterministic(tmp_path):
    root = tmp_path / "external"
    csv_path, _ = _assembly(root)
    request = _request(tmp_path, root)
    first = resolve_corpus_evidence(
        request, output_root=tmp_path / "one",
        selection={"HISTORICAL_SOURCE_ASSEMBLY": str(csv_path)},
        emit_materialisation_request=True,
    )
    second = resolve_corpus_evidence(
        request, output_root=tmp_path / "two",
        selection={"HISTORICAL_SOURCE_ASSEMBLY": str(csv_path)},
        emit_materialisation_request=True,
    )
    assert first["status"] == "BLOCKED"
    assert not (tmp_path / "one" / "canonical_materialisation_request.json").exists()
    one = json.loads(
        (tmp_path / "one" / "canonical_materialisation_request.draft.json").read_text()
    )
    two = json.loads(
        (tmp_path / "two" / "canonical_materialisation_request.draft.json").read_text()
    )
    assert one["logical_request_identity"] == two["logical_request_identity"]


def test_multiple_assemblies_and_missing_provider_fail_closed(tmp_path):
    root = tmp_path / "external"
    _assembly(root, "one")
    _assembly(root, "two")
    request = _request(tmp_path, root)
    result = resolve_corpus_evidence(
        request, output_root=tmp_path / "ambiguous"
    )
    assert result["assembly_status"] == "AMBIGUOUS_SOURCE_ASSEMBLY"

    other = tmp_path / "missing-provider"
    _assembly(other, providers=("Benzinga",))
    result = resolve_corpus_evidence(
        _request(tmp_path, other), output_root=tmp_path / "missing"
    )
    assert result["assembly_status"] == "SOURCE_ASSEMBLY_INCOMPLETE"


def test_smoke_and_partial_evidence_are_rejected(tmp_path):
    root = tmp_path / "external"
    _assembly(root / "smoke", "assembly")
    partial = root / "assembly.partial.json"
    _json(partial, {"assembly_checksum": "x"})
    candidates, _ = discover_external_evidence(
        _request(tmp_path, root), max_depth=3, max_candidates=20,
        max_metadata_bytes=1024,
    )
    reasons = {
        reason for row in candidates for reason in row["reason_codes"]
    }
    classes = {row["artifact_class"] for row in candidates}
    assert "DEVELOPMENT_OR_SMOKE_PATH" in reasons
    assert "PARTIAL_OR_TEMPORARY_ARTIFACT" in classes
