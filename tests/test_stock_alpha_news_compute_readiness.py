from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from core.research.compute.run_contracts import checksum
from core.research.ml.stock_level.stock_alpha_news_compute_readiness import (
    READ_ONLY_MARKER,
    NewsComputeReadinessRequest,
    audit_news_compute_readiness,
)


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path, *, selected=("CORPUS",), chunks=True, cache=True,
            certificate=True, feature=False):
    production = tmp_path / "production"
    corpus_root = production / "corpus"
    score_root = production / "scores"
    feature_root = production / "features"
    source = production / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("bounded source inventory\n", encoding="utf-8")
    corpus_artifact = corpus_root / "corpus.csv"
    corpus_artifact.parent.mkdir(parents=True)
    corpus_artifact.write_text("identity only\n", encoding="utf-8")
    corpus = {
        "schema_version": "stock_alpha_news.historical_canonical_corpus.v2",
        "canonical_corpus_identity": "CORPUS-ID",
        "canonical_corpus_checksum": "CORPUS-CHECKSUM",
        "logical_manifest_checksum": "CORPUS-MANIFEST",
        "canonical_schema_checksum": "SCHEMA",
        "canonical_row_count": 2,
        "source_assembly_identity": "SOURCE-ID",
        "duplicate_group_count": 1,
        "ingested_at_utc": "2026-07-18T00:00:00Z",
        "canonical_artifact_path": str(corpus_artifact),
        "source_metadata": {"providers": ["a", "b"]},
    }
    corpus_manifest = corpus_root / "manifest.json"
    _json(corpus_manifest, corpus)
    model = {
        "model_id": "ProsusAI/finbert", "model_revision": "model-rev",
        "tokenizer_id": "ProsusAI/finbert",
        "tokenizer_revision": "tokenizer-rev",
    }
    expected = [{
        "ordinal": 1, "chunk_id": "chunk-1", "article_count": 2,
        "identity": {"chunk_id": "chunk-1"},
    }]
    plan = {
        "scoring_plan_contract":
            "stock_alpha_finbert_production_scoring_plan.v1",
        "logical_checksum": "PLAN-ID", "plan_artifact_checksum": "PLAN-FILE",
        "canonical_corpus_identity": "CORPUS-ID",
        "canonical_corpus_checksum": "CORPUS-CHECKSUM",
        "finbert_model_identity": model, "expected_chunks": expected,
        "expected_chunk_count": 1, "configuration_checksum": "CONFIG",
        "maximum_token_length": 256,
    }
    plan_path = production / "plan.json"
    _json(plan_path, plan)
    chunk_path = score_root / "chunks" / "chunk-1.json"
    if chunks:
        _json(chunk_path, {"bounded": True})
        _csv(score_root / "chunk_manifest.csv", [{
            "chunk_id": "chunk-1", "status": "completed",
            "chunk_path": str(chunk_path), "scoring_plan_identity": "PLAN-ID",
            "chunk_artifact_sha256": "ARTIFACT",
            "scored_rows_logical_checksum": "ROWS", "article_count": "2",
        }])
    cert = {
        "score_store_contract":
            "stock_alpha_finbert_production_score_store.v1",
        "status": "COMPLETE", "production_scoring_complete": True,
        "production_scoring_plan_identity": "PLAN-ID",
        "canonical_corpus_identity": "CORPUS-ID",
        "canonical_corpus_checksum": "CORPUS-CHECKSUM",
        "finbert_model_identity": model, "score_store_identity": "SCORES",
        "certified_completed_chunk_count": 1,
        "certified_scored_row_count": 2,
    }
    cert_path = score_root / "certificate.json"
    if certificate:
        _json(cert_path, cert)
    spine_path = production / "spine.json"
    mapping_path = production / "mapping.json"
    alias_path = production / "aliases.json"
    _json(spine_path, {"daily_spine_identity": "SPINE",
                       "daily_spine_checksum": "SPINE-CHECKSUM"})
    _json(mapping_path, {"ticker_mapping_identity": "MAPPING",
                         "ticker_mapping_checksum": "MAPPING-CHECKSUM"})
    _json(alias_path, {"identity": "ALIASES"})
    feature_manifest = feature_root / "manifest.json"
    partition = feature_root / "decision_date=2024-01-01" / "part.jsonl"
    if feature:
        partition.parent.mkdir(parents=True)
        partition.write_text('{"bounded":true}\n', encoding="utf-8")
        _json(feature_manifest, {
            "feature_store_contract":
                "canonical_partitioned_pit_news_feature_store.v1",
            "canonical_corpus_identity": "CORPUS-ID",
            "canonical_corpus_checksum": "CORPUS-CHECKSUM",
            "score_store_identity": "SCORES",
            "canonical_daily_spine_identity": "SPINE",
            "ticker_mapping_identity": "MAPPING",
            "feature_schema_checksum": "FEATURE-SCHEMA", "row_count": 1,
            "partitions": [{"relative_path":
                            "decision_date=2024-01-01/part.jsonl",
                            "artifact_checksum": "PART", "row_count": 1}],
        })
    cache_root = tmp_path / "cache"
    if cache:
        for revision in ("model-rev", "tokenizer-rev"):
            snapshot = cache_root / "repo" / "snapshots" / revision
            snapshot.mkdir(parents=True, exist_ok=True)
        _json(cache_root / "repo" / "snapshots" / "model-rev" / "config.json",
              {"bounded": True})
        _json(cache_root / "repo" / "snapshots" / "tokenizer-rev"
              / "tokenizer_config.json", {"bounded": True})
    runtime = tmp_path / "runtime.json"
    _json(runtime, {"explicit": True})
    request = NewsComputeReadinessRequest(
        selected_stages=tuple(selected), canonical_source_path=str(source),
        canonical_corpus_root=str(corpus_root),
        canonical_manifest_path=str(corpus_manifest),
        scoring_plan_path=str(plan_path), score_store_root=str(score_root),
        chunk_manifest_path=str(score_root / "chunk_manifest.csv"),
        certification_path=str(cert_path),
        pit_feature_store_root=str(feature_root),
        pit_feature_manifest_path=str(feature_manifest),
        daily_spine_manifest_path=str(spine_path),
        ticker_mapping_manifest_path=str(mapping_path),
        alias_parent_path=str(alias_path), runtime_config_path=str(runtime),
        shared_run_root=str(tmp_path / "shared-runs"),
        resource_ledger_path=str(tmp_path / "compute" / "ledger.json"),
        run_registry_path=str(tmp_path / "compute" / "registry.json"),
        model_cache_root=str(cache_root),
        audit_output_path=str(tmp_path / "audit"),
    )
    return request


def audit(request):
    return audit_news_compute_readiness(
        request, repository_root=Path.cwd()
    )


def test_request_identity_report_checksum_and_ready_are_deterministic(tmp_path):
    request = fixture(tmp_path)
    assert request.identity == request.identity
    first = audit(request)
    second = audit(request)
    assert first["logical_checksum"] == second["logical_checksum"]
    assert first["overall_readiness"] == "READY"
    assert first["audit_mode"] == READ_ONLY_MARKER


def test_explicit_paths_and_no_defaults():
    with pytest.raises((TypeError, ValueError)):
        NewsComputeReadinessRequest(selected_stages=("CORPUS",))  # type: ignore


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing_manifest", "MANIFEST_NOT_FOUND"),
        ("corpus_mismatch", "CORPUS_ANCESTRY_MISMATCH"),
        ("model_revision", "MODEL_REVISION_MISMATCH"),
        ("malformed_ledger", "LEDGER_MALFORMED"),
    ],
)
def test_blocking_evidence(tmp_path, mutation, code):
    request = fixture(tmp_path, selected=("CORPUS", "SCORING"))
    if mutation == "missing_manifest":
        Path(request.canonical_manifest_path).unlink()
    elif mutation == "corpus_mismatch":
        plan = json.loads(Path(request.scoring_plan_path).read_text())
        plan["canonical_corpus_identity"] = "WRONG"
        _json(Path(request.scoring_plan_path), plan)
    elif mutation == "model_revision":
        plan = json.loads(Path(request.scoring_plan_path).read_text())
        plan["finbert_model_identity"]["model_revision"] = ""
        _json(Path(request.scoring_plan_path), plan)
    else:
        _json(Path(request.resource_ledger_path), {"contract_version": "wrong"})
    report = audit(request)
    assert report["overall_readiness"] == "BLOCKED"
    assert code in {row["code"] for row in report["blockers"]}


def test_missing_chunks_and_cache_are_conditional(tmp_path):
    request = fixture(
        tmp_path, selected=("CORPUS", "SCORING"), chunks=False, cache=False
    )
    report = audit(request)
    assert report["overall_readiness"] == "READY_WITH_CONDITIONS"
    assert report["scoring_resume_inventory"]["missing"] == 1
    assert report["model_cache_reference_state"] == (
        "MODEL_CACHE_REFERENCE_INCOMPLETE"
    )


@pytest.mark.parametrize("certificate,feature,expected", [
    (False, False, "BLOCKED"),
    (True, False, "READY_WITH_CONDITIONS"),
    (True, True, "READY_WITH_CONDITIONS"),
])
def test_pit_certification_and_feature_states(
    tmp_path, certificate, feature, expected
):
    request = fixture(
        tmp_path, selected=("CORPUS", "SCORING", "CERTIFICATION", "PIT"),
        certificate=certificate, feature=feature,
    )
    report = audit(request)
    assert report["overall_readiness"] == expected


def test_read_only_privacy_runbook_and_non_mutating_probes(tmp_path):
    request = fixture(tmp_path)
    before = {
        path: path.read_bytes() for path in (
            Path(request.canonical_manifest_path),
            Path(request.canonical_source_path),
        )
    }
    report = audit(request)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not Path(request.resource_ledger_path).exists()
    assert not Path(request.run_registry_path).exists()
    output = Path(request.audit_output_path)
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )
    assert "article body sentinel" not in combined
    runbook = (output / "operator_runbook.md").read_text()
    assert "Canonical corpus" in runbook
    assert "Stop new launches" in runbook
    assert "news-transformer trainer is not part" in runbook
    assert report["execution_lease_acquired"] is False
    assert report["network_access_performed"] is False
    assert report["model_activation_performed"] is False


def test_script_strict_exit_codes_and_help(tmp_path):
    help_result = subprocess.run([
        sys.executable, "scripts/audit_stock_alpha_news_compute_readiness.py",
        "--help",
    ], capture_output=True, text=True)
    assert help_result.returncode == 0
    for index, (request, expected) in enumerate((
        (fixture(tmp_path / "ready"), 0),
        (fixture(tmp_path / "conditional", selected=("CORPUS", "SCORING"),
                 chunks=False), 1),
        (fixture(tmp_path / "blocked", selected=("CORPUS", "SCORING")), 2),
    )):
        if expected == 2:
            Path(request.canonical_manifest_path).unlink()
        request_path = tmp_path / f"request-{index}.json"
        _json(request_path, {
            **asdict(request), "selected_stages": list(request.selected_stages)
        })
        result = subprocess.run([
            sys.executable,
            "scripts/audit_stock_alpha_news_compute_readiness.py",
            "--request", str(request_path),
            "--output-root", str(tmp_path / f"output-{index}"),
            "--strict", "--json",
        ], capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
