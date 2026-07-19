from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.stock_alpha_news_compute_readiness_request import (
    NewsReadinessDiscoveryRequest,
    build_news_readiness_request,
    discover_candidates,
)


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request(tmp_path: Path, root: Path) -> NewsReadinessDiscoveryRequest:
    return NewsReadinessDiscoveryRequest(
        discovery_roots=(str(root),),
        selected_stages=("CORPUS",),
        expected_provider_scope="alpaca_benzinga",
        expected_model_id="ProsusAI/finbert",
        expected_model_revision="model-rev",
        expected_tokenizer_id="ProsusAI/finbert",
        expected_tokenizer_revision="tokenizer-rev",
        canonical_source_path=str(tmp_path / "source.csv"),
        runtime_config_path=str(tmp_path / "runtime.yaml"),
        candidate_canonical_corpus_root=str(tmp_path / "corpus"),
        candidate_score_store_root=str(tmp_path / "scores"),
        candidate_pit_feature_store_root=str(tmp_path / "features"),
        candidate_model_cache_root=str(tmp_path / "models"),
        candidate_run_root=str(tmp_path / "runs"),
        candidate_resource_ledger=str(tmp_path / "ledger.json"),
        candidate_registry=str(tmp_path / "registry.json"),
        final_audit_output_root=str(tmp_path / "audit"),
    )


def _corpus(identity="corpus-1"):
    return {
        "schema_version": "stock_alpha_news.historical_canonical_corpus.v1",
        "canonical_corpus_identity": identity,
        "canonical_corpus_checksum": f"{identity}-checksum",
        "logical_manifest_checksum": "logical",
        "canonical_schema_checksum": "schema",
        "canonical_row_count": 10,
        "source_assembly_identity": "source",
        "duplicate_group_count": 0,
        "ingested_at_utc": "2026-07-19T00:00:00Z",
    }


def test_draft_is_deterministic_and_cannot_invoke_audit(tmp_path):
    root = tmp_path / "discover"
    manifest = _write(root / "manifest.json", _corpus())
    request = _request(tmp_path, root)
    selection = {"CANONICAL_CORPUS": str(manifest.resolve())}

    first = build_news_readiness_request(
        request, output_root=tmp_path / "out-1", selection=selection
    )
    second = build_news_readiness_request(
        request, output_root=tmp_path / "out-2", selection=selection
    )

    assert request.identity == _request(tmp_path, root).identity
    assert first["status"] == second["status"] == "READY_WITH_CONDITIONS"
    assert not first["approved_request_emitted"]
    assert not first["audit_invoked"]
    assert not (tmp_path / "out-1" / "readiness_request.json").exists()
    assert json.loads(
        (tmp_path / "out-1" / "readiness_request.draft.json").read_text()
    )["request"]["canonical_manifest_path"] == str(manifest.resolve())


@pytest.mark.parametrize("run_audit", [False, True])
def test_approval_emits_request_but_audit_requires_explicit_flag(
    tmp_path, run_audit
):
    root = tmp_path / "discover"
    manifest = _write(root / "manifest.json", _corpus())
    result = build_news_readiness_request(
        _request(tmp_path, root),
        output_root=tmp_path / "out",
        selection={"CANONICAL_CORPUS": str(manifest.resolve())},
        approve_selection=True,
        run_audit=run_audit,
    )

    assert result["status"] == ("BLOCKED" if run_audit else "READY")
    assert result["approved_request_emitted"]
    assert result["audit_invoked"] is run_audit
    assert (tmp_path / "out" / "readiness_request.json").exists()


@pytest.mark.parametrize(
    ("relative_path", "payload", "classification"),
    [
        ("development/manifest.json", _corpus(), "INELIGIBLE_DEVELOPMENT_ONLY"),
        ("smoke/manifest.json", _corpus(), "INELIGIBLE_SMOKE_OR_PROBE"),
        ("bad.json", "{", "INELIGIBLE_MALFORMED"),
        ("unknown.json", {"hello": "world"}, "INELIGIBLE_WRONG_CONTRACT"),
    ],
)
def test_candidate_rejection_classifications(
    tmp_path, relative_path, payload, classification
):
    root = tmp_path / "discover"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    candidates, _ = discover_candidates(
        _request(tmp_path, root),
        max_depth=6,
        max_candidates=10,
        max_file_bytes=1024 * 1024,
    )
    assert candidates[0]["classification"] == classification


def test_ambiguity_and_bounds_are_fail_closed(tmp_path):
    root = tmp_path / "discover"
    _write(root / "one.json", _corpus("one"))
    _write(root / "two.json", _corpus("two"))
    result = build_news_readiness_request(
        _request(tmp_path, root), output_root=tmp_path / "out",
        max_candidates=1,
    )
    inventory = json.loads(
        (tmp_path / "out" / "candidate_inventory.json").read_text()
    )["candidates"]
    assert sum(row["artifact_type"] != "MODEL_CACHE" for row in inventory) == 1

    result = build_news_readiness_request(
        _request(tmp_path, root), output_root=tmp_path / "out-all"
    )
    assert result["status"] == "BLOCKED"
    assert any(
        row["code"] == "AMBIGUOUS_MULTIPLE_CANDIDATES"
        for row in result["blockers"]
    )
    assert not result["approved_request_emitted"]


def test_nested_nonproduction_config_is_ineligible(tmp_path):
    root = tmp_path / "discover"
    _write(root / "config.yaml", {
        "ml": {
            "production_validated": False,
            "stock_alpha_news_canonical_corpus": {
                "source_assembly_csv_path": "source.csv",
                "output_dir": "output",
                "write_enabled": False,
            },
        },
    })
    candidates, _ = discover_candidates(
        _request(tmp_path, root), max_depth=2, max_candidates=10,
        max_file_bytes=1024 * 1024,
    )
    assert candidates[0]["classification"] == "INELIGIBLE_INCOMPLETE"
    assert candidates[0]["reason_codes"] == [
        "CANONICAL_WRITE_DISABLED", "PRODUCTION_VALIDATION_FALSE"
    ]
