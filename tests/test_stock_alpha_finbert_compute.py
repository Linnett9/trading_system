from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.compute.machine_profile import GIB, dell_i5_10500_profile
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.ml.stock_level.stock_alpha_finbert_compute import (
    FinBertExecutionPolicy,
    build_chunk_resource_request,
    deterministic_chunk_item_id,
    deterministic_run_id,
    execute_finbert_compute_run,
)
from tests.test_stock_alpha_finbert_score_store import _plan_fixture


class Adapter:
    def __init__(self, *, reused: bool = False, fail: str | None = None):
        self.reused = reused
        self.fail = fail
        self.calls = []

    def compatible_output(self, chunk):
        self.calls.append("compatible")
        if self.reused:
            return {
                "chunk_id": chunk["chunk_id"], "chunk_path": "existing.json",
                "chunk_artifact_sha256": "ABC", "row_count": chunk["article_count"],
            }
        return None

    def load_model(self, reference, policy):
        self.calls.append("load")
        if self.fail == "load":
            raise RuntimeError("load failed")
        return object()

    def tokenize(self, model, chunk):
        self.calls.append("tokenize")
        return ["tokens"]

    def infer(self, model, tokenized, chunk):
        self.calls.append("infer")
        if self.fail == "infer":
            raise RuntimeError("infer failed")
        return ["prediction"]

    def publish(self, chunk, predictions):
        self.calls.append("publish")
        if self.fail == "publish":
            raise RuntimeError("publish failed")
        return {
            "chunk_id": chunk["chunk_id"], "chunk_path": "new.json",
            "chunk_artifact_sha256": "DEF", "row_count": chunk["article_count"],
            "publication_result": "PUBLISHED",
        }


def ledger(tmp_path: Path):
    profile = dell_i5_10500_profile(
        source_git_commit="commit", generated_at="fixed"
    )
    service = ResourceLeaseLedger(
        profile=profile, path=tmp_path / "ledger.json",
        available_memory=lambda: 32 * GIB,
    )
    service.initialise_ledger()
    return profile, service


def plan():
    return _plan_fixture()[0]


def run(tmp_path: Path, adapter: Adapter, *, certification_complete=True):
    profile, service = ledger(tmp_path)
    result = execute_finbert_compute_run(
        scoring_plan=plan(), adapter=adapter,
        certify=lambda _: {
            "status": "COMPLETE" if certification_complete else "INCOMPLETE",
            "production_scoring_complete": certification_complete,
            "score_store_identity": "certificate",
        },
        machine_profile=profile, lease_ledger=service,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    return result, service


def test_deterministic_identity_request_and_explicit_policy():
    fixture = plan()
    policy = FinBertExecutionPolicy()
    assert deterministic_run_id(fixture, policy) == deterministic_run_id(
        fixture, policy
    )
    item = deterministic_chunk_item_id(
        deterministic_run_id(fixture), fixture["expected_chunks"][0], fixture
    )
    assert item == deterministic_chunk_item_id(
        deterministic_run_id(fixture), fixture["expected_chunks"][0], fixture
    )
    request = build_chunk_resource_request(
        run_id="run", item_id=item, attempt_identity="attempt", policy=policy
    )
    assert request.estimated_peak_ram_bytes == 10 * GIB
    assert request.cpu_weight == 2
    assert request.estimate_source == "CONSERVATIVE_DEFAULT"
    assert request.gpu_required is False
    with pytest.raises(ValueError, match="explicit"):
        FinBertExecutionPolicy(device="cuda")


def test_compatible_skip_precedes_load_and_certification_has_no_request(tmp_path):
    result, service = run(tmp_path, Adapter(reused=True))
    assert result["summary"]["compatible_reused_chunks"] == len(
        plan()["expected_chunks"]
    )
    assert result["summary"]["newly_scored_chunks"] == 0
    assert result["resource_summary"]["requests_created"] == len(
        plan()["expected_chunks"]
    )
    assert service.read_ledger_status()["active_leases"] == []
    spans = json.loads(
        (Path(result["run_root"]) / "telemetry_spans.json").read_text()
    )["spans"]
    assert {row["name"] for row in spans} == {"compatible_resume_check"}


def test_lease_before_load_release_telemetry_reference_and_registry(tmp_path):
    adapter = Adapter()
    result, service = run(tmp_path, adapter)
    assert adapter.calls == [
        step
        for _ in plan()["expected_chunks"]
        for step in ("compatible", "load", "tokenize", "infer", "publish")
    ]
    assert service.read_ledger_status()["active_leases"] == []
    assert result["summary"]["newly_scored_chunks"] == len(
        plan()["expected_chunks"]
    )
    assert result["model_reference"]["artifact_type"] == (
        "EXTERNAL_PINNED_MODEL_REFERENCE"
    )
    assert result["model_reference"]["resolution_policy"] == (
        "reference_only_no_weights_copied"
    )
    names = {
        row["name"] for row in json.loads(
            (Path(result["run_root"]) / "telemetry_spans.json").read_text()
        )["spans"]
    }
    assert names == {
        "base_model_loading", "tokenisation", "inference",
        "atomic_chunk_publication",
    }
    assert result["registry"]["health"] == "HEALTHY"


@pytest.mark.parametrize("phase", ["load", "infer", "publish"])
def test_failure_releases_lease_and_fails_run(tmp_path, phase):
    result, service = run(tmp_path, Adapter(fail=phase))
    assert service.read_ledger_status()["active_leases"] == []
    assert result["summary"]["failed_chunks"] == len(plan()["expected_chunks"])
    assert result["summary"]["final_run_status"] == "FAILED"


def test_certification_failure_is_terminal_and_non_model(tmp_path):
    result, service = run(
        tmp_path, Adapter(reused=True), certification_complete=False
    )
    assert result["summary"]["certification_status"] == "INCOMPLETE"
    assert result["summary"]["final_run_status"] == "FAILED"
    requests = json.loads(
        (Path(result["run_root"]) / "resource_requests.json").read_text()
    )["requests"]
    assert len(requests) == len(plan()["expected_chunks"])
    assert service.read_ledger_status()["active_leases"] == []
