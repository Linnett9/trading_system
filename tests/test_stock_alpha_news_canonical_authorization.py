from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.research.ml.stock_level.stock_alpha_news_canonical_authorization import (
    AUTHORIZATION_V2,
    AUTHORIZATION_V3,
    NOTICE,
    authorization_template,
    build_v2_request,
    load_authorization,
    normalize_ingested_at_utc,
    runtime_configuration,
    validate_authorization,
    validate_v2_request,
)
from scripts.run_authorized_stock_alpha_news_canonical_materialisation import (
    _plan,
    main as authorization_main,
)
from core.research.ml.stock_level.stock_alpha_news_data_compute import (
    deterministic_news_data_run_id,
)


def _v1(tmp_path):
    source = tmp_path / "source" / "assembly.csv"
    metadata = source.with_suffix(".json")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("provider,provider_article_id,symbol,published_at_utc\n")
    metadata.write_text("{}")
    return {
        "contract_version":
            "stock_alpha_news_canonical_materialisation_request.v1",
        "source_assembly_path": str(source),
        "source_metadata_path": str(metadata),
        "source_assembly_identity": "source-id",
        "source_assembly_checksum": "a" * 64,
        "canonical_output_root": str(tmp_path / "output"),
        "canonical_contract_version":
            "stock_alpha_news.historical_canonical_corpus.v2",
        "expected_published_min": "2020-01-01T00:00:00Z",
        "expected_published_max": "2021-01-01T00:00:00Z",
        "expected_symbol_count": 2,
        "availability_time_policy_identity": "availability-v1",
        "resource_profile": {
            "estimated_peak_ram_bytes": 4 * 1024**3, "cpu_weight": 1,
            "inner_threads": 1, "gpu_required": False,
            "estimate_source": "CONSERVATIVE_DEFAULT",
        },
        "shared_compute_run_root": str(tmp_path / "runs"),
        "resource_ledger_path": str(tmp_path / "ledger.json"),
        "run_registry_path": str(tmp_path / "registry.json"),
        "source_git_commit_evidence": "commit",
        "execution_authorized": False,
    }


def _provenance():
    return {
        "provenance_type": "OPERATOR_REVIEWED_FIXED_TIMESTAMP",
        "evidence_path": "selection.json",
        "evidence_contract": "stock_alpha_news_ingested_at_utc_selection.v1",
        "evidence_identity": "selection-id",
        "source_field": "ingested_at_utc",
        "assembly_sha256": "a" * 64,
        "selection_status": "APPROVED_FOR_PLAN_CONSTRUCTION",
    }


def _v2(tmp_path, timestamp="2026-07-10T22:00:00Z"):
    return build_v2_request(
        _v1(tmp_path), ingested_at_utc=timestamp,
        provenance=_provenance(),
    )


def _authorization_bundle(tmp_path):
    request = _v2(tmp_path)
    runtime = runtime_configuration(request)
    run_id = deterministic_news_data_run_id(
        _plan(request, runtime["configuration_checksum"])
    )
    authorization = authorization_template(
        request, runtime["configuration_checksum"], run_id
    )
    return request, runtime, run_id, authorization


def _historical_v2(authorization, *, execution_authorized=False):
    return {
        "notice": NOTICE,
        **authorization,
        "authorization_contract": AUTHORIZATION_V2,
        "execution_authorized": execution_authorized,
    }


@pytest.mark.parametrize("value", ["", "2026-07-10", "2026-07-10T22:00:00+01:00"])
def test_timestamp_requires_explicit_utc(value):
    with pytest.raises(ValueError):
        normalize_ingested_at_utc(value)


def test_v1_rejected_and_v2_identity_is_deterministic(tmp_path):
    with pytest.raises(ValueError, match="INCOMPLETE"):
        validate_v2_request(_v1(tmp_path))
    first = _v2(tmp_path)
    second = _v2(tmp_path)
    changed = _v2(tmp_path, "2026-07-10T22:00:01Z")
    assert first["logical_request_identity"] == second["logical_request_identity"]
    assert first["logical_request_identity"] != changed["logical_request_identity"]
    assert first["ingested_at_utc"] == "2026-07-10T22:00:00Z"


def test_provenance_checksum_must_match(tmp_path):
    provenance = _provenance()
    provenance["assembly_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="ASSEMBLY_MISMATCH"):
        build_v2_request(
            _v1(tmp_path), ingested_at_utc="2026-07-10T22:00:00Z",
            provenance=provenance,
        )


def test_authorization_exact_matching_and_false_execution_gate(tmp_path):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    assert authorization["authorization_contract"] == AUTHORIZATION_V3
    assert "notice" not in authorization
    result = validate_authorization(
        authorization, request, runtime["configuration_checksum"], run_id,
        execution_required=False,
    )
    assert result["execution_authorized"] is False
    with pytest.raises(ValueError, match="PRODUCTION_AUTHORIZATION_REQUIRED"):
        validate_authorization(
            authorization, request, runtime["configuration_checksum"], run_id,
            execution_required=True,
        )
    for key in (
        "materialisation_request_identity", "source_assembly_sha256",
        "ingested_at_utc", "ingested_at_utc_provenance_identity",
        "expected_shared_run_id", "approved_output_root",
        "authorized_stage", "reviewed_runtime_configuration_checksum",
    ):
        wrong = dict(authorization)
        wrong[key] = "wrong"
        with pytest.raises(ValueError):
            validate_authorization(
                wrong, request, runtime["configuration_checksum"], run_id,
                execution_required=False,
            )


def test_unexpected_scope_key_and_old_authorization_rejected(tmp_path):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    authorization["scoring_authorized"] = True
    with pytest.raises(ValueError, match="SCOPE_KEYS"):
        validate_authorization(
            authorization, request, runtime["configuration_checksum"], run_id,
            execution_required=False,
        )
    authorization.pop("scoring_authorized")
    authorization["authorization_contract"] = (
        "stock_alpha_news_canonical_materialisation_authorization.v1"
    )
    with pytest.raises(
        ValueError,
        match="INCOMPLETE_OR_SUPERSEDED_AUTHORIZATION_CONTRACT",
    ):
        validate_authorization(
            authorization, request, runtime["configuration_checksum"], run_id,
            execution_required=False,
        )


def test_plan_only_cli_accepts_false_bom_template_without_mutation(tmp_path):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    request_path = tmp_path / "request.json"
    runtime_path = tmp_path / "runtime.json"
    auth_path = tmp_path / "authorization.json"
    request_path.write_text(json.dumps(request))
    runtime_path.write_text(json.dumps(runtime))
    auth_path.write_text("\ufeff" + json.dumps(authorization), encoding="utf-8")
    result = subprocess.run([
        sys.executable,
        "scripts/run_authorized_stock_alpha_news_canonical_materialisation.py",
        "--request", str(request_path), "--authorization", str(auth_path),
        "--runtime-config", str(runtime_path),
        "--run-root", request["shared_compute_run_root"],
        "--resource-ledger", request["resource_ledger_path"],
        "--registry", request["run_registry_path"], "--plan-only", "--json",
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == (
        "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION"
    )
    assert not Path(request["resource_ledger_path"]).exists()
    assert not Path(request["run_registry_path"]).exists()
    assert not Path(request["shared_compute_run_root"]).exists()
    assert not Path(request["canonical_output_root"]).exists()


@pytest.mark.parametrize("key", [
    "materialisation_request_identity", "source_assembly_sha256",
    "ingested_at_utc", "ingested_at_utc_provenance_identity",
    "expected_shared_run_id", "approved_output_root", "authorized_stage",
    "reviewed_runtime_configuration_checksum",
])
def test_every_v3_immutable_field_changes_identity_and_is_rejected(
    tmp_path, key,
):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    changed = dict(authorization)
    changed[key] = "changed"
    with pytest.raises(ValueError):
        validate_authorization(
            changed, request, runtime["configuration_checksum"], run_id,
            execution_required=False,
        )
    identity = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert identity(authorization) != identity(changed)


def test_v3_exact_keys_boolean_and_relative_path(tmp_path):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    for mutation, code in (
        (lambda value: value.pop("authorized_stage"), "SCOPE_KEYS"),
        (lambda value: value.update({"notice": NOTICE}), "SCOPE_KEYS"),
        (lambda value: value.update({"reviewer": "operator"}), "SCOPE_KEYS"),
        (
            lambda value: value.update({"execution_authorized": "false"}),
            "FLAG_MUST_BE_BOOLEAN",
        ),
        (
            lambda value: value.update({"approved_output_root": "relative"}),
            "OUTPUT_ROOT",
        ),
    ):
        changed = dict(authorization)
        mutation(changed)
        with pytest.raises(ValueError, match=code):
            validate_authorization(
                changed, request, runtime["configuration_checksum"], run_id,
                execution_required=False,
            )


def test_v3_identity_covers_execution_flag_and_complete_payload(tmp_path):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    false_result = validate_authorization(
        authorization, request, runtime["configuration_checksum"], run_id,
        execution_required=False,
    )
    approved = {**authorization, "execution_authorized": True}
    true_result = validate_authorization(
        approved, request, runtime["configuration_checksum"], run_id,
        execution_required=True,
    )
    assert false_result["authorization_identity"] != (
        true_result["authorization_identity"]
    )


def test_v2_execute_rejection_and_contradiction_codes(tmp_path):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    historical = _historical_v2(authorization)
    assert validate_authorization(
        historical, request, runtime["configuration_checksum"], run_id,
        execution_required=False,
    )["historical_contract"] is True
    with pytest.raises(ValueError, match="SUPERSEDED_AUTHORIZATION_CONTRACT"):
        validate_authorization(
            historical, request, runtime["configuration_checksum"], run_id,
            execution_required=True,
        )
    contradictory = {
        **historical, "execution_authorized": True,
    }
    with pytest.raises(
        ValueError, match="AUTHORIZATION_NOTICE_CONTRACT_CONTRADICTION"
    ):
        validate_authorization(
            contradictory, request, runtime["configuration_checksum"], run_id,
            execution_required=True,
        )


def test_malformed_authorization_rejected(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        load_authorization(path)


def test_synthetic_v3_execution_reaches_orchestrator_after_validation(
    tmp_path, monkeypatch,
):
    request, runtime, run_id, authorization = _authorization_bundle(tmp_path)
    authorization["execution_authorized"] = True
    request_path = tmp_path / "request.json"
    runtime_path = tmp_path / "runtime.json"
    auth_path = tmp_path / "authorization.json"
    request_path.write_text(json.dumps(request))
    runtime_path.write_text(json.dumps(runtime))
    auth_path.write_text(json.dumps(authorization))
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "scripts.run_authorized_stock_alpha_news_canonical_materialisation."
        "subprocess.run",
        fake_run,
    )
    result = authorization_main([
        "--request", str(request_path), "--authorization", str(auth_path),
        "--runtime-config", str(runtime_path),
        "--run-root", request["shared_compute_run_root"],
        "--resource-ledger", request["resource_ledger_path"],
        "--registry", request["run_registry_path"], "--execute", "--json",
    ])
    assert result == 0
    assert "run_stock_alpha_news_data_compute.py" in " ".join(
        str(value) for value in observed["command"]
    )
    assert not Path(request["shared_compute_run_root"]).exists()
    assert not Path(request["resource_ledger_path"]).exists()
    assert not Path(request["run_registry_path"]).exists()
