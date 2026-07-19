from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.research.ml.stock_level.stock_alpha_finbert_scoring_foundation import (
    AUTHORIZATION_KEYS,
    SCORING_SELECTION_CONTRACT,
    authorization_template,
    deterministic_chunk_identity,
    deterministic_run_id,
    execute_authorized_boundary,
    logical_identity,
    publish_model_package,
    validate_authorization,
    validate_model_package,
    validate_selection,
)

REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"


def _source(root: Path) -> Path:
    root.mkdir()
    files = {
        "config.json": json.dumps({"id2label": {"0": "positive"}}),
        "pytorch_model.bin": "weights",
        "special_tokens_map.json": "{}",
        "tokenizer_config.json": "{}",
        "vocab.txt": "token\n",
    }
    for name, value in files.items():
        (root / name).write_text(value)
    return root


def _package(tmp_path: Path):
    return publish_model_package(
        source_root=_source(tmp_path / "source"),
        package_root=(tmp_path / "package").resolve(),
        model_name="ProsusAI/finbert",
        revision=REVISION,
    )


def _request(tmp_path: Path, package):
    payload = {
        "scoring_request_contract": "stock_alpha_news_finbert_scoring_request.v1",
        "canonical_identity": "c" * 64,
        "canonical_csv_sha256": "d" * 64,
        "model_package_identity": package["package_logical_identity"],
        "model_package_root": package["package_root"],
        "model_name": "ProsusAI/finbert",
        "model_revision": REVISION,
        "score_output_root": str((tmp_path / "scores").resolve()),
        "scored_at_selection_identity": "s" * 64,
    }
    payload["scoring_request_identity"] = logical_identity(payload)
    return payload


def _bundle(tmp_path):
    package = _package(tmp_path)
    request = _request(tmp_path, package)
    runtime_checksum = "r" * 64
    run_id = deterministic_run_id(request, runtime_checksum)
    authorization = authorization_template(
        request, runtime_checksum=runtime_checksum, expected_run_id=run_id
    )
    return package, request, runtime_checksum, authorization


def test_complete_package_is_deterministic_and_missing_weight_rejected(tmp_path):
    package = _package(tmp_path)
    assert validate_model_package(Path(package["package_root"])) == package
    assert package["package_logical_identity"] == validate_model_package(
        Path(package["package_root"])
    )["package_logical_identity"]
    source = _source(tmp_path / "incomplete")
    (source / "pytorch_model.bin").unlink()
    with pytest.raises(ValueError, match="SOURCE_FILE_INVALID"):
        publish_model_package(
            source_root=source, package_root=(tmp_path / "bad").resolve(),
            model_name="ProsusAI/finbert", revision=REVISION,
        )


def test_package_rejects_checksum_revision_and_symlink(tmp_path):
    package = _package(tmp_path)
    root = Path(package["package_root"])
    (root / "vocab.txt").write_text("changed")
    with pytest.raises(ValueError, match="SIZE|CHECKSUM"):
        validate_model_package(root)
    with pytest.raises(ValueError, match="REVISION"):
        publish_model_package(
            source_root=_source(tmp_path / "short-source"),
            package_root=(tmp_path / "short").resolve(),
            model_name="ProsusAI/finbert", revision="4556d130",
        )
    if hasattr(Path, "symlink_to"):
        source = _source(tmp_path / "link-source")
        target = source / "real.bin"
        target.write_text("weights")
        (source / "pytorch_model.bin").unlink()
        try:
            (source / "pytorch_model.bin").symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")
        with pytest.raises(ValueError, match="SOURCE_FILE_INVALID"):
            publish_model_package(
                source_root=source, package_root=(tmp_path / "linked").resolve(),
                model_name="ProsusAI/finbert", revision=REVISION,
            )


def test_authorization_exact_keys_boolean_bindings_and_identity(tmp_path):
    _, request, runtime, authorization = _bundle(tmp_path)
    run_id = deterministic_run_id(request, runtime)
    result = validate_authorization(
        authorization, request, runtime_checksum=runtime,
        expected_run_id=run_id, execution_required=False,
    )
    assert set(authorization) == AUTHORIZATION_KEYS
    assert result["execution_authorized"] is False
    with pytest.raises(ValueError, match="EXECUTION_AUTHORIZATION_REQUIRED"):
        validate_authorization(
            authorization, request, runtime_checksum=runtime,
            expected_run_id=run_id, execution_required=True,
        )
    for mutation in (
        lambda value: value.pop("authorized_stage"),
        lambda value: value.update({"notice": "no"}),
        lambda value: value.update({"execution_authorized": "false"}),
        lambda value: value.update({"canonical_identity": "wrong"}),
        lambda value: value.update({"canonical_csv_sha256": "wrong"}),
        lambda value: value.update({"model_package_identity": "wrong"}),
        lambda value: value.update({"model_revision": "wrong"}),
        lambda value: value.update({"expected_shared_run_id": "wrong"}),
        lambda value: value.update(
            {"reviewed_runtime_configuration_checksum": "wrong"}
        ),
        lambda value: value.update({"approved_score_output_root": "wrong"}),
        lambda value: value.update({"authorized_stage": "wrong"}),
    ):
        changed = dict(authorization)
        mutation(changed)
        with pytest.raises(ValueError):
            validate_authorization(
                changed, request, runtime_checksum=runtime,
                expected_run_id=run_id, execution_required=False,
            )
        assert logical_identity(changed) != logical_identity(authorization)


def test_plan_only_has_no_side_effects_and_true_boundary_orders_lease(tmp_path):
    package, request, runtime, authorization = _bundle(tmp_path)
    events = []
    callbacks = {
        "validate_package": lambda: events.append("package") or package,
        "persist_resource_request": lambda: events.append("persist"),
        "acquire_lease": lambda: events.append("lease") or "lease",
        "activate_source": lambda: events.append("source") or "source",
        "activate_model": lambda: events.append("model") or "model",
    }
    result = execute_authorized_boundary(
        request=request, runtime_checksum=runtime, authorization=authorization,
        plan_only=True, **callbacks,
    )
    assert result["status"] == "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION"
    assert events == ["package"]
    authorization["execution_authorized"] = True
    result = execute_authorized_boundary(
        request=request, runtime_checksum=runtime, authorization=authorization,
        plan_only=False, **callbacks,
    )
    assert result["status"] == "AUTHORIZED_EXECUTION_BOUNDARY_REACHED"
    assert events == ["package", "package", "persist", "lease", "source", "model"]


def test_invalid_authorization_precedes_every_lifecycle_side_effect(tmp_path):
    package, request, runtime, authorization = _bundle(tmp_path)
    authorization["canonical_identity"] = "wrong"
    events = []
    with pytest.raises(ValueError):
        execute_authorized_boundary(
            request=request, runtime_checksum=runtime,
            authorization=authorization, plan_only=False,
            validate_package=lambda: events.append("package") or package,
            persist_resource_request=lambda: events.append("persist"),
            acquire_lease=lambda: events.append("lease"),
            activate_source=lambda: events.append("source"),
            activate_model=lambda: events.append("model"),
        )
    assert events == []


def test_missing_selection_blocks_and_exclusion_is_explicit():
    selection = {
        "selection_contract": SCORING_SELECTION_CONTRACT,
        "approved_for_plan_construction": False,
        "execution_authorized": False,
        "scored_at_utc": None,
        "maximum_sequence_length": None,
        "inference_batch_size": None,
        "chunk_size": None,
        "worker_count": None,
        "inner_thread_count": 1,
        "device_policy": "CPU_ONLY",
        "memory_request_bytes": 10 * 1024**3,
        "retry_limit": None,
        "text_ineligible_row_policy": "GOVERNED_EXCLUSION",
        "canonical_row_count": 486757,
        "expected_eligible_row_count": 486756,
        "expected_excluded_row_count": 1,
        "score_output_root": "C:\\scores",
        "coverage_unit": "ARTICLE_SYMBOL_PAIR",
        "certification_contract":
            "stock_alpha_finbert_production_score_store.v1",
    }
    with pytest.raises(ValueError, match="MANDATORY_SCORING_SELECTIONS_MISSING"):
        validate_selection(selection)
    selection["scored_at_utc"] = "CURRENT_CLOCK"
    selection["maximum_sequence_length"] = 256
    selection["inference_batch_size"] = 8
    selection["chunk_size"] = 1000
    selection["worker_count"] = 1
    selection["retry_limit"] = 0
    selection["approved_for_plan_construction"] = True
    with pytest.raises(ValueError, match="CURRENT_CLOCK"):
        validate_selection(selection)


def test_chunk_identity_binds_all_resume_compatibility_inputs():
    values = {
        "canonical_identity": "c", "canonical_csv_sha256": "d",
        "model_package_identity": "m", "algorithm_contract": "a",
        "text_selection_contract": "t", "maximum_sequence_length": 256,
        "inference_batch_size": 8, "chunk_size": 1000, "start": 0,
        "stop": 1000, "scored_at_selection_identity": "s",
        "expected_eligible_rows": 486756, "output_schema_identity": "o",
    }
    baseline = deterministic_chunk_identity(**values)
    for key in values:
        changed = dict(values)
        changed[key] = changed[key] + "x" if isinstance(changed[key], str) else changed[key] + 1
        assert deterministic_chunk_identity(**changed) != baseline


def test_authorized_cli_plan_only_smoke_loads_no_model(tmp_path):
    package, request, _, _ = _bundle(tmp_path)
    runtime = {"runtime_contract": "stock_alpha_news_finbert_scoring_runtime.v1"}
    runtime_checksum = logical_identity(runtime)
    authorization = authorization_template(
        request, runtime_checksum=runtime_checksum,
        expected_run_id=deterministic_run_id(request, runtime_checksum),
    )
    paths = {}
    for name, payload in (
        ("request", request), ("runtime", runtime),
        ("authorization", authorization),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    result = subprocess.run([
        sys.executable, "scripts/run_authorized_stock_alpha_finbert_scoring.py",
        "--request", str(paths["request"]),
        "--runtime-config", str(paths["runtime"]),
        "--authorization", str(paths["authorization"]), "--plan-only",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == (
        "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION"
    )
    assert not Path(request["score_output_root"]).exists()
