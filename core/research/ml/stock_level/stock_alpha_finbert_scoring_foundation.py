from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

MODEL_PACKAGE_CONTRACT = "stock_alpha_finbert_immutable_model_package.v1"
SCORING_REQUEST_CONTRACT = "stock_alpha_news_finbert_scoring_request.v1"
SCORING_RUNTIME_CONTRACT = "stock_alpha_news_finbert_scoring_runtime.v1"
SCORING_AUTHORIZATION_CONTRACT = (
    "stock_alpha_news_finbert_scoring_authorization.v1"
)
SCORING_SELECTION_CONTRACT = "stock_alpha_news_finbert_scoring_selection.v1"
AUTHORIZED_STAGE = "FINBERT_ARTICLE_SCORING"
FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)
AUTHORIZATION_KEYS = {
    "authorization_contract",
    "execution_authorized",
    "scoring_request_identity",
    "canonical_identity",
    "canonical_csv_sha256",
    "model_package_identity",
    "model_name",
    "model_revision",
    "expected_shared_run_id",
    "approved_score_output_root",
    "authorized_stage",
    "reviewed_runtime_configuration_checksum",
    "scored_at_selection_identity",
}


def logical_identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_model_package(
    *,
    source_root: Path,
    package_root: Path,
    model_name: str,
    revision: str,
) -> dict[str, Any]:
    if not FULL_REVISION.fullmatch(revision):
        raise ValueError("MODEL_REVISION_MUST_BE_FULL_IMMUTABLE_SHA")
    if package_root.exists():
        raise FileExistsError("MODEL_PACKAGE_DESTINATION_ALREADY_EXISTS")
    inventory = _source_inventory(source_root)
    temporary = package_root.with_name(f".{package_root.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError("MODEL_PACKAGE_TEMPORARY_DESTINATION_EXISTS")
    temporary.mkdir(parents=True)
    try:
        copied = []
        for item in inventory:
            source = source_root / item["name"]
            destination = temporary / item["name"]
            shutil.copyfile(source, destination)
            verified = _file_record(destination)
            if verified["size"] != item["size"] or verified["sha256"] != item["sha256"]:
                raise ValueError("MODEL_PACKAGE_COPY_VERIFICATION_FAILED")
            copied.append({**verified, "source_path": str(source.resolve())})
        config = json.loads((temporary / "config.json").read_text(encoding="utf-8"))
        manifest = {
            "package_contract": MODEL_PACKAGE_CONTRACT,
            "model_name": model_name,
            "model_revision": revision,
            "package_root": str(package_root.resolve()),
            "required_files": list(REQUIRED_FILES),
            "files": copied,
            "model_configuration_sha256": _by_name(copied, "config.json")["sha256"],
            "tokenizer_configuration_sha256": _by_name(
                copied, "tokenizer_config.json"
            )["sha256"],
            "vocabulary_sha256": _by_name(copied, "vocab.txt")["sha256"],
            "label_mapping": config.get("id2label"),
            "offline_local_only_required": True,
            "publication_complete": True,
            "total_size": sum(item["size"] for item in copied),
        }
        manifest["package_logical_identity"] = logical_identity(manifest)
        (temporary / "model_package_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, package_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_model_package(package_root)


def validate_model_package(
    package_root: Path,
    *,
    expected_model_name: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if not package_root.is_absolute():
        raise ValueError("MODEL_PACKAGE_ROOT_MUST_BE_ABSOLUTE")
    if (package_root / ".publication-incomplete").exists():
        raise ValueError("MODEL_PACKAGE_PUBLICATION_INCOMPLETE")
    manifest_path = package_root / "model_package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "package_contract", "model_name", "model_revision", "package_root",
        "required_files", "files", "model_configuration_sha256",
        "tokenizer_configuration_sha256", "vocabulary_sha256", "label_mapping",
        "offline_local_only_required", "publication_complete", "total_size",
        "package_logical_identity",
    }
    if set(manifest) != expected_keys:
        raise ValueError("MODEL_PACKAGE_MANIFEST_SCOPE_MISMATCH")
    if manifest["package_contract"] != MODEL_PACKAGE_CONTRACT:
        raise ValueError("MODEL_PACKAGE_CONTRACT_UNSUPPORTED")
    if not FULL_REVISION.fullmatch(str(manifest["model_revision"])):
        raise ValueError("MODEL_REVISION_MUST_BE_FULL_IMMUTABLE_SHA")
    if expected_model_name and manifest["model_name"] != expected_model_name:
        raise ValueError("MODEL_PACKAGE_NAME_MISMATCH")
    if expected_revision and manifest["model_revision"] != expected_revision:
        raise ValueError("MODEL_PACKAGE_REVISION_MISMATCH")
    if Path(manifest["package_root"]).resolve() != package_root.resolve():
        raise ValueError("MODEL_PACKAGE_ROOT_MISMATCH")
    if manifest["required_files"] != list(REQUIRED_FILES):
        raise ValueError("MODEL_PACKAGE_REQUIRED_FILES_MISMATCH")
    if manifest["publication_complete"] is not True:
        raise ValueError("MODEL_PACKAGE_PUBLICATION_INCOMPLETE")
    records = manifest["files"]
    if {row["name"] for row in records} != set(REQUIRED_FILES):
        raise ValueError("MODEL_PACKAGE_FILE_INVENTORY_MISMATCH")
    for record in records:
        path = package_root / record["name"]
        if path.is_symlink():
            raise ValueError("MODEL_PACKAGE_REQUIRED_FILE_IS_SYMLINK")
        actual = _file_record(path)
        if actual["size"] != record["size"]:
            raise ValueError("MODEL_PACKAGE_FILE_SIZE_MISMATCH")
        if actual["sha256"] != record["sha256"]:
            raise ValueError("MODEL_PACKAGE_FILE_CHECKSUM_MISMATCH")
    identity_payload = {
        key: value for key, value in manifest.items()
        if key != "package_logical_identity"
    }
    if logical_identity(identity_payload) != manifest["package_logical_identity"]:
        raise ValueError("MODEL_PACKAGE_IDENTITY_MISMATCH")
    return manifest


def validate_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "selection_contract", "approved_for_plan_construction",
        "execution_authorized", "scored_at_utc", "maximum_sequence_length",
        "inference_batch_size", "chunk_size", "worker_count",
        "inner_thread_count", "device_policy", "memory_request_bytes",
        "retry_limit", "text_ineligible_row_policy", "canonical_row_count",
        "expected_eligible_row_count", "expected_excluded_row_count",
        "score_output_root", "coverage_unit", "certification_contract",
    }
    if set(selection) != required:
        raise ValueError("SCORING_SELECTION_SCOPE_MISMATCH")
    if selection["selection_contract"] != SCORING_SELECTION_CONTRACT:
        raise ValueError("SCORING_SELECTION_CONTRACT_UNSUPPORTED")
    if selection["execution_authorized"] is not False:
        raise ValueError("SELECTION_MUST_NOT_AUTHORIZE_EXECUTION")
    mandatory = required - {
        "selection_contract", "approved_for_plan_construction",
        "execution_authorized",
    }
    missing = sorted(key for key in mandatory if selection[key] in {None, ""})
    if missing:
        raise ValueError("MANDATORY_SCORING_SELECTIONS_MISSING:" + ",".join(missing))
    if selection["scored_at_utc"] == "CURRENT_CLOCK":
        raise ValueError("SCORING_TIMESTAMP_MUST_NOT_USE_CURRENT_CLOCK")
    if (
        selection["canonical_row_count"] != 486757
        or selection["expected_eligible_row_count"] != 486756
        or selection["expected_excluded_row_count"] != 1
        or selection["text_ineligible_row_policy"] != "GOVERNED_EXCLUSION"
    ):
        raise ValueError("TEXT_INELIGIBLE_ROW_ACCOUNTING_MISMATCH")
    if selection["approved_for_plan_construction"] is not True:
        raise ValueError("SCORING_SELECTION_NOT_APPROVED_FOR_PLAN")
    return dict(selection)


def authorization_template(
    request: Mapping[str, Any],
    *,
    runtime_checksum: str,
    expected_run_id: str,
) -> dict[str, Any]:
    return {
        "authorization_contract": SCORING_AUTHORIZATION_CONTRACT,
        "execution_authorized": False,
        "scoring_request_identity": request["scoring_request_identity"],
        "canonical_identity": request["canonical_identity"],
        "canonical_csv_sha256": request["canonical_csv_sha256"],
        "model_package_identity": request["model_package_identity"],
        "model_name": request["model_name"],
        "model_revision": request["model_revision"],
        "expected_shared_run_id": expected_run_id,
        "approved_score_output_root": request["score_output_root"],
        "authorized_stage": AUTHORIZED_STAGE,
        "reviewed_runtime_configuration_checksum": runtime_checksum,
        "scored_at_selection_identity": request["scored_at_selection_identity"],
    }


def validate_authorization(
    authorization: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    runtime_checksum: str,
    expected_run_id: str,
    execution_required: bool,
) -> dict[str, Any]:
    if set(authorization) != AUTHORIZATION_KEYS:
        raise ValueError("SCORING_AUTHORIZATION_SCOPE_KEYS_MISMATCH")
    if authorization["authorization_contract"] != SCORING_AUTHORIZATION_CONTRACT:
        raise ValueError("SCORING_AUTHORIZATION_CONTRACT_UNSUPPORTED")
    if type(authorization["execution_authorized"]) is not bool:
        raise ValueError("SCORING_AUTHORIZATION_FLAG_MUST_BE_BOOLEAN")
    expected = authorization_template(
        request, runtime_checksum=runtime_checksum, expected_run_id=expected_run_id
    )
    for key, value in expected.items():
        if key != "execution_authorized" and authorization[key] != value:
            raise ValueError(f"SCORING_AUTHORIZATION_{key.upper()}_MISMATCH")
    if execution_required and authorization["execution_authorized"] is not True:
        raise ValueError("FINBERT_SCORING_EXECUTION_AUTHORIZATION_REQUIRED")
    return {
        "authorization_identity": logical_identity(authorization),
        "execution_authorized": authorization["execution_authorized"],
    }


def deterministic_run_id(request: Mapping[str, Any], runtime_checksum: str) -> str:
    return "finbert-" + logical_identity({
        "request": request["scoring_request_identity"],
        "runtime": runtime_checksum,
        "canonical": request["canonical_identity"],
        "model_package": request["model_package_identity"],
    })[:24]


def deterministic_chunk_identity(
    *,
    canonical_identity: str,
    canonical_csv_sha256: str,
    model_package_identity: str,
    algorithm_contract: str,
    text_selection_contract: str,
    maximum_sequence_length: int,
    inference_batch_size: int,
    chunk_size: int,
    start: int,
    stop: int,
    scored_at_selection_identity: str,
    expected_eligible_rows: int,
    output_schema_identity: str,
) -> str:
    return logical_identity({
        "canonical_identity": canonical_identity,
        "canonical_csv_sha256": canonical_csv_sha256,
        "model_package_identity": model_package_identity,
        "algorithm_contract": algorithm_contract,
        "text_selection_contract": text_selection_contract,
        "maximum_sequence_length": maximum_sequence_length,
        "inference_batch_size": inference_batch_size,
        "chunk_size": chunk_size,
        "start": start,
        "stop": stop,
        "scored_at_selection_identity": scored_at_selection_identity,
        "expected_eligible_rows": expected_eligible_rows,
        "output_schema_identity": output_schema_identity,
    })


def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "scoring_request_contract", "canonical_identity",
        "canonical_csv_sha256", "model_package_identity",
        "model_package_root", "model_name", "model_revision",
        "score_output_root", "scored_at_selection_identity",
        "scoring_request_identity",
    }
    if set(request) != required:
        raise ValueError("SCORING_REQUEST_SCOPE_MISMATCH")
    if request["scoring_request_contract"] != SCORING_REQUEST_CONTRACT:
        raise ValueError("SCORING_REQUEST_CONTRACT_UNSUPPORTED")
    if not FULL_REVISION.fullmatch(str(request["model_revision"])):
        raise ValueError("MODEL_REVISION_MUST_BE_FULL_IMMUTABLE_SHA")
    for key in ("model_package_root", "score_output_root"):
        if not Path(request[key]).is_absolute():
            raise ValueError(f"{key.upper()}_MUST_BE_ABSOLUTE")
    identity_payload = {
        key: value for key, value in request.items()
        if key != "scoring_request_identity"
    }
    if logical_identity(identity_payload) != request["scoring_request_identity"]:
        raise ValueError("SCORING_REQUEST_IDENTITY_MISMATCH")
    package = Path(request["model_package_root"]).resolve()
    output = Path(request["score_output_root"]).resolve()
    if output == package or package in output.parents or output in package.parents:
        raise ValueError("SCORING_OUTPUT_ALIASES_MODEL_PACKAGE")
    return dict(request)


def execute_authorized_boundary(
    *,
    request: Mapping[str, Any],
    runtime_checksum: str,
    authorization: Mapping[str, Any],
    plan_only: bool,
    validate_package: Callable[[], Any],
    persist_resource_request: Callable[[], Any],
    acquire_lease: Callable[[], Any],
    activate_source: Callable[[], Any],
    activate_model: Callable[[], Any],
) -> dict[str, Any]:
    validate_request(request)
    run_id = deterministic_run_id(request, runtime_checksum)
    auth = validate_authorization(
        authorization, request, runtime_checksum=runtime_checksum,
        expected_run_id=run_id, execution_required=not plan_only,
    )
    package = validate_package()
    if plan_only:
        return {
            "status": "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION",
            "run_id": run_id,
            "model_package_identity": package["package_logical_identity"],
            "authorization_identity": auth["authorization_identity"],
        }
    persist_resource_request()
    lease = acquire_lease()
    try:
        source = activate_source()
        model = activate_model()
        return {
            "status": "AUTHORIZED_EXECUTION_BOUNDARY_REACHED",
            "run_id": run_id, "lease": lease, "source": source, "model": model,
        }
    except BaseException:
        close = getattr(lease, "close_failure", None)
        if close:
            close()
        raise


def _source_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"MODEL_PACKAGE_SOURCE_FILE_INVALID:{name}")
        inventory.append(_file_record(path))
    return inventory


def _file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)}


def _by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row["name"] == name)
