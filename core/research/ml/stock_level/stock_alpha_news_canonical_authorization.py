from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REQUEST_V2 = "stock_alpha_news_canonical_materialisation_request.v2"
AUTHORIZATION_V2 = "stock_alpha_news_canonical_materialisation_authorization.v2"
SELECTION_CONTRACT = "stock_alpha_news_ingested_at_utc_selection.v1"
STAGE = "CANONICAL_CORPUS_MATERIALISATION"
NOTICE = "NOT PRODUCTION EXECUTION AUTHORIZATION"


def normalize_ingested_at_utc(value: str) -> str:
    text = str(value or "").strip()
    if not text.endswith("Z"):
        raise ValueError("INGESTED_AT_UTC_MUST_BE_EXPLICIT_UTC")
    parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("INGESTED_AT_UTC_MUST_BE_EXPLICIT_UTC")
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_v2_request(
    v1: Mapping[str, Any], *, ingested_at_utc: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if v1.get("contract_version") not in {
        "stock_alpha_news_canonical_materialisation_request.v1", REQUEST_V2,
    }:
        raise ValueError("INCOMPLETE_MATERIALISATION_REQUEST_CONTRACT")
    timestamp = normalize_ingested_at_utc(ingested_at_utc)
    required = {
        "provenance_type", "evidence_path", "evidence_contract",
        "evidence_identity", "source_field", "assembly_sha256",
        "selection_status",
    }
    if set(provenance) != required or not all(
        str(provenance[key]).strip() for key in required
    ):
        raise ValueError("INGESTED_AT_UTC_PROVENANCE_INCOMPLETE")
    if provenance["assembly_sha256"] != v1["source_assembly_checksum"]:
        raise ValueError("INGESTED_AT_UTC_PROVENANCE_ASSEMBLY_MISMATCH")
    payload = {
        **{key: value for key, value in v1.items()
           if key not in {"logical_request_identity", "notice"}},
        "contract_version": REQUEST_V2,
        "notice": NOTICE,
        "ingested_at_utc": timestamp,
        "ingested_at_utc_provenance": dict(provenance),
        "ingested_at_utc_provenance_identity": _identity(provenance),
        "execution_authorized": False,
    }
    payload["logical_request_identity"] = _identity({
        key: value for key, value in payload.items()
        if key not in {"logical_request_identity", "notice"}
    })
    return payload


def runtime_configuration(v2: Mapping[str, Any]) -> dict[str, Any]:
    validate_v2_request(v2)
    material = {
        "source_assembly_path": v2["source_assembly_path"],
        "source_metadata_path": v2["source_metadata_path"],
        "source_assembly_checksum": v2["source_assembly_checksum"],
        "canonical_output_root": v2["canonical_output_root"],
        "ingested_at_utc": v2["ingested_at_utc"],
        "ingested_at_utc_provenance_identity":
            v2["ingested_at_utc_provenance_identity"],
        "write_enabled": True, "production_validated": False,
        "execution_authorized": False,
        "execution_policy": "CPU_ONLY_NON_MODEL",
        "resource_profile": v2["resource_profile"],
        "shared_compute_run_root": v2["shared_compute_run_root"],
        "resource_ledger_path": v2["resource_ledger_path"],
        "run_registry_path": v2["run_registry_path"],
        "network_access_allowed": False,
    }
    return {
        "notice": NOTICE, "materialisation": material,
        "configuration_checksum": configuration_checksum(material),
    }


def configuration_checksum(material):
    return _identity(material)


def authorization_template(v2, runtime_checksum, expected_run_id):
    validate_v2_request(v2)
    return {
        "notice": NOTICE,
        "authorization_contract": AUTHORIZATION_V2,
        "execution_authorized": False,
        "materialisation_request_identity": v2["logical_request_identity"],
        "source_assembly_sha256": v2["source_assembly_checksum"],
        "ingested_at_utc": v2["ingested_at_utc"],
        "ingested_at_utc_provenance_identity":
            v2["ingested_at_utc_provenance_identity"],
        "expected_shared_run_id": expected_run_id,
        "approved_output_root": v2["canonical_output_root"],
        "authorized_stage": STAGE,
        "reviewed_runtime_configuration_checksum": runtime_checksum,
    }


def validate_v2_request(request):
    if request.get("contract_version") != REQUEST_V2:
        raise ValueError("INCOMPLETE_MATERIALISATION_REQUEST_CONTRACT")
    timestamp = normalize_ingested_at_utc(request.get("ingested_at_utc", ""))
    if timestamp != request.get("ingested_at_utc"):
        raise ValueError("INGESTED_AT_UTC_NOT_NORMALIZED")
    provenance = request.get("ingested_at_utc_provenance")
    required = {
        "provenance_type", "evidence_path", "evidence_contract",
        "evidence_identity", "source_field", "assembly_sha256",
        "selection_status",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("INGESTED_AT_UTC_PROVENANCE_INCOMPLETE")
    if any(not str(provenance[key]).strip() for key in required):
        raise ValueError("INGESTED_AT_UTC_PROVENANCE_INCOMPLETE")
    if provenance["assembly_sha256"] != request.get("source_assembly_checksum"):
        raise ValueError("INGESTED_AT_UTC_PROVENANCE_ASSEMBLY_MISMATCH")
    if request.get("ingested_at_utc_provenance_identity") != _identity(provenance):
        raise ValueError("INGESTED_AT_UTC_PROVENANCE_IDENTITY_MISMATCH")
    if request.get("execution_authorized") is not False:
        raise ValueError("REQUEST_MUST_NOT_AUTHORIZE_EXECUTION")
    return True


def load_authorization(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("AUTHORIZATION_OBJECT_REQUIRED")
    return payload


def validate_authorization(
    authorization, request, runtime_checksum, expected_run_id, *,
    execution_required: bool,
):
    validate_v2_request(request)
    if authorization.get("authorization_contract") != AUTHORIZATION_V2:
        raise ValueError(
            "INCOMPLETE_OR_SUPERSEDED_AUTHORIZATION_CONTRACT"
        )
    expected = authorization_template(
        request, runtime_checksum, expected_run_id
    )
    if set(authorization) != set(expected):
        raise ValueError("AUTHORIZATION_SCOPE_KEYS_MISMATCH")
    if any(
        value is None or (isinstance(value, str) and not value.strip())
        for value in authorization.values()
    ):
        raise ValueError("AUTHORIZATION_BLANK_FIELD")
    for key, value in expected.items():
        if key == "execution_authorized":
            continue
        if authorization.get(key) != value:
            raise ValueError(f"AUTHORIZATION_{key.upper()}_MISMATCH")
    if execution_required and authorization.get("execution_authorized") is not True:
        raise ValueError("PRODUCTION_AUTHORIZATION_REQUIRED")
    if not execution_required and authorization.get("execution_authorized") not in {
        False, True,
    }:
        raise ValueError("AUTHORIZATION_FLAG_INVALID")
    output = Path(authorization["approved_output_root"])
    if not output.is_absolute():
        raise ValueError("AUTHORIZATION_OUTPUT_ROOT_MUST_BE_ABSOLUTE")
    return {
        "authorization_identity": _identity(authorization),
        "execution_authorized": authorization["execution_authorized"],
    }


def timestamp_selection_template(assembly_sha256):
    return {
        "selection_contract": SELECTION_CONTRACT,
        "ingested_at_utc": "",
        "source_assembly_sha256": assembly_sha256,
        "selection_reason": "",
        "acknowledges_canonical_content_and_identity_effect": False,
        "approved_for_plan_construction": False,
        "execution_authorized": False,
    }


def _identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()
