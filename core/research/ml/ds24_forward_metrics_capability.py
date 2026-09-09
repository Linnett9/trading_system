from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.ds24_metrics_only_evaluator import (
    extended_performance_metrics_contract_hash,
    extended_performance_metrics_contract_registry,
    resolved_performance_contract_v3_hash,
    validate_extended_metrics_writer_capability,
)


CAPABILITY_SCHEMA_ID = "DS24_FORWARD_METRICS_WRITER_CAPABILITY_V1"


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_forward_metrics_capability(
    *,
    family: str,
    worker_kind: str,
    worker_script: str,
    metrics_root_name: str,
    evidence_basis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = extended_performance_metrics_contract_registry()
    capability: dict[str, Any] = {
        "schema_id": CAPABILITY_SCHEMA_ID,
        "family": family,
        "worker_kind": worker_kind,
        "worker_script": worker_script,
        "metrics_root_name": metrics_root_name,
        "writer_initialised_before_first_prediction": True,
        "base_evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        "extended_contract_hash": extended_performance_metrics_contract_hash(),
        "available_per_timestamp_fields": list(registry["mandatory_per_timestamp_fields"]),
        "available_artifacts": list(registry["mandatory_artifacts"]),
        "atomic_retention_gate_available": True,
        "retention_gate_requirements": list(registry["atomic_retention_gate_requirements"]),
        "full_prediction_persistence": "disabled",
        "paper_orders": 0,
        "live_orders": 0,
        "holdout_accessed": False,
        "evidence_basis": dict(evidence_basis or {}),
    }
    validation = validate_extended_metrics_writer_capability(capability, family=family)
    capability["validation"] = validation
    capability["capability_hash"] = stable_payload_hash(
        {key: value for key, value in capability.items() if key not in {"validation", "capability_hash"}}
    )
    return capability


def write_forward_metrics_capability(path: Path, capability: Mapping[str, Any]) -> dict[str, Any]:
    family = str(capability.get("family", ""))
    validation = validate_extended_metrics_writer_capability(capability, family=family)
    if not validation.get("admitted"):
        raise RuntimeError(f"DS24_FORWARD_METRICS_CAPABILITY_INVALID:{family}:{validation.get('missing_requirements')}")
    payload = dict(capability)
    payload["validation"] = validation
    return write_json_atomic(path, payload, advisory=True)
