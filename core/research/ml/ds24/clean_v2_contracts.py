from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPOSITORY_ROOT / "config" / "ds24_clean_v2"
CLEAN_SOURCE_PATHS = (
    "core/research/ml/ds24/clean_v2_certification.py",
    "core/research/ml/ds24/clean_v2_checkpoint_compatibility.py",
    "core/research/ml/ds24/clean_v2_contracts.py",
    "core/research/ml/ds24/clean_v2_data.py",
    "core/research/ml/ds24/clean_v2_features.py",
    "core/research/ml/ds24/clean_v2_package_cache.py",
    "core/research/ml/ds24/clean_v2_resources.py",
    "core/research/ml/ds24/clean_v2_runtime.py",
    "core/research/ml/ds24/incremental_evaluator_state.py",
    "core/research/ml/ds24_metrics_only_evaluator.py",
    "core/research/ml/stock_level/stock_level_sequence_regressors.py",
    "scripts/local/ds24_clean_v2_certify.py",
    "scripts/local/ds24_clean_v2_family_worker.py",
    "scripts/local/ds24_clean_v2_mac_preflight.py",
    "scripts/local/ds24_clean_v2_monitor.py",
    "scripts/local/ds24_clean_v2_reader_preflight.py",
    "scripts/local/ds24_clean_v2_reconcile_failures.py",
    "scripts/local/ds24_clean_v2_supervisor.py",
    "scripts/local/ds24_incremental_evaluator_state.py",
)


class CleanV2ContractError(ValueError):
    """Raised when a clean DS24 authority is missing or internally inconsistent."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_source_hash(*, repository_root: Path = REPOSITORY_ROOT) -> str:
    """Return the cross-host identity of the complete CLEAN V2 runtime."""

    return stable_hash(
        {
            relative: file_sha256(repository_root / relative)
            for relative in CLEAN_SOURCE_PATHS
        }
    )


def load_contract(name: str, *, config_root: Path = CONFIG_ROOT) -> dict[str, Any]:
    if Path(name).name != name or not name.endswith(".json"):
        raise CleanV2ContractError(f"Invalid clean-v2 contract name: {name!r}")
    path = config_root / name
    if not path.is_file():
        raise CleanV2ContractError(f"Missing clean-v2 contract: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanV2ContractError(f"Unreadable clean-v2 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise CleanV2ContractError(f"Clean-v2 contract must be a JSON object: {path}")
    return payload


def logical_contract_hash(payload: Mapping[str, Any]) -> str:
    """Hash a contract while excluding its optional self-reported hash field."""

    return stable_hash({key: value for key, value in payload.items() if key != "contract_sha256"})


def validate_declared_hash(payload: Mapping[str, Any], *, label: str) -> str:
    actual = logical_contract_hash(payload)
    declared = payload.get("contract_sha256")
    if declared is not None and declared != actual:
        raise CleanV2ContractError(
            f"{label} contract hash mismatch: declared={declared}, actual={actual}"
        )
    return actual


def authority_bundle() -> dict[str, Any]:
    names = (
        "predictor_manifest.json",
        "feature_authority.json",
        "target_contract.json",
        "eligibility_contract.json",
        "model_registry.json",
        "tournament_contract.json",
        "cross_host_ownership.json",
        "prior_evidence_manifest.json",
    )
    files: dict[str, dict[str, str]] = {}
    for name in names:
        payload = load_contract(name)
        files[name] = {
            "file_sha256": file_sha256(CONFIG_ROOT / name),
            "logical_sha256": validate_declared_hash(payload, label=name),
        }
    result: dict[str, Any] = {
        "authority_id": "DS24_CLEAN_V2_STATIC_AUTHORITY_BUNDLE_V1",
        "files": files,
    }
    result["bundle_sha256"] = stable_hash(result)
    return result
