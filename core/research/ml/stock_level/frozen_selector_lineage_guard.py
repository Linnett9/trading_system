from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.artifact_lineage import (
    INSUFFICIENT_EVIDENCE,
    NOT_APPLICABLE,
    VERIFIED_STRICT_OOS,
    verify_lineage_graph,
)
from core.research.ml.dataset_build_manifest import (
    PERMITTED_BLOCKED,
    PERMITTED_DIAGNOSTIC,
    PERMITTED_PROMOTION,
    PERMITTED_RESEARCH,
    STATUS_CURRENT,
    STATUS_LEGACY_NO_MANIFEST,
    STATUS_UNVERIFIED,
    check_dataset_lineage,
    dataset_manifest_path,
    normalize_intended_use,
)
from core.research.ml.selector_dataset_lineage import verify_dataset_lineage_manifest
from core.research.ml.stock_level.selector_dataset import (
    frozen_selector_dataset_lineage_expectation,
)


FROZEN_SELECTOR_COMBINED_GUARD_VERSION = "frozen_selector_dataset_combined_lineage_guard_v1"
SELECTOR_LINEAGE_VERIFIED = "VERIFIED"

_PERMITTED_ORDER = {
    PERMITTED_BLOCKED: 0,
    PERMITTED_DIAGNOSTIC: 1,
    PERMITTED_RESEARCH: 2,
    PERMITTED_PROMOTION: 3,
}


def check_frozen_selector_dataset_lineage(
    *,
    dataset_root: Path,
    intended_use: str = PERMITTED_RESEARCH,
    expected_parents: Mapping[str, Any] | None = None,
    artifact_manifest_path: Path | None = None,
    require_artifact_lineage: bool = False,
    expected_artifact_kind: str = "BOUNDED_SELECTOR_PREDICTION",
) -> dict[str, Any]:
    intended = normalize_intended_use(intended_use)
    selector_manifest_path = dataset_root / "manifest.json"
    selector_manifest = _read_json_object(selector_manifest_path)
    expected = (
        frozen_selector_dataset_lineage_expectation(selector_manifest)
        if selector_manifest is not None
        else {}
    )
    expected.update(dict(expected_parents or {}))
    rows_path = dataset_root / "rows.parquet"
    generic = check_dataset_lineage(
        dataset_path=rows_path,
        manifest_path=dataset_manifest_path(rows_path),
        expected=expected,
        intended_use=intended,
    )
    selector = _selector_lineage_status(selector_manifest_path, dataset_root)
    artifact = _artifact_lineage_status(
        artifact_manifest_path,
        intended_use=intended,
        require_artifact_lineage=require_artifact_lineage,
        expected_artifact_kind=expected_artifact_kind,
    )
    permitted = _combined_permitted_use(
        _generic_selector_permitted_use(generic),
        selector.get("permitted_use"),
        artifact.get("permitted_use"),
    )
    promotion_eligible = (
        generic.get("status") == STATUS_CURRENT
        and selector.get("status") == SELECTOR_LINEAGE_VERIFIED
        and artifact.get("status") in {VERIFIED_STRICT_OOS, NOT_APPLICABLE}
        and _PERMITTED_ORDER[permitted] >= _PERMITTED_ORDER[PERMITTED_PROMOTION]
    )
    use_authorized = _PERMITTED_ORDER[permitted] >= _PERMITTED_ORDER[intended]
    blocking_reasons = _blocking_reasons(generic, selector, artifact)
    if not use_authorized:
        blocking_reasons.append(f"INTENDED_USE_NOT_PERMITTED:{intended}")
    blocking_reasons = sorted(set(blocking_reasons))
    return {
        "combined_guard_version": FROZEN_SELECTOR_COMBINED_GUARD_VERSION,
        "dataset_root": str(dataset_root),
        "dataset_path": str(rows_path),
        "generic_dataset_status": generic.get("status"),
        "selector_lineage_status": selector.get("status"),
        "artifact_lineage_status": artifact.get("status"),
        "generic_dataset": generic,
        "selector_lineage": selector,
        "artifact_lineage": artifact,
        "intended_use": intended,
        "permitted_use": permitted,
        "use_authorized": use_authorized,
        "blocking_reasons": blocking_reasons,
        "changed_parents": generic.get("changed_parents", []),
        "missing_parents": generic.get("missing_parents", []),
        "authority_versions": _authority_versions(
            generic_manifest_path=dataset_manifest_path(rows_path),
            expected=expected,
            selector_manifest=selector_manifest,
        ),
        "promotion_eligible": promotion_eligible,
        "promotion": {
            "promotion_eligible": promotion_eligible,
            "blocking_reasons": blocking_reasons,
        },
        "diagnostic_label_required": permitted == PERMITTED_DIAGNOSTIC,
        "dataset_rebuilt": False,
        "dataset_modified": False,
        "source_modified": False,
    }


def enforce_frozen_selector_dataset_lineage(
    *,
    dataset_root: Path,
    intended_use: str = PERMITTED_RESEARCH,
    expected_parents: Mapping[str, Any] | None = None,
    artifact_manifest_path: Path | None = None,
    require_artifact_lineage: bool = False,
    expected_artifact_kind: str = "BOUNDED_SELECTOR_PREDICTION",
) -> dict[str, Any]:
    report = check_frozen_selector_dataset_lineage(
        dataset_root=dataset_root,
        intended_use=intended_use,
        expected_parents=expected_parents,
        artifact_manifest_path=artifact_manifest_path,
        require_artifact_lineage=require_artifact_lineage,
        expected_artifact_kind=expected_artifact_kind,
    )
    if not report["use_authorized"]:
        reasons = ",".join(report["blocking_reasons"])
        raise RuntimeError(
            "Frozen selector dataset lineage guard blocked "
            f"{dataset_root}: permitted_use={report['permitted_use']} "
            f"intended_use={report['intended_use']} reasons={reasons}"
        )
    return report


def _selector_lineage_status(manifest_path: Path, dataset_root: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "status": STATUS_LEGACY_NO_MANIFEST,
            "reasons": ["SELECTOR_DATASET_MANIFEST_MISSING"],
            "permitted_use": PERMITTED_DIAGNOSTIC,
            "manifest_path": str(manifest_path),
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": STATUS_UNVERIFIED,
            "reasons": ["SELECTOR_DATASET_MANIFEST_UNREADABLE"],
            "permitted_use": PERMITTED_DIAGNOSTIC,
            "manifest_path": str(manifest_path),
        }
    try:
        verified = verify_dataset_lineage_manifest(
            manifest_path,
            dataset_root=dataset_root,
        )
    except ValueError as exc:
        return {
            "status": STATUS_UNVERIFIED,
            "reasons": ["SELECTOR_DATASET_LINEAGE_UNVERIFIED", str(exc)],
            "permitted_use": PERMITTED_DIAGNOSTIC,
            "manifest_path": str(manifest_path),
        }
    preflight = payload.get("frozen_preflight") if isinstance(payload, Mapping) else None
    if isinstance(preflight, Mapping) and preflight.get("status") not in (None, "READY"):
        return {
            "status": str(preflight.get("status")),
            "reasons": list(preflight.get("blockers") or ("FROZEN_PREFLIGHT_NOT_READY",)),
            "permitted_use": PERMITTED_DIAGNOSTIC,
            "manifest_path": str(manifest_path),
            "verification": verified,
            "frozen_preflight": dict(preflight),
        }
    return {
        "status": SELECTOR_LINEAGE_VERIFIED,
        "reasons": [],
        "permitted_use": PERMITTED_PROMOTION,
        "manifest_path": str(manifest_path),
        "verification": verified,
        "frozen_preflight": dict(preflight) if isinstance(preflight, Mapping) else None,
    }


def _artifact_lineage_status(
    manifest_path: Path | None,
    *,
    intended_use: str,
    require_artifact_lineage: bool,
    expected_artifact_kind: str,
) -> dict[str, Any]:
    if manifest_path is None:
        status = INSUFFICIENT_EVIDENCE if require_artifact_lineage else NOT_APPLICABLE
        return {
            "status": status,
            "reasons": ["ARTIFACT_LINEAGE_REQUIRED"] if require_artifact_lineage else [],
            "permitted_use": PERMITTED_DIAGNOSTIC if require_artifact_lineage else PERMITTED_PROMOTION,
            "manifest_path": None,
        }
    result = verify_lineage_graph(
        manifest_path,
        expected_artifact_kind=expected_artifact_kind,
        require_promotion_grade=intended_use == PERMITTED_PROMOTION,
    )
    reasons = sorted(
        set(result.get("verification_reasons") or ())
        | set((result.get("promotion") or {}).get("blocking_reasons") or ())
    )
    promotion_eligible = bool((result.get("promotion") or {}).get("promotion_eligible"))
    return {
        "status": result.get("verification_status"),
        "reasons": reasons,
        "permitted_use": PERMITTED_PROMOTION if promotion_eligible else PERMITTED_DIAGNOSTIC,
        "manifest_path": str(manifest_path),
        "verification": result,
    }


def _combined_permitted_use(*values: Any) -> str:
    normalized = [
        str(value)
        for value in values
        if str(value) in _PERMITTED_ORDER
    ]
    if not normalized:
        return PERMITTED_BLOCKED
    return min(normalized, key=lambda value: _PERMITTED_ORDER[value])


def _generic_selector_permitted_use(generic: Mapping[str, Any]) -> str:
    if generic.get("status") == STATUS_UNVERIFIED:
        return PERMITTED_DIAGNOSTIC
    return str(generic.get("permitted_use") or PERMITTED_BLOCKED)


def _blocking_reasons(
    generic: Mapping[str, Any],
    selector: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if generic.get("status") != STATUS_CURRENT:
        reasons.extend(str(value) for value in generic.get("reasons", ()) or ())
    if selector.get("status") != SELECTOR_LINEAGE_VERIFIED:
        reasons.extend(str(value) for value in selector.get("reasons", ()) or ())
    if artifact.get("status") not in {VERIFIED_STRICT_OOS, NOT_APPLICABLE}:
        reasons.extend(str(value) for value in artifact.get("reasons", ()) or ())
    return reasons


def _authority_versions(
    *,
    generic_manifest_path: Path,
    expected: Mapping[str, Any],
    selector_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    generic = _read_json_object(generic_manifest_path) or {}
    fields = (
        "canonical_price_authority_version",
        "universe_authority_version",
        "identity_authority_version",
        "corporate_action_authority_version",
        "market_calendar_authority_version",
        "target_contract_version",
        "feature_code_version",
        "label_code_version",
        "configuration_hash",
    )
    return {
        "recorded": {field: generic.get(field) for field in fields},
        "expected": {field: expected.get(field) for field in fields},
        "selector_manifest": {
            "daily_stock_spine_identity": (selector_manifest or {}).get("daily_stock_spine_identity"),
            "daily_stock_spine_checksum": (selector_manifest or {}).get("daily_stock_spine_checksum"),
            "daily_feature_store_identity": (selector_manifest or {}).get("daily_feature_store_identity"),
            "daily_feature_store_checksum": (selector_manifest or {}).get("daily_feature_store_checksum"),
            "symbol_registry_identity": (selector_manifest or {}).get("symbol_registry_identity"),
            "symbol_registry_checksum": (selector_manifest or {}).get("symbol_registry_checksum"),
            "target_contract": (selector_manifest or {}).get("target_contract")
            or (selector_manifest or {}).get("economic_target_id"),
            "target_contract_checksum": (selector_manifest or {}).get("target_contract_checksum")
            or (selector_manifest or {}).get("target_registry_entry_checksum"),
        },
    }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None
