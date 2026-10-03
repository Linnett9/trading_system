from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from core.research.ml.ds24.clean_v2_contracts import stable_hash


SCIENTIFIC_IDENTITY_VERSION = "DS24_CLEAN_V2_SCIENTIFIC_IDENTITY_V2"
CHECKPOINT_IDENTITY_VERSION = "DS24_CLEAN_V2_CHECKPOINT_IDENTITY_V2"
NO_MODEL_CONTROL_RESUME_IDENTITY_VERSION = (
    "DS24_CLEAN_V2_NO_MODEL_CONTROL_RESUME_IDENTITY_V1"
)

CLEAN_V2_FEATURE_AUTHORITY_HASH = (
    "ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d"
)
CLEAN_V2_TARGET_AUTHORITY_HASH = (
    "41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1"
)
CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH = (
    "4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc"
)

_LEGACY_CLEAN_V2_FIELDS = frozenset(
    {
        "clean_source_hash",
        "family",
        "feature_authority_hash",
        "model_config_hash",
        "policy_hash",
        "refit_T",
        "refit_policy_hash",
        "static_authority_bundle_sha256",
        "target_authority_hash",
        "target_contract_hash",
    }
)
_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "checkpoint_identity_version",
        "scientific_identity",
        "scientific_identity_hash",
        "operational_identity_at_creation",
    }
)
_OPERATIONAL_IDENTITY_FIELDS = frozenset(
    {
        "clean_source_hash",
        "resource_policy_id",
        "resource_policy_hash",
        "maximum_model_workers",
        "attempt_generation",
        "attempt_id",
        "host",
        "pid",
        "process_creation_time_utc",
    }
)


class CheckpointCompatibilityError(ValueError):
    """Raised when checkpoint reuse cannot be proven scientifically safe."""


def no_model_control_config(
    *,
    family: str,
    model_registry: Mapping[str, Any],
    refit_policy_id: str,
) -> dict[str, Any]:
    """Return the canonical no-fit configuration shared by worker and repair."""

    if family not in model_registry.get("controls", []):
        raise CheckpointCompatibilityError("FAMILY_IS_NOT_A_REGISTERED_CONTROL")
    return {
        "lane": "CONTROLS",
        "status": "FIXED_NO_FIT_CONTROL",
        "training": {
            "lookback_sessions": 20,
            "refit_policy": refit_policy_id,
        },
        "parameters": {},
    }


def make_no_model_control_resume_identity(
    *,
    family: str,
    model_config: Mapping[str, Any],
    feature_authority_hash: str,
    target_authority_hash: str,
    target_contract_hash: str,
    static_authority_bundle_sha256: str,
    predictor_manifest: Mapping[str, Any],
    eligibility_contract: Mapping[str, Any],
    tournament_contract: Mapping[str, Any],
    evaluation_contract_hash: str,
) -> dict[str, Any]:
    """Build the complete resume identity for a no-fit scientific control.

    Controls have no fitted estimator artifact, but their population, schedule,
    target, and evaluation identities remain scientific authority.
    """

    if feature_authority_hash != CLEAN_V2_FEATURE_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_FEATURE_AUTHORITY")
    if target_authority_hash != CLEAN_V2_TARGET_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_TARGET_AUTHORITY")
    if static_authority_bundle_sha256 != CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_STATIC_AUTHORITY_BUNDLE")
    controls = list(tournament_contract.get("lanes", {}).get("CONTROLS", []))
    if family not in controls:
        raise CheckpointCompatibilityError("FAMILY_IS_NOT_A_NO_MODEL_CONTROL")
    candidates = [
        dict(candidate)
        for candidate in tournament_contract.get("execution_budget", {}).get(
            "candidates", []
        )
        if candidate.get("family") == family
    ]
    if len(candidates) != 1:
        raise CheckpointCompatibilityError("CONTROL_SCHEDULE_AUTHORITY_MISSING")
    schedule = candidates[0]
    if (
        schedule.get("lane") != "CONTROLS"
        or int(schedule.get("expected_fit_count", -1)) != 0
        or schedule.get("sample_cap_unit") != "not_applicable_no_fit"
    ):
        raise CheckpointCompatibilityError("CONTROL_SCHEDULE_AUTHORITY_INVALID")
    control_semantics = {
        "momentum": {
            "score": "ret_60m feature value",
            "fitted_model_required": False,
            "checkpoint_required": False,
        },
        "equal_weight_no_model": {
            "score": "constant 0.0 for every eligible asset",
            "fitted_model_required": False,
            "checkpoint_required": False,
        },
    }
    if family not in control_semantics:
        raise CheckpointCompatibilityError("UNKNOWN_NO_MODEL_CONTROL_SEMANTICS")
    return {
        "identity_version": NO_MODEL_CONTROL_RESUME_IDENTITY_VERSION,
        "family": family,
        "model_config_hash": stable_hash(dict(model_config)),
        "control_semantics": control_semantics[family],
        "feature_authority_hash": feature_authority_hash,
        "predictor_order_hash": predictor_manifest.get(
            "ordered_predictor_sha256"
        ),
        "predictor_manifest_hash": stable_hash(dict(predictor_manifest)),
        "target_authority_hash": target_authority_hash,
        "target_contract_hash": target_contract_hash,
        "eligibility_contract_hash": stable_hash(dict(eligibility_contract)),
        "population_authority": eligibility_contract.get("population_authority"),
        "decision_schedule": dict(
            tournament_contract.get("execution_budget", {}).get(
                "decision_spine", {}
            )
        ),
        "control_schedule": schedule,
        "refit_policy": dict(tournament_contract.get("refit_policy", {})),
        "measurement_policy_hash": stable_hash(
            dict(tournament_contract.get("measurement_policy", {}))
        ),
        "evaluation_contract_hash": evaluation_contract_hash,
        "static_authority_bundle_sha256": static_authority_bundle_sha256,
    }


def validate_no_model_control_resume_identity(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Fail closed unless a control resume identity is exactly authoritative."""

    if observed.get("identity_version") != NO_MODEL_CONTROL_RESUME_IDENTITY_VERSION:
        raise CheckpointCompatibilityError(
            "NO_MODEL_CONTROL_RESUME_IDENTITY_VERSION_MISMATCH"
        )
    if observed.get("feature_authority_hash") != CLEAN_V2_FEATURE_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_FEATURE_AUTHORITY")
    if observed.get("target_authority_hash") != CLEAN_V2_TARGET_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_TARGET_AUTHORITY")
    if (
        observed.get("static_authority_bundle_sha256")
        != CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH
    ):
        raise CheckpointCompatibilityError("NON_CLEAN_V2_STATIC_AUTHORITY_BUNDLE")
    semantics = observed.get("control_semantics")
    if not isinstance(semantics, Mapping):
        raise CheckpointCompatibilityError("NO_MODEL_CONTROL_SEMANTICS_MISSING")
    if semantics.get("fitted_model_required") is not False or semantics.get(
        "checkpoint_required"
    ) is not False:
        raise CheckpointCompatibilityError("NO_MODEL_CONTROL_FIT_SEMANTICS_INVALID")
    if dict(observed) != dict(expected):
        raise CheckpointCompatibilityError(
            "NO_MODEL_CONTROL_SCIENTIFIC_IDENTITY_MISMATCH"
        )


@dataclass(frozen=True)
class CheckpointCompatibilityAssessment:
    classification: str
    reusable: bool
    checkpoint_source_hash: str
    current_source_hash: str
    scientific_identity_hash: str
    operational_differences: Mapping[str, Mapping[str, Any]]
    legacy_clean_v2_checkpoint: bool
    prior_completed_package: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def make_checkpoint_identity(
    *,
    scientific_identity: Mapping[str, Any],
    operational_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_scientific_authority(scientific_identity)
    _validate_operational_identity(operational_identity)
    scientific = dict(scientific_identity)
    return {
        "checkpoint_identity_version": CHECKPOINT_IDENTITY_VERSION,
        "scientific_identity": scientific,
        "scientific_identity_hash": stable_hash(scientific),
        "operational_identity_at_creation": dict(operational_identity),
    }


def assess_checkpoint_compatibility(
    *,
    checkpoint_identity: Mapping[str, Any],
    expected_scientific_identity: Mapping[str, Any],
    current_operational_identity: Mapping[str, Any],
    completed_refits: Sequence[str] = (),
) -> CheckpointCompatibilityAssessment:
    """Classify exact, operational-only, or prior-package compatibility."""

    expected_scientific = dict(expected_scientific_identity)
    current_operational = dict(current_operational_identity)
    _validate_scientific_authority(expected_scientific)
    _validate_operational_identity(current_operational)
    identity = dict(checkpoint_identity)
    if set(identity) == _LEGACY_CLEAN_V2_FIELDS:
        return _assess_legacy_clean_v2(
            identity=identity,
            expected_scientific=expected_scientific,
            current_operational=current_operational,
            completed_refits=completed_refits,
        )
    if set(identity) != _CHECKPOINT_IDENTITY_FIELDS:
        raise CheckpointCompatibilityError(
            "UNKNOWN_CHECKPOINT_IDENTITY_FIELDS"
        )
    if identity.get("checkpoint_identity_version") != CHECKPOINT_IDENTITY_VERSION:
        raise CheckpointCompatibilityError("CHECKPOINT_IDENTITY_VERSION_MISMATCH")
    checkpoint_scientific = identity.get("scientific_identity")
    checkpoint_operational = identity.get("operational_identity_at_creation")
    if not isinstance(checkpoint_scientific, dict) or not isinstance(
        checkpoint_operational, dict
    ):
        raise CheckpointCompatibilityError("MALFORMED_CHECKPOINT_IDENTITY")
    if identity.get("scientific_identity_hash") != stable_hash(
        checkpoint_scientific
    ):
        raise CheckpointCompatibilityError("CHECKPOINT_SCIENTIFIC_HASH_MISMATCH")
    _validate_scientific_authority(checkpoint_scientific)
    _validate_operational_identity(checkpoint_operational)
    prior_completed = _is_prior_completed_package(
        checkpoint_scientific,
        expected_scientific,
        completed_refits,
    )
    if checkpoint_scientific != expected_scientific and not prior_completed:
        raise CheckpointCompatibilityError("SCIENTIFIC_IDENTITY_MISMATCH")
    differences = _operational_differences(
        checkpoint_operational, current_operational
    )
    checkpoint_source = str(checkpoint_operational["clean_source_hash"])
    current_source = str(current_operational["clean_source_hash"])
    if prior_completed:
        classification = "PRIOR_COMPLETED_PACKAGE_CHECKPOINT_NOT_REUSED"
        reusable = False
    elif differences:
        classification = "RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE"
        reusable = True
    else:
        classification = "RESUME_EXACT_SCIENTIFIC_AND_OPERATIONAL_IDENTITY"
        reusable = True
    return CheckpointCompatibilityAssessment(
        classification=classification,
        reusable=reusable,
        checkpoint_source_hash=checkpoint_source,
        current_source_hash=current_source,
        scientific_identity_hash=stable_hash(checkpoint_scientific),
        operational_differences=differences,
        legacy_clean_v2_checkpoint=False,
        prior_completed_package=prior_completed,
    )


def _assess_legacy_clean_v2(
    *,
    identity: Mapping[str, Any],
    expected_scientific: Mapping[str, Any],
    current_operational: Mapping[str, Any],
    completed_refits: Sequence[str],
) -> CheckpointCompatibilityAssessment:
    if not identity.get("clean_source_hash"):
        raise CheckpointCompatibilityError("LEGACY_SOURCE_PROVENANCE_MISSING")
    expected_projection = _legacy_projection(expected_scientific)
    observed_projection = {
        key: identity[key] for key in expected_projection
    }
    prior_completed = False
    if observed_projection != expected_projection:
        without_refit = {
            key: value
            for key, value in observed_projection.items()
            if key != "refit_T"
        }
        expected_without_refit = {
            key: value
            for key, value in expected_projection.items()
            if key != "refit_T"
        }
        checkpoint_refit = str(observed_projection["refit_T"])
        prior_completed = bool(
            without_refit == expected_without_refit
            and checkpoint_refit in set(map(str, completed_refits))
        )
        if not prior_completed:
            raise CheckpointCompatibilityError("SCIENTIFIC_IDENTITY_MISMATCH")
    _validate_legacy_clean_v2_authority(identity)
    checkpoint_source = str(identity["clean_source_hash"])
    current_source = str(current_operational["clean_source_hash"])
    differences = (
        {}
        if checkpoint_source == current_source
        else {
            "clean_source_hash": {
                "checkpoint": checkpoint_source,
                "current": current_source,
            }
        }
    )
    if prior_completed:
        classification = "PRIOR_COMPLETED_PACKAGE_CHECKPOINT_NOT_REUSED"
        reusable = False
    elif differences:
        classification = "RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE"
        reusable = True
    else:
        classification = "RESUME_COMPATIBLE_LEGACY_CLEAN_V2_IDENTITY"
        reusable = True
    return CheckpointCompatibilityAssessment(
        classification=classification,
        reusable=reusable,
        checkpoint_source_hash=checkpoint_source,
        current_source_hash=current_source,
        scientific_identity_hash=stable_hash(expected_scientific),
        operational_differences=differences,
        legacy_clean_v2_checkpoint=True,
        prior_completed_package=prior_completed,
    )


def _legacy_projection(scientific_identity: Mapping[str, Any]) -> dict[str, Any]:
    package = scientific_identity.get("package_contract")
    if not isinstance(package, Mapping):
        raise CheckpointCompatibilityError("PACKAGE_CONTRACT_MISSING")
    return {
        "family": scientific_identity.get("family"),
        "feature_authority_hash": scientific_identity.get(
            "feature_authority_hash"
        ),
        "model_config_hash": scientific_identity.get("model_config_hash"),
        "policy_hash": package.get("policy_hash"),
        "refit_T": package.get("refit_T"),
        "refit_policy_hash": scientific_identity.get("refit_policy_hash"),
        "static_authority_bundle_sha256": scientific_identity.get(
            "static_authority_bundle_sha256"
        ),
        "target_authority_hash": scientific_identity.get(
            "target_authority_hash"
        ),
        "target_contract_hash": scientific_identity.get("target_contract_hash"),
    }


def _is_prior_completed_package(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    completed_refits: Sequence[str],
) -> bool:
    checkpoint_package = checkpoint.get("package_contract")
    expected_package = expected.get("package_contract")
    if not isinstance(checkpoint_package, Mapping) or not isinstance(
        expected_package, Mapping
    ):
        return False
    checkpoint_without_package = {
        key: value for key, value in checkpoint.items() if key != "package_contract"
    }
    expected_without_package = {
        key: value for key, value in expected.items() if key != "package_contract"
    }
    schedule_contract = expected.get("schedule_contract")
    package_hashes = (
        schedule_contract.get("package_contract_hashes", {})
        if isinstance(schedule_contract, Mapping)
        else {}
    )
    checkpoint_refit = checkpoint_package.get("refit_T")
    return bool(
        checkpoint_without_package == expected_without_package
        and checkpoint_refit in set(map(str, completed_refits))
        and package_hashes.get(checkpoint_refit)
        == stable_hash(dict(checkpoint_package))
        and checkpoint_package.get("policy_hash")
        == expected_package.get("policy_hash")
    )


def _validate_scientific_authority(identity: Mapping[str, Any]) -> None:
    if identity.get("scientific_identity_version") != SCIENTIFIC_IDENTITY_VERSION:
        raise CheckpointCompatibilityError("SCIENTIFIC_IDENTITY_VERSION_MISMATCH")
    required = {
        "family",
        "model_config_hash",
        "feature_authority_hash",
        "predictor_order_hash",
        "predictor_manifest_hash",
        "target_authority_hash",
        "target_contract_hash",
        "training_lookback_sessions",
        "training_sample_contract",
        "refit_policy_id",
        "refit_policy_hash",
        "schedule_contract",
        "schedule_hash",
        "estimator_contract",
        "preprocessing_semantics",
        "static_authority_bundle_sha256",
        "package_contract",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise CheckpointCompatibilityError(
            "SCIENTIFIC_IDENTITY_FIELDS_MISSING:" + ",".join(missing)
        )
    if identity.get("feature_authority_hash") != CLEAN_V2_FEATURE_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_FEATURE_AUTHORITY")
    if identity.get("target_authority_hash") != CLEAN_V2_TARGET_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_TARGET_AUTHORITY")
    if (
        identity.get("static_authority_bundle_sha256")
        != CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH
    ):
        raise CheckpointCompatibilityError("NON_CLEAN_V2_STATIC_AUTHORITY_BUNDLE")
    schedule = identity.get("schedule_contract")
    package = identity.get("package_contract")
    if not isinstance(schedule, Mapping) or not isinstance(package, Mapping):
        raise CheckpointCompatibilityError("SCIENTIFIC_SCHEDULE_OR_PACKAGE_MALFORMED")
    package_hashes = schedule.get("package_contract_hashes")
    refit_t = package.get("refit_T")
    if (
        not isinstance(package_hashes, Mapping)
        or package_hashes.get(refit_t) != stable_hash(dict(package))
    ):
        raise CheckpointCompatibilityError("PACKAGE_OUTSIDE_SCIENTIFIC_SCHEDULE")


def _validate_legacy_clean_v2_authority(identity: Mapping[str, Any]) -> None:
    if identity.get("feature_authority_hash") != CLEAN_V2_FEATURE_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_FEATURE_AUTHORITY")
    if identity.get("target_authority_hash") != CLEAN_V2_TARGET_AUTHORITY_HASH:
        raise CheckpointCompatibilityError("NON_CLEAN_V2_TARGET_AUTHORITY")
    if (
        identity.get("static_authority_bundle_sha256")
        != CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH
    ):
        raise CheckpointCompatibilityError("NON_CLEAN_V2_STATIC_AUTHORITY_BUNDLE")


def _validate_operational_identity(identity: Mapping[str, Any]) -> None:
    if set(identity) != _OPERATIONAL_IDENTITY_FIELDS:
        raise CheckpointCompatibilityError("UNKNOWN_OPERATIONAL_IDENTITY_FIELDS")
    if not identity.get("clean_source_hash") or not identity.get(
        "resource_policy_id"
    ):
        raise CheckpointCompatibilityError("OPERATIONAL_IDENTITY_INCOMPLETE")


def _operational_differences(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"checkpoint": checkpoint.get(key), "current": current.get(key)}
        for key in sorted(_OPERATIONAL_IDENTITY_FIELDS)
        if checkpoint.get(key) != current.get(key)
    }
