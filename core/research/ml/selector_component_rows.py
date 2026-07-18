from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from core.research.ml.registries.io import canonical_hash


TRAINING_ROWS_CONTRACT = "selector_component_training_rows.v2"
PREDICTION_ROWS_CONTRACT = "selector_component_prediction_rows.v2"
EVALUATION_OUTCOMES_CONTRACT = "selector_component_evaluation_outcomes.v1"
TRAINING_ROLES = {"TRAINING", "FIT_VALIDATION"}
MODEL_INPUT_ROLES = {*TRAINING_ROLES, "PREDICTION"}
PROHIBITED_PREDICTION_FIELDS = {
    "target_value", "target_values", "relevance_label", "label",
    "target_maturity_timestamp", "target_maturity_timestamps",
    "target_availability_state", "maturity_status",
    "evaluation_outcome", "realized_return", "realised_return",
}
COMMON_IDENTITY_FIELDS = (
    "dataset_identity", "campaign_identity", "plan_job_identity", "model_id",
    "symbol_identity", "decision_date", "target_horizon", "fold_identity",
    "dataset_row_identity", "feature_schema_identity",
    "feature_order_checksum", "ordered_feature_values",
)
TRAINING_EVIDENCE_FIELDS = (
    "training_boundary_identity", "purge_sessions", "embargo_sessions",
    "target_contract", "target_availability_timestamp",
    "target_maturity_timestamp", "target_value",
)


def training_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    _require(row, (*COMMON_IDENTITY_FIELDS, *TRAINING_EVIDENCE_FIELDS))
    target = float(row["target_value"])
    if not math.isfinite(target):
        raise ValueError("Training target must be finite")
    boundary = _instant(row["training_boundary_identity"])
    if (
        _instant(row["target_availability_timestamp"]) > boundary
        or _instant(row["target_maturity_timestamp"]) > boundary
    ):
        raise ValueError("Training outcome is not mature at the boundary")
    row["target_value"] = target
    row["role"] = "TRAINING"
    row["contract_version"] = TRAINING_ROWS_CONTRACT
    row["logical_row_checksum"] = canonical_hash(row)
    return row


def prediction_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    _require(row, COMMON_IDENTITY_FIELDS)
    prohibited = _prediction_outcome_fields(row)
    if prohibited:
        raise ValueError(
            "Prediction row contains prohibited outcome fields: "
            + ",".join(prohibited)
        )
    row["role"] = "PREDICTION"
    row["contract_version"] = PREDICTION_ROWS_CONTRACT
    row["logical_row_checksum"] = canonical_hash(row)
    return row


def validate_model_row_roles(
    rows: Sequence[Mapping[str, Any]],
    *,
    role_field: str,
    target_fields: Sequence[str],
    require_prediction: bool = True,
) -> None:
    """Fail closed when labeled and strict-OOS populations are conflated."""
    if not rows:
        raise ValueError("Selector component rows are empty")
    identities: dict[str, str] = {}
    for row in rows:
        role = str(row.get(role_field) or "")
        if role not in MODEL_INPUT_ROLES:
            raise ValueError(f"Ambiguous selector row role: {role}")
        row_id = str(row.get("row_id") or "")
        if not row_id:
            raise ValueError("Selector dataset row identity is required")
        if row_id in identities:
            raise ValueError("Training and prediction populations overlap")
        identities[row_id] = role
        if role in TRAINING_ROLES:
            for field in target_fields:
                if field not in row:
                    raise ValueError(f"Training row missing outcome: {field}")
            if role == "FIT_VALIDATION" and not row.get(
                "training_population_identity"
            ):
                raise ValueError(
                    "FIT_VALIDATION must bind its training population"
                )
        else:
            prohibited = _prediction_outcome_fields(row)
            if prohibited:
                raise ValueError(
                    "Prediction row contains prohibited outcome fields: "
                    + ",".join(prohibited)
                )
    if not any(role == "TRAINING" for role in identities.values()):
        raise ValueError("Training population is empty")
    if (
        require_prediction
        and not any(role == "PREDICTION" for role in identities.values())
    ):
        raise ValueError("Prediction population is empty")


def prediction_row_checksum(row: Mapping[str, Any]) -> str:
    prohibited = _prediction_outcome_fields(row)
    if prohibited:
        raise ValueError(
            "Prediction row contains prohibited outcome fields: "
            + ",".join(prohibited)
        )
    return canonical_hash(dict(row))


def evaluation_outcome(
    *,
    prediction_join_identity: Mapping[str, Any],
    target_value: float,
    target_availability_timestamp: str,
    outcome_maturity_timestamp: str,
    maturity_contract: str,
    target_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    value = float(target_value)
    if not math.isfinite(value):
        raise ValueError("Evaluation outcome must be finite")
    logical = {
        "contract_version": EVALUATION_OUTCOMES_CONTRACT,
        "prediction_join_identity": dict(prediction_join_identity),
        "prediction_join_checksum": canonical_hash(prediction_join_identity),
        "target_value": value,
        "target_availability_timestamp": str(target_availability_timestamp),
        "outcome_maturity_timestamp": str(outcome_maturity_timestamp),
        "maturity_contract": str(maturity_contract),
        "target_provenance": dict(target_provenance),
        "evaluation_only": True,
        "fit_eligible": False,
        "prediction_input_eligible": False,
    }
    logical["logical_checksum"] = canonical_hash(logical)
    return logical


def join_prediction_outcomes(
    prediction_rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    join_identity,
) -> list[dict[str, Any]]:
    by_identity: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        _validate_evaluation_outcome(outcome)
        identity = str(outcome["prediction_join_checksum"])
        if identity in by_identity:
            raise ValueError("Duplicate evaluation-outcome ownership")
        by_identity[identity] = outcome
    joined = []
    for prediction in prediction_rows:
        identity_payload = join_identity(prediction)
        identity = canonical_hash(identity_payload)
        outcome = by_identity.pop(identity, None)
        if outcome is None:
            raise ValueError("Prediction is missing evaluation outcome")
        joined.append({
            "prediction": dict(prediction),
            "evaluation_outcome": dict(outcome),
        })
    if by_identity:
        raise ValueError("Evaluation outcome has no prediction owner")
    return joined


def reject_evaluation_artifact(value: Mapping[str, Any]) -> None:
    if (
        value.get("evaluation_only") is True
        or value.get("contract_version") == EVALUATION_OUTCOMES_CONTRACT
    ):
        raise ValueError("Evaluation outcomes are not model inputs")


def _prediction_outcome_fields(row: Mapping[str, Any]) -> list[str]:
    fields = {
        str(field) for field in row
        if (
            field in PROHIBITED_PREDICTION_FIELDS
            or str(field).startswith("actual_forward_return")
            or str(field).startswith("realized_")
            or str(field).startswith("realised_")
        )
    }
    return sorted(fields)


def _validate_evaluation_outcome(value: Mapping[str, Any]) -> None:
    expected = canonical_hash({
        key: item for key, item in value.items()
        if key != "logical_checksum"
    })
    if (
        value.get("contract_version") != EVALUATION_OUTCOMES_CONTRACT
        or value.get("evaluation_only") is not True
        or value.get("fit_eligible") is not False
        or value.get("prediction_input_eligible") is not False
        or value.get("logical_checksum") != expected
    ):
        raise ValueError("Invalid evaluation-outcome contract")


def _require(value: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [
        field for field in fields
        if value.get(field) in (None, "")
    ]
    if missing:
        raise ValueError(
            "Selector row identity/evidence missing: " + ",".join(missing)
        )


def _instant(value: Any) -> datetime:
    text = str(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
