from __future__ import annotations

import json
import math
import platform
from datetime import date, datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence


PROTOCOL_CONTRACT = "research_protocol_v1"
DATE_DOMAIN_CONTRACT = "research_date_domain_v1"
FROZEN_HISTORY_CONTRACT = "frozen_development_history_v1"
FOLD_CONTRACT = "research_protocol_fold_v1"
PANEL_CONTRACT = "research_protocol_panel_v1"
BUDGET_CONTRACT = "research_budget_policy_v1"
AUDIT_EVENT_CONTRACT = "final_audit_access_event_v1"
PROMOTION_CONTRACT = "promotion_governance_v1"
READINESS_CONTRACT = "research_protocol_readiness_v1"
DOMAINS = {
    "TRAINING", "INNER_VALIDATION", "DEVELOPMENT_EVALUATION", "FINAL_AUDIT",
    "EXCLUDED_EMBARGO", "EXCLUDED_PURGE", "UNASSIGNED",
}
AUDIT_OUTCOMES = {
    "AUTHORIZED_AUDIT", "DUPLICATE_READ", "UNAUTHORIZED_ACCESS",
    "POST_AUDIT_MODEL_CHANGE", "AUDIT_BUDGET_EXCEEDED", "IDENTITY_MISMATCH", "INVALID_EVENT",
}
STATES = {
    "DRAFT", "DEVELOPMENT_FROZEN", "DEVELOPMENT_ACTIVE", "DEVELOPMENT_CLOSED",
    "FINAL_AUDIT_AUTHORIZED", "FINAL_AUDIT_COMPLETE", "PROMOTION_DECIDED", "INVALIDATED",
}
TRANSITIONS = {
    "DRAFT": {"DEVELOPMENT_FROZEN", "INVALIDATED"},
    "DEVELOPMENT_FROZEN": {"DEVELOPMENT_ACTIVE", "INVALIDATED"},
    "DEVELOPMENT_ACTIVE": {"DEVELOPMENT_CLOSED", "INVALIDATED"},
    "DEVELOPMENT_CLOSED": {"FINAL_AUDIT_AUTHORIZED", "INVALIDATED"},
    "FINAL_AUDIT_AUTHORIZED": {"FINAL_AUDIT_COMPLETE", "INVALIDATED"},
    "FINAL_AUDIT_COMPLETE": {"PROMOTION_DECIDED", "INVALIDATED"},
    "PROMOTION_DECIDED": {"INVALIDATED"},
}


class ResearchProtocolError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def research_protocol(
    protocol_id: str,
    protocol_version: str,
    *,
    hypothesis_family: str,
    eligible_model_families: Sequence[str],
    development_history_start: str,
    development_history_end: str,
    development_freeze_timestamp: str,
    inner_fold_specification_identity: str,
    purging_horizon: int,
    embargo_sessions: int,
    multi_regime_panel_identity: str,
    cost_model_panel_identity: str,
    portfolio_policy_panel_identity: str,
    primary_promotion_metrics: Sequence[str],
    secondary_diagnostic_metrics: Sequence[str],
    final_audit_period_start: str,
    final_audit_period_end: str,
    final_audit_population_identity: str,
    permitted_final_audit_access_count: int,
    search_budget_policy_identity: str,
    source_commit: str,
    final_audit_required: bool = True,
    dsr_required: bool = True,
    pbo_required: bool = True,
) -> dict[str, Any]:
    logical = {
        "contract_version": PROTOCOL_CONTRACT,
        "protocol_id": str(protocol_id),
        "protocol_version": str(protocol_version),
        "hypothesis_family": str(hypothesis_family),
        "eligible_model_families": sorted(str(value) for value in eligible_model_families),
        "development_history_start": _date(development_history_start),
        "development_history_end": _date(development_history_end),
        "development_freeze_timestamp": _timestamp(development_freeze_timestamp),
        "inner_fold_specification_identity": str(inner_fold_specification_identity),
        "purging_horizon": int(purging_horizon),
        "embargo_sessions": int(embargo_sessions),
        "multi_regime_panel_identity": str(multi_regime_panel_identity),
        "cost_model_panel_identity": str(cost_model_panel_identity),
        "portfolio_policy_panel_identity": str(portfolio_policy_panel_identity),
        "primary_promotion_metrics": list(primary_promotion_metrics),
        "secondary_diagnostic_metrics": list(secondary_diagnostic_metrics),
        "final_audit_period_start": _date(final_audit_period_start),
        "final_audit_period_end": _date(final_audit_period_end),
        "final_audit_population_identity": str(final_audit_population_identity),
        "permitted_final_audit_access_count": int(permitted_final_audit_access_count),
        "search_budget_policy_identity": str(search_budget_policy_identity),
        "source_commit": str(source_commit),
        "final_audit_required": bool(final_audit_required),
        "dsr_required": bool(dsr_required),
        "pbo_required": bool(pbo_required),
    }
    if logical["purging_horizon"] < 0 or logical["embargo_sessions"] < 0 or logical["permitted_final_audit_access_count"] < 0:
        raise ResearchProtocolError("DRAFT_ONLY", "PROTOCOL_NUMERIC_POLICY_INVALID")
    if logical["development_history_start"] > logical["development_history_end"]:
        raise ResearchProtocolError("DRAFT_ONLY", "DEVELOPMENT_PERIOD_INVALID")
    if logical["final_audit_period_start"] > logical["final_audit_period_end"]:
        raise ResearchProtocolError("FINAL_AUDIT_NOT_PROTECTED", "FINAL_AUDIT_PERIOD_INVALID")
    if logical["development_history_end"] >= logical["final_audit_period_start"]:
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "DEVELOPMENT_FINAL_AUDIT_OVERLAP")
    logical["protocol_checksum"] = canonical_hash(logical)
    return logical


def frozen_development_history(
    *,
    protocol_id: str,
    protocol_version: str,
    dataset_identity: str,
    row_population_checksum: str,
    decision_date_checksum: str,
    target_contract_checksum: str,
    maximum_available_source_timestamp: str,
    freeze_timestamp: str,
    source_commit: str,
    allowed_correction_policy: str,
    amendment_history_identity: str,
    prior_version_checksum: str | None = None,
) -> dict[str, Any]:
    logical = {
        "contract_version": FROZEN_HISTORY_CONTRACT,
        "protocol_id": str(protocol_id),
        "protocol_version": str(protocol_version),
        "dataset_identity": str(dataset_identity),
        "row_population_checksum": str(row_population_checksum),
        "decision_date_checksum": str(decision_date_checksum),
        "target_contract_checksum": str(target_contract_checksum),
        "maximum_available_source_timestamp": _timestamp(maximum_available_source_timestamp),
        "freeze_timestamp": _timestamp(freeze_timestamp),
        "source_commit": str(source_commit),
        "allowed_correction_policy": str(allowed_correction_policy),
        "amendment_history_identity": str(amendment_history_identity),
        "prior_version_checksum": prior_version_checksum,
    }
    if logical["maximum_available_source_timestamp"] > logical["freeze_timestamp"]:
        raise ResearchProtocolError("MISSING_DEVELOPMENT_FREEZE", "SOURCE_AVAILABLE_AFTER_FREEZE")
    if any(not logical[key] for key in (
        "dataset_identity", "row_population_checksum", "decision_date_checksum",
        "target_contract_checksum", "allowed_correction_policy", "amendment_history_identity",
    )):
        raise ResearchProtocolError("MISSING_DEVELOPMENT_FREEZE", "FROZEN_HISTORY_IDENTITY_MISSING")
    logical["frozen_history_checksum"] = canonical_hash(logical)
    return logical


def date_domain_contract(protocol_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalised = []
    identities = []
    for row in rows:
        identity = str(row["date_identity"])
        domain = str(row["domain"])
        if domain not in DOMAINS:
            raise ResearchProtocolError("TEMPORAL_OVERLAP", "DATE_DOMAIN_UNSUPPORTED")
        identities.append(identity)
        normalised.append({
            "date_identity": _date(identity),
            "domain": domain,
            "target_maturity_date": _date(row["target_maturity_date"]) if row.get("target_maturity_date") else None,
        })
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "DATE_IDENTITIES_NOT_UNIQUE_AND_ORDERED")
    logical = {
        "contract_version": DATE_DOMAIN_CONTRACT,
        "protocol_id": str(protocol_id),
        "rows": normalised,
        "date_population_checksum": canonical_hash(normalised),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def fold_identity(
    fold_id: str,
    protocol_id: str,
    *,
    training_dates: Sequence[str],
    training_target_maturity_dates: Sequence[str],
    validation_dates: Sequence[str],
    purge_dates: Sequence[str],
    embargo_dates: Sequence[str],
) -> dict[str, Any]:
    training = _ordered_dates(training_dates, "TRAINING")
    maturity = [_date(value) for value in training_target_maturity_dates]
    validation = _ordered_dates(validation_dates, "VALIDATION")
    purge = _ordered_dates(purge_dates, "PURGE", allow_empty=True)
    embargo = _ordered_dates(embargo_dates, "EMBARGO", allow_empty=True)
    if len(training) != len(maturity):
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "TRAINING_MATURITY_DIMENSION_MISMATCH")
    if training and validation and max(training) >= min(validation):
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "TRAINING_VALIDATION_OVERLAP")
    purge_set = set(purge)
    if any(mature >= min(validation) and train not in purge_set for train, mature in zip(training, maturity)):
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "PURGE_VIOLATION")
    if set(training) & set(embargo):
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "EMBARGO_VIOLATION")
    logical = {
        "contract_version": FOLD_CONTRACT,
        "fold_id": str(fold_id),
        "protocol_id": str(protocol_id),
        "training_dates": training,
        "training_target_maturity_dates": maturity,
        "validation_dates": validation,
        "purge_dates": purge,
        "embargo_dates": embargo,
    }
    logical["fold_checksum"] = canonical_hash(logical)
    return logical


def panel_identity(
    panel_id: str,
    protocol_id: str,
    *,
    panel_dates: Sequence[str],
    selection_rule_identity: str,
    candidate_outcomes_used: bool = False,
) -> dict[str, Any]:
    dates = _ordered_dates(panel_dates, "PANEL")
    if candidate_outcomes_used:
        raise ResearchProtocolError("FINAL_AUDIT_CONTAMINATED", "PANEL_SELECTION_DEPENDS_ON_CANDIDATE_OUTCOMES")
    logical = {
        "contract_version": PANEL_CONTRACT,
        "panel_id": str(panel_id),
        "protocol_id": str(protocol_id),
        "panel_dates": dates,
        "selection_rule_identity": str(selection_rule_identity),
        "candidate_outcomes_used": False,
    }
    logical["panel_checksum"] = canonical_hash(logical)
    return logical


def search_budget_policy(
    policy_id: str,
    *,
    maximum_model_families: int,
    maximum_hyperparameter_configurations: int,
    maximum_seeds_per_configuration: int,
    maximum_total_material_trials: int,
    maximum_campaign_extensions: int,
    maximum_final_audit_accesses: int,
    permitted_extension_reasons: Sequence[str],
    authorised_approver_identity: str,
    campaign_close_condition: str,
    shared_hypothesis_budget: int | None = None,
) -> dict[str, Any]:
    numeric = [
        maximum_model_families, maximum_hyperparameter_configurations,
        maximum_seeds_per_configuration, maximum_total_material_trials,
        maximum_campaign_extensions, maximum_final_audit_accesses,
    ]
    if any(int(value) < 0 for value in numeric) or maximum_total_material_trials < 1:
        raise ResearchProtocolError("MISSING_SEARCH_BUDGET", "SEARCH_BUDGET_INVALID")
    shared = int(shared_hypothesis_budget if shared_hypothesis_budget is not None else maximum_total_material_trials)
    logical = {
        "contract_version": BUDGET_CONTRACT,
        "policy_id": str(policy_id),
        "maximum_model_families": int(maximum_model_families),
        "maximum_hyperparameter_configurations": int(maximum_hyperparameter_configurations),
        "maximum_seeds_per_configuration": int(maximum_seeds_per_configuration),
        "maximum_total_material_trials": int(maximum_total_material_trials),
        "shared_hypothesis_budget": shared,
        "maximum_campaign_extensions": int(maximum_campaign_extensions),
        "maximum_final_audit_accesses": int(maximum_final_audit_accesses),
        "permitted_extension_reasons": sorted(str(value) for value in permitted_extension_reasons),
        "authorised_approver_identity": str(authorised_approver_identity),
        "campaign_close_condition": str(campaign_close_condition),
    }
    logical["budget_checksum"] = canonical_hash(logical)
    return logical


def final_audit_access_event(
    access_event_id: str,
    *,
    protocol_id: str,
    hypothesis_id: str,
    campaign_id: str,
    requesting_experiment_or_trial_id: str,
    requester_role: str,
    access_timestamp: str,
    final_audit_dataset_identity: str,
    purpose: str,
    result_path: str,
    model_or_policy_changed_afterward: bool,
    access_outcome: str,
    source_commit: str,
    rerun_of_event_id: str | None = None,
    infrastructure_rerun: bool = False,
    model_identity_unchanged: bool = True,
    policy_identity_unchanged: bool = True,
    population_identity_unchanged: bool = True,
    change_categories: Sequence[str] = (),
) -> dict[str, Any]:
    if access_outcome not in AUDIT_OUTCOMES:
        raise ResearchProtocolError("FINAL_AUDIT_NOT_PROTECTED", "AUDIT_OUTCOME_INVALID")
    logical = {
        "contract_version": AUDIT_EVENT_CONTRACT,
        "access_event_id": str(access_event_id),
        "protocol_id": str(protocol_id),
        "hypothesis_id": str(hypothesis_id),
        "campaign_id": str(campaign_id),
        "requesting_experiment_or_trial_id": str(requesting_experiment_or_trial_id),
        "requester_role": str(requester_role),
        "access_timestamp": _timestamp(access_timestamp),
        "final_audit_dataset_identity": str(final_audit_dataset_identity),
        "purpose": str(purpose),
        "result_path": str(result_path),
        "model_or_policy_changed_afterward": bool(model_or_policy_changed_afterward),
        "access_outcome": access_outcome,
        "source_commit": str(source_commit),
        "rerun_of_event_id": rerun_of_event_id,
        "infrastructure_rerun": bool(infrastructure_rerun),
        "model_identity_unchanged": bool(model_identity_unchanged),
        "policy_identity_unchanged": bool(policy_identity_unchanged),
        "population_identity_unchanged": bool(population_identity_unchanged),
        "change_categories": sorted(str(value) for value in change_categories),
    }
    logical["event_checksum"] = canonical_hash(logical)
    return logical


def promotion_governance(
    *,
    protocol: Mapping[str, Any],
    search_accounting: Mapping[str, Any],
    audit_summary: Mapping[str, Any],
    selected_trial_id: str,
    search_accounting_result_id: str,
    development_panel_result_identity: str | None,
    final_audit_result_identity: str | None,
    cost_model_identity: str,
    portfolio_policy_identity: str,
    statistical_safeguard_identity: str | None,
    dsr_evidence_identity: str | None,
    pbo_evidence_identity: str | None,
    decision: str,
    decision_reason: str,
    approval_record: str,
) -> dict[str, Any]:
    if decision not in {"PROMOTE", "REJECT", "CONTINUE_DEVELOPMENT", "INVALIDATE", "BLOCKED"}:
        raise ResearchProtocolError("IDENTITY_MISMATCH", "PROMOTION_DECISION_UNSUPPORTED")
    reasons = []
    if not search_accounting.get("valid"):
        reasons.append("INCOMPLETE_SEARCH_ACCOUNTING")
    if search_accounting.get("effective_search_count") is None:
        reasons.append("EFFECTIVE_SEARCH_COUNT_MISSING")
    if not development_panel_result_identity:
        reasons.append("DEVELOPMENT_METRICS_MISSING")
    if protocol.get("final_audit_required") and not final_audit_result_identity:
        reasons.append("FINAL_AUDIT_RESULT_MISSING")
    if audit_summary.get("contamination_findings"):
        reasons.append("FINAL_AUDIT_CONTAMINATED")
    if audit_summary.get("access_budget_exceeded"):
        reasons.append("AUDIT_ACCESS_EXCEEDED")
    if protocol.get("dsr_required") and not dsr_evidence_identity:
        reasons.append("DSR_EVIDENCE_MISSING")
    if protocol.get("pbo_required") and not pbo_evidence_identity:
        reasons.append("PBO_EVIDENCE_MISSING")
    trials = {row["trial_id"]: row for row in search_accounting.get("logical_trials", [])}
    if selected_trial_id not in trials:
        reasons.append("SELECTED_TRIAL_MISSING")
    valid = not reasons and decision in {"PROMOTE", "REJECT"}
    logical = {
        "contract_version": PROMOTION_CONTRACT,
        "status": "VALID" if valid else "BLOCKED",
        "valid": valid,
        "blocking_reasons": sorted(set(reasons)),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("protocol_version"),
        "hypothesis_id": trials.get(selected_trial_id, {}).get("hypothesis_id"),
        "campaign_id": trials.get(selected_trial_id, {}).get("search_campaign_id"),
        "selected_trial_id": selected_trial_id,
        "search_accounting_result_id": search_accounting_result_id,
        "effective_search_count": search_accounting.get("effective_search_count"),
        "development_panel_result_identity": development_panel_result_identity,
        "final_audit_result_identity": final_audit_result_identity,
        "final_audit_access_count": audit_summary.get("counted_accesses"),
        "cost_model_identity": cost_model_identity,
        "portfolio_policy_identity": portfolio_policy_identity,
        "statistical_safeguard_identity": statistical_safeguard_identity,
        "dsr_evidence_identity": dsr_evidence_identity,
        "pbo_evidence_identity": pbo_evidence_identity,
        "decision": decision if valid else "BLOCKED",
        "requested_decision": decision,
        "decision_reason": decision_reason,
        "approval_record": approval_record,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def evaluate_research_protocol(
    protocol: Mapping[str, Any],
    *,
    date_domains: Mapping[str, Any],
    frozen_history: Mapping[str, Any] | None,
    folds: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
    budget: Mapping[str, Any] | None,
    search_accounting: Mapping[str, Any],
    audit_events: Sequence[Mapping[str, Any]],
    state_history: Sequence[str],
    promotion: Mapping[str, Any] | None = None,
    campaign_extensions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    reasons = []
    warnings = []
    if protocol.get("protocol_checksum") != canonical_hash({key: value for key, value in protocol.items() if key != "protocol_checksum"}):
        reasons.append("PROTOCOL_CHECKSUM_MISMATCH")
    if not frozen_history:
        reasons.append("MISSING_DEVELOPMENT_FREEZE")
    elif frozen_history.get("protocol_id") != protocol.get("protocol_id"):
        reasons.append("FROZEN_HISTORY_PROTOCOL_MISMATCH")
    elif frozen_history.get("frozen_history_checksum") != canonical_hash(
        {key: value for key, value in frozen_history.items() if key != "frozen_history_checksum"}
    ):
        reasons.append("FROZEN_HISTORY_CHECKSUM_MISMATCH")
    if date_domains.get("date_population_checksum") != canonical_hash(date_domains.get("rows", [])):
        reasons.append("DATE_POPULATION_CHECKSUM_MISMATCH")
    if date_domains.get("logical_result_checksum") != canonical_hash(
        {key: value for key, value in date_domains.items() if key != "logical_result_checksum"}
    ):
        reasons.append("DATE_DOMAIN_CHECKSUM_MISMATCH")
    for fold in folds:
        if fold.get("fold_checksum") != canonical_hash({key: value for key, value in fold.items() if key != "fold_checksum"}):
            reasons.append("FOLD_CHECKSUM_MISMATCH")
    if panel.get("panel_checksum") != canonical_hash({key: value for key, value in panel.items() if key != "panel_checksum"}):
        reasons.append("PANEL_CHECKSUM_MISMATCH")
    if budget and budget.get("budget_checksum") != canonical_hash({key: value for key, value in budget.items() if key != "budget_checksum"}):
        reasons.append("BUDGET_CHECKSUM_MISMATCH")
    reasons.extend(_verify_dates(protocol, date_domains))
    reasons.extend(_verify_linkage(protocol, folds, panel, budget))
    reasons.extend(_verify_budget(budget, search_accounting, campaign_extensions))
    state_reasons = _verify_states(state_history, bool(protocol.get("final_audit_required")))
    reasons.extend(state_reasons)
    audit = _audit_summary(protocol, audit_events)
    reasons.extend(audit["blocking_reasons"])
    if promotion and not promotion.get("valid"):
        reasons.extend(promotion.get("blocking_reasons", []))
    status = _readiness_status(reasons, frozen_history, state_history)
    logical = {
        "contract_version": READINESS_CONTRACT,
        "status": status,
        "valid": not reasons and status == "READY",
        "blocking_reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "protocol_status": state_history[-1] if state_history else "DRAFT",
        "development_identity": frozen_history.get("frozen_history_checksum") if frozen_history else None,
        "fold_identity": canonical_hash([fold.get("fold_checksum") for fold in folds]),
        "panel_identity": panel.get("panel_checksum"),
        "final_audit_identity": protocol.get("final_audit_population_identity"),
        "budget_identity": budget.get("budget_checksum") if budget else None,
        "trial_counts": search_accounting.get("counts", {}),
        "effective_search_count": search_accounting.get("effective_search_count"),
        "final_audit_access_count": audit["counted_accesses"],
        "contamination_findings": audit["contamination_findings"],
        "promotion_eligibility": not reasons and state_history and state_history[-1] == "FINAL_AUDIT_COMPLETE",
        "date_population_checksum": date_domains.get("date_population_checksum"),
        "protocol_checksum": protocol.get("protocol_checksum"),
        "configuration_checksum": canonical_hash({
            "folds": [fold.get("fold_checksum") for fold in folds],
            "panel": panel.get("panel_checksum"), "budget": budget.get("budget_checksum") if budget else None,
            "state_history": list(state_history), "campaign_extensions": list(campaign_extensions),
        }),
        "audit_summary": audit,
        "promotion_checksum": promotion.get("logical_result_checksum") if promotion else None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def verify_research_protocol_result(
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    **inputs,
) -> dict[str, Any]:
    expected = evaluate_research_protocol(protocol, **inputs)
    fields = (
        "status", "blocking_reasons", "development_identity", "fold_identity",
        "panel_identity", "budget_identity", "effective_search_count",
        "final_audit_access_count", "contamination_findings",
        "configuration_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": "research_protocol_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def verify_promotion_governance(result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    expected = canonical_hash({key: value for key, value in result.items() if key != "logical_result_checksum"})
    if result.get("logical_result_checksum") != expected:
        reasons.append("PROMOTION_LOGICAL_CHECKSUM_MISMATCH")
    if result.get("decision") not in {"PROMOTE", "REJECT", "CONTINUE_DEVELOPMENT", "INVALIDATE", "BLOCKED"}:
        reasons.append("PROMOTION_DECISION_INVALID")
    if result.get("valid") and result.get("blocking_reasons"):
        reasons.append("PROMOTION_VALIDITY_INCONSISTENT")
    return {"contract_version": "promotion_governance_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def validate_frozen_history_amendment(previous: Mapping[str, Any], amended: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if previous.get("protocol_version") == amended.get("protocol_version"):
        reasons.append("PROTOCOL_VERSION_NOT_INCREMENTED")
    if previous.get("row_population_checksum") == amended.get("row_population_checksum"):
        reasons.append("AMENDMENT_RETAINED_OLD_ROW_CHECKSUM")
    if amended.get("prior_version_checksum") != previous.get("frozen_history_checksum"):
        reasons.append("AMENDMENT_PRIOR_VERSION_LINK_MISSING")
    return {"contract_version": "frozen_history_amendment_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def _verify_dates(protocol, domains):
    reasons = []
    rows = domains.get("rows", [])
    final_start, final_end = protocol["final_audit_period_start"], protocol["final_audit_period_end"]
    for row in rows:
        day, domain = row["date_identity"], row["domain"]
        inside_final = final_start <= day <= final_end
        if inside_final and domain != "FINAL_AUDIT":
            reasons.append("FINAL_AUDIT_DATE_IN_DEVELOPMENT_DOMAIN")
        if not inside_final and domain == "FINAL_AUDIT":
            reasons.append("FINAL_AUDIT_DOMAIN_OUTSIDE_PROTOCOL_PERIOD")
        if domain in {"TRAINING", "INNER_VALIDATION", "DEVELOPMENT_EVALUATION"} and day > protocol["development_history_end"]:
            reasons.append("DEVELOPMENT_DATE_AFTER_FROZEN_HISTORY")
    return reasons


def _verify_linkage(protocol, folds, panel, budget):
    reasons = []
    if any(fold.get("protocol_id") != protocol.get("protocol_id") for fold in folds):
        reasons.append("FOLD_PROTOCOL_MISMATCH")
    if panel.get("protocol_id") != protocol.get("protocol_id") or panel.get("panel_id") != protocol.get("multi_regime_panel_identity"):
        reasons.append("PANEL_IDENTITY_MISMATCH")
    if budget and budget.get("policy_id") != protocol.get("search_budget_policy_identity"):
        reasons.append("BUDGET_IDENTITY_MISMATCH")
    if any(protocol["final_audit_period_start"] <= day <= protocol["final_audit_period_end"] for day in panel.get("panel_dates", [])):
        reasons.append("FINAL_AUDIT_DATE_IN_SELECTION_PANEL")
    return reasons


def _verify_budget(budget, accounting, extensions):
    if not budget:
        return ["MISSING_SEARCH_BUDGET"]
    reasons = []
    count = accounting.get("effective_search_count")
    if not accounting.get("valid") or count is None:
        reasons.append("INCOMPLETE_SEARCH_ACCOUNTING")
        return reasons
    authorised = 0
    for extension in sorted(extensions, key=lambda row: row.get("authorised_at", "")):
        if extension.get("reason") not in budget["permitted_extension_reasons"] or not extension.get("approval_record"):
            reasons.append("UNAUTHORISED_BUDGET_EXTENSION")
        else:
            authorised += int(extension.get("additional_trials", 0))
    if len(extensions) > budget["maximum_campaign_extensions"]:
        reasons.append("CAMPAIGN_EXTENSION_LIMIT_EXCEEDED")
    if count > budget["maximum_total_material_trials"] + authorised:
        reasons.append("BUDGET_EXCEEDED")
    if count > budget["shared_hypothesis_budget"] + authorised:
        reasons.append("SHARED_HYPOTHESIS_BUDGET_EXCEEDED")
    return reasons


def _audit_summary(protocol, events):
    ordered = sorted((dict(event) for event in events), key=lambda row: (row.get("access_timestamp", ""), row.get("access_event_id", "")))
    ids = [row.get("access_event_id") for row in ordered]
    reasons, findings = [], []
    if len(ids) != len(set(ids)):
        reasons.append("DUPLICATE_AUDIT_EVENT_ID")
    counted = 0
    seen_results = set()
    for event in ordered:
        if event.get("event_checksum") != canonical_hash({key: value for key, value in event.items() if key != "event_checksum"}):
            reasons.append("AUDIT_EVENT_CHECKSUM_MISMATCH")
        if event.get("protocol_id") != protocol.get("protocol_id") or event.get("final_audit_dataset_identity") != protocol.get("final_audit_population_identity"):
            reasons.append("AUDIT_IDENTITY_MISMATCH")
        duplicate_rerun = event.get("infrastructure_rerun") and event.get("rerun_of_event_id")
        if duplicate_rerun:
            if not all(event.get(key) for key in ("model_identity_unchanged", "policy_identity_unchanged", "population_identity_unchanged")):
                findings.append("IDENTITY_CHANGING_AUDIT_RERUN")
        else:
            counted += 1
        if event.get("model_or_policy_changed_afterward"):
            findings.extend(event.get("change_categories") or ["MODEL_OR_POLICY_CHANGED_AFTER_AUDIT"])
        if event.get("access_outcome") in {"UNAUTHORIZED_ACCESS", "POST_AUDIT_MODEL_CHANGE", "IDENTITY_MISMATCH", "INVALID_EVENT"}:
            findings.append(event["access_outcome"])
        seen_results.add(event.get("result_path"))
    allowed = min(protocol["permitted_final_audit_access_count"], protocol["permitted_final_audit_access_count"])
    if counted > allowed:
        reasons.append("AUDIT_ACCESS_EXCEEDED")
    if findings:
        reasons.append("FINAL_AUDIT_CONTAMINATED")
    return {
        "event_count": len(ordered), "counted_accesses": counted,
        "access_budget_exceeded": counted > allowed,
        "contamination_findings": sorted(set(findings)),
        "blocking_reasons": sorted(set(reasons)),
        "event_population_checksum": canonical_hash(ordered),
    }


def _verify_states(history, audit_required):
    if not history:
        return ["INVALID_STATE_TRANSITION"]
    if any(state not in STATES for state in history):
        return ["INVALID_STATE_TRANSITION"]
    reasons = []
    for previous, current in zip(history, history[1:]):
        if current not in TRANSITIONS.get(previous, set()):
            reasons.append(f"INVALID_STATE_TRANSITION:{previous}->{current}")
    if audit_required and "FINAL_AUDIT_AUTHORIZED" in history and "DEVELOPMENT_CLOSED" not in history[:history.index("FINAL_AUDIT_AUTHORIZED")]:
        reasons.append("FINAL_AUDIT_BEFORE_DEVELOPMENT_CLOSED")
    return reasons


def _readiness_status(reasons, frozen, history):
    joined = " ".join(reasons)
    for needle, status in (
        ("TEMPORAL", "TEMPORAL_OVERLAP"), ("FINAL_AUDIT_DATE", "TEMPORAL_OVERLAP"),
        ("MISSING_SEARCH_BUDGET", "MISSING_SEARCH_BUDGET"), ("BUDGET_EXCEEDED", "BUDGET_EXCEEDED"),
        ("AUDIT_ACCESS_EXCEEDED", "AUDIT_ACCESS_EXCEEDED"), ("CONTAMINATED", "FINAL_AUDIT_CONTAMINATED"),
        ("INCOMPLETE_SEARCH", "INCOMPLETE_SEARCH_ACCOUNTING"), ("IDENTITY", "IDENTITY_MISMATCH"),
        ("STATE_TRANSITION", "INVALID_STATE_TRANSITION"),
    ):
        if needle in joined:
            return status
    if not frozen:
        return "MISSING_DEVELOPMENT_FREEZE"
    if reasons:
        return "FINAL_AUDIT_NOT_PROTECTED"
    if not history or history[-1] in {"DRAFT", "DEVELOPMENT_FROZEN", "DEVELOPMENT_ACTIVE"}:
        return "DRAFT_ONLY"
    return "READY"


def _ordered_dates(values, owner, allow_empty=False):
    rows = [_date(value) for value in values]
    if (not rows and not allow_empty) or rows != sorted(rows) or len(rows) != len(set(rows)):
        raise ResearchProtocolError("TEMPORAL_OVERLAP", f"{owner}_DATES_NOT_UNIQUE_AND_ORDERED")
    return rows


def _date(value):
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ResearchProtocolError("TEMPORAL_OVERLAP", "DATE_INVALID") from exc


def _timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except ValueError as exc:
        raise ResearchProtocolError("IDENTITY_MISMATCH", "TIMESTAMP_INVALID") from exc


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
