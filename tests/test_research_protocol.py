from __future__ import annotations

import copy

import pytest

from core.research.ml.research_protocol import (
    ResearchProtocolError,
    canonical_json,
    date_domain_contract,
    evaluate_research_protocol,
    final_audit_access_event,
    fold_identity,
    frozen_development_history,
    panel_identity,
    promotion_governance,
    research_protocol,
    search_budget_policy,
    validate_frozen_history_amendment,
    verify_promotion_governance,
    verify_research_protocol_result,
)


def _protocol(**overrides):
    values = {
        "protocol_id": "P1", "protocol_version": "1.0",
        "hypothesis_family": "selector", "eligible_model_families": ["ridge", "rf"],
        "development_history_start": "2020-01-01", "development_history_end": "2023-12-31",
        "development_freeze_timestamp": "2024-01-02T00:00:00Z",
        "inner_fold_specification_identity": "fold_spec_v1", "purging_horizon": 5,
        "embargo_sessions": 1, "multi_regime_panel_identity": "panel_v1",
        "cost_model_panel_identity": "cost_v1", "portfolio_policy_panel_identity": "policy_v1",
        "primary_promotion_metrics": ["after_cost_return"], "secondary_diagnostic_metrics": ["ndcg"],
        "final_audit_period_start": "2024-01-01", "final_audit_period_end": "2024-12-31",
        "final_audit_population_identity": "audit_population_v1",
        "permitted_final_audit_access_count": 1, "search_budget_policy_identity": "budget_v1",
        "source_commit": "abc",
    }
    values.update(overrides)
    return research_protocol(**values)


def _freeze(**overrides):
    values = {
        "protocol_id": "P1", "protocol_version": "1.0", "dataset_identity": "development_v1",
        "row_population_checksum": "rows1", "decision_date_checksum": "dates1",
        "target_contract_checksum": "target1",
        "maximum_available_source_timestamp": "2024-01-01T00:00:00Z",
        "freeze_timestamp": "2024-01-02T00:00:00Z", "source_commit": "abc",
        "allowed_correction_policy": "new protocol version and amendment",
        "amendment_history_identity": "amendments_v1",
    }
    values.update(overrides)
    return frozen_development_history(**values)


def _domains(**overrides):
    rows = [
        {"date_identity": "2023-01-01", "domain": "TRAINING", "target_maturity_date": "2023-01-05"},
        {"date_identity": "2023-06-01", "domain": "INNER_VALIDATION"},
        {"date_identity": "2023-12-01", "domain": "DEVELOPMENT_EVALUATION"},
        {"date_identity": "2024-06-01", "domain": "FINAL_AUDIT"},
    ]
    if overrides:
        rows = overrides["rows"]
    return date_domain_contract("P1", rows)


def _fold(**overrides):
    values = {
        "fold_id": "F1", "protocol_id": "P1",
        "training_dates": ["2023-01-01", "2023-01-02"],
        "training_target_maturity_dates": ["2023-01-03", "2023-01-04"],
        "validation_dates": ["2023-02-01"], "purge_dates": [], "embargo_dates": ["2023-01-31"],
    }
    values.update(overrides)
    return fold_identity(**values)


def _panel(**overrides):
    values = {
        "panel_id": "panel_v1", "protocol_id": "P1",
        "panel_dates": ["2023-03-01", "2023-09-01"],
        "selection_rule_identity": "calendar_regimes_v1",
    }
    values.update(overrides)
    return panel_identity(**values)


def _budget(**overrides):
    values = {
        "policy_id": "budget_v1", "maximum_model_families": 2,
        "maximum_hyperparameter_configurations": 3, "maximum_seeds_per_configuration": 2,
        "maximum_total_material_trials": 3, "maximum_campaign_extensions": 1,
        "maximum_final_audit_accesses": 1, "permitted_extension_reasons": ["infrastructure_failure"],
        "authorised_approver_identity": "research_lead", "campaign_close_condition": "budget or rule",
        "shared_hypothesis_budget": 3,
    }
    values.update(overrides)
    return search_budget_policy(**values)


def _accounting(count=3, valid=True):
    trials = [{
        "trial_id": f"T{i}", "hypothesis_id": "H1", "search_campaign_id": "C1",
        "terminal_status": "COMPLETED", "counts_as_material_search": True,
    } for i in range(1, count + 1)]
    return {
        "valid": valid, "effective_search_count": count if valid else None,
        "counts": {"material_effective_search_count": count},
        "logical_trials": trials, "logical_result_checksum": "accounting",
    }


def _audit(**overrides):
    values = {
        "access_event_id": "A1", "protocol_id": "P1", "hypothesis_id": "H1",
        "campaign_id": "C1", "requesting_experiment_or_trial_id": "T1",
        "requester_role": "research_lead", "access_timestamp": "2025-01-01T00:00:00Z",
        "final_audit_dataset_identity": "audit_population_v1", "purpose": "final audit",
        "result_path": "synthetic/audit.json", "model_or_policy_changed_afterward": False,
        "access_outcome": "AUTHORIZED_AUDIT", "source_commit": "abc",
    }
    values.update(overrides)
    return final_audit_access_event(**values)


def _readiness(**overrides):
    values = {
        "protocol": _protocol(), "date_domains": _domains(), "frozen_history": _freeze(),
        "folds": [_fold()], "panel": _panel(), "budget": _budget(),
        "search_accounting": _accounting(), "audit_events": [_audit()],
        "state_history": [
            "DRAFT", "DEVELOPMENT_FROZEN", "DEVELOPMENT_ACTIVE", "DEVELOPMENT_CLOSED",
            "FINAL_AUDIT_AUTHORIZED", "FINAL_AUDIT_COMPLETE",
        ],
    }
    values.update(overrides)
    protocol = values.pop("protocol")
    return evaluate_research_protocol(protocol, **values)


def test_valid_frozen_development_and_disjoint_final_audit():
    result = _readiness()
    assert result["status"] == "READY"
    assert result["valid"]


def test_training_validation_overlap_purge_and_embargo_violations():
    with pytest.raises(ResearchProtocolError, match="TRAINING_VALIDATION_OVERLAP"):
        _fold(training_dates=["2023-02-01"], training_target_maturity_dates=["2023-02-02"])
    with pytest.raises(ResearchProtocolError, match="PURGE_VIOLATION"):
        _fold(training_target_maturity_dates=["2023-02-02", "2023-01-04"])
    with pytest.raises(ResearchProtocolError, match="EMBARGO_VIOLATION"):
        _fold(embargo_dates=["2023-01-02"])


def test_panel_is_deterministic_and_outcome_independent():
    assert _panel()["panel_dates"] == ["2023-03-01", "2023-09-01"]
    with pytest.raises(ResearchProtocolError, match="PANEL_SELECTION_DEPENDS"):
        _panel(candidate_outcomes_used=True)


def test_final_audit_overlap_is_rejected():
    with pytest.raises(ResearchProtocolError, match="DEVELOPMENT_FINAL_AUDIT_OVERLAP"):
        _protocol(final_audit_period_start="2023-12-01")
    rows = [
        {"date_identity": "2024-06-01", "domain": "DEVELOPMENT_EVALUATION"},
    ]
    result = _readiness(date_domains=_domains(rows=rows))
    assert result["status"] == "TEMPORAL_OVERLAP"


def test_missing_budget_exact_budget_and_exceeded_budget():
    assert _readiness(budget=None)["status"] == "MISSING_SEARCH_BUDGET"
    assert _readiness()["valid"]
    exceeded = _readiness(search_accounting=_accounting(4))
    assert exceeded["status"] == "BUDGET_EXCEEDED"


def test_authorised_and_unauthorised_extensions():
    extension = {
        "additional_trials": 1, "reason": "infrastructure_failure",
        "approval_record": "approval", "authorised_at": "2024-01-01T00:00:00Z",
    }
    assert _readiness(search_accounting=_accounting(4), campaign_extensions=[extension])["valid"]
    bad = {**extension, "reason": "try more models"}
    result = _readiness(search_accounting=_accounting(4), campaign_extensions=[bad])
    assert "UNAUTHORISED_BUDGET_EXTENSION" in result["blocking_reasons"]


def test_audit_access_allowance_and_repeated_access():
    assert _readiness()["final_audit_access_count"] == 1
    result = _readiness(audit_events=[_audit(), _audit(access_event_id="A2", result_path="other")])
    assert result["status"] == "AUDIT_ACCESS_EXCEEDED"


def test_documented_infrastructure_rerun_does_not_consume_access():
    rerun = _audit(
        access_event_id="A2", access_outcome="DUPLICATE_READ",
        rerun_of_event_id="A1", infrastructure_rerun=True,
    )
    result = _readiness(audit_events=[_audit(), rerun])
    assert result["valid"]
    assert result["final_audit_access_count"] == 1


def test_identity_changing_rerun_is_contamination():
    rerun = _audit(
        access_event_id="A2", access_outcome="DUPLICATE_READ",
        rerun_of_event_id="A1", infrastructure_rerun=True,
        model_identity_unchanged=False,
    )
    result = _readiness(audit_events=[_audit(), rerun])
    assert result["status"] == "FINAL_AUDIT_CONTAMINATED"


@pytest.mark.parametrize(
    "category",
    ["HYPERPARAMETER_CHANGE", "FEATURE_SELECTION_CHANGE", "MODEL_FAMILY_CHANGE",
     "PORTFOLIO_POLICY_CHANGE", "COST_ASSUMPTION_CHANGE", "PROMOTION_THRESHOLD_CHANGE"],
)
def test_post_audit_research_changes_contaminate(category):
    event = _audit(
        access_outcome="POST_AUDIT_MODEL_CHANGE",
        model_or_policy_changed_afterward=True, change_categories=[category],
    )
    result = _readiness(audit_events=[event])
    assert result["status"] == "FINAL_AUDIT_CONTAMINATED"
    assert category in result["contamination_findings"]


def test_promotion_governance_valid_promotion_and_rejection():
    protocol, accounting = _protocol(), _accounting()
    audit_summary = _readiness()["audit_summary"]
    for decision in ("PROMOTE", "REJECT"):
        promotion = promotion_governance(
            protocol=protocol, search_accounting=accounting, audit_summary=audit_summary,
            selected_trial_id="T1", search_accounting_result_id="accounting",
            development_panel_result_identity="development_result",
            final_audit_result_identity="audit_result", cost_model_identity="cost_v1",
            portfolio_policy_identity="policy_v1", statistical_safeguard_identity="safeguards",
            dsr_evidence_identity="dsr", pbo_evidence_identity="pbo",
            decision=decision, decision_reason="synthetic", approval_record="approved",
        )
        assert promotion["valid"]
        assert promotion["decision"] == decision
        assert verify_promotion_governance(promotion)["valid"]
        changed = copy.deepcopy(promotion)
        changed["decision"] = "INVALIDATE"
        assert not verify_promotion_governance(changed)["valid"]


def test_promotion_blocks_missing_count_and_contamination():
    protocol = _protocol()
    incomplete = promotion_governance(
        protocol=protocol, search_accounting=_accounting(valid=False),
        audit_summary=_readiness()["audit_summary"], selected_trial_id="T1",
        search_accounting_result_id="x", development_panel_result_identity="dev",
        final_audit_result_identity="audit", cost_model_identity="cost_v1",
        portfolio_policy_identity="policy_v1", statistical_safeguard_identity="s",
        dsr_evidence_identity="d", pbo_evidence_identity="p", decision="PROMOTE",
        decision_reason="x", approval_record="a",
    )
    assert not incomplete["valid"]
    contaminated_summary = _readiness(
        audit_events=[_audit(model_or_policy_changed_afterward=True)]
    )["audit_summary"]
    blocked = promotion_governance(
        protocol=protocol, search_accounting=_accounting(), audit_summary=contaminated_summary,
        selected_trial_id="T1", search_accounting_result_id="x",
        development_panel_result_identity="dev", final_audit_result_identity="audit",
        cost_model_identity="cost_v1", portfolio_policy_identity="policy_v1",
        statistical_safeguard_identity="s", dsr_evidence_identity="d",
        pbo_evidence_identity="p", decision="PROMOTE", decision_reason="x", approval_record="a",
    )
    assert not blocked["valid"]


def test_development_reopened_after_audit_and_invalid_transition():
    result = _readiness(state_history=[
        "DRAFT", "DEVELOPMENT_FROZEN", "DEVELOPMENT_ACTIVE", "DEVELOPMENT_CLOSED",
        "FINAL_AUDIT_AUTHORIZED", "FINAL_AUDIT_COMPLETE", "DEVELOPMENT_ACTIVE",
    ])
    assert result["status"] == "INVALID_STATE_TRANSITION"
    result = _readiness(state_history=["DRAFT", "FINAL_AUDIT_AUTHORIZED"])
    assert result["status"] == "INVALID_STATE_TRANSITION"


def test_protocol_version_amendment_and_changed_dataset_checksum():
    previous = _freeze()
    amended = _freeze(
        protocol_version="2.0", row_population_checksum="rows2",
        prior_version_checksum=previous["frozen_history_checksum"],
        amendment_history_identity="amendments_v2",
    )
    assert validate_frozen_history_amendment(previous, amended)["valid"]
    bad = copy.deepcopy(amended)
    bad["row_population_checksum"] = previous["row_population_checksum"]
    assert not validate_frozen_history_amendment(previous, bad)["valid"]


def test_stable_json_checksum_and_verifier_mutations():
    protocol = _protocol()
    inputs = {
        "date_domains": _domains(), "frozen_history": _freeze(), "folds": [_fold()],
        "panel": _panel(), "budget": _budget(), "search_accounting": _accounting(),
        "audit_events": [_audit()], "state_history": [
            "DRAFT", "DEVELOPMENT_FROZEN", "DEVELOPMENT_ACTIVE", "DEVELOPMENT_CLOSED",
            "FINAL_AUDIT_AUTHORIZED", "FINAL_AUDIT_COMPLETE",
        ],
    }
    result = evaluate_research_protocol(protocol, **inputs)
    assert verify_research_protocol_result(protocol, result, **inputs)["valid"]
    changed_result = copy.deepcopy(result)
    changed_result["creation_metadata"]["created_at"] = "different"
    assert changed_result["logical_result_checksum"] == result["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    mutations = []
    changed = copy.deepcopy(inputs); changed["date_domains"]["rows"][0]["date_identity"] = "2022-01-01"; mutations.append(changed)
    changed = copy.deepcopy(inputs); changed["budget"]["maximum_total_material_trials"] = 2; mutations.append(changed)
    changed = copy.deepcopy(inputs); changed["audit_events"][0]["purpose"] = "changed"; mutations.append(changed)
    for changed_inputs in mutations:
        assert not verify_research_protocol_result(protocol, result, **changed_inputs)["valid"]
