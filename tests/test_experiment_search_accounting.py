from __future__ import annotations

import copy
import json

import pyarrow.parquet as pq
import pytest

from core.research.ml.experiment_search_accounting import (
    account_experiment_search,
    canonical_json,
    hypothesis_identity,
    materialise_search_views,
    promotion_accounting,
    search_campaign_identity,
    verify_materialised_views,
    verify_promotion_accounting,
    verify_search_accounting,
)


def _hypothesis():
    return hypothesis_identity(
        "H1", "Synthetic model improves a registered metric",
        research_family="selector", primary_metric="ndcg", benchmark="control",
        continuation_rule="continue within budget", rejection_rule="reject invalid",
        registered_at="2025-01-01T00:00:00Z",
    )


def _campaign(**overrides):
    values = {
        "search_campaign_id": "C1",
        "hypothesis_id": "H1",
        "model_family": "ridge",
        "dataset_identity": "synthetic_dataset",
        "feature_schema_hash": "features",
        "target_contract_hash": "target",
        "portfolio_policy_panel": ["equal"],
        "cost_model_panel": ["fixed"],
        "risk_model_panel": ["none"],
        "training_windows": ["train"],
        "validation_windows": ["valid"],
        "planned_configuration_budget": 10,
        "seed_budget": 2,
        "campaign_start": "2025-01-01T00:00:00Z",
    }
    values.update(overrides)
    return search_campaign_identity(**values)


def _event(
    event_id,
    run_id,
    status,
    *,
    trial_id="T1",
    seed=1,
    hyperparameters=None,
    campaign_id="C1",
    hypothesis_id="H1",
    execution_kind="initial",
    material=None,
    timestamp=None,
):
    metadata = {
        "trial_id": trial_id,
        "search_campaign_id": campaign_id,
        "hypothesis_id": hypothesis_id,
        "attempt_id": run_id,
        "hyperparameters": hyperparameters or {"alpha": 1.0},
        "random_seed": seed,
        "dataset_identity": "synthetic_dataset",
        "fold_identity": "fold_1",
        "training_dates": ["2020", "2021"],
        "validation_dates": ["2022"],
        "execution_kind": execution_kind,
        "metrics_path": f"synthetic/{trial_id}.json",
        "continuation_or_rejection_reason": "synthetic reason",
    }
    if material is not None:
        metadata["material_evaluation"] = material
    return {
        "ledger_contract_version": "experiment_ledger_event_v1",
        "event_id": event_id,
        "event_timestamp": timestamp or f"2025-01-01T00:00:{int(event_id[1:]):02d}Z",
        "experiment_spec_hash": "spec",
        "experiment_run_id": run_id,
        "event_status": status,
        "artifact_kind": "MODEL",
        "canonical_model_id": "ridge",
        "requested_model_id": "ridge",
        "registry_hashes": {},
        "source_commit": "abc",
        "artifact_paths": [],
        "error_summary": None,
        "rejection_summary": None,
        "parent_experiment": None,
        "metadata": metadata,
    }


def _history(trial_id="T1", terminal="COMPLETED", **kwargs):
    digits = "".join(value for value in trial_id if value.isdigit())
    offset = int(digits or "0") * 10
    return [
        _event(f"E{offset + 1}", f"R-{trial_id}", "PLANNED", trial_id=trial_id, **kwargs),
        _event(f"E{offset + 2}", f"R-{trial_id}", "STARTED", trial_id=trial_id, **kwargs),
        _event(f"E{offset + 3}", f"R-{trial_id}", terminal, trial_id=trial_id, **kwargs),
    ]


def _account(events, campaign=None):
    hypothesis = _hypothesis()
    campaign = campaign or _campaign()
    return account_experiment_search(events, hypotheses={"H1": hypothesis}, campaigns={campaign["search_campaign_id"]: campaign})


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED"])
def test_successful_and_failed_trials_count_as_material(terminal):
    result = _account(_history(terminal=terminal))
    assert result["valid"]
    assert result["effective_search_count"] == 1
    assert result["counts"][f"{terminal.lower()}_trial_count"] == 1


def test_material_rejected_and_preexecution_rejected_are_distinguished():
    material = [_event("E1", "R1", "PLANNED"), _event("E2", "R1", "REJECTED", material=True)]
    counted = _account(material)
    assert counted["effective_search_count"] == 1
    pre = [_event("E1", "R1", "PLANNED"), _event("E2", "R1", "REJECTED", material=False)]
    excluded = _account(pre)
    assert excluded["effective_search_count"] == 0
    assert excluded["counts"]["pre_execution_rejection_count"] == 1


@pytest.mark.parametrize("execution_kind", ["retry", "resumed"])
def test_retry_and_resume_collapse_to_one_logical_trial(execution_kind):
    events = _history(terminal="FAILED")
    events += [
        _event("E4", "R2", "PLANNED", execution_kind=execution_kind),
        _event("E5", "R2", "STARTED", execution_kind=execution_kind),
        _event("E6", "R2", "COMPLETED", execution_kind=execution_kind),
    ]
    result = _account(events)
    assert result["counts"]["process_attempt_count"] == 2
    assert result["counts"]["logical_trial_count"] == 1
    assert result["effective_search_count"] == 1
    assert result["logical_trials"][0]["retry_count"] == 1


def test_distinct_seed_and_hyperparameter_are_distinct_trials():
    events = _history("T1", seed=1, hyperparameters={"alpha": 1})
    events += _history("T2", seed=2, hyperparameters={"alpha": 1})
    events += _history("T3", seed=1, hyperparameters={"alpha": 2})
    result = _account(events)
    assert result["effective_search_count"] == 3
    assert result["seed_count"] == 2
    assert result["hyperparameter_configuration_count"] == 2


def test_cache_reuse_does_not_increase_material_count():
    events = [
        _event("E1", "R1", "PLANNED", execution_kind="cache"),
        _event("E2", "R1", "STARTED", execution_kind="cache"),
        _event("E3", "R1", "SKIPPED_COMPLETE", execution_kind="cache"),
    ]
    result = _account(events)
    assert result["effective_search_count"] == 0
    assert result["counts"]["cache_reuse_count"] == 1


def test_duplicate_event_and_invalid_transition_block():
    duplicate = _history()
    duplicate.append(copy.deepcopy(duplicate[-1]))
    assert account_experiment_search(duplicate)["status"] == "INVALID_EVENT_HISTORY"
    invalid = [_event("E1", "R1", "PLANNED"), _event("E2", "R1", "COMPLETED")]
    assert account_experiment_search(invalid)["status"] == "INVALID_EVENT_HISTORY"


def test_incomplete_lifecycle_is_visible_warning():
    result = _account([_event("E1", "R1", "PLANNED"), _event("E2", "R1", "STARTED")])
    assert result["valid"]
    assert result["warnings"] == ["INCOMPLETE_LIFECYCLE:T1"]
    assert result["effective_search_count"] == 0


def test_completed_then_invalidated_remains_material():
    events = _history()
    events.append(_event("E14", "R-T1", "INVALIDATED"))
    result = _account(events)
    assert result["effective_search_count"] == 1
    assert result["counts"]["invalidated_trial_count"] == 1


@pytest.mark.parametrize(
    ("campaign_id", "hypothesis_id", "status"),
    [(None, "H1", "MISSING_CAMPAIGN_IDENTITY"), ("C1", None, "MISSING_HYPOTHESIS_IDENTITY")],
)
def test_missing_governance_identity_is_blocked(campaign_id, hypothesis_id, status):
    result = _account(_history(campaign_id=campaign_id, hypothesis_id=hypothesis_id))
    assert result["status"] == status
    assert not result["valid"]


def test_legacy_event_is_not_assigned_an_invented_campaign():
    events = _history()
    for event in events:
        event["metadata"] = {}
    result = account_experiment_search(events)
    assert result["status"] == "LEGACY_UNASSIGNED"
    assert result["effective_search_count"] is None


def test_missing_budget_exact_budget_exceeded_and_extension():
    campaign = _campaign()
    campaign.pop("planned_total_material_trials")
    assert _account(_history(), campaign)["status"] == "MISSING_SEARCH_BUDGET"
    exact = _campaign(planned_configuration_budget=1, seed_budget=1)
    assert _account(_history(), exact)["campaign_summaries"][0]["trials_remaining"] == 0
    events = _history("T1") + _history("T2")
    exceeded = _account(events, exact)
    assert exceeded["status"] == "BUDGET_EXCEEDED"
    extended = _campaign(planned_configuration_budget=1, seed_budget=1, authorised_extension=1)
    assert _account(events, extended)["valid"]


def test_trial_assigned_to_two_campaigns_blocks():
    events = _history(terminal="FAILED")
    events += [
        _event("E4", "R2", "PLANNED", campaign_id="C2"),
        _event("E5", "R2", "STARTED", campaign_id="C2"),
        _event("E6", "R2", "COMPLETED", campaign_id="C2"),
    ]
    result = account_experiment_search(events)
    assert result["status"] == "DUPLICATE_TRIAL_IDENTITY"


def test_event_and_trial_ordering_and_checksums_are_deterministic():
    events = _history("T2") + _history("T1")
    first = _account(list(reversed(events)))
    second = _account(events)
    assert [row["trial_id"] for row in first["logical_trials"]] == ["T1", "T2"]
    assert first["source_event_checksum"] == second["source_event_checksum"]
    assert first["logical_result_checksum"] == second["logical_result_checksum"]


def test_verifier_detects_status_seed_hyperparameter_and_budget_changes():
    events = _history()
    hypothesis, campaign = _hypothesis(), _campaign()
    result = account_experiment_search(events, hypotheses={"H1": hypothesis}, campaigns={"C1": campaign})
    assert verify_search_accounting(events, result, hypotheses={"H1": hypothesis}, campaigns={"C1": campaign})["valid"]
    for mutator in (
        lambda rows, camp: rows[-1].__setitem__("event_status", "FAILED"),
        lambda rows, camp: rows[-1]["metadata"].__setitem__("random_seed", 2),
        lambda rows, camp: rows[-1]["metadata"].__setitem__("hyperparameters", {"alpha": 2}),
        lambda rows, camp: camp.__setitem__("planned_total_material_trials", 1),
    ):
        changed_events, changed_campaign = copy.deepcopy(events), copy.deepcopy(campaign)
        mutator(changed_events, changed_campaign)
        verification = verify_search_accounting(
            changed_events, result, hypotheses={"H1": hypothesis}, campaigns={"C1": changed_campaign}
        )
        assert not verification["valid"]


def test_canonical_json_and_parquet_materialisation_round_trip(tmp_path):
    accounting = _account(_history("T2") + _history("T1"))
    paths = materialise_search_views(accounting, tmp_path)
    snapshot = json.loads((tmp_path / "experiment_search_snapshot.json").read_text())
    assert snapshot["logical_result_checksum"] == accounting["logical_result_checksum"]
    table = pq.read_table(paths["logical_trials"])
    assert table.column_names == [
        "trial_id", "search_campaign_id", "hypothesis_id", "model_id", "random_seed",
        "dataset_identity", "fold_identity", "terminal_status", "attempt_count",
        "retry_count", "counts_as_material_search", "hyperparameter_checksum", "trial_checksum",
    ]
    assert table["trial_id"].to_pylist() == ["T1", "T2"]
    assert verify_materialised_views(accounting, paths)["valid"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_promotion_linkage_success_and_incomplete_count_block():
    accounting = _account(_history())
    promotion = promotion_accounting(
        accounting, hypothesis_id="H1", search_campaign_id="C1",
        selected_trial_id="T1", benchmark_trial_ids=[],
        reported_effective_search_count=1, final_validation_panel_identity="panel",
        final_holdout_use_state="unused", decision_reason="synthetic",
        promotion_report_checksum="report",
    )
    assert promotion["valid"]
    assert verify_promotion_accounting(accounting, promotion)["valid"]
    blocked = promotion_accounting(
        accounting, hypothesis_id="H1", search_campaign_id="C1",
        selected_trial_id="T1", benchmark_trial_ids=[],
        reported_effective_search_count=2, final_validation_panel_identity="panel",
        final_holdout_use_state="unused", decision_reason="synthetic",
        promotion_report_checksum="report",
    )
    assert not blocked["valid"]
