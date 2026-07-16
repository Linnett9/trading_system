from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


HYPOTHESIS_CONTRACT = "experiment_hypothesis_v1"
CAMPAIGN_CONTRACT = "experiment_search_campaign_v1"
TRIAL_CONTRACT = "experiment_logical_trial_v1"
ACCOUNTING_CONTRACT = "experiment_search_accounting_v1"
PROMOTION_CONTRACT = "experiment_promotion_accounting_v1"
MATERIALISATION_CONTRACT = "experiment_search_materialisation_v1"
COUNTING_POLICY_ID = "material_trial_counting_policy_v1"
SUPPORTED_EVENT_VERSION = "experiment_ledger_event_v1"
TERMINAL = {"COMPLETED", "FAILED", "REJECTED", "SKIPPED_COMPLETE", "CANCELLED", "INVALIDATED"}
MATERIAL_TERMINAL = {"COMPLETED", "FAILED", "REJECTED", "INVALIDATED"}
STATUSES = {
    "VALID", "LEGACY_UNASSIGNED", "INCOMPLETE_LIFECYCLE", "INVALID_EVENT_HISTORY",
    "MISSING_HYPOTHESIS_IDENTITY", "MISSING_CAMPAIGN_IDENTITY", "MISSING_SEARCH_BUDGET",
    "BUDGET_EXCEEDED", "DUPLICATE_TRIAL_IDENTITY", "UNRECONSTRUCTABLE_SEARCH_COUNT",
    "UNSUPPORTED_EVENT_VERSION",
}


class SearchAccountingError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def hypothesis_identity(
    hypothesis_id: str,
    hypothesis_statement: str,
    *,
    research_family: str,
    primary_metric: str,
    benchmark: str,
    continuation_rule: str,
    rejection_rule: str,
    registered_at: str,
) -> dict[str, Any]:
    logical = {
        "contract_version": HYPOTHESIS_CONTRACT,
        "hypothesis_id": str(hypothesis_id),
        "hypothesis_statement": str(hypothesis_statement),
        "research_family": str(research_family),
        "primary_metric": str(primary_metric),
        "benchmark": str(benchmark),
        "continuation_rule": str(continuation_rule),
        "rejection_rule": str(rejection_rule),
        "registered_at": str(registered_at),
    }
    if any(not logical[key] for key in logical if key != "contract_version"):
        raise SearchAccountingError("MISSING_HYPOTHESIS_IDENTITY", "HYPOTHESIS_FIELD_MISSING")
    logical["hypothesis_checksum"] = canonical_hash(logical)
    return logical


def search_campaign_identity(
    search_campaign_id: str,
    hypothesis_id: str,
    *,
    model_family: str,
    dataset_identity: str,
    feature_schema_hash: str,
    target_contract_hash: str,
    portfolio_policy_panel: Sequence[str],
    cost_model_panel: Sequence[str],
    risk_model_panel: Sequence[str],
    training_windows: Sequence[str],
    validation_windows: Sequence[str],
    planned_configuration_budget: int,
    seed_budget: int,
    campaign_start: str,
    campaign_status: str = "OPEN",
    authorised_extension: int = 0,
    continuation_authorised: bool = False,
    continuation_reason: str | None = None,
    campaign_stop_reason: str | None = None,
) -> dict[str, Any]:
    configuration_budget = int(planned_configuration_budget)
    seeds = int(seed_budget)
    extension = int(authorised_extension)
    if configuration_budget < 1 or seeds < 1 or extension < 0:
        raise SearchAccountingError("MISSING_SEARCH_BUDGET", "CAMPAIGN_SEARCH_BUDGET_INVALID")
    if continuation_authorised and not continuation_reason:
        raise SearchAccountingError("INVALID_EVENT_HISTORY", "CAMPAIGN_CONTINUATION_REASON_MISSING")
    logical = {
        "contract_version": CAMPAIGN_CONTRACT,
        "search_campaign_id": str(search_campaign_id),
        "hypothesis_id": str(hypothesis_id),
        "model_family": str(model_family),
        "dataset_identity": str(dataset_identity),
        "feature_schema_hash": str(feature_schema_hash),
        "target_contract_hash": str(target_contract_hash),
        "portfolio_policy_panel": sorted(str(value) for value in portfolio_policy_panel),
        "cost_model_panel": sorted(str(value) for value in cost_model_panel),
        "risk_model_panel": sorted(str(value) for value in risk_model_panel),
        "training_windows": list(training_windows),
        "validation_windows": list(validation_windows),
        "planned_configuration_budget": configuration_budget,
        "seed_budget": seeds,
        "planned_total_material_trials": configuration_budget * seeds,
        "authorised_extension": extension,
        "campaign_start": str(campaign_start),
        "campaign_status": str(campaign_status),
        "continuation_authorised": bool(continuation_authorised),
        "continuation_reason": continuation_reason,
        "campaign_stop_reason": campaign_stop_reason,
    }
    if not logical["search_campaign_id"] or not logical["hypothesis_id"]:
        raise SearchAccountingError("MISSING_CAMPAIGN_IDENTITY", "CAMPAIGN_IDENTITY_MISSING")
    logical["campaign_checksum"] = canonical_hash(logical)
    return logical


def account_experiment_search(
    events: Sequence[Mapping[str, Any]],
    *,
    hypotheses: Mapping[str, Mapping[str, Any]] | None = None,
    campaigns: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        ordered = _validated_events(events)
        hypothesis_map = {str(key): dict(value) for key, value in (hypotheses or {}).items()}
        campaign_map = {str(key): dict(value) for key, value in (campaigns or {}).items()}
        attempts = _reconstruct_attempts(ordered)
        trials = _collapse_trials(attempts)
        validation = _validate_governance(trials, hypothesis_map, campaign_map)
        if validation["blocking_reasons"]:
            return _blocked(ordered, attempts, trials, validation["status"], validation["blocking_reasons"])
        campaign_rows = _campaign_accounting(trials, campaign_map)
        budget_blocks = [reason for row in campaign_rows for reason in row["blocking_reasons"]]
        if budget_blocks:
            status = "BUDGET_EXCEEDED" if any("BUDGET_EXCEEDED" in reason for reason in budget_blocks) else "MISSING_SEARCH_BUDGET"
            return _blocked(ordered, attempts, trials, status, budget_blocks, campaign_rows=campaign_rows)
        material = [trial for trial in trials if trial["counts_as_material_search"]]
        excluded = [
            {"trial_id": trial["trial_id"], "reason": trial["material_exclusion_reason"]}
            for trial in trials if not trial["counts_as_material_search"]
        ]
        counts = _counts(ordered, attempts, trials, material)
        logical = {
            "contract_version": ACCOUNTING_CONTRACT,
            "status": "VALID",
            "valid": True,
            "blocking_reasons": [],
            "warnings": sorted(set(validation["warnings"])),
            "counting_policy_id": COUNTING_POLICY_ID,
            "counting_policy_version": "1.0",
            "included_statuses": sorted(MATERIAL_TERMINAL),
            "counts": counts,
            "hypothesis_ids": sorted({trial["hypothesis_id"] for trial in trials}),
            "campaign_ids": sorted({trial["search_campaign_id"] for trial in trials}),
            "included_logical_trial_ids": [trial["trial_id"] for trial in material],
            "excluded_trials": excluded,
            "process_attempts": attempts,
            "logical_trials": trials,
            "campaign_summaries": campaign_rows,
            "hypothesis_summaries": _hypothesis_summaries(trials, hypothesis_map),
            "source_event_checksum": canonical_hash(ordered),
            "logical_trial_population_checksum": canonical_hash(
                [{"trial_id": row["trial_id"], "trial_checksum": row["trial_checksum"]} for row in trials]
            ),
            "configuration_checksum": canonical_hash(
                {"counting_policy_id": COUNTING_POLICY_ID, "campaigns": campaign_map, "hypotheses": hypothesis_map}
            ),
            "dsr_effective_search_count": len(material),
            "effective_search_count": len(material),
            "dsr_count_difference_reasons": [],
            "seed_count": len({trial["random_seed"] for trial in material}),
            "hyperparameter_configuration_count": len({trial["hyperparameter_checksum"] for trial in material}),
            "model_family_count": len({campaign_map[trial["search_campaign_id"]]["model_family"] for trial in material}),
            "dataset_count": len({trial["dataset_identity"] for trial in material}),
            "date_panel_count": len({(tuple(trial["training_dates"]), tuple(trial["validation_dates"])) for trial in material}),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except SearchAccountingError as exc:
        return _blocked([], [], [], exc.status, [exc.reason])


def promotion_accounting(
    accounting: Mapping[str, Any],
    *,
    hypothesis_id: str,
    search_campaign_id: str,
    selected_trial_id: str,
    benchmark_trial_ids: Sequence[str],
    reported_effective_search_count: int,
    final_validation_panel_identity: str,
    final_holdout_use_state: str,
    decision_reason: str,
    promotion_report_checksum: str,
) -> dict[str, Any]:
    reasons = []
    if not accounting.get("valid"):
        reasons.append("CAMPAIGN_ACCOUNTING_INCOMPLETE")
    trials = {row["trial_id"]: row for row in accounting.get("logical_trials", [])}
    selected = trials.get(selected_trial_id)
    if selected is None or selected.get("search_campaign_id") != search_campaign_id:
        reasons.append("SELECTED_TRIAL_NOT_IN_CAMPAIGN")
    if selected and selected.get("hypothesis_id") != hypothesis_id:
        reasons.append("PROMOTION_HYPOTHESIS_MISMATCH")
    if any(trial_id not in trials for trial_id in benchmark_trial_ids):
        reasons.append("BENCHMARK_TRIAL_NOT_FOUND")
    actual_count = accounting.get("effective_search_count")
    if actual_count is None or int(reported_effective_search_count) != actual_count:
        reasons.append("PROMOTION_SEARCH_COUNT_MISMATCH")
    counts = accounting.get("counts", {})
    if counts.get("failed_trial_count", 0) + counts.get("rejected_trial_count", 0) > 0:
        included = set(accounting.get("included_logical_trial_ids", []))
        material_failures = {
            row["trial_id"] for row in trials.values()
            if row["terminal_status"] in {"FAILED", "REJECTED"} and row["counts_as_material_search"]
        }
        if not material_failures.issubset(included):
            reasons.append("FAILED_OR_REJECTED_TRIALS_ABSENT")
    logical = {
        "contract_version": PROMOTION_CONTRACT,
        "status": "VALID" if not reasons else "UNRECONSTRUCTABLE_SEARCH_COUNT",
        "valid": not reasons,
        "blocking_reasons": sorted(set(reasons)),
        "hypothesis_id": hypothesis_id,
        "search_campaign_id": search_campaign_id,
        "selected_trial_id": selected_trial_id,
        "benchmark_trial_ids": sorted(benchmark_trial_ids),
        "full_material_effective_search_count": actual_count,
        "reported_effective_search_count": int(reported_effective_search_count),
        "counting_policy_id": accounting.get("counting_policy_id"),
        "final_validation_panel_identity": final_validation_panel_identity,
        "final_holdout_use_state": final_holdout_use_state,
        "decision_reason": decision_reason,
        "promotion_report_checksum": promotion_report_checksum,
        "accounting_result_checksum": accounting.get("logical_result_checksum"),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def materialise_search_views(accounting: Mapping[str, Any], directory: str | Path) -> dict[str, str]:
    if not accounting.get("valid"):
        raise SearchAccountingError("UNRECONSTRUCTABLE_SEARCH_COUNT", "INVALID_ACCOUNTING_CANNOT_BE_MATERIALISED")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SearchAccountingError("UNSUPPORTED_EVENT_VERSION", "PYARROW_UNAVAILABLE") from exc
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = {key: value for key, value in accounting.items() if key != "creation_metadata"}
    json_path = root / "experiment_search_snapshot.json"
    json_path.write_text(canonical_json(snapshot) + "\n", encoding="utf-8")
    table_specs = {
        "logical_trials": (
            accounting["logical_trials"],
            ["trial_id", "search_campaign_id", "hypothesis_id", "model_id", "random_seed", "dataset_identity",
             "fold_identity", "terminal_status", "attempt_count", "retry_count", "counts_as_material_search",
             "hyperparameter_checksum", "trial_checksum"],
        ),
        "process_attempts": (
            accounting["process_attempts"],
            ["attempt_id", "experiment_run_id", "trial_id", "search_campaign_id", "hypothesis_id",
             "terminal_status", "execution_kind", "material_evaluation", "event_count", "attempt_checksum"],
        ),
        "campaign_summaries": (
            accounting["campaign_summaries"],
            ["search_campaign_id", "hypothesis_id", "planned_total_material_trials", "authorised_extension",
             "effective_budget", "trials_attempted", "trials_completed", "trials_failed", "trials_rejected",
             "trials_remaining", "budget_exceeded", "campaign_closed", "continuation_authorised"],
        ),
        "hypothesis_summaries": (
            accounting["hypothesis_summaries"],
            ["hypothesis_id", "campaign_count", "logical_trial_count", "material_effective_search_count"],
        ),
        "effective_search_counts": (
            [{
                "counting_policy_id": accounting["counting_policy_id"],
                "effective_search_count": accounting["effective_search_count"],
                "dsr_effective_search_count": accounting["dsr_effective_search_count"],
                "logical_trial_population_checksum": accounting["logical_trial_population_checksum"],
                "source_event_checksum": accounting["source_event_checksum"],
            }],
            ["counting_policy_id", "effective_search_count", "dsr_effective_search_count",
             "logical_trial_population_checksum", "source_event_checksum"],
        ),
    }
    paths = {"json_snapshot": str(json_path)}
    for name, (rows, columns) in table_specs.items():
        normalised = [{column: _parquet_value(row.get(column)) for column in columns} for row in rows]
        table = pa.Table.from_pylist(normalised).select(columns)
        metadata = dict(table.schema.metadata or {})
        metadata[b"contract_version"] = MATERIALISATION_CONTRACT.encode()
        table = table.replace_schema_metadata(metadata)
        path = root / f"{name}.parquet"
        pq.write_table(table, path, compression="NONE", write_statistics=False)
        paths[name] = str(path)
    validation = {
        "contract_version": MATERIALISATION_CONTRACT,
        "status": "VALID",
        "row_counts": {name: len(rows) for name, (rows, _) in table_specs.items()},
        "accounting_result_checksum": accounting["logical_result_checksum"],
    }
    validation["logical_result_checksum"] = canonical_hash(validation)
    validation_path = root / "validation_report.json"
    validation_path.write_text(canonical_json(validation) + "\n", encoding="utf-8")
    paths["validation_report"] = str(validation_path)
    return paths


def verify_search_accounting(
    events: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    *,
    hypotheses: Mapping[str, Mapping[str, Any]],
    campaigns: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = account_experiment_search(events, hypotheses=hypotheses, campaigns=campaigns)
    fields = (
        "status", "counts", "source_event_checksum", "logical_trial_population_checksum",
        "configuration_checksum", "effective_search_count", "dsr_effective_search_count",
        "logical_trials", "process_attempts", "campaign_summaries", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": "experiment_search_accounting_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def verify_materialised_views(accounting: Mapping[str, Any], paths: Mapping[str, str]) -> dict[str, Any]:
    reasons = []
    try:
        import pyarrow.parquet as pq
        snapshot = json.loads(Path(paths["json_snapshot"]).read_text(encoding="utf-8"))
        expected = {key: value for key, value in accounting.items() if key != "creation_metadata"}
        if snapshot != expected:
            reasons.append("JSON_MATERIALISATION_MISMATCH")
        expected_rows = {
            "logical_trials": accounting["logical_trials"],
            "process_attempts": accounting["process_attempts"],
            "campaign_summaries": accounting["campaign_summaries"],
            "hypothesis_summaries": accounting["hypothesis_summaries"],
        }
        for name, rows in expected_rows.items():
            table = pq.read_table(paths[name])
            if table.num_rows != len(rows) or table.schema.metadata.get(b"contract_version") != MATERIALISATION_CONTRACT.encode():
                reasons.append(f"{name.upper()}_PARQUET_MISMATCH")
                continue
            columns = table.column_names
            expected_values = [
                {column: _parquet_value(row.get(column)) for column in columns}
                for row in rows
            ]
            if table.to_pylist() != expected_values:
                reasons.append(f"{name.upper()}_PARQUET_CONTENT_MISMATCH")
    except (KeyError, OSError, ValueError):
        reasons.append("MATERIALISATION_READ_FAILURE")
    return {"contract_version": "experiment_search_materialisation_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def verify_promotion_accounting(accounting: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    expected = promotion_accounting(
        accounting,
        hypothesis_id=result["hypothesis_id"],
        search_campaign_id=result["search_campaign_id"],
        selected_trial_id=result["selected_trial_id"],
        benchmark_trial_ids=result["benchmark_trial_ids"],
        reported_effective_search_count=result["reported_effective_search_count"],
        final_validation_panel_identity=result["final_validation_panel_identity"],
        final_holdout_use_state=result["final_holdout_use_state"],
        decision_reason=result["decision_reason"],
        promotion_report_checksum=result["promotion_report_checksum"],
    )
    valid = result.get("logical_result_checksum") == expected.get("logical_result_checksum")
    return {"contract_version": "experiment_promotion_accounting_verification_v1", "valid": valid, "blocking_reasons": [] if valid else ["PROMOTION_ACCOUNTING_MISMATCH"]}


def _validated_events(events):
    rows = [dict(event) for event in events]
    event_ids = [str(row.get("event_id", "")) for row in rows]
    if any(not value for value in event_ids) or len(event_ids) != len(set(event_ids)):
        raise SearchAccountingError("INVALID_EVENT_HISTORY", "DUPLICATE_OR_MISSING_EVENT_ID")
    if any(row.get("ledger_contract_version") != SUPPORTED_EVENT_VERSION for row in rows):
        raise SearchAccountingError("UNSUPPORTED_EVENT_VERSION", "LEDGER_EVENT_VERSION_UNSUPPORTED")
    allowed = {"PLANNED", "STARTED", *TERMINAL}
    if any(row.get("event_status") not in allowed for row in rows):
        raise SearchAccountingError("INVALID_EVENT_HISTORY", "EVENT_STATUS_INVALID")
    return sorted(rows, key=lambda row: (str(row.get("event_timestamp", "")), str(row["event_id"])))


def _reconstruct_attempts(events):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        run_id = str(event.get("experiment_run_id", ""))
        if not run_id:
            raise SearchAccountingError("INVALID_EVENT_HISTORY", "EXPERIMENT_RUN_ID_MISSING")
        grouped.setdefault(run_id, []).append(event)
    attempts = []
    for run_id, rows in sorted(grouped.items()):
        statuses = [row["event_status"] for row in rows]
        _validate_transitions(statuses)
        metadata = {}
        for row in rows:
            metadata.update(dict(row.get("metadata") or {}))
        terminal = statuses[-1] if statuses[-1] in TERMINAL else "INCOMPLETE"
        attempt_id = str(metadata.get("attempt_id") or run_id)
        attempt = {
            "attempt_id": attempt_id,
            "experiment_run_id": run_id,
            "trial_id": metadata.get("trial_id"),
            "search_campaign_id": metadata.get("search_campaign_id"),
            "hypothesis_id": metadata.get("hypothesis_id"),
            "terminal_status": terminal,
            "execution_kind": str(metadata.get("execution_kind", "initial")),
            "material_evaluation": bool(metadata.get("material_evaluation", any(status in {"COMPLETED", "FAILED"} for status in statuses))),
            "event_count": len(rows),
            "last_event_timestamp": str(rows[-1].get("event_timestamp", "")),
            "raw_event_ids": [row["event_id"] for row in rows],
            "model_id": rows[-1].get("canonical_model_id") or rows[-1].get("requested_model_id"),
            "hyperparameters": metadata.get("hyperparameters"),
            "random_seed": metadata.get("random_seed"),
            "dataset_identity": metadata.get("dataset_identity"),
            "fold_identity": metadata.get("fold_identity"),
            "training_dates": list(metadata.get("training_dates") or []),
            "validation_dates": list(metadata.get("validation_dates") or []),
            "source_commit": rows[-1].get("source_commit"),
            "metrics_path": metadata.get("metrics_path"),
            "continuation_or_rejection_reason": metadata.get("continuation_or_rejection_reason") or rows[-1].get("rejection_summary"),
        }
        identity = {key: value for key, value in attempt.items() if key != "raw_event_ids"}
        attempt["attempt_checksum"] = canonical_hash(identity)
        attempts.append(attempt)
    return attempts


def _validate_transitions(statuses):
    allowed = {
        "PLANNED": {"STARTED", "REJECTED", "CANCELLED"},
        "STARTED": {"COMPLETED", "FAILED", "REJECTED", "CANCELLED", "SKIPPED_COMPLETE"},
        "COMPLETED": {"INVALIDATED"},
    }
    for previous, current in zip(statuses, statuses[1:]):
        if current not in allowed.get(previous, set()):
            raise SearchAccountingError("INVALID_EVENT_HISTORY", f"INVALID_STATUS_TRANSITION:{previous}->{current}")


def _collapse_trials(attempts):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        if attempt["trial_id"]:
            trial_id = str(attempt["trial_id"])
        elif all(attempt.get(key) is not None for key in ("hyperparameters", "random_seed", "dataset_identity", "fold_identity")):
            trial_id = "trial-" + canonical_hash({
                "hyperparameters": attempt["hyperparameters"], "random_seed": attempt["random_seed"],
                "dataset_identity": attempt["dataset_identity"], "fold_identity": attempt["fold_identity"],
            })[:16].lower()
        else:
            trial_id = f"legacy-{attempt['experiment_run_id']}"
        grouped.setdefault(trial_id, []).append(attempt)
    trials = []
    campaign_owner = {}
    for trial_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row["last_event_timestamp"], row["attempt_id"]))
        campaigns = {row["search_campaign_id"] for row in rows if row["search_campaign_id"]}
        if len(campaigns) > 1:
            raise SearchAccountingError("DUPLICATE_TRIAL_IDENTITY", f"TRIAL_ASSIGNED_TO_MULTIPLE_CAMPAIGNS:{trial_id}")
        first = rows[0]
        comparable = ("model_id", "hyperparameters", "random_seed", "dataset_identity", "fold_identity", "training_dates", "validation_dates")
        if any(row.get(key) != first.get(key) for row in rows[1:] for key in comparable):
            raise SearchAccountingError("DUPLICATE_TRIAL_IDENTITY", f"TRIAL_IDENTITY_CONFLICT:{trial_id}")
        terminal = rows[-1]["terminal_status"]
        material = any(row["material_evaluation"] for row in rows) and terminal in MATERIAL_TERMINAL
        if terminal == "INVALIDATED":
            material = any(row["material_evaluation"] for row in rows)
        execution_kinds = [row["execution_kind"] for row in rows]
        trial = {
            "contract_version": TRIAL_CONTRACT,
            "trial_id": trial_id,
            "search_campaign_id": first["search_campaign_id"],
            "hypothesis_id": first["hypothesis_id"],
            "model_id": first["model_id"],
            "hyperparameters": first["hyperparameters"],
            "hyperparameter_checksum": canonical_hash(first["hyperparameters"]) if first["hyperparameters"] is not None else None,
            "random_seed": first["random_seed"],
            "dataset_identity": first["dataset_identity"],
            "fold_identity": first["fold_identity"],
            "training_dates": first["training_dates"],
            "validation_dates": first["validation_dates"],
            "source_commit": first["source_commit"],
            "attempt_ids": [row["attempt_id"] for row in rows],
            "attempt_count": len(rows),
            "retry_count": max(len(rows) - 1, 0),
            "resumed_execution_count": execution_kinds.count("resumed"),
            "cache_reuse_count": sum(row["terminal_status"] == "SKIPPED_COMPLETE" or row["execution_kind"] == "cache" for row in rows),
            "terminal_status": terminal,
            "metrics_path": rows[-1]["metrics_path"],
            "continuation_or_rejection_reason": rows[-1]["continuation_or_rejection_reason"],
            "counts_as_material_search": material,
            "material_exclusion_reason": None if material else (
                "PRE_EXECUTION_REJECTION" if terminal == "REJECTED" else
                "CACHE_REUSE" if terminal == "SKIPPED_COMPLETE" else
                "INCOMPLETE_LIFECYCLE" if terminal == "INCOMPLETE" else "NOT_MATERIALLY_EVALUATED"
            ),
        }
        trial["trial_checksum"] = canonical_hash(trial)
        trials.append(trial)
        campaign_owner[trial_id] = first["search_campaign_id"]
    return trials


def _validate_governance(trials, hypotheses, campaigns):
    warnings = []
    for trial in trials:
        if not trial["hypothesis_id"] and not trial["search_campaign_id"]:
            return {"status": "LEGACY_UNASSIGNED", "blocking_reasons": ["LEGACY_EVENT_GOVERNANCE_IDENTITIES_MISSING"], "warnings": warnings}
        if not trial["hypothesis_id"]:
            return {"status": "MISSING_HYPOTHESIS_IDENTITY", "blocking_reasons": ["TRIAL_HYPOTHESIS_ID_MISSING"], "warnings": warnings}
        if not trial["search_campaign_id"]:
            return {"status": "MISSING_CAMPAIGN_IDENTITY", "blocking_reasons": ["TRIAL_CAMPAIGN_ID_MISSING"], "warnings": warnings}
        if trial["hypothesis_id"] not in hypotheses:
            return {"status": "MISSING_HYPOTHESIS_IDENTITY", "blocking_reasons": ["HYPOTHESIS_REGISTRATION_MISSING"], "warnings": warnings}
        campaign = campaigns.get(trial["search_campaign_id"])
        if not campaign:
            return {"status": "MISSING_CAMPAIGN_IDENTITY", "blocking_reasons": ["CAMPAIGN_REGISTRATION_MISSING"], "warnings": warnings}
        if campaign.get("hypothesis_id") != trial["hypothesis_id"]:
            return {"status": "INVALID_EVENT_HISTORY", "blocking_reasons": ["TRIAL_CAMPAIGN_HYPOTHESIS_MISMATCH"], "warnings": warnings}
        if trial["terminal_status"] == "INCOMPLETE":
            warnings.append(f"INCOMPLETE_LIFECYCLE:{trial['trial_id']}")
    return {"status": "VALID", "blocking_reasons": [], "warnings": warnings}


def _campaign_accounting(trials, campaigns):
    rows = []
    for campaign_id in sorted({trial["search_campaign_id"] for trial in trials}):
        campaign = campaigns[campaign_id]
        relevant = [trial for trial in trials if trial["search_campaign_id"] == campaign_id]
        planned = campaign.get("planned_total_material_trials")
        if planned is None:
            rows.append({"search_campaign_id": campaign_id, "hypothesis_id": campaign.get("hypothesis_id"), "blocking_reasons": ["MISSING_SEARCH_BUDGET"]})
            continue
        extension = int(campaign.get("authorised_extension", 0))
        effective_budget = int(planned) + extension
        material = [trial for trial in relevant if trial["counts_as_material_search"]]
        exceeded = len(material) > effective_budget
        row = {
            "search_campaign_id": campaign_id,
            "hypothesis_id": campaign["hypothesis_id"],
            "planned_total_material_trials": int(planned),
            "authorised_extension": extension,
            "effective_budget": effective_budget,
            "trials_attempted": len(material),
            "trials_completed": sum(trial["terminal_status"] == "COMPLETED" for trial in material),
            "trials_failed": sum(trial["terminal_status"] == "FAILED" for trial in material),
            "trials_rejected": sum(trial["terminal_status"] == "REJECTED" for trial in material),
            "trials_remaining": max(effective_budget - len(material), 0),
            "budget_utilisation": len(material) / effective_budget,
            "budget_exceeded": exceeded,
            "campaign_closed": campaign.get("campaign_status") == "CLOSED",
            "continuation_authorised": bool(campaign.get("continuation_authorised", False)),
            "campaign_stop_reason": campaign.get("campaign_stop_reason"),
            "blocking_reasons": ["BUDGET_EXCEEDED_WITHOUT_AUTHORISED_EXTENSION"] if exceeded else [],
        }
        rows.append(row)
    return rows


def _counts(events, attempts, trials, material):
    return {
        "raw_event_count": len(events),
        "experiment_count": len({event["experiment_run_id"] for event in events}),
        "process_attempt_count": len(attempts),
        "logical_trial_count": len(trials),
        "completed_trial_count": sum(row["terminal_status"] == "COMPLETED" for row in trials),
        "failed_trial_count": sum(row["terminal_status"] == "FAILED" for row in trials),
        "rejected_trial_count": sum(row["terminal_status"] == "REJECTED" for row in trials),
        "invalidated_trial_count": sum(row["terminal_status"] == "INVALIDATED" for row in trials),
        "pre_execution_rejection_count": sum(row["material_exclusion_reason"] == "PRE_EXECUTION_REJECTION" for row in trials),
        "retry_count": sum(row["retry_count"] for row in trials),
        "resumed_execution_count": sum(row["resumed_execution_count"] for row in trials),
        "cache_reuse_count": sum(row["cache_reuse_count"] for row in trials),
        "material_effective_search_count": len(material),
        "dsr_effective_search_count": len(material),
    }


def _hypothesis_summaries(trials, hypotheses):
    rows = []
    for hypothesis_id in sorted({trial["hypothesis_id"] for trial in trials}):
        relevant = [trial for trial in trials if trial["hypothesis_id"] == hypothesis_id]
        rows.append({
            "hypothesis_id": hypothesis_id,
            "campaign_count": len({trial["search_campaign_id"] for trial in relevant}),
            "logical_trial_count": len(relevant),
            "material_effective_search_count": sum(trial["counts_as_material_search"] for trial in relevant),
            "hypothesis_checksum": hypotheses[hypothesis_id]["hypothesis_checksum"],
        })
    return rows


def _blocked(events, attempts, trials, status, reasons, *, campaign_rows=()):
    logical = {
        "contract_version": ACCOUNTING_CONTRACT,
        "status": status if status in STATUSES else "INVALID_EVENT_HISTORY",
        "valid": False,
        "blocking_reasons": sorted(set(reasons)),
        "warnings": [],
        "counting_policy_id": COUNTING_POLICY_ID,
        "counting_policy_version": "1.0",
        "counts": _counts(events, attempts, trials, []) if events else {},
        "process_attempts": attempts,
        "logical_trials": trials,
        "campaign_summaries": list(campaign_rows),
        "source_event_checksum": canonical_hash(events),
        "logical_trial_population_checksum": canonical_hash(
            [{"trial_id": row["trial_id"], "trial_checksum": row["trial_checksum"]} for row in trials]
        ),
        "effective_search_count": None,
        "dsr_effective_search_count": None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _parquet_value(value):
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
