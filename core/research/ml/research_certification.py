from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.artifact_lineage import (
    VERIFIED_STRICT_OOS,
    verify_lineage_graph,
)
from core.research.ml.dataset_build_manifest import (
    STATUS_CONFLICTING_PARENT,
    STATUS_CURRENT,
    STATUS_LEGACY_NO_MANIFEST,
    STATUS_MISSING_PARENT,
    STATUS_STALE,
    check_dataset_lineage,
    dataset_manifest_path,
    file_sha256,
)
from core.research.ml.experiment_ledger import read_ledger
from core.research.ml.policy_evaluation.contracts import normalise_trial_accounting
from core.research.ml.provenance import dependency_identity, file_identity, source_provenance
from core.research.ml.registries import load_registry_bundle
from core.research.ml.registries.io import canonical_hash


RESEARCH_CERTIFICATION_ENVELOPE_VERSION = "research_certification_envelope_v1"
RESEARCH_CERTIFICATION_REPLAY_VERIFIER_VERSION = "research_certification_replay_verifier_v1"

CERTIFIABLE = "CERTIFIABLE"
RESEARCH_ONLY = "RESEARCH_ONLY"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
BLOCKED = "BLOCKED"

HASH_EXACT = "EXACT"
HASH_MISMATCH = "MISMATCH"
HASH_MISSING = "MISSING"

STANDARD_ENVELOPE_FIELDS = (
    "run_id",
    "parent_run_ids",
    "trial_family_id",
    "git_sha",
    "git_dirty",
    "dirty_patch_hash",
    "environment_hash",
    "resolved_config_hash",
    "registry_versions",
    "authority_versions",
    "random_seeds",
    "input_snapshots",
    "dataset_manifests",
    "fold_definition_hash",
    "prediction_artifacts",
    "portfolio_replay_artifacts",
    "execution_scenario",
    "trial_accounting",
    "output_hashes",
    "certification_status",
)

REQUIRED_AUTHORITY_FIELDS = (
    "universe_authority_version",
    "identity_authority_version",
    "market_calendar_authority_version",
    "target_contract_version",
    "feature_code_version",
    "label_code_version",
)

SEQUENCE_MODEL_IDS = frozenset(
    {
        "dlinear",
        "patchtst",
        "itransformer",
        "momentum_transformer",
        "multitask_transformer",
        "market_context_encoder",
        "news_analysis_transformer",
        "temporal_fusion_transformer",
        "transformer",
    }
)


def write_research_certification_envelope(
    output_path: Path,
    *,
    config: Mapping[str, Any] | None = None,
    source_control: Mapping[str, Any] | None = None,
    dirty_patch_hash: str | None = None,
    capture_dirty_patch: bool = False,
    parent_run_ids: Sequence[str] = (),
    trial_family_id: str | None = None,
    dataset_manifest_paths: Sequence[Path] = (),
    dataset_roots: Sequence[Path] = (),
    prediction_manifest_paths: Sequence[Path] = (),
    portfolio_replay_paths: Sequence[Path] = (),
    input_paths: Sequence[Path] = (),
    output_paths: Sequence[Path] = (),
    execution_scenario: Mapping[str, Any] | None = None,
    trial_accounting: Mapping[str, Any] | None = None,
    experiment_ledger_path: Path | None = None,
    sequence_context: Mapping[str, Any] | None = None,
    promotion_evidence: Mapping[str, Any] | None = None,
    diagnostic_legacy: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    envelope = build_research_certification_envelope(
        config=config,
        source_control=source_control,
        dirty_patch_hash=dirty_patch_hash,
        capture_dirty_patch=capture_dirty_patch,
        parent_run_ids=parent_run_ids,
        trial_family_id=trial_family_id,
        dataset_manifest_paths=dataset_manifest_paths,
        dataset_roots=dataset_roots,
        prediction_manifest_paths=prediction_manifest_paths,
        portfolio_replay_paths=portfolio_replay_paths,
        input_paths=input_paths,
        output_paths=output_paths,
        execution_scenario=execution_scenario,
        trial_accounting=trial_accounting,
        experiment_ledger_path=experiment_ledger_path,
        sequence_context=sequence_context,
        promotion_evidence=promotion_evidence,
        diagnostic_legacy=diagnostic_legacy,
        run_id=run_id,
    )
    _write_json_atomic(output_path, envelope)
    return envelope


def build_research_certification_envelope(
    *,
    config: Mapping[str, Any] | None = None,
    source_control: Mapping[str, Any] | None = None,
    dirty_patch_hash: str | None = None,
    capture_dirty_patch: bool = False,
    parent_run_ids: Sequence[str] = (),
    trial_family_id: str | None = None,
    dataset_manifest_paths: Sequence[Path] = (),
    dataset_roots: Sequence[Path] = (),
    prediction_manifest_paths: Sequence[Path] = (),
    portfolio_replay_paths: Sequence[Path] = (),
    input_paths: Sequence[Path] = (),
    output_paths: Sequence[Path] = (),
    execution_scenario: Mapping[str, Any] | None = None,
    trial_accounting: Mapping[str, Any] | None = None,
    experiment_ledger_path: Path | None = None,
    sequence_context: Mapping[str, Any] | None = None,
    promotion_evidence: Mapping[str, Any] | None = None,
    diagnostic_legacy: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    ml_config = dict((config or {}).get("ml", {}) or {})
    certification_config = dict(ml_config.get("research_certification", {}) or {})
    source = dict(source_control or source_provenance())
    git_dirty = bool(
        source.get("dirty_worktree")
        or source.get("git_dirty")
        or source.get("dirty_tree")
    )
    if git_dirty and dirty_patch_hash is None and capture_dirty_patch:
        dirty_patch_hash = _captured_dirty_patch_hash()

    roots = _unique_paths(dataset_roots)
    resolved_dataset_manifests = _unique_paths(
        [
            *dataset_manifest_paths,
            *(dataset_manifest_path(root / "rows.parquet") for root in roots),
        ]
    )
    discovered_inputs = _unique_paths(
        [
            *input_paths,
            *_config_input_paths(config or {}),
            *(root / "manifest.json" for root in roots),
            *(root / "rows.parquet" for root in roots),
            *(root / "baseline_scores.parquet" for root in roots),
        ]
    )
    predictions = [
        _prediction_artifact_record(path)
        for path in _unique_paths(prediction_manifest_paths)
    ]
    replay_artifacts = [
        _portfolio_replay_record(path)
        for path in _unique_paths(portfolio_replay_paths)
    ]
    dataset_records = [
        _dataset_manifest_record(path)
        for path in resolved_dataset_manifests
    ]
    discovered_inputs = _unique_paths(
        [
            *discovered_inputs,
            *_dataset_input_paths(dataset_records),
        ]
    )
    output_records = _output_records(
        [
            *output_paths,
            *(
                Path(str(record["artifact_path"]))
                for record in predictions
                if record.get("artifact_path")
            ),
            *(
                Path(str(record["summary_path"]))
                for record in replay_artifacts
                if record.get("summary_path")
            ),
            *(
                Path(str(item["path"]))
                for replay in replay_artifacts
                for item in replay.get("related_artifacts", [])
                if item.get("path")
            ),
        ]
    )
    registry_versions = _registry_versions()
    authority_versions = _authority_versions(
        dataset_records=dataset_records,
        prediction_records=predictions,
        replay_records=replay_artifacts,
        sequence_context=sequence_context,
    )
    resolved_trial_family_id = str(
        trial_family_id
        or certification_config.get("trial_family_id")
        or ml_config.get("trial_family_id")
        or ""
    )
    accounting = _trial_accounting(
        trial_family_id=resolved_trial_family_id,
        trial_accounting=trial_accounting
        or certification_config.get("trial_accounting"),
        ledger_path=experiment_ledger_path
        or _optional_path(certification_config.get("experiment_ledger_path"))
        or _optional_path(ml_config.get("experiment_ledger_path")),
        prediction_records=predictions,
    )
    folds = sorted(
        str(record.get("fold_identity"))
        for record in predictions
        if record.get("fold_identity")
    )
    random_seeds = _random_seeds(
        config=config or {},
        prediction_records=predictions,
    )
    scenario = _execution_scenario(
        config=config or {},
        supplied=execution_scenario
        or certification_config.get("execution_scenario"),
        replay_records=replay_artifacts,
    )
    evidence = _promotion_evidence(
        supplied=promotion_evidence
        or certification_config.get("promotion_evidence"),
        replay_records=replay_artifacts,
    )
    envelope: dict[str, Any] = {
        "contract_version": RESEARCH_CERTIFICATION_ENVELOPE_VERSION,
        "run_id": str(run_id or ""),
        "parent_run_ids": sorted(str(value) for value in parent_run_ids),
        "trial_family_id": resolved_trial_family_id,
        "git_sha": source.get("git_commit") or source.get("git_sha") or "",
        "git_dirty": git_dirty,
        "dirty_patch_hash": dirty_patch_hash,
        "environment_hash": dependency_identity(
            ("numpy", "pyarrow", "scikit-learn", "exchange-calendars")
        )["hash"],
        "resolved_config_hash": canonical_hash(config or {}),
        "registry_versions": registry_versions,
        "authority_versions": authority_versions,
        "random_seeds": random_seeds,
        "input_snapshots": _file_snapshots(discovered_inputs),
        "dataset_manifests": dataset_records,
        "fold_definition_hash": canonical_hash(folds) if folds else "",
        "prediction_artifacts": predictions,
        "portfolio_replay_artifacts": replay_artifacts,
        "execution_scenario": scenario,
        "trial_accounting": accounting,
        "ticket66_enforcement": _ticket66_enforcement_summary(accounting),
        "output_hashes": output_records,
        "certification_status": "",
        "promotion_evidence": evidence,
        "diagnostic_legacy": bool(diagnostic_legacy),
        "components_reused": [
            "source_worktree_provenance_v1",
            "dataset_build_manifest_v1",
            "frozen_selector_dataset_combined_lineage_guard_v1",
            "artifact_link_contract_v1",
            "selector_experiment_ledger.v1",
            "canonical_policy_trial_accounting.v1",
            "sequence_window_authority_v1",
            "portfolio_replay_identity_v2_registry",
        ],
    }
    gates = certification_gates(envelope)
    envelope["certification_status"] = gates["status"]
    envelope["certification_gates"] = gates
    envelope["ticket66_enforcement"]["blockers"] = sorted(
        set(envelope["ticket66_enforcement"].get("blockers", []) + gates["all_reasons"])
    )
    envelope["ticket66_enforcement"]["enforcement_hash"] = canonical_hash(
        {
            key: value
            for key, value in envelope["ticket66_enforcement"].items()
            if key != "enforcement_hash"
        }
    )
    identity_payload = _envelope_identity_payload(envelope)
    envelope["deterministic_envelope_identity"] = canonical_hash(identity_payload)
    if not envelope["run_id"]:
        envelope["run_id"] = (
            "research-cert-"
            + envelope["deterministic_envelope_identity"][:16].lower()
        )
    envelope["promotion"] = {
        "automatic_promotion": False,
        "promotion_action": "not_triggered",
        "promotion_allowed_by_certification": envelope["certification_status"]
        == CERTIFIABLE,
        "promotion_prohibited": envelope["certification_status"] != CERTIFIABLE,
        "blocking_reasons": gates["all_reasons"]
        if envelope["certification_status"] != CERTIFIABLE
        else [],
    }
    envelope["envelope_hash"] = canonical_hash(
        {key: value for key, value in envelope.items() if key != "envelope_hash"}
    )
    return envelope


def certification_gates(envelope: Mapping[str, Any]) -> dict[str, Any]:
    hard: list[str] = []
    research: list[str] = []
    diagnostic: list[str] = []

    if envelope.get("git_dirty") and not envelope.get("dirty_patch_hash"):
        research.append("DIRTY_TREE_WITHOUT_CAPTURED_PATCH")

    dataset_records = list(envelope.get("dataset_manifests") or [])
    if not dataset_records:
        research.append("DATASET_MANIFEST_MISSING")
    for record in dataset_records:
        status = str(record.get("lineage_status") or "")
        if status in {
            STATUS_STALE,
            STATUS_MISSING_PARENT,
            STATUS_CONFLICTING_PARENT,
        }:
            hard.append(f"STALE_OR_MISSING_DATASET_PARENT:{status}")
        elif status == STATUS_LEGACY_NO_MANIFEST:
            diagnostic.append("LEGACY_DATASET_MANIFEST_MISSING")
        elif status and status != STATUS_CURRENT:
            lineage_reasons = set(record.get("lineage_reasons") or [])
            dirty_only = lineage_reasons <= {
                "DIRTY_TREE_BUILD",
                "INTENDED_USE_NOT_PERMITTED:PROMOTION_GRADE",
            }
            if not (envelope.get("dirty_patch_hash") and dirty_only):
                research.append(f"DATASET_LINEAGE_NOT_CURRENT:{status}")

    authority = envelope.get("authority_versions")
    for field in REQUIRED_AUTHORITY_FIELDS:
        if not _meaningful_authority(_find_authority_value(authority, field)):
            research.append(f"AUTHORITY_VERSION_MISSING:{field}")
    universe = _find_authority_value(authority, "universe_authority_version")
    if _is_non_pit_universe(universe):
        hard.append("NON_PIT_UNIVERSE")
    identity = _find_authority_value(authority, "identity_authority_version")
    if _is_unresolved_identity(identity):
        hard.append("UNRESOLVED_IDENTITY")

    predictions = list(envelope.get("prediction_artifacts") or [])
    if not predictions:
        research.append("OOS_PREDICTIONS_MISSING")
    for record in predictions:
        if record.get("artifact_kind") in {"RESEARCH_DIAGNOSTIC", "TREE_DIAGNOSTIC"}:
            diagnostic.append("DIAGNOSTIC_LEGACY_RUN")
            continue
        if record.get("hash_status") == HASH_MISMATCH:
            hard.append("PREDICTION_HASH_MISMATCH")
        elif record.get("hash_status") == HASH_MISSING:
            research.append("OOS_PREDICTION_ARTIFACT_MISSING")
        status = str(record.get("verification_status") or "")
        if status and status != VERIFIED_STRICT_OOS:
            if status == "CONFLICTING_EVIDENCE":
                hard.append("OOS_PREDICTION_LINEAGE_CONFLICT")
            else:
                research.append(f"OOS_PREDICTION_NOT_VERIFIED:{status}")
        if not record.get("maximum_label_available_timestamp"):
            hard.append("TARGET_AVAILABILITY_MISSING")

    sequence_present = bool(
        (envelope.get("authority_versions") or {}).get("sequence_authority")
        or any(record.get("sequence_model") for record in predictions)
    )
    sequence_context = (envelope.get("authority_versions") or {}).get(
        "sequence_authority"
    )
    if sequence_present and not (
        isinstance(sequence_context, Mapping)
        and sequence_context.get("strict_context_recorded") is True
    ):
        hard.append("STRICT_SEQUENCE_CONTEXT_MISSING")

    replay_artifacts = list(envelope.get("portfolio_replay_artifacts") or [])
    if not replay_artifacts:
        research.append("PORTFOLIO_REPLAY_MISSING")
    for record in replay_artifacts:
        if record.get("hash_status") == HASH_MISMATCH:
            hard.append("REPLAY_HASH_MISMATCH")
        elif record.get("hash_status") == HASH_MISSING:
            research.append("PORTFOLIO_REPLAY_MISSING")
        status = str(record.get("verification_status") or "")
        if status and status != VERIFIED_STRICT_OOS:
            research.append(f"PORTFOLIO_REPLAY_NOT_REPRODUCIBLE:{status}")

    trial = envelope.get("trial_accounting")
    if not envelope.get("trial_family_id") or not isinstance(trial, Mapping):
        research.append("TRIAL_FAMILY_RECORD_MISSING")
    elif trial.get("evidence_complete") is not True:
        research.append("TRIAL_FAMILY_RECORD_MISSING")
        research.extend(_ticket66_trial_accounting_gate_reasons(trial))

    scenario = envelope.get("execution_scenario")
    if not _execution_assumptions_complete(scenario):
        research.append("EXECUTION_ASSUMPTIONS_MISSING")

    promotion = envelope.get("promotion_evidence")
    if not isinstance(promotion, Mapping) or promotion.get("evidence_complete") is not True:
        research.append("PROMOTION_EVIDENCE_INCOMPLETE")

    if envelope.get("diagnostic_legacy"):
        diagnostic.append("DIAGNOSTIC_LEGACY_RUN")

    hard = sorted(set(hard))
    research = sorted(set(research))
    diagnostic = sorted(set(diagnostic))
    if hard:
        status = BLOCKED
    elif diagnostic:
        status = DIAGNOSTIC_ONLY
    elif research:
        status = RESEARCH_ONLY
    else:
        status = CERTIFIABLE
    return {
        "contract_version": "research_certification_gates_v1",
        "status": status,
        "hard_blocking_reasons": hard,
        "research_only_reasons": research,
        "diagnostic_reasons": diagnostic,
        "all_reasons": sorted(set(hard + research + diagnostic)),
    }


def _ticket66_trial_accounting_gate_reasons(trial: Mapping[str, Any]) -> list[str]:
    missing = set(str(reason) for reason in trial.get("missing_evidence", []) or [])
    reasons: list[str] = []
    if "TRIAL_FAMILY_COUNT_MISSING" in missing or not trial.get("trial_family_id"):
        reasons.append("TRIAL_FAMILY_MISSING")
    if "TRIAL_FAMILY_ACCOUNTING_INCOMPLETE" in missing or trial.get("trial_family_complete") is False:
        reasons.append("TRIAL_ATTEMPTS_INCOMPLETE")
    if "FAILED_TRIALS_NOT_ACCOUNTED" in missing:
        reasons.append("FAILED_TRIALS_NOT_ACCOUNTED")
    if "SKIPPED_TRIALS_NOT_ACCOUNTED" in missing:
        reasons.append("SKIPPED_TRIALS_NOT_ACCOUNTED")
    if "DSR_EVIDENCE_MISSING" in missing:
        reasons.append("DSR_FULL_FAMILY_EVIDENCE_MISSING")
    if "PBO_EVIDENCE_MISSING" in missing:
        reasons.append("PBO_DECLARED_FAMILY_EVIDENCE_MISSING")
    if "MULTIPLICITY_EVIDENCE_MISSING" in missing:
        reasons.append("MULTIPLICITY_EVIDENCE_MISSING")
    if "LOCKED_HOLDOUT_MISSING" in missing:
        reasons.append("LOCKED_HOLDOUT_MISSING")
    if "LOCKED_HOLDOUT_INVALIDATED" in missing or trial.get("holdout_invalidation_status") == "INVALIDATED":
        reasons.append("LOCKED_HOLDOUT_REUSED")
    if "HOLDOUT_ACCESS_UNAUTHORISED" in missing:
        reasons.append("HOLDOUT_ACCESS_UNAUTHORISED")
    if "DSR_FAMILY_MISMATCH" in missing:
        reasons.append("DSR_FAMILY_MISMATCH")
    if "PBO_FAMILY_MISMATCH" in missing:
        reasons.append("PBO_FAMILY_MISMATCH")
    if trial.get("effective_search_count") != trial.get("trial_family_count"):
        reasons.append("DSR_FAMILY_SIZE_MISMATCH")
    return reasons


def _ticket66_enforcement_summary(trial: Mapping[str, Any]) -> dict[str, Any]:
    missing = [str(reason) for reason in trial.get("missing_evidence", []) or []]
    summary = {
        "contract_version": "ticket65_ticket66_enforcement_envelope_v1",
        "family_id": trial.get("trial_family_id"),
        "family_identity": trial.get("trial_family_identity"),
        "family_manifest_hash": trial.get("accounting_result_checksum"),
        "holdout_identity": trial.get("holdout_identity"),
        "holdout_validity": trial.get("holdout_valid"),
        "attempt_count": trial.get("trial_family_count"),
        "attempt_status_counts": trial.get("attempt_status_counts"),
        "dsr_evidence_hash": trial.get("dsr_evidence_hash"),
        "pbo_evidence_hash": trial.get("pbo_evidence_hash"),
        "multiplicity_evidence_hash": trial.get("multiplicity_evidence_hash"),
        "blockers": sorted(set(missing + _ticket66_trial_accounting_gate_reasons(trial))),
    }
    summary["enforcement_hash"] = canonical_hash(summary)
    return summary


def verify_research_certification_envelope(
    envelope_path: Path,
    *,
    reproduced_artifact_paths: Mapping[str, str | Path] | None = None,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    overrides = {
        str(key): Path(value)
        for key, value in dict(reproduced_artifact_paths or {}).items()
    }
    commit = _recorded_commit_status(str(envelope.get("git_sha") or ""))
    input_comparisons = [
        _compare_snapshot(row, overrides=overrides)
        for row in envelope.get("input_snapshots", [])
    ]
    dataset_comparisons = [
        _compare_snapshot(row, path_key="manifest_path", overrides=overrides)
        for row in envelope.get("dataset_manifests", [])
        if row.get("manifest_path")
    ]
    prediction_comparisons = []
    for row in envelope.get("prediction_artifacts", []):
        if row.get("manifest_path"):
            prediction_comparisons.append(
                _compare_snapshot(
                    {
                        "path": row["manifest_path"],
                        "sha256": row.get("manifest_sha256"),
                    },
                    overrides=overrides,
                )
            )
        if row.get("artifact_path"):
            prediction_comparisons.append(
                _compare_snapshot(
                    {
                        "path": row["artifact_path"],
                        "sha256": row.get("artifact_sha256"),
                    },
                    overrides=overrides,
                    tolerance=tolerance,
                )
            )
    replay_comparisons = []
    for row in envelope.get("portfolio_replay_artifacts", []):
        if row.get("summary_path"):
            replay_comparisons.append(
                _compare_snapshot(
                    {
                        "path": row["summary_path"],
                        "sha256": row.get("summary_sha256"),
                    },
                    overrides=overrides,
                    tolerance=tolerance,
                )
            )
        for item in row.get("related_artifacts", []) or []:
            replay_comparisons.append(
                _compare_snapshot(item, overrides=overrides, tolerance=tolerance)
            )
    output_comparisons = [
        _compare_snapshot(row, overrides=overrides, tolerance=tolerance)
        for row in envelope.get("output_hashes", [])
    ]
    all_comparisons = (
        input_comparisons
        + dataset_comparisons
        + prediction_comparisons
        + replay_comparisons
        + output_comparisons
    )
    mismatches = [
        row
        for row in all_comparisons
        if row.get("status") not in {HASH_EXACT, "TOLERANCE_MATCH"}
    ]
    tolerance_matches = [
        row for row in all_comparisons if row.get("status") == "TOLERANCE_MATCH"
    ]
    if not mismatches and tolerance_matches:
        status = "TOLERANCE_REPLAY"
    elif not mismatches:
        status = "EXACT_ARTIFACT_REPLAY"
    else:
        status = "HASH_MISMATCH"
    result = {
        "contract_version": RESEARCH_CERTIFICATION_REPLAY_VERIFIER_VERSION,
        "envelope_path": str(envelope_path),
        "recorded_run_id": envelope.get("run_id"),
        "recorded_commit": commit,
        "input_hashes": input_comparisons,
        "dataset_manifest_hashes": dataset_comparisons,
        "prediction_hashes": prediction_comparisons,
        "replay_hashes": replay_comparisons,
        "output_hashes": output_comparisons,
        "reproduction_status": status,
        "exact_reproduction": status == "EXACT_ARTIFACT_REPLAY",
        "tolerance_based_reproduction": status == "TOLERANCE_REPLAY",
        "training_rerun_performed": False,
        "blocking_reasons": sorted(
            {
                row["reason"]
                for row in mismatches
                if row.get("reason")
            }
        ),
    }
    result["verification_hash"] = canonical_hash(result)
    return result


def write_bounded_selector_certification_envelope(
    config: Mapping[str, Any],
    *,
    settings: Any,
    run_summary: Mapping[str, Any],
) -> Path:
    output_root = Path(settings.output_root)
    prediction_manifests = sorted(output_root.glob("date=*/manifest.json"))
    path = output_root / "research_certification_envelope.json"
    ticket66 = _bounded_selector_ticket66_evidence(output_root)
    scenario = {
        "integration": "ticket_63_bounded_selector",
        "execution_assumptions": {
            "bounded_training": True,
            "trading_impact": "none",
            "production_validated": False,
            "promotion_automated": False,
        },
        "no_trade_comparison": _nested_get(
            config,
            ("ml", "research_certification", "no_trade_comparison"),
        ),
        "null_policy": _nested_get(
            config,
            ("ml", "research_certification", "null_policy"),
        )
        or {
            "policy_id": "research_null_no_orders",
            "trade_allowed": False,
            "promotion_effect": "prohibit_automatic_promotion",
        },
        "promotion_prohibition": {
            "automatic_promotion": False,
            "reason": "bounded_selector_envelope_only",
        },
        "run_summary_hash": canonical_hash(run_summary),
    }
    envelope = write_research_certification_envelope(
        path,
        config=config,
        dataset_roots=(Path(settings.dataset_root),),
        prediction_manifest_paths=prediction_manifests,
        input_paths=(
            Path(settings.dataset_root) / "feature_schema.json",
            *ticket66.get("input_paths", ()),
        ),
        output_paths=tuple(ticket66.get("output_paths", ())),
        execution_scenario=scenario,
        trial_family_id=_nested_get(
            config,
            ("ml", "research_certification", "trial_family_id"),
        )
        or ticket66.get("trial_family_id"),
        trial_accounting=ticket66.get("trial_accounting")
        or _nested_get(
            config,
            ("ml", "research_certification", "trial_accounting"),
        ),
        promotion_evidence={
            "evidence_complete": False,
            "promotion_allowed": False,
            "promotion_prohibited": True,
            "reason": "portfolio_replay_and_promotion_evidence_not_part_of_bounded_training",
        },
    )
    if ticket66.get("trial_family_id"):
        _write_json_atomic(
            output_root / "certification_gate_validation.json",
            _bounded_selector_certification_gate_validation(envelope, ticket66),
        )
    return path


def _bounded_selector_ticket66_evidence(output_root: Path) -> dict[str, Any]:
    family_path = output_root / "trial_family_manifest.json"
    holdout_path = output_root / "locked_outer_holdout.json"
    policy_evidence_path = output_root / "trial_family_policy_evidence.json"
    family = _read_json_object(family_path)
    if not family:
        return {"input_paths": (), "output_paths": ()}
    holdout = _read_json_object(holdout_path)
    supplied_policy_evidence = _read_json_object(policy_evidence_path)
    if supplied_policy_evidence:
        trial_accounting = supplied_policy_evidence
    else:
        from core.research.ml.trial_family_protocol import (
            policy_trial_accounting_evidence,
        )

        trial_accounting = policy_trial_accounting_evidence(
            family,
            locked_holdout=holdout or None,
        )
    paths = [
        path
        for path in (
            output_root / "active_path_inventory.json",
            family_path,
            output_root / "trial_family_example.json",
            output_root / "trial_family_attempts.json",
            output_root / "trial_family_planned_attempts.json",
            output_root / "trial_attempts.csv",
            policy_evidence_path,
            holdout_path,
            output_root / "locked_holdout_example.json",
            output_root / "holdout_access_log.json",
            output_root / "holdout_access_log.csv",
            output_root / "ticket63_integration_validation.json",
            output_root / "ticket50_integration_validation.json",
            output_root / "ticket_66a_summary.md",
        )
        if path.exists()
    ]
    return {
        "trial_family_id": family.get("family_id"),
        "trial_accounting": trial_accounting,
        "family": family,
        "locked_holdout": holdout,
        "input_paths": tuple(paths),
        "output_paths": tuple(paths),
    }


def _bounded_selector_certification_gate_validation(
    envelope: Mapping[str, Any],
    ticket66: Mapping[str, Any],
) -> dict[str, Any]:
    gates = dict(envelope.get("certification_gates") or {})
    enforcement = dict(envelope.get("ticket66_enforcement") or {})
    validation = {
        "contract_version": "ticket66a_certification_gate_validation_v1",
        "status": gates.get("status"),
        "trial_family_id": enforcement.get("family_id") or ticket66.get("trial_family_id"),
        "family_manifest_hash": enforcement.get("family_manifest_hash"),
        "holdout_identity": enforcement.get("holdout_identity"),
        "holdout_validity": enforcement.get("holdout_validity"),
        "attempt_count": enforcement.get("attempt_count"),
        "dsr_evidence_hash": enforcement.get("dsr_evidence_hash"),
        "pbo_evidence_hash": enforcement.get("pbo_evidence_hash"),
        "multiplicity_evidence_hash": enforcement.get("multiplicity_evidence_hash"),
        "hard_blocking_reasons": gates.get("hard_blocking_reasons", []),
        "research_only_reasons": gates.get("research_only_reasons", []),
        "diagnostic_reasons": gates.get("diagnostic_reasons", []),
        "blockers": gates.get("all_reasons", []),
        "promotion": envelope.get("promotion"),
        "fails_closed_for_missing_or_invalid_evidence": gates.get("status") != CERTIFIABLE,
        "no_candidate_promoted": True,
    }
    validation["validation_checksum"] = canonical_hash(validation)
    return validation


def write_portfolio_replay_certification_envelope(
    config: Mapping[str, Any],
    *,
    replay_summary_path: Path,
) -> Path:
    ml = dict(config.get("ml", {}) or {})
    certification = dict(ml.get("research_certification", {}) or {})
    selector_manifests = [
        Path(str(path))
        for path in ml.get("selector_artifact_manifests", ()) or ()
    ]
    dataset_roots = [
        Path(str(path))
        for path in certification.get("dataset_roots", ()) or ()
    ]
    output_path = Path(
        certification.get("output_path")
        or replay_summary_path.parent / "research_certification_envelope.json"
    )
    scenario = {
        "integration": "ticket_63_portfolio_replay",
        "execution_assumptions": {
            "trading_impact": "none",
            "production_validated": False,
            "promotion_automated": False,
            "cost_bps": ml.get("stock_portfolio_replay_cost_bps"),
            "slippage_bps": ml.get("stock_portfolio_replay_slippage_bps"),
            "top_n": ml.get("stock_portfolio_replay_top_n"),
            "policies": ml.get("stock_portfolio_replay_policies"),
        },
        "no_trade_comparison": certification.get("no_trade_comparison"),
        "null_policy": certification.get("null_policy")
        or {
            "policy_id": "research_null_no_orders",
            "trade_allowed": False,
            "promotion_effect": "prohibit_automatic_promotion",
        },
        "promotion_prohibition": {
            "automatic_promotion": False,
            "reason": "certification_does_not_promote",
        },
    }
    write_research_certification_envelope(
        output_path,
        config=config,
        dataset_roots=dataset_roots,
        prediction_manifest_paths=selector_manifests,
        portfolio_replay_paths=(replay_summary_path,),
        execution_scenario=scenario,
        trial_family_id=certification.get("trial_family_id"),
        trial_accounting=certification.get("trial_accounting"),
        promotion_evidence=certification.get("promotion_evidence"),
    )
    return output_path


def _dataset_manifest_record(path: Path) -> dict[str, Any]:
    result = {
        "manifest_path": str(path),
        "exists": path.exists(),
        "manifest_sha256": file_sha256(path) if path.exists() else None,
        "dataset_id": None,
        "dataset_type": None,
        "lineage_status": STATUS_LEGACY_NO_MANIFEST if not path.exists() else None,
        "lineage_reasons": ["DATASET_MANIFEST_MISSING"] if not path.exists() else [],
        "source_paths": [],
        "source_manifest_hashes": [],
        "authority_versions": {},
    }
    payload = _read_json_object(path)
    if payload:
        result.update(
            {
                "dataset_id": payload.get("dataset_id"),
                "dataset_type": payload.get("dataset_type"),
                "manifest_hash": payload.get("manifest_hash"),
                "source_paths": list(payload.get("source_paths") or []),
                "source_manifest_hashes": list(
                    payload.get("source_manifest_hashes") or []
                ),
                "authority_versions": {
                    field: payload.get(field)
                    for field in (
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
                },
            }
        )
        if isinstance(payload.get("market_calendar_authority"), Mapping):
            result["authority_versions"]["market_calendar_authority"] = dict(
                payload.get("market_calendar_authority") or {}
            )
    lineage = check_dataset_lineage(
        manifest_path=path,
        intended_use="promotion-grade",
    )
    result["lineage_status"] = lineage.get("status")
    result["lineage_reasons"] = list(lineage.get("reasons") or [])
    result["permitted_use"] = lineage.get("permitted_use")
    result["use_authorized"] = lineage.get("use_authorized")
    return result


def _prediction_artifact_record(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    link = payload.get("artifact_link") if isinstance(payload.get("artifact_link"), Mapping) else {}
    artifact_path = payload.get("prediction_artifact_path") or link.get("artifact_path")
    recorded = payload.get("prediction_checksum") or link.get("artifact_checksum")
    artifact_identity = file_identity(Path(str(artifact_path))) if artifact_path else {
        "exists": False,
        "sha256": None,
    }
    manifest_sha = file_sha256(path) if path.exists() else None
    lineage = _safe_lineage(path, expected_artifact_kind=link.get("artifact_kind"))
    model_ids = _split_model_ids(
        link.get("canonical_model_or_policy_id")
        or payload.get("selector_model_identity")
        or ""
    )
    return {
        "manifest_path": str(path),
        "manifest_exists": path.exists(),
        "manifest_sha256": manifest_sha,
        "artifact_kind": link.get("artifact_kind"),
        "artifact_id": link.get("artifact_id"),
        "artifact_link_hash": link.get("artifact_link_hash"),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "artifact_exists": artifact_identity.get("exists"),
        "artifact_sha256": artifact_identity.get("sha256"),
        "recorded_artifact_sha256": str(recorded).upper()
        if recorded not in (None, "")
        else None,
        "hash_status": _hash_status(artifact_identity.get("sha256"), recorded),
        "verification_status": lineage.get("verification_status")
        or payload.get("validation_status"),
        "verification_reasons": lineage.get("verification_reasons", []),
        "decision_start": link.get("decision_start") or payload.get("decision_date"),
        "decision_end": link.get("decision_end") or payload.get("decision_date"),
        "training_cutoff": link.get("training_cutoff")
        or payload.get("training_cutoff"),
        "maximum_label_available_timestamp": link.get(
            "maximum_label_available_timestamp"
        )
        or payload.get("training_label_available_timestamp_max")
        or payload.get("label_availability_cutoff"),
        "fold_identity": payload.get("fold_identity"),
        "row_population_hash": link.get("row_population_hash")
        or payload.get("prediction_population_checksum"),
        "prediction_row_count": payload.get("prediction_row_count")
        or payload.get("oos_row_count"),
        "strict_oos_claim": link.get("strict_oos_claim"),
        "model_ids": sorted(model_ids),
        "sequence_model": bool(SEQUENCE_MODEL_IDS.intersection(model_ids)),
        "experiments": list(payload.get("experiments") or []),
        "random_seed": payload.get("random_seed"),
        "authority_versions": {
            "target_contract_version": payload.get("target_contract_version"),
            "feature_contract_version": payload.get("feature_contract_version"),
            "daily_stock_spine_identity": payload.get("daily_stock_spine_identity"),
            "symbol_registry_identity": payload.get("symbol_registry_identity"),
        },
    }


def _portfolio_replay_record(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    link = payload.get("artifact_link") if isinstance(payload.get("artifact_link"), Mapping) else {}
    related = [
        file_identity(candidate)
        for candidate in (
            path.parent / "stock_level_portfolio_replay_summary.csv",
            path.parent / "stock_level_portfolio_replay_equity_curves.csv",
            path.parent / "stock_level_portfolio_replay_holdings.csv",
        )
        if candidate.exists()
    ]
    summary_identity = file_identity(path)
    return {
        "summary_path": str(path),
        "summary_exists": summary_identity.get("exists"),
        "summary_sha256": summary_identity.get("sha256"),
        "artifact_kind": link.get("artifact_kind") or "PORTFOLIO_REPLAY",
        "artifact_id": link.get("artifact_id"),
        "artifact_link_hash": link.get("artifact_link_hash"),
        "verification_status": link.get("verification_status"),
        "verification_reasons": list(link.get("verification_reasons") or []),
        "hash_status": HASH_EXACT if summary_identity.get("exists") else HASH_MISSING,
        "strict_oos_claim": link.get("strict_oos_claim"),
        "upstream_link_count": len(link.get("upstream_links") or []),
        "promotion": dict(payload.get("promotion") or {}),
        "lineage_mode": payload.get("lineage_mode"),
        "execution_assumptions": {
            "policies": payload.get("policies"),
            "signal_columns": payload.get("signal_columns"),
            "cost_bps": payload.get("cost_bps"),
            "slippage_bps": payload.get("slippage_bps"),
            "top_n": payload.get("top_n"),
            "oos_only": payload.get("oos_only"),
        },
        "related_artifacts": related,
    }


def _safe_lineage(path: Path, *, expected_artifact_kind: Any = None) -> dict[str, Any]:
    if not path.exists():
        return {"verification_status": "INSUFFICIENT_EVIDENCE", "verification_reasons": ["ARTIFACT_MANIFEST_MISSING"]}
    try:
        return verify_lineage_graph(
            path,
            expected_artifact_kind=str(expected_artifact_kind)
            if expected_artifact_kind
            else None,
        )
    except Exception as exc:
        return {
            "verification_status": "INSUFFICIENT_EVIDENCE",
            "verification_reasons": [f"ARTIFACT_LINEAGE_UNREADABLE:{type(exc).__name__}"],
        }


def _registry_versions() -> dict[str, Any]:
    try:
        bundle = load_registry_bundle()
    except Exception as exc:
        return {
            "registry_contract_version": "ml_registry_set_v1",
            "status": "UNAVAILABLE",
            "error": type(exc).__name__,
        }
    return {
        "registry_contract_version": "ml_registry_set_v1",
        "registry_set_hash": bundle.registry_set_hash,
        "registry_hashes": {
            kind: document.registry_hash
            for kind, document in sorted(bundle.documents.items())
        },
    }


def _authority_versions(
    *,
    dataset_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    sequence_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in dataset_records:
        for key, value in dict(record.get("authority_versions") or {}).items():
            _set_if_meaningful(result, key, value)
    for record in prediction_records:
        for key, value in dict(record.get("authority_versions") or {}).items():
            _set_if_meaningful(result, key, value)
    if sequence_context:
        result["sequence_authority"] = dict(sequence_context)
    if replay_records:
        result["portfolio_replay_identity_version"] = "portfolio_replay_identity_v2_registry"
    return result


def _set_if_meaningful(target: dict[str, Any], key: str, value: Any) -> None:
    if key not in target and value not in (None, ""):
        target[key] = value


def _trial_accounting(
    *,
    trial_family_id: str,
    trial_accounting: Mapping[str, Any] | None,
    ledger_path: Path | None,
    prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = dict(trial_accounting or {})
    if trial_family_id and not payload.get("trial_family_id"):
        payload["trial_family_id"] = trial_family_id
    normalised = normalise_trial_accounting(payload)
    ledger_events: list[dict[str, Any]] = []
    if ledger_path is not None:
        try:
            ledger_events = read_ledger(ledger_path)
        except ValueError:
            ledger_events = []
    experiment_run_ids = sorted(
        {
            str(event.get("experiment_run_id"))
            for event in ledger_events
            if event.get("experiment_run_id")
        }
        | {
            str(item.get("experiment_run_id"))
            for record in prediction_records
            for item in record.get("experiments", [])
            if item.get("experiment_run_id")
        }
    )
    result = {
        **normalised,
        "trial_family_id": trial_family_id,
        "ledger_path": str(ledger_path) if ledger_path else None,
        "ledger_event_count": len(ledger_events),
        "experiment_run_ids": experiment_run_ids,
    }
    if payload.get("evidence_complete") is True:
        result["evidence_complete"] = True
    result["certification_trial_accounting_hash"] = canonical_hash(result)
    return result


def _promotion_evidence(
    *,
    supplied: Mapping[str, Any] | None,
    replay_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if supplied:
        payload = dict(supplied)
        payload.setdefault("evidence_complete", False)
        payload.setdefault("promotion_triggered", False)
        return payload
    replay_promotions = [
        dict(record.get("promotion") or {})
        for record in replay_records
        if record.get("promotion")
    ]
    complete = bool(replay_promotions) and all(
        row.get("promotion_eligible") is True
        and not row.get("blocking_reasons")
        for row in replay_promotions
    )
    return {
        "contract_version": "research_certification_promotion_evidence_v1",
        "evidence_complete": complete,
        "promotion_triggered": False,
        "promotion_eligible_artifact_count": sum(
            row.get("promotion_eligible") is True for row in replay_promotions
        ),
        "blocking_reasons": sorted(
            {
                str(reason)
                for row in replay_promotions
                for reason in row.get("blocking_reasons", []) or []
            }
        ),
    }


def _execution_scenario(
    *,
    config: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
    replay_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if supplied:
        return dict(supplied)
    ml = dict(config.get("ml", {}) or {})
    replay = replay_records[0] if replay_records else {}
    return {
        "execution_assumptions": replay.get("execution_assumptions")
        or {
            "trading_impact": "none",
            "production_validated": False,
            "promotion_automated": False,
            "cost_bps": ml.get("stock_portfolio_replay_cost_bps"),
            "slippage_bps": ml.get("stock_portfolio_replay_slippage_bps"),
            "top_n": ml.get("stock_portfolio_replay_top_n"),
        },
        "no_trade_comparison": _nested_get(
            config,
            ("ml", "research_certification", "no_trade_comparison"),
        ),
        "null_policy": _nested_get(
            config,
            ("ml", "research_certification", "null_policy"),
        ),
        "promotion_prohibition": {
            "automatic_promotion": False,
            "reason": "certification_envelope_never_promotes",
        },
    }


def _execution_assumptions_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        value.get(key) not in (None, "", {})
        for key in ("execution_assumptions", "no_trade_comparison", "null_policy")
    )


def _random_seeds(
    *,
    config: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    seeds: dict[str, Any] = {}
    if ml.get("random_seed") is not None:
        seeds["config.random_seed"] = ml["random_seed"]
    for index, record in enumerate(prediction_records):
        if record.get("random_seed") is not None:
            seeds[f"prediction_artifact[{index}]"] = record["random_seed"]
    return seeds


def _dataset_input_paths(records: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        for row in record.get("source_paths", []) or []:
            if isinstance(row, Mapping) and row.get("path"):
                paths.append(Path(str(row["path"])))
        for row in record.get("source_manifest_hashes", []) or []:
            if isinstance(row, Mapping) and row.get("path"):
                paths.append(Path(str(row["path"])))
    return paths


def _config_input_paths(config: Mapping[str, Any]) -> list[Path]:
    path = config.get("config_path")
    return [Path(str(path))] if path else []


def _file_snapshots(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return sorted(
        (file_identity(path) for path in _unique_paths(paths) if path.exists()),
        key=lambda row: str(row["path"]),
    )


def _output_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return sorted(
        (file_identity(path) for path in _unique_paths(paths) if path.exists()),
        key=lambda row: str(row["path"]),
    )


def _hash_status(current: Any, recorded: Any) -> str:
    if current in (None, "") or recorded in (None, ""):
        return HASH_MISSING
    return HASH_EXACT if str(current).upper() == str(recorded).upper() else HASH_MISMATCH


def _find_authority_value(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
        for child in value.values():
            found = _find_authority_value(child, field)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_authority_value(child, field)
            if found not in (None, ""):
                return found
    return None


def _meaningful_authority(value: Any) -> bool:
    if value in (None, ""):
        return False
    text = str(value).strip().upper()
    return text not in {"UNKNOWN", "UNVERIFIED", "NONE", "NULL"}


def _is_non_pit_universe(value: Any) -> bool:
    if value in (None, ""):
        return False
    text = str(value).strip().upper().replace("-", "_")
    return "NON_PIT" in text or ("STATIC" in text and "PIT" not in text)


def _is_unresolved_identity(value: Any) -> bool:
    if value in (None, ""):
        return False
    text = str(value).strip().upper()
    return any(
        marker in text
        for marker in (
            "UNRESOLVED",
            "UNKNOWN",
            "AMBIGUOUS",
            "LEGACY_SYMBOL_IDENTITY",
        )
    )


def _envelope_identity_payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "contract_version": envelope.get("contract_version"),
        **{
            key: envelope.get(key)
            for key in STANDARD_ENVELOPE_FIELDS
            if key != "run_id"
        },
        "promotion_evidence": envelope.get("promotion_evidence"),
        "diagnostic_legacy": envelope.get("diagnostic_legacy"),
    }
    return payload


def _captured_dirty_patch_hash() -> str | None:
    tracked_patch = _git("diff", "HEAD", "--binary")
    untracked = sorted(filter(None, _git("ls-files", "--others", "--exclude-standard").splitlines()))
    identities = []
    for item in untracked:
        path = Path(item)
        if path.exists() and path.is_file():
            identities.append(file_identity(path))
    if not tracked_patch and not identities:
        return None
    return canonical_hash(
        {
            "tracked_patch": tracked_patch,
            "untracked_file_identities": identities,
        }
    )


def _recorded_commit_status(recorded: str) -> dict[str, Any]:
    current = _git("rev-parse", "HEAD").strip()
    exists = bool(recorded and _git_success("cat-file", "-e", f"{recorded}^{{commit}}"))
    return {
        "recorded_git_sha": recorded or None,
        "current_git_sha": current or None,
        "current_matches_recorded": bool(recorded and current == recorded),
        "recorded_commit_available": exists,
        "checkout_performed": False,
        "checkout_required_for_exact_context": bool(recorded and current != recorded),
    }


def _compare_snapshot(
    row: Mapping[str, Any],
    *,
    path_key: str = "path",
    overrides: Mapping[str, Path],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    original_path = str(row.get(path_key) or row.get("path") or "")
    comparison_path = overrides.get(original_path, Path(original_path))
    expected = row.get("sha256")
    if expected is None:
        expected = row.get("manifest_sha256") or row.get("summary_sha256")
    if not original_path:
        return {"path": original_path, "status": HASH_MISSING, "reason": "RECORDED_PATH_MISSING"}
    if not comparison_path.exists() or not comparison_path.is_file():
        return {
            "path": original_path,
            "comparison_path": str(comparison_path),
            "expected_sha256": expected,
            "actual_sha256": None,
            "status": HASH_MISSING,
            "reason": "ARTIFACT_FILE_MISSING",
        }
    actual = file_sha256(comparison_path)
    if expected and actual == str(expected).upper():
        return {
            "path": original_path,
            "comparison_path": str(comparison_path),
            "expected_sha256": str(expected).upper(),
            "actual_sha256": actual,
            "status": HASH_EXACT,
            "reason": None,
        }
    if tolerance > 0 and comparison_path != Path(original_path):
        if _within_tolerance(Path(original_path), comparison_path, tolerance=tolerance):
            return {
                "path": original_path,
                "comparison_path": str(comparison_path),
                "expected_sha256": str(expected).upper() if expected else None,
                "actual_sha256": actual,
                "status": "TOLERANCE_MATCH",
                "reason": None,
                "tolerance": tolerance,
            }
    return {
        "path": original_path,
        "comparison_path": str(comparison_path),
        "expected_sha256": str(expected).upper() if expected else None,
        "actual_sha256": actual,
        "status": HASH_MISMATCH,
        "reason": "HASH_MISMATCH",
    }


def _within_tolerance(left: Path, right: Path, *, tolerance: float) -> bool:
    if not left.exists() or not right.exists():
        return False
    if left.suffix.lower() == ".json" and right.suffix.lower() == ".json":
        try:
            return _json_within_tolerance(
                json.loads(left.read_text(encoding="utf-8")),
                json.loads(right.read_text(encoding="utf-8")),
                tolerance=tolerance,
            )
        except (OSError, ValueError, TypeError):
            return False
    if left.suffix.lower() == ".csv" and right.suffix.lower() == ".csv":
        try:
            return _csv_within_tolerance(left, right, tolerance=tolerance)
        except (OSError, ValueError, TypeError):
            return False
    return False


def _json_within_tolerance(left: Any, right: Any, *, tolerance: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_json_within_tolerance(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_within_tolerance(a, b, tolerance=tolerance)
            for a, b in zip(left, right)
        )
    if _is_number(left) and _is_number(right):
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
    return left == right


def _csv_within_tolerance(left: Path, right: Path, *, tolerance: float) -> bool:
    with left.open("r", encoding="utf-8", newline="") as handle:
        left_rows = list(csv.DictReader(handle))
    with right.open("r", encoding="utf-8", newline="") as handle:
        right_rows = list(csv.DictReader(handle))
    if len(left_rows) != len(right_rows):
        return False
    for left_row, right_row in zip(left_rows, right_rows):
        if set(left_row) != set(right_row):
            return False
        for key in left_row:
            a = left_row[key]
            b = right_row[key]
            if _is_number(a) and _is_number(b):
                if not math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0.0):
                    return False
            elif a != b:
                return False
    return True


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(float(value))


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    return sorted({Path(path) for path in paths if path not in (None, "")}, key=lambda item: item.as_posix())


def _split_model_ids(value: Any) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _optional_path(value: Any) -> Path | None:
    return Path(str(value)) if value not in (None, "") else None


def _nested_get(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_success(*args: str) -> bool:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
