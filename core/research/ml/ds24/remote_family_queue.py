from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import pandas as pd

from core.research.ml.ds24.ensemble_oof import (
    COMPACT_SCORE_CONTRACT_ID,
    build_compact_oof_v2_manifest,
    compact_oof_v2_contract_payload,
    compact_oof_v2_metadata_from_v1,
    expand_compact_oof_v2_to_v1,
    openable_path,
    prepare_oof_score_frame,
    publish_compact_oof_v2_manifest,
    reproduce_v3_metrics_from_scores,
    stable_hash,
    validate_compact_oof_v2_manifest,
    write_compact_oof_v2_partitions,
    write_csv,
    write_json,
)
from core.research.ml.ds24_metrics_only_evaluator import resolved_performance_contract_v3_hash


QUEUE_ID = "DS24_VAST_REMOTE_NINE_FAMILY_R1"
OWNERSHIP_CONTRACT_ID = "DS24_LOCAL_REMOTE_FAMILY_OWNERSHIP_V1"
PREDICTOR_CONTRACT_ID = "DS24_R44B_101_PREDICTOR_AUTHORITY"
PREDICTOR_CONTRACT_HASH = "ds24_r44b_101_predictor_contract_hash"
TARGET_CONTRACT_ID = "forward_return_60m__decision_5m"
R44B_TFT_CONFIGURATION_HASH = "529058a37fa3731390e4631e9dd97696622e670d699e98411fbc89bfe5a00227"
R44B_SOURCE_BUNDLE_HASH = "6d5cc3f09e24160c7d26cdff29d98546742a6c082928202356b689a94fe62109"

LOCAL_FAMILIES = (
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge",
    "elastic_net",
    "rff_ridge",
    "huber",
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)
REMOTE_QUEUE_ORDER = (
    "temporal_fusion_transformer",
    "market_context_encoder",
    "momentum_transformer",
    "itransformer",
    "transformer",
    "patchtst",
    "dlinear",
    "lightgbm_lambdarank",
    "lightgbm_rank_xendcg",
)
GPU_SEQUENCE_FAMILIES = REMOTE_QUEUE_ORDER[:7]
CPU_RANKING_FAMILIES = REMOTE_QUEUE_ORDER[7:]
ALL_OWNED_FAMILIES = LOCAL_FAMILIES + REMOTE_QUEUE_ORDER
QUEUE_LEDGER_COLUMNS = [
    "queue_ordinal",
    "family",
    "family_class",
    "ownership_lane",
    "certification_state",
    "trial_id",
    "attempt",
    "remote_pid",
    "start_timestamp",
    "last_heartbeat",
    "checkpoint_cursor",
    "metrics_cursor",
    "oof_cursor",
    "terminal_state",
    "result_sync_state",
    "blocker",
]
ACCEPTABLE_CERTIFICATION_STATES = (
    "REMOTE_ADAPTER_CERTIFIED",
    "REMOTE_ADAPTER_READY_FOR_SYNTHETIC_SMOKE",
    "CONFIGURATION_AUTHORITY_REQUIRED",
    "V3_ADAPTER_REQUIRED",
    "OOF_ADAPTER_REQUIRED",
    "LINUX_CUDA_PORTABILITY_REQUIRED",
    "IMPLEMENTATION_BLOCKED",
)
BLOCKING_CERTIFICATION_STATES = {
    "CONFIGURATION_AUTHORITY_REQUIRED",
    "V3_ADAPTER_REQUIRED",
    "OOF_ADAPTER_REQUIRED",
    "LINUX_CUDA_PORTABILITY_REQUIRED",
    "IMPLEMENTATION_BLOCKED",
}
REMOTE_SOURCE_ROOT = PurePosixPath("/workspace/ds24/source")
REMOTE_DATA_AUTHORITY_ROOT = PurePosixPath("/workspace/ds24/data/authority")
REMOTE_QUEUE_ROOT = PurePosixPath("/workspace/ds24/queue")
REMOTE_OUTPUT_ROOT = PurePosixPath("/workspace/ds24/output/remote_vast_runs") / f"queue={QUEUE_ID}"

MODEL_CLASSES = {
    "temporal_fusion_transformer": "core.research.ml.models.temporal_fusion_transformer_model:TemporalFusionTransformerMLModel",
    "market_context_encoder": "core.research.ml.models.market_context_encoder_model:MarketContextEncoderMLModel",
    "momentum_transformer": "core.research.ml.models.momentum_transformer_model:MomentumTransformerSequenceMLModel",
    "itransformer": "core.research.ml.models.itransformer_model:ITransformerSequenceMLModel",
    "transformer": "core.research.ml.models.transformer_model:TransformerSequenceMLModel",
    "patchtst": "core.research.ml.models.patchtst_model:PatchTSTSequenceMLModel",
    "dlinear": "core.research.ml.models.dlinear_model:DLinearSequenceMLModel",
    "lightgbm_lambdarank": "core.research.ml.stock_level.lightgbm_production_selector:FittedLightGBMRanker",
    "lightgbm_rank_xendcg": "core.research.ml.stock_level.lightgbm_production_selector:FittedLightGBMRanker",
}


class RemoteQueueError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFamilyAdapter:
    family: str
    queue_ordinal: int
    family_class: str
    certification_state: str
    trial_id: str
    run_id: str
    model_class: str
    configuration_source: str
    configuration_hash: str
    predictor_contract: str
    predictor_contract_hash: str
    target_contract: str
    target_contract_hash: str
    evaluation_contract: str
    evaluation_contract_hash: str
    train_score_schedule: str
    refit_policy: str
    score_orientation: str
    deterministic_seed_policy: str
    device_policy: str
    checkpoint_policy: str
    v3_evaluation_adapter: str
    ensemble_oof_adapter: str
    full_development_gate: str


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_text(path: Path, text: str) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def family_class(family: str) -> str:
    if family in GPU_SEQUENCE_FAMILIES:
        return "GPU_SEQUENCE"
    if family in CPU_RANKING_FAMILIES:
        return "CPU_RANKING"
    raise RemoteQueueError(f"DS24_R44D_UNKNOWN_REMOTE_FAMILY:{family}")


def queue_ordinal(family: str) -> int:
    if family not in REMOTE_QUEUE_ORDER:
        raise RemoteQueueError(f"DS24_R44D_UNKNOWN_REMOTE_FAMILY:{family}")
    return REMOTE_QUEUE_ORDER.index(family) + 1


def remote_trial_id(family: str) -> str:
    return f"{QUEUE_ID}_{queue_ordinal(family):02d}_{family}_TRIAL_0001"


def assert_remote_family(family: str) -> None:
    if family in LOCAL_FAMILIES:
        raise RemoteQueueError(f"DS24_R44D_LOCAL_FAMILY_REMOTE_LAUNCH_REFUSED:{family}")
    if family not in REMOTE_QUEUE_ORDER:
        raise RemoteQueueError(f"DS24_R44D_UNKNOWN_REMOTE_FAMILY:{family}")


def remote_namespace_for_family(family: str) -> str:
    assert_remote_family(family)
    return str(REMOTE_OUTPUT_ROOT / f"family={family}")


def family_run_root(local_output_root: Path, family: str) -> Path:
    assert_remote_family(family)
    return local_output_root / "remote_vast_runs" / f"queue={QUEUE_ID}" / f"family={family}"


def ownership_payload() -> dict[str, Any]:
    assignments = [
        {
            "family": family,
            "ownership_lane": "local",
            "remote_launch_allowed": False,
            "local_launch_allowed": True,
        }
        for family in LOCAL_FAMILIES
    ] + [
        {
            "family": family,
            "ownership_lane": "vast_remote",
            "remote_launch_allowed": True,
            "local_launch_allowed": False,
        }
        for family in REMOTE_QUEUE_ORDER
    ]
    family_names = [row["family"] for row in assignments]
    payload = {
        "contract_id": OWNERSHIP_CONTRACT_ID,
        "version": 1,
        "assignments": assignments,
        "local_families": list(LOCAL_FAMILIES),
        "vast_remote_families": list(REMOTE_QUEUE_ORDER),
        "all_families_assigned_exactly_once": len(family_names) == len(set(family_names)) == len(ALL_OWNED_FAMILIES),
        "local_supervisor_modified": False,
        "remote_launcher_enforces_authority": True,
        "duplicate_guard_checks": [
            "same_family_active_local_and_remote",
            "identical_trial_ids",
            "overlapping_output_namespaces",
            "imported_results_from_unowned_family",
            "same_refit_package_executed_by_two_remote_workers",
        ],
    }
    payload["authority_hash"] = stable_hash(payload)
    return payload


def queue_order_payload() -> dict[str, Any]:
    rows = [
        {
            "queue_ordinal": ordinal,
            "family": family,
            "family_class": family_class(family),
            "default_concurrency_lane": "gpu_singleton" if family in GPU_SEQUENCE_FAMILIES else "cpu_ranking_deferred",
        }
        for ordinal, family in enumerate(REMOTE_QUEUE_ORDER, start=1)
    ]
    payload = {
        "queue_id": QUEUE_ID,
        "queue_order": rows,
        "starts_with": REMOTE_QUEUE_ORDER[0],
        "bounded_retry_counts_required": True,
        "blocked_family_never_skipped_silently": True,
        "user_defer_supported": True,
        "durable_ledger_columns": list(QUEUE_LEDGER_COLUMNS),
    }
    payload["queue_hash"] = stable_hash(payload)
    return payload


def registry_entries(repo_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(repo_root / "config/ml_registries/selector_models.v1.json")
    return {
        str(row.get("canonical_id")): row
        for row in payload.get("entries", [])
        if isinstance(row, dict) and row.get("canonical_id") in set(REMOTE_QUEUE_ORDER)
    }


def _family_config_path(repo_root: Path, family: str) -> Path:
    return repo_root / "config/ticket_63_wave2_model_family" / f"{family}_daily_v1.json"


def _configuration_source(repo_root: Path, family: str) -> tuple[str, dict[str, Any]]:
    if family == "temporal_fusion_transformer":
        path = repo_root / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44b_vast_ai_isolated_remote_tft_execution/04_tft_configuration_authority.yaml"
        return repo_relative(repo_root, path), {
            "configuration_hash": R44B_TFT_CONFIGURATION_HASH,
            "target_identity": TARGET_CONTRACT_ID,
            "predictor_count": 101,
        }
    if family in CPU_RANKING_FAMILIES:
        return "config/ml_registries/selector_models.v1.json:fitting_configuration_checksum", {}
    path = _family_config_path(repo_root, family)
    return repo_relative(repo_root, path), read_json(path)


def _configuration_hash(repo_root: Path, family: str, entry: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    if family == "temporal_fusion_transformer":
        return R44B_TFT_CONFIGURATION_HASH
    if family in CPU_RANKING_FAMILIES:
        return str(entry.get("fitting_configuration_checksum", ""))
    path = _family_config_path(repo_root, family)
    file_hash = sha256_file(path) if path.exists() else ""
    return stable_hash(
        {
            "family": family,
            "registry_entry": entry,
            "ticket63_daily_architecture_config": config,
            "ticket63_config_sha256": file_hash,
            "ds24_queue_target_contract": TARGET_CONTRACT_ID,
            "ds24_queue_predictor_contract": PREDICTOR_CONTRACT_ID,
        }
    )


def _certification_state(family: str, entry: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    if family == "temporal_fusion_transformer":
        return "REMOTE_ADAPTER_CERTIFIED"
    if family in CPU_RANKING_FAMILIES:
        ready = bool(entry) and entry.get("readiness_status") == "CAMPAIGN_READY" and entry.get("strict_oos_capable") is True
        return "REMOTE_ADAPTER_READY_FOR_SYNTHETIC_SMOKE" if ready else "CONFIGURATION_AUTHORITY_REQUIRED"
    if not entry or not config:
        return "CONFIGURATION_AUTHORITY_REQUIRED"
    if entry.get("implementation_status") not in {"IMPLEMENTED_BUT_UNVALIDATED", "IMPLEMENTED_AND_AUTHORITATIVE_RUNNABLE"}:
        return "IMPLEMENTATION_BLOCKED"
    return "REMOTE_ADAPTER_READY_FOR_SYNTHETIC_SMOKE"


def family_configuration_authority(repo_root: Path) -> dict[str, Any]:
    entries = registry_entries(repo_root)
    rows: list[dict[str, Any]] = []
    for family in REMOTE_QUEUE_ORDER:
        entry = entries.get(family, {})
        source, config = _configuration_source(repo_root, family)
        state = _certification_state(family, entry, config)
        config_hash = _configuration_hash(repo_root, family, entry, config)
        row = {
            "family": family,
            "queue_ordinal": queue_ordinal(family),
            "family_class": family_class(family),
            "model_class": MODEL_CLASSES[family],
            "implementation_owner": entry.get("implementation_owner", MODEL_CLASSES[family]),
            "configuration_source": source,
            "configuration_hash": config_hash,
            "registry_implementation_status": entry.get("implementation_status", ""),
            "existing_default_target_identity": config.get("target_identity", entry.get("target_contract", "")),
            "predictor_contract": PREDICTOR_CONTRACT_ID,
            "predictor_contract_hash": PREDICTOR_CONTRACT_HASH,
            "target_contract": TARGET_CONTRACT_ID,
            "target_contract_hash": stable_hash({"target_contract": TARGET_CONTRACT_ID}),
            "evaluation_contract": "RESOLVED_PERFORMANCE_CONTRACT_V3",
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
            "train_score_schedule": "daily-session refit with five-minute DS24 decision scoring",
            "refit_policy": "daily_session_v1; first uncommitted timestamp resume",
            "score_orientation": "higher long_selection_score is better",
            "deterministic_seed_policy": "single fixed seed per trial plus deterministic trial_id/run_id namespace",
            "device_policy": (
                "one CUDA GPU for paid smoke/full sequence execution; CPU allowed only for synthetic adapter proof"
                if family in GPU_SEQUENCE_FAMILIES
                else "single CPU worker, n_jobs=1, optional alongside GPU only after resource smoke certification"
            ),
            "checkpoint_policy": (
                "latest/previous atomic checkpoints with hash verification"
                if family in GPU_SEQUENCE_FAMILIES
                else "completed refit package cursor plus model artifact hash verification"
            ),
            "v3_evaluation_adapter": "core.research.ml.ds24_metrics_only_evaluator:MetricsOnlyEvidenceWriter",
            "ensemble_oof_adapter": f"core.research.ml.ds24.ensemble_oof:{COMPACT_SCORE_CONTRACT_ID}",
            "certification_state": state,
            "authority_notes": (
                "R44B/R44C frozen DS24 5m TFT authority"
                if family == "temporal_fusion_transformer"
                else "existing repository architecture/defaults are bound to the DS24 5m queue target for smoke certification; no performance tuning used"
            ),
            "full_development_gate": (
                "paid CUDA smoke plus explicit full-history user approval"
                if family in GPU_SEQUENCE_FAMILIES
                else "paid CPU smoke or certified GPU+CPU concurrency profile plus explicit full-history user approval"
            ),
        }
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["certification_state"]] = counts.get(row["certification_state"], 0) + 1
    blocking = [row for row in rows if row["certification_state"] in BLOCKING_CERTIFICATION_STATES]
    payload = {
        "authority_id": "DS24_R44D_NINE_FAMILY_CONFIGURATION_AUTHORITY_V1",
        "queue_id": QUEUE_ID,
        "target_contract": TARGET_CONTRACT_ID,
        "predictor_contract": PREDICTOR_CONTRACT_ID,
        "families": rows,
        "classification_counts": counts,
        "blocking_families": [row["family"] for row in blocking],
        "no_scientific_hyperparameter_tuning": True,
        "tft_r44b_configuration_frozen": True,
        "status": "PASS" if not blocking else "FAIL",
    }
    payload["authority_hash"] = stable_hash(payload)
    return payload


def adapter_registry(repo_root: Path) -> dict[str, RemoteFamilyAdapter]:
    authority = family_configuration_authority(repo_root)
    adapters: dict[str, RemoteFamilyAdapter] = {}
    for row in authority["families"]:
        if row["certification_state"] in BLOCKING_CERTIFICATION_STATES:
            continue
        adapters[row["family"]] = RemoteFamilyAdapter(
            family=row["family"],
            queue_ordinal=int(row["queue_ordinal"]),
            family_class=row["family_class"],
            certification_state=row["certification_state"],
            trial_id=remote_trial_id(row["family"]),
            run_id=QUEUE_ID,
            model_class=row["model_class"],
            configuration_source=row["configuration_source"],
            configuration_hash=row["configuration_hash"],
            predictor_contract=row["predictor_contract"],
            predictor_contract_hash=row["predictor_contract_hash"],
            target_contract=row["target_contract"],
            target_contract_hash=row["target_contract_hash"],
            evaluation_contract=row["evaluation_contract"],
            evaluation_contract_hash=row["evaluation_contract_hash"],
            train_score_schedule=row["train_score_schedule"],
            refit_policy=row["refit_policy"],
            score_orientation=row["score_orientation"],
            deterministic_seed_policy=row["deterministic_seed_policy"],
            device_policy=row["device_policy"],
            checkpoint_policy=row["checkpoint_policy"],
            v3_evaluation_adapter=row["v3_evaluation_adapter"],
            ensemble_oof_adapter=row["ensemble_oof_adapter"],
            full_development_gate=row["full_development_gate"],
        )
    return adapters


def common_remote_worker_contract() -> dict[str, Any]:
    fields = [
        "family",
        "trial_id",
        "run_id",
        "resume",
        "checkpoint_root",
        "metrics_root",
        "ensemble_oof_root",
        "dataset_manifest",
        "predictor_contract",
        "target_contract",
        "evaluation_contract",
        "execution_profile",
        "bounded_smoke",
        "full_development",
    ]
    payload = {
        "contract_id": "DS24_R44D_COMMON_REMOTE_WORKER_INTERFACE_V1",
        "queue_id": QUEUE_ID,
        "fields": fields,
        "family_neutral_supervisor": True,
        "family_specific_training_logic_location": "adapter implementation behind common interface",
        "local_supervisor_import_or_control": False,
        "full_development_requires_user_approval": True,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def initial_queue_ledger(certification_by_family: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    certification_by_family = certification_by_family or {}
    rows: list[dict[str, Any]] = []
    for ordinal, family in enumerate(REMOTE_QUEUE_ORDER, start=1):
        rows.append(
            {
                "queue_ordinal": ordinal,
                "family": family,
                "family_class": family_class(family),
                "ownership_lane": "vast_remote",
                "certification_state": certification_by_family.get(family, "REMOTE_ADAPTER_READY_FOR_SYNTHETIC_SMOKE"),
                "trial_id": remote_trial_id(family),
                "attempt": 0,
                "remote_pid": "",
                "start_timestamp": "",
                "last_heartbeat": "",
                "checkpoint_cursor": "",
                "metrics_cursor": "",
                "oof_cursor": "",
                "terminal_state": "PENDING",
                "result_sync_state": "NOT_STARTED",
                "blocker": "",
            }
        )
    return rows


def validate_queue_ledger(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    required_columns = set(QUEUE_LEDGER_COLUMNS)
    bad_columns = [row for row in records if set(row) != required_columns]
    bad_families = [row.get("family", "") for row in records if row.get("family") not in REMOTE_QUEUE_ORDER]
    ordinals = [int(row.get("queue_ordinal", 0)) for row in records]
    trial_ids = [str(row.get("trial_id", "")) for row in records]
    status = not bad_columns and not bad_families and ordinals == list(range(1, len(REMOTE_QUEUE_ORDER) + 1)) and len(trial_ids) == len(set(trial_ids))
    return {
        "status": "PASS" if status else "FAIL",
        "row_count": len(records),
        "columns": list(QUEUE_LEDGER_COLUMNS),
        "bad_column_rows": len(bad_columns),
        "bad_families": bad_families,
        "unique_trial_ids": len(trial_ids) == len(set(trial_ids)),
    }


def write_queue_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, [{column: row.get(column, "") for column in QUEUE_LEDGER_COLUMNS} for row in rows])


def read_queue_ledger(path: Path) -> list[dict[str, Any]]:
    with open(openable_path(path), "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_active(row: Mapping[str, Any]) -> bool:
    state = str(row.get("terminal_state") or row.get("state") or "").upper()
    return state in {"", "ACTIVE", "CLAIMED", "RUNNING", "STARTING"}


def _namespace_overlap(left: str, right: str) -> bool:
    a = left.rstrip("/\\").replace("\\", "/")
    b = right.rstrip("/\\").replace("\\", "/")
    return bool(a and b and (a == b or a.startswith(b + "/") or b.startswith(a + "/")))


def duplicate_guard(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    violations: list[dict[str, Any]] = []
    active = [row for row in rows if _is_active(row)]
    for family in LOCAL_FAMILIES:
        local = [row for row in active if row.get("family") == family and row.get("ownership_lane") == "local"]
        remote = [row for row in active if row.get("family") == family and row.get("ownership_lane") == "vast_remote"]
        if local and remote:
            violations.append({"code": "SAME_FAMILY_ACTIVE_LOCAL_AND_REMOTE", "family": family})
    trial_seen: dict[str, dict[str, Any]] = {}
    for row in active:
        trial = str(row.get("trial_id", ""))
        if not trial:
            continue
        if trial in trial_seen:
            violations.append({"code": "IDENTICAL_TRIAL_ID", "trial_id": trial})
        trial_seen[trial] = row
    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            if _namespace_overlap(str(left.get("output_namespace", "")), str(right.get("output_namespace", ""))):
                violations.append(
                    {
                        "code": "OVERLAPPING_OUTPUT_NAMESPACE",
                        "left_family": left.get("family", ""),
                        "right_family": right.get("family", ""),
                    }
                )
    for row in rows:
        if row.get("record_type") == "imported_result" and row.get("family") not in REMOTE_QUEUE_ORDER:
            violations.append({"code": "IMPORTED_RESULT_FROM_UNOWNED_FAMILY", "family": row.get("family", "")})
    refit_seen: dict[str, dict[str, Any]] = {}
    for row in active:
        if row.get("ownership_lane") != "vast_remote":
            continue
        refit = str(row.get("refit_package_id") or row.get("refit_package") or "")
        if not refit:
            continue
        if refit in refit_seen:
            violations.append({"code": "DUPLICATE_REMOTE_REFIT_PACKAGE", "refit_package_id": refit})
        refit_seen[refit] = row
    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "records_checked": len(rows),
        "active_records_checked": len(active),
    }


def namespace_contract_payload() -> dict[str, Any]:
    rows = [
        {
            "family": family,
            "source_root": str(REMOTE_SOURCE_ROOT),
            "data_authority_root": str(REMOTE_DATA_AUTHORITY_ROOT),
            "queue_root": str(REMOTE_QUEUE_ROOT),
            "output_root": remote_namespace_for_family(family),
            "checkpoint_root": str(PurePosixPath(remote_namespace_for_family(family)) / "checkpoints"),
            "metrics_root": str(PurePosixPath(remote_namespace_for_family(family)) / "metrics_only_v3"),
            "ensemble_oof_root": str(PurePosixPath(remote_namespace_for_family(family)) / "ensemble_oof_scores_v2"),
        }
        for family in REMOTE_QUEUE_ORDER
    ]
    payload = {
        "contract_id": "DS24_R44D_REMOTE_NAMESPACE_CONTRACT_V1",
        "shared_source_root": str(REMOTE_SOURCE_ROOT),
        "shared_read_only_data_authority_root": str(REMOTE_DATA_AUTHORITY_ROOT),
        "queue_root": str(REMOTE_QUEUE_ROOT),
        "family_namespaces": rows,
        "local_active_worker_roots_modified": False,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def resource_classification_payload() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44D_RESOURCE_CLASSIFICATION_V1",
        "gpu_sequence_families": list(GPU_SEQUENCE_FAMILIES),
        "cpu_ranking_families": list(CPU_RANKING_FAMILIES),
        "default_safe_concurrency": {
            "gpu_workers": 1,
            "cpu_ranking_workers": 0,
            "total_active_families": 1,
            "status": "CERTIFIED_DEFAULT",
        },
        "certified_gpu_plus_cpu_concurrency": {
            "gpu_workers": 1,
            "cpu_ranking_workers": 1,
            "total_active_families": 2,
            "status": "PAID_SMOKE_REQUIRED",
            "criteria": [
                "host RAM retains at least 12 GiB headroom",
                "no swap thrashing",
                "GPU throughput falls by less than 10 percent",
                "CPU model throughput remains useful",
                "disk I/O is not saturated",
                "checkpoints remain timely",
                "output growth remains inside budget",
            ],
        },
        "multi_gpu_allowed": False,
        "do_not_assume_parallel_beneficial": True,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def admit_resource_profile(
    *,
    gpu_workers: int,
    cpu_ranking_workers: int,
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = observed or {}
    if gpu_workers == 1 and cpu_ranking_workers == 0:
        return {"accepted": True, "profile": "DEFAULT_SAFE_SERIAL", "reason": "default safe R44D profile"}
    criteria = {
        "host_ram_headroom_gib": float(observed.get("host_ram_headroom_gib", 0.0)) >= 12.0,
        "swap_thrashing": observed.get("swap_thrashing", True) is False,
        "gpu_throughput_drop_fraction": float(observed.get("gpu_throughput_drop_fraction", 1.0)) < 0.10,
        "cpu_throughput_useful": observed.get("cpu_throughput_useful", False) is True,
        "disk_io_saturated": observed.get("disk_io_saturated", True) is False,
        "checkpoints_timely": observed.get("checkpoints_timely", False) is True,
        "output_growth_inside_budget": observed.get("output_growth_inside_budget", False) is True,
    }
    accepted = gpu_workers == 1 and cpu_ranking_workers == 1 and all(criteria.values())
    return {
        "accepted": accepted,
        "profile": "GPU_PLUS_CPU_RANKING" if gpu_workers == 1 and cpu_ranking_workers == 1 else "UNSUPPORTED",
        "criteria": criteria,
        "blocker": "" if accepted else "DS24_R44D_RESOURCE_PROFILE_PAID_SMOKE_REQUIRED_OR_FAILED",
    }


def _synthetic_predictions(family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = float(queue_ordinal(family)) * 0.001
    for day_index, day in enumerate(("2024-01-02", "2024-01-03")):
        for asset_index, asset in enumerate(("AAA", "BBB", "CCC")):
            rows.append(
                {
                    "family": family,
                    "decision_timestamp": f"{day}T14:{35 + 5 * day_index:02d}:00Z",
                    "asset_id": asset,
                    "prediction": float(0.2 + asset_index * 0.1 + day_index * 0.01 + offset),
                }
            )
    return pd.DataFrame(rows)


def run_synthetic_family_smoke(adapter: RemoteFamilyAdapter, output_root: Path) -> dict[str, Any]:
    assert_remote_family(adapter.family)
    run_root = family_run_root(output_root, adapter.family)
    scores_v1 = prepare_oof_score_frame(
        _synthetic_predictions(adapter.family),
        trial_id=adapter.trial_id,
        run_id=adapter.run_id,
        family=adapter.family,
        training_cutoff_timestamp="2024-01-01T21:00:00Z",
        refit_id="refit-000001",
        refit_ordinal=1,
        model_config_hash=adapter.configuration_hash,
        dataset_manifest_hash="synthetic_shared_data_manifest_hash",
        predictor_contract_hash=adapter.predictor_contract_hash,
        target_contract_hash=adapter.target_contract_hash,
        evaluation_contract_hash=adapter.evaluation_contract_hash,
    )
    metadata = compact_oof_v2_metadata_from_v1(
        scores_v1,
        source_bundle_hash=R44B_SOURCE_BUNDLE_HASH if adapter.family == "temporal_fusion_transformer" else "",
        terminal_completeness_state="SYNTHETIC_CONTRACT_COMPLETE",
        provisional=True,
    )
    ledger = write_compact_oof_v2_partitions(run_root, scores_v1)
    manifest = build_compact_oof_v2_manifest(run_root, ledger, metadata=metadata)
    publish_compact_oof_v2_manifest(run_root, manifest)
    compact_validation = validate_compact_oof_v2_manifest(run_root, manifest)
    compact_frames = [
        pd.read_parquet(openable_path(run_root / row["relative_path"]))
        for row in manifest.get("files", [])
    ]
    compact_frame = pd.concat(compact_frames, ignore_index=True)
    expanded = expand_compact_oof_v2_to_v1(compact_frame, manifest)
    score_columns = ["decision_timestamp", "asset_id", "long_selection_score", "cross_sectional_rank"]
    equivalence = (
        scores_v1[score_columns].astype(str).to_dict("records")
        == expanded[score_columns].astype(str).to_dict("records")
    )
    targets = pd.DataFrame(
        {
            "decision_timestamp": scores_v1["decision_timestamp"].astype(str),
            "asset_id": scores_v1["asset_id"].astype(str),
            "target_value": scores_v1["long_selection_score"].astype(float) * 0.01,
            "target_available_timestamp": (
                pd.to_datetime(scores_v1["decision_timestamp"], utc=True) + pd.Timedelta(minutes=60)
            ).astype(str),
            "target_is_trainable": True,
        }
    )
    v3_metrics = reproduce_v3_metrics_from_scores(scores_v1, targets, top_n=2)
    write_json(
        run_root / "metrics_only_v3" / "resolved_performance_summary_v3.json",
        {
            "family": adapter.family,
            "trial_id": adapter.trial_id,
            "run_id": adapter.run_id,
            "evaluation_contract_hash": adapter.evaluation_contract_hash,
            "mean_spearman_rank_ic": v3_metrics["mean_spearman_rank_ic"],
            "synthetic_only": True,
            "status": v3_metrics["status"],
        },
    )
    status = "PASS" if compact_validation["valid"] and v3_metrics["status"] == "PASS" and equivalence else "FAIL"
    return {
        "family": adapter.family,
        "status": status,
        "certification_state": adapter.certification_state,
        "run_root": str(run_root),
        "trial_id": adapter.trial_id,
        "v3_metrics_status": v3_metrics["status"],
        "v3_metrics": v3_metrics,
        "compact_oof_contract_id": COMPACT_SCORE_CONTRACT_ID,
        "compact_manifest_hash": manifest["manifest_hash"],
        "compact_validation": compact_validation,
        "expanded_v2_matches_v1_scores": equivalence,
        "row_count": int(len(scores_v1)),
        "partition_count": int(manifest["partition_count"]),
        "terminal_success_requires_v3_and_oof": True,
        "paid_hardware_run": False,
        "full_history_run": False,
    }


def run_all_synthetic_smokes(repo_root: Path, output_root: Path) -> dict[str, Any]:
    adapters = adapter_registry(repo_root)
    missing = [family for family in REMOTE_QUEUE_ORDER if family not in adapters]
    if missing:
        raise RemoteQueueError("DS24_R44D_REMOTE_ADAPTER_CERTIFICATION_BLOCKED:" + ",".join(missing))
    results = [run_synthetic_family_smoke(adapters[family], output_root) for family in REMOTE_QUEUE_ORDER]
    payload = {
        "smoke_id": "DS24_R44D_NINE_FAMILY_SYNTHETIC_V3_OOF_SMOKE_V1",
        "queue_id": QUEUE_ID,
        "families": results,
        "all_remote_families_exercised": [row["family"] for row in results] == list(REMOTE_QUEUE_ORDER),
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def per_family_storage_projection() -> list[dict[str, Any]]:
    rows_per_decision_day = 514 * 78
    decision_days = 252 * 14
    compressed_bytes_per_row = 48
    partition_overhead_per_day = 16 * 1024
    score_bytes = decision_days * (rows_per_decision_day * compressed_bytes_per_row + partition_overhead_per_day)
    rows: list[dict[str, Any]] = []
    for family in REMOTE_QUEUE_ORDER:
        if family == "temporal_fusion_transformer":
            checkpoint_bytes = 8 * 1024**3
        elif family in GPU_SEQUENCE_FAMILIES:
            checkpoint_bytes = 3 * 1024**3
        else:
            checkpoint_bytes = 512 * 1024**2
        metrics_bytes = 512 * 1024**2 if family in GPU_SEQUENCE_FAMILIES else 256 * 1024**2
        total = int(score_bytes + checkpoint_bytes + metrics_bytes)
        rows.append(
            {
                "family": family,
                "family_class": family_class(family),
                "estimated_oof_rows": decision_days * rows_per_decision_day,
                "compact_oof_v2_bytes": int(score_bytes),
                "checkpoint_budget_bytes": int(checkpoint_bytes),
                "metrics_budget_bytes": int(metrics_bytes),
                "estimated_total_remote_output_bytes": total,
                "external_oof_recommended": True,
                "full_oof_copy_to_current_c_drive_allowed": False,
            }
        )
    return rows


def full_queue_storage_projection() -> dict[str, Any]:
    rows = per_family_storage_projection()
    remote_output = sum(int(row["estimated_total_remote_output_bytes"]) for row in rows)
    shared_data = 47_297_267_964
    source_and_env = 5 * 1024**3
    required_remote_disk = remote_output + shared_data + source_and_env
    payload = {
        "projection_id": "DS24_R44D_FULL_QUEUE_STORAGE_PROJECTION_V1",
        "queue_id": QUEUE_ID,
        "compact_oof_contract": compact_oof_v2_contract_payload(),
        "remote_family_output_bytes": int(remote_output),
        "shared_data_transfer_once_bytes": int(shared_data),
        "source_environment_budget_bytes": int(source_and_env),
        "required_remote_disk_before_headroom_bytes": int(required_remote_disk),
        "minimum_offer_disk_bytes": 250 * 1024**3,
        "accepted_under_minimum_offer_disk": required_remote_disk <= 250 * 1024**3,
        "local_c_drive_policy": "manifest/checkpoint/metrics only unless external OOF destination is supplied",
        "external_complete_oof_store_required": True,
    }
    payload["status"] = "PASS" if payload["accepted_under_minimum_offer_disk"] else "FAIL"
    payload["projection_hash"] = stable_hash(payload)
    return payload


def register_external_oof_manifest(
    *,
    family: str,
    manifest_hash: str,
    external_uri: str,
    row_count: int,
    total_bytes: int,
) -> dict[str, Any]:
    assert_remote_family(family)
    if not manifest_hash or not external_uri:
        raise RemoteQueueError("DS24_R44D_EXTERNAL_OOF_MANIFEST_REGISTRATION_INCOMPLETE")
    payload = {
        "status": "PASS",
        "family": family,
        "manifest_hash": manifest_hash,
        "external_uri": external_uri,
        "row_count": int(row_count),
        "total_bytes": int(total_bytes),
        "copied_to_c_drive": False,
        "registered_by_manifest_without_duplication": True,
    }
    payload["registration_hash"] = stable_hash(payload)
    return payload


class RemoteQueueSupervisor:
    def __init__(self, queue_root: Path, *, max_attempts: int = 2) -> None:
        self.queue_root = queue_root
        self.max_attempts = int(max_attempts)
        self.state_path = queue_root / "queue_state.json"
        self.ledger_path = queue_root / "queue_ledger.csv"

    def initialise(self, certification_by_family: Mapping[str, str] | None = None) -> dict[str, Any]:
        os.makedirs(openable_path(self.queue_root), exist_ok=True)
        state = {
            "queue_id": QUEUE_ID,
            "max_attempts": self.max_attempts,
            "ledger": initial_queue_ledger(certification_by_family),
            "history": [],
        }
        self.save(state)
        return state

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self.initialise()
        return read_json(self.state_path)

    def save(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["ledger"] = [
            {column: dict(row).get(column, "") for column in QUEUE_LEDGER_COLUMNS}
            for row in state.get("ledger", [])
        ]
        write_json(self.state_path, payload)
        write_queue_ledger(self.ledger_path, payload["ledger"])

    def claim_next_family(self, *, now: str = "2026-08-31T00:00:00Z", remote_pid: int | str = "") -> dict[str, Any] | None:
        state = self.load()
        for row in state["ledger"]:
            if row["terminal_state"] == "RUNNING":
                return row
        for row in state["ledger"]:
            if row["terminal_state"] in {"BLOCKED", "DEFERRED", "SUCCESS"}:
                if row["terminal_state"] == "BLOCKED":
                    return None
                continue
            if row["terminal_state"] in {"PENDING", "FAILED"}:
                if int(row["attempt"]) >= self.max_attempts:
                    row["terminal_state"] = "BLOCKED"
                    row["blocker"] = "DS24_R44D_BOUNDED_RETRY_COUNT_EXHAUSTED"
                    self.save(state)
                    return None
                row["attempt"] = int(row["attempt"]) + 1
                row["remote_pid"] = str(remote_pid or os.getpid())
                row["start_timestamp"] = row["start_timestamp"] or now
                row["last_heartbeat"] = now
                row["terminal_state"] = "RUNNING"
                row["result_sync_state"] = "NOT_STARTED"
                state["history"].append({"event": "CLAIM", "family": row["family"], "attempt": row["attempt"], "timestamp": now})
                self.save(state)
                return row
        self.save(state)
        return None

    def heartbeat(
        self,
        family: str,
        *,
        now: str,
        checkpoint_cursor: str = "",
        metrics_cursor: str = "",
        oof_cursor: str = "",
    ) -> dict[str, Any]:
        state = self.load()
        for row in state["ledger"]:
            if row["family"] == family:
                row["last_heartbeat"] = now
                row["checkpoint_cursor"] = checkpoint_cursor or row["checkpoint_cursor"]
                row["metrics_cursor"] = metrics_cursor or row["metrics_cursor"]
                row["oof_cursor"] = oof_cursor or row["oof_cursor"]
                state["history"].append({"event": "HEARTBEAT", "family": family, "timestamp": now})
                self.save(state)
                return row
        raise RemoteQueueError(f"DS24_R44D_HEARTBEAT_UNKNOWN_FAMILY:{family}")

    def mark_family_failed(self, family: str, blocker: str, *, now: str = "2026-08-31T00:00:00Z") -> dict[str, Any]:
        state = self.load()
        for row in state["ledger"]:
            if row["family"] == family:
                row["terminal_state"] = "BLOCKED" if int(row["attempt"]) >= self.max_attempts else "FAILED"
                row["blocker"] = blocker
                row["last_heartbeat"] = now
                state["history"].append({"event": row["terminal_state"], "family": family, "blocker": blocker, "timestamp": now})
                self.save(state)
                return row
        raise RemoteQueueError(f"DS24_R44D_FAIL_UNKNOWN_FAMILY:{family}")

    def mark_family_complete(
        self,
        family: str,
        *,
        checkpoint_cursor: str,
        metrics_cursor: str,
        oof_cursor: str,
        now: str = "2026-08-31T00:00:00Z",
    ) -> dict[str, Any]:
        if metrics_cursor and not oof_cursor:
            raise RemoteQueueError("DS24_R44D_TERMINAL_SUCCESS_REQUIRES_OOF")
        if oof_cursor and not metrics_cursor:
            raise RemoteQueueError("DS24_R44D_TERMINAL_SUCCESS_REQUIRES_V3_METRICS")
        state = self.load()
        for row in state["ledger"]:
            if row["family"] == family:
                row["checkpoint_cursor"] = checkpoint_cursor
                row["metrics_cursor"] = metrics_cursor
                row["oof_cursor"] = oof_cursor
                row["terminal_state"] = "SUCCESS"
                row["result_sync_state"] = "READY_FOR_SYNC"
                row["last_heartbeat"] = now
                row["blocker"] = ""
                state["history"].append({"event": "SUCCESS", "family": family, "timestamp": now})
                self.save(state)
                return row
        raise RemoteQueueError(f"DS24_R44D_COMPLETE_UNKNOWN_FAMILY:{family}")

    def skip_blocked_family(self, family: str, reason: str, *, now: str = "2026-08-31T00:00:00Z") -> dict[str, Any]:
        state = self.load()
        for row in state["ledger"]:
            if row["family"] == family:
                if row["terminal_state"] != "BLOCKED":
                    raise RemoteQueueError("DS24_R44D_SKIP_REQUIRES_BLOCKED_FAMILY")
                row["terminal_state"] = "DEFERRED"
                row["result_sync_state"] = "DEFERRED_BY_USER"
                row["blocker"] = reason
                row["last_heartbeat"] = now
                state["history"].append({"event": "DEFERRED", "family": family, "reason": reason, "timestamp": now})
                self.save(state)
                return row
        raise RemoteQueueError(f"DS24_R44D_SKIP_UNKNOWN_FAMILY:{family}")


def queue_resume_determinism_proof(root: Path) -> dict[str, Any]:
    supervisor = RemoteQueueSupervisor(root, max_attempts=2)
    state = supervisor.initialise({"temporal_fusion_transformer": "REMOTE_ADAPTER_CERTIFIED"})
    first = supervisor.claim_next_family(now="2026-08-31T01:00:00Z", remote_pid=1234)
    if first is None:
        raise RemoteQueueError("DS24_R44D_QUEUE_CLAIM_FAILED")
    supervisor.heartbeat(
        "temporal_fusion_transformer",
        now="2026-08-31T01:05:00Z",
        checkpoint_cursor="checkpoint:0001",
        metrics_cursor="metrics:0001",
        oof_cursor="oof:0001",
    )
    restarted = RemoteQueueSupervisor(root, max_attempts=2)
    resumed = restarted.claim_next_family(now="2026-08-31T01:10:00Z", remote_pid=5678)
    running_preserved = resumed is not None and resumed["family"] == "temporal_fusion_transformer" and str(resumed["remote_pid"]) == "1234"
    restarted.mark_family_failed("temporal_fusion_transformer", "synthetic forced failure", now="2026-08-31T01:15:00Z")
    retry = restarted.claim_next_family(now="2026-08-31T01:20:00Z", remote_pid=5678)
    restarted.mark_family_failed("temporal_fusion_transformer", "synthetic forced failure", now="2026-08-31T01:25:00Z")
    blocked_claim = restarted.claim_next_family(now="2026-08-31T01:30:00Z", remote_pid=5678)
    blocked = restarted.load()["ledger"][0]["terminal_state"] == "BLOCKED" and blocked_claim is None
    deferred = restarted.skip_blocked_family(
        "temporal_fusion_transformer",
        "user-marked deferred after bounded retry proof",
        now="2026-08-31T01:35:00Z",
    )
    next_family = restarted.claim_next_family(now="2026-08-31T01:40:00Z", remote_pid=9012)
    validation = validate_queue_ledger(restarted.load()["ledger"])
    result = {
        "queue_id": QUEUE_ID,
        "initial_row_count": len(state["ledger"]),
        "running_family_preserved_after_process_restart": bool(running_preserved),
        "retry_family": "" if retry is None else retry["family"],
        "bounded_retry_exhaustion_blocks_family": bool(blocked),
        "user_defer_allows_next_family": deferred["terminal_state"] == "DEFERRED"
        and next_family is not None
        and next_family["family"] == "market_context_encoder",
        "queue_ledger_validation": validation,
    }
    result["status"] = "PASS" if all(
        [
            result["running_family_preserved_after_process_restart"],
            result["retry_family"] == "temporal_fusion_transformer",
            result["bounded_retry_exhaustion_blocks_family"],
            result["user_defer_allows_next_family"],
            validation["status"] == "PASS",
        ]
    ) else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result
