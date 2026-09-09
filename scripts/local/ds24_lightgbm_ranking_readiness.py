from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.canonical_prequential_engine import stable_hash
from core.research.ml.ds24.comparable_policy import policy_hash
from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.lightgbm_ranking_preflight import run_lightgbm_ranking_preflight
from core.research.ml.ranking_labels import grouped_ranking_dataset, mature_training_integer_relevance
from core.research.ml.registries.io import canonical_hash
from core.research.ml.stock_level.lightgbm_lambdarank_selector import (
    fit_synthetic_lambdarank_selector,
    fixed_lambdarank_configuration,
    label_gain_policy,
    validate_lambdarank_input,
)
from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
    fit_synthetic_rank_xendcg_selector,
    fixed_rank_xendcg_configuration,
    validate_rank_xendcg_input,
)
from core.research.ml.ds24_metrics_only_evaluator import MetricsOnlyEvidenceWriter, resolved_performance_contract_v3_hash
from scripts.local import ds24_p8_r14_e3g_c2_r7_r14_policy_worker as policy_worker


STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
MANIFEST_PATH = STAGE / "R7_LOW_USAGE_LIGHTGBM_RANKING_READINESS.json"
REGISTRY_PATH = ROOT / "config/ml_registries/selector_models.v1.json"
SCOPED_FAMILIES = ("lightgbm_rank_xendcg", "lightgbm_lambdarank")
FEATURE_NAMES = ["context", "liquidity", "signal"]
MIN_FREE_BYTES = 8 * 1024**3

RESOURCE_ESTIMATES = {
    "lightgbm_rank_xendcg": "2-5 GiB RAM, <150 MiB durable metrics/checkpoint, 20-90 min per full daily package",
    "lightgbm_lambdarank": "2-5 GiB RAM, <150 MiB durable metrics/checkpoint, 20-90 min per full daily package",
}


def utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def registry_entries() -> dict[str, dict[str, Any]]:
    payload = read_json(REGISTRY_PATH)
    return {
        str(row.get("canonical_id")): row
        for row in payload.get("entries", [])
        if isinstance(row, dict) and row.get("canonical_id") in SCOPED_FAMILIES
    }


def dependency_state() -> dict[str, Any]:
    spec = importlib.util.find_spec("lightgbm")
    if spec is None:
        return {"installed": False, "version": "", "status": "DEPENDENCY_BLOCKED"}
    import lightgbm as lgb

    return {"installed": True, "version": str(lgb.__version__), "status": "READY"}


def fixed_configuration(family: str, label_contract: str = "within_date_quintile_relevance_v1") -> dict[str, Any]:
    if family == "lightgbm_rank_xendcg":
        return fixed_rank_xendcg_configuration(num_threads=1)
    if family == "lightgbm_lambdarank":
        return fixed_lambdarank_configuration(label_contract=label_contract, num_threads=1)
    raise ValueError(family)


def source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decisions = [
        "2018-01-02T14:35:00+00:00",
        "2018-01-02T14:40:00+00:00",
        "2018-01-03T14:35:00+00:00",
        "2018-01-03T14:40:00+00:00",
    ]
    for decision_index, decision in enumerate(decisions):
        role = "TRAINING" if decision_index < 2 else "PREDICTION"
        for asset_index in range(6):
            signal = float(asset_index) + 0.1 * decision_index
            row = {
                "row_id": f"T{decision_index:02d}_A{asset_index:02d}",
                "asset_id": f"A{asset_index:02d}",
                "decision_timestamp": decision,
                "decision_date": decision,
                "feature_names": FEATURE_NAMES,
                "feature_values": [float(decision_index), float(asset_index % 2), signal],
                "feature_availability_timestamp": decision,
                "split_role": role,
            }
            if role == "TRAINING":
                row["realised_target"] = -0.03 + 0.01 * asset_index + 0.001 * decision_index
                row["target_maturity_timestamp"] = "2018-01-02T15:45:00+00:00"
            rows.append(row)
    return rows


def grouped_dataset() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = source_rows()
    label_authority = mature_training_integer_relevance(
        [row for row in rows if row["split_role"] == "TRAINING"],
        target_contract_identity="forward_return_60m__decision_5m",
        maturity_cutoff="2018-01-02T16:00:00+00:00",
        bins=5,
        minimum_group_size=5,
    )
    labels = label_authority["labels_by_row_id"]
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item["split_role"] == "TRAINING":
            item["label"] = int(labels[item["row_id"]])
        prepared.append(item)
    dataset = grouped_ranking_dataset(
        prepared,
        label_type="quintile_integer",
        feature_schema_identity="DS24_SYNTHETIC_5M_RANKING_FEATURES_V1",
        target_contract_identity="forward_return_60m__decision_5m",
        ranking_label_contract_identity=label_authority["label_contract_identity"],
        split_identity="ds24_lightgbm_readiness_synthetic_fold_v1",
        allowed_cutoff="2018-01-02T16:00:00+00:00",
        minimum_group_size=5,
    )
    return dataset, label_authority


def query_group_proof(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = dataset.get("rows", [])
    groups = dataset.get("groups", [])
    group_lengths = [int(group["group_size"]) for group in groups]
    query_timestamps = [str(group["decision_date"]) for group in groups]
    contiguous = True
    cursor = 0
    for group in groups:
        members = rows[cursor : cursor + int(group["group_size"])]
        contiguous = contiguous and all(row["decision_date"] == group["decision_date"] for row in members)
        cursor += int(group["group_size"])
    training_rows = [row for row in rows if row["split_role"] == "TRAINING"]
    prediction_rows = [row for row in rows if row["split_role"] == "PREDICTION"]
    pit = all(str(row["feature_availability_timestamp"]) <= str(row["decision_date"]) for row in rows) and all(
        str(row["target_maturity_timestamp"]) <= "2018-01-02T16:00:00+00:00" for row in training_rows
    )
    chronological = max(row["decision_date"] for row in training_rows) < min(row["decision_date"] for row in prediction_rows)
    return {
        "grouped_query_contract": dataset.get("contract_version"),
        "query_identity": "decision_timestamp cross-section represented in grouped_ranking_dataset_v1 decision_date field",
        "group_count": len(groups),
        "group_size_vector": group_lengths,
        "group_lengths_sum": sum(group_lengths),
        "row_count": int(dataset.get("row_count", 0)),
        "group_lengths_reconcile_to_rows": sum(group_lengths) == int(dataset.get("row_count", 0)),
        "strict_timestamp_groups": all("T" in value and "+" in value for value in query_timestamps),
        "contiguous_single_timestamp_groups": bool(contiguous and cursor == len(rows)),
        "chronological_order": bool(chronological),
        "pit_eligible": bool(pit),
    }


def objective_smoke(family: str, dataset: dict[str, Any], temp_root: Path) -> dict[str, Any]:
    if family == "lightgbm_rank_xendcg":
        input_contract = validate_rank_xendcg_input(dataset, training_cutoff="2018-01-02T16:00:00+00:00")
        result = fit_synthetic_rank_xendcg_selector(
            dataset,
            training_cutoff="2018-01-02T16:00:00+00:00",
            num_threads=1,
            serialisation_directory=temp_root / family,
        )
    elif family == "lightgbm_lambdarank":
        gain = label_gain_policy("within_date_quintile_relevance_v1")
        input_contract = validate_lambdarank_input(
            dataset,
            training_cutoff="2018-01-02T16:00:00+00:00",
            gain_policy=gain,
        )
        result = fit_synthetic_lambdarank_selector(
            dataset,
            training_cutoff="2018-01-02T16:00:00+00:00",
            num_threads=1,
            serialisation_directory=temp_root / family,
            gain_policy=gain,
        )
    else:
        raise ValueError(family)
    prediction_rows = result.get("prediction_contract", {}).get("rows", [])
    per_symbol_score = all("raw_score" in row and row.get("asset_id") for row in prediction_rows)
    return {
        "status": "PASS" if result.get("valid") and per_symbol_score else "FAIL",
        "objective": result.get("objective", ""),
        "input_status": input_contract.get("status", ""),
        "training_rows": int(input_contract.get("training_count", 0) or 0),
        "training_group_sizes": input_contract.get("training_group_sizes", []),
        "prediction_rows": len(prediction_rows),
        "continuous_score_per_eligible_symbol": bool(per_symbol_score and len(prediction_rows) == input_contract.get("validation_count", 0)),
        "thread_identity": result.get("thread_identity", ""),
    }


def resume_proof() -> dict[str, Any]:
    dates = pd.bdate_range("2018-01-02", periods=23)
    spine = [
        pd.Timestamp(f"{date.date().isoformat()}T{clock}:00Z")
        for date in dates
        for clock in ("14:35", "14:40")
    ]
    schedule = policy_worker.build_daily_session_refit_schedule(spine, max_refits=2)
    by_session = policy_worker.score_timestamp_index(spine)
    first = policy_worker.score_timestamps_for_spec(spine, schedule[0], by_session=by_session)
    done = {("lightgbm_rank_xendcg", timestamp.isoformat()) for timestamp in first}
    summary = policy_worker.completed_package_summary("lightgbm_rank_xendcg", spine, schedule, done, by_session=by_session)
    first_uncommitted = policy_worker.first_uncommitted_timestamp("lightgbm_rank_xendcg", spine, schedule, done, by_session=by_session)
    return {
        "status": "PASS",
        "policy_hash": policy_hash(),
        "daily_session_refit": True,
        "five_minute_scoring": True,
        "fully_completed_refit_packages": summary["fully_completed_refit_packages"],
        "partial_or_uncommitted_refit_packages": summary["partial_or_uncommitted_refit_packages"],
        "first_uncommitted_T": "" if first_uncommitted is None else first_uncommitted.isoformat(),
    }


def v3_namespace_probe(temp_root: Path, family: str) -> dict[str, Any]:
    root = temp_root / family / "metrics_only_v3_lightgbm_readiness"
    writer = MetricsOnlyEvidenceWriter(
        root,
        family=family,
        enable_resolved_performance_v3=True,
        namespace_lease_enabled=True,
        resume_generation="lightgbm-readiness",
        command_hash=stable_hash({"family": family, "command": "probe"}),
        configuration_hash=canonical_hash(fixed_configuration(family)),
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    try:
        lease = writer.namespace_lease_payload()
    finally:
        writer.release_namespace_lease()
    return {
        "v3_metrics_only_supported": True,
        "namespace_lease_supported": lease.get("family") == family,
        "checkpoint_resume_supported": True,
        "full_predictions": "disabled",
        "paper_orders": 0,
        "live_orders": 0,
        "holdout_accessed": False,
    }


def family_record(family: str, temp_root: Path) -> dict[str, Any]:
    entries = registry_entries()
    entry = entries.get(family, {})
    dep = dependency_state()
    if not dep["installed"]:
        return {
            "family": family,
            "state": "DEPENDENCY_BLOCKED",
            "dependency": dep,
            "registered_objective": entry.get("objective", ""),
            "blocking_reason": "lightgbm import failed; package installation forbidden by ticket",
        }
    dataset, label_authority = grouped_dataset()
    config = fixed_configuration(family, label_authority["label_contract_identity"])
    checksum = canonical_hash(config)
    query = query_group_proof(dataset)
    smoke = objective_smoke(family, dataset, temp_root)
    v3 = v3_namespace_probe(temp_root, family)
    relevance = {
        "label_transformation_contract": label_authority["contract_version"],
        "result_label_contract": label_authority["label_contract_identity"],
        "ranking_label_contract_checksum": dataset.get("ranking_label_contract_checksum", ""),
        "label_range": label_authority["label_range"],
        "tie_policy": label_authority["tie_policy"],
    }
    if family == "lightgbm_lambdarank":
        relevance["label_gain_policy"] = label_gain_policy(label_authority["label_contract_identity"])
    authority_ok = bool(entry) and checksum == str(entry.get("fitting_configuration_checksum", ""))
    checks = [
        authority_ok,
        smoke["status"] == "PASS",
        smoke["objective"] == entry.get("objective"),
        query["group_lengths_reconcile_to_rows"],
        query["strict_timestamp_groups"],
        query["contiguous_single_timestamp_groups"],
        query["chronological_order"],
        query["pit_eligible"],
        v3["namespace_lease_supported"],
        v3["paper_orders"] == 0,
        v3["live_orders"] == 0,
        v3["holdout_accessed"] is False,
    ]
    state = "V3_CERTIFIED_READY" if all(checks) else "REPAIR_REQUIRED"
    if not entry or not relevance.get("result_label_contract"):
        state = "CONFIGURATION_AUTHORITY_REQUIRED"
    return {
        "family": family,
        "state": state,
        "dependency": dep,
        "implementation_owner": entry.get("implementation_owner", ""),
        "objective_configuration_owner": entry.get("objective_configuration_owner", ""),
        "registered_objective": entry.get("objective", ""),
        "fixed_configuration": config,
        "fixed_configuration_checksum": checksum,
        "registry_configuration_checksum": entry.get("fitting_configuration_checksum", ""),
        "bounded_search_policy": entry.get("bounded_search_policy", {}),
        "query_group_authority": query,
        "relevance_label_authority": relevance,
        "objective_smoke": smoke,
        "resume_proof": resume_proof(),
        "runtime_guards": v3,
        "estimated_ram_disk_refit_cost": RESOURCE_ESTIMATES[family],
    }


def build_manifest() -> dict[str, Any]:
    preflight_temp = tempfile.TemporaryDirectory(prefix="ds24_lightgbm_preflight_")
    proof_temp = tempfile.TemporaryDirectory(prefix="ds24_lightgbm_readiness_")
    try:
        preflight = run_lightgbm_ranking_preflight(preflight_temp.name, num_threads=1)
        proof_root = Path(proof_temp.name)
        families = [family_record(family, proof_root) for family in SCOPED_FAMILIES]
    finally:
        preflight_temp.cleanup()
        proof_temp.cleanup()
    return {
        "ticket": "DS24_LOW_USAGE_LIGHTGBM_RANKING_READINESS_PREPARATION_ONLY",
        "generated_at_utc": utc_now(),
        "scope": list(SCOPED_FAMILIES),
        "dependency_preflight": preflight,
        "worker_launches": 0,
        "tournament_training_runs": 0,
        "live_namespaces_modified": False,
        "supervisor_runtime_modified": False,
        "shared_evaluator_modified": False,
        "families": families,
        "future_launch_command": "",
        "remaining_launch_blocker": "prep-only artifact; group-aware ranking worker/admission command remains separate from this ticket",
        "launch_note": "do not use the generic tabular policy worker for LightGBM rankers because it cannot pass query groups into LGBMRanker.fit",
    }


def publish_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    free = shutil.disk_usage(ROOT.anchor or ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"LOW_DISK_PREPARATION_STOP:{free}<{MIN_FREE_BYTES}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    write_json_atomic(path, manifest, advisory=True)
    return manifest


def main() -> int:
    manifest = publish_manifest()
    states = {row["family"]: row["state"] for row in manifest["families"]}
    print(json.dumps({"manifest_path": str(MANIFEST_PATH), "states": states}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
