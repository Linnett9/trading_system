from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


STAGE = supervisor.STAGE
MODEL_ARTIFACT_ROOT = STAGE / "model_artifacts"
OOF_ROOT = STAGE / "oof_predictions"
R47_PRE_IMPORT_SNAPSHOT_PATH = STAGE / "R47_cross_host_pre_import_snapshot.json"
R47_IMPORT_AUTHORITY_PATH = STAGE / "R47_cross_host_import_authority.json"
R47_IMPORT_LEDGER_PATH = STAGE / "R47_cross_host_import_ledger.jsonl"
R47_FAMILY_IMPORT_STATUS_PATH = STAGE / "R47_family_import_status.json"
R47_OWNERSHIP_STATE_PATH = STAGE / "R47_cross_host_ownership_state.json"
R47_DELL_EFFECTIVE_READY_QUEUE_PATH = STAGE / "R47_dell_effective_ready_queue.json"
R47_MAC_TRANSFER_CONTRACT_PATH = STAGE / "R47_mac_future_transfer_contract.json"
R47_CLASSIFICATION = "DS24_R47_CROSS_HOST_IMPORT_AUTHORITY_ACTIVE"
R47A_CLASSIFICATION = "DS24_R47A_XENDCG_CROSS_HOST_IMPORT_COMPLETE"
MAC_AUX_QUEUE_ID = "DS24_MAC_AUX_NINE_FAMILY_R1"
MAC_AUX_ROOT = ROOT / "mac_aux_runs" / f"queue={MAC_AUX_QUEUE_ID}"
XENDCG_MAC_AUX_FAMILY_ROOT = MAC_AUX_ROOT / "family=lightgbm_rank_xendcg"
LAMBDARANK_MAC_AUX_FAMILY_ROOT = MAC_AUX_ROOT / "family=lightgbm_lambdarank"
PRODUCER_AUTHORITY_ROOT = Path(r"C:\Users\Brandon\Desktop\ds24-prospective-paper\authority\xendcg\ds24_xendcg_producer_authority_r1")
R47A_XENDCG_SOURCE_DISCOVERY_PATH = STAGE / "R47A_xendcg_source_discovery.json"
R47A_XENDCG_ARTIFACT_VALIDATION_PATH = STAGE / "R47A_xendcg_artifact_validation.json"
R47A_XENDCG_IMPORT_AUTHORITY_PATH = STAGE / "R47A_xendcg_import_authority.json"
R47A_XENDCG_IMPORT_RESULT_PATH = STAGE / "R47A_xendcg_import_result.json"
R47A_OWNERSHIP_STATE_PATH = STAGE / "R47A_cross_host_ownership_state.json"
R47A_DELL_EFFECTIVE_READY_QUEUE_PATH = STAGE / "R47A_dell_effective_ready_queue.json"
IMPORTABLE_FAMILIES = ("lightgbm_rank_xendcg", "lightgbm_lambdarank", "DLinear")
XENDCG_EXPECTED_MODEL_ARTIFACTS = 2262
XENDCG_EXPECTED_FIRST_ORDINAL = 2
XENDCG_EXPECTED_LAST_ORDINAL = 2263
XENDCG_FIXTURE_REFITS = ("000002", "001132", "002263")
XENDCG_PRODUCER_SHA = "5c5e8ff1486d870f8c2d3b1ceb2c406f5544e38a34e5a768a816abc06076b17d"
XENDCG_MAC_HEAD = "b2ffdf0966c0424c82632ea21f83cefb29905cc1"
XENDCG_FEATURE_ORDER_SHA = "22db0fcfe1219a30e0f6926dd8f0af11b4cd8ead2a6462a0df278549fadeb5a0"
XENDCG_MODEL_CONFIG_HASH = "1ca718aebd461365b59c1503acdff5c8f0899aabfd699fe59995bdceccd435d3"
XENDCG_TARGET_CONTRACT_HASH = "295656c825cb70986359c2b6457c44224ad45ede3aff97bdb97dbb85e02a2fda"
XENDCG_SOURCE_RUN_ID = "MAC_LIGHTGBM_RANK_XENDCG_COMPLETE_TRANSFERRED_AUTHORITY"
MAC_AUX_SOURCE_ROOT_TYPE = "MAC_AUX_FAMILY_ROOT"
IMPORT_REQUIRED_FIELDS = (
    "family",
    "mac_run_id",
    "mac_producing_host",
    "producer_source_hash",
    "model_config_authority_hash",
    "artifact_manifest",
    "artifact_hashes",
    "metrics_manifest",
    "metrics_hashes",
    "score_oof_population",
    "date_range",
    "target",
    "feature_count",
    "feature_hash",
    "refit_count",
    "terminal_status",
    "holdout_accessed",
    "paper_orders",
    "live_orders",
)
KNOWN_XENDCG_METRICS = {
    "mean_rank_ic": 0.250646544554,
    "mean_spearman_rank_ic": 0.250646544554,
    "pearson_ic": 0.247020330954,
    "ndcg_at_20": 0.228239431375,
    "directional_accuracy": 0.580621053511,
    "mean_hit_rate": 0.703762726870,
    "mean_daily_net_return": 0.007043190071,
    "daily_sharpe": 13.233750175466,
    "annualised_arithmetic_return": 1.774883897978,
    "annual_volatility": 0.134117984278,
    "max_drawdown": -0.029690054964,
    "caution": "Extraordinary Mac-origin result; import is provenance acceptance, not independent prospective confirmation.",
}


@dataclass(frozen=True)
class ImportVerification:
    family: str
    classification: str
    source_run_id: str
    source_host: str
    artifact_root: str
    model_artifact_count: int
    apple_double_sidecar_count: int
    missing_requirements: tuple[str, ...]
    artifact_hash: str
    original_metrics: Mapping[str, Any]
    tournament_comparable_metrics_status: str
    previous_ownership: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "classification": self.classification,
            "source_run_id": self.source_run_id,
            "source_host": self.source_host,
            "artifact_root": self.artifact_root,
            "model_artifact_count": self.model_artifact_count,
            "apple_double_sidecar_count": self.apple_double_sidecar_count,
            "missing_requirements": list(self.missing_requirements),
            "artifact_hash": self.artifact_hash,
            "original_metrics": dict(self.original_metrics),
            "tournament_comparable_metrics_status": self.tournament_comparable_metrics_status,
            "previous_ownership": dict(self.previous_ownership),
        }


def read_json(path: Path) -> dict[str, Any]:
    return supervisor.read_json(path)


def write_json(path: Path, payload: Any) -> None:
    supervisor.write_json(path, payload)


def utc_now() -> str:
    return supervisor.utc_now()


def display_path(path: Path) -> str:
    return supervisor.display_path(path)


def state_hash(payload: Any) -> str:
    return supervisor.state_hash(payload)


def file_hash(path: Path) -> str:
    return supervisor.file_hash(path)


def family_tokens(family: str) -> tuple[str, ...]:
    if family == "DLinear":
        return ("DLinear", "dlinear")
    return (family,)


def mac_aux_family_root(family: str) -> Path:
    if family == "lightgbm_rank_xendcg":
        return XENDCG_MAC_AUX_FAMILY_ROOT
    if family == "lightgbm_lambdarank":
        return LAMBDARANK_MAC_AUX_FAMILY_ROOT
    return MAC_AUX_ROOT / "family=DLinear"


def artifact_root_for_family(family: str) -> Path:
    if family in {"lightgbm_rank_xendcg", "lightgbm_lambdarank"}:
        return mac_aux_family_root(family) / "model_artifacts"
    return MODEL_ARTIFACT_ROOT


def family_model_artifacts(family: str, artifact_root: Path | None = None) -> list[Path]:
    root = artifact_root or MODEL_ARTIFACT_ROOT
    if not root.exists():
        return []
    matches: list[Path] = []
    for token in family_tokens(family):
        matches.extend(path for path in root.glob(f"*{token}*.pkl") if path.is_file() and not path.name.startswith("._"))
    return sorted(set(matches))


def apple_double_sidecars(family: str, artifact_root: Path | None = None) -> list[Path]:
    root = artifact_root or MODEL_ARTIFACT_ROOT
    if not root.exists():
        return []
    matches: list[Path] = []
    for token in family_tokens(family):
        matches.extend(path for path in root.glob(f"._*{token}*.pkl") if path.is_file())
    return sorted(set(matches))


def manifest_hash(paths: Iterable[Path]) -> str:
    rows = [
        {
            "path": display_path(path),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    return state_hash(rows) if rows else ""


def csv_family_rows(path: Path, family: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("family") == family]


def ownership_by_family(path: Path = supervisor.R44_CROSS_HOST_OWNERSHIP_PATH) -> dict[str, dict[str, Any]]:
    return supervisor.validate_cross_host_ownership_authority(path).get("by_family", {})


def family_status_file(family: str) -> Path:
    if family == "lightgbm_rank_xendcg":
        return STAGE / "54_rankxendcg_status.json"
    if family == "lightgbm_lambdarank":
        return STAGE / "55_lambdarank_status.json"
    return STAGE / "46_dlinear_status.json"


def source_run_id_for(family: str, ownership: Mapping[str, Any]) -> str:
    if family == "lightgbm_rank_xendcg":
        return XENDCG_SOURCE_RUN_ID
    if family == "lightgbm_lambdarank":
        return "MAC_LIGHTGBM_LAMBDARANK_PENDING_COMPLETION"
    return "MAC_DLINEAR_RESERVED_NEXT"


def expected_model_count(family: str) -> int:
    return XENDCG_EXPECTED_MODEL_ARTIFACTS if family == "lightgbm_rank_xendcg" else 0


def build_import_contract(family: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    root = artifact_root or artifact_root_for_family(family)
    ownership = ownership_by_family().get(family, {})
    artifacts = family_model_artifacts(family, root)
    sidecars = apple_double_sidecars(family, root)
    inventory_rows = csv_family_rows(STAGE / "20_model_artifact_inventory.csv", family) + csv_family_rows(STAGE / "20_model_artifact_inventory_r1_live.csv", family)
    status = read_json(family_status_file(family))
    artifact_hash = manifest_hash(artifacts)
    if family == "lightgbm_rank_xendcg" and root == XENDCG_MAC_AUX_FAMILY_ROOT / "model_artifacts":
        return build_xendcg_mac_aux_import_contract()
    if family == "lightgbm_rank_xendcg":
        terminal_status = "MAC_COMPLETE" if ownership.get("owner_state") == "MAC_COMPLETE" else ""
        feature_count: int | str = 101
        target = "forward_return_60m__decision_5m"
        refit_count: int | str = 3
        metrics = KNOWN_XENDCG_METRICS
        comparable = "NOT_DERIVABLE_FROM_RETAINED_DELL_ARTIFACTS" if len(artifacts) != XENDCG_EXPECTED_MODEL_ARTIFACTS else "DERIVABLE_FROM_RETAINED_OOF_TOPN_PENDING_COMPUTATION"
    elif family == "lightgbm_lambdarank":
        terminal_status = "AWAITING_MAC_COMPLETION"
        feature_count = ""
        target = "forward_return_60m__decision_5m"
        refit_count = ""
        metrics = {}
        comparable = "AWAITING_MAC_COMPLETION"
    else:
        terminal_status = str(ownership.get("owner_state") or "MAC_RESERVED_NEXT")
        feature_count = ""
        target = "forward_return_60m__decision_5m"
        refit_count = ""
        metrics = {}
        comparable = "MAC_RESERVED_NEXT_NO_IMPORT_ARTIFACTS"
    return {
        "family": family,
        "mac_run_id": source_run_id_for(family, ownership),
        "mac_producing_host": "MAC",
        "producer_source_hash": status.get("producer_source_hash", ""),
        "model_config_authority_hash": status.get("model_config_authority_hash", ""),
        "artifact_manifest": {
            "root": display_path(root),
            "model_artifact_count": len(artifacts),
            "expected_model_artifact_count": expected_model_count(family),
            "inventory_rows": len(inventory_rows),
            "apple_double_sidecar_count": len(sidecars),
        },
        "artifact_hashes": {"manifest_sha256": artifact_hash},
        "metrics_manifest": status.get("metrics_manifest", {}),
        "metrics_hashes": status.get("metrics_hashes", {}),
        "score_oof_population": status.get("score_oof_population", {}),
        "date_range": status.get("date_range", {}),
        "target": target,
        "feature_count": feature_count,
        "feature_hash": status.get("feature_hash", ""),
        "refit_count": refit_count,
        "terminal_status": terminal_status,
        "holdout_accessed": bool(status.get("holdout_accessed", False)),
        "paper_orders": int(status.get("paper_orders", 0) or 0),
        "live_orders": int(status.get("live_orders", 0) or 0),
        "original_metrics": metrics,
        "tournament_comparable_metrics_status": comparable,
        "previous_ownership": ownership,
    }


def missing_contract_fields(contract: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in IMPORT_REQUIRED_FIELDS:
        value = contract.get(field)
        if value in ("", None, {}, []):
            missing.append(field)
    if contract.get("holdout_accessed"):
        missing.append("holdout_must_be_false")
    if int(contract.get("paper_orders", 0) or 0) or int(contract.get("live_orders", 0) or 0):
        missing.append("paper_live_orders_must_be_zero")
    return missing


def verify_family_import(family: str, *, artifact_root: Path | None = None) -> ImportVerification:
    contract = build_import_contract(family, artifact_root=artifact_root)
    artifacts = contract["artifact_manifest"]
    missing = missing_contract_fields(contract)
    model_artifact_count = int(artifacts.get("model_artifact_count") or artifacts.get("genuine_pkl_count") or 0)
    if family == "lightgbm_rank_xendcg" and model_artifact_count != XENDCG_EXPECTED_MODEL_ARTIFACTS:
        missing.append("expected_2262_non_appledouble_model_artifacts")
    if family == "lightgbm_lambdarank" and contract["previous_ownership"].get("owner_state") == "MAC_RUNNING":
        missing.append("mac_completion_artifacts_not_present")
    if family == "DLinear" and contract["previous_ownership"].get("owner_state") == "MAC_RESERVED_NEXT":
        missing.append("mac_dlinear_not_started_or_completed")
    classification = "COMPLETE_IMPORTED" if not missing else "IMPORT_BLOCKED_MISSING_AUTHORITY_OR_ARTIFACTS"
    return ImportVerification(
        family=family,
        classification=classification,
        source_run_id=str(contract["mac_run_id"]),
        source_host=str(contract["mac_producing_host"]),
        artifact_root=str(artifacts.get("root") or artifacts.get("model_root", "")),
        model_artifact_count=model_artifact_count,
        apple_double_sidecar_count=int(contract["artifact_manifest"]["apple_double_sidecar_count"]),
        missing_requirements=tuple(dict.fromkeys(missing)),
        artifact_hash=str(contract["artifact_hashes"]["manifest_sha256"]),
        original_metrics=contract.get("original_metrics", {}),
        tournament_comparable_metrics_status=str(contract["tournament_comparable_metrics_status"]),
        previous_ownership=contract.get("previous_ownership", {}),
    )


def ledger_events(path: Path | None = None) -> list[dict[str, Any]]:
    ledger_path = path or R47_IMPORT_LEDGER_PATH
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def duplicate_guard(verification: ImportVerification, path: Path | None = None) -> str:
    if verification.classification != "COMPLETE_IMPORTED":
        return "NOT_APPLICABLE_INCOMPLETE_IMPORT"
    for event in ledger_events(path or R47_IMPORT_LEDGER_PATH):
        if event.get("family") != verification.family or event.get("source_run_id") != verification.source_run_id:
            continue
        if event.get("validation_classification") != "COMPLETE_IMPORTED":
            continue
        if event.get("artifact_hash") == verification.artifact_hash:
            return "ALREADY_IMPORTED_IDENTICAL"
        return "CROSS_HOST_IMPORT_HASH_CONFLICT"
    return "NEW_IMPORT"


def append_ledger_event(event: Mapping[str, Any], path: Path | None = None) -> None:
    ledger_path = path or R47_IMPORT_LEDGER_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, default=str, separators=(",", ":")) + "\n")


def model_refit_ordinal(path: Path) -> int | None:
    match = re.search(r"refit=(\d{6})\.pkl$", path.name)
    return int(match.group(1)) if match else None


def xendcg_model_inventory(model_root: Path | None = None) -> dict[str, Any]:
    model_root = model_root or XENDCG_MAC_AUX_FAMILY_ROOT / "model_artifacts"
    genuine = family_model_artifacts("lightgbm_rank_xendcg", model_root)
    sidecars = apple_double_sidecars("lightgbm_rank_xendcg", model_root)
    ordinals = [ordinal for path in genuine if (ordinal := model_refit_ordinal(path)) is not None]
    ordinal_counts = {ordinal: ordinals.count(ordinal) for ordinal in sorted(set(ordinals))}
    expected = set(range(XENDCG_EXPECTED_FIRST_ORDINAL, XENDCG_EXPECTED_LAST_ORDINAL + 1))
    observed = set(ordinals)
    missing = sorted(expected - observed)
    duplicates = sorted(ordinal for ordinal, count in ordinal_counts.items() if count > 1)
    return {
        "root": display_path(model_root),
        "model_root": display_path(model_root),
        "model_artifact_count": len(genuine),
        "genuine_pkl_count": len(genuine),
        "apple_double_sidecar_count": len(sidecars),
        "missing_ordinals": [f"{ordinal:06d}" for ordinal in missing],
        "missing_ordinal_count": len(missing),
        "duplicate_ordinals": [f"{ordinal:06d}" for ordinal in duplicates],
        "duplicate_ordinal_count": len(duplicates),
        "first_ordinal": f"{min(ordinals):06d}" if ordinals else "",
        "last_ordinal": f"{max(ordinals):06d}" if ordinals else "",
        "expected_first_ordinal": f"{XENDCG_EXPECTED_FIRST_ORDINAL:06d}",
        "expected_last_ordinal": f"{XENDCG_EXPECTED_LAST_ORDINAL:06d}",
        "expected_genuine_pkl_count": XENDCG_EXPECTED_MODEL_ARTIFACTS,
        "unexpected_filenames": [path.name for path in genuine if model_refit_ordinal(path) is None],
    }


def load_producer_authority() -> dict[str, Any]:
    source_manifest = read_json(PRODUCER_AUTHORITY_ROOT / "source_manifest.json")
    parity = read_json(PRODUCER_AUTHORITY_ROOT / "three_refit_parity.json")
    primary_source = next(
        row for row in source_manifest.get("source_files", []) if row.get("relative_path") == "core/research/ml/ds24/mac_aux_queue_r44f2.py"
    )
    fixture_hashes = {
        str(row["refit"]).zfill(6): {
            "model_sha256": row["model_sha256"],
            "oof_sha256": row["oof_sha256"],
            "model_size_bytes": row["model_size_bytes"],
            "oof_size_bytes": row["oof_size_bytes"],
        }
        for row in source_manifest.get("xendcg_artifact_hashes", [])
    }
    return {
        "authority_root": display_path(PRODUCER_AUTHORITY_ROOT),
        "source_manifest_sha256": file_hash(PRODUCER_AUTHORITY_ROOT / "source_manifest.json"),
        "three_refit_parity_sha256": file_hash(PRODUCER_AUTHORITY_ROOT / "three_refit_parity.json"),
        "producer_sha": primary_source["sha256"],
        "mac_head": source_manifest.get("mac_head", ""),
        "producer_source_classification": source_manifest.get("producer_source_classification", ""),
        "predictor_count": source_manifest.get("feature_authority", {}).get("predictor_count"),
        "feature_order_sha256": source_manifest.get("feature_authority", {}).get("feature_order_sha256", ""),
        "estimator_params": source_manifest.get("estimator_configuration_source", {}).get("params", {}),
        "runtime_arguments": source_manifest.get("runtime_arguments", {}),
        "parity_classification": parity.get("classification", ""),
        "fixture_hashes": fixture_hashes,
        "parity_results": parity.get("results", []),
    }


def validate_producer_authority(authority: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if authority.get("producer_sha") != XENDCG_PRODUCER_SHA:
        missing.append("producer_sha_mismatch")
    if authority.get("mac_head") != XENDCG_MAC_HEAD:
        missing.append("mac_head_mismatch")
    if authority.get("producer_source_classification") != "MAC_WORKTREE_UNCOMMITTED_SOURCE_OVER_ADVERTISED_HEAD":
        missing.append("producer_source_classification_mismatch")
    if authority.get("predictor_count") != 101:
        missing.append("predictor_count_mismatch")
    if authority.get("feature_order_sha256") != XENDCG_FEATURE_ORDER_SHA:
        missing.append("feature_order_sha_mismatch")
    params = authority.get("estimator_params", {})
    required_params = {
        "objective": "rank_xendcg",
        "n_estimators": 25,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "random_state": 1729,
        "n_jobs": 4,
        "num_threads": 4,
    }
    for key, expected in required_params.items():
        if params.get(key) != expected:
            missing.append(f"estimator_{key}_mismatch")
    runtime = authority.get("runtime_arguments", {})
    if runtime.get("lookback_sessions") != 20:
        missing.append("lookback_sessions_mismatch")
    if runtime.get("max_training_rows") != 24000:
        missing.append("max_training_rows_mismatch")
    if authority.get("parity_classification") != "XENDCG_PRODUCTION_PARITY_PASS":
        missing.append("parity_classification_mismatch")
    return missing


def validate_parity_results(authority: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    by_refit = {f"{int(row['refit_ordinal']):06d}": row for row in authority.get("parity_results", [])}
    for refit in XENDCG_FIXTURE_REFITS:
        row = by_refit.get(refit)
        if not row:
            missing.append(f"parity_fixture_{refit}_missing")
            continue
        if row.get("fresh_vs_saved_model_max_abs_diff") != 0.0:
            missing.append(f"parity_fixture_{refit}_fresh_saved_diff_nonzero")
        if float(row.get("fresh_vs_oof_max_abs_diff", 1.0)) > 5e-8:
            missing.append(f"parity_fixture_{refit}_fresh_oof_diff_exceeds_tolerance")
        if float(row.get("saved_model_vs_oof_max_abs_diff", 1.0)) > 5e-8:
            missing.append(f"parity_fixture_{refit}_saved_oof_diff_exceeds_tolerance")
        if not row.get("fresh_top20_membership_match"):
            missing.append(f"parity_fixture_{refit}_top20_membership_mismatch")
        if not row.get("fresh_top20_order_match"):
            missing.append(f"parity_fixture_{refit}_top20_order_mismatch")
        if not row.get("saved_model_top20_membership_match"):
            missing.append(f"parity_fixture_{refit}_saved_top20_membership_mismatch")
        if not row.get("saved_model_top20_order_match"):
            missing.append(f"parity_fixture_{refit}_saved_top20_order_mismatch")
    return missing


def xendcg_fixture_hash_validation(authority: Mapping[str, Any]) -> dict[str, Any]:
    fixtures: dict[str, Any] = {}
    missing: list[str] = []
    for refit, expected in authority.get("fixture_hashes", {}).items():
        model = XENDCG_MAC_AUX_FAMILY_ROOT / "model_artifacts" / f"lightgbm_rank_xendcg_refit={refit}.pkl"
        manifest = read_json(XENDCG_MAC_AUX_FAMILY_ROOT / "ensemble_oof_scores_manifest_v2.json")
        oof_row = next((row for row in manifest.get("files", []) if int(row.get("refit_ordinal")) == int(refit)), {})
        oof = XENDCG_MAC_AUX_FAMILY_ROOT / str(oof_row.get("relative_path", ""))
        actual_model_hash = file_hash(model) if model.exists() else ""
        actual_oof_hash = file_hash(oof) if oof.exists() else ""
        model_match = actual_model_hash == expected["model_sha256"]
        oof_match = actual_oof_hash == expected["oof_sha256"]
        if not model_match:
            missing.append(f"fixture_{refit}_model_hash_mismatch")
        if not oof_match:
            missing.append(f"fixture_{refit}_oof_hash_mismatch")
        fixtures[refit] = {
            "model_path": display_path(model),
            "expected_model_sha256": expected["model_sha256"],
            "actual_model_sha256": actual_model_hash,
            "model_hash_match": model_match,
            "oof_path": display_path(oof),
            "expected_oof_sha256": expected["oof_sha256"],
            "actual_oof_sha256": actual_oof_hash,
            "oof_hash_match": oof_match,
        }
    return {"fixtures": fixtures, "missing_requirements": missing}


def stable_xendcg_artifact_hash(
    inventory: Mapping[str, Any],
    authority: Mapping[str, Any],
    fixture_validation: Mapping[str, Any],
    oof_manifest_sha256: str,
    metrics_sha256: str,
) -> str:
    return state_hash(
        {
            "source_root_type": MAC_AUX_SOURCE_ROOT_TYPE,
            "source_root": display_path(XENDCG_MAC_AUX_FAMILY_ROOT),
            "model_inventory": inventory,
            "producer_sha": authority.get("producer_sha", ""),
            "mac_head": authority.get("mac_head", ""),
            "feature_order_sha256": authority.get("feature_order_sha256", ""),
            "source_manifest_sha256": authority.get("source_manifest_sha256", ""),
            "three_refit_parity_sha256": authority.get("three_refit_parity_sha256", ""),
            "fixture_hash_validation": fixture_validation,
            "oof_manifest_sha256": oof_manifest_sha256,
            "metrics_sha256": metrics_sha256,
        }
    )


def xendcg_source_discovery() -> dict[str, Any]:
    root = XENDCG_MAC_AUX_FAMILY_ROOT
    summary = read_json(root / "family_execution_summary.json") if (root / "family_execution_summary.json").exists() else {}
    checkpoint = read_json(root / "checkpoints" / "latest.json") if (root / "checkpoints" / "latest.json").exists() else {}
    oof_manifest = read_json(root / "ensemble_oof_scores_manifest_v2.json") if (root / "ensemble_oof_scores_manifest_v2.json").exists() else {}
    metrics = read_json(root / "metrics_only_v3" / "resolved_performance_summary_v3.json") if (root / "metrics_only_v3" / "resolved_performance_summary_v3.json").exists() else {}
    files = oof_manifest.get("files", [])
    decision_dates = [str(row.get("decision_date")) for row in files if row.get("decision_date")]
    payload = {
        "ticket": "DS24_R47A_XENDCG_SOURCE_DISCOVERY",
        "generated_at_utc": utc_now(),
        "classification": "PASS" if root.exists() else "MISSING_SOURCE_ROOT",
        "source_root_type": MAC_AUX_SOURCE_ROOT_TYPE,
        "queue_id": MAC_AUX_QUEUE_ID,
        "family": "lightgbm_rank_xendcg",
        "source_root": display_path(root),
        "root_exists": root.exists(),
        "top_level_entries": sorted(path.name for path in root.iterdir()) if root.exists() else [],
        "terminal_family_status": summary.get("status", ""),
        "checkpoint_cursor": checkpoint.get("cursor", ""),
        "run_id": summary.get("run_id") or oof_manifest.get("run_id", ""),
        "refit_count": oof_manifest.get("distinct_decision_timestamps") or summary.get("metrics_rows"),
        "decision_date_range": {"start": min(decision_dates) if decision_dates else "", "end": max(decision_dates) if decision_dates else ""},
        "score_row_count": oof_manifest.get("row_count"),
        "retained_oof_row_count": oof_manifest.get("row_count"),
        "top_n_contract": "Top-20 retained membership/order authority from three_refit_parity.json",
        "target": metrics.get("target_contract", "forward_return_60m__decision_5m"),
        "feature_order_sha256": XENDCG_FEATURE_ORDER_SHA,
        "metrics_authority": display_path(root / "metrics_only_v3" / "resolved_performance_summary_v3.json"),
        "oof_manifest_sha256": file_hash(root / "ensemble_oof_scores_manifest_v2.json") if (root / "ensemble_oof_scores_manifest_v2.json").exists() else "",
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(R47A_XENDCG_SOURCE_DISCOVERY_PATH, payload)
    return payload


def xendcg_artifact_validation() -> dict[str, Any]:
    inventory = xendcg_model_inventory()
    authority = load_producer_authority()
    fixture_validation = xendcg_fixture_hash_validation(authority)
    oof_manifest_sha256 = file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "ensemble_oof_scores_manifest_v2.json")
    metrics_sha256 = file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "resolved_performance_summary_v3.json")
    artifact_hash = stable_xendcg_artifact_hash(
        inventory,
        authority,
        fixture_validation["fixtures"],
        oof_manifest_sha256,
        metrics_sha256,
    )
    missing = []
    if inventory["genuine_pkl_count"] != XENDCG_EXPECTED_MODEL_ARTIFACTS:
        missing.append("expected_2262_non_appledouble_model_artifacts")
    if inventory["missing_ordinal_count"]:
        missing.append("missing_model_ordinals")
    if inventory["duplicate_ordinal_count"]:
        missing.append("duplicate_model_ordinals")
    missing.extend(validate_producer_authority(authority))
    missing.extend(validate_parity_results(authority))
    missing.extend(fixture_validation["missing_requirements"])
    payload = {
        "ticket": "DS24_R47A_XENDCG_ARTIFACT_VALIDATION",
        "generated_at_utc": utc_now(),
        "classification": "PASS" if not missing else "FAIL_CLOSED",
        "source_root_type": MAC_AUX_SOURCE_ROOT_TYPE,
        "source_root": display_path(XENDCG_MAC_AUX_FAMILY_ROOT),
        "model_inventory": inventory,
        "producer_authority": {
            key: value
            for key, value in authority.items()
            if key not in {"parity_results"}
        },
        "fixture_hash_validation": fixture_validation["fixtures"],
        "parity_fixture_validation": {
            f"{int(row['refit_ordinal']):06d}": {
                "fresh_vs_saved_model_max_abs_diff": row.get("fresh_vs_saved_model_max_abs_diff"),
                "fresh_vs_oof_max_abs_diff": row.get("fresh_vs_oof_max_abs_diff"),
                "saved_model_vs_oof_max_abs_diff": row.get("saved_model_vs_oof_max_abs_diff"),
                "fresh_top20_membership_match": row.get("fresh_top20_membership_match"),
                "fresh_top20_order_match": row.get("fresh_top20_order_match"),
                "saved_model_top20_membership_match": row.get("saved_model_top20_membership_match"),
                "saved_model_top20_order_match": row.get("saved_model_top20_order_match"),
            }
            for row in authority.get("parity_results", [])
        },
        "missing_requirements": list(dict.fromkeys(missing)),
        "stable_artifact_hash": artifact_hash,
        "no_model_fits": True,
        "no_new_predictions": True,
    }
    payload["artifact_validation_hash"] = state_hash({key: value for key, value in payload.items() if key != "artifact_validation_hash"})
    write_json(R47A_XENDCG_ARTIFACT_VALIDATION_PATH, payload)
    return payload


def build_xendcg_mac_aux_import_contract() -> dict[str, Any]:
    ownership = ownership_by_family().get("lightgbm_rank_xendcg", {})
    discovery = xendcg_source_discovery()
    validation = xendcg_artifact_validation()
    summary = read_json(XENDCG_MAC_AUX_FAMILY_ROOT / "family_execution_summary.json")
    metrics = read_json(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "resolved_performance_summary_v3.json")
    oof_manifest = read_json(XENDCG_MAC_AUX_FAMILY_ROOT / "ensemble_oof_scores_manifest_v2.json")
    return {
        "family": "lightgbm_rank_xendcg",
        "mac_run_id": XENDCG_SOURCE_RUN_ID,
        "mac_producing_host": "MAC",
        "source_root_type": MAC_AUX_SOURCE_ROOT_TYPE,
        "queue_id": MAC_AUX_QUEUE_ID,
        "source_root": display_path(XENDCG_MAC_AUX_FAMILY_ROOT),
        "source_terminal_manifest": discovery,
        "producer_source_hash": XENDCG_PRODUCER_SHA,
        "model_config_authority_hash": summary.get("validation", {}).get("source_configuration_hash") or XENDCG_MODEL_CONFIG_HASH,
        "artifact_manifest": validation["model_inventory"],
        "artifact_hashes": {
            "manifest_sha256": validation["stable_artifact_hash"],
            "model_artifact_manifest_sha256": validation["stable_artifact_hash"],
            "fixture_hashes": validation["fixture_hash_validation"],
        },
        "metrics_manifest": {
            "path": display_path(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "resolved_performance_summary_v3.json"),
            "sha256": file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "resolved_performance_summary_v3.json"),
            "rows": metrics.get("rank_ic_rows"),
        },
        "metrics_hashes": {
            "resolved_performance_summary_v3": file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "resolved_performance_summary_v3.json"),
            "per_t_metrics": file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "per_t_metrics.parquet"),
            "decision_trace": file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "metrics_only_v3" / "decision_trace.parquet"),
        },
        "score_oof_population": {
            "path": display_path(XENDCG_MAC_AUX_FAMILY_ROOT / "ensemble_oof_scores_manifest_v2.json"),
            "manifest_sha256": file_hash(XENDCG_MAC_AUX_FAMILY_ROOT / "ensemble_oof_scores_manifest_v2.json"),
            "row_count": oof_manifest.get("row_count"),
            "distinct_assets": oof_manifest.get("distinct_assets"),
            "distinct_decision_timestamps": oof_manifest.get("distinct_decision_timestamps"),
        },
        "date_range": discovery["decision_date_range"],
        "target": metrics.get("target_contract", "forward_return_60m__decision_5m"),
        "target_contract_hash": XENDCG_TARGET_CONTRACT_HASH,
        "feature_count": 101,
        "feature_hash": XENDCG_FEATURE_ORDER_SHA,
        "refit_count": validation["model_inventory"]["genuine_pkl_count"],
        "terminal_status": summary.get("status", ""),
        "holdout_accessed": not bool(summary.get("zero_holdout", False)),
        "paper_orders": int(summary.get("paper_orders", 0) or 0),
        "live_orders": int(summary.get("live_orders", 0) or 0),
        "original_metrics": KNOWN_XENDCG_METRICS,
        "retained_metrics": metrics,
        "tournament_comparable_metrics_status": "DERIVED_FROM_RETAINED_MAC_OOF",
        "previous_ownership": ownership,
    }


def write_r47a_import_authority(verification: ImportVerification) -> dict[str, Any]:
    contract = build_xendcg_mac_aux_import_contract()
    payload = {
        "ticket": "DS24_R47A_XENDCG_IMPORT_AUTHORITY",
        "generated_at_utc": utc_now(),
        "classification": R47A_CLASSIFICATION if verification.classification == "COMPLETE_IMPORTED" else "FAIL_CLOSED",
        "contract": contract,
        "verification": verification.as_dict(),
        "old_r47_incorrect_search_root": display_path(MODEL_ARTIFACT_ROOT),
        "corrected_source_root": display_path(XENDCG_MAC_AUX_FAMILY_ROOT),
        "no_model_fits": True,
        "no_new_predictions": True,
        "no_worker_launches": True,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(R47A_XENDCG_IMPORT_AUTHORITY_PATH, payload)
    return payload


def write_r47a_ownership_state(verification: ImportVerification) -> dict[str, Any]:
    r44 = supervisor.validate_cross_host_ownership_authority(supervisor.R44_CROSS_HOST_OWNERSHIP_PATH)
    rows = []
    for row in r44["families"]:
        current = dict(row)
        if row.get("family") == "lightgbm_rank_xendcg" and verification.classification == "COMPLETE_IMPORTED":
            current.update(
                {
                    "execution_owner": "MAC",
                    "ownership_state": "COMPLETE_IMPORTED",
                    "owner_state": "COMPLETE_IMPORTED",
                    "dell_eligible": False,
                    "mac_eligible": False,
                    "ownership_reason": "R47A imported verified Mac auxiliary XENDCG result; Dell recomputation forbidden.",
                    "import_source_root_type": MAC_AUX_SOURCE_ROOT_TYPE,
                    "import_source_run_id": verification.source_run_id,
                    "import_artifact_hash": verification.artifact_hash,
                }
            )
        current["authority_hash"] = state_hash(current)
        rows.append(current)
    payload = {
        "ticket": "DS24_R47A_CROSS_HOST_OWNERSHIP_STATE",
        "generated_at_utc": utc_now(),
        "classification": R47A_CLASSIFICATION if verification.classification == "COMPLETE_IMPORTED" else "FAIL_CLOSED",
        "derived_from_r44_hash": r44["manifest_hash"],
        "historical_r44_preserved": True,
        "families": rows,
        "by_family": {str(row["family"]): row for row in rows},
    }
    write_json(R47A_OWNERSHIP_STATE_PATH, payload)
    return payload


def write_r47a_dell_effective_queue(ownership_state: Mapping[str, Any]) -> dict[str, Any]:
    ready = supervisor.validate_ready_family_queue_manifest(supervisor.R42_READY_QUEUE_PATH)
    queue = [family for family in ready["ready_family_queue"] if not cross_host_skip_reason_r47(family, ownership_state)]
    skipped = [
        {"family": family, "reason": cross_host_skip_reason_r47(family, ownership_state)}
        for family in ready["ready_family_queue"]
        if cross_host_skip_reason_r47(family, ownership_state)
    ]
    payload = {
        "ticket": "DS24_R47A_DELL_EFFECTIVE_READY_QUEUE",
        "generated_at_utc": utc_now(),
        "classification": R47A_CLASSIFICATION,
        "derived_from": {
            "r42_ready_queue_hash": ready["manifest_hash"],
            "r47a_ownership_state_hash": state_hash(ownership_state),
        },
        "dell_effective_ready_queue": queue,
        "excluded_families": skipped,
        "worker_launches": 0,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(R47A_DELL_EFFECTIVE_READY_QUEUE_PATH, payload)
    return payload


def run_r47a_xendcg_import() -> dict[str, Any]:
    verification = verify_family_import("lightgbm_rank_xendcg")
    duplicate = duplicate_guard(verification)
    if duplicate == "CROSS_HOST_IMPORT_HASH_CONFLICT":
        raise RuntimeError(f"CROSS_HOST_IMPORT_HASH_CONFLICT:lightgbm_rank_xendcg:{verification.source_run_id}")
    authority = write_r47a_import_authority(verification)
    ownership = write_r47a_ownership_state(verification)
    queue = write_r47a_dell_effective_queue(ownership)
    prior_failures = [
        event
        for event in ledger_events()
        if event.get("family") == "lightgbm_rank_xendcg" and event.get("validation_classification") != "COMPLETE_IMPORTED"
    ]
    if duplicate == "ALREADY_IMPORTED_IDENTICAL":
        result_artifact = {
            "ticket": "DS24_R47A_XENDCG_IMPORT_RESULT",
            "generated_at_utc": utc_now(),
            "classification": "ALREADY_IMPORTED_IDENTICAL",
            "terminal_classification": R47A_CLASSIFICATION,
            "family": "lightgbm_rank_xendcg",
            "source_run_id": verification.source_run_id,
            "artifact_hash": verification.artifact_hash,
            "import_result": "COMPLETE_IMPORTED",
            "duplicate_import_result": duplicate,
            "ledger_appended": False,
            "ownership_state": ownership["by_family"]["lightgbm_rank_xendcg"]["owner_state"],
            "effective_queue": queue["dell_effective_ready_queue"],
        }
        write_json(R47A_XENDCG_IMPORT_RESULT_PATH, result_artifact)
        return {
            "ticket": "DS24_R47A_XENDCG_IMPORT_RESULT",
            "generated_at_utc": result_artifact["generated_at_utc"],
            "classification": duplicate,
            "family": "lightgbm_rank_xendcg",
            "source_run_id": verification.source_run_id,
            "artifact_hash": verification.artifact_hash,
            "ledger_appended": False,
            "ownership_state": result_artifact["ownership_state"],
            "effective_queue": queue["dell_effective_ready_queue"],
        }
    event = {
        "timestamp_utc": utc_now(),
        "ticket": "DS24_R47A",
        "family": "lightgbm_rank_xendcg",
        "source_host": verification.source_host,
        "source_run_id": verification.source_run_id,
        "import_action": "IMPORT_FAMILY",
        "corrected_source_root": display_path(XENDCG_MAC_AUX_FAMILY_ROOT),
        "prior_r47_failure_count": len(prior_failures),
        "verified_artifact_count": verification.model_artifact_count,
        "artifact_hash": verification.artifact_hash,
        "authority_hashes": {
            "r47a_import_authority": authority["authority_hash"],
            "producer_source_hash": XENDCG_PRODUCER_SHA,
            "feature_order_sha256": XENDCG_FEATURE_ORDER_SHA,
        },
        "metrics_authority": verification.tournament_comparable_metrics_status,
        "previous_ownership": verification.previous_ownership,
        "resulting_ownership": "COMPLETE_IMPORTED" if verification.classification == "COMPLETE_IMPORTED" else verification.previous_ownership.get("owner_state", ""),
        "duplicate_status": duplicate,
        "validation_classification": verification.classification,
        "missing_requirements": list(verification.missing_requirements),
        "worker_launches": 0,
        "workers_stopped": 0,
        "holdout_accessed": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    append_ledger_event(event)
    result = {
        "ticket": "DS24_R47A_XENDCG_IMPORT_RESULT",
        "generated_at_utc": utc_now(),
        "classification": verification.classification,
        "terminal_classification": R47A_CLASSIFICATION if verification.classification == "COMPLETE_IMPORTED" else verification.classification,
        "family": "lightgbm_rank_xendcg",
        "import_result": verification.classification,
        "source_run_id": verification.source_run_id,
        "duplicate_status": duplicate,
        "ledger_appended": True,
        "event": event,
        "ownership_transition": {
            "previous": verification.previous_ownership.get("owner_state", ""),
            "resulting": ownership["by_family"]["lightgbm_rank_xendcg"]["owner_state"],
        },
        "effective_queue": queue["dell_effective_ready_queue"],
    }
    write_json(R47A_XENDCG_IMPORT_RESULT_PATH, result)
    return result


def imported_statuses(import_family: str | None = None) -> dict[str, ImportVerification]:
    statuses = {family: verify_family_import(family) for family in IMPORTABLE_FAMILIES}
    return statuses if import_family is None else {import_family: statuses[import_family]}


def write_import_authority(statuses: Mapping[str, ImportVerification]) -> dict[str, Any]:
    payload = {
        "ticket": "DS24_R47_CROSS_HOST_IMPORT_AUTHORITY",
        "generated_at_utc": utc_now(),
        "classification": R47_CLASSIFICATION,
        "required_fields": list(IMPORT_REQUIRED_FIELDS),
        "families": {family: status.as_dict() for family, status in statuses.items()},
        "no_model_fits": True,
        "no_worker_launches": True,
        "no_holdout_access": True,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(R47_IMPORT_AUTHORITY_PATH, payload)
    return payload


def write_family_import_status(statuses: Mapping[str, ImportVerification]) -> dict[str, Any]:
    payload = {
        "ticket": "DS24_R47_FAMILY_IMPORT_STATUS",
        "generated_at_utc": utc_now(),
        "classification": R47_CLASSIFICATION,
        "families": {family: status.as_dict() for family, status in statuses.items()},
    }
    write_json(R47_FAMILY_IMPORT_STATUS_PATH, payload)
    return payload


def write_r47_ownership_state(statuses: Mapping[str, ImportVerification]) -> dict[str, Any]:
    r44 = supervisor.validate_cross_host_ownership_authority(supervisor.R44_CROSS_HOST_OWNERSHIP_PATH)
    rows = []
    for row in r44["families"]:
        current = dict(row)
        imported = statuses.get(str(row.get("family")))
        if imported and imported.classification == "COMPLETE_IMPORTED":
            current.update(
                {
                    "execution_owner": "MAC",
                    "ownership_state": "COMPLETE_IMPORTED",
                    "owner_state": "COMPLETE_IMPORTED",
                    "dell_eligible": False,
                    "mac_eligible": False,
                    "ownership_reason": "Mac result imported and terminally accepted; Dell recomputation forbidden.",
                    "import_source_run_id": imported.source_run_id,
                    "import_artifact_hash": imported.artifact_hash,
                }
            )
        current["authority_hash"] = state_hash(current)
        rows.append(current)
    payload = {
        "ticket": "DS24_R47_CROSS_HOST_OWNERSHIP_STATE",
        "generated_at_utc": utc_now(),
        "classification": R47_CLASSIFICATION,
        "derived_from_r44_hash": r44["manifest_hash"],
        "families": rows,
        "by_family": {str(row["family"]): row for row in rows},
    }
    write_json(R47_OWNERSHIP_STATE_PATH, payload)
    return payload


def cross_host_skip_reason_r47(family: str, ownership: Mapping[str, Any]) -> str:
    row = ownership.get("by_family", {}).get(family, {})
    if row.get("owner_state") == "COMPLETE_IMPORTED" or row.get("ownership_state") == "COMPLETE_IMPORTED":
        return "SKIP_COMPLETE_IMPORTED"
    return supervisor.cross_host_skip_reason(family, ownership)


def write_r47_dell_effective_queue(ownership_state: Mapping[str, Any]) -> dict[str, Any]:
    ready = supervisor.validate_ready_family_queue_manifest(supervisor.R42_READY_QUEUE_PATH)
    queue = [family for family in ready["ready_family_queue"] if not cross_host_skip_reason_r47(family, ownership_state)]
    skipped = [
        {"family": family, "reason": cross_host_skip_reason_r47(family, ownership_state)}
        for family in ready["ready_family_queue"]
        if cross_host_skip_reason_r47(family, ownership_state)
    ]
    payload = {
        "ticket": "DS24_R47_DELL_EFFECTIVE_READY_QUEUE",
        "generated_at_utc": utc_now(),
        "classification": R47_CLASSIFICATION,
        "derived_from": {
            "r42_ready_queue_hash": ready["manifest_hash"],
            "r47_ownership_state_hash": supervisor.hash_if_exists(R47_OWNERSHIP_STATE_PATH),
        },
        "dell_effective_ready_queue": queue,
        "excluded_families": skipped,
        "worker_launches": 0,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(R47_DELL_EFFECTIVE_READY_QUEUE_PATH, payload)
    return payload


def write_future_transfer_contract(statuses: Mapping[str, ImportVerification]) -> dict[str, Any]:
    pending = {}
    for family in ("lightgbm_lambdarank", "DLinear"):
        pending[family] = {
            "required_bundle": [
                "terminal_manifest",
                "metrics_manifest",
                "oof_or_topn_artifacts",
                "model_artifact_hash_manifest",
                "checkpoint_or_final_status",
                "producer_config_authority",
                "producer_source_authority",
                "compact_provenance",
            ],
            "current_import_readiness": statuses[family].classification,
            "missing_requirements": list(statuses[family].missing_requirements),
            "transfer_large_model_artifacts": "ONLY_IF_REQUIRED_FOR_RETAINED_AUTHORITY_OR_FUTURE_USE",
        }
    payload = {
        "ticket": "DS24_R47_MAC_FUTURE_TRANSFER_CONTRACT",
        "generated_at_utc": utc_now(),
        "classification": R47_CLASSIFICATION,
        "families": pending,
        "do_not_transfer_large_datasets": True,
    }
    write_json(R47_MAC_TRANSFER_CONTRACT_PATH, payload)
    return payload


def pre_import_snapshot() -> dict[str, Any]:
    ready = supervisor.validate_ready_family_queue_manifest(supervisor.R42_READY_QUEUE_PATH)
    ownership = supervisor.validate_cross_host_ownership_authority(supervisor.R44_CROSS_HOST_OWNERSHIP_PATH)
    r46 = read_json(supervisor.R46_AUTOSTART_REGISTRATION_VALIDATION_PATH)
    board = supervisor.lightweight_certified_queue_board(ready["ready_family_queue"])
    gate = supervisor.resource_gate(board, supervisor.GateConfig(max_active_model_processes=3, max_policy_workers=3, max_system_commit_percent=95.0, admission_commit_percent=92.0))
    statuses = {family: verify_family_import(family).as_dict() for family in IMPORTABLE_FAMILIES}
    payload = {
        "ticket": "DS24_R47_CROSS_HOST_PRE_IMPORT_SNAPSHOT",
        "generated_at_utc": utc_now(),
        "classification": "PASS",
        "r42_ready_queue_hash": ready["manifest_hash"],
        "r44_ownership_hash": ownership["manifest_hash"],
        "r46_supervisor_autostart_state": r46.get("classification", ""),
        "supervisor_pid": read_json(supervisor.LEASE_PATH).get("pid"),
        "dell_active_workers": [row for row in board if row.get("pid_alive")],
        "mac_owned_families": ownership.get("excluded_mac_owned", []),
        "mac_reserved_families": ownership.get("excluded_mac_reserved", []),
        "family_artifact_presence": statuses,
        "resource_snapshot": gate.get("resource_snapshot", {}),
        "zero_full_prediction_guard": gate.get("zero_full_prediction_guard", {}),
        "worker_launches": 0,
        "workers_stopped": 0,
        "supervisor_restarted": 0,
    }
    write_json(R47_PRE_IMPORT_SNAPSHOT_PATH, payload)
    return payload


def run_import(family: str) -> dict[str, Any]:
    if family not in IMPORTABLE_FAMILIES:
        raise RuntimeError(f"UNSUPPORTED_CROSS_HOST_IMPORT_FAMILY:{family}")
    statuses = imported_statuses()
    verification = statuses[family]
    duplicate = duplicate_guard(verification)
    if duplicate == "ALREADY_IMPORTED_IDENTICAL":
        return {"classification": duplicate, "family": family, "source_run_id": verification.source_run_id}
    if duplicate == "CROSS_HOST_IMPORT_HASH_CONFLICT":
        raise RuntimeError(f"CROSS_HOST_IMPORT_HASH_CONFLICT:{family}:{verification.source_run_id}")
    event = {
        "timestamp_utc": utc_now(),
        "family": family,
        "source_host": verification.source_host,
        "source_run_id": verification.source_run_id,
        "import_action": "IMPORT_FAMILY",
        "artifact_hash": verification.artifact_hash,
        "metrics_authority": verification.tournament_comparable_metrics_status,
        "previous_ownership": verification.previous_ownership,
        "resulting_ownership": "COMPLETE_IMPORTED" if verification.classification == "COMPLETE_IMPORTED" else verification.previous_ownership.get("owner_state", ""),
        "duplicate_status": duplicate,
        "validation_classification": verification.classification,
        "missing_requirements": list(verification.missing_requirements),
        "worker_launches": 0,
        "workers_stopped": 0,
        "holdout_accessed": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    append_ledger_event(event)
    write_import_authority(statuses)
    write_family_import_status(statuses)
    ownership = write_r47_ownership_state(statuses)
    queue = write_r47_dell_effective_queue(ownership)
    transfer = write_future_transfer_contract(statuses)
    return {
        "classification": verification.classification,
        "family": family,
        "event": event,
        "effective_queue": queue.get("dell_effective_ready_queue", []),
        "future_transfer_contract": transfer.get("families", {}),
    }


def status() -> dict[str, Any]:
    snapshot = pre_import_snapshot()
    statuses = imported_statuses()
    authority = write_import_authority(statuses)
    family_status = write_family_import_status(statuses)
    ownership = write_r47_ownership_state(statuses)
    queue = write_r47_dell_effective_queue(ownership)
    transfer = write_future_transfer_contract(statuses)
    return {
        "classification": R47_CLASSIFICATION,
        "snapshot": snapshot,
        "authority": authority,
        "family_import_status": family_status,
        "ownership_state": ownership,
        "dell_effective_ready_queue": queue,
        "future_transfer_contract": transfer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--audit-family")
    mode.add_argument("--import-family")
    mode.add_argument("--r47a-xendcg-source-discovery", action="store_true")
    mode.add_argument("--r47a-xendcg-artifact-validation", action="store_true")
    mode.add_argument("--r47a-import-xendcg", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.r47a_xendcg_source_discovery:
        print(json.dumps(xendcg_source_discovery(), indent=2, sort_keys=True))
        return 0
    if args.r47a_xendcg_artifact_validation:
        print(json.dumps(xendcg_artifact_validation(), indent=2, sort_keys=True))
        return 0
    if args.r47a_import_xendcg:
        print(json.dumps(run_r47a_xendcg_import(), indent=2, sort_keys=True))
        return 0
    if args.audit_family:
        print(json.dumps(verify_family_import(args.audit_family).as_dict(), indent=2, sort_keys=True))
        return 0
    if args.import_family:
        print(json.dumps(run_import(args.import_family), indent=2, sort_keys=True))
        return 0
    print(json.dumps(status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
