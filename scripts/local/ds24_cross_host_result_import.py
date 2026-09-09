from __future__ import annotations

import argparse
import csv
import json
import os
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
IMPORTABLE_FAMILIES = ("lightgbm_rank_xendcg", "lightgbm_lambdarank", "DLinear")
XENDCG_EXPECTED_MODEL_ARTIFACTS = 2262
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
    "ndcg_at_20": 0.228239431375,
    "directional_accuracy": 0.580621053511,
    "daily_sharpe": 13.233750175466,
    "annualised_arithmetic_return": 1.774883897978,
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
        return "MAC_LIGHTGBM_RANK_XENDCG_COMPLETE_TRANSFERRED_AUTHORITY"
    if family == "lightgbm_lambdarank":
        return "MAC_LIGHTGBM_LAMBDARANK_PENDING_COMPLETION"
    return "MAC_DLINEAR_RESERVED_NEXT"


def expected_model_count(family: str) -> int:
    return XENDCG_EXPECTED_MODEL_ARTIFACTS if family == "lightgbm_rank_xendcg" else 0


def build_import_contract(family: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    root = artifact_root or MODEL_ARTIFACT_ROOT
    ownership = ownership_by_family().get(family, {})
    artifacts = family_model_artifacts(family, root)
    sidecars = apple_double_sidecars(family, root)
    inventory_rows = csv_family_rows(STAGE / "20_model_artifact_inventory.csv", family) + csv_family_rows(STAGE / "20_model_artifact_inventory_r1_live.csv", family)
    status = read_json(family_status_file(family))
    artifact_hash = manifest_hash(artifacts)
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
    contract = build_import_contract(family, artifact_root=artifact_root or MODEL_ARTIFACT_ROOT)
    artifacts = contract["artifact_manifest"]
    missing = missing_contract_fields(contract)
    if family == "lightgbm_rank_xendcg" and artifacts.get("model_artifact_count") != XENDCG_EXPECTED_MODEL_ARTIFACTS:
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
        artifact_root=str(contract["artifact_manifest"]["root"]),
        model_artifact_count=int(contract["artifact_manifest"]["model_artifact_count"]),
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
        if event.get("artifact_hash") == verification.artifact_hash:
            return "ALREADY_IMPORTED_IDENTICAL"
        return "CROSS_HOST_IMPORT_HASH_CONFLICT"
    return "NEW_IMPORT"


def append_ledger_event(event: Mapping[str, Any], path: Path | None = None) -> None:
    ledger_path = path or R47_IMPORT_LEDGER_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, default=str, separators=(",", ":")) + "\n")


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
