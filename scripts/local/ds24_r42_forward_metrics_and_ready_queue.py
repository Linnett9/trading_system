from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.ds24_forward_metrics_capability import (
    build_forward_metrics_capability,
    write_forward_metrics_capability,
)
from scripts.local import ds24_full_family_readiness_audit as r41
from scripts.local import ds24_lightgbm_ranking_readiness as ranking_readiness
from scripts.local import ds24_v3_sequence_policy_worker as sequence_worker
from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


STAGE = r41.STAGE
CAPABILITY_ROOT = STAGE / "R42_forward_metrics_capabilities"
OUTPUT_JSON = STAGE / "R42_full_family_readiness_matrix.json"
OUTPUT_CSV = STAGE / "R42_full_family_readiness_matrix.csv"
REPAIR_RESULTS_JSON = STAGE / "R42_readiness_repair_results.json"
READY_QUEUE_JSON = STAGE / "R42_ready_family_queue.json"
R41_MATRIX_JSON = STAGE / "R41_full_family_readiness_matrix.json"

REPAIRABLE_FORWARD_FAMILIES = (
    "elastic_net",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
)
SEQUENCE_FAMILIES = (
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
)
READY_ORDER = (
    "random_forest",
    "elastic_net",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
)
CSV_COLUMNS = (
    "family",
    "previous_r41_state",
    "current_r42_state",
    "repair_performed",
    "forward_metrics_admitted",
    "implementation_ready",
    "bounded_certification_state",
    "supervisor_launch_enabled",
    "remaining_blocker",
    "evidence_paths",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def family_slug(family: str) -> str:
    return r41.DISPLAY_TO_SLUG.get(family, re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_"))


def previous_r41_rows() -> dict[str, dict[str, Any]]:
    matrix = read_json(R41_MATRIX_JSON) or r41.build_matrix()
    return {str(row.get("family")): row for row in matrix.get("families", []) if isinstance(row, dict)}


def record_from_manifest(path: Path, family: str) -> dict[str, Any]:
    payload = read_json(path)
    slug = family_slug(family)
    if str(payload.get("family", "")) in {family, slug}:
        return dict(payload.get("evidence", payload))
    for row in payload.get("families", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("family", "")) in {family, slug}:
            return dict(row)
    return {}


def certification_evidence(family: str) -> dict[str, Any]:
    if family == "elastic_net":
        record = record_from_manifest(STAGE / "R7_LOW_USAGE_ELASTIC_NET_READINESS.json", family)
        return {
            "state": record.get("state", ""),
            "path": display_path(STAGE / "R7_LOW_USAGE_ELASTIC_NET_READINESS.json"),
            "record": record,
        }
    if family in {"lightgbm_rank_xendcg", "lightgbm_lambdarank"}:
        record = record_from_manifest(STAGE / "R7_LOW_USAGE_LIGHTGBM_RANKING_READINESS.json", family)
        if not record:
            record = ranking_readiness.family_record(family, CAPABILITY_ROOT / "_tmp_ranking")
        return {
            "state": record.get("state", ""),
            "path": display_path(STAGE / "R7_LOW_USAGE_LIGHTGBM_RANKING_READINESS.json"),
            "record": record,
        }
    if family in SEQUENCE_FAMILIES:
        record = record_from_manifest(STAGE / "R7_LOW_USAGE_V3_SEQUENCE_WORKER_READINESS.json", family)
        return {
            "state": record.get("state", ""),
            "path": display_path(STAGE / "R7_LOW_USAGE_V3_SEQUENCE_WORKER_READINESS.json"),
            "record": record,
        }
    return {"state": "", "path": "", "record": {}}


def worker_route_ready(family: str) -> bool:
    if family in SEQUENCE_FAMILIES:
        try:
            sequence_worker.assert_supported_family(family)
        except Exception:
            return False
    row = supervisor.execution_registry_row(family)
    script = ROOT / str(row.get("worker_script", "")).replace("/", "\\")
    return bool(script.exists() and row.get("worker_kind") in {"TABULAR", "LIGHTGBM_RANKING", "PYTORCH_SEQUENCE"})


def capability_for_family(family: str, evidence: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    row = supervisor.execution_registry_row(family)
    capability = build_forward_metrics_capability(
        family=family,
        worker_kind=str(row.get("worker_kind", "")),
        worker_script=str(row.get("worker_script", "")),
        metrics_root_name=supervisor.default_metrics_root_name(family, "metrics_only_v3"),
        evidence_basis={
            "ticket": "DS24_R42_FORWARD_METRICS_CAPABILITY_REPAIR",
            "source_evidence_path": evidence.get("path", ""),
            "source_state": evidence.get("state", ""),
            "live_supervisor_visible": False,
        },
    )
    path = CAPABILITY_ROOT / f"{family_slug(family)}.forward_metrics_capability.json"
    write_forward_metrics_capability(path, capability)
    return capability, display_path(path)


def repaired_record(family: str, previous: Mapping[str, Any]) -> dict[str, Any]:
    evidence = certification_evidence(family)
    capability, capability_path = capability_for_family(family, evidence)
    admission = supervisor.forward_metrics_contract_admission_decision(
        family,
        evaluation_version="v3",
        metrics_root_name=supervisor.default_metrics_root_name(family, "metrics_only_v3"),
        capability=capability,
    )
    implementation_ready = worker_route_ready(family)
    certification_state = str(evidence.get("state", ""))
    certification_ready = certification_state in {"V3_CERTIFIED_READY", "V3_SEQUENCE_WORKER_CERTIFIED_READY"}
    if not admission.get("admitted"):
        state = "FORWARD_METRICS_CONTRACT_REQUIRED"
        blocker = ",".join(admission.get("missing_requirements", []))
    elif not implementation_ready:
        state = "IMPLEMENTATION_BLOCKED"
        blocker = "worker route missing or family does not resolve"
    elif not certification_ready:
        state = "V3_REPLAY_REQUIRED"
        blocker = f"bounded certification state={certification_state or 'missing'}"
    else:
        state = "READY_TO_LAUNCH"
        blocker = ""
    return {
        "family": family,
        "previous_r41_state": previous.get("state", ""),
        "current_r42_state": state,
        "repair_performed": "forward_metrics_capability_published_and_validated",
        "forward_metrics": {
            "admitted": bool(admission.get("admitted")),
            "capability_decision": admission.get("capability_decision", ""),
            "capability_path": capability_path,
            "contract_hash": admission.get("contract_hash", ""),
        },
        "implementation": {
            "ready": implementation_ready,
            "worker_kind": supervisor.worker_kind_for_family(family),
            "worker_script": supervisor.worker_script_for_family(family),
            "supervisor_launch_enabled": bool(supervisor.launch_enabled_for_family(family)),
        },
        "bounded_certification": {
            "state": certification_state,
            "evidence_path": evidence.get("path", ""),
            "record_keys": sorted(evidence.get("record", {}).keys()),
        },
        "safety": {
            "worker_launches": 0,
            "workers_stopped": 0,
            "live_supervisor_queue_modified": False,
            "live_namespaces_modified": False,
            "holdout_accessed": False,
            "full_prediction_files_written": 0,
            "paper_orders": 0,
            "live_orders": 0,
        },
        "remaining_blocker": blocker,
        "evidence_paths": [item for item in [capability_path, str(evidence.get("path", ""))] if item],
    }


def inherited_record(family: str, previous: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": family,
        "previous_r41_state": previous.get("state", ""),
        "current_r42_state": previous.get("state", ""),
        "repair_performed": "inherited_from_r41",
        "forward_metrics": previous.get("v3_forward_metrics", {}),
        "implementation": previous.get("implementation", {}),
        "bounded_certification": previous.get("bounded_replay", {}),
        "safety": {
            "worker_launches": 0,
            "workers_stopped": 0,
            "live_supervisor_queue_modified": False,
            "live_namespaces_modified": False,
        },
        "remaining_blocker": previous.get("blocker", ""),
        "evidence_paths": previous.get("evidence_paths", []),
    }


def tft_record(previous: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": "Temporal Fusion Transformer",
        "previous_r41_state": previous.get("state", ""),
        "current_r42_state": "CONFIGURATION_AUTHORITY_REQUIRED",
        "repair_performed": "not_repaired_without_accepted_tft_tournament_authority",
        "forward_metrics": {"admitted": False, "capability_decision": "TFT_AUTHORITY_HELD_CLOSED"},
        "implementation": {
            "ready": False,
            "worker_kind": supervisor.worker_kind_for_family("Temporal Fusion Transformer"),
            "worker_script": supervisor.worker_script_for_family("Temporal Fusion Transformer"),
            "supervisor_launch_enabled": False,
        },
        "bounded_certification": {
            "state": "CONFIGURATION_AUTHORITY_REQUIRED",
            "evidence_path": display_path(STAGE / "R7_R40_06_tft_certification.json"),
        },
        "safety": {
            "worker_launches": 0,
            "workers_stopped": 0,
            "live_supervisor_queue_modified": False,
            "live_namespaces_modified": False,
            "holdout_accessed": False,
        },
        "remaining_blocker": "R40/R41 accepted full TFT static/recurrent tournament authority not found",
        "evidence_paths": [display_path(STAGE / "R7_R40_06_tft_certification.json")],
    }


def build_matrix() -> dict[str, Any]:
    previous = previous_r41_rows()
    records: list[dict[str, Any]] = []
    for family in r41.UNFINISHED_FAMILIES:
        prior = previous.get(family, {})
        if family in REPAIRABLE_FORWARD_FAMILIES:
            records.append(repaired_record(family, prior))
        elif family == "Temporal Fusion Transformer":
            records.append(tft_record(prior))
        else:
            records.append(inherited_record(family, prior))
    ready = [family for family in READY_ORDER if any(row["family"] == family and row["current_r42_state"] == "READY_TO_LAUNCH" for row in records)]
    repair_results = [row for row in records if row["repair_performed"] != "inherited_from_r41"]
    return {
        "ticket": "DS24_R42_FORWARD_METRICS_CAPABILITY_REPAIR_AND_READY_QUEUE",
        "generated_at_utc": r41.utc_now(),
        "parent_r41_commit": "198aca19e",
        "r41_matrix_path": display_path(R41_MATRIX_JSON),
        "capability_root": display_path(CAPABILITY_ROOT),
        "read_only_runtime": True,
        "worker_launches": 0,
        "workers_stopped": 0,
        "live_supervisor_queue_modified": False,
        "running_verified_families_before_after_expected": ["mlp", "extra_trees", "gradient_boosting"],
        "ready_to_launch_families": ready,
        "recommended_supervisor_queue_when_slot_opens": ready,
        "families": records,
        "repair_results": repair_results,
        "remaining_repair_queue": [
            {
                "family": row["family"],
                "state": row["current_r42_state"],
                "blocker": row["remaining_blocker"],
                "evidence_paths": row["evidence_paths"],
            }
            for row in records
            if row["current_r42_state"] not in {"READY_TO_LAUNCH", "RUNNING_VERIFIED"}
        ],
    }


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


def publish(matrix: Mapping[str, Any]) -> None:
    write_json_atomic(OUTPUT_JSON, matrix, advisory=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in matrix["families"]:
            writer.writerow(
                {
                    "family": row["family"],
                    "previous_r41_state": row["previous_r41_state"],
                    "current_r42_state": row["current_r42_state"],
                    "repair_performed": row["repair_performed"],
                    "forward_metrics_admitted": csv_cell(row.get("forward_metrics", {}).get("admitted", "")),
                    "implementation_ready": csv_cell(row.get("implementation", {}).get("ready", "")),
                    "bounded_certification_state": csv_cell(row.get("bounded_certification", {}).get("state", "")),
                    "supervisor_launch_enabled": csv_cell(row.get("implementation", {}).get("supervisor_launch_enabled", "")),
                    "remaining_blocker": row["remaining_blocker"],
                    "evidence_paths": csv_cell(row["evidence_paths"]),
                }
            )
    write_json_atomic(
        REPAIR_RESULTS_JSON,
        {
            "ticket": matrix["ticket"],
            "generated_at_utc": matrix["generated_at_utc"],
            "repair_results": matrix["repair_results"],
            "remaining_repair_queue": matrix["remaining_repair_queue"],
        },
        advisory=True,
    )
    write_json_atomic(
        READY_QUEUE_JSON,
        {
            "ticket": matrix["ticket"],
            "generated_at_utc": matrix["generated_at_utc"],
            "ready_family_queue": matrix["ready_to_launch_families"],
            "recommended_supervisor_queue_when_slot_opens": matrix["recommended_supervisor_queue_when_slot_opens"],
            "not_applied_to_live_supervisor": True,
        },
        advisory=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS24 R42 forward metrics capability repair and ready queue publisher")
    parser.add_argument("--no-publish", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix()
    if not args.no_publish:
        publish(matrix)
    print(
        json.dumps(
            {
                "matrix_path": display_path(OUTPUT_JSON) if not args.no_publish else "",
                "ready_to_launch": matrix["ready_to_launch_families"],
                "remaining_repair_count": len(matrix["remaining_repair_queue"]),
                "worker_launches": matrix["worker_launches"],
                "workers_stopped": matrix["workers_stopped"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
