from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_storage import validate_artifact_package
from .lease_storage import atomic_write_json, exclusive_file_lock
from .run_contracts import (
    RUN_STATUS_CONTRACT, build_item_status, checksum, derive_run_status,
    semantic_payload, validate_item_status, validate_run_manifest,
)
from .run_results import render_summary, results_payload

DEFAULT_RUNS_ROOT = Path("reports/runs")


class StaleRunRevision(RuntimeError):
    pass


def initialise_run(
    manifest: Mapping[str, Any], *, runs_root: Path = DEFAULT_RUNS_ROOT,
) -> Path:
    validate_run_manifest(manifest)
    root = runs_root / str(manifest["run_root_relative_path"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    with exclusive_file_lock(root / "run.lock"):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            validate_run_manifest(existing)
            if existing.get("compatibility_identity") != manifest.get("compatibility_identity"):
                raise ValueError("INCOMPATIBLE: existing run ownership differs")
            return root
        if any(root.iterdir()):
            allowed = {"run.lock"}
            if any(child.name not in allowed for child in root.iterdir()):
                raise ValueError("INCOMPATIBLE: non-empty unowned run root")
        atomic_write_json(path, manifest)
        _write_csv(root / "component_status.csv", [])
        _write_csv(root / "failures.csv", [])
        _write_csv(root / "artifact_inventory.csv", [])
    return root


def publish_item_status(
    run_root: Path, row: Mapping[str, Any]
) -> None:
    validate_item_status(row)
    manifest = read_run_manifest(run_root)
    expected = {item["item_id"]: item for item in manifest["expected_inventory"]}
    if row["item_id"] not in expected or row["run_identity"] != manifest["run_identity"]:
        raise ValueError("Unknown or incompatible run item identity")
    _validate_authoritative_completion(row, manifest)
    with exclusive_file_lock(run_root / "run.lock"):
        rows = _read_csv(run_root / "component_status.csv")
        by_id = {item["item_id"]: item for item in rows}
        by_id[str(row["item_id"])] = _json_safe_row(row)
        _write_csv(
            run_root / "component_status.csv",
            sorted(by_id.values(), key=lambda item: int(item["ordered_position"])),
        )


def update_run_status(
    run_root: Path, *, expected_revision: int | None,
    inputs_valid: bool, evaluation_required: bool = False,
    evaluation_artifacts_valid: bool = False, cancelled: bool = False,
    fail_run_on_required_failure: bool = True,
    resource_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with exclusive_file_lock(run_root / "run.lock"):
        manifest = read_run_manifest(run_root)
        current = _read_json_optional(run_root / "run_status.json")
        revision = int(current.get("state_revision", -1)) if current else -1
        if expected_revision is not None and expected_revision != revision:
            raise StaleRunRevision(
                f"Expected run revision {expected_revision}, found {revision}"
            )
        rows = _read_csv(run_root / "component_status.csv")
        for row in rows:
            _restore_json_fields(row)
            validate_item_status(row)
        known = {row["item_id"] for row in rows}
        for expected in manifest["expected_inventory"]:
            if expected["item_id"] not in known:
                rows.append(build_item_status(
                    run_identity=manifest["run_identity"],
                    item_id=expected["item_id"],
                    ordered_position=int(expected["ordered_position"]),
                    pipeline=manifest["pipeline"], stage=manifest["stage"],
                    attempt_identity="", status="PLANNED",
                ))
        current_status = derive_run_status(
            rows, inputs_valid=inputs_valid,
            evaluation_required=evaluation_required,
            evaluation_artifacts_valid=evaluation_artifacts_valid,
            cancelled=cancelled,
            fail_run_on_required_failure=fail_run_on_required_failure,
        )
        counts = _counts(rows, len(manifest["expected_inventory"]))
        now = datetime.now(timezone.utc).isoformat()
        resources = dict(resource_evidence or {})
        payload = {
            "contract_version": RUN_STATUS_CONTRACT,
            "run_identity": manifest["run_identity"],
            "state_revision": revision + 1, "current_status": current_status,
            "started_timestamp": (
                current.get("started_timestamp") if current else None
            ) or (now if current_status not in {"PLANNED", "INPUTS_READY"} else None),
            "completed_timestamp": (
                now if current_status in {
                    "COMPONENTS_COMPLETE", "EVALUATION_COMPLETE", "FAILED", "CANCELLED"
                } else None
            ),
            "counts": counts,
            "active_resource_lease_identities": resources.get("active_lease_identities", []),
            "latest_telemetry_identity": resources.get("telemetry_identity"),
            "latest_resource_summary_identity": resources.get("resource_summary_identity"),
            "latest_artifact_inventory_identity": resources.get("artifact_inventory_identity"),
            "latest_results_identity": resources.get("results_identity"),
            "latest_reconciliation_timestamp": resources.get("reconciliation_timestamp"),
            "reserved_ram_bytes": resources.get("reserved_ram_bytes"),
            "measured_peak_ram_bytes": resources.get("measured_peak_ram_bytes"),
            "resource_wait_seconds": resources.get("resource_wait_seconds"),
            "estimate_exceeded": resources.get("estimate_exceeded"),
            "blocker_summary": sorted({
                str(row.get("blocker_reason")) for row in rows if row.get("blocker_reason")
            }),
            "failure_summary": sorted({
                str(row.get("failure_reason")) for row in rows if row.get("failure_reason")
            }),
            "next_required_action": _next_action(current_status),
            "latest_update_timestamp": now,
        }
        payload["logical_checksum"] = checksum(semantic_payload(payload, mutable=True))
        atomic_write_json(run_root / "run_status.json", payload)
        return payload


def build_artifact_inventory(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    inventory = []
    for source in entries:
        root = Path(str(source["package_root"]))
        manifest = validate_artifact_package(root)
        records = manifest.get("file_inventory", [])
        inventory.append({
            "artifact_identity": manifest["artifact_id"],
            "artifact_type": manifest["artifact_type"],
            "artifact_subtype": manifest["artifact_subtype"],
            "artifact_role": manifest["artifact_role"],
            "manifest_path": str(root / "manifest.json"),
            "package_checksum": manifest["package_checksum"],
            "owning_run_identity": source["owning_run_identity"],
            "owning_item_identity": source["owning_item_identity"],
            "status": manifest["completion_status"],
            "fitted_model_applicable": manifest["artifact_type"] == "FITTED_MODEL",
            "prediction_applicable": manifest["artifact_type"] == "PREDICTION_ARTIFACT",
            "evaluation_applicable": manifest["artifact_type"] == "EVALUATION_ARTIFACT",
            "promotion_status": manifest["promotion_status"],
            "file_count": len(records),
            "total_bytes": sum(int(row["size_bytes"]) for row in records),
            "compatibility_validation_status": "VALID",
        })
    return sorted(inventory, key=lambda row: (
        row["owning_item_identity"], row["artifact_identity"]
    ))


def publish_artifact_inventory(
    run_root: Path, entries: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    inventory = build_artifact_inventory(entries)
    identity = checksum(inventory)
    with exclusive_file_lock(run_root / "run.lock"):
        _write_csv(run_root / "artifact_inventory.csv", inventory)
    return inventory, identity


def publish_results_snapshot(
    run_root: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    manifest = read_run_manifest(run_root)
    known = {row["item_id"] for row in manifest["expected_inventory"]}
    if any(row.get("item_identity") not in known for row in records):
        raise ValueError("Result record references unknown run item")
    payload, csv_text = results_payload(manifest["run_identity"], records)
    with exclusive_file_lock(run_root / "run.lock"):
        atomic_write_json(run_root / "results.json", payload)
        _write_text_atomic(run_root / "results.csv", csv_text)
    return payload


def publish_leaderboard(
    run_root: Path, leaderboard: Mapping[str, Any]
) -> None:
    if leaderboard.get("contract_version") != "compute_model_leaderboard.v1":
        raise ValueError("Leaderboard contract mismatch")
    rows = list(leaderboard.get("ordered_entries", [])) + list(
        leaderboard.get("excluded_entries", [])
    )
    with exclusive_file_lock(run_root / "run.lock"):
        atomic_write_json(run_root / "leaderboard.json", leaderboard)
        _write_csv(run_root / "leaderboard.csv", rows)


def update_global_registry_snapshot(
    run_root: Path, *, registry_path: Path,
    expected_registry_revision: int | None = None,
) -> dict[str, Any]:
    manifest = read_run_manifest(run_root)
    status = _validated_status(run_root)
    from .run_registry import registry_record, update_run_registry

    try:
        registry = update_run_registry(
            registry_record(
                manifest, status,
                summary_path=str(run_root / "summary.md"),
                result_path=str(run_root / "results.json"),
            ),
            path=registry_path,
            expected_revision=expected_registry_revision,
        )
        return {"health": "HEALTHY", "registry_revision": registry["revision"]}
    except Exception as exc:
        return {
            "health": "DEGRADED_REGISTRY",
            "status_revision": status["state_revision"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def publish_failure_and_blocker_records(
    run_root: Path, *, failures: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
) -> None:
    manifest = read_run_manifest(run_root)
    known = {row["item_id"] for row in manifest["expected_inventory"]}
    if any(
        row.get("run_identity") != manifest["run_identity"]
        or row.get("item_identity") not in known
        for row in failures
    ):
        raise ValueError("Failure record references unknown run/item")
    if any(row.get("run_identity") != manifest["run_identity"] for row in blockers):
        raise ValueError("Blocker record references incompatible run")
    with exclusive_file_lock(run_root / "run.lock"):
        _write_csv(
            run_root / "failures.csv",
            sorted(failures, key=lambda row: (
                str(row["item_identity"]), str(row["failure_code"])
            )),
        )
        _write_csv(
            run_root / "blockers.csv",
            sorted(blockers, key=lambda row: str(row["blocker_code"])),
        )


def publish_summary(
    run_root: Path, *, artifact_inventory: Sequence[Mapping[str, Any]] = (),
    leaderboard: Mapping[str, Any] | None = None,
) -> str:
    manifest = read_run_manifest(run_root)
    status = _validated_status(run_root)
    important = [
        str(row["manifest_path"]) for row in artifact_inventory
    ] + [str(run_root / "results.json"), str(run_root / "run_status.json")]
    text = render_summary(
        manifest, status, artifact_inventory=artifact_inventory,
        leaderboard=leaderboard, important_paths=important,
    )
    with exclusive_file_lock(run_root / "run.lock"):
        _write_text_atomic(run_root / "summary.md", text)
    return text


def read_run_manifest(run_root: Path) -> dict[str, Any]:
    payload = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    validate_run_manifest(payload)
    return payload


def validate_run_compatibility(
    run_root: Path, expected_manifest: Mapping[str, Any]
) -> bool:
    return (
        read_run_manifest(run_root).get("compatibility_identity")
        == expected_manifest.get("compatibility_identity")
    )


def _validated_status(run_root: Path) -> dict[str, Any]:
    payload = json.loads((run_root / "run_status.json").read_text(encoding="utf-8"))
    if payload.get("contract_version") != RUN_STATUS_CONTRACT:
        raise ValueError("Run status contract mismatch")
    if payload.get("logical_checksum") != checksum(semantic_payload(payload, mutable=True)):
        raise ValueError("Run status checksum mismatch")
    return payload


def _counts(rows: Sequence[Mapping[str, Any]], expected: int) -> dict[str, int]:
    statuses = [row["status"] for row in rows]
    return {
        "expected": expected,
        "completed": sum(row in {"COMPLETE", "SKIPPED_COMPATIBLE"} for row in statuses),
        "running": statuses.count("RUNNING"),
        "waiting": statuses.count("WAITING_FOR_RESOURCES"),
        "blocked": statuses.count("BLOCKED") + statuses.count("CORRUPT"),
        "failed": statuses.count("FAILED"),
        "cancelled": statuses.count("CANCELLED"),
        "skipped_compatible": statuses.count("SKIPPED_COMPATIBLE"),
        "incomplete": sum(row in {"PLANNED", "INPUTS_READY", "INCOMPLETE"} for row in statuses),
    }


def _next_action(status: str) -> str:
    return {
        "PLANNED": "Validate required input artifacts.",
        "INPUTS_READY": "Request compute resources.",
        "WAITING_FOR_RESOURCES": "Wait for the machine-wide resource lease.",
        "RUNNING": "Monitor component and telemetry evidence.",
        "PARTIALLY_COMPLETE": "Resume remaining compatible components.",
        "COMPONENTS_COMPLETE": "Publish required evaluation artifacts.",
        "EVALUATION_COMPLETE": "Review pipeline-owned eligibility evidence.",
        "BLOCKED": "Resolve authoritative blockers.",
        "FAILED": "Review failure evidence and retry policy.",
        "CANCELLED": "Review cancellation audit evidence.",
    }[status]


def _validate_authoritative_completion(
    row: Mapping[str, Any], run_manifest: Mapping[str, Any]
) -> None:
    if row.get("status") not in {"COMPLETE", "SKIPPED_COMPATIBLE"}:
        return
    kind = row.get("required_artifact_kind", "NONE")
    if kind == "STAGE":
        package = row.get("stage_artifact_package_path")
        if not package:
            raise ValueError("Completed stage requires authoritative artifact package")
        artifact = validate_artifact_package(Path(str(package)))
        if artifact.get("run_id") != run_manifest.get("run_id"):
            raise ValueError("Stage artifact run ancestry mismatch")
    elif kind == "MODEL":
        model_path = row.get("fitted_model_package_path")
        if not model_path:
            raise ValueError("Completed model requires authoritative model package")
        model_manifest = validate_artifact_package(Path(str(model_path)))
        from .model_artifacts import validate_model_artifact_manifest

        validate_model_artifact_manifest(model_manifest)
        if model_manifest.get("run_id") != run_manifest.get("run_id"):
            raise ValueError("Model artifact run ancestry mismatch")
        if row.get("predictions_required"):
            prediction_path = row.get("prediction_package_path")
            if not prediction_path:
                raise ValueError("Completed model requires prediction package")
            prediction = validate_artifact_package(Path(str(prediction_path)))
            from .artifact_contracts import validate_prediction_binding

            validate_prediction_binding(prediction, model_manifest)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["item_id"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(_json_safe_row(row))
    _write_text_atomic(path, output.getvalue())


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _json_safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple, bool)) or value is None else value
        )
        for key, value in row.items()
    }


def _restore_json_fields(row: dict[str, Any]) -> None:
    for key, value in list(row.items()):
        if isinstance(value, str) and (
            value.startswith("{") or value.startswith("[")
            or value in {"true", "false", "null"}
        ):
            try:
                row[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    row["ordered_position"] = int(row["ordered_position"])


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("logical_checksum") != checksum(semantic_payload(payload, mutable=True)):
        raise ValueError("Run status checksum mismatch")
    return payload
