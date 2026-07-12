from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_MANIFEST_NAME = "run_manifest.json"
LATEST_COMPLETED_NAME = "latest_completed.json"
CHAMPION_NAME = "champion.json"


@dataclass(frozen=True)
class ImmutableRunRecord:
    run_id: str
    run_dir: Path
    manifest_path: Path
    latest_completed_path: Path


def deterministic_run_id(kind: str, identity: Mapping[str, Any]) -> str:
    payload = {
        "kind": kind,
        "identity": identity,
    }
    digest = _sha256_json(payload)[:16]
    return f"{kind}-{digest}"


def immutable_run_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / "runs" / run_id


def latest_completed_pointer_path(output_dir: Path) -> Path:
    return output_dir / LATEST_COMPLETED_NAME


def champion_pointer_path(output_dir: Path) -> Path:
    return output_dir / CHAMPION_NAME


def read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / RUN_MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_complete_run_dir(run_dir: Path) -> bool:
    manifest = read_run_manifest(run_dir)
    return bool(manifest and manifest.get("run_status") == "complete")


def run_dir_from_latest_completed(output_dir: Path) -> Path | None:
    payload = _read_pointer(latest_completed_pointer_path(output_dir))
    if not payload:
        return None
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    run_dir = immutable_run_dir(output_dir, run_id)
    return run_dir if run_dir.exists() else None


def preserve_immutable_run(
    *,
    output_dir: Path,
    run_id: str,
    kind: str,
    identity: Mapping[str, Any],
    artifact_paths: Sequence[Path],
    extra_manifest: Mapping[str, Any] | None = None,
) -> ImmutableRunRecord:
    run_dir = immutable_run_dir(output_dir, run_id)
    manifest_path = run_dir / RUN_MANIFEST_NAME
    latest_path = latest_completed_pointer_path(output_dir)
    existing = read_run_manifest(run_dir)
    if existing and existing.get("run_status") == "complete":
        existing_check = _complete_manifest_compatibility(
            existing,
            run_dir=run_dir,
            identity=identity,
            artifact_paths=artifact_paths,
        )
        if existing_check:
            raise RuntimeError(
                f"Immutable run already exists with incompatible contents: "
                f"{run_dir}; reason={existing_check}"
            )
        _write_json_atomic(
            latest_path,
            _pointer_payload(
                pointer_type="latest_completed",
                run_id=run_id,
                run_dir=run_dir,
                kind=kind,
                identity=identity,
                extra=extra_manifest,
            ),
        )
        return ImmutableRunRecord(run_id, run_dir, manifest_path, latest_path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Immutable run already exists but is not complete: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_records = _artifact_records(artifact_paths)
    _write_json_atomic(
        manifest_path,
        _manifest_payload(
            run_id=run_id,
            kind=kind,
            run_status="writing",
            identity=identity,
            artifacts=artifact_records,
            extra_manifest=extra_manifest,
        ),
    )
    for path in artifact_paths:
        destination = run_dir / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    copied_records = _artifact_records(run_dir / path.name for path in artifact_paths)
    _write_json_atomic(
        manifest_path,
        _manifest_payload(
            run_id=run_id,
            kind=kind,
            run_status="complete",
            identity=identity,
            artifacts=copied_records,
            extra_manifest=extra_manifest,
        ),
    )
    _write_json_atomic(
        latest_path,
        _pointer_payload(
            pointer_type="latest_completed",
            run_id=run_id,
            run_dir=run_dir,
            kind=kind,
            identity=identity,
            extra=extra_manifest,
        ),
    )
    return ImmutableRunRecord(run_id, run_dir, manifest_path, latest_path)


def update_champion_pointer(
    *,
    output_dir: Path,
    run_id: str,
    kind: str,
    model_name: str | None = None,
    identity: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> Path:
    pointer_path = champion_pointer_path(output_dir)
    run_dir = immutable_run_dir(output_dir, run_id)
    if not is_complete_run_dir(run_dir):
        raise RuntimeError(f"Cannot champion incomplete immutable run: {run_dir}")
    _write_json_atomic(
        pointer_path,
        _pointer_payload(
            pointer_type="champion",
            run_id=run_id,
            run_dir=run_dir,
            kind=kind,
            identity=identity or {},
            extra={"model_name": model_name, "reason": reason},
        ),
    )
    return pointer_path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing immutable run artifact: {path}")
        records.append(
            {
                "name": path.name,
                "sha256": file_digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _complete_manifest_compatibility(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    identity: Mapping[str, Any],
    artifact_paths: Sequence[Path],
) -> str:
    if manifest.get("identity") != dict(identity):
        return "identity_mismatch"
    expected_by_name = {
        record["name"]: record for record in manifest.get("artifacts", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    if not expected_by_name:
        return "missing_artifact_manifest"
    for source_path in artifact_paths:
        record = expected_by_name.get(source_path.name)
        if not record:
            return f"missing_artifact_record:{source_path.name}"
        copied_path = run_dir / source_path.name
        if not copied_path.exists():
            return f"missing_artifact:{source_path.name}"
        if copied_path.stat().st_size != record.get("size_bytes"):
            return f"artifact_size_mismatch:{source_path.name}"
        if file_digest(copied_path) != record.get("sha256"):
            return f"artifact_hash_mismatch:{source_path.name}"
    return ""


def _manifest_payload(
    *,
    run_id: str,
    kind: str,
    run_status: str,
    identity: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    extra_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": kind,
        "workflow": kind,
        "run_status": run_status,
        "identity": dict(identity),
        "artifacts": list(artifacts),
        "updated_at": _utc_now(),
        "updated_at_utc": _utc_now(),
        "research_only": True,
        "trading_impact": "none",
    }
    if run_status == "writing":
        payload["created_at_utc"] = payload["updated_at_utc"]
    if run_status == "complete":
        payload["completed_at_utc"] = payload["updated_at_utc"]
    if extra_manifest:
        payload.update(dict(extra_manifest))
    return payload


def _pointer_payload(
    *,
    pointer_type: str,
    run_id: str,
    run_dir: Path,
    kind: str,
    identity: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "pointer_type": pointer_type,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "kind": kind,
        "workflow": kind,
        "identity": dict(identity),
        "updated_at": _utc_now(),
        "updated_at_utc": _utc_now(),
    }
    if extra:
        payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def _read_pointer(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
