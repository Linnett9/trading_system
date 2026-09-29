from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (  # noqa: E402
    CLEAN_SOURCE_PATHS,
    file_sha256,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_features import REPAIRED_FEATURES  # noqa: E402


SCHEMA_VERSION = "ds24_clean_v2_mac_metadata.v1"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "docs" / "audits" / "ds24_clean_v2_mac_reconstruction_20260929"
)
CAUSALITY_CERTIFICATE_RELATIVE_PATH = Path(
    "research_runs/ds24_clean_v2/DS24_CLEAN_V2_TOURNAMENT_R1_20260926/"
    "causality_certificate.json"
)
BUILDER_ROOTS = (
    "scripts/local/ds24_clean_v2_build_sidecar.py",
    "scripts/local/ds24_clean_v2_build_target_delta.py",
    "scripts/local/ds24_clean_v2_certify.py",
    "core/research/ml/ds24/clean_v2_data.py",
)
EXPECTED_BUILDER_CLOSURE = (
    "core/research/ml/ds24/clean_v2_certification.py",
    "core/research/ml/ds24/clean_v2_contracts.py",
    "core/research/ml/ds24/clean_v2_data.py",
    "core/research/ml/ds24/clean_v2_features.py",
    "core/research/ml/ds24/master_5m_validation_stats.py",
    "core/research/ml/five_minute_target_dataset.py",
    "core/research/ml/target_authority.py",
    "infrastructure/data/calendar_authority.py",
    "infrastructure/data/market_sessions.py",
    "scripts/local/ds24_clean_v2_build_sidecar.py",
    "scripts/local/ds24_clean_v2_build_target_delta.py",
    "scripts/local/ds24_clean_v2_certify.py",
)
EXPECTED_REPAIRED_FIELDS = frozenset(
    {
        "overnight_gap",
        "previous_session_return",
        "two_session_return",
        "opening_range_position",
        "session_return_30m",
        "opening_return_30m",
        "minutes_since_open",
        "minutes_until_close",
        "session_progress",
        "opening_period_flag",
        "early_close_session_flag",
    }
)
EXPECTED_COUNTS = {
    "feature_base": {
        "partition_count": 5654,
        "row_count": 114497377,
        "total_bytes": 36970830328,
    },
    "feature_sidecar": {
        "partition_count": 5654,
        "row_count": 114497377,
        "total_bytes": 8427792776,
    },
    "target_base": {
        "partition_count": 5654,
        "row_count": 100180654,
        "total_bytes": 18820658712,
    },
    "target_delta": {
        "partition_count": 514,
        "row_count": 2251012,
        "trainable_row_count": 1699821,
        "total_bytes": 432608762,
        "empty_symbols": ["JHG"],
    },
}
FROZEN_AUTHORITIES = {
    "feature_authority_sha256": (
        "ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d"
    ),
    "target_authority_sha256": (
        "41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1"
    ),
    "static_bundle_sha256": (
        "4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc"
    ),
    "refit_policy": "REFIT_EVERY_5_TRADING_SESSIONS_V1",
}


class MacMetadataExportError(RuntimeError):
    """Raised when a source or metadata authority cannot be exported safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MacMetadataExportError(f"Unable to read JSON authority: {path}") from exc
    if not isinstance(payload, dict):
        raise MacMetadataExportError(f"JSON authority must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_metadata(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise MacMetadataExportError(f"Required metadata file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)


def _git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository_root}", *args],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise MacMetadataExportError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _git_bytes(repository_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository_root}", *args],
        cwd=repository_root,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MacMetadataExportError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _clean_source_hash_for(repository_root: Path) -> str:
    return stable_hash(
        {
            relative: file_sha256(repository_root / relative)
            for relative in CLEAN_SOURCE_PATHS
        }
    )


def _clean_source_hash_from_commit(commit: str) -> str:
    return stable_hash(
        {
            relative: hashlib.sha256(
                _git_bytes(ROOT, "show", f"{commit}:{relative}")
            ).hexdigest()
            for relative in CLEAN_SOURCE_PATHS
        }
    )


def _authority_bundle_for(repository_root: Path) -> dict[str, Any]:
    names = (
        "predictor_manifest.json",
        "feature_authority.json",
        "target_contract.json",
        "eligibility_contract.json",
        "model_registry.json",
        "tournament_contract.json",
        "cross_host_ownership.json",
        "prior_evidence_manifest.json",
    )
    files: dict[str, dict[str, str]] = {}
    config_root = repository_root / "config" / "ds24_clean_v2"
    for name in names:
        path = config_root / name
        payload = _read_json(path)
        logical = stable_hash(
            {key: value for key, value in payload.items() if key != "contract_sha256"}
        )
        declared = payload.get("contract_sha256")
        if declared is not None and declared != logical:
            raise MacMetadataExportError(
                f"{name} declared hash mismatch: declared={declared}, actual={logical}"
            )
        files[name] = {
            "file_sha256": file_sha256(path),
            "logical_sha256": logical,
        }
    result: dict[str, Any] = {
        "authority_id": "DS24_CLEAN_V2_STATIC_AUTHORITY_BUNDLE_V1",
        "files": files,
    }
    result["bundle_sha256"] = stable_hash(result)
    return result


def _module_path(repository_root: Path, module: str) -> Path | None:
    module_path = Path(*module.split("."))
    candidates = (module_path.with_suffix(".py"), module_path / "__init__.py")
    for relative in candidates:
        if (repository_root / relative).is_file():
            return relative
    return None


def _import_closure(
    repository_root: Path,
    roots: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]]:
    pending = list(roots)
    visited: set[str] = set()
    edges: dict[str, tuple[str, ...]] = {}
    external: set[str] = set()
    while pending:
        relative_value = pending.pop(0)
        if relative_value in visited:
            continue
        visited.add(relative_value)
        path = repository_root / relative_value
        if not path.is_file():
            raise MacMetadataExportError(f"Import-closure source is missing: {path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_value)
        except (OSError, SyntaxError) as exc:
            raise MacMetadataExportError(f"Cannot parse import-closure source: {path}") from exc
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            modules: Iterable[str]
            if isinstance(node, ast.Import):
                modules = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            else:
                continue
            for module in modules:
                local = _module_path(repository_root, module)
                if local is None:
                    if module != "__future__":
                        external.add(module.split(".", 1)[0])
                    continue
                local_value = local.as_posix()
                dependencies.add(local_value)
                if local_value not in visited:
                    pending.append(local_value)
        edges[relative_value] = tuple(sorted(dependencies))
    return (
        tuple(sorted(visited)),
        {key: edges[key] for key in sorted(edges)},
        tuple(sorted(external)),
    )


def _self_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in payload.items() if key != "inventory_sha256"}
    )


def _inventory(
    *,
    inventory_id: str,
    root_relative_path: str,
    entries: list[dict[str, Any]],
    source_manifest_file_sha256: str,
    source_manifest_logical_sha256: str,
    trainable_row_count: int | None = None,
    empty_symbols: Sequence[str] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "ds24_clean_v2_normalized_partition_inventory.v1",
        "inventory_id": inventory_id,
        "root_relative_path": root_relative_path,
        "partition_count": len(entries),
        "row_count": sum(int(row["row_count"]) for row in entries),
        "total_bytes": sum(int(row["size_bytes"]) for row in entries),
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_manifest_logical_sha256": source_manifest_logical_sha256,
        "entries": entries,
    }
    if trainable_row_count is not None:
        payload["trainable_row_count"] = trainable_row_count
    if empty_symbols:
        payload["empty_symbols"] = list(empty_symbols)
    payload["inventory_sha256"] = _self_hash(payload)
    return payload


def _normalized_inventories(source_repository: Path) -> dict[str, dict[str, Any]]:
    feature_contract = load_contract("feature_authority.json")
    target_contract = load_contract("target_contract.json")
    feature_root = source_repository / feature_contract["sidecar"]["path"]
    target_root = source_repository / target_contract["physical_authority"]["delta_path"]
    feature_manifest_path = feature_root / "authority_manifest.json"
    target_manifest_path = target_root / "authority_manifest.json"
    feature_manifest = _read_json(feature_manifest_path)
    target_manifest = _read_json(target_manifest_path)
    if feature_manifest.get("complete") is not True:
        raise MacMetadataExportError("Feature sidecar authority manifest is not complete")
    if target_manifest.get("complete") is not True:
        raise MacMetadataExportError("Target delta authority manifest is not complete")

    feature_base_entries = [
        {
            "relative_path": row["base_relative_path"],
            "sha256": str(row["base_sha256"]).lower(),
            "size_bytes": int(row["base_bytes"]),
            "row_count": int(row["row_count"]),
        }
        for row in feature_manifest["partitions"]
    ]
    feature_sidecar_entries = [
        {
            "relative_path": row["relative_path"],
            "sha256": str(row["sha256"]).lower(),
            "size_bytes": int(row["bytes"]),
            "row_count": int(row["row_count"]),
            "base_relative_path": row["base_relative_path"],
        }
        for row in feature_manifest["partitions"]
    ]

    target_base_root_value = target_contract["physical_authority"]["base_path"]
    target_base_root = source_repository / target_base_root_value
    target_base_entries: list[dict[str, Any]] = []
    for row in target_manifest["base_snapshot"]["partitions"]:
        relative = Path(row["base_relative_path"])
        path = target_base_root / relative
        if not path.is_file():
            raise MacMetadataExportError(f"Target base partition is missing: {path}")
        target_base_entries.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": str(row["base_sha256"]).lower(),
                "size_bytes": int(row["base_bytes"]),
                "row_count": int(pq.ParquetFile(path).metadata.num_rows),
            }
        )
    target_delta_entries = [
        {
            "relative_path": row["relative_path"],
            "sha256": str(row["sha256"]).lower(),
            "size_bytes": int(row["bytes"]),
            "row_count": int(row["rows"]),
            "trainable_row_count": int(row["trainable_rows"]),
            "symbol": row["symbol"],
            "empty": bool(row.get("empty_source_window", False)),
        }
        for row in target_manifest["partitions"]
    ]

    feature_manifest_sha256 = file_sha256(feature_manifest_path)
    target_manifest_sha256 = file_sha256(target_manifest_path)
    return {
        "feature_base": _inventory(
            inventory_id="DS24_CLEAN_V2_FEATURE_BASE",
            root_relative_path=feature_contract["base_authority"]["path"],
            entries=feature_base_entries,
            source_manifest_file_sha256=feature_manifest_sha256,
            source_manifest_logical_sha256=feature_manifest["logical_sha256"],
        ),
        "feature_sidecar": _inventory(
            inventory_id="DS24_CLEAN_V2_FEATURE_SIDECAR",
            root_relative_path=feature_contract["sidecar"]["path"],
            entries=feature_sidecar_entries,
            source_manifest_file_sha256=feature_manifest_sha256,
            source_manifest_logical_sha256=feature_manifest["logical_sha256"],
        ),
        "target_base": _inventory(
            inventory_id="DS24_CLEAN_V2_TARGET_BASE",
            root_relative_path=target_base_root_value,
            entries=target_base_entries,
            source_manifest_file_sha256=target_manifest_sha256,
            source_manifest_logical_sha256=target_manifest["logical_sha256"],
        ),
        "target_delta": _inventory(
            inventory_id="DS24_CLEAN_V2_TARGET_DELTA",
            root_relative_path=target_contract["physical_authority"]["delta_path"],
            entries=target_delta_entries,
            source_manifest_file_sha256=target_manifest_sha256,
            source_manifest_logical_sha256=target_manifest["logical_sha256"],
            trainable_row_count=sum(
                int(row["trainable_row_count"]) for row in target_delta_entries
            ),
            empty_symbols=target_manifest["empty_symbols"],
        ),
    }


def _validate_authorities(
    inventories: Mapping[str, Mapping[str, Any]],
    certificate: Mapping[str, Any],
    *,
    source_repository: Path,
) -> None:
    for name, expected in EXPECTED_COUNTS.items():
        actual = inventories[name]
        for key, value in expected.items():
            if actual.get(key) != value:
                raise MacMetadataExportError(
                    f"{name} {key} mismatch: expected={value!r}, actual={actual.get(key)!r}"
                )
    if set(REPAIRED_FEATURES) != EXPECTED_REPAIRED_FIELDS or len(REPAIRED_FEATURES) != 11:
        raise MacMetadataExportError(
            f"CLEAN V2 repaired-field contract changed: {REPAIRED_FEATURES!r}"
        )
    bundle = _authority_bundle_for(source_repository)
    if bundle["bundle_sha256"] != FROZEN_AUTHORITIES["static_bundle_sha256"]:
        raise MacMetadataExportError("Static authority bundle hash changed")
    if (
        inventories["feature_sidecar"]["source_manifest_logical_sha256"]
        != FROZEN_AUTHORITIES["feature_authority_sha256"]
    ):
        raise MacMetadataExportError("Frozen feature authority hash changed")
    if (
        inventories["target_delta"]["source_manifest_logical_sha256"]
        != FROZEN_AUTHORITIES["target_authority_sha256"]
    ):
        raise MacMetadataExportError("Frozen target authority hash changed")
    if certificate.get("passed") is not True:
        raise MacMetadataExportError("Causality certificate is not passing")
    if certificate.get("certification_scope") != "FULL_MATERIALIZED_AUTHORITY":
        raise MacMetadataExportError("Causality certificate is not full-authority scope")


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("numpy", "pandas", "pyarrow", "psutil", "exchange-calendars"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def _source_information(
    *,
    source_repository: Path,
    source_snapshot_commit: str,
    import_closure: Sequence[str],
    import_edges: Mapping[str, Sequence[str]],
    external_imports: Sequence[str],
) -> dict[str, Any]:
    status = _git(source_repository, "status", "--porcelain=v1")
    status_lines = [line for line in status.splitlines() if line.strip()]
    expanded_status = _git(
        source_repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    expanded_status_lines = [
        line for line in expanded_status.splitlines() if line.strip()
    ]
    working_source_hash = _clean_source_hash_for(source_repository)
    committed_source_hash = _clean_source_hash_from_commit(source_snapshot_commit)
    if working_source_hash != committed_source_hash:
        raise MacMetadataExportError(
            "Publication commit does not reproduce the current CLEAN source bytes: "
            f"working={working_source_hash}, committed={committed_source_hash}"
        )
    path_records = [
        {
            "relative_path": relative,
            "sha256": file_sha256(source_repository / relative),
        }
        for relative in CLEAN_SOURCE_PATHS
    ]
    closure_records = [
        {
            "relative_path": relative,
            "sha256": file_sha256(source_repository / relative),
        }
        for relative in import_closure
    ]
    for record in closure_records:
        committed_bytes = _git_bytes(
            ROOT,
            "show",
            f"{source_snapshot_commit}:{record['relative_path']}",
        )
        committed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
        if committed_sha256 != record["sha256"]:
            raise MacMetadataExportError(
                "Publication commit does not reproduce builder dependency: "
                f"{record['relative_path']}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_repository": {
            "git_head": _git(source_repository, "rev-parse", "HEAD"),
            "git_branch": _git(source_repository, "branch", "--show-current"),
            "dirty_worktree": bool(status_lines),
            "dirty_entry_count": len(status_lines),
            "dirty_entry_count_with_untracked_files_all": len(expanded_status_lines),
            "dirty_status_sha256": hashlib.sha256(
                (
                    "\n".join(expanded_status_lines)
                    + ("\n" if expanded_status_lines else "")
                ).encode("utf-8")
            ).hexdigest(),
        },
        "publication": {
            "branch": "clean-v2/mac-reconstruction-current",
            "source_snapshot_commit": source_snapshot_commit,
            "clean_source_hash": working_source_hash,
        },
        "clean_source_paths": path_records,
        "builder_roots": list(BUILDER_ROOTS),
        "builder_import_closure": closure_records,
        "builder_import_edges": {key: list(value) for key, value in import_edges.items()},
        "external_imports": list(external_imports),
        "package_versions_on_dell": _package_versions(),
        "target_builder_module": "core/research/ml/five_minute_target_dataset.py",
        "production_reader": "core/research/ml/ds24/clean_v2_data.py",
        "repaired_fields": list(REPAIRED_FEATURES),
        "frozen_authorities": FROZEN_AUTHORITIES,
    }


def _export_summary(
    inventories: Mapping[str, Mapping[str, Any]],
    source_information: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "DS24_CLEAN_V2_MAC_RECONSTRUCTION_SOURCE_AND_METADATA_READY",
        "bulk_parquet_included": False,
        "model_checkpoints_included": False,
        "runtime_state_included": False,
        "research_metrics_included": False,
        "publication": source_information["publication"],
        "frozen_authorities": FROZEN_AUTHORITIES,
        "inventories": {
            name: {
                key: inventory[key]
                for key in (
                    "inventory_sha256",
                    "partition_count",
                    "row_count",
                    "total_bytes",
                )
            }
            for name, inventory in inventories.items()
        },
        "target_delta_trainable_row_count": inventories["target_delta"][
            "trainable_row_count"
        ],
        "target_delta_empty_symbols": inventories["target_delta"]["empty_symbols"],
        "mac_reconstruction_order": [
            "reuse exact V1 partitions only when relative_path, size_bytes, and sha256 match",
            "locally reconstruct any missing V1 feature/target base partitions",
            "run scripts/local/ds24_clean_v2_build_sidecar.py",
            "run scripts/local/ds24_clean_v2_build_target_delta.py",
            "run scripts/local/ds24_clean_v2_certify.py --full-materialized-authority",
        ],
    }


def export_metadata(
    *,
    source_repository: Path,
    output_root: Path,
    source_snapshot_commit: str,
) -> dict[str, Any]:
    source_repository = source_repository.resolve()
    output_root = output_root.resolve()
    closure, edges, external = _import_closure(ROOT, BUILDER_ROOTS)
    if closure != EXPECTED_BUILDER_CLOSURE:
        raise MacMetadataExportError(
            "Builder import closure changed; review before publication: "
            f"expected={EXPECTED_BUILDER_CLOSURE!r}, actual={closure!r}"
        )
    certificate_path = source_repository / CAUSALITY_CERTIFICATE_RELATIVE_PATH
    certificate = _read_json(certificate_path)
    inventories = _normalized_inventories(source_repository)
    _validate_authorities(
        inventories,
        certificate,
        source_repository=source_repository,
    )
    source_information = _source_information(
        source_repository=source_repository,
        source_snapshot_commit=source_snapshot_commit,
        import_closure=closure,
        import_edges=edges,
        external_imports=external,
    )
    for name, inventory in inventories.items():
        _write_json(output_root / "inventories" / f"{name}.json", inventory)
    _write_json(output_root / "builder_source_information.json", source_information)
    _write_json(output_root / "export_summary.json", _export_summary(inventories, source_information))

    feature_contract = load_contract("feature_authority.json")
    target_contract = load_contract("target_contract.json")
    _copy_metadata(
        source_repository / feature_contract["sidecar"]["path"] / "authority_manifest.json",
        output_root / "authority_manifests" / "feature_sidecar_authority_manifest.json",
    )
    _copy_metadata(
        source_repository
        / target_contract["physical_authority"]["delta_path"]
        / "authority_manifest.json",
        output_root / "authority_manifests" / "target_delta_authority_manifest.json",
    )
    _copy_metadata(certificate_path, output_root / "causality_certificate.json")
    for contract_path in sorted(
        (source_repository / "config" / "ds24_clean_v2").glob("*.json")
    ):
        _copy_metadata(contract_path, output_root / "authority_contracts" / contract_path.name)

    file_inventory = []
    for path in sorted(candidate for candidate in output_root.rglob("*") if candidate.is_file()):
        if path.name == "metadata_file_inventory.json":
            continue
        file_inventory.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    inventory_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "file_count": len(file_inventory),
        "total_bytes": sum(int(row["size_bytes"]) for row in file_inventory),
        "files": file_inventory,
    }
    inventory_payload["inventory_sha256"] = stable_hash(inventory_payload)
    _write_json(output_root / "metadata_file_inventory.json", inventory_payload)
    return {
        "output_root": str(output_root),
        "file_count": len(file_inventory) + 1,
        "clean_source_hash": source_information["publication"]["clean_source_hash"],
        "inventories": {
            name: {
                "partition_count": value["partition_count"],
                "row_count": value["row_count"],
                "total_bytes": value["total_bytes"],
            }
            for name, value in inventories.items()
        },
    }


def build_archive(
    *,
    metadata_root: Path,
    archive_path: Path,
    publication_commit: str,
) -> dict[str, Any]:
    metadata_root = metadata_root.resolve()
    archive_path = archive_path.resolve()
    if not metadata_root.is_dir():
        raise MacMetadataExportError(f"Metadata root is missing: {metadata_root}")
    export_summary = _read_json(metadata_root / "export_summary.json")
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "publication_branch": "clean-v2/mac-reconstruction-current",
        "publication_commit": publication_commit,
        "clean_source_hash": export_summary["publication"]["clean_source_hash"],
        "git_fetch_command": (
            "git fetch origin clean-v2/mac-reconstruction-current"
        ),
        "git_checkout_command": f"git checkout --detach {publication_commit}",
    }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".partial")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(candidate for candidate in metadata_root.rglob("*") if candidate.is_file()):
            archive.write(path, path.relative_to(metadata_root).as_posix())
        archive.writestr(
            "publication_pointer.json",
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        )
    os.replace(temporary, archive_path)
    return {
        "archive_path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "sha256": file_sha256(archive_path),
        "publication": pointer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the source-only DS24 CLEAN V2 Mac reconstruction metadata package."
    )
    parser.add_argument("--source-repository", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-snapshot-commit")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--publication-commit")
    args = parser.parse_args()
    if args.archive is not None:
        if not args.publication_commit:
            parser.error("--publication-commit is required with --archive")
        result = build_archive(
            metadata_root=args.output_root,
            archive_path=args.archive,
            publication_commit=args.publication_commit,
        )
    else:
        if args.source_repository is None or not args.source_snapshot_commit:
            parser.error(
                "--source-repository and --source-snapshot-commit are required for export"
            )
        result = export_metadata(
            source_repository=args.source_repository,
            output_root=args.output_root,
            source_snapshot_commit=args.source_snapshot_commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
