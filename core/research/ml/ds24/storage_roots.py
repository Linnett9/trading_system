from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping


DS24_CANONICAL_5M_SOURCE = "DS24_CANONICAL_5M_SOURCE"
DS24_RESEARCH_OUTPUT_ROOT = "DS24_RESEARCH_OUTPUT_ROOT"
DS24_P6_RUN_ID = "ds24_p6_preholdout_20260807T211438Z"
DS24_P6_FEATURE_VERSION = "ds24_preholdout_five_minute_features_v1"
DS24_P6_AUTHORITY_ID = "ds24_preholdout_five_minute_feature_authority"
DS24_P6_FEATURE_CONTRACT_IDENTITY = "565f9af428664ba204168302c8670d78bf8ae93013559f20783510abf2a76b80"
DS24_P6_STORAGE_LAYOUT_IDENTITY = "62f81bb0d836644a262744abc68004f05490f124209c7053af374574646a0929"
DS24_P5_CANONICAL_SOURCE_PARTITION_IDENTITY = "e451ac59c80882b8f28e57b7c704528512d39abd79bb93b8ec5c8501e476811a"
DS24_P5_CANONICAL_SOURCE_SCHEMA_IDENTITY = "bb0d5dac766df390f1b903528b8ab5f5521a9f6243dce252ce2757c7d7d088b7"

DS24_STORAGE_ROOT_UNBOUND = "DS24_STORAGE_ROOT_UNBOUND"
DS24_EXTERNAL_VOLUME_UNAVAILABLE = "DS24_EXTERNAL_VOLUME_UNAVAILABLE"
DS24_EXTERNAL_VOLUME_IDENTITY_MISMATCH = "DS24_EXTERNAL_VOLUME_IDENTITY_MISMATCH"
DS24_STORAGE_PATH_ESCAPE = "DS24_STORAGE_PATH_ESCAPE"
DS24_STORAGE_BINDING_INVALID = "DS24_STORAGE_BINDING_INVALID"
DS24_P6_OUTPUT_ROOT_NOT_ADMITTED = "DS24_P6_OUTPUT_ROOT_NOT_ADMITTED"


class BindingState(str, Enum):
    UNBOUND = "UNBOUND"
    CANDIDATE = "CANDIDATE"
    ADMITTED = "ADMITTED"
    UNAVAILABLE = "UNAVAILABLE"
    REVOKED = "REVOKED"


@dataclass(frozen=True)
class VolumeIdentity:
    volume_label: str = ""
    volume_serial_or_stable_id: str = ""
    filesystem: str = ""
    drive_hint: str = ""
    physical_root_observed: str = ""


@dataclass(frozen=True)
class StorageBinding:
    logical_root_id: str
    physical_root: Path | None
    binding_state: BindingState
    binding_source: str
    expected_device_class: str = ""
    volume_label: str = ""
    volume_serial_or_stable_id: str = ""
    filesystem: str = ""
    admitted_at: str = ""
    authority_ticket: str = ""


@dataclass(frozen=True)
class ResolvedStoragePath:
    logical_root_id: str
    resolved_physical_root: Path
    resolved_path: Path
    binding_source: str
    volume_expectation: dict[str, str]
    availability_state: str


class DS24StorageRootError(RuntimeError):
    def __init__(self, code: str, details: Mapping[str, object] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = dict(details or {})


VolumeProbe = Callable[[Path], VolumeIdentity]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_bindings(repo_root: Path | None = None) -> dict[str, StorageBinding]:
    root = repo_root or repository_root()
    return {
        DS24_CANONICAL_5M_SOURCE: StorageBinding(
            logical_root_id=DS24_CANONICAL_5M_SOURCE,
            physical_root=root / "data/processed/alpaca/symbol_bars/sip/5m",
            binding_state=BindingState.ADMITTED,
            binding_source="safe_built_in_local_binding",
            expected_device_class="LOCAL",
            authority_ticket="DS24_R4_KEEP_CANONICAL_SOURCE_LOCAL",
        ),
        DS24_RESEARCH_OUTPUT_ROOT: StorageBinding(
            logical_root_id=DS24_RESEARCH_OUTPUT_ROOT,
            physical_root=None,
            binding_state=BindingState.UNBOUND,
            binding_source="built_in_fail_closed_unbound",
            expected_device_class="ADMITTED_EXTERNAL_AUTHORITY_VOLUME",
            authority_ticket="DS24_R4_EXTERNAL_DEVICE_REQUIRED_BEFORE_PLAN_FINALISATION",
        ),
    }


def env_research_output_binding(env: Mapping[str, str] | None = None) -> StorageBinding | None:
    values = env or os.environ
    root = values.get("DS24_RESEARCH_OUTPUT_ROOT")
    if not root:
        return None
    return StorageBinding(
        logical_root_id=DS24_RESEARCH_OUTPUT_ROOT,
        physical_root=Path(root),
        binding_state=BindingState(values.get("DS24_RESEARCH_OUTPUT_BINDING_STATE", BindingState.CANDIDATE.value)),
        binding_source="approved_environment_override",
        expected_device_class=values.get("DS24_RESEARCH_OUTPUT_EXPECTED_DEVICE_CLASS", "ADMITTED_EXTERNAL_AUTHORITY_VOLUME"),
        volume_label=values.get("DS24_RESEARCH_OUTPUT_VOLUME_LABEL", ""),
        volume_serial_or_stable_id=values.get("DS24_RESEARCH_OUTPUT_VOLUME_ID", ""),
        filesystem=values.get("DS24_RESEARCH_OUTPUT_FILESYSTEM", ""),
        admitted_at=values.get("DS24_RESEARCH_OUTPUT_ADMITTED_AT", ""),
        authority_ticket=values.get("DS24_RESEARCH_OUTPUT_AUTHORITY_TICKET", ""),
    )


def merge_bindings(
    explicit_bindings: Mapping[str, StorageBinding] | None = None,
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, StorageBinding]:
    bindings = default_bindings(repo_root)
    env_binding = env_research_output_binding(env)
    if env_binding is not None:
        bindings[DS24_RESEARCH_OUTPUT_ROOT] = env_binding
    if explicit_bindings:
        bindings.update(explicit_bindings)
    return bindings


def local_volume_identity(root: Path) -> VolumeIdentity:
    resolved = root.resolve()
    return VolumeIdentity(
        drive_hint=resolved.drive.upper() if resolved.drive else resolved.anchor,
        physical_root_observed=str(resolved),
    )


def _normalise_relative_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.drive:
        raise DS24StorageRootError(DS24_STORAGE_PATH_ESCAPE, {"relative_path": str(relative_path)})
    parts = candidate.parts
    if any(part in ("..", "") for part in parts):
        raise DS24StorageRootError(DS24_STORAGE_PATH_ESCAPE, {"relative_path": str(relative_path)})
    return candidate


def _assert_under_root(root: Path, path: Path, logical_root_id: str) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise DS24StorageRootError(
            DS24_STORAGE_PATH_ESCAPE,
            {"logical_root_id": logical_root_id, "root": str(root), "resolved_path": str(path)},
        ) from exc


def _expected_volume(binding: StorageBinding) -> dict[str, str]:
    return {
        "volume_label": binding.volume_label,
        "volume_serial_or_stable_id": binding.volume_serial_or_stable_id,
        "filesystem": binding.filesystem,
        "expected_device_class": binding.expected_device_class,
    }


def _validate_external_binding(binding: StorageBinding, volume_probe: VolumeProbe) -> None:
    if binding.physical_root is None:
        raise DS24StorageRootError(DS24_STORAGE_ROOT_UNBOUND, {"logical_root_id": binding.logical_root_id})
    if binding.binding_state is not BindingState.ADMITTED:
        raise DS24StorageRootError(
            DS24_P6_OUTPUT_ROOT_NOT_ADMITTED,
            {"logical_root_id": binding.logical_root_id, "binding_state": binding.binding_state.value},
        )
    if not binding.physical_root.exists() or not binding.physical_root.is_dir():
        raise DS24StorageRootError(
            DS24_EXTERNAL_VOLUME_UNAVAILABLE,
            {"logical_root_id": binding.logical_root_id, "physical_root": str(binding.physical_root)},
        )
    actual = volume_probe(binding.physical_root)
    mismatches: dict[str, tuple[str, str]] = {}
    if binding.volume_serial_or_stable_id and binding.volume_serial_or_stable_id != actual.volume_serial_or_stable_id:
        mismatches["volume_serial_or_stable_id"] = (binding.volume_serial_or_stable_id, actual.volume_serial_or_stable_id)
    if binding.filesystem and binding.filesystem.lower() != actual.filesystem.lower():
        mismatches["filesystem"] = (binding.filesystem, actual.filesystem)
    if binding.volume_label and binding.volume_label != actual.volume_label:
        mismatches["volume_label"] = (binding.volume_label, actual.volume_label)
    if mismatches:
        raise DS24StorageRootError(
            DS24_EXTERNAL_VOLUME_IDENTITY_MISMATCH,
            {"logical_root_id": binding.logical_root_id, "mismatches": mismatches, "actual": asdict(actual)},
        )


def resolve_ds24_storage_path(
    logical_root_id: str,
    relative_path: str | Path = ".",
    *,
    require_available: bool = True,
    bindings: Mapping[str, StorageBinding] | None = None,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    volume_probe: VolumeProbe = local_volume_identity,
) -> ResolvedStoragePath:
    all_bindings = merge_bindings(bindings, repo_root=repo_root, env=env)
    binding = all_bindings.get(logical_root_id)
    if binding is None:
        raise DS24StorageRootError(DS24_STORAGE_BINDING_INVALID, {"logical_root_id": logical_root_id})
    if binding.physical_root is None:
        raise DS24StorageRootError(DS24_STORAGE_ROOT_UNBOUND, {"logical_root_id": logical_root_id})
    if logical_root_id == DS24_RESEARCH_OUTPUT_ROOT:
        _validate_external_binding(binding, volume_probe)
    elif require_available and (not binding.physical_root.exists() or not binding.physical_root.is_dir()):
        raise DS24StorageRootError(
            DS24_EXTERNAL_VOLUME_UNAVAILABLE,
            {"logical_root_id": logical_root_id, "physical_root": str(binding.physical_root)},
        )

    relative = _normalise_relative_path(relative_path)
    root = binding.physical_root.resolve(strict=False)
    resolved_path = (root / relative).resolve(strict=False)
    _assert_under_root(root, resolved_path, logical_root_id)
    availability_state = "AVAILABLE" if root.exists() else "BOUND_BUT_NOT_PRESENT"
    return ResolvedStoragePath(
        logical_root_id=logical_root_id,
        resolved_physical_root=root,
        resolved_path=resolved_path,
        binding_source=binding.binding_source,
        volume_expectation=_expected_volume(binding),
        availability_state=availability_state,
    )


def ds24_p6_relative_run_path(run_id: str = DS24_P6_RUN_ID) -> Path:
    if run_id != DS24_P6_RUN_ID:
        raise DS24StorageRootError(
            DS24_STORAGE_BINDING_INVALID,
            {"expected_run_id": DS24_P6_RUN_ID, "actual_run_id": run_id},
        )
    return Path("ml_features") / "five_minute" / f"version={DS24_P6_FEATURE_VERSION}" / f"run={run_id}"


def resolve_ds24_p6_run_root(
    *,
    run_id: str = DS24_P6_RUN_ID,
    require_available: bool = True,
    bindings: Mapping[str, StorageBinding] | None = None,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    volume_probe: VolumeProbe = local_volume_identity,
) -> ResolvedStoragePath:
    return resolve_ds24_storage_path(
        DS24_RESEARCH_OUTPUT_ROOT,
        ds24_p6_relative_run_path(run_id),
        require_available=require_available,
        bindings=bindings,
        repo_root=repo_root,
        env=env,
        volume_probe=volume_probe,
    )

