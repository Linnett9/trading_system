from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CHECKPOINT_CONTRACT_VERSION = "stock_selector_checkpoint_v1"
OPERATING_MODES = (
    "daily_cold_refit_strict",
    "daily_checkpoint_update",
    "daily_score_periodic_refit",
)


@dataclass(frozen=True)
class LoadedSelectorCheckpoint:
    path: Path
    manifest: dict[str, Any]
    model_state: Any
    preprocessing_state: Any
    optimizer_state: Any
    scheduler_state: Any
    rng_state: Any


def write_selector_checkpoint(
    root: Path, *, model_id: str, model_family: str, model_state_date: str,
    parent_checkpoint_id: str | None, last_training_decision_timestamp: str,
    last_included_label_availability_timestamp: str, frozen_dataset_id: str,
    feature_schema_hash: str, target_schema_hash: str, model_config_hash: str,
    git_commit: str, preprocessing_state_identity: str, random_seed: int,
    training_row_ids: Sequence[str], model_state: Any, preprocessing_state: Any,
    optimizer_state: Any = None, scheduler_state: Any = None, rng_state: Any = None,
    completion_status: str = "complete", operating_mode: str = "daily_cold_refit_strict",
) -> Path:
    if operating_mode not in OPERATING_MODES:
        raise ValueError(f"Unknown selector checkpoint operating mode: {operating_mode}")
    if preprocessing_state is None:
        raise ValueError("Selector checkpoints require preprocessing state")
    row_checksum = _hash_lines(sorted(training_row_ids))
    identity = {
        "contract_version": CHECKPOINT_CONTRACT_VERSION, "model_id": model_id,
        "model_family": model_family, "model_state_date": model_state_date,
        "parent_checkpoint_id": parent_checkpoint_id,
        "last_training_decision_timestamp": last_training_decision_timestamp,
        "last_included_label_availability_timestamp": last_included_label_availability_timestamp,
        "frozen_dataset_id": frozen_dataset_id, "feature_schema_hash": feature_schema_hash,
        "target_schema_hash": target_schema_hash, "model_config_hash": model_config_hash,
        "git_commit": git_commit, "preprocessing_state_identity": preprocessing_state_identity,
        "random_seed": random_seed, "training_row_count": len(training_row_ids),
        "training_row_id_checksum": row_checksum, "operating_mode": operating_mode,
    }
    checkpoint_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = root / model_id / model_state_date / checkpoint_id
    path.mkdir(parents=True, exist_ok=False)
    state_path = path / "state.pkl"
    state_path.write_bytes(pickle.dumps({
        "model": model_state, "preprocessing": preprocessing_state,
        "optimizer": optimizer_state, "scheduler": scheduler_state, "rng": rng_state,
    }))
    manifest = {
        **identity, "checkpoint_id": checkpoint_id,
        "checkpoint_checksum": _sha256(state_path), "completion_status": completion_status,
        "optimizer_state_present": optimizer_state is not None,
        "scheduler_state_present": scheduler_state is not None,
        "rng_state_present": rng_state is not None,
    }
    (path / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def load_selector_checkpoint(
    path: Path, *, decision_timestamp: str, frozen_dataset_id: str,
    feature_schema_hash: str, target_schema_hash: str, model_config_hash: str,
    require_optimizer_state: bool = False, require_scheduler_state: bool = False,
) -> LoadedSelectorCheckpoint:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Selector checkpoint manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != CHECKPOINT_CONTRACT_VERSION:
        raise RuntimeError("Selector checkpoint contract is incompatible")
    if manifest.get("completion_status") != "complete":
        raise RuntimeError("Selector checkpoint is incomplete")
    expected = {
        "frozen_dataset_id": frozen_dataset_id, "feature_schema_hash": feature_schema_hash,
        "target_schema_hash": target_schema_hash, "model_config_hash": model_config_hash,
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise RuntimeError(f"Selector checkpoint identity mismatch: {mismatches}")
    if str(manifest["model_state_date"]) > str(decision_timestamp):
        raise RuntimeError("A future-dated selector checkpoint cannot be loaded")
    state_path = path / "state.pkl"
    if _sha256(state_path) != manifest.get("checkpoint_checksum"):
        raise RuntimeError("Selector checkpoint checksum mismatch")
    state = pickle.loads(state_path.read_bytes())
    if state.get("preprocessing") is None:
        raise RuntimeError("Selector checkpoint preprocessing state is absent")
    if require_optimizer_state and state.get("optimizer") is None:
        raise RuntimeError("Selector checkpoint optimizer state is absent")
    if require_scheduler_state and state.get("scheduler") is None:
        raise RuntimeError("Selector checkpoint scheduler state is absent")
    return LoadedSelectorCheckpoint(
        path, manifest, state["model"], state["preprocessing"],
        state.get("optimizer"), state.get("scheduler"), state.get("rng"),
    )


def newly_matured_rows(
    rows: Sequence[Mapping[str, Any]], *, previous_label_timestamp: str,
    current_decision_timestamp: str,
) -> list[Mapping[str, Any]]:
    if previous_label_timestamp > current_decision_timestamp:
        raise ValueError("Previous checkpoint label timestamp is in the future")
    return [
        row for row in rows
        if previous_label_timestamp < str(row["label_available_timestamp"]) <= current_decision_timestamp
        and str(row["decision_timestamp"]) < current_decision_timestamp
    ]


def _hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest.upper()
