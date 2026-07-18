from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.research.compute.run_registry import (
    CorruptRunRegistry,
    StaleRegistryRevision,
    read_run_registry,
    update_run_registry,
)


def record(identity: str):
    return {
        "run_identity": identity, "run_id": f"run-{identity}",
        "pipeline": "test", "stage": "stage", "status": "RUNNING",
        "source_git_commit": "commit", "machine_profile_identity": "machine",
        "run_root_relative_path": f"test/stage/run-{identity}",
        "latest_status_revision": 1,
    }


def test_registry_initialise_revision_stale_and_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "run_registry.json"
    assert read_run_registry(path)["revision"] == 0
    first = update_run_registry(record("a"), path=path, expected_revision=0)
    assert first["revision"] == 1
    with pytest.raises(StaleRegistryRevision):
        update_run_registry(record("b"), path=path, expected_revision=0)
    payload = json.loads(path.read_text())
    payload["logical_checksum"] = "bad"
    path.write_text(json.dumps(payload))
    with pytest.raises(CorruptRunRegistry):
        update_run_registry(record("b"), path=path)


def test_independent_concurrent_updates_preserve_both_runs(tmp_path: Path) -> None:
    path = tmp_path / "run_registry.json"

    def update(identity: str):
        return update_run_registry(record(identity), path=path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(update, ("a", "b")))
    registry = read_run_registry(path)
    assert registry["revision"] == 2
    assert {row["run_identity"] for row in registry["runs"]} == {"a", "b"}


def test_run_identity_collision_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "run_registry.json"
    update_run_registry(record("a"), path=path)
    collision = record("b")
    collision["run_id"] = "run-a"
    with pytest.raises(ValueError, match="collision"):
        update_run_registry(collision, path=path)
