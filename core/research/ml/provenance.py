from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.research.ml.registries.io import canonical_hash


DEPENDENCY_DISTRIBUTIONS = {
    "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow",
    "scikit-learn": "scikit-learn", "torch": "torch",
    "scipy": "scipy",
}


def dependency_provenance(requirements: Iterable[str]) -> dict[str, str]:
    requested = {str(value).strip().lower() for value in requirements}
    names = {"python"}
    for requirement in requested:
        if requirement in {"scikit-learn", "sklearn"}:
            names.update({"numpy", "scikit-learn"})
        elif requirement == "torch":
            names.update({"numpy", "torch"})
        elif requirement in DEPENDENCY_DISTRIBUTIONS:
            names.add(requirement)
    result = {"python": platform.python_version()}
    for name in sorted(names - {"python"}):
        distribution = DEPENDENCY_DISTRIBUTIONS[name]
        try:
            result[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def dependency_identity(requirements: Iterable[str]) -> dict[str, Any]:
    versions = dependency_provenance(requirements)
    return {"contract_version": "relevant_dependencies_v1", "versions": versions, "hash": canonical_hash(versions)}


def source_provenance(repo_root: Path = Path("."), *, changed_path_limit: int = 100) -> dict[str, Any]:
    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    tracked = sorted(set(filter(None, _git(root, "diff", "--name-only").splitlines())) | set(filter(None, _git(root, "diff", "--cached", "--name-only").splitlines())))
    untracked = sorted(filter(None, _git(root, "ls-files", "--others", "--exclude-standard").splitlines()))
    bounded = tracked[:changed_path_limit]
    return {
        "contract_version": "source_worktree_provenance_v1", "git_commit": commit or None,
        "git_branch": branch or None, "dirty_worktree": bool(tracked or untracked),
        "tracked_changed_path_count": len(tracked), "tracked_changed_paths": bounded,
        "tracked_changed_paths_truncated": len(tracked) > changed_path_limit,
        "tracked_changed_paths_hash": canonical_hash(tracked),
        "untracked_file_count": len(untracked), "untracked_files_present": bool(untracked),
        "untracked_paths_hash": canonical_hash(untracked),
    }


def normalize_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip through strict canonical JSON to reject unsupported metadata."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(encoded)


def file_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "size_bytes": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "exists": True, "sha256": digest.hexdigest().upper(), "size_bytes": path.stat().st_size}


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
