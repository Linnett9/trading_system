from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from core.research.ml.ds24.clean_v2_contracts import file_sha256, stable_hash
from core.research.ml.ds24.clean_v2_resources import (
    GIB,
    LEGACY_CACHE_MINIMUM_POST_WRITE_FREE_BYTES,
    evaluate_disk_write_feasibility,
    process_identity,
    process_identity_matches,
    write_json_object_atomic,
)


PACKAGE_CACHE_SCHEMA = "DS24_CLEAN_V2_SHARED_PANEL_PACKAGE_CACHE_V1"
# This accepted global cap bounds reusable acceleration data independently of
# host admission policy. Host-specific post-write floors remain separate.
DEFAULT_PACKAGE_CACHE_MAX_BYTES = 7 * GIB // 4
MIN_POST_CACHE_FREE_DISK_BYTES = LEGACY_CACHE_MINIMUM_POST_WRITE_FREE_BYTES
SOURCE_NAMESPACE_PREFIX = "source="
SOURCE_NAMESPACE_HEX_LENGTH = 16


class PackageCacheError(RuntimeError):
    """Raised when a shared package cache cannot be trusted."""


@dataclass(frozen=True)
class PanelPackageKey:
    feature_authority_hash: str
    target_authority_hash: str
    static_authority_bundle_sha256: str
    predictor_order_hash: str
    assembly_source_hash: str
    session_dates: tuple[str, ...]
    maximum_assets: int | None

    def payload(self) -> dict[str, Any]:
        return {
            "schema": PACKAGE_CACHE_SCHEMA,
            "feature_authority_hash": self.feature_authority_hash,
            "target_authority_hash": self.target_authority_hash,
            "static_authority_bundle_sha256": self.static_authority_bundle_sha256,
            "predictor_order_hash": self.predictor_order_hash,
            "assembly_source_hash": self.assembly_source_hash,
            "session_dates": list(self.session_dates),
            "maximum_assets": self.maximum_assets,
        }

    @property
    def digest(self) -> str:
        return stable_hash(self.payload())


@dataclass(frozen=True)
class CachedPanelResult:
    frame: pd.DataFrame
    disposition: str
    cache_key: str
    artifact_bytes: int
    read_wall_seconds: float
    write_wall_seconds: float


@dataclass(frozen=True)
class CacheRetentionResult:
    """Outcome of one identity-safe retention pass."""

    maximum_bytes: int
    before_bytes: int
    after_bytes: int
    evicted_entries: tuple[str, ...]
    evicted_bytes: int
    stale_locks_reclaimed: tuple[str, ...]
    stale_leases_reclaimed: tuple[str, ...]
    skipped_live_entries: tuple[str, ...]
    inactive_entries_remaining: tuple[str, ...]

    @property
    def retention_satisfied(self) -> bool:
        return (
            self.after_bytes <= self.maximum_bytes
            and not self.inactive_entries_remaining
        )

    def payload(self) -> dict[str, Any]:
        return {
            "maximum_bytes": self.maximum_bytes,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "evicted_entries": list(self.evicted_entries),
            "evicted_bytes": self.evicted_bytes,
            "stale_locks_reclaimed": list(self.stale_locks_reclaimed),
            "stale_leases_reclaimed": list(self.stale_leases_reclaimed),
            "skipped_live_entries": list(self.skipped_live_entries),
            "inactive_entries_remaining": list(self.inactive_entries_remaining),
            "retention_satisfied": self.retention_satisfied,
        }


@dataclass(frozen=True)
class _CacheEntry:
    namespace_root: Path
    artifact_id: str
    manifest_path: Path
    data_path: Path
    artifact_bytes: int
    manifest_bytes: int
    modified_ns: int
    active_namespace: bool

    @property
    def total_bytes(self) -> int:
        return self.artifact_bytes + self.manifest_bytes

    @property
    def identity(self) -> str:
        return f"{self.namespace_root.name}/{self.artifact_id}"


class SharedPanelPackageCache:
    """Bounded, read-only-after-publication cache for exact panel packages.

    Cache artifacts are operational accelerators, never scientific authority.
    A consumer accepts an artifact only when its full authority/session key,
    byte hash, size, row count, ordered columns, and dtypes all validate.
    """

    def __init__(
        self,
        root: Path,
        *,
        maximum_bytes: int = DEFAULT_PACKAGE_CACHE_MAX_BYTES,
        minimum_post_write_free_bytes: int = MIN_POST_CACHE_FREE_DISK_BYTES,
        lock_timeout_seconds: float = 600.0,
        retention_root: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.retention_root = (
            Path(retention_root) if retention_root is not None else self.root
        )
        self.maximum_bytes = int(maximum_bytes)
        self.minimum_post_write_free_bytes = int(minimum_post_write_free_bytes)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        if self.maximum_bytes <= 0:
            raise ValueError("Shared package cache maximum must be positive")
        if self.minimum_post_write_free_bytes < 0:
            raise ValueError("Shared package cache post-write floor cannot be negative")
        if retention_root is not None:
            if self.root.parent != self.retention_root:
                raise ValueError(
                    "A grouped cache namespace must be a direct child of its "
                    "retention root"
                )
            if not self._is_source_namespace_name(self.root.name):
                raise ValueError(
                    "A grouped cache namespace must use source=<16 hex>"
                )
        self.retention_root.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        if self._is_link_like(self.retention_root) or self._is_link_like(self.root):
            raise PackageCacheError(
                "Shared package cache roots must not be links or junctions"
            )

    def get_or_build(
        self,
        key: PanelPackageKey,
        builder: Callable[[], pd.DataFrame],
    ) -> CachedPanelResult:
        if (
            self.retention_root != self.root
            and self.root.name
            != f"{SOURCE_NAMESPACE_PREFIX}{key.assembly_source_hash[:SOURCE_NAMESPACE_HEX_LENGTH]}"
        ):
            raise PackageCacheError(
                "Shared package cache source namespace does not match the key"
            )
        with self._lock(self._key_lock_path(key.digest)):
            cached = self._load_locked(key)
            if cached is not None:
                return cached
            superset_key = self._smallest_compatible_superset_key(key)
            if superset_key is not None:
                with self._lock(self._key_lock_path(superset_key.digest)):
                    superset = self._load_locked(superset_key)
                if superset is not None:
                    if "session_date" not in superset.frame:
                        raise PackageCacheError(
                            "Shared package cache superset lacks session_date"
                        )
                    requested_dates = set(key.session_dates)
                    filtered = superset.frame[
                        superset.frame["session_date"].astype(str).isin(
                            requested_dates
                        )
                    ].copy()
                    if filtered.empty:
                        raise PackageCacheError(
                            "Shared package cache superset produced an empty exact slice"
                        )
                    return CachedPanelResult(
                        frame=filtered,
                        disposition="HIT_SUPERSET",
                        cache_key=superset.cache_key,
                        artifact_bytes=superset.artifact_bytes,
                        read_wall_seconds=superset.read_wall_seconds,
                        write_wall_seconds=0.0,
                    )
            frame = builder()
            if frame.empty:
                raise PackageCacheError("Refusing to cache an empty panel package")
            return self._publish_locked(key, frame)

    def _smallest_compatible_superset_key(
        self, key: PanelPackageKey
    ) -> PanelPackageKey | None:
        requested_payload = key.payload()
        requested_dates = set(key.session_dates)
        candidates: list[PanelPackageKey] = []
        for manifest_path in self.root.glob("*.json"):
            artifact_id = manifest_path.stem
            if len(artifact_id) != 32 or any(
                character not in "0123456789abcdef" for character in artifact_id
            ):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError) as exc:
                raise PackageCacheError(
                    f"Unreadable shared package cache manifest: {manifest_path}"
                ) from exc
            if manifest.get("schema") != PACKAGE_CACHE_SCHEMA:
                raise PackageCacheError("Shared package cache schema mismatch")
            payload = dict(manifest.get("key_payload") or {})
            comparable_fields = (
                "schema",
                "feature_authority_hash",
                "target_authority_hash",
                "static_authority_bundle_sha256",
                "predictor_order_hash",
                "assembly_source_hash",
                "maximum_assets",
            )
            if any(
                payload.get(field) != requested_payload.get(field)
                for field in comparable_fields
            ):
                continue
            candidate_dates = tuple(str(value) for value in payload.get("session_dates", ()))
            if not requested_dates < set(candidate_dates):
                continue
            candidate = PanelPackageKey(
                feature_authority_hash=str(payload["feature_authority_hash"]),
                target_authority_hash=str(payload["target_authority_hash"]),
                static_authority_bundle_sha256=str(
                    payload["static_authority_bundle_sha256"]
                ),
                predictor_order_hash=str(payload["predictor_order_hash"]),
                assembly_source_hash=str(payload["assembly_source_hash"]),
                session_dates=candidate_dates,
                maximum_assets=(
                    int(payload["maximum_assets"])
                    if payload.get("maximum_assets") is not None
                    else None
                ),
            )
            if candidate.digest != manifest.get("cache_key"):
                raise PackageCacheError(
                    "Shared package cache superset key mismatch"
                )
            candidates.append(candidate)
        return min(candidates, key=lambda item: len(item.session_dates), default=None)

    def _load_locked(self, key: PanelPackageKey) -> CachedPanelResult | None:
        manifest_path = self._manifest_path(key.digest)
        data_path = self._data_path(key.digest)
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PackageCacheError(
                f"Unreadable shared package cache manifest: {manifest_path}"
            ) from exc
        if manifest.get("schema") != PACKAGE_CACHE_SCHEMA:
            raise PackageCacheError("Shared package cache schema mismatch")
        if manifest.get("cache_key") != key.digest:
            raise PackageCacheError("Shared package cache key mismatch")
        if manifest.get("key_payload") != key.payload():
            raise PackageCacheError("Shared package cache authority mismatch")
        if not data_path.is_file():
            raise PackageCacheError("Shared package cache data file is missing")
        expected_size = int(manifest.get("artifact_bytes", -1))
        if data_path.stat().st_size != expected_size:
            raise PackageCacheError("Shared package cache size mismatch")
        if file_sha256(data_path) != manifest.get("artifact_sha256"):
            raise PackageCacheError("Shared package cache byte hash mismatch")

        lease_path = self._lease_path(key.digest)
        self._write_lease(lease_path, key.digest)
        started = time.perf_counter()
        try:
            frame = pd.read_parquet(data_path)
        finally:
            lease_path.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        self._validate_frame(frame, manifest)
        return CachedPanelResult(
            frame=frame,
            disposition="HIT",
            cache_key=key.digest,
            artifact_bytes=expected_size,
            read_wall_seconds=elapsed,
            write_wall_seconds=0.0,
        )

    def _publish_locked(
        self,
        key: PanelPackageKey,
        frame: pd.DataFrame,
    ) -> CachedPanelResult:
        data_path = self._data_path(key.digest)
        manifest_path = self._manifest_path(key.digest)
        temporary = self.root / (
            f".{self._artifact_id(key.digest)}.{os.getpid()}."
            f"{uuid4().hex[:8]}.partial"
        )
        started = time.perf_counter()
        try:
            frame.to_parquet(temporary, index=False)
            artifact_bytes = int(temporary.stat().st_size)
            disk_free = int(shutil.disk_usage(self.root).free)
            write_feasibility = evaluate_disk_write_feasibility(
                disk_free_bytes=disk_free,
                required_remaining_bytes=self.minimum_post_write_free_bytes,
                additional_write_bytes=artifact_bytes,
            )
            if (
                artifact_bytes > self.maximum_bytes
                or not write_feasibility.safe
            ):
                return CachedPanelResult(
                    frame=frame,
                    disposition="MISS_NOT_CACHED_CAPACITY",
                    cache_key=key.digest,
                    artifact_bytes=artifact_bytes,
                    read_wall_seconds=0.0,
                    write_wall_seconds=time.perf_counter() - started,
                )
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            artifact_hash = file_sha256(temporary)
            manifest = {
                "schema": PACKAGE_CACHE_SCHEMA,
                "cache_key": key.digest,
                "key_payload": key.payload(),
                "data_file": data_path.name,
                "artifact_bytes": artifact_bytes,
                "artifact_sha256": artifact_hash,
                "row_count": int(len(frame)),
                "columns": [str(column) for column in frame.columns],
                "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
                "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
            }
            manifest_bytes = len(
                (
                    json.dumps(manifest, indent=2, sort_keys=True, default=str)
                    + "\n"
                ).encode("utf-8")
            )
            incoming_bytes = artifact_bytes + manifest_bytes
            with self._lock(self.retention_root / ".cleanup.lock"):
                self._make_room_locked(
                    incoming_bytes,
                    exclude=(self.root, self._artifact_id(key.digest)),
                )
                if self._cache_bytes_locked() + incoming_bytes > self.maximum_bytes:
                    return CachedPanelResult(
                        frame=frame,
                        disposition="MISS_NOT_CACHED_CAPACITY",
                        cache_key=key.digest,
                        artifact_bytes=artifact_bytes,
                        read_wall_seconds=0.0,
                        write_wall_seconds=time.perf_counter() - started,
                    )
                os.replace(temporary, data_path)
                try:
                    write_json_object_atomic(manifest_path, manifest)
                except Exception:
                    data_path.unlink(missing_ok=True)
                    raise
        finally:
            temporary.unlink(missing_ok=True)
        return CachedPanelResult(
            frame=frame,
            disposition="MISS_PUBLISHED",
            cache_key=key.digest,
            artifact_bytes=artifact_bytes,
            read_wall_seconds=0.0,
            write_wall_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
        if len(frame) != int(manifest.get("row_count", -1)):
            raise PackageCacheError("Shared package cache row-count mismatch")
        columns = [str(column) for column in frame.columns]
        if columns != list(manifest.get("columns") or []):
            raise PackageCacheError("Shared package cache column-order mismatch")
        dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
        if dtypes != dict(manifest.get("dtypes") or {}):
            raise PackageCacheError("Shared package cache dtype mismatch")

    def enforce_retention(self) -> CacheRetentionResult:
        """Evict unpinned reproducible entries under the global cache budget.

        Entries from inactive source namespaces are removed first because the
        active worker cannot address them. Active-source entries are then
        evicted oldest-first only as needed to enforce ``maximum_bytes``.
        """

        with self._lock(self.retention_root / ".cleanup.lock"):
            return self._enforce_retention_locked(incoming_bytes=0, exclude=None)

    def _make_room_locked(
        self,
        incoming_bytes: int,
        *,
        exclude: tuple[Path, str] | None,
    ) -> CacheRetentionResult:
        return self._enforce_retention_locked(
            incoming_bytes=incoming_bytes,
            exclude=exclude,
        )

    def _enforce_retention_locked(
        self,
        *,
        incoming_bytes: int,
        exclude: tuple[Path, str] | None,
    ) -> CacheRetentionResult:
        stale_locks = self._reclaim_stale_group_locks()
        entries = self._retention_entries()
        before_bytes = sum(entry.total_bytes for entry in entries)
        current_bytes = before_bytes
        evicted: list[str] = []
        evicted_bytes = 0
        stale_leases: list[str] = []
        skipped_live: list[str] = []

        inactive = sorted(
            (entry for entry in entries if not entry.active_namespace),
            key=lambda entry: (entry.modified_ns, entry.identity),
        )
        active = sorted(
            (entry for entry in entries if entry.active_namespace),
            key=lambda entry: (entry.modified_ns, entry.identity),
        )
        for entry in (*inactive, *active):
            if exclude == (entry.namespace_root, entry.artifact_id):
                continue
            must_remove_inactive = not entry.active_namespace
            must_make_room = current_bytes + incoming_bytes > self.maximum_bytes
            if not must_remove_inactive and not must_make_room:
                continue
            try:
                with self._lock(
                    entry.namespace_root / f"{entry.artifact_id}.lock.json",
                    timeout_seconds=0.0,
                ):
                    reclaimed, live = self._reconcile_entry_leases(entry)
                    stale_leases.extend(reclaimed)
                    if live:
                        skipped_live.append(entry.identity)
                        continue
                    if not entry.manifest_path.is_file():
                        continue
                    entry.manifest_path.unlink()
                    entry.data_path.unlink()
                    current_bytes -= entry.total_bytes
                    evicted.append(entry.identity)
                    evicted_bytes += entry.total_bytes
            except TimeoutError:
                skipped_live.append(entry.identity)

        remaining = self._retention_entries()
        after_bytes = sum(entry.total_bytes for entry in remaining)
        return CacheRetentionResult(
            maximum_bytes=self.maximum_bytes,
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            evicted_entries=tuple(evicted),
            evicted_bytes=evicted_bytes,
            stale_locks_reclaimed=tuple(sorted(stale_locks)),
            stale_leases_reclaimed=tuple(sorted(stale_leases)),
            skipped_live_entries=tuple(sorted(set(skipped_live))),
            inactive_entries_remaining=tuple(
                entry.identity for entry in remaining if not entry.active_namespace
            ),
        )

    def _retention_entries(self) -> list[_CacheEntry]:
        entries: list[_CacheEntry] = []
        data_paths: set[Path] = set()
        namespace_roots = self._namespace_roots()
        for namespace_root in namespace_roots:
            for manifest_path in namespace_root.glob("*.json"):
                artifact_id = manifest_path.stem
                if not self._is_artifact_id(artifact_id):
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise PackageCacheError(
                        f"Unreadable shared package cache manifest: {manifest_path}"
                    ) from exc
                if manifest.get("schema") != PACKAGE_CACHE_SCHEMA:
                    raise PackageCacheError(
                        f"Shared package cache schema mismatch: {manifest_path}"
                    )
                cache_key = str(manifest.get("cache_key", ""))
                payload = manifest.get("key_payload")
                if (
                    not isinstance(payload, dict)
                    or stable_hash(payload) != cache_key
                    or self._artifact_id(cache_key) != artifact_id
                ):
                    raise PackageCacheError(
                        f"Shared package cache key mismatch: {manifest_path}"
                    )
                expected_data_name = f"{artifact_id}.parquet"
                if manifest.get("data_file") != expected_data_name:
                    raise PackageCacheError(
                        f"Shared package cache data path mismatch: {manifest_path}"
                    )
                if self.retention_root != self.root:
                    source_hash = str(payload.get("assembly_source_hash", ""))
                    if namespace_root.name != (
                        f"{SOURCE_NAMESPACE_PREFIX}"
                        f"{source_hash[:SOURCE_NAMESPACE_HEX_LENGTH]}"
                    ):
                        raise PackageCacheError(
                            f"Shared package cache namespace mismatch: {manifest_path}"
                        )
                data_path = namespace_root / expected_data_name
                if not data_path.is_file():
                    raise PackageCacheError(
                        f"Shared package cache data file is missing: {data_path}"
                    )
                artifact_bytes = int(manifest.get("artifact_bytes", -1))
                if artifact_bytes < 0 or data_path.stat().st_size != artifact_bytes:
                    raise PackageCacheError(
                        f"Shared package cache size mismatch: {data_path}"
                    )
                data_paths.add(data_path)
                entries.append(
                    _CacheEntry(
                        namespace_root=namespace_root,
                        artifact_id=artifact_id,
                        manifest_path=manifest_path,
                        data_path=data_path,
                        artifact_bytes=artifact_bytes,
                        manifest_bytes=manifest_path.stat().st_size,
                        modified_ns=manifest_path.stat().st_mtime_ns,
                        active_namespace=namespace_root == self.root,
                    )
                )
        orphaned = sorted(
            str(path)
            for namespace_root in namespace_roots
            for path in namespace_root.glob("*.parquet")
            if path.is_file() and path not in data_paths
        )
        if orphaned:
            raise PackageCacheError(
                "Unowned shared package cache data must be reconciled before "
                f"retention: {orphaned}"
            )
        return entries

    def _namespace_roots(self) -> tuple[Path, ...]:
        if self.retention_root == self.root:
            return (self.root,)
        roots: list[Path] = []
        for candidate in self.retention_root.iterdir():
            if not self._is_source_namespace_name(candidate.name):
                continue
            if self._is_link_like(candidate):
                raise PackageCacheError(
                    f"Refusing to follow cache namespace link or junction: {candidate}"
                )
            if not candidate.is_dir():
                raise PackageCacheError(
                    f"Cache source namespace is not a directory: {candidate}"
                )
            roots.append(candidate)
        return tuple(sorted(roots, key=lambda path: path.name))

    def _cache_bytes_locked(self) -> int:
        return sum(entry.total_bytes for entry in self._retention_entries())

    def _reclaim_stale_group_locks(self) -> list[str]:
        reclaimed: list[str] = []
        for namespace_root in self._namespace_roots():
            for lock_path in namespace_root.glob("*.lock.json"):
                if self._reclaim_stale_lock(lock_path):
                    reclaimed.append(
                        f"{namespace_root.name}/{lock_path.name}"
                    )
        return reclaimed

    def _reconcile_entry_leases(
        self,
        entry: _CacheEntry,
    ) -> tuple[list[str], list[str]]:
        reclaimed: list[str] = []
        live: list[str] = []
        for lease_path in entry.namespace_root.glob(
            f"{entry.artifact_id}.lease.*.json"
        ):
            try:
                payload = json.loads(lease_path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", 0) or 0)
                creation = str(
                    payload.get("process_creation_time_utc", "") or ""
                )
                cache_key = str(payload.get("cache_key", ""))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                live.append(str(lease_path))
                continue
            if (
                self._artifact_id(cache_key) != entry.artifact_id
                or pid <= 0
                or not creation
            ):
                live.append(str(lease_path))
                continue
            if process_identity_matches(
                process_identity(pid),
                expected_creation_time=creation,
                required_command_fragments=(),
            ):
                live.append(str(lease_path))
                continue
            lease_path.unlink(missing_ok=True)
            reclaimed.append(f"{entry.namespace_root.name}/{lease_path.name}")
        return reclaimed, live

    @staticmethod
    def _is_artifact_id(value: str) -> bool:
        return len(value) == 32 and all(
            character in "0123456789abcdef" for character in value
        )

    @staticmethod
    def _is_source_namespace_name(value: str) -> bool:
        if not value.startswith(SOURCE_NAMESPACE_PREFIX):
            return False
        suffix = value[len(SOURCE_NAMESPACE_PREFIX) :]
        return len(suffix) == SOURCE_NAMESPACE_HEX_LENGTH and all(
            character in "0123456789abcdef" for character in suffix
        )

    @staticmethod
    def _is_link_like(path: Path) -> bool:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(is_junction and is_junction())

    def _manifest_path(self, cache_key: str) -> Path:
        return self.root / f"{self._artifact_id(cache_key)}.json"

    def _data_path(self, cache_key: str) -> Path:
        return self.root / f"{self._artifact_id(cache_key)}.parquet"

    def _key_lock_path(self, cache_key: str) -> Path:
        return self.root / f"{self._artifact_id(cache_key)}.lock.json"

    def _lease_path(self, cache_key: str) -> Path:
        return self.root / (
            f"{self._artifact_id(cache_key)}.lease.{os.getpid()}."
            f"{uuid4().hex[:8]}.json"
        )

    @staticmethod
    def _artifact_id(cache_key: str) -> str:
        """Use a collision-checked 128-bit filename token on legacy Windows paths."""

        return str(cache_key)[:32]

    @staticmethod
    def _write_lease(path: Path, cache_key: str) -> None:
        identity = process_identity(os.getpid())
        write_json_object_atomic(
            path,
            {
                "cache_key": cache_key,
                "pid": identity.pid,
                "process_creation_time_utc": identity.creation_time_utc,
                "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
            },
        )

    @contextmanager
    def _lock(
        self,
        path: Path,
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[None]:
        timeout = self.lock_timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + max(0.0, timeout)
        token = uuid4().hex
        identity = process_identity(os.getpid())
        payload = {
            "token": token,
            "pid": identity.pid,
            "process_creation_time_utc": identity.creation_time_utc,
            "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        }
        while True:
            try:
                descriptor = os.open(
                    path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            except FileExistsError:
                if self._reclaim_stale_lock(path):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring package-cache lock: {path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            if current.get("token") == token:
                path.unlink(missing_ok=True)

    @staticmethod
    def _reclaim_stale_lock(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0) or 0)
            creation = str(payload.get("process_creation_time_utc", "") or "")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if pid <= 0 or not creation or not process_identity_matches(
            process_identity(pid),
            expected_creation_time=creation,
            required_command_fragments=(),
        ):
            path.unlink(missing_ok=True)
            return True
        return False
