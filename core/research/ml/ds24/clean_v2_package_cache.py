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
    process_identity,
    process_identity_matches,
    write_json_object_atomic,
)


PACKAGE_CACHE_SCHEMA = "DS24_CLEAN_V2_SHARED_PANEL_PACKAGE_CACHE_V1"
DEFAULT_PACKAGE_CACHE_MAX_BYTES = 2 * GIB
MIN_POST_CACHE_FREE_DISK_BYTES = 4 * GIB


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
        lock_timeout_seconds: float = 600.0,
    ) -> None:
        self.root = Path(root)
        self.maximum_bytes = int(maximum_bytes)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        if self.maximum_bytes <= 0:
            raise ValueError("Shared package cache maximum must be positive")
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_build(
        self,
        key: PanelPackageKey,
        builder: Callable[[], pd.DataFrame],
    ) -> CachedPanelResult:
        with self._lock(self._key_lock_path(key.digest)):
            cached = self._load_locked(key)
            if cached is not None:
                return cached
            superset_key = self._smallest_compatible_superset_key(key)
            if superset_key is not None:
                with self._lock(self._key_lock_path(superset_key.digest)):
                    superset = self._load_locked(superset_key)
                if superset is None:
                    raise PackageCacheError(
                        "Shared package cache superset disappeared during read"
                    )
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
            if (
                artifact_bytes > self.maximum_bytes
                or disk_free - artifact_bytes < MIN_POST_CACHE_FREE_DISK_BYTES
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
            self._make_room(artifact_bytes, exclude_key=key.digest)
            if self._cache_bytes() + artifact_bytes > self.maximum_bytes:
                return CachedPanelResult(
                    frame=frame,
                    disposition="MISS_NOT_CACHED_CAPACITY",
                    cache_key=key.digest,
                    artifact_bytes=artifact_bytes,
                    read_wall_seconds=0.0,
                    write_wall_seconds=time.perf_counter() - started,
                )
            artifact_hash = file_sha256(temporary)
            os.replace(temporary, data_path)
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
            write_json_object_atomic(manifest_path, manifest)
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

    def _make_room(self, incoming_bytes: int, *, exclude_key: str) -> None:
        if self._cache_bytes() + incoming_bytes <= self.maximum_bytes:
            return
        with self._lock(self.root / ".cleanup.lock"):
            manifests = sorted(
                self.root.glob("*.json"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            for manifest_path in manifests:
                artifact_id = manifest_path.stem
                if artifact_id == self._artifact_id(exclude_key):
                    continue
                if self._cache_bytes() + incoming_bytes <= self.maximum_bytes:
                    break
                try:
                    with self._lock(
                        self.root / f"{artifact_id}.lock.json",
                        timeout_seconds=0.0,
                    ):
                        if any(self.root.glob(f"{artifact_id}.lease.*.json")):
                            continue
                        data_path = self.root / f"{artifact_id}.parquet"
                        manifest_path.unlink(missing_ok=True)
                        data_path.unlink(missing_ok=True)
                except TimeoutError:
                    continue

    def _cache_bytes(self) -> int:
        return sum(
            int(path.stat().st_size)
            for path in self.root.glob("*.parquet")
            if path.is_file()
        )

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
                self._reclaim_stale_lock(path)
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
    def _reclaim_stale_lock(path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0) or 0)
            creation = str(payload.get("process_creation_time_utc", "") or "")
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if pid <= 0 or not creation or not process_identity_matches(
            pid,
            creation,
            require_alive=True,
        ):
            path.unlink(missing_ok=True)

