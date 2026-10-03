from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from core.research.ml.ds24.clean_v2_package_cache import (
    DEFAULT_PACKAGE_CACHE_MAX_BYTES,
    PanelPackageKey,
    SharedPanelPackageCache,
)
from core.research.ml.ds24.clean_v2_resources import GIB, process_identity


MEBIBYTE = 1024**2
OLD_SOURCE = "1" * 64
CURRENT_SOURCE = "2" * 64


def test_default_global_budget_reserves_one_panel_staging_window() -> None:
    assert DEFAULT_PACKAGE_CACHE_MAX_BYTES == 7 * GIB // 4


def _cache(
    group_root: Path,
    source_hash: str,
    *,
    maximum_bytes: int = 16 * MEBIBYTE,
) -> SharedPanelPackageCache:
    return SharedPanelPackageCache(
        group_root / f"source={source_hash[:16]}",
        retention_root=group_root,
        maximum_bytes=maximum_bytes,
        lock_timeout_seconds=0.0,
    )


def _key(source_hash: str, session_date: str = "2024-01-02") -> PanelPackageKey:
    return PanelPackageKey(
        feature_authority_hash="feature",
        target_authority_hash="target",
        static_authority_bundle_sha256="bundle",
        predictor_order_hash="predictors",
        assembly_source_hash=source_hash,
        session_dates=(session_date,),
        maximum_assets=None,
    )


def _frame(session_date: str = "2024-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["A", "B"],
            "session_date": [session_date, session_date],
            "predictor": [1.25, 2.5],
            "target_value": [0.01, -0.02],
        }
    )


def _artifact_id(key: PanelPackageKey) -> str:
    return SharedPanelPackageCache._artifact_id(key.digest)


def test_source_transition_evicts_only_reproducible_inactive_entries_and_rebuilds(
    tmp_path: Path,
) -> None:
    group_root = tmp_path / "shared"
    old_cache = _cache(group_root, OLD_SOURCE)
    expected = _frame()
    old_cache.get_or_build(_key(OLD_SOURCE), lambda: expected.copy())

    current_cache = _cache(group_root, CURRENT_SOURCE)
    retention = current_cache.enforce_retention()

    assert retention.retention_satisfied is True
    assert len(retention.evicted_entries) == 1
    assert retention.after_bytes == 0
    assert not list((group_root / f"source={OLD_SOURCE[:16]}").glob("*.parquet"))

    rebuild_calls = 0

    def rebuild() -> pd.DataFrame:
        nonlocal rebuild_calls
        rebuild_calls += 1
        return expected.copy()

    rebuilt = current_cache.get_or_build(_key(CURRENT_SOURCE), rebuild)

    assert rebuild_calls == 1
    assert rebuilt.disposition == "MISS_PUBLISHED"
    assert_frame_equal(rebuilt.frame, expected, check_exact=True)


def test_retention_is_global_and_bounded_within_the_active_source(
    tmp_path: Path,
) -> None:
    group_root = tmp_path / "shared"
    cache = _cache(group_root, CURRENT_SOURCE)
    cache.get_or_build(_key(CURRENT_SOURCE, "2024-01-02"), lambda: _frame())
    cache.get_or_build(
        _key(CURRENT_SOURCE, "2024-01-03"),
        lambda: _frame("2024-01-03"),
    )
    paths = list(cache.root.glob("*.parquet")) + list(cache.root.glob("*.json"))
    total_bytes = sum(path.stat().st_size for path in paths)

    bounded = _cache(
        group_root,
        CURRENT_SOURCE,
        maximum_bytes=total_bytes - 1,
    ).enforce_retention()

    assert bounded.retention_satisfied is True
    assert bounded.before_bytes == total_bytes
    assert bounded.after_bytes <= bounded.maximum_bytes
    assert len(bounded.evicted_entries) == 1
    assert len(list(cache.root.glob("*.parquet"))) == 1


def test_live_entry_lock_is_preserved_but_dead_owner_lock_is_reclaimed(
    tmp_path: Path,
) -> None:
    group_root = tmp_path / "shared"
    old_cache = _cache(group_root, OLD_SOURCE)
    key = _key(OLD_SOURCE)
    old_cache.get_or_build(key, lambda: _frame())
    artifact_id = _artifact_id(key)
    lock_path = old_cache.root / f"{artifact_id}.lock.json"
    identity = process_identity(os.getpid())
    lock_path.write_text(
        json.dumps(
            {
                "token": "live",
                "pid": identity.pid,
                "process_creation_time_utc": identity.creation_time_utc,
                "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
            }
        ),
        encoding="utf-8",
    )
    current_cache = _cache(group_root, CURRENT_SOURCE)

    live_result = current_cache.enforce_retention()

    assert live_result.retention_satisfied is False
    assert live_result.skipped_live_entries == (
        f"source={OLD_SOURCE[:16]}/{artifact_id}",
    )
    assert lock_path.is_file()
    assert (old_cache.root / f"{artifact_id}.parquet").is_file()

    lock_path.write_text(
        json.dumps(
            {
                "token": "dead",
                "pid": 2147483647,
                "process_creation_time_utc": "2000-01-01T00:00:00+00:00",
                "created_at_utc": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    dead_result = current_cache.enforce_retention()

    assert dead_result.retention_satisfied is True
    assert dead_result.stale_locks_reclaimed == (
        f"source={OLD_SOURCE[:16]}/{artifact_id}.lock.json",
    )
    assert not lock_path.exists()
    assert not (old_cache.root / f"{artifact_id}.parquet").exists()


def test_live_reader_lease_is_preserved_but_dead_owner_lease_is_reclaimed(
    tmp_path: Path,
) -> None:
    group_root = tmp_path / "shared"
    old_cache = _cache(group_root, OLD_SOURCE)
    key = _key(OLD_SOURCE)
    old_cache.get_or_build(key, lambda: _frame())
    artifact_id = _artifact_id(key)
    lease_path = old_cache.root / f"{artifact_id}.lease.owner.json"
    identity = process_identity(os.getpid())
    lease = {
        "cache_key": key.digest,
        "pid": identity.pid,
        "process_creation_time_utc": identity.creation_time_utc,
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    lease_path.write_text(json.dumps(lease), encoding="utf-8")
    current_cache = _cache(group_root, CURRENT_SOURCE)

    live_result = current_cache.enforce_retention()

    assert live_result.retention_satisfied is False
    assert live_result.skipped_live_entries == (
        f"source={OLD_SOURCE[:16]}/{artifact_id}",
    )
    assert lease_path.is_file()
    assert (old_cache.root / f"{artifact_id}.parquet").is_file()

    lease["pid"] = 2147483647
    lease["process_creation_time_utc"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(lease), encoding="utf-8")
    dead_result = current_cache.enforce_retention()

    assert dead_result.retention_satisfied is True
    assert dead_result.stale_leases_reclaimed == (
        f"source={OLD_SOURCE[:16]}/{lease_path.name}",
    )
    assert not lease_path.exists()
    assert not (old_cache.root / f"{artifact_id}.parquet").exists()


def test_new_cache_instance_resumes_from_an_exact_current_source_entry(
    tmp_path: Path,
) -> None:
    group_root = tmp_path / "shared"
    key = _key(CURRENT_SOURCE)
    expected = _frame()
    _cache(group_root, CURRENT_SOURCE).get_or_build(
        key,
        lambda: expected.copy(),
    )

    def unexpected_rebuild() -> pd.DataFrame:
        raise AssertionError("resume should reuse the exact current-source entry")

    resumed = _cache(group_root, CURRENT_SOURCE).get_or_build(
        key,
        unexpected_rebuild,
    )

    assert resumed.disposition == "HIT"
    assert_frame_equal(resumed.frame, expected, check_exact=True)


def test_concurrently_evicted_superset_falls_back_to_equivalent_cold_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group_root = tmp_path / "shared"
    cache = _cache(group_root, CURRENT_SOURCE)
    superset_key = PanelPackageKey(
        **{
            **_key(CURRENT_SOURCE).__dict__,
            "session_dates": ("2024-01-02", "2024-01-03"),
        }
    )
    superset_frame = pd.concat(
        [_frame("2024-01-02"), _frame("2024-01-03")],
        ignore_index=True,
    )
    cache.get_or_build(superset_key, lambda: superset_frame.copy())
    original_selector = cache._smallest_compatible_superset_key

    def select_then_evict(key: PanelPackageKey) -> PanelPackageKey | None:
        selected = original_selector(key)
        assert selected == superset_key
        artifact_id = _artifact_id(superset_key)
        (cache.root / f"{artifact_id}.json").unlink()
        (cache.root / f"{artifact_id}.parquet").unlink()
        return selected

    monkeypatch.setattr(cache, "_smallest_compatible_superset_key", select_then_evict)
    expected = _frame("2024-01-03")

    result = cache.get_or_build(
        _key(CURRENT_SOURCE, "2024-01-03"),
        lambda: expected.copy(),
    )

    assert result.disposition == "MISS_PUBLISHED"
    assert_frame_equal(result.frame, expected, check_exact=True)
