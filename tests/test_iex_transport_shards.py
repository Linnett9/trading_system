from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import application.services.hf_data_cutover_perf1 as perf1_module
from application.services.hf_data_cutover_perf1 import (
    Perf1TransportConfiguration,
    _acquire_shards,
    _assemble_transport_partition,
    _shard_batches,
    select_transport,
)
from application.services.hf_data_cutover_iex_contract import IexBackfillSettings
from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths
from core.research.iex_market_data import (
    IexPartitionRequest,
    canonical_raw_bar_frame,
    validate_raw_bar_partition,
)
from infrastructure.data.iex_market_data_store import IexMarketDataStore, sha256_file
from infrastructure.data.iex_transport_shards import (
    IexTransportShardStore,
    plan_monthly_transport_shards,
)


def _request() -> IexPartitionRequest:
    return IexPartitionRequest(
        "AAPL",
        "5m",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 4, 1, tzinfo=timezone.utc),
        year=2024,
    )


def _frame(symbol: str, timestamp: str) -> pd.DataFrame:
    return canonical_raw_bar_frame(
        [
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "trade_count": 2,
                "vwap": 10.25,
            }
        ]
    )


def _publish(store: IexTransportShardStore, shard, timestamp: str) -> dict[str, object]:
    frame = _frame(shard.canonical.symbol, timestamp)
    quality = validate_raw_bar_partition(frame, shard.validation_request())
    return store.publish(shard, frame, quality=quality, source={"temporary": True})


def test_monthly_shards_preserve_canonical_identity_and_bounds() -> None:
    request = _request()
    shards = plan_monthly_transport_shards(request)
    assert len(shards) == 3
    assert [(row.start.month, row.end.month) for row in shards] == [(1, 2), (2, 3), (3, 4)]
    assert {row.canonical.partition_id for row in shards} == {request.partition_id}


def test_shard_resume_skips_one_and_several_verified_months(tmp_path: Path) -> None:
    request = _request()
    shards = plan_monthly_transport_shards(request)
    store = IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000)
    first = _publish(store, shards[0], "2024-01-02T14:30:00Z")
    first_hash = first["parquet_sha256"]

    resumed = IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000)
    assert resumed.completed_shard_count(shards) == 1
    assert _publish(resumed, shards[0], "2024-01-02T14:30:00Z")["parquet_sha256"] == first_hash
    _publish(resumed, shards[1], "2024-02-02T14:30:00Z")
    assert IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000).completed_shard_count(shards) == 2


def test_canonical_assembly_is_atomic_and_cleans_shards_only_after_verification(tmp_path: Path) -> None:
    request = _request()
    shards = plan_monthly_transport_shards(request)
    shard_store = IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000)
    for shard, timestamp in zip(
        shards,
        ("2024-01-02T14:30:00Z", "2024-02-02T14:30:00Z", "2024-03-04T14:30:00Z"),
        strict=True,
    ):
        _publish(shard_store, shard, timestamp)
    canonical = IexMarketDataStore(tmp_path / "canonical")
    result = _assemble_transport_partition(
        request,
        shards,
        shard_store,
        Perf1TransportConfiguration("C", "monthly", "persistent_per_worker", 8),
        publish_canonical=True,
        canonical_store=canonical,
    )
    assert result is not None and result["published"] == 1
    assert canonical.completed_manifest(request)["quality"]["row_count"] == 3
    assert not (shard_store.root / request.partition_id).exists()


def test_interruption_before_manifest_publication_exposes_no_completed_authority(tmp_path: Path) -> None:
    request = _request()
    canonical = IexMarketDataStore(tmp_path / "canonical")
    parquet = canonical.parquet_path(request)
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"orphaned pre-manifest bytes")
    assert canonical.completed_manifest(request) is None
    frame = _frame("AAPL", "2024-01-02T14:30:00Z")
    manifest = canonical.publish(
        request,
        frame,
        quality=validate_raw_bar_partition(frame, request),
        source={"test": True},
    )
    assert canonical.completed_manifest(request) == manifest


def test_completed_canonical_partition_is_never_a_benchmark_write_target(tmp_path: Path) -> None:
    request = _request()
    canonical = IexMarketDataStore(tmp_path / "canonical")
    frame = _frame("AAPL", "2024-01-02T14:30:00Z")
    canonical.publish(
        request,
        frame,
        quality=validate_raw_bar_partition(frame, request),
        source={"existing": True},
    )
    before_bytes = canonical.parquet_path(request).read_bytes()
    before_hash = sha256_file(canonical.parquet_path(request))
    pending = [candidate for candidate in (request,) if canonical.completed_manifest(candidate) is None]
    assert pending == []
    assert canonical.parquet_path(request).read_bytes() == before_bytes
    assert sha256_file(canonical.parquet_path(request)) == before_hash


def test_monthly_batch_geometry_never_exceeds_five_symbols() -> None:
    base = _request()
    groups = []
    for index in range(12):
        request = IexPartitionRequest(
            f"S{index}", base.timeframe, base.start, base.end, year=base.year
        )
        groups.append(plan_monthly_transport_shards(request))
    batches = _shard_batches(groups)
    assert batches
    assert max(len(batch.shards) for batch in batches) == 5
    assert all(len({(shard.start, shard.end) for shard in batch.shards}) == 1 for batch in batches)


def test_corrupted_verified_shard_fails_closed(tmp_path: Path) -> None:
    shard = plan_monthly_transport_shards(_request())[0]
    store = IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000)
    _publish(store, shard, "2024-01-02T14:30:00Z")
    store.parquet_path(shard).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.completed_manifest(shard)


def test_acquisition_resume_requests_only_missing_months(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = _request()
    shards = plan_monthly_transport_shards(request)
    shard_store = IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000)
    _publish(shard_store, shards[0], "2024-01-02T14:30:00Z")
    requested_bounds: list[tuple[datetime, datetime]] = []

    monkeypatch.setattr(
        perf1_module.AlpacaBasicHistoricalBarProvider,
        "check_authentication",
        lambda _self: {"can_attempt_authenticated_request": True},
    )

    def fake_fetch(provider, provider_request, **_kwargs):
        requested_bounds.append((provider_request.start, provider_request.end))
        provider.metrics.requests_attempted += 1
        provider.metrics.requests_successful += 1
        provider.metrics.pages_downloaded += 1
        timestamp = provider_request.start + pd.Timedelta(days=1, hours=14, minutes=30)
        rows = _frame("AAPL", timestamp.isoformat()).to_dict("records")
        provider.metrics.rows_downloaded += len(rows)
        provider.metrics.response_latency_seconds.append(0.1)
        return rows, {"pages": 1, "raw_page_sha256": ["a" * 64]}

    monkeypatch.setattr(perf1_module, "fetch_chunk_with_retries", fake_fetch)
    repo = tmp_path / "repo"
    audit = repo / "audit"
    repo.mkdir()
    audit.mkdir()
    paths = CutoverPaths(repo, audit, tmp_path / "r1", tmp_path / "r2")
    settings = IexBackfillSettings(request.start, request.end, ("AAPL",), tmp_path / "canonical", ())
    result = _acquire_shards(
        paths,
        settings,
        (request,),
        Perf1TransportConfiguration("B", "monthly", "fresh_per_attempt", 8),
        shard_store,
        publish_canonical=False,
    )

    assert set(requested_bounds) == {
        (shards[1].start, shards[1].end),
        (shards[2].start, shards[2].end),
    }
    assert result["duplicated_work_avoided"] == 1
    assert result["validated_rows"] == 3


def test_perf1_fails_before_provider_use_when_disk_reserve_is_breached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = _request()
    paths = CutoverPaths(tmp_path, tmp_path / "audit", tmp_path / "r1", tmp_path / "r2")
    settings = IexBackfillSettings(request.start, request.end, ("AAPL",), tmp_path / "canonical", ())
    shard_store = IexTransportShardStore(tmp_path / "scratch", scratch_ceiling_bytes=10_000_000)
    monkeypatch.setattr(
        perf1_module.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"total": 100, "used": 99, "free": 1})(),
    )
    with pytest.raises(CutoverAuditError, match="15 GiB reserve"):
        _acquire_shards(
            paths,
            settings,
            (request,),
            Perf1TransportConfiguration("B", "monthly", "fresh_per_attempt", 8),
            shard_store,
            publish_canonical=False,
        )


def test_transport_selection_requires_material_gain_and_integrity(tmp_path: Path) -> None:
    paths = CutoverPaths(tmp_path, tmp_path / "audit", tmp_path / "r1", tmp_path / "r2")
    paths.output_root.mkdir()
    baseline = {
        "variant": "A",
        "validated_rows_per_second": 100.0,
        "session_mode": "fresh_per_attempt",
        "shard_geometry": "symbol_year",
        "workers": 8,
        "status": "HISTORICAL_TELEMETRY",
    }
    marginal = {
        **baseline,
        "variant": "B",
        "validated_rows_per_second": 119.0,
        "shard_geometry": "monthly",
        "status": "PASSED",
        "irrecoverable_partitions": 0,
        "failed_shards": 0,
        "http_429_count": 0,
    }
    unsafe = {
        **marginal,
        "variant": "C",
        "validated_rows_per_second": 150.0,
        "failed_shards": 1,
    }
    assert select_transport(paths, (baseline, marginal, unsafe))["selected_variant"] == "A"
    safe = {**unsafe, "failed_shards": 0}
    assert select_transport(paths, (baseline, marginal, safe))["selected_variant"] == "C"
