"""Build the separate provider-adjusted IEX daily research view."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from application.services.hf_data_cutover_iex_backfill import BACKFILL_ASOF, alpaca_symbol_aliases
from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, utc_now
from core.research.iex_market_data import IexPartitionRequest, canonical_raw_bar_frame, validate_raw_bar_partition
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    HistoricalBarRequest,
    SharedRateLimiter,
    fetch_chunk_with_retries,
)
from infrastructure.data.iex_market_data_store import sha256_file
from infrastructure.data.iex_research_artifact_store import IexResearchArtifactStore


ADJUSTED_DAILY_WORKERS = 4


def run_iex_adjusted_daily_build(paths: CutoverPaths) -> dict[str, Any]:
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("state") not in {
        "IEX_RAW_DATA_COMPLETE_FEATURE_BUILD_PENDING",
        "IEX_ADJUSTED_DAILY_IN_PROGRESS",
    }:
        raise CutoverAuditError(f"Adjusted daily view requires complete raw data, found {run_manifest.get('state')}")
    coverage = _read_json(paths.output_root / "IEX_HISTORICAL_COVERAGE_PROBE.json")
    start = datetime.fromisoformat(coverage["selected_uniform_acquisition_start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(coverage["selected_acquisition_end_exclusive"].replace("Z", "+00:00"))
    roster = tuple(_read_json(paths.output_root / "IEX_ACQUISITION_ROSTER.json")["symbols"])
    aliases = dict(alpaca_symbol_aliases(paths.repository_root, roster))
    provider = AlpacaBasicHistoricalBarProvider(
        feed="iex",
        rate_limiter=SharedRateLimiter(requests_per_minute=150, max_in_flight_requests=ADJUSTED_DAILY_WORKERS),
    )
    store = IexResearchArtifactStore(paths.repository_root / "data/market/iex")
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_ADJUSTED_DAILY_IN_PROGRESS",
            "next_gate": "Complete separately identified provider-adjusted daily research view.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    requests = tuple(
        IexPartitionRequest(symbol=symbol, timeframe="1Day", start=start, end=end)
        for symbol in roster
    )
    batches = tuple(tuple(requests[index : index + 25]) for index in range(0, len(requests), 25))
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=ADJUSTED_DAILY_WORKERS, thread_name_prefix="iex-adjusted-daily") as executor:
        futures: dict[Future[list[dict[str, Any]]], tuple[IexPartitionRequest, ...]] = {
            executor.submit(_collect_adjusted_batch, store, provider, batch, aliases): batch for batch in batches
        }
        for future in as_completed(futures):
            completed.extend(future.result())
    completed.sort(key=lambda row: row["identity"]["symbol"])
    manifest = {
        "schema_version": "hf_data_cutover_r1.iex_adjusted_daily_manifest.v1",
        "created_at_utc": utc_now(),
        "provider": "Alpaca",
        "feed": "IEX",
        "adjustment": "all",
        "identity": "SEPARATE_ADJUSTED_RESEARCH_VIEW_NOT_RAW_SOURCE_TRUTH",
        "partition_count": len(completed),
        "row_count": sum(int(row["row_count"]) for row in completed),
        "parquet_bytes": sum(int(row["parquet_bytes"]) for row in completed),
        "partition_manifest_sha256": sorted(
            sha256_file(
                paths.repository_root
                / "data/market/iex"
                / Path(row["identity"]["relative_directory"])
                / "manifest.json"
            )
            for row in completed
        ),
        "status": "COMPLETE" if len(completed) == len(roster) else "INCOMPLETE",
    }
    if manifest["status"] != "COMPLETE":
        raise CutoverAuditError(f"Adjusted daily view incomplete: {len(completed)}/{len(roster)}")
    output_path = paths.output_root / "IEX_DAILY_ADJUSTED_MANIFEST.json"
    atomic_json(output_path, manifest)
    run_manifest = _read_json(run_manifest_path)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_ADJUSTED_DAILY_COMPLETE_1H_PENDING",
            "iex_adjusted_daily": {
                "manifest_sha256": sha256_file(output_path),
                "rows": manifest["row_count"],
            },
            "next_gate": "Derive canonical completed 1h bars.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def _collect_adjusted_batch(
    store: IexResearchArtifactStore,
    provider: AlpacaBasicHistoricalBarProvider,
    requests: tuple[IexPartitionRequest, ...],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    pending: list[IexPartitionRequest] = []
    results: list[dict[str, Any]] = []
    for request in requests:
        identity = _identity(request)
        existing = store.completed(_relative(request.symbol), identity)
        if existing is None:
            pending.append(request)
        else:
            results.append(existing)
    if not pending:
        return results
    provider_symbols = tuple(aliases.get(request.symbol, request.symbol) for request in pending)
    remaps = tuple(
        (request.symbol, provider_symbols[index])
        for index, request in enumerate(pending)
        if provider_symbols[index] != request.symbol
    )
    first = pending[0]
    provider_request = HistoricalBarRequest(
        symbols=provider_symbols,
        timeframe="1Day",
        start=first.start,
        end=first.end,
        feed="iex",
        adjustment="all",
        limit=10_000,
        raw_chunk_id=f"iex-adjusted-daily-{'-'.join(request.symbol for request in pending)}",
        canonical_symbols=tuple(request.symbol for request in pending),
        provider_symbol_by_canonical=remaps,
        currency="USD",
        asof=BACKFILL_ASOF,
    )
    rows, chunk = fetch_chunk_with_retries(
        provider,
        provider_request,
        max_retries=5,
        retain_raw_pages=False,
    )
    combined = canonical_raw_bar_frame(rows)
    for request in pending:
        frame = canonical_raw_bar_frame(combined.loc[combined["symbol"].eq(request.symbol)].to_dict("records"))
        quality = validate_raw_bar_partition(frame, request)
        results.append(
            store.publish(
                _relative(request.symbol),
                frame,
                identity=_identity(request),
                lineage={
                    "provider": "Alpaca",
                    "feed": "IEX",
                    "adjustment": "all",
                    "asof": BACKFILL_ASOF,
                    "raw_page_sha256_for_shared_batch": chunk["raw_page_sha256"],
                    "provider_symbol_map": dict(remaps),
                },
                quality=quality,
            )
        )
    return results


def _identity(request: IexPartitionRequest) -> dict[str, Any]:
    relative = _relative(request.symbol)
    return {
        "dataset": "iex_adjusted_daily_research_view_v1",
        "table": "1d_adjusted",
        "symbol": request.symbol,
        "year": None,
        "relative_directory": str(relative),
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "feed": "IEX",
        "adjustment": "all",
        "builder_version": "iex_adjusted_daily_v1",
    }


def _relative(symbol: str) -> Path:
    return Path("derived/1d_adjusted") / f"symbol={symbol}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["run_iex_adjusted_daily_build"]
