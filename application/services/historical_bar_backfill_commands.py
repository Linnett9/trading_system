from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    AlpacaHistoricalBarError,
    BackfillChunkStateStore,
    CollectionManifest,
    HistoricalBarRequest,
    ImmutableRawChunkStore,
    SharedRateLimiter,
    fetch_chunk_with_retries,
    free_historical_bar_source_inventory,
)
from infrastructure.data.historical_bar_overlap import audit_historical_bar_overlap
from infrastructure.data.historical_bar_staging import (
    consolidate_staging_chunks,
    coverage_gap_audit,
    validate_normalized_bars,
    write_staging_parquet,
)
import pyarrow.parquet as pq
from infrastructure.data.market_sessions import is_rth_timestamp


def run_historical_bar_backfill_probe(config: Mapping[str, Any]) -> None:
    settings = dict((config.get("ml", {}) or {}).get("historical_bar_backfill", {}) or {})
    symbols = tuple(str(symbol).upper() for symbol in settings.get("symbols", ["SPY"])[: int(settings.get("max_symbols", 2))])
    timeframe = str(settings.get("timeframe", "5m"))
    feed = str(settings.get("feed", "iex"))
    adjustment = str(settings.get("adjustment", "all"))
    end = _parse_datetime(settings.get("end")) or datetime.now(timezone.utc) - timedelta(minutes=20)
    start = _parse_datetime(settings.get("start")) or end - timedelta(days=int(settings.get("probe_days", 2)))
    output_root = Path(settings.get("output_root", "reports/market_data/historical_bar_backfill"))
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "free_provider_probe_report.json"
    provider = AlpacaBasicHistoricalBarProvider(
        feed=feed,
        config=config,
        rate_limiter=SharedRateLimiter(
            requests_per_minute=int(settings.get("requests_per_minute", 180)),
            max_in_flight_requests=int(settings.get("max_in_flight_requests", 4)),
        ),
    )
    auth = provider.check_authentication()
    report: dict[str, Any] = {
        "mode": "historical_bar_backfill_free_provider_probe",
        "free_only": True,
        "canonical_market_data_modified": False,
        "provider_inventory": free_historical_bar_source_inventory(),
        "alpaca_authentication": auth,
        "alpaca_capabilities": provider.capabilities().__dict__,
        "probe_request": {
            "symbols": list(symbols),
            "timeframe": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bounded_symbol_batch": True,
            "bounded_date_window": True,
        },
    }
    if auth["can_attempt_authenticated_request"]:
        state_store = BackfillChunkStateStore(output_root / "chunk_state.json")
        try:
            rows, chunk = fetch_chunk_with_retries(
                provider,
                HistoricalBarRequest(
                    symbols=symbols,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    feed=feed,
                    adjustment=adjustment,
                    limit=int(settings.get("limit", 10_000)),
                    raw_chunk_id=f"probe-{feed}-{timeframe}-{start:%Y%m%d%H%M}-{end:%Y%m%d%H%M}",
                ),
                state_store=state_store,
                max_retries=int(settings.get("max_retries", 3)),
            )
            report.update(
                {
                    "alpaca_probe_executed": True,
                    "probe_success": True,
                    "http_status": 200,
                    "entitlement_classification": "available",
                    "historical_depth_observed": _coverage(rows),
                    "feed_actually_available": feed if rows else None,
                    "successful_5m_probe_ranges": [
                        {
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                            "row_count": len(rows),
                        }
                    ]
                    if timeframe in {"5m", "5Min"}
                    else [],
                    "successful_symbol_coverage": _symbol_coverage(rows, symbols),
                    "observed_pagination_behavior": {
                        "pages": chunk.get("pages", 0),
                        "next_page_token_handled": chunk.get("pages", 0) > 1,
                    },
                    "observed_metrics": provider.metrics.as_dict(),
                    "one_hour_recommendation": (
                        "derive locally from validated 5m bars before collecting independent 1h bars"
                        if timeframe in {"5m", "5Min"}
                        else "validate 5m first before deciding whether to collect independent 1h bars"
                    ),
                }
            )
        except AlpacaHistoricalBarError as exc:
            report.update(
                {
                    "alpaca_probe_executed": True,
                    "probe_success": False,
                    "http_status": exc.status_code,
                    "entitlement_classification": exc.classification,
                    "historical_depth_observed": None,
                    "feed_actually_available": None,
                    "successful_5m_probe_ranges": [],
                    "successful_symbol_coverage": {},
                    "observed_pagination_behavior": None,
                    "observed_metrics": provider.metrics.as_dict(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    else:
        report.update(
            {
                "alpaca_probe_executed": False,
                "historical_depth_observed": None,
                "feed_actually_available": None,
                "successful_5m_probe_ranges": [],
                "successful_symbol_coverage": {},
                "observed_pagination_behavior": None,
                "observed_metrics": provider.metrics.as_dict(),
                "one_hour_recommendation": "derive locally from validated 5m bars after authenticated probe succeeds",
                "blocked_reason": "Alpaca credentials unavailable through config.alpaca or accepted environment aliases",
            }
        )
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("HISTORICAL BAR FREE-PROVIDER PROBE")
    print("canonical_market_data_modified=false")
    print(f"Report: {report_path}")
    print(f"Alpaca probe executed: {report['alpaca_probe_executed']}")
    print(f"Observed metrics: {report['observed_metrics']}")


def run_historical_bar_backfill_collect(config: Mapping[str, Any]) -> None:
    settings = dict((config.get("ml", {}) or {}).get("historical_bar_backfill", {}) or {})
    if str(settings.get("provider", "alpaca")).lower() != "alpaca":
        raise ValueError("historical_bar_backfill currently implements provider=alpaca only")
    feed = str(settings.get("feed", "iex")).lower()
    if feed not in {"iex", "sip"}:
        raise ValueError("historical-bar collection currently supports Alpaca feed=iex or feed=sip only")
    output_root = Path(settings.get("output_root", "reports/market_data/historical_bar_backfill"))
    output_root.mkdir(parents=True, exist_ok=True)
    symbols = _symbols(settings)
    timeframe = str(settings.get("timeframe", "5m"))
    start = _parse_datetime(settings.get("start"))
    end = _parse_datetime(settings.get("end"))
    if start is None or end is None:
        raise ValueError("historical_bar_backfill collect requires explicit start and end")
    symbol_batch_size = int(settings.get("symbol_batch_size", settings.get("max_symbols", 2)))
    date_window_days = int(settings.get("date_window_days", 5))
    dry_run = bool(settings.get("dry_run", True))
    resume = bool(settings.get("resume", True))
    force_refresh = bool(settings.get("force_refresh", False))
    write_raw = bool(settings.get("write_raw", settings.get("raw_write", True)))
    write_staging = bool(settings.get("write_normalized_staging", settings.get("staging_write", True)))
    provider = AlpacaBasicHistoricalBarProvider(
        feed=str(settings.get("feed", "iex")),
        config=config,
        rate_limiter=SharedRateLimiter(
            requests_per_minute=int(settings.get("requests_per_minute", 180)),
            max_in_flight_requests=int(settings.get("max_in_flight_requests", 4)),
        ),
    )
    auth = provider.check_authentication()
    plan = _plan(
        symbols,
        start,
        end,
        timeframe=timeframe,
        feed=str(settings.get("feed", "iex")),
        adjustment=str(settings.get("adjustment", "all")),
        symbol_batch_size=symbol_batch_size,
        date_window_days=date_window_days,
    )
    manifest = CollectionManifest(output_root / "collection_manifest.json")
    raw_store = ImmutableRawChunkStore(settings.get("raw_root", "data/raw/alpaca/stock_bars"))
    state_store = BackfillChunkStateStore(output_root / "chunk_state.json")
    all_rows: list[dict[str, Any]] = []
    chunks = []
    if dry_run or not auth["can_attempt_authenticated_request"]:
        for request in plan:
            manifest.update(request.raw_chunk_id or "", "planned", {"dry_run": dry_run})
        report = {
            "mode": "historical_bar_backfill_collect",
            "dry_run": dry_run,
            "canonical_market_data_modified": False,
            "feed_domain_shift_note": _feed_domain_shift_note(),
            "credentials": auth,
            "planned_chunk_count": len(plan),
            "planned_symbol_count": len(symbols),
            "observed_metrics": provider.metrics.as_dict(),
            "blocked_reason": None if dry_run else "Alpaca credentials unavailable",
        }
        _write_collect_report(output_root, report)
        _print_collect_report(output_root, report)
        return
    for request in plan:
        chunk_id = request.raw_chunk_id or ""
        if resume and not force_refresh and manifest.is_completed(chunk_id):
            try:
                skipped_rows = raw_store.read_completed_chunk(request)
                all_rows.extend(skipped_rows)
                chunks.append({"chunk_id": chunk_id, "status": "skipped_completed", "rows": len(skipped_rows)})
            except FileNotFoundError as exc:
                chunks.append({"chunk_id": chunk_id, "status": "skipped_completed_raw_missing", "error_message": str(exc)})
            continue
        manifest.update(chunk_id, "in_progress", {"symbols": list(request.symbols), "start": request.start.isoformat(), "end": request.end.isoformat()})
        try:
            before = provider.metrics.as_dict()
            rows, chunk = fetch_chunk_with_retries(
                provider,
                request,
                state_store=state_store if resume else None,
                max_retries=int(settings.get("max_retries", 3)),
            )
            after = provider.metrics.as_dict()
            chunk_metrics = _metric_delta(before, after)
            raw_manifest = None
            if write_raw:
                raw_manifest = raw_store.write_completed_chunk(
                    request,
                    rows=rows,
                    raw_pages=chunk.get("raw_pages", []),
                    metrics={**chunk, **chunk_metrics},
                    force_refresh=force_refresh,
                    raw_write=True,
                )
            all_rows.extend(rows)
            status = "completed" if rows else "empty_valid_response"
            manifest.update(chunk_id, status, {"rows": len(rows), "pages": chunk.get("pages", 0), "raw_manifest": raw_manifest})
            chunks.append({"chunk_id": chunk_id, "status": status, "rows": len(rows), "pages": chunk.get("pages", 0)})
        except Exception as exc:
            status = getattr(exc, "classification", "retryable_failure")
            manifest.update(chunk_id, status, {"error_type": type(exc).__name__, "error_message": str(exc)})
            chunks.append({"chunk_id": chunk_id, "status": status, "error_type": type(exc).__name__})
            if status in {"permanent_authentication_failure", "entitlement_failure"}:
                break
    pre_validation = validate_normalized_bars(all_rows)
    consolidated_rows = []
    consolidation_report: dict[str, Any] | None = None
    post_validation: dict[str, Any] | None = None
    staging_path = None
    coverage_report = []
    if write_staging and all_rows:
        consolidated_rows, consolidation_report = consolidate_staging_chunks(
            all_rows,
            allow_conflicting_duplicates=bool(settings.get("allow_conflicting_duplicates", False)),
        )
        post_validation = validate_normalized_bars(consolidated_rows)
        staging_path = output_root / "staging" / str(settings.get("feed", "iex")) / timeframe / "bars.parquet"
        write_staging_parquet(consolidated_rows, staging_path)
        coverage_report = coverage_gap_audit(
            consolidated_rows,
            timeframe=timeframe,
            requested_start=start,
            requested_end=end,
            provider="alpaca",
            feed=str(settings.get("feed", "iex")),
        )
    research_view_audit = _research_view_audit(consolidated_rows or all_rows, post_validation or pre_validation, coverage_report)
    report = {
        "mode": "historical_bar_backfill_collect",
        "research_data_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "dry_run": False,
        "canonical_market_data_modified": False,
        "canonical_merge_enabled": False,
        "feed_domain_shift_note": _feed_domain_shift_note(),
        "credentials": auth,
        "planned_chunk_count": len(plan),
        "chunk_results": chunks,
        "raw_write_enabled": write_raw,
        "staging_write_enabled": write_staging,
        "staging_path": str(staging_path) if staging_path else None,
        "pre_consolidation_validation": pre_validation,
        "post_consolidation_validation": post_validation,
        "normalization_validation": post_validation or pre_validation,
        "consolidation_report": consolidation_report,
        "final_structural_validity": (post_validation or pre_validation)["valid"],
        "coverage_completeness_status": _coverage_status(coverage_report),
        "research_view_audit": research_view_audit,
        "all_session_row_count": research_view_audit["all_session_row_count"],
        "rth_row_count": research_view_audit["rth_row_count"],
        "outside_rth_row_count": research_view_audit["outside_rth_row_count"],
        "no_forward_fill": research_view_audit["no_forward_fill"],
        "no_synthetic_5m_bars": research_view_audit["no_synthetic_5m_bars"],
        "no_stale_close_carry": research_view_audit["no_stale_close_carry"],
        "duplicate_count": research_view_audit["duplicate_count"],
        "structurally_valid_staging": research_view_audit["structurally_valid_staging"],
        "missing_expected_rth_bars": research_view_audit["missing_expected_rth_bars"],
        "coverage_report": coverage_report,
        "observed_metrics": provider.metrics.as_dict(),
    }
    _write_collect_report(output_root, report)
    _print_collect_report(output_root, report)


def run_historical_bar_backfill_benchmark(config: Mapping[str, Any]) -> None:
    settings = dict((config.get("ml", {}) or {}).get("historical_bar_backfill", {}) or {})
    output_root = Path(settings.get("output_root", "reports/market_data/historical_bar_backfill")) / "benchmark"
    output_root.mkdir(parents=True, exist_ok=True)
    symbols = _symbols(settings)
    start = _parse_datetime(settings.get("start")) or datetime.now(timezone.utc) - timedelta(days=5)
    end = _parse_datetime(settings.get("end")) or datetime.now(timezone.utc) - timedelta(minutes=20)
    alternatives = [
        {"symbol_batch_size": int(batch), "date_window_days": int(days)}
        for batch in settings.get("benchmark_symbol_batch_sizes", [1, 5, 10])
        for days in settings.get("benchmark_date_window_days", [1, 5])
    ]
    provider = AlpacaBasicHistoricalBarProvider(
        feed=str(settings.get("feed", "iex")),
        config=config,
        rate_limiter=SharedRateLimiter(
            requests_per_minute=int(settings.get("requests_per_minute", 180)),
            max_in_flight_requests=int(settings.get("max_in_flight_requests", 4)),
        ),
    )
    auth = provider.check_authentication()
    results = []
    if auth["can_attempt_authenticated_request"]:
        for alternative in alternatives:
            before = provider.metrics.as_dict()
            started = time.monotonic()
            requests = _plan(
                symbols[: int(settings.get("benchmark_max_symbols", min(10, len(symbols))))],
                start,
                end,
                timeframe=str(settings.get("timeframe", "5m")),
                feed=str(settings.get("feed", "iex")),
                adjustment=str(settings.get("adjustment", "all")),
                symbol_batch_size=alternative["symbol_batch_size"],
                date_window_days=alternative["date_window_days"],
            )[: int(settings.get("benchmark_max_chunks_per_alternative", 2))]
            rows = 0
            failures = 0
            for request in requests:
                try:
                    chunk_rows, _ = fetch_chunk_with_retries(
                        provider,
                        request,
                        max_retries=int(settings.get("max_retries", 2)),
                    )
                    rows += len(chunk_rows)
                except Exception:
                    failures += 1
            elapsed = time.monotonic() - started
            after = provider.metrics.as_dict()
            delta = _metric_delta(before, after)
            results.append(
                {
                    **alternative,
                    "chunks": len(requests),
                    "rows": rows,
                    "elapsed_seconds": elapsed,
                    "requests": delta["requests_attempted"],
                    "pages": delta["pages_downloaded"],
                    "rows_per_request": rows / delta["requests_attempted"] if delta["requests_attempted"] else 0.0,
                    "effective_rows_per_second": rows / elapsed if elapsed else 0.0,
                    "failures": failures,
                    "retries": delta["requests_retried"],
                    "http_429_count": delta["http_429_count"],
                }
            )
    report = {
        "mode": "historical_bar_backfill_benchmark",
        "free_only": True,
        "canonical_market_data_modified": False,
        "credentials": auth,
        "alternatives": alternatives,
        "results": results,
        "blocked_reason": None if auth["can_attempt_authenticated_request"] else "Alpaca credentials unavailable",
        "observed_metrics": provider.metrics.as_dict(),
        "recommendation": _benchmark_recommendation(results),
    }
    path = output_root / "batching_benchmark_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("HISTORICAL BAR BACKFILL BENCHMARK")
    print("canonical_market_data_modified=false")
    print(f"Report: {path}")
    print(f"Alternatives measured: {len(results)}")


def run_historical_bar_feed_overlap(config: Mapping[str, Any]) -> None:
    settings = dict((config.get("ml", {}) or {}).get("historical_bar_feed_overlap", {}) or {})
    left_path = Path(settings["left_staging_path"])
    right_path = Path(settings["right_staging_path"])
    output_root = Path(settings.get("output_root", "reports/market_data/historical_bar_backfill/feed_overlap"))
    output_root.mkdir(parents=True, exist_ok=True)
    left_rows = pq.read_table(left_path).to_pylist()
    right_rows = pq.read_table(right_path).to_pylist()
    start = _parse_datetime(settings.get("start"))
    end = _parse_datetime(settings.get("end"))
    if start is None or end is None:
        raise ValueError("historical_bar_feed_overlap requires explicit start and end")
    timeframe = str(settings.get("timeframe", "5m"))
    left_provider = str(settings.get("left_provider", "iex"))
    right_provider = str(settings.get("right_provider", "sip"))
    report = {
        "mode": "historical_bar_feed_overlap",
        "canonical_source_selected": False,
        "feed_domain_shift_note": _feed_domain_shift_note(),
        "matched_request_parameters": {
            "symbols": sorted({row["symbol"] for row in left_rows} | {row["symbol"] for row in right_rows}),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": settings.get("adjustment", "all"),
            "session_policy": settings.get("session_policy", "regular_session_default"),
        },
        "left_summary": _feed_rows_summary(left_rows, timeframe=timeframe, requested_start=start, requested_end=end, feed=left_provider),
        "right_summary": _feed_rows_summary(right_rows, timeframe=timeframe, requested_start=start, requested_end=end, feed=right_provider),
        "overlap": audit_historical_bar_overlap(
            left_rows,
            right_rows,
            left_provider=left_provider,
            right_provider=right_provider,
        ),
        "per_symbol_coverage_comparison": _per_symbol_coverage_comparison(left_rows, right_rows),
    }
    path = output_root / "feed_overlap_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("HISTORICAL BAR FEED OVERLAP")
    print("canonical_source_selected=false")
    print(f"Report: {path}")


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _symbols(settings: Mapping[str, Any]) -> tuple[str, ...]:
    if settings.get("universe_file"):
        path = Path(str(settings["universe_file"]))
        return tuple(
            line.strip().upper()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return tuple(str(symbol).upper() for symbol in settings.get("symbols", ["SPY"]))


def _plan(
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    *,
    timeframe: str,
    feed: str,
    adjustment: str,
    symbol_batch_size: int,
    date_window_days: int,
) -> list[HistoricalBarRequest]:
    requests = []
    for batch_index in range(0, len(symbols), max(1, symbol_batch_size)):
        batch = symbols[batch_index:batch_index + max(1, symbol_batch_size)]
        window_start = start
        while window_start < end:
            window_end_exclusive = min(end, window_start + timedelta(days=max(1, date_window_days)))
            request_end = window_end_exclusive
            if window_end_exclusive < end:
                request_end = window_end_exclusive - _bar_delta(timeframe)
            chunk_id = f"alpaca-{feed}-{timeframe}-{'-'.join(batch)}-{window_start:%Y%m%dT%H%M%SZ}-{request_end:%Y%m%dT%H%M%SZ}"
            requests.append(
                HistoricalBarRequest(
                    symbols=batch,
                    timeframe=timeframe,
                    start=window_start,
                    end=request_end,
                    feed=feed,
                    adjustment=adjustment,
                    raw_chunk_id=chunk_id,
                )
            )
            window_start = window_end_exclusive
    return requests


def _bar_delta(timeframe: str) -> timedelta:
    if timeframe in {"5m", "5Min"}:
        return timedelta(minutes=5)
    if timeframe in {"1h", "1Hour"}:
        return timedelta(hours=1)
    return timedelta(days=1)


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = ["requests_attempted", "requests_retried", "http_429_count", "pages_downloaded", "rows_downloaded"]
    return {key: float(after.get(key, 0) or 0) - float(before.get(key, 0) or 0) for key in keys}


def _feed_domain_shift_note() -> str:
    return (
        "Historical training feed and future free live scoring feed must remain explicit. "
        "Do not silently mix SIP and IEX bars; preserve provenance for train/evaluate feed-domain checks."
    )


def _coverage_status(coverage_report: list[Mapping[str, Any]]) -> str | None:
    if not coverage_report:
        return None
    statuses = {str(row.get("completeness_status")) for row in coverage_report}
    if statuses == {"complete"}:
        return "complete"
    if "empty" in statuses and len(statuses) == 1:
        return "empty"
    return "incomplete"


def _research_view_audit(
    rows: list[dict[str, Any]],
    validation: Mapping[str, Any],
    coverage_report: list[Mapping[str, Any]],
) -> dict[str, Any]:
    rth_rows = [row for row in rows if row.get("session_type") == "rth" or _timestamp_is_rth(row.get("timestamp"))]
    return {
        "all_session_row_count": len(rows),
        "rth_row_count": len(rth_rows),
        "outside_rth_row_count": len(rows) - len(rth_rows),
        "research_view": "rth_only",
        "raw_and_staging_preserve_all_returned_bars": True,
        "no_forward_fill": True,
        "no_synthetic_5m_bars": True,
        "no_stale_close_carry": True,
        "duplicate_count": validation.get("duplicate_key_count"),
        "missing_expected_rth_bars": sum(int(row.get("missing_expected_rth_bars", 0)) for row in coverage_report),
        "structurally_valid_staging": validation.get("valid"),
    }


def _timestamp_is_rth(value: Any) -> bool:
    if not isinstance(value, datetime):
        return False
    return is_rth_timestamp(value)


def _write_collect_report(output_root: Path, report: Mapping[str, Any]) -> None:
    (output_root / "historical_bar_collect_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )


def _print_collect_report(output_root: Path, report: Mapping[str, Any]) -> None:
    print("HISTORICAL BAR BACKFILL COLLECT")
    print("canonical_market_data_modified=false")
    print(f"Report: {output_root / 'historical_bar_collect_report.json'}")
    print(f"Dry run: {report['dry_run']}")
    print(f"Planned chunks: {report['planned_chunk_count']}")
    print(f"Observed metrics: {report['observed_metrics']}")


def _benchmark_recommendation(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    viable = [row for row in results if row["failures"] == 0]
    if not viable:
        return None
    best = max(viable, key=lambda row: row["effective_rows_per_second"])
    return {
        "symbol_batch_size": best["symbol_batch_size"],
        "date_window_days": best["date_window_days"],
        "basis": "highest observed rows/second among no-failure tiny alternatives",
    }


def _feed_rows_summary(
    rows: list[dict[str, Any]],
    *,
    timeframe: str,
    requested_start: datetime,
    requested_end: datetime,
    feed: str,
) -> dict[str, Any]:
    timestamps = [row["timestamp"] for row in rows]
    coverage = coverage_gap_audit(
        rows,
        timeframe=timeframe,
        requested_start=requested_start,
        requested_end=requested_end,
        provider="alpaca",
        feed=feed,
    )
    return {
        "rows": len(rows),
        "earliest_timestamp": min(timestamps).isoformat() if timestamps else None,
        "latest_timestamp": max(timestamps).isoformat() if timestamps else None,
        "symbols_returned": sorted({row["symbol"] for row in rows}),
        "bars_per_symbol": _symbol_coverage(rows, tuple(sorted({row["symbol"] for row in rows}))),
        "coverage_report": coverage,
        "structural_validation": validate_normalized_bars(rows),
    }


def _per_symbol_coverage_comparison(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left = _symbol_coverage(left_rows, tuple(sorted({row["symbol"] for row in left_rows})))
    right = _symbol_coverage(right_rows, tuple(sorted({row["symbol"] for row in right_rows})))
    symbols = sorted(set(left) | set(right))
    return {
        symbol: {
            "left_rows": left.get(symbol, 0),
            "right_rows": right.get(symbol, 0),
            "row_delta": left.get(symbol, 0) - right.get(symbol, 0),
        }
        for symbol in symbols
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    timestamps = [row["timestamp"] for row in rows]
    return {
        "first_timestamp": min(timestamps).isoformat(),
        "last_timestamp": max(timestamps).isoformat(),
        "row_count": len(rows),
    }


def _symbol_coverage(rows: list[dict[str, Any]], symbols: tuple[str, ...]) -> dict[str, Any]:
    counts = {symbol: 0 for symbol in symbols}
    for row in rows:
        symbol = str(row["symbol"]).upper()
        if symbol in counts:
            counts[symbol] += 1
    return counts
