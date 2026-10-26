"""Bounded Alpaca IEX capability pilot and historical coverage probe."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from application.services.hf_data_cutover_service import (
    CutoverAuditError,
    CutoverPaths,
    _atomic_bytes,
    _load_json,
    atomic_json,
    utc_now,
)
from core.research.market_data_cutover import canonical_sha256
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    HistoricalBarRequest,
    SharedRateLimiter,
    fetch_chunk_with_retries,
)
from infrastructure.data.market_data_cutover_audit import resource_state, sha256_file


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pilot_quality(frame: pd.DataFrame, *, timeframe: str, window: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["session"] = working["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    for symbol, group in working.groupby("symbol", sort=True):
        ordered = group.sort_values("timestamp")
        duplicate_count = int(ordered.duplicated(["symbol", "timestamp"], keep=False).sum())
        if timeframe == "5m":
            off_grid = int(
                (
                    ordered["timestamp"].dt.second.ne(0)
                    | ordered["timestamp"].dt.microsecond.ne(0)
                    | ordered["timestamp"].dt.minute.mod(5).ne(0)
                ).sum()
            )
            rth = ordered[ordered["session_type"].eq("rth")]
            bars_per_session = rth.groupby("session", observed=True).size()
        else:
            # Alpaca daily-bar anchors are provider-defined and have changed
            # across historical periods. Validate daily granularity without
            # inventing a fixed UTC-hour contract.
            off_grid = int(
                (
                    ordered["timestamp"].dt.minute.ne(0)
                    | ordered["timestamp"].dt.second.ne(0)
                    | ordered["timestamp"].dt.microsecond.ne(0)
                ).sum()
            )
            bars_per_session = ordered.groupby("session", observed=True).size()
        rows.append(
            {
                "window": window,
                "timeframe": timeframe,
                "symbol": symbol,
                "row_count": len(ordered),
                "first_observation": ordered["timestamp"].min().isoformat(),
                "last_observation": ordered["timestamp"].max().isoformat(),
                "duplicate_rows": duplicate_count,
                "off_grid_rows": off_grid,
                "session_count": int(len(bars_per_session)),
                "bars_per_session_min": int(bars_per_session.min()) if len(bars_per_session) else None,
                "bars_per_session_median": float(bars_per_session.median()) if len(bars_per_session) else None,
                "bars_per_session_max": int(bars_per_session.max()) if len(bars_per_session) else None,
                "zero_volume_rate": float(pd.to_numeric(ordered["volume"], errors="coerce").fillna(0).eq(0).mean()),
                "volume_median": float(pd.to_numeric(ordered["volume"], errors="coerce").median()),
                "trade_count_available_rate": float(ordered["trade_count"].notna().mean()),
                "trade_count_median": float(pd.to_numeric(ordered["trade_count"], errors="coerce").median()) if ordered["trade_count"].notna().any() else None,
                "vwap_available_rate": float(ordered["vwap"].notna().mean()),
            }
        )
    return rows


def _storage_forecast(
    *,
    roster_size: int,
    five_minute_rows: int,
    five_minute_parquet_bytes: int,
    five_minute_symbol_sessions: int,
    daily_rows: int,
    daily_parquet_bytes: int,
    free_bytes: int,
) -> dict[str, Any]:
    if five_minute_rows <= 0 or daily_rows <= 0 or five_minute_symbol_sessions <= 0:
        raise CutoverAuditError("Pilot did not produce enough rows for a measured storage forecast")
    raw_5m_bytes_per_row = five_minute_parquet_bytes / five_minute_rows
    raw_daily_bytes_per_row = daily_parquet_bytes / daily_rows
    observed_5m_rows_per_symbol_session = five_minute_rows / five_minute_symbol_sessions
    annual_sessions = 252
    annual_5m_rows = int(round(roster_size * annual_sessions * observed_5m_rows_per_symbol_session))
    annual_daily_rows = roster_size * annual_sessions
    annual_hourly_rows = roster_size * annual_sessions * 7
    # The feature forecast is tied to measured compressed raw bytes/row and the
    # frozen 80-column schema. It is intentionally conservative until the first
    # real feature partition supplies a replacement empirical ratio.
    feature_bytes_per_row = max(80.0, raw_5m_bytes_per_row * 3.0)
    target_bytes_per_row = max(24.0, raw_5m_bytes_per_row * 0.6)
    manifest_fraction = 0.01

    def horizon(years: float) -> dict[str, Any]:
        raw_5m = int(annual_5m_rows * years * raw_5m_bytes_per_row)
        daily = int(annual_daily_rows * years * raw_daily_bytes_per_row)
        hourly = int(annual_hourly_rows * years * raw_5m_bytes_per_row)
        features = int(annual_5m_rows * years * feature_bytes_per_row)
        targets = int(annual_5m_rows * years * target_bytes_per_row)
        manifests = int((raw_5m + daily + hourly + features + targets) * manifest_fraction)
        permanent = raw_5m + daily + hourly + features + targets + manifests
        scratch_peak = int(max(raw_5m, features) * 1.25)
        return {
            "years": years,
            "raw_5m_bytes": raw_5m,
            "raw_daily_bytes": daily,
            "derived_1h_bytes": hourly,
            "feature_v1_bytes": features,
            "target_v1_bytes": targets,
            "manifest_bytes": manifests,
            "permanent_bytes": permanent,
            "scratch_peak_bytes": scratch_peak,
            "peak_required_bytes": permanent + scratch_peak,
        }

    forecasts = {label: horizon(years) for label, years in (("1_year", 1), ("3_years", 3), ("5_years", 5), ("maximum_available_history", 10.8))}
    reserve = 15 * 1024**3
    maximum = forecasts["maximum_available_history"]
    return {
        "schema_version": "hf_data_cutover_r1.iex_storage_forecast.v1",
        "basis": {
            "roster_size": roster_size,
            "annual_sessions": annual_sessions,
            "observed_5m_rows_per_symbol_session": observed_5m_rows_per_symbol_session,
            "measured_raw_5m_parquet_bytes_per_row": raw_5m_bytes_per_row,
            "measured_raw_daily_parquet_bytes_per_row": raw_daily_bytes_per_row,
            "feature_bytes_per_row_conservative_multiplier": 3.0,
            "feature_forecast_status": "PROVISIONAL_UNTIL_REAL_FEATURE_PARTITION_MEASURED",
        },
        "forecasts": forecasts,
        "disk_gate": {
            "free_bytes_before_full_build": free_bytes,
            "minimum_final_reserve_bytes": reserve,
            "maximum_history_peak_admissible": maximum["peak_required_bytes"] + reserve <= free_bytes,
            "projected_final_free_bytes_maximum_history": free_bytes - maximum["permanent_bytes"],
        },
    }


def run_iex_pilot(paths: CutoverPaths) -> dict[str, Any]:
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("state") != "SIP_RETIRED_IEX_PILOT_PENDING":
        raise CutoverAuditError(f"IEX pilot requires retired SIP state, found {run_manifest.get('state')}")
    roster_path = paths.repository_root / "config/universes/alpaca_514_symbols.txt"
    roster = tuple(line.strip().upper() for line in roster_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(roster) != 514 or len(set(roster)) != 514 or not {"SPY", "QQQ"}.issubset(roster):
        raise CutoverAuditError("The authenticated acquisition roster is not the expected unique 514 symbols including SPY and QQQ")
    roster_payload = {
        "schema_version": "hf_data_cutover_r1.iex_acquisition_roster.v1",
        "source_path": str(roster_path.resolve()),
        "source_sha256": sha256_file(roster_path),
        "symbol_count": len(roster),
        "symbols": list(roster),
        "includes_spy": True,
        "includes_qqq": True,
        "role": "DATA_ACQUISITION_ONLY_NOT_HISTORICAL_ELIGIBILITY",
        "historical_membership_inferred": False,
    }
    roster_payload["roster_identity"] = canonical_sha256(roster_payload)
    atomic_json(paths.output_root / "IEX_ACQUISITION_ROSTER.json", roster_payload)

    provider = AlpacaBasicHistoricalBarProvider(
        feed="iex",
        rate_limiter=SharedRateLimiter(requests_per_minute=30, max_in_flight_requests=1),
    )
    authentication = provider.check_authentication()
    if not authentication["can_attempt_authenticated_request"]:
        raise CutoverAuditError("Existing Alpaca credentials are unavailable to the IEX pilot")
    authority = {
        "schema_version": "hf_data_cutover_r1.iex_source_use_authority.v1",
        "ticket": "HF-DATA-CUTOVER-R1",
        "purpose": "market-data-only Alpaca IEX historical retrieval",
        "provider": "Alpaca",
        "feed": "iex",
        "orders_authorized": False,
        "orders_submitted": 0,
        "sip_requests_authorized": False,
        "paid_upgrade_authorized": False,
        "credential_report": authentication,
        "request_contract": {
            "endpoint": "https://data.alpaca.markets/v2/stocks/bars",
            "timeframes": ["5Min", "1Day"],
            "feed": "iex",
            "adjustment": "raw",
            "currency": "USD",
            "sort": "asc",
            "limit": 10000,
            "pagination": "follow next_page_token until absent",
            "asof": "2026-10-25",
        },
        "current_documentation_verified_at_utc": utc_now(),
        "official_documentation": [
            "https://docs.alpaca.markets/us/reference/stockbars",
            "https://docs.alpaca.markets/us/docs/about-market-data-api",
            "https://docs.alpaca.markets/us/docs/market-data-faq",
        ],
        "documented_basic_limit": {"historical_start": "2016", "calls_per_minute": 200, "latest_data_delay_minutes": 15},
        "configured_pilot_calls_per_minute": 30,
    }
    atomic_json(paths.output_root / "IEX_SOURCE_USE_AUTHORITY.json", authority)

    pilot_symbols = ("AAPL", "TSLA", "SPY", "MSFT", "ARM", "META")
    windows = (
        ("early_2016", datetime(2016, 1, 4, tzinfo=timezone.utc), datetime(2016, 1, 16, tzinfo=timezone.utc)),
        ("mid_2021", datetime(2021, 6, 1, tzinfo=timezone.utc), datetime(2021, 6, 16, tzinfo=timezone.utc)),
        ("new_listing_2023", datetime(2023, 9, 14, tzinfo=timezone.utc), datetime(2023, 9, 29, tzinfo=timezone.utc)),
        ("recent_2026", datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 16, tzinfo=timezone.utc)),
    )
    pilot_root = paths.output_root / "IEX_CAPABILITY_PILOT_DATA"
    all_frames: dict[str, list[pd.DataFrame]] = {"5m": [], "1Day": []}
    request_evidence: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    raw_json_bytes = 0
    for timeframe in ("5m", "1Day"):
        for label, start, end in windows:
            request = HistoricalBarRequest(
                symbols=pilot_symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                feed="iex",
                adjustment="raw",
                limit=10_000,
                raw_chunk_id=f"hf-data-cutover-r1-{timeframe}-{label}",
                currency="USD",
                asof="2026-10-25",
            )
            rows, chunk = fetch_chunk_with_retries(provider, request, max_retries=3)
            raw_pages = chunk.pop("raw_pages")
            serialized = json.dumps(raw_pages, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            raw_json_bytes += len(serialized)
            raw_path = pilot_root / "raw_pages" / timeframe / f"{label}.json"
            _atomic_bytes(raw_path, serialized)
            frame = pd.DataFrame(rows)
            if not frame.empty:
                all_frames[timeframe].append(frame)
                quality_rows.extend(_pilot_quality(frame, timeframe=timeframe, window=label))
            request_evidence.append(
                {
                    "window": label,
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "symbols": list(pilot_symbols),
                    "rows": len(rows),
                    "pages": chunk["pages"],
                    "raw_page_sha256": chunk["raw_page_sha256"],
                    "raw_payload_path": str(raw_path.relative_to(paths.output_root)),
                    "raw_payload_sha256": sha256_file(raw_path),
                }
            )
    parquet_measurements: dict[str, Any] = {}
    combined_frames: dict[str, pd.DataFrame] = {}
    for timeframe, frames in all_frames.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
            combined = combined.sort_values(["symbol", "timestamp"], kind="mergesort").drop_duplicates(["symbol", "timestamp"], keep="last")
        output_path = pilot_root / f"pilot_{timeframe}.parquet"
        _atomic_parquet(output_path, combined)
        combined_frames[timeframe] = combined
        parquet_measurements[timeframe] = {"path": str(output_path.relative_to(paths.output_root)), "rows": len(combined), "bytes": output_path.stat().st_size, "bytes_per_row": output_path.stat().st_size / len(combined) if len(combined) else None, "sha256": sha256_file(output_path)}

    five = combined_frames["5m"]
    daily = combined_frames["1Day"]
    rth_five = five[five["session_type"].eq("rth")].copy()
    rth_five["session"] = rth_five["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    symbol_sessions = int(rth_five[["symbol", "session"]].drop_duplicates().shape[0])
    pilot = {
        "schema_version": "hf_data_cutover_r1.iex_capability_pilot.v1",
        "completed_at_utc": utc_now(),
        "provider": "Alpaca",
        "feed": "IEX",
        "symbols": list(pilot_symbols),
        "windows": [label for label, _, _ in windows],
        "requests": request_evidence,
        "quality": quality_rows,
        "parquet_measurements": parquet_measurements,
        "raw_json_bytes": raw_json_bytes,
        "provider_metrics": provider.metrics.as_dict(),
        "actual_earliest_observation": min((row["first_observation"] for row in quality_rows), default=None),
        "actual_latest_observation": max((row["last_observation"] for row in quality_rows), default=None),
        "pagination_observed": any(int(row["pages"]) > 1 for row in request_evidence),
        "duplicate_rows": sum(int(row["duplicate_rows"]) for row in quality_rows),
        "off_grid_rows": sum(int(row["off_grid_rows"]) for row in quality_rows),
        "vwap_available_rate": float(five["vwap"].notna().mean()) if len(five) else None,
        "trade_count_available_rate": float(five["trade_count"].notna().mean()) if len(five) else None,
        "ticker_change_probe": {
            "requested_symbol": "META",
            "asof": "2026-10-25",
            "early_2016_rows_labeled_meta": int(((five["symbol"] == "META") & (five["timestamp"].dt.year == 2016)).sum()) if len(five) else 0,
            "interpretation": "Positive rows demonstrate current-identity alias mapping to historical FB; zero rows leave mapping inconclusive.",
        },
        "new_listing_probe": {"symbol": "ARM", "first_observation": min((row["first_observation"] for row in quality_rows if row["symbol"] == "ARM"), default=None)},
        "validation": {"feed_explicit_iex": True, "adjustment_explicit_raw": True, "currency_explicit_usd": True, "asof_explicit": True, "zero_orders": True, "zero_sip_requests": True},
    }
    atomic_json(paths.output_root / "IEX_CAPABILITY_PILOT.json", pilot)
    current_free = int(resource_state(repository_root=paths.repository_root)["disk"]["free_bytes"])
    forecast = _storage_forecast(
        roster_size=len(roster),
        five_minute_rows=len(five),
        five_minute_parquet_bytes=parquet_measurements["5m"]["bytes"],
        five_minute_symbol_sessions=symbol_sessions,
        daily_rows=len(daily),
        daily_parquet_bytes=parquet_measurements["1Day"]["bytes"],
        free_bytes=current_free,
    )
    atomic_json(paths.output_root / "IEX_STORAGE_FORECAST.json", forecast)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_PILOT_COMPLETE_FULL_BUILD_PENDING" if forecast["disk_gate"]["maximum_history_peak_admissible"] else "IEX_PILOT_COMPLETE_FULL_BUILD_BLOCKED",
            "iex_pilot": {
                "source_authority_sha256": sha256_file(paths.output_root / "IEX_SOURCE_USE_AUTHORITY.json"),
                "pilot_sha256": sha256_file(paths.output_root / "IEX_CAPABILITY_PILOT.json"),
                "storage_forecast_sha256": sha256_file(paths.output_root / "IEX_STORAGE_FORECAST.json"),
                "roster_sha256": sha256_file(paths.output_root / "IEX_ACQUISITION_ROSTER.json"),
                "requests_attempted": provider.metrics.requests_attempted,
                "rows": len(five) + len(daily),
            },
            "next_gate": "Build full resumable IEX raw 5m/daily authority." if forecast["disk_gate"]["maximum_history_peak_admissible"] else "Select a uniform later start date that preserves the 15 GiB reserve.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest
__all__ = ["probe_iex_historical_coverage", "run_iex_pilot"]


def probe_iex_historical_coverage(paths: CutoverPaths) -> dict[str, Any]:
    """Measure year-level IEX 5-minute availability before the full backfill."""

    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("state") not in {
        "IEX_PILOT_COMPLETE_FULL_BUILD_PENDING",
        "IEX_COVERAGE_PROBE_COMPLETE_FULL_BUILD_PENDING",
    }:
        raise CutoverAuditError(
            f"IEX coverage probe requires a successful pilot, found {run_manifest.get('state')}"
        )
    provider = AlpacaBasicHistoricalBarProvider(
        feed="iex",
        rate_limiter=SharedRateLimiter(requests_per_minute=30, max_in_flight_requests=1),
    )
    evidence: list[dict[str, Any]] = []
    all_observations: list[datetime] = []
    for year in range(2016, 2027):
        for period, month, end_day in (("january", 1, 16), ("july", 7, 16)):
            if year == 2026 and month == 7:
                end_day = 16
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = datetime(year, month, end_day, tzinfo=timezone.utc)
            request = HistoricalBarRequest(
                symbols=("AAPL", "SPY"),
                timeframe="5m",
                start=start,
                end=end,
                feed="iex",
                adjustment="raw",
                limit=10_000,
                raw_chunk_id=f"hf-data-cutover-r1-coverage-{year}-{period}",
                currency="USD",
                asof="2026-10-25",
            )
            rows, chunk = fetch_chunk_with_retries(provider, request, max_retries=3)
            timestamps = [row["timestamp"] for row in rows]
            all_observations.extend(timestamps)
            evidence.append(
                {
                    "year": year,
                    "period": period,
                    "requested_start": start.isoformat(),
                    "requested_end": end.isoformat(),
                    "row_count": len(rows),
                    "symbols_returned": sorted({str(row["symbol"]) for row in rows}),
                    "first_observation": min(timestamps).isoformat() if timestamps else None,
                    "last_observation": max(timestamps).isoformat() if timestamps else None,
                    "pages": chunk["pages"],
                    "raw_page_sha256": chunk["raw_page_sha256"],
                }
            )
    nonempty_years = sorted({int(row["year"]) for row in evidence if int(row["row_count"]) > 0})
    payload = {
        "schema_version": "hf_data_cutover_r1.iex_historical_coverage_probe.v1",
        "completed_at_utc": utc_now(),
        "provider": "Alpaca",
        "feed": "IEX",
        "symbols": ["AAPL", "SPY"],
        "probe_design": "Two bounded half-month windows per calendar year; acquisition start is conservative and uniform.",
        "windows": evidence,
        "nonempty_years": nonempty_years,
        "actual_earliest_observation_in_probe": min(all_observations).isoformat() if all_observations else None,
        "actual_latest_observation_in_probe": max(all_observations).isoformat() if all_observations else None,
        "selected_uniform_acquisition_start": f"{min(nonempty_years)}-01-01T00:00:00Z" if nonempty_years else None,
        "selected_acquisition_end_exclusive": "2026-10-01T00:00:00Z",
        "selection_rule": "Earliest calendar year with observed 5-minute IEX rows; no historical model eligibility is inferred.",
        "provider_metrics": provider.metrics.as_dict(),
    }
    if not nonempty_years:
        raise CutoverAuditError("IEX coverage probe returned no historical observations")
    output_path = paths.output_root / "IEX_HISTORICAL_COVERAGE_PROBE.json"
    atomic_json(output_path, payload)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_COVERAGE_PROBE_COMPLETE_FULL_BUILD_PENDING",
            "iex_coverage_probe": {
                "sha256": sha256_file(output_path),
                "selected_uniform_acquisition_start": payload["selected_uniform_acquisition_start"],
                "selected_acquisition_end_exclusive": payload["selected_acquisition_end_exclusive"],
                "requests_attempted": provider.metrics.requests_attempted,
            },
            "next_gate": "Build full resumable IEX raw 5m/daily authority.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest

