from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import pyarrow as pa

from core.research.ml.reference.canonical_assets import (
    alpaca_provider_symbol,
    read_aliases_csv,
    read_assets_csv,
)


REPORT_SCHEMA_VERSION = "alpaca_daily_extension_preflight.v1"
DEFAULT_REPORT_ROOT = Path("reports/data_lineage/alpaca_daily_extension_preflight")
DEFAULT_DAILY_ARCHIVE_ROOT = Path("data/processed/alpaca/symbol_bars/sip/1d")
DEFAULT_STOOQ_ROOT = Path("data/processed/stooq_parquet")
DEFAULT_ASSET_REGISTRY = Path("data/reference/assets/canonical_asset_registry.csv")
DEFAULT_ALIAS_REGISTRY = Path("data/reference/assets/provider_symbol_aliases.csv")


@dataclass(frozen=True)
class PreflightPlan:
    latest_spy_session: str | None
    median_source_endpoint: str | None
    recommended_overlap_start: str | None
    production_end_semantics: str
    estimated_request_count: int | None


def run_preflight(
    *,
    report_root: Path = DEFAULT_REPORT_ROOT,
    stooq_root: Path = DEFAULT_STOOQ_ROOT,
    asset_registry: Path = DEFAULT_ASSET_REGISTRY,
    alias_registry: Path = DEFAULT_ALIAS_REGISTRY,
    daily_archive_root: Path = DEFAULT_DAILY_ARCHIVE_ROOT,
    smoke_output_root: Path = Path("reports/market_data/historical_bar_backfill/daily_sip_smoke_abcb"),
    smoke_archive_root: Path = Path("data/processed/alpaca/symbol_bars/sip/1d_smoke"),
    smoke_symbols: Sequence[str] = ("AAPL", "SPY", "BRK-B", "ABCB"),
    full_universe_symbol_batch_size: int = 50,
    date_window_days: int = 31,
    requests_per_minute: int = 180,
    dry_run: bool = False,
) -> dict[str, Any]:
    if requests_per_minute > 180:
        raise ValueError("requests_per_minute must be <= 180 for Alpaca daily extension preflight")
    assets = read_assets_csv(asset_registry)
    aliases = read_aliases_csv(alias_registry)
    collection_assets = [asset for asset in assets if asset.collection_universe_514 and asset.is_active]
    stooq_rows = stooq_freshness(stooq_root)
    universe_rows = classify_universe(collection_assets, aliases, stooq_rows=stooq_rows)
    plan = production_plan(
        stooq_rows,
        universe_rows,
        stooq_root=stooq_root,
        daily_archive_root=daily_archive_root,
        smoke_output_root=smoke_output_root,
        full_universe_symbol_batch_size=full_universe_symbol_batch_size,
        date_window_days=date_window_days,
        requests_per_minute=requests_per_minute,
    )
    smoke_rows = smoke_coverage_rows(smoke_symbols, universe_rows, stooq_rows)
    alpaca_smoke_rows = read_alpaca_daily_archive(smoke_archive_root, smoke_symbols)
    if alpaca_smoke_rows:
        smoke_start = min(str(row["session_date"]) for row in alpaca_smoke_rows)
        smoke_end = max(str(row["session_date"]) for row in alpaca_smoke_rows)
    else:
        smoke_start = plan.recommended_overlap_start
        smoke_end = plan.latest_spy_session
    reconciliation_rows, reconciliation = reconcile_smoke_rows(
        read_stooq_daily_rows(stooq_root, smoke_symbols, smoke_start, smoke_end),
        alpaca_smoke_rows,
    )
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "direct_daily_supported": True,
        "alpaca_timeframe": "1Day",
        "provider": "alpaca",
        "feed": "sip",
        "adjustment": "all",
        "canonical_market_data_writes_enabled": False,
        "stooq_modified": False,
        "five_minute_archive_modified": False,
        "daily_archive_root": str(daily_archive_root),
        "smoke_output_root": str(smoke_output_root),
        "smoke_archive_root": str(smoke_archive_root),
        "smoke_archive_row_count": len(alpaca_smoke_rows),
        "source_freshness_rows": len(stooq_rows),
        "universe_rows": len(universe_rows),
        "valid_alpaca_mappings": sum(1 for row in universe_rows if row["alpaca_mapping_status"] == "mapped"),
        "smoke_symbols": list(smoke_symbols),
        "plan": plan.__dict__,
        "reports": {
            "source_freshness_csv": str(report_root / "source_freshness.csv"),
            "universe_classification_csv": str(report_root / "universe_classification.csv"),
            "smoke_collection_report_json": str(report_root / "smoke_collection_report.json"),
            "smoke_coverage_csv": str(report_root / "smoke_coverage.csv"),
            "smoke_reconciliation_csv": str(report_root / "smoke_reconciliation.csv"),
            "smoke_reconciliation_json": str(report_root / "smoke_reconciliation.json"),
            "production_plan_json": str(report_root / "production_plan.json"),
        },
    }
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_csv(report_root / "source_freshness.csv", stooq_rows, SOURCE_FRESHNESS_FIELDS)
        _write_csv(report_root / "universe_classification.csv", universe_rows, UNIVERSE_FIELDS)
        _write_csv(report_root / "smoke_coverage.csv", smoke_rows, SMOKE_COVERAGE_FIELDS)
        _write_csv(report_root / "smoke_reconciliation.csv", reconciliation_rows, RECONCILIATION_FIELDS)
        _write_json(report_root / "smoke_reconciliation.json", reconciliation)
        _write_json(report_root / "production_plan.json", {**payload, "manual_command": manual_production_command()})
        _write_json(report_root / "smoke_collection_report.json", smoke_collection_report(smoke_symbols, plan, smoke_output_root))
    return payload


SOURCE_FRESHNESS_FIELDS = (
    "symbol",
    "latest_session",
    "earliest_session",
    "row_count",
    "source",
    "endpoint_status",
)
UNIVERSE_FIELDS = (
    "asset_id",
    "canonical_symbol",
    "security_type",
    "asset_role",
    "alpaca_provider_symbol",
    "alpaca_mapping_status",
    "mapping_reason",
    "stooq_available",
    "latest_stooq_session",
)
SMOKE_COVERAGE_FIELDS = (
    "canonical_symbol",
    "alpaca_provider_symbol",
    "mapping_status",
    "latest_stooq_session",
    "included_reason",
)
RECONCILIATION_FIELDS = (
    "session_date",
    "symbol",
    "stooq_close",
    "alpaca_close",
    "relative_close_difference",
    "abs_relative_close_difference",
    "stooq_return",
    "alpaca_return",
    "return_difference",
    "abs_return_difference",
    "stooq_volume",
    "alpaca_volume",
    "volume_difference",
    "relative_volume_difference",
    "missing_stooq",
    "missing_alpaca",
    "possible_adjustment_discrepancy",
    "possible_corporate_action",
    "classification",
)


def stooq_freshness(stooq_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(stooq_root.glob("*.parquet")):
        try:
            table = pq.read_table(path, columns=["timestamp", "source"])
            payload = table.to_pylist()
        except Exception as exc:
            rows.append(
                {
                    "symbol": path.stem.upper(),
                    "latest_session": "",
                    "earliest_session": "",
                    "row_count": 0,
                    "source": "",
                    "endpoint_status": f"unreadable:{type(exc).__name__}",
                }
            )
            continue
        dates = sorted({_session_date(row.get("timestamp")) for row in payload if row.get("timestamp") is not None})
        sources = sorted({str(row.get("source") or "") for row in payload if row.get("source")})
        rows.append(
            {
                "symbol": path.stem.upper(),
                "latest_session": dates[-1].isoformat() if dates else "",
                "earliest_session": dates[0].isoformat() if dates else "",
                "row_count": len(payload),
                "source": "|".join(sources),
                "endpoint_status": "ok" if dates else "empty",
            }
        )
    latest_dates = sorted(row["latest_session"] for row in rows if row["latest_session"])
    median = latest_dates[len(latest_dates) // 2] if latest_dates else ""
    for row in rows:
        row["endpoint_status"] = "before_median_endpoint" if row["latest_session"] and row["latest_session"] < median else row["endpoint_status"]
    return rows


def classify_universe(assets: Sequence[Any], aliases: Sequence[Any], *, stooq_rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    alias_by_asset = {
        alias.asset_id: alias
        for alias in aliases
        if alias.provider == "alpaca" and alias.is_primary
    }
    stooq_latest = {row["symbol"]: row["latest_session"] for row in (stooq_rows if stooq_rows is not None else stooq_freshness(DEFAULT_STOOQ_ROOT))}
    rows = []
    for asset in sorted(assets, key=lambda row: row.canonical_symbol):
        alias = alias_by_asset.get(asset.asset_id)
        provider_symbol = alias.provider_symbol if alias else alpaca_provider_symbol(asset.canonical_symbol)
        rows.append(
            {
                "asset_id": asset.asset_id,
                "canonical_symbol": asset.canonical_symbol,
                "security_type": asset.security_type,
                "asset_role": asset_role(asset.canonical_symbol, asset.security_type),
                "alpaca_provider_symbol": provider_symbol,
                "alpaca_mapping_status": "mapped" if alias else "missing",
                "mapping_reason": alias.mapping_reason if alias else "",
                "stooq_available": str(asset.canonical_symbol in stooq_latest).lower(),
                "latest_stooq_session": stooq_latest.get(asset.canonical_symbol, ""),
            }
        )
    return rows


def production_plan(
    stooq_rows: Sequence[Mapping[str, Any]],
    universe_rows: Sequence[Mapping[str, Any]],
    *,
    stooq_root: Path,
    daily_archive_root: Path,
    smoke_output_root: Path,
    full_universe_symbol_batch_size: int,
    date_window_days: int,
    requests_per_minute: int,
) -> PreflightPlan:
    latest_spy = next((row["latest_session"] for row in stooq_rows if row["symbol"] == "SPY"), None)
    endpoints = sorted(row["latest_session"] for row in stooq_rows if row.get("latest_session"))
    median = endpoints[len(endpoints) // 2] if endpoints else None
    overlap_start = recommended_overlap_start(stooq_rows, anchor=latest_spy or median, stooq_root=stooq_root)
    mapped = sum(1 for row in universe_rows if row.get("alpaca_mapping_status") == "mapped")
    request_count = estimate_request_count(mapped, overlap_start, latest_completed_session_semantic_date(), full_universe_symbol_batch_size, date_window_days) if overlap_start else None
    return PreflightPlan(
        latest_spy_session=latest_spy,
        median_source_endpoint=median,
        recommended_overlap_start=overlap_start,
        production_end_semantics="latest completed US equity session at command runtime; config may be refreshed by rerunning preflight",
        estimated_request_count=request_count,
    )


def recommended_overlap_start(stooq_rows: Sequence[Mapping[str, Any]], *, anchor: str | None, stooq_root: Path = DEFAULT_STOOQ_ROOT) -> str | None:
    if not anchor:
        return None
    spy_dates = _stooq_symbol_dates(stooq_root / "SPY.parquet")
    if not spy_dates:
        latest = date.fromisoformat(anchor)
        return (latest - timedelta(days=90)).isoformat()
    latest = date.fromisoformat(anchor)
    prior = [value for value in spy_dates if value <= latest]
    if len(prior) >= 61:
        return prior[-61].isoformat()
    return prior[0].isoformat() if prior else None


def latest_completed_session_semantic_date(now: datetime | None = None) -> str:
    current = (now or datetime.now(timezone.utc)).date()
    day = current - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.isoformat()


def estimate_request_count(symbol_count: int, start: str, end: str, symbol_batch_size: int, date_window_days: int) -> int:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    symbol_batches = (symbol_count + max(1, symbol_batch_size) - 1) // max(1, symbol_batch_size)
    windows = max(1, ((end_date - start_date).days + max(1, date_window_days) - 1) // max(1, date_window_days))
    return symbol_batches * windows


def smoke_collection_report(symbols: Sequence[str], plan: PreflightPlan, output_root: Path) -> dict[str, Any]:
    existing = output_root / "historical_bar_collect_report.json"
    if existing.exists():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        return {
            "mode": "bounded_preflight_actual_collect_report",
            "live_collection_executed_by_preflight": False,
            "actual_collect_report_path": str(existing),
            "direct_daily_supported": True,
            "timeframe": "1Day",
            "feed": "sip",
            "adjustment": "all",
            "planned_chunk_count": payload.get("planned_chunk_count"),
            "dry_run": payload.get("dry_run"),
            "blocked_reason": payload.get("blocked_reason"),
            "credentials": payload.get("credentials"),
            "observed_metrics": payload.get("observed_metrics"),
            "staging_path": payload.get("staging_path"),
            "staging_deferred_reason": payload.get("staging_deferred_reason"),
            "bounded_consolidation_performed": payload.get("bounded_consolidation_performed"),
            "daily_archive_report": payload.get("daily_archive_report"),
            "api_collection_succeeded": any(row.get("status") == "completed" for row in payload.get("chunk_results", [])),
            "raw_chunk_completed_or_reused": any(row.get("status") in {"completed", "skipped_completed"} for row in payload.get("chunk_results", [])),
            "staging_completed": bool(payload.get("staging_path")),
            "daily_archive_completed": bool(payload.get("daily_archive_report")),
            "production_validated": payload.get("production_validated", False),
            "canonical_market_data_modified": payload.get("canonical_market_data_modified"),
            "raw_write_enabled": payload.get("raw_write_enabled"),
            "staging_write_enabled": payload.get("staging_write_enabled"),
        }
    start = plan.recommended_overlap_start or "2026-04-20"
    end = plan.latest_spy_session or start
    return {
        "mode": "bounded_preflight_plan_only",
        "live_collection_executed_by_preflight": False,
        "direct_daily_supported": True,
        "timeframe": "1Day",
        "feed": "sip",
        "adjustment": "all",
        "symbols": list(symbols),
        "start": start,
        "end": end,
        "output_root": str(output_root),
        "expected_archive_root": str(DEFAULT_DAILY_ARCHIVE_ROOT),
        "rerun_idempotency": "existing collection manifest skips completed chunks when resume=true",
    }


def smoke_coverage_rows(symbols: Sequence[str], universe_rows: Sequence[Mapping[str, Any]], stooq_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_symbol = {row["canonical_symbol"]: row for row in universe_rows}
    freshness = {row["symbol"]: row for row in stooq_rows}
    rows = []
    for symbol in symbols:
        canonical = symbol.upper()
        info = by_symbol.get(canonical, {})
        rows.append(
            {
                "canonical_symbol": canonical,
                "alpaca_provider_symbol": info.get("alpaca_provider_symbol", alpaca_provider_symbol(canonical)),
                "mapping_status": info.get("alpaca_mapping_status", "missing"),
                "latest_stooq_session": freshness.get(canonical, {}).get("latest_session", ""),
                "included_reason": "representative_smoke_symbol",
            }
        )
    return rows


def read_stooq_daily_rows(stooq_root: Path, symbols: Sequence[str], start: str | None, end: str | None) -> list[dict[str, Any]]:
    if not start or not end:
        return []
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    rows = []
    for symbol in symbols:
        path = stooq_root / f"{symbol.upper()}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path, columns=["symbol", "timestamp", "close", "volume"])
        for row in table.to_pylist():
            session = _session_date(row["timestamp"])
            if start_date <= session <= end_date:
                rows.append(
                    {
                        "symbol": symbol.upper(),
                        "session_date": session.isoformat(),
                        "timestamp": datetime.combine(session, time(0), tzinfo=timezone.utc),
                        "close": row.get("close"),
                        "volume": row.get("volume"),
                    }
                )
    return rows


def read_alpaca_daily_archive(
    archive_root: Path,
    symbols: Sequence[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, Any]]:
    wanted = {symbol.upper() for symbol in symbols or []}
    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None
    rows: list[dict[str, Any]] = []
    for path in sorted(archive_root.glob("symbol=*/year=*/bars.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            symbol = str(row.get("canonical_symbol") or row.get("symbol") or "").upper()
            if wanted and symbol not in wanted:
                continue
            timestamp = _parse_timestamp(row.get("timestamp_utc"))
            session_date = str(row.get("session_date") or (timestamp.date().isoformat() if timestamp else ""))
            parsed_session = date.fromisoformat(session_date)
            if start_date and parsed_session < start_date:
                continue
            if end_date and parsed_session > end_date:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session_date,
                    "timestamp": timestamp,
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "provider_symbol": row.get("provider_symbol"),
                    "adjustment_policy": row.get("adjustment_policy"),
                }
            )
    return sorted(rows, key=lambda row: (row["symbol"], row["session_date"]))


def reconcile_smoke_rows(stooq_rows: Sequence[Mapping[str, Any]], alpaca_rows: Sequence[Mapping[str, Any]] = ()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    stooq_by_key = {(row["symbol"], row["session_date"]): row for row in stooq_rows}
    alpaca_by_key = {(row.get("symbol"), str(row.get("session_date") or _session_date(row.get("timestamp")).isoformat())): row for row in alpaca_rows if row.get("symbol")}
    keys = sorted(set(stooq_by_key) | set(alpaca_by_key))
    previous_stooq: dict[str, float] = {}
    previous_alpaca: dict[str, float] = {}
    for symbol, session in keys:
        left = stooq_by_key.get((symbol, session))
        right = alpaca_by_key.get((symbol, session))
        stooq_close = _float(left.get("close")) if left else None
        alpaca_close = _float(right.get("close")) if right else None
        stooq_return = _return(previous_stooq.get(symbol), stooq_close)
        alpaca_return = _return(previous_alpaca.get(symbol), alpaca_close)
        if stooq_close is not None:
            previous_stooq[symbol] = stooq_close
        if alpaca_close is not None:
            previous_alpaca[symbol] = alpaca_close
        rel = (alpaca_close - stooq_close) / stooq_close if stooq_close and alpaca_close is not None else ""
        abs_rel = abs(rel) if isinstance(rel, float) else ""
        return_diff = alpaca_return - stooq_return if isinstance(alpaca_return, float) and isinstance(stooq_return, float) else ""
        abs_return_diff = abs(return_diff) if isinstance(return_diff, float) else ""
        stooq_volume = _float(left.get("volume")) if left else None
        alpaca_volume = _float(right.get("volume")) if right else None
        volume_diff = alpaca_volume - stooq_volume if alpaca_volume is not None and stooq_volume is not None else ""
        rel_volume_diff = volume_diff / stooq_volume if isinstance(volume_diff, float) and stooq_volume else ""
        classification = _reconciliation_classification(
            missing_stooq=left is None,
            missing_alpaca=right is None,
            abs_relative_close_difference=abs_rel,
            abs_return_difference=abs_return_diff,
        )
        rows.append(
            {
                "session_date": session,
                "symbol": symbol,
                "stooq_close": stooq_close if stooq_close is not None else "",
                "alpaca_close": alpaca_close if alpaca_close is not None else "",
                "relative_close_difference": rel,
                "abs_relative_close_difference": abs_rel,
                "stooq_return": stooq_return,
                "alpaca_return": alpaca_return,
                "return_difference": return_diff,
                "abs_return_difference": abs_return_diff,
                "stooq_volume": stooq_volume if stooq_volume is not None else "",
                "alpaca_volume": alpaca_volume if alpaca_volume is not None else "",
                "volume_difference": volume_diff,
                "relative_volume_difference": rel_volume_diff,
                "missing_stooq": str(left is None).lower(),
                "missing_alpaca": str(right is None).lower(),
                "possible_adjustment_discrepancy": str(classification == "POSSIBLE_ADJUSTMENT_DIFFERENCE").lower(),
                "possible_corporate_action": str(classification == "POSSIBLE_CORPORATE_ACTION").lower(),
                "classification": classification,
            }
        )
    close_diffs = [float(row["abs_relative_close_difference"]) for row in rows if isinstance(row["abs_relative_close_difference"], float)]
    return_diffs = [float(row["abs_return_difference"]) for row in rows if isinstance(row["abs_return_difference"], float)]
    volume_diffs = [abs(float(row["relative_volume_difference"])) for row in rows if isinstance(row["relative_volume_difference"], float)]
    classifications = Counter(str(row["classification"]) for row in rows)
    per_symbol = _per_symbol_reconciliation_stats(rows)
    compatibility = _provider_compatibility_decision(rows, close_diffs, return_diffs, classifications)
    return rows, {
        "canonical_source_selected": False,
        "alpaca_archive_row_count": len(alpaca_rows),
        "stooq_comparison_row_count": len(stooq_rows),
        "output_row_count": len(rows),
        "matched_rows": sum(1 for row in rows if row["missing_stooq"] == "false" and row["missing_alpaca"] == "false"),
        "stooq_only_rows": sum(1 for row in rows if row["missing_alpaca"] == "true"),
        "alpaca_only_rows": sum(1 for row in rows if row["missing_stooq"] == "true"),
        "relative_close_difference_count": sum(1 for value in close_diffs if value > 0.0005),
        "max_abs_relative_close_difference": max(close_diffs, default=0.0),
        "median_abs_relative_close_difference": _percentile(close_diffs, 0.50),
        "p95_abs_relative_close_difference": _percentile(close_diffs, 0.95),
        "return_difference_count": sum(1 for value in return_diffs if value > 0.0005),
        "max_abs_return_difference": max(return_diffs, default=0.0),
        "median_abs_return_difference": _percentile(return_diffs, 0.50),
        "p95_abs_return_difference": _percentile(return_diffs, 0.95),
        "max_abs_relative_volume_difference": max(volume_diffs, default=0.0),
        "median_abs_relative_volume_difference": _percentile(volume_diffs, 0.50),
        "p95_abs_relative_volume_difference": _percentile(volume_diffs, 0.95),
        "rows_by_classification": dict(sorted(classifications.items())),
        "possible_adjustment_discrepancy_count": classifications.get("POSSIBLE_ADJUSTMENT_DIFFERENCE", 0),
        "possible_corporate_action_count": classifications.get("POSSIBLE_CORPORATE_ACTION", 0),
        "large_unexplained_difference_count": classifications.get("LARGE_UNEXPLAINED_DIFFERENCE", 0),
        "per_symbol": per_symbol,
        "provider_compatibility_decision": compatibility["decision"],
        "provider_compatibility_evidence": compatibility["evidence"],
        "classification": "blocked_until_live_smoke" if not alpaca_rows else "live_smoke_reconciled",
    }


def run_reconcile_only(
    *,
    alpaca_archive_root: Path,
    stooq_root: Path = DEFAULT_STOOQ_ROOT,
    report_root: Path,
    symbols: Sequence[str],
    start: str,
    end: str,
    smoke_output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    canonical_symbols = [str(symbol).upper() for symbol in symbols]
    alpaca_rows = read_alpaca_daily_archive(alpaca_archive_root, canonical_symbols, start=start, end=end)
    stooq_rows = read_stooq_daily_rows(stooq_root, canonical_symbols, start, end)
    reconciliation_rows, summary = reconcile_smoke_rows(stooq_rows, alpaca_rows)
    collection_report = _offline_collection_report(
        alpaca_archive_root=alpaca_archive_root,
        smoke_output_root=smoke_output_root,
        alpaca_rows=alpaca_rows,
        summary=summary,
    )
    coverage_rows = [
        {
            "canonical_symbol": symbol,
            "alpaca_provider_symbol": alpaca_provider_symbol(symbol),
            "mapping_status": "mapped",
            "latest_stooq_session": max((row["session_date"] for row in stooq_rows if row["symbol"] == symbol), default=""),
            "included_reason": "reconcile_only_overlap_symbol",
        }
        for symbol in canonical_symbols
    ]
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "smoke_collection_report.json", collection_report)
        _write_csv(report_root / "smoke_coverage.csv", coverage_rows, SMOKE_COVERAGE_FIELDS)
        _write_csv(report_root / "smoke_reconciliation.csv", reconciliation_rows, RECONCILIATION_FIELDS)
        _write_json(report_root / "smoke_reconciliation.json", summary)
    return {
        "alpaca_archive_root": str(alpaca_archive_root),
        "stooq_root": str(stooq_root),
        "report_root": str(report_root),
        "symbols": canonical_symbols,
        "start": start,
        "end": end,
        "dry_run": dry_run,
        "collection_report": collection_report,
        "reconciliation": summary,
    }


def manual_production_command() -> str:
    return "python .\\main.py --mode ml-historical-bar-backfill-collect --config .\\config\\config.historical_bar_backfill_alpaca_daily_sip_514_symbol_template.yaml"


def _reconciliation_classification(
    *,
    missing_stooq: bool,
    missing_alpaca: bool,
    abs_relative_close_difference: float | str,
    abs_return_difference: float | str,
) -> str:
    if missing_stooq and not missing_alpaca:
        return "ALPACA_ONLY"
    if missing_alpaca and not missing_stooq:
        return "STOOQ_ONLY"
    close_diff = float(abs_relative_close_difference or 0.0)
    return_diff = float(abs_return_difference or 0.0)
    if close_diff <= 0.0005 and return_diff <= 0.0005:
        return "MATCH"
    if close_diff <= 0.005 and return_diff <= 0.001:
        return "SMALL_VENDOR_DIFFERENCE"
    if close_diff > 0.02 and return_diff > 0.02:
        return "POSSIBLE_CORPORATE_ACTION"
    if close_diff > 0.005 and return_diff <= 0.002:
        return "POSSIBLE_ADJUSTMENT_DIFFERENCE"
    return "LARGE_UNEXPLAINED_DIFFERENCE"


def _per_symbol_reconciliation_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["symbol"])].append(row)
    output = {}
    for symbol, symbol_rows in sorted(grouped.items()):
        close_diffs = [float(row["abs_relative_close_difference"]) for row in symbol_rows if isinstance(row["abs_relative_close_difference"], float)]
        return_diffs = [float(row["abs_return_difference"]) for row in symbol_rows if isinstance(row["abs_return_difference"], float)]
        output[symbol] = {
            "output_rows": len(symbol_rows),
            "matched_rows": sum(1 for row in symbol_rows if row["missing_stooq"] == "false" and row["missing_alpaca"] == "false"),
            "alpaca_only_rows": sum(1 for row in symbol_rows if row["missing_stooq"] == "true"),
            "stooq_only_rows": sum(1 for row in symbol_rows if row["missing_alpaca"] == "true"),
            "rows_by_classification": dict(sorted(Counter(str(row["classification"]) for row in symbol_rows).items())),
            "max_abs_relative_close_difference": max(close_diffs, default=0.0),
            "p95_abs_relative_close_difference": _percentile(close_diffs, 0.95),
            "max_abs_return_difference": max(return_diffs, default=0.0),
            "p95_abs_return_difference": _percentile(return_diffs, 0.95),
        }
    return output


def _provider_compatibility_decision(
    rows: Sequence[Mapping[str, Any]],
    close_diffs: Sequence[float],
    return_diffs: Sequence[float],
    classifications: Counter[str],
) -> dict[str, Any]:
    matched = sum(1 for row in rows if row["missing_stooq"] == "false" and row["missing_alpaca"] == "false")
    evidence = {
        "matched_rows": matched,
        "alpaca_only_rows": classifications.get("ALPACA_ONLY", 0),
        "stooq_only_rows": classifications.get("STOOQ_ONLY", 0),
        "max_abs_relative_close_difference": max(close_diffs, default=0.0),
        "p95_abs_relative_close_difference": _percentile(close_diffs, 0.95),
        "max_abs_return_difference": max(return_diffs, default=0.0),
        "p95_abs_return_difference": _percentile(return_diffs, 0.95),
        "large_unexplained_difference_count": classifications.get("LARGE_UNEXPLAINED_DIFFERENCE", 0),
        "possible_corporate_action_count": classifications.get("POSSIBLE_CORPORATE_ACTION", 0),
    }
    if matched == 0:
        decision = "PROVIDER_COMPATIBILITY_BLOCKED"
    elif evidence["large_unexplained_difference_count"] or evidence["possible_corporate_action_count"]:
        decision = "PROVIDER_COMPATIBILITY_REVIEW_REQUIRED"
    elif evidence["p95_abs_return_difference"] <= 0.001 and evidence["max_abs_return_difference"] <= 0.005:
        decision = "PROVIDER_COMPATIBILITY_ACCEPTABLE"
    else:
        decision = "PROVIDER_COMPATIBILITY_REVIEW_REQUIRED"
    return {"decision": decision, "evidence": evidence}


def _offline_collection_report(
    *,
    alpaca_archive_root: Path,
    smoke_output_root: Path | None,
    alpaca_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    collect_report_path = (smoke_output_root / "historical_bar_collect_report.json") if smoke_output_root else None
    collect_payload = {}
    if collect_report_path and collect_report_path.exists():
        collect_payload = json.loads(collect_report_path.read_text(encoding="utf-8"))
    return {
        "mode": "alpaca_daily_reconcile_only",
        "offline_reconciliation": True,
        "credentials_required": False,
        "api_requests_attempted": 0,
        "blocked_reason": None,
        "actual_collect_report_path": str(collect_report_path) if collect_report_path else None,
        "api_collection_succeeded": bool(collect_payload.get("observed_metrics", {}).get("requests_successful") or collect_payload.get("daily_archive_report")),
        "raw_chunk_completed_or_reused": bool(collect_payload.get("chunk_results") or alpaca_rows),
        "staging_completed": bool(collect_payload.get("staging_path")),
        "daily_archive_completed": bool(alpaca_rows),
        "production_validated": False,
        "canonical_market_data_modified": False,
        "alpaca_archive_root": str(alpaca_archive_root),
        "smoke_archive_row_count": len(alpaca_rows),
        "reconciliation_status": "completed" if alpaca_rows else "blocked_no_alpaca_archive_rows",
        "production_compatibility_decision": summary.get("provider_compatibility_decision"),
        "collection_status": "completed_archive_available" if alpaca_rows else "missing_archive_rows",
        "observed_metrics": {"requests_attempted": 0},
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def write_daily_archive(
    rows: Sequence[Mapping[str, Any]],
    *,
    archive_root: Path,
    asset_registry: Path = DEFAULT_ASSET_REGISTRY,
    alias_registry: Path = DEFAULT_ALIAS_REGISTRY,
    dataset_version: str = "alpaca_daily_bars_v1",
) -> dict[str, Any]:
    assets = {asset.canonical_symbol: asset.asset_id for asset in read_assets_csv(asset_registry)}
    aliases = {
        alias.asset_id: alias.provider_symbol
        for alias in read_aliases_csv(alias_registry)
        if alias.provider == "alpaca" and alias.is_primary
    }
    asset_to_symbol = {asset_id: symbol for symbol, asset_id in assets.items()}
    serial: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for row in rows:
        canonical = str(row.get("symbol", "")).upper()
        timestamp = _parse_timestamp(row.get("timestamp"))
        if not canonical or timestamp is None:
            continue
        session = timestamp.date().isoformat()
        key = (canonical, session)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        asset_id = assets.get(canonical, "")
        provider_symbol = row.get("provider_symbol") or aliases.get(asset_id) or canonical
        serial.append(
            {
                "asset_id": asset_id,
                "canonical_symbol": canonical,
                "provider_symbol": provider_symbol,
                "session_date": session,
                "timestamp_utc": timestamp,
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
                "volume": _float(row.get("volume")),
                "trade_count": row.get("trade_count"),
                "vwap": _float(row.get("vwap")),
                "provider": "alpaca",
                "feed": row.get("feed", "sip"),
                "timeframe": "1Day",
                "adjustment_policy": row.get("adjustment_mode", "all"),
                "request_chunk_id": row.get("raw_chunk_identifier", ""),
                "dataset_version": dataset_version,
            }
        )
    by_partition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in serial:
        year = str(row["session_date"])[:4]
        by_partition.setdefault((str(row["canonical_symbol"]), year), []).append(row)
    written = []
    schema = _daily_archive_schema()
    for (symbol, year), partition_rows in sorted(by_partition.items()):
        target = archive_root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        table = pa.Table.from_pylist(sorted(partition_rows, key=lambda item: item["session_date"]), schema=schema)
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(target)
        written.append({"path": str(target), "row_count": len(partition_rows), "symbol": symbol, "year": year})
    return {
        "archive_root": str(archive_root),
        "dataset_version": dataset_version,
        "input_rows": len(rows),
        "written_rows": len(serial),
        "duplicate_symbol_session_rows_dropped": duplicates,
        "partition_count": len(written),
        "partitions": written,
        "symbols_without_asset_id": sorted({row["canonical_symbol"] for row in serial if not row["asset_id"]}),
        "asset_registry": str(asset_registry),
        "alias_registry": str(alias_registry),
        "asset_id_mapping_count": len(asset_to_symbol),
    }


def _daily_archive_schema() -> pa.Schema:
    return pa.schema(
        [
            ("asset_id", pa.string()),
            ("canonical_symbol", pa.string()),
            ("provider_symbol", pa.string()),
            ("session_date", pa.string()),
            ("timestamp_utc", pa.timestamp("us", tz="UTC")),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
            ("trade_count", pa.int64()),
            ("vwap", pa.float64()),
            ("provider", pa.string()),
            ("feed", pa.string()),
            ("timeframe", pa.string()),
            ("adjustment_policy", pa.string()),
            ("request_chunk_id", pa.string()),
            ("dataset_version", pa.string()),
        ]
    )


def asset_role(symbol: str, security_type: str) -> str:
    if symbol.upper() in {"SPY", "QQQ", "IWM"} or "ETF" in security_type.upper():
        return "etf_or_reference"
    return "selectable_equity_candidate"


def _stooq_symbol_dates(path: Path) -> list[date]:
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["timestamp"])
    return sorted({_session_date(row["timestamp"]) for row in table.to_pylist() if row.get("timestamp") is not None})


def _session_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _return(previous: float | None, current: float | None) -> float | str:
    if previous in (None, 0) or current is None:
        return ""
    return current / previous - 1.0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value is None:
        return None
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
