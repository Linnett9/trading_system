from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from core.research.ml.reference.canonical_assets import read_aliases_csv, read_assets_csv


DEFAULT_ALPACA_ARCHIVE_ROOT = Path("data/processed/alpaca/symbol_bars/sip/1d")
DEFAULT_STOOQ_ROOT = Path("data/processed/stooq_parquet")
DEFAULT_ASSET_REGISTRY = Path("data/reference/assets/canonical_asset_registry.csv")
DEFAULT_ALIAS_REGISTRY = Path("data/reference/assets/provider_symbol_aliases.csv")
DEFAULT_REPORT_ROOT = Path("reports/data_lineage/alpaca_daily_full_universe_reconciliation")
PRICE_MATCH_RELATIVE_THRESHOLD = 0.0005
SMALL_PRICE_RELATIVE_THRESHOLD = 0.005
RETURN_MATCH_THRESHOLD = 0.0005
SMALL_RETURN_THRESHOLD = 0.001
LARGE_RETURN_THRESHOLD = 0.02
VOLUME_DEFINITION_THRESHOLD = 0.25


RECONCILIATION_FIELDS = (
    "asset_id",
    "canonical_symbol",
    "session_date",
    "alpaca_present",
    "stooq_present",
    "alpaca_open",
    "stooq_open",
    "open_abs_diff",
    "open_rel_diff",
    "alpaca_high",
    "stooq_high",
    "high_abs_diff",
    "high_rel_diff",
    "alpaca_low",
    "stooq_low",
    "low_abs_diff",
    "low_rel_diff",
    "alpaca_close",
    "stooq_close",
    "close_abs_diff",
    "close_rel_diff",
    "alpaca_volume",
    "stooq_volume",
    "volume_abs_diff",
    "volume_rel_diff",
    "alpaca_return",
    "stooq_return",
    "return_abs_diff",
    "missing_source_indicator",
    "classification",
)


SYMBOL_STAT_FIELDS = (
    "asset_id",
    "canonical_symbol",
    "overlap_row_count",
    "alpaca_only_row_count",
    "stooq_only_row_count",
    "match_count",
    "small_difference_count",
    "large_difference_count",
    "median_absolute_close_difference",
    "p95_absolute_close_difference",
    "maximum_absolute_close_difference",
    "median_absolute_return_difference",
    "p95_absolute_return_difference",
    "maximum_absolute_return_difference",
    "median_relative_volume_difference",
    "p95_relative_volume_difference",
    "maximum_relative_volume_difference",
    "missing_session_count",
)


COVERAGE_FIELDS = (
    "asset_id",
    "canonical_symbol",
    "row_count",
    "first_session",
    "last_session",
    "missing_session_count",
    "missing_sessions",
)


MISSING_FIELDS = (
    "asset_id",
    "symbol",
    "missing_session",
    "previous_available_session",
    "next_available_session",
    "spy_traded_that_day",
    "other_assets_traded_that_day",
    "stooq_has_bar",
    "likely_explanation",
    "classification",
)


@dataclass(frozen=True)
class SourceRows:
    rows: list[dict[str, Any]]
    file_count: int
    source_paths: tuple[str, ...]


def run_full_universe_reconciliation(
    *,
    alpaca_archive_root: Path = DEFAULT_ALPACA_ARCHIVE_ROOT,
    stooq_root: Path = DEFAULT_STOOQ_ROOT,
    asset_registry: Path = DEFAULT_ASSET_REGISTRY,
    alias_registry: Path = DEFAULT_ALIAS_REGISTRY,
    report_root: Path = DEFAULT_REPORT_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    assets = [asset for asset in read_assets_csv(asset_registry) if asset.collection_universe_514 and asset.is_active]
    asset_by_symbol = {asset.canonical_symbol: asset for asset in assets}
    aliases = read_aliases_csv(alias_registry)
    alias_by_asset_provider = {
        (alias.asset_id, alias.provider): alias.provider_symbol
        for alias in aliases
        if alias.is_primary
    }
    alpaca = read_alpaca_archive(alpaca_archive_root)
    if not alpaca.rows:
        raise FileNotFoundError(f"No Alpaca daily rows found under {alpaca_archive_root}")
    alpaca_dates = sorted({row["session_date"] for row in alpaca.rows})
    symbols = sorted(asset_by_symbol)
    stooq = read_stooq_rows(stooq_root, symbols, alpaca_dates[0], alpaca_dates[-1], asset_by_symbol)

    audit, coverage_rows, missing_rows = audit_alpaca_archive(
        alpaca,
        asset_by_symbol=asset_by_symbol,
        expected_sessions=alpaca_dates,
        stooq_rows=stooq.rows,
        alpaca_archive_root=alpaca_archive_root,
    )
    reconciliation_rows = reconcile_provider_rows(alpaca.rows, stooq.rows, asset_by_symbol)
    summary, symbol_stats, flagged_rows, boundary_policy, decision = summarize_reconciliation(
        reconciliation_rows,
        audit=audit,
        stooq_row_count=len(stooq.rows),
    )
    outputs = {
        "alpaca_archive_audit.json": audit,
        "alpaca_archive_audit.md": render_audit_markdown(audit, missing_rows),
        "alpaca_symbol_coverage.csv": coverage_rows,
        "alpaca_missing_sessions.csv": missing_rows,
        "provider_reconciliation.csv": reconciliation_rows,
        "provider_reconciliation_summary.json": summary,
        "provider_reconciliation_summary.md": render_summary_markdown(summary, symbol_stats, flagged_rows),
        "provider_symbol_statistics.csv": symbol_stats,
        "provider_flagged_rows.csv": flagged_rows,
        "provider_boundary_policy.json": boundary_policy,
        "compatibility_decision.json": decision,
    }
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "alpaca_archive_audit.json", audit)
        (report_root / "alpaca_archive_audit.md").write_text(outputs["alpaca_archive_audit.md"], encoding="utf-8")
        _write_csv(report_root / "alpaca_symbol_coverage.csv", coverage_rows, COVERAGE_FIELDS)
        _write_csv(report_root / "alpaca_missing_sessions.csv", missing_rows, MISSING_FIELDS)
        _write_csv(report_root / "provider_reconciliation.csv", reconciliation_rows, RECONCILIATION_FIELDS)
        _write_json(report_root / "provider_reconciliation_summary.json", summary)
        (report_root / "provider_reconciliation_summary.md").write_text(outputs["provider_reconciliation_summary.md"], encoding="utf-8")
        _write_csv(report_root / "provider_symbol_statistics.csv", symbol_stats, SYMBOL_STAT_FIELDS)
        _write_csv(report_root / "provider_flagged_rows.csv", flagged_rows, RECONCILIATION_FIELDS)
        _write_json(report_root / "provider_boundary_policy.json", boundary_policy)
        _write_json(report_root / "compatibility_decision.json", decision)
    return {
        "report_root": str(report_root),
        "dry_run": dry_run,
        "alpaca_archive_root": str(alpaca_archive_root),
        "stooq_root": str(stooq_root),
        "asset_registry": str(asset_registry),
        "alias_registry": str(alias_registry),
        "alpaca_source_file_count": alpaca.file_count,
        "stooq_source_file_count": stooq.file_count,
        "audit": audit,
        "summary": summary,
        "symbol_stats": symbol_stats,
        "flagged_rows": flagged_rows,
        "boundary_policy": boundary_policy,
        "compatibility_decision": decision,
        "outputs": sorted(outputs),
        "source_archives_modified": False,
        "canonical_market_data_modified": False,
        "alpaca_api_requests": 0,
    }


def read_alpaca_archive(root: Path) -> SourceRows:
    paths = sorted(root.glob("symbol=*/year=*/bars.parquet"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            canonical = str(row.get("canonical_symbol") or row.get("symbol") or "").upper()
            session = str(row.get("session_date") or _date_text(row.get("timestamp_utc")))
            rows.append(
                {
                    "asset_id": str(row.get("asset_id") or ""),
                    "canonical_symbol": canonical,
                    "provider_symbol": str(row.get("provider_symbol") or canonical),
                    "session_date": session,
                    "timestamp": row.get("timestamp_utc"),
                    "open": _float(row.get("open")),
                    "high": _float(row.get("high")),
                    "low": _float(row.get("low")),
                    "close": _float(row.get("close")),
                    "volume": _float(row.get("volume")),
                    "provider": "alpaca",
                    "adjustment_policy": str(row.get("adjustment_policy") or ""),
                    "source_path": str(path),
                }
            )
    return SourceRows(sorted(rows, key=lambda row: (row["canonical_symbol"], row["session_date"])), len(paths), tuple(str(path) for path in paths))


def read_stooq_rows(
    root: Path,
    symbols: Sequence[str],
    start: str,
    end: str,
    asset_by_symbol: Mapping[str, Any],
) -> SourceRows:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for symbol in symbols:
        path = root / f"{symbol}.parquet"
        if not path.exists():
            continue
        paths.append(path)
        table = pq.read_table(path)
        for row in table.to_pylist():
            session = _session_date(row.get("timestamp"))
            if session is None or session < start_date or session > end_date:
                continue
            asset = asset_by_symbol.get(symbol)
            rows.append(
                {
                    "asset_id": asset.asset_id if asset else "",
                    "canonical_symbol": symbol,
                    "provider_symbol": symbol,
                    "session_date": session.isoformat(),
                    "timestamp": row.get("timestamp"),
                    "open": _float(row.get("open")),
                    "high": _float(row.get("high")),
                    "low": _float(row.get("low")),
                    "close": _float(row.get("close")),
                    "volume": _float(row.get("volume")),
                    "provider": "stooq",
                    "adjustment_policy": "stooq_bulk_close_semantics",
                    "source_path": str(path),
                }
            )
    return SourceRows(sorted(rows, key=lambda row: (row["canonical_symbol"], row["session_date"])), len(paths), tuple(str(path) for path in paths))


def audit_alpaca_archive(
    alpaca: SourceRows,
    *,
    asset_by_symbol: Mapping[str, Any],
    expected_sessions: Sequence[str],
    stooq_rows: Sequence[Mapping[str, Any]],
    alpaca_archive_root: Path = DEFAULT_ALPACA_ARCHIVE_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = alpaca.rows
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[row["canonical_symbol"]].append(row)
    key_counts = Counter((row["canonical_symbol"], row["session_date"]) for row in rows)
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]
    invalid_ohlc = [row for row in rows if not _valid_ohlc(row)]
    invalid_volume = [row for row in rows if row.get("volume") is None or float(row["volume"]) < 0]
    invalid_timestamps = [row for row in rows if not row.get("session_date")]
    nonpositive_prices = [
        row for row in rows if any(row.get(field) is None or float(row[field]) <= 0 for field in ("open", "high", "low", "close"))
    ]
    mapping_failures = [
        row for row in rows
        if row["canonical_symbol"] not in asset_by_symbol or row["asset_id"] != asset_by_symbol[row["canonical_symbol"]].asset_id
    ]
    unsorted_partitions = []
    coverage_rows = []
    missing_rows = []
    all_sessions = set(expected_sessions)
    stooq_keys = {(row["canonical_symbol"], row["session_date"]) for row in stooq_rows}
    sessions_with_any = Counter(row["session_date"] for row in rows)
    spy_sessions = {row["session_date"] for row in by_symbol.get("SPY", [])}
    for symbol in sorted(asset_by_symbol):
        symbol_rows = sorted(by_symbol.get(symbol, []), key=lambda row: row["session_date"])
        observed = [row["session_date"] for row in symbol_rows]
        observed_set = set(observed)
        missing = sorted(all_sessions - observed_set)
        if observed != sorted(observed):
            unsorted_partitions.append(symbol)
        coverage_rows.append(
            {
                "asset_id": asset_by_symbol[symbol].asset_id,
                "canonical_symbol": symbol,
                "row_count": len(symbol_rows),
                "first_session": observed[0] if observed else "",
                "last_session": observed[-1] if observed else "",
                "missing_session_count": len(missing),
                "missing_sessions": "|".join(missing),
            }
        )
        for session in missing:
            previous = max((value for value in observed if value < session), default="")
            following = min((value for value in observed if value > session), default="")
            stooq_has_bar = (symbol, session) in stooq_keys
            classification, explanation = _missing_classification(
                symbol=symbol,
                session=session,
                previous=previous,
                following=following,
                stooq_has_bar=stooq_has_bar,
                spy_traded=session in spy_sessions,
                other_assets_traded=sessions_with_any[session] > 0,
            )
            missing_rows.append(
                {
                    "asset_id": asset_by_symbol[symbol].asset_id,
                    "symbol": symbol,
                    "missing_session": session,
                    "previous_available_session": previous,
                    "next_available_session": following,
                    "spy_traded_that_day": str(session in spy_sessions).lower(),
                    "other_assets_traded_that_day": str(sessions_with_any[session] > 0).lower(),
                    "stooq_has_bar": str(stooq_has_bar).lower(),
                    "likely_explanation": explanation,
                    "classification": classification,
                }
            )
    dates = sorted({row["session_date"] for row in rows})
    audit = {
        "schema_version": "alpaca_daily_full_universe_audit.v1",
        "alpaca_archive_root": str(alpaca_archive_root),
        "file_count": alpaca.file_count,
        "row_count": len(rows),
        "symbol_count": len(by_symbol),
        "asset_id_count": len({row["asset_id"] for row in rows if row.get("asset_id")}),
        "date_minimum": dates[0] if dates else None,
        "date_maximum": dates[-1] if dates else None,
        "unique_trading_sessions": len(dates),
        "expected_full_grid_rows": len(asset_by_symbol) * len(dates),
        "missing_grid_rows": len(missing_rows),
        "duplicate_symbol_session_rows": sum(count - 1 for count in key_counts.values() if count > 1),
        "duplicate_keys": [{"symbol": key[0], "session_date": key[1]} for key in sorted(duplicate_keys)],
        "invalid_ohlc_rows": len(invalid_ohlc),
        "invalid_volume_rows": len(invalid_volume),
        "invalid_timestamp_rows": len(invalid_timestamps),
        "nonpositive_price_rows": len(nonpositive_prices),
        "unsorted_partitions": sorted(unsorted_partitions),
        "symbol_asset_mapping_failures": len(mapping_failures),
        "symbols_without_asset_id": sorted({row["canonical_symbol"] for row in rows if not row.get("asset_id")}),
        "rows_per_symbol_minimum": min((len(value) for value in by_symbol.values()), default=0),
        "rows_per_symbol_maximum": max((len(value) for value in by_symbol.values()), default=0),
        "rows_per_symbol_mode": Counter(len(value) for value in by_symbol.values()).most_common(1)[0][0] if by_symbol else 0,
        "final_structural_validity": not duplicate_keys and not invalid_ohlc and not invalid_volume and not invalid_timestamps and not nonpositive_prices and not mapping_failures,
        "api_requests_attempted": 0,
        "canonical_market_data_modified": False,
        "source_archive_modified": False,
    }
    return audit, coverage_rows, sorted(missing_rows, key=lambda row: (row["symbol"], row["missing_session"]))


def reconcile_provider_rows(
    alpaca_rows: Sequence[Mapping[str, Any]],
    stooq_rows: Sequence[Mapping[str, Any]],
    asset_by_symbol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    alpaca_returns = _returns_by_key(alpaca_rows)
    stooq_returns = _returns_by_key(stooq_rows)
    alpaca_by_key = {(row["canonical_symbol"], row["session_date"]): row for row in alpaca_rows}
    stooq_by_key = {(row["canonical_symbol"], row["session_date"]): row for row in stooq_rows}
    output = []
    for symbol, session in sorted(set(alpaca_by_key) | set(stooq_by_key)):
        left = alpaca_by_key.get((symbol, session))
        right = stooq_by_key.get((symbol, session))
        asset = asset_by_symbol.get(symbol)
        row = {
            "asset_id": asset.asset_id if asset else (left or right or {}).get("asset_id", ""),
            "canonical_symbol": symbol,
            "session_date": session,
            "alpaca_present": str(left is not None).lower(),
            "stooq_present": str(right is not None).lower(),
            "missing_source_indicator": _missing_source_indicator(left, right),
        }
        for field in ("open", "high", "low", "close"):
            a = _float(left.get(field)) if left else None
            s = _float(right.get(field)) if right else None
            row[f"alpaca_{field}"] = _csv_value(a)
            row[f"stooq_{field}"] = _csv_value(s)
            row[f"{field}_abs_diff"] = _csv_value(abs(a - s) if a is not None and s is not None else None)
            row[f"{field}_rel_diff"] = _csv_value(abs(a - s) / abs(s) if a is not None and s not in (None, 0.0) else None)
        av = _float(left.get("volume")) if left else None
        sv = _float(right.get("volume")) if right else None
        ar = alpaca_returns.get((symbol, session))
        sr = stooq_returns.get((symbol, session))
        row["alpaca_volume"] = _csv_value(av)
        row["stooq_volume"] = _csv_value(sv)
        row["volume_abs_diff"] = _csv_value(abs(av - sv) if av is not None and sv is not None else None)
        row["volume_rel_diff"] = _csv_value(abs(av - sv) / abs(sv) if av is not None and sv not in (None, 0.0) else None)
        row["alpaca_return"] = _csv_value(ar)
        row["stooq_return"] = _csv_value(sr)
        row["return_abs_diff"] = _csv_value(abs(ar - sr) if ar is not None and sr is not None else None)
        row["classification"] = classify_reconciliation_row(row)
        output.append(row)
    return output


def classify_reconciliation_row(row: Mapping[str, Any]) -> str:
    alpaca_present = row.get("alpaca_present") == "true"
    stooq_present = row.get("stooq_present") == "true"
    if alpaca_present and not stooq_present:
        return "ALPACA_ONLY"
    if stooq_present and not alpaca_present:
        return "STOOQ_ONLY"
    close_rel = _float(row.get("close_rel_diff")) or 0.0
    return_diff = _float(row.get("return_abs_diff")) or 0.0
    volume_rel = _float(row.get("volume_rel_diff")) or 0.0
    if close_rel <= PRICE_MATCH_RELATIVE_THRESHOLD and return_diff <= RETURN_MATCH_THRESHOLD:
        return "VOLUME_DEFINITION_DIFFERENCE" if volume_rel > VOLUME_DEFINITION_THRESHOLD else "MATCH"
    if close_rel <= SMALL_PRICE_RELATIVE_THRESHOLD and return_diff <= SMALL_RETURN_THRESHOLD:
        return "VOLUME_DEFINITION_DIFFERENCE" if volume_rel > VOLUME_DEFINITION_THRESHOLD else "SMALL_VENDOR_DIFFERENCE"
    if close_rel > 0.02 and return_diff <= SMALL_RETURN_THRESHOLD:
        return "POSSIBLE_ADJUSTMENT_DIFFERENCE"
    if close_rel > 0.02 and return_diff > LARGE_RETURN_THRESHOLD:
        return "POSSIBLE_CORPORATE_ACTION"
    return "LARGE_UNEXPLAINED_DIFFERENCE"


def summarize_reconciliation(
    rows: Sequence[Mapping[str, Any]],
    *,
    audit: Mapping[str, Any],
    stooq_row_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    classifications = Counter(str(row["classification"]) for row in rows)
    matched = [row for row in rows if row["alpaca_present"] == "true" and row["stooq_present"] == "true"]
    close_abs = [_float(row.get("close_abs_diff")) for row in matched if _float(row.get("close_abs_diff")) is not None]
    close_rel = [_float(row.get("close_rel_diff")) for row in matched if _float(row.get("close_rel_diff")) is not None]
    returns = [_float(row.get("return_abs_diff")) for row in matched if _float(row.get("return_abs_diff")) is not None]
    volumes = [_float(row.get("volume_rel_diff")) for row in matched if _float(row.get("volume_rel_diff")) is not None]
    symbol_stats = per_symbol_statistics(rows)
    flagged_rows = [
        row for row in rows
        if row["classification"] in {"POSSIBLE_ADJUSTMENT_DIFFERENCE", "POSSIBLE_CORPORATE_ACTION", "LARGE_UNEXPLAINED_DIFFERENCE", "MISSING_SESSION_REVIEW"}
    ]
    worst = {
        "return_disagreement": sorted(symbol_stats, key=lambda row: (-_float(row["maximum_absolute_return_difference"]), row["canonical_symbol"]))[:10],
        "close_disagreement": sorted(symbol_stats, key=lambda row: (-_float(row["maximum_absolute_close_difference"]), row["canonical_symbol"]))[:10],
        "volume_disagreement": sorted(symbol_stats, key=lambda row: (-_float(row["maximum_relative_volume_difference"]), row["canonical_symbol"]))[:10],
        "missing_sessions": sorted(symbol_stats, key=lambda row: (-int(row["missing_session_count"]), row["canonical_symbol"]))[:10],
    }
    decision = compatibility_decision(
        matched_rows=len(matched),
        classification_counts=classifications,
        p95_return=_percentile([value for value in returns if value is not None], 0.95),
        max_return=max([value for value in returns if value is not None], default=0.0),
        p95_close_rel=_percentile([value for value in close_rel if value is not None], 0.95),
        max_close_rel=max([value for value in close_rel if value is not None], default=0.0),
    )
    boundary_policy = provider_boundary_policy(decision)
    summary = {
        "schema_version": "alpaca_stooq_full_universe_reconciliation.v1",
        "alpaca_rows_read": audit["row_count"],
        "stooq_rows_read": stooq_row_count,
        "output_union_rows": len(rows),
        "overlapping_rows": len(matched),
        "alpaca_only_rows": classifications.get("ALPACA_ONLY", 0),
        "stooq_only_rows": classifications.get("STOOQ_ONLY", 0),
        "classification_totals": dict(sorted(classifications.items())),
        "close_difference_statistics": _stats(close_abs),
        "close_relative_difference_statistics": _stats(close_rel),
        "return_difference_statistics": _stats(returns),
        "volume_relative_difference_statistics": _stats(volumes),
        "worst_symbols": worst,
        "flagged_row_count": len(flagged_rows),
        "compatibility_decision": decision["decision"],
        "compatibility_thresholds": decision["thresholds"],
        "api_requests_attempted": 0,
        "source_archives_modified": False,
        "canonical_market_data_modified": False,
    }
    return summary, symbol_stats, flagged_rows, boundary_policy, decision


def per_symbol_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_symbol"])].append(row)
    output = []
    for symbol, symbol_rows in sorted(grouped.items()):
        overlap = [row for row in symbol_rows if row["alpaca_present"] == "true" and row["stooq_present"] == "true"]
        close = [_float(row.get("close_abs_diff")) for row in overlap if _float(row.get("close_abs_diff")) is not None]
        returns = [_float(row.get("return_abs_diff")) for row in overlap if _float(row.get("return_abs_diff")) is not None]
        volumes = [_float(row.get("volume_rel_diff")) for row in overlap if _float(row.get("volume_rel_diff")) is not None]
        counts = Counter(row["classification"] for row in symbol_rows)
        asset_id = str(symbol_rows[0].get("asset_id") or "")
        output.append(
            {
                "asset_id": asset_id,
                "canonical_symbol": symbol,
                "overlap_row_count": len(overlap),
                "alpaca_only_row_count": counts.get("ALPACA_ONLY", 0),
                "stooq_only_row_count": counts.get("STOOQ_ONLY", 0),
                "match_count": counts.get("MATCH", 0),
                "small_difference_count": counts.get("SMALL_VENDOR_DIFFERENCE", 0) + counts.get("VOLUME_DEFINITION_DIFFERENCE", 0),
                "large_difference_count": counts.get("LARGE_UNEXPLAINED_DIFFERENCE", 0) + counts.get("POSSIBLE_ADJUSTMENT_DIFFERENCE", 0) + counts.get("POSSIBLE_CORPORATE_ACTION", 0),
                "median_absolute_close_difference": _percentile(close, 0.50),
                "p95_absolute_close_difference": _percentile(close, 0.95),
                "maximum_absolute_close_difference": max(close, default=0.0),
                "median_absolute_return_difference": _percentile(returns, 0.50),
                "p95_absolute_return_difference": _percentile(returns, 0.95),
                "maximum_absolute_return_difference": max(returns, default=0.0),
                "median_relative_volume_difference": _percentile(volumes, 0.50),
                "p95_relative_volume_difference": _percentile(volumes, 0.95),
                "maximum_relative_volume_difference": max(volumes, default=0.0),
                "missing_session_count": counts.get("ALPACA_ONLY", 0) + counts.get("STOOQ_ONLY", 0),
            }
        )
    return output


def compatibility_decision(
    *,
    matched_rows: int,
    classification_counts: Counter[str],
    p95_return: float,
    max_return: float,
    p95_close_rel: float,
    max_close_rel: float,
) -> dict[str, Any]:
    thresholds = {
        "blocked_if_matched_rows_zero": True,
        "review_if_large_unexplained_rows": 1,
        "acceptable_with_controls_p95_return_max": 0.0025,
        "acceptable_with_controls_max_return_max": 0.05,
        "acceptable_with_controls_p95_relative_close_max": 0.01,
        "acceptable_with_controls_max_relative_close_max": 0.10,
    }
    if matched_rows == 0:
        decision = "FULL_UNIVERSE_COMPATIBILITY_BLOCKED"
    elif classification_counts.get("LARGE_UNEXPLAINED_DIFFERENCE", 0) >= thresholds["review_if_large_unexplained_rows"]:
        decision = "FULL_UNIVERSE_REVIEW_REQUIRED"
    elif (
        p95_return <= thresholds["acceptable_with_controls_p95_return_max"]
        and max_return <= thresholds["acceptable_with_controls_max_return_max"]
        and p95_close_rel <= thresholds["acceptable_with_controls_p95_relative_close_max"]
        and max_close_rel <= thresholds["acceptable_with_controls_max_relative_close_max"]
    ):
        decision = "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE_WITH_CONTROLS"
    else:
        decision = "FULL_UNIVERSE_REVIEW_REQUIRED"
    return {
        "decision": decision,
        "thresholds": thresholds,
        "evidence": {
            "matched_rows": matched_rows,
            "classification_counts": dict(sorted(classification_counts.items())),
            "p95_absolute_return_difference": p95_return,
            "maximum_absolute_return_difference": max_return,
            "p95_relative_close_difference": p95_close_rel,
            "maximum_relative_close_difference": max_close_rel,
        },
        "canonical_daily_market_dataset_v2_approved_for_construction": decision in {
            "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE",
            "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE_WITH_CONTROLS",
        },
    }


def provider_boundary_policy(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": decision["decision"],
        "source_provenance_required": True,
        "required_columns": [
            "provider",
            "source_dataset_version",
            "provider_transition_flag",
            "provider_local_rolling_volume_percentile",
            "provider_local_rolling_volume_z_score",
        ],
        "return_controls": [
            "validate transition-day and adjacent-session returns by asset",
            "flag provider boundary rows for downstream feature generation",
        ],
        "price_controls": [
            "retain raw provider prices",
            "avoid silent adjusted/unadjusted mixing",
        ],
        "volume_controls": [
            "do not treat Stooq and Alpaca volume definitions as identical",
            "use provider-local rolling volume percentiles near transition",
            "use provider-local rolling volume z-scores near transition",
            "avoid raw cross-provider volume ratios near transition",
        ],
        "approved_next_step": decision.get("canonical_daily_market_dataset_v2_approved_for_construction", False),
    }


def render_audit_markdown(audit: Mapping[str, Any], missing_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Alpaca Archive Audit",
        "",
        f"- Files: {audit['file_count']}",
        f"- Rows: {audit['row_count']}",
        f"- Symbols: {audit['symbol_count']}",
        f"- Date range: {audit['date_minimum']} to {audit['date_maximum']}",
        f"- Unique sessions: {audit['unique_trading_sessions']}",
        f"- Missing grid rows: {audit['missing_grid_rows']}",
        f"- Duplicate rows: {audit['duplicate_symbol_session_rows']}",
        f"- Invalid OHLC rows: {audit['invalid_ohlc_rows']}",
        "",
        "## Missing Sessions",
        "",
    ]
    for row in missing_rows:
        lines.append(f"- {row['symbol']} {row['missing_session']}: {row['classification']} - {row['likely_explanation']}")
    return "\n".join(lines) + "\n"


def render_summary_markdown(
    summary: Mapping[str, Any],
    symbol_stats: Sequence[Mapping[str, Any]],
    flagged_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Provider Reconciliation Summary",
        "",
        f"- Overlapping rows: {summary['overlapping_rows']}",
        f"- Alpaca-only rows: {summary['alpaca_only_rows']}",
        f"- Stooq-only rows: {summary['stooq_only_rows']}",
        f"- Decision: {summary['compatibility_decision']}",
        f"- Flagged rows: {len(flagged_rows)}",
        "",
        "## Classification Totals",
        "",
    ]
    for key, value in summary["classification_totals"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Worst Return Symbols")
    lines.append("")
    for row in summary["worst_symbols"]["return_disagreement"][:10]:
        lines.append(f"- {row['canonical_symbol']}: max return diff {row['maximum_absolute_return_difference']}")
    return "\n".join(lines) + "\n"


def _returns_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], float | None]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_symbol"])].append(row)
    output: dict[tuple[str, str], float | None] = {}
    for symbol, symbol_rows in grouped.items():
        previous = None
        for row in sorted(symbol_rows, key=lambda item: item["session_date"]):
            close = _float(row.get("close"))
            output[(symbol, row["session_date"])] = None if previous in (None, 0.0) or close is None else close / previous - 1.0
            if close is not None:
                previous = close
    return output


def _missing_classification(
    *,
    symbol: str,
    session: str,
    previous: str,
    following: str,
    stooq_has_bar: bool,
    spy_traded: bool,
    other_assets_traded: bool,
) -> tuple[str, str]:
    if not previous or not following:
        return "LISTING_OR_DELISTING_BOUNDARY", "missing row lies at observed availability boundary"
    if stooq_has_bar and spy_traded:
        return "PROVIDER_OMISSION", "Stooq and SPY have bars for the session but Alpaca has no symbol bar"
    if spy_traded and other_assets_traded:
        return "LEGITIMATE_NO_BAR", "market was open; symbol likely had no reportable daily bar in Alpaca SIP"
    if not spy_traded:
        return "REQUEST_BOUNDARY_ISSUE", "SPY does not trade on the expected session grid"
    return "UNEXPLAINED_MISSING_SESSION", f"{symbol} lacks an Alpaca bar on {session}"


def _valid_ohlc(row: Mapping[str, Any]) -> bool:
    values = [_float(row.get(field)) for field in ("open", "high", "low", "close")]
    if any(value is None for value in values):
        return False
    open_, high, low, close = values
    return bool(low <= open_ <= high and low <= close <= high)


def _missing_source_indicator(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> str:
    if left is None and right is None:
        return "both_missing"
    if left is None:
        return "missing_alpaca"
    if right is None:
        return "missing_stooq"
    return "present_both"


def _stats(values: Sequence[float | None]) -> dict[str, float]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "median": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
        "maximum": max(clean, default=0.0),
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


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _session_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    return None


def _date_text(value: Any) -> str:
    session = _session_date(value)
    return session.isoformat() if session else ""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Alpaca daily archive and reconcile against Stooq.")
    parser.add_argument("--alpaca-archive-root", type=Path, default=DEFAULT_ALPACA_ARCHIVE_ROOT)
    parser.add_argument("--stooq-root", type=Path, default=DEFAULT_STOOQ_ROOT)
    parser.add_argument("--asset-registry", type=Path, default=DEFAULT_ASSET_REGISTRY)
    parser.add_argument("--alias-registry", type=Path, default=DEFAULT_ALIAS_REGISTRY)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_full_universe_reconciliation(
        alpaca_archive_root=args.alpaca_archive_root,
        stooq_root=args.stooq_root,
        asset_registry=args.asset_registry,
        alias_registry=args.alias_registry,
        report_root=args.report_root,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "report_root": result["report_root"],
        "dry_run": result["dry_run"],
        "alpaca_rows": result["audit"]["row_count"],
        "stooq_rows": result["summary"]["stooq_rows_read"],
        "decision": result["compatibility_decision"]["decision"],
        "api_requests_attempted": result["alpaca_api_requests"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
