from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.target_authority import (
    INELIGIBLE_SOURCE_BAR,
    MATURED_VALID,
    MISSING_SOURCE_BAR,
    QUARANTINED_SOURCE_BAR,
    RIGHT_CENSORED,
    SESSION_BOUNDARY_CONFLICT,
    TARGET_CATALOGUE_VERSION,
    UNKNOWN_SOURCE_GAP,
    build_target_manifest,
    calculate_target,
    canonical_hash,
    normalise_target_bars,
    resolve_target_contract,
    target_catalogue_payload,
)
from infrastructure.data.calendar_authority import default_calendar_authority


DEFAULT_OUTPUT_DIR = Path("docs/audits/ticket_71a")
DEFAULT_SOURCE_ROOT = Path("data/processed/alpaca/symbol_bars/sip/5m")
REQUESTED_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "AVGO",
    "JPM",
    "XOM",
    "SPY",
    "QQQ",
    "IWM",
)
DECISION_DATES = (
    date(2024, 1, 2),
    date(2024, 1, 5),
    date(2024, 1, 12),
    date(2024, 1, 31),
    date(2024, 3, 8),
    date(2024, 7, 3),
)
TARGET_IDS = (
    "forward_return_30m__decision_5m",
    "forward_return_60m__decision_5m",
    "forward_return_to_close__decision_5m",
    "forward_return_next_open__decision_5m",
)
GENERATED_AT = "2026-07-29T00:00:00Z"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded Ticket 71A five-minute target timing pilot."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-symbols", type=int, default=14)
    args = parser.parse_args(argv)
    result = run_pilot(
        output_dir=args.output_dir,
        source_root=args.source_root,
        year=args.year,
        max_symbols=args.max_symbols,
    )
    print(json.dumps({"classification": result["classification"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_pilot(
    *,
    output_dir: Path,
    source_root: Path,
    year: int,
    max_symbols: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_symbols, missing_symbols = _resolve_symbols(source_root, year, max_symbols)
    source_dates = _source_dates()
    source_by_symbol, source_manifest = _load_source_rows(
        source_root,
        year=year,
        symbols=selected_symbols,
        source_dates=source_dates,
        missing_symbols=missing_symbols,
    )
    source_cutoff = max(
        _bar_end_timestamp(row)
        for rows in source_by_symbol.values()
        for row in rows
    )
    pilot_config = _pilot_config(
        source_root=source_root,
        year=year,
        selected_symbols=selected_symbols,
        missing_symbols=missing_symbols,
        source_dates=source_dates,
        source_cutoff=source_cutoff,
    )
    target_rows = _target_rows(source_by_symbol, source_cutoff=source_cutoff)

    rows_path = output_dir / "five_minute_target_rows.parquet"
    _write_parquet(rows_path, target_rows)
    source_manifest["target_row_count"] = len(target_rows)
    source_manifest["target_rows_sha256"] = _file_sha256(rows_path)

    coverage_by_contract = _coverage_by_contract(target_rows)
    coverage_by_symbol = _coverage_by_symbol(target_rows)
    coverage_by_date = _coverage_by_date(target_rows)
    session_validation = _session_boundary_validation(target_rows, source_by_symbol, source_dates)
    pit_validation = _pit_validation(target_rows)
    missing_bar_analysis = _missing_bar_analysis(source_by_symbol, source_dates)
    manifest = build_target_manifest(
        target_rows,
        selected_target=resolve_target_contract("forward_return_30m__decision_5m"),
        source_cutoff=source_cutoff,
        output_paths=(rows_path,),
        source_paths=source_manifest["source_paths_raw"],
        configuration=pilot_config,
        calendar_identity=default_calendar_authority().identity(
            start=min(source_dates).isoformat(),
            end=max(source_dates).isoformat(),
        ),
        producer_command="python scripts/ticket_71a_bounded_five_minute_target_pilot.py",
        producer_module="scripts.ticket_71a_bounded_five_minute_target_pilot",
    )
    source_manifest.pop("source_paths_raw", None)

    _write_json(output_dir / "pilot_config.json", pilot_config)
    _write_json(output_dir / "five_minute_source_manifest.json", source_manifest)
    _write_csv(output_dir / "target_coverage_by_contract.csv", coverage_by_contract, _coverage_contract_fields())
    _write_csv(output_dir / "target_coverage_by_symbol.csv", coverage_by_symbol, _coverage_symbol_fields())
    _write_csv(output_dir / "target_coverage_by_date.csv", coverage_by_date, _coverage_date_fields())
    _write_json(output_dir / "session_boundary_validation.json", session_validation)
    _write_json(output_dir / "pit_validation.json", pit_validation)
    _write_csv(output_dir / "missing_bar_analysis.csv", missing_bar_analysis, _missing_fields())
    _write_json(output_dir / "target_manifest_example.json", manifest)

    classification = _classification(
        selected_symbols=selected_symbols,
        pit_validation=pit_validation,
        session_validation=session_validation,
    )
    summary = _summary_markdown(
        classification=classification,
        pilot_config=pilot_config,
        source_manifest=source_manifest,
        coverage_by_contract=coverage_by_contract,
        session_validation=session_validation,
        pit_validation=pit_validation,
        missing_bar_analysis=missing_bar_analysis,
        output_dir=output_dir,
    )
    (output_dir / "ticket_71a_summary.md").write_text(summary, encoding="utf-8")
    return {"classification": classification, "output_dir": str(output_dir)}


def _resolve_symbols(source_root: Path, year: int, max_symbols: int) -> tuple[list[str], list[str]]:
    selected = []
    missing = []
    for symbol in REQUESTED_SYMBOLS:
        path = source_root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
        if path.exists():
            selected.append(symbol)
        else:
            missing.append(symbol)
        if len(selected) >= max_symbols:
            break
    return selected, missing


def _source_dates() -> list[date]:
    authority = default_calendar_authority()
    dates = set(DECISION_DATES)
    for day in DECISION_DATES:
        next_day = authority.next_session(day)
        if next_day is not None:
            dates.add(next_day)
    return sorted(dates)


def _load_source_rows(
    source_root: Path,
    *,
    year: int,
    symbols: Sequence[str],
    source_dates: Sequence[date],
    missing_symbols: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    wanted = {day.isoformat() for day in source_dates}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    source_paths = []
    row_counts = {}
    versions = defaultdict(set)
    feeds = defaultdict(set)
    for symbol in symbols:
        path = source_root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
        table = pq.ParquetFile(path).read(columns=[
            "asset_id",
            "canonical_symbol",
            "provider_symbol",
            "timestamp_utc",
            "session_date",
            "session_type",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "provider",
            "feed",
            "timeframe",
            "adjustment_policy",
            "raw_chunk_id",
            "source_row_hash",
            "dataset_version",
        ])
        session_mask = pc.is_in(table["session_date"], value_set=pa.array(sorted(wanted)))
        rows = [_normalise_source_row(row) for row in table.filter(session_mask).to_pylist()]
        rows.sort(key=lambda item: str(item["timestamp_utc"]))
        by_symbol[symbol] = rows
        row_counts[symbol] = len(rows)
        source_paths.append(path)
        for row in rows:
            versions[symbol].add(str(row.get("dataset_version") or ""))
            feeds[symbol].add(str(row.get("feed") or ""))
    manifest = {
        "manifest_version": "ticket_71a_five_minute_source_manifest.v1",
        "generated_at": GENERATED_AT,
        "source_root": str(source_root),
        "year": year,
        "requested_symbols": list(REQUESTED_SYMBOLS),
        "selected_symbols": list(symbols),
        "missing_requested_symbols": list(missing_symbols),
        "source_dates": [day.isoformat() for day in source_dates],
        "source_paths": [_file_identity(path) for path in source_paths],
        "source_paths_raw": source_paths,
        "row_counts_by_symbol": row_counts,
        "total_source_rows_loaded": sum(row_counts.values()),
        "dataset_versions_by_symbol": {symbol: sorted(values) for symbol, values in versions.items()},
        "feeds_by_symbol": {symbol: sorted(values) for symbol, values in feeds.items()},
        "bounded": True,
        "whole_archive_scanned": False,
        "model_training_performed": False,
    }
    manifest["content_hash"] = canonical_hash(
        {
            "selected_symbols": manifest["selected_symbols"],
            "source_dates": manifest["source_dates"],
            "row_counts_by_symbol": manifest["row_counts_by_symbol"],
        }
    )
    return by_symbol, manifest


def _normalise_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    timestamp = result.get("timestamp_utc")
    if isinstance(timestamp, datetime):
        result["timestamp_utc"] = _format(timestamp)
    result["timeframe"] = "5m"
    result["bar_status"] = "OK"
    result["source_bar_id"] = str(result.get("source_row_hash") or "")
    return result


def _target_rows(
    source_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_cutoff: datetime,
) -> list[dict[str, Any]]:
    output = []
    authority = default_calendar_authority()
    contracts = {target_id: resolve_target_contract(target_id) for target_id in TARGET_IDS}
    prepared_by_symbol = {
        symbol: normalise_target_bars(
            rows,
            asset_id=symbol,
            timeframe="5m",
            calendar_authority=authority,
        )
        for symbol, rows in source_by_symbol.items()
    }
    for symbol, rows in sorted(source_by_symbol.items()):
        decision_timestamps = _decision_timestamps(rows)
        for decision in decision_timestamps:
            for target_id in TARGET_IDS:
                result = calculate_target(
                    asset_id=symbol,
                    decision_timestamp=decision,
                    bar_source=prepared_by_symbol[symbol],
                    target_contract=contracts[target_id],
                    calendar_authority=authority,
                    source_cutoff=source_cutoff,
                    training_cutoff=source_cutoff,
                ).payload()
                result["row_id"] = f"{symbol}|{target_id}|{result['decision_timestamp']}"
                result["symbol"] = symbol
                result["decision_session_date"] = result["decision_timestamp"][:10]
                result["feature_cutoff"] = result["decision_timestamp"]
                result["feature_columns"] = "feature_cutoff"
                result["target_value_column"] = "value"
                output.append(result)
    return sorted(output, key=lambda row: (row["symbol"], row["decision_timestamp"], row["target_id"]))


def _decision_timestamps(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    by_session: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        if str(row.get("session_type") or "").lower() not in {"rth", "regular"}:
            continue
        by_session[str(row["session_date"])].append(_bar_end_timestamp(row))
    selected = []
    for session_date, values in sorted(by_session.items()):
        ordered = sorted(set(values))
        if session_date not in {day.isoformat() for day in DECISION_DATES}:
            continue
        if not ordered:
            continue
        candidates = {
            ordered[0],
            ordered[min(11, len(ordered) - 1)],
            ordered[min(35, len(ordered) - 1)],
            ordered[max(0, len(ordered) - 12)],
            ordered[max(0, len(ordered) - 6)],
            ordered[-1],
        }
        selected.extend(_format(value) for value in sorted(candidates))
    return selected


def _coverage_by_contract(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["target_id"])].append(row)
    output = []
    for target_id, target_rows in sorted(by_target.items()):
        counts = Counter(str(row.get("target_resolution_classification") or "") for row in target_rows)
        values = [
            float(row["value"])
            for row in target_rows
            if row.get("value") is not None
            and str(row.get("target_resolution_classification")) == MATURED_VALID
        ]
        lags = [
            (_parse(str(row["target_available_timestamp"])) - _parse(str(row["target_end_timestamp"]))).total_seconds()
            for row in target_rows
            if row.get("target_available_timestamp") and row.get("target_end_timestamp")
        ]
        output.append(
            {
                "target_id": target_id,
                "decision_rows": len(target_rows),
                "matured_rows": sum(_truthy(row.get("target_is_mature")) for row in target_rows),
                "trainable_rows": sum(_truthy(row.get("target_is_trainable")) for row in target_rows),
                "right_censored_rows": counts.get(RIGHT_CENSORED, 0),
                "missing_source_bars": counts.get(MISSING_SOURCE_BAR, 0),
                "quarantined_or_ineligible_rows": counts.get(QUARANTINED_SOURCE_BAR, 0) + counts.get(INELIGIBLE_SOURCE_BAR, 0),
                "session_boundary_failures": counts.get(SESSION_BOUNDARY_CONFLICT, 0),
                "unknown_source_gaps": counts.get(UNKNOWN_SOURCE_GAP, 0),
                "target_value_distribution": json.dumps(_distribution(values), sort_keys=True),
                "timing_coverage": json.dumps(_timing_coverage(target_rows), sort_keys=True),
                "availability_lag_seconds": json.dumps(_distribution(lags), sort_keys=True),
                "symbol_count": len({str(row["symbol"]) for row in target_rows}),
                "date_count": len({str(row["decision_session_date"]) for row in target_rows}),
            }
        )
    return output


def _coverage_by_symbol(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row["symbol"])].append(row)
    return [_coverage_row("symbol", symbol, symbol_rows) for symbol, symbol_rows in sorted(by_symbol.items())]


def _coverage_by_date(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row["decision_session_date"])].append(row)
    return [_coverage_row("decision_session_date", day, date_rows) for day, date_rows in sorted(by_date.items())]


def _coverage_row(key: str, value: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("target_resolution_classification") or "") for row in rows)
    return {
        key: value,
        "decision_rows": len(rows),
        "matured_rows": sum(_truthy(row.get("target_is_mature")) for row in rows),
        "trainable_rows": sum(_truthy(row.get("target_is_trainable")) for row in rows),
        "right_censored_rows": counts.get(RIGHT_CENSORED, 0),
        "missing_source_bars": counts.get(MISSING_SOURCE_BAR, 0),
        "quarantined_or_ineligible_rows": counts.get(QUARANTINED_SOURCE_BAR, 0) + counts.get(INELIGIBLE_SOURCE_BAR, 0),
        "session_boundary_failures": counts.get(SESSION_BOUNDARY_CONFLICT, 0),
        "unknown_source_gaps": counts.get(UNKNOWN_SOURCE_GAP, 0),
        "classification_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
    }


def _session_boundary_validation(
    target_rows: Sequence[Mapping[str, Any]],
    source_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    source_dates: Sequence[date],
) -> dict[str, Any]:
    rows_by_case = defaultdict(list)
    for row in target_rows:
        decision = _parse(str(row["decision_timestamp"]))
        session = default_calendar_authority().session(decision.astimezone(timezone.utc).date())
        close = session.close_timestamp.astimezone(timezone.utc) if session.close_timestamp else None
        if close and decision <= close - timedelta(minutes=60):
            rows_by_case["regular_session"].append(row)
        if close and decision >= close - timedelta(minutes=60):
            rows_by_case["final_hour"].append(row)
        if close and decision >= close - timedelta(minutes=30):
            rows_by_case["final_30_minutes"].append(row)
        if str(row.get("target_id")) == "forward_return_next_open__decision_5m":
            rows_by_case["next_session_open"].append(row)
        if row.get("decision_session_date") == "2024-01-05":
            rows_by_case["weekend_rollover"].append(row)
        if row.get("decision_session_date") == "2024-01-12":
            rows_by_case["holiday_rollover"].append(row)
        if row.get("decision_session_date") == "2024-03-08":
            rows_by_case["dst_boundary"].append(row)
        if row.get("decision_session_date") == "2024-07-03":
            rows_by_case["early_close"].append(row)
    missing = _missing_bar_analysis(source_by_symbol, source_dates)
    missing_count = sum(int(row["missing_bar_count"]) for row in missing)
    checks = {
        name: {
            "status": "PASSED" if rows_by_case[name] else "FAILED",
            "row_count": len(rows_by_case[name]),
        }
        for name in (
            "regular_session",
            "final_hour",
            "final_30_minutes",
            "early_close",
            "next_session_open",
            "weekend_rollover",
            "holiday_rollover",
            "dst_boundary",
        )
    }
    checks["missing_five_minute_bars"] = {
        "status": "PASSED" if missing else "FAILED",
        "row_count": len(missing),
        "missing_bar_count": missing_count,
        "classification": "MISSING_SOURCE_BAR" if missing_count else "NO_MISSING_BARS_OBSERVED",
    }
    return {
        "validation_version": "ticket_71a_session_boundary_validation.v1",
        "status": "PASSED" if all(item["status"] == "PASSED" for item in checks.values()) else "FAILED",
        "checks": checks,
        "calendar_identity": default_calendar_authority().identity(
            start=min(source_dates).isoformat(),
            end=max(source_dates).isoformat(),
        ),
    }


def _pit_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations = []
    checked = 0
    excluded_conflicts = 0
    for row in rows:
        if row.get("target_resolution_classification") == SESSION_BOUNDARY_CONFLICT:
            excluded_conflicts += 1
            continue
        decision = _parse(str(row["decision_timestamp"]))
        feature_cutoff = _parse(str(row["feature_cutoff"]))
        end = _parse_optional(row.get("target_end_timestamp"))
        available = _parse_optional(row.get("target_available_timestamp"))
        if end is None or available is None:
            continue
        checked += 1
        if feature_cutoff > decision:
            violations.append({"row_id": row["row_id"], "reason": "feature_cutoff_after_decision"})
        if end <= decision:
            violations.append({"row_id": row["row_id"], "reason": "target_end_not_after_decision"})
        if available < end:
            violations.append({"row_id": row["row_id"], "reason": "target_available_before_target_end"})
        if _truthy(row.get("target_is_trainable")) and available > _parse(str(row["training_cutoff"])):
            violations.append({"row_id": row["row_id"], "reason": "trainable_after_training_cutoff"})
    feature_columns = sorted({name for row in rows for name in str(row.get("feature_columns") or "").split("|") if name})
    leakage_columns = [name for name in feature_columns if "target" in name or name == "value"]
    return {
        "validation_version": "ticket_71a_pit_validation.v1",
        "status": "PASSED" if not violations and not leakage_columns else "FAILED",
        "checked_rows": checked,
        "excluded_session_boundary_conflict_rows": excluded_conflicts,
        "violation_count": len(violations),
        "violations": violations[:100],
        "feature_columns": feature_columns,
        "target_value_feature_leakage_columns": leakage_columns,
        "target_value_feature_leakage": bool(leakage_columns),
        "checked_invariants": [
            "feature_cutoff <= decision_timestamp",
            "target_end > decision_timestamp",
            "target_available >= target_end",
            "target_available <= training_cutoff for trainable rows",
            "target values are not feature columns",
        ],
    }


def _missing_bar_analysis(
    source_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    source_dates: Sequence[date],
) -> list[dict[str, Any]]:
    rows = []
    for symbol, symbol_rows in sorted(source_by_symbol.items()):
        observed_by_date: dict[str, set[str]] = defaultdict(set)
        for row in symbol_rows:
            if str(row.get("session_type") or "").lower() in {"rth", "regular"}:
                observed_by_date[str(row["session_date"])].add(str(row["timestamp_utc"]))
        for day in source_dates:
            expected = _expected_bar_starts(day)
            observed = observed_by_date.get(day.isoformat(), set())
            missing = sorted(set(expected) - observed)
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": day.isoformat(),
                    "expected_regular_bars": len(expected),
                    "observed_regular_bars": len(observed & set(expected)),
                    "missing_bar_count": len(missing),
                    "missing_timestamps": "|".join(missing[:20]),
                    "missing_timestamps_truncated": len(missing) > 20,
                    "classification": "MISSING_SOURCE_BAR" if missing else "NO_MISSING_BARS_OBSERVED",
                }
            )
    return rows


def _expected_bar_starts(day: date) -> list[str]:
    record = default_calendar_authority().session(day)
    if not record.open_timestamp or not record.close_timestamp:
        return []
    current = record.open_timestamp.astimezone(timezone.utc)
    close = record.close_timestamp.astimezone(timezone.utc)
    output = []
    while current < close:
        output.append(_format(current))
        current += timedelta(minutes=5)
    return output


def _pilot_config(
    *,
    source_root: Path,
    year: int,
    selected_symbols: Sequence[str],
    missing_symbols: Sequence[str],
    source_dates: Sequence[date],
    source_cutoff: datetime,
) -> dict[str, Any]:
    catalogue = target_catalogue_payload()
    config = {
        "pilot_version": "ticket_71a_bounded_five_minute_target_pilot.v1",
        "generated_at": GENERATED_AT,
        "depends_on": {
            "ticket": "71",
            "target_catalogue_version": TARGET_CATALOGUE_VERSION,
            "target_catalogue_hash": catalogue["content_hash"],
        },
        "source_root": str(source_root),
        "source_year": year,
        "requested_symbols": list(REQUESTED_SYMBOLS),
        "selected_symbols": list(selected_symbols),
        "missing_requested_symbols": list(missing_symbols),
        "decision_dates": [day.isoformat() for day in DECISION_DATES],
        "source_dates_loaded": [day.isoformat() for day in source_dates],
        "target_ids": list(TARGET_IDS),
        "source_cutoff": _format(source_cutoff),
        "bounded": True,
        "model_training_performed": False,
        "strategy_performance_compared": False,
        "positions_created": False,
        "portfolio_replay_run": False,
        "production_or_paper_trading_altered": False,
    }
    config["content_hash"] = canonical_hash(config)
    return config


def _classification(
    *,
    selected_symbols: Sequence[str],
    pit_validation: Mapping[str, Any],
    session_validation: Mapping[str, Any],
) -> str:
    if not selected_symbols:
        return "BLOCKED_SOURCE_COVERAGE"
    if pit_validation.get("status") != "PASSED":
        return "BLOCKED_PIT"
    if session_validation.get("status") != "PASSED":
        return "BLOCKED_SESSION_SEMANTICS"
    return "FIVE_MINUTE_TARGET_PILOT_VALIDATED"


def _summary_markdown(
    *,
    classification: str,
    pilot_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    coverage_by_contract: Sequence[Mapping[str, Any]],
    session_validation: Mapping[str, Any],
    pit_validation: Mapping[str, Any],
    missing_bar_analysis: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    total_missing = sum(int(row["missing_bar_count"]) for row in missing_bar_analysis)
    lines = [
        "# Ticket 71A Bounded Five-Minute Target Pilot",
        "",
        f"- Classification: `{classification}`",
        f"- Output directory: `{output_dir}`",
        f"- Selected symbols: `{len(pilot_config.get('selected_symbols') or [])}`",
        f"- Source rows loaded: `{source_manifest.get('total_source_rows_loaded')}`",
        f"- Target rows: `{source_manifest.get('target_row_count')}`",
        f"- Session validation: `{session_validation.get('status')}`",
        f"- PIT validation: `{pit_validation.get('status')}`",
        f"- Missing five-minute bars observed: `{total_missing}`",
        "",
        "Coverage by contract:",
    ]
    for row in coverage_by_contract:
        lines.append(
            f"- `{row['target_id']}` rows={row['decision_rows']} trainable={row['trainable_rows']} "
            f"session_boundary_failures={row['session_boundary_failures']} missing={row['missing_source_bars']}"
        )
    lines.extend(
        [
            "",
            "The pilot uses Ticket 71 target contracts only. No fitting, Sharpe calculation, stock ranking, position creation, replay, or production/paper trading changes were performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": mean(ordered),
        "median": median(ordered),
        "max": ordered[-1],
    }


def _timing_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "target_start_present": sum(bool(row.get("target_start_timestamp")) for row in rows),
        "target_end_present": sum(bool(row.get("target_end_timestamp")) for row in rows),
        "target_available_present": sum(bool(row.get("target_available_timestamp")) for row in rows),
    }


def _bar_end_timestamp(row: Mapping[str, Any]) -> datetime:
    timestamp = _parse(str(row["timestamp_utc"]))
    return timestamp + timedelta(minutes=5)


def _parse_optional(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    return _parse(str(value))


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path, compression="zstd")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _coverage_contract_fields() -> tuple[str, ...]:
    return (
        "target_id",
        "decision_rows",
        "matured_rows",
        "trainable_rows",
        "right_censored_rows",
        "missing_source_bars",
        "quarantined_or_ineligible_rows",
        "session_boundary_failures",
        "unknown_source_gaps",
        "target_value_distribution",
        "timing_coverage",
        "availability_lag_seconds",
        "symbol_count",
        "date_count",
    )


def _coverage_symbol_fields() -> tuple[str, ...]:
    return (
        "symbol",
        "decision_rows",
        "matured_rows",
        "trainable_rows",
        "right_censored_rows",
        "missing_source_bars",
        "quarantined_or_ineligible_rows",
        "session_boundary_failures",
        "unknown_source_gaps",
        "classification_counts",
    )


def _coverage_date_fields() -> tuple[str, ...]:
    return (
        "decision_session_date",
        "decision_rows",
        "matured_rows",
        "trainable_rows",
        "right_censored_rows",
        "missing_source_bars",
        "quarantined_or_ineligible_rows",
        "session_boundary_failures",
        "unknown_source_gaps",
        "classification_counts",
    )


def _missing_fields() -> tuple[str, ...]:
    return (
        "symbol",
        "session_date",
        "expected_regular_bars",
        "observed_regular_bars",
        "missing_bar_count",
        "missing_timestamps",
        "missing_timestamps_truncated",
        "classification",
    )


if __name__ == "__main__":
    raise SystemExit(main())
