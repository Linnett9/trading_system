from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_REPORT_ROOT = Path("reports/data_lineage/alpaca_daily_full_universe_reconciliation")
DEFAULT_RECONCILIATION = DEFAULT_REPORT_ROOT / "provider_reconciliation.csv"
OPEN_HIGH_LOW_REL_THRESHOLD = 0.005
OPEN_HIGH_LOW_ABS_THRESHOLD = 1.0
CLOSE_REL_THRESHOLD = 0.005
CLOSE_ABS_THRESHOLD = 1.0
RETURN_THRESHOLD = 0.001
EXTREME_RETURN_THRESHOLD = 0.002
GENUINE_RETURN_THRESHOLD = 0.002
VOLUME_REL_THRESHOLD = 0.25
STABLE_RATIO_MAD_MAX = 0.003
STABLE_RATIO_P95_DEV_MAX = 0.01
STABLE_RATIO_RETURN_P95_MAX = 0.0025
RATIO_REGIME_JUMP_THRESHOLD = 0.02
MIN_STABLE_OVERLAP_ROWS = 20


REVISED_FIELDS = (
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
    "original_classification",
    "trigger_reasons",
    "price_ratio",
    "price_ratio_deviation",
    "symbol_median_price_ratio",
    "symbol_ratio_regime",
)


TRIGGER_COUNT_FIELDS = ("trigger", "row_count")
TRIGGER_BY_SYMBOL_FIELDS = ("canonical_symbol", "trigger", "row_count")
REGIME_FIELDS = (
    "canonical_symbol",
    "overlap_row_count",
    "median_price_ratio",
    "mad_log_price_ratio",
    "p95_ratio_deviation",
    "ratio_regime_changes",
    "median_return_difference",
    "p95_return_difference",
    "maximum_return_difference",
    "stable_price_level_offset",
    "regime_status",
)
REGIME_CHANGE_FIELDS = (
    "canonical_symbol",
    "session_date",
    "previous_session_date",
    "previous_price_ratio",
    "price_ratio",
    "ratio_change",
    "return_abs_diff",
    "classification_hint",
)
EXTREME_RETURN_FIELDS = (
    "canonical_symbol",
    "session_date",
    "previous_session_date",
    "stooq_previous_close",
    "stooq_current_close",
    "alpaca_previous_close",
    "alpaca_current_close",
    "stooq_return",
    "alpaca_return",
    "return_difference",
    "price_ratio_change",
    "classification_trigger",
    "diagnosis",
)
BOUNDARY_FIELDS = ("session_date", "row_count", "classification", "explanation")
ALPACA_ONLY_FIELDS = ("session_date", "row_count", "classification", "explanation")
WORST_CONTEXT_FIELDS = (
    "canonical_symbol",
    "session_date",
    "context_role",
    "alpaca_close",
    "stooq_close",
    "alpaca_return",
    "stooq_return",
    "return_abs_diff",
    "close_rel_diff",
    "volume_rel_diff",
    "price_ratio",
    "revised_classification",
    "trigger_reasons",
)


def run_reclassification_diagnostics(
    *,
    reconciliation_path: Path = DEFAULT_RECONCILIATION,
    report_root: Path = DEFAULT_REPORT_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows = read_reconciliation(reconciliation_path)
    symbol_stats = price_ratio_regime_statistics(rows)
    stats_by_symbol = {row["canonical_symbol"]: row for row in symbol_stats}
    regime_changes = price_ratio_regime_changes(rows)
    stooq_boundary = stooq_only_boundary_analysis(rows)
    alpaca_extension = alpaca_only_extension_analysis(rows)
    revised_rows = revised_reconciliation(rows, stats_by_symbol, stooq_boundary, alpaca_extension, regime_changes)
    trigger_counts, trigger_by_symbol = trigger_summaries(revised_rows, original_large_only=True)
    extreme_rows, extreme_summary = extreme_return_review(rows, revised_rows)
    worst_context = worst_symbol_context(revised_rows)
    summary = revised_summary(rows, revised_rows, symbol_stats, trigger_counts, stooq_boundary, alpaca_extension)
    decision = revised_compatibility_decision(summary)
    markdown = render_revised_summary(summary, decision)
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_csv(report_root / "classification_trigger_counts.csv", trigger_counts, TRIGGER_COUNT_FIELDS)
        _write_csv(report_root / "classification_trigger_by_symbol.csv", trigger_by_symbol, TRIGGER_BY_SYMBOL_FIELDS)
        _write_csv(report_root / "price_ratio_regime_statistics.csv", symbol_stats, REGIME_FIELDS)
        _write_csv(report_root / "price_ratio_regime_changes.csv", regime_changes, REGIME_CHANGE_FIELDS)
        _write_csv(report_root / "extreme_return_review.csv", extreme_rows, EXTREME_RETURN_FIELDS)
        _write_json(report_root / "extreme_return_review.json", extreme_summary)
        _write_csv(report_root / "stooq_only_boundary_analysis.csv", stooq_boundary, BOUNDARY_FIELDS)
        _write_csv(report_root / "alpaca_only_extension_analysis.csv", alpaca_extension, ALPACA_ONLY_FIELDS)
        _write_csv(report_root / "worst_symbol_context.csv", worst_context, WORST_CONTEXT_FIELDS)
        _write_csv(report_root / "revised_provider_reconciliation.csv", revised_rows, REVISED_FIELDS)
        _write_json(report_root / "revised_provider_reconciliation_summary.json", summary)
        (report_root / "revised_provider_reconciliation_summary.md").write_text(markdown, encoding="utf-8")
        _write_json(report_root / "revised_compatibility_decision.json", decision)
    return {
        "dry_run": dry_run,
        "report_root": str(report_root),
        "row_count": len(rows),
        "original_large_rows": sum(1 for row in rows if row["classification"] == "LARGE_UNEXPLAINED_DIFFERENCE"),
        "trigger_counts": trigger_counts,
        "trigger_by_symbol": trigger_by_symbol,
        "price_ratio_regime_statistics": symbol_stats,
        "price_ratio_regime_changes": regime_changes,
        "extreme_return_review": extreme_rows,
        "extreme_return_summary": extreme_summary,
        "stooq_only_boundary_analysis": stooq_boundary,
        "alpaca_only_extension_analysis": alpaca_extension,
        "worst_symbol_context": worst_context,
        "summary": summary,
        "compatibility_decision": decision,
        "api_requests_attempted": 0,
        "source_archives_modified": False,
        "canonical_market_data_modified": False,
    }


def read_reconciliation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return sorted(rows, key=lambda row: (row["canonical_symbol"], row["session_date"]))


def large_row_triggers(row: Mapping[str, Any]) -> list[str]:
    triggers: list[str] = []
    for field in ("open", "high", "low", "close"):
        abs_value = _float(row.get(f"{field}_abs_diff"))
        rel_value = _float(row.get(f"{field}_rel_diff"))
        abs_threshold = CLOSE_ABS_THRESHOLD if field == "close" else OPEN_HIGH_LOW_ABS_THRESHOLD
        rel_threshold = CLOSE_REL_THRESHOLD if field == "close" else OPEN_HIGH_LOW_REL_THRESHOLD
        if abs_value is not None and abs_value > abs_threshold:
            triggers.append(f"{field} absolute threshold")
        if rel_value is not None and rel_value > rel_threshold:
            triggers.append(f"{field} relative threshold")
    return_diff = _float(row.get("return_abs_diff"))
    volume_rel = _float(row.get("volume_rel_diff"))
    if row.get("alpaca_return") == "" or row.get("stooq_return") == "":
        triggers.append("missing previous-provider close")
    if return_diff is not None and return_diff > RETURN_THRESHOLD:
        triggers.append("return threshold")
    if volume_rel is not None and volume_rel > VOLUME_REL_THRESHOLD:
        triggers.append("volume threshold")
    if not triggers:
        triggers.append("other")
    return triggers


def price_ratio_regime_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_symbol(rows)
    output = []
    for symbol, symbol_rows in sorted(grouped.items()):
        overlap = [row for row in symbol_rows if _present_both(row) and _price_ratio(row) is not None]
        ratios = [_price_ratio(row) for row in overlap]
        log_ratios = [math.log(value) for value in ratios if value and value > 0]
        returns = [_float(row.get("return_abs_diff")) for row in overlap if _float(row.get("return_abs_diff")) is not None]
        median_ratio = _percentile(ratios, 0.50)
        deviations = [abs(value - median_ratio) for value in ratios]
        median_log = _percentile(log_ratios, 0.50)
        mad_log = _percentile([abs(value - median_log) for value in log_ratios], 0.50)
        changes = _symbol_regime_changes(overlap)
        stable = (
            len(overlap) >= MIN_STABLE_OVERLAP_ROWS
            and mad_log <= STABLE_RATIO_MAD_MAX
            and _percentile(deviations, 0.95) <= STABLE_RATIO_P95_DEV_MAX
            and _percentile(returns, 0.95) <= STABLE_RATIO_RETURN_P95_MAX
        )
        if not overlap:
            status = "no_overlap"
        elif len(changes) == 0 and stable:
            status = "one_stable_ratio"
        elif len(changes) > 0 and _percentile(returns, 0.95) <= STABLE_RATIO_RETURN_P95_MAX:
            status = "multiple_stable_ratio_regimes"
        elif len(changes) > 0:
            status = "sudden_ratio_changes"
        else:
            status = "genuinely_unstable_ratios"
        output.append(
            {
                "canonical_symbol": symbol,
                "overlap_row_count": len(overlap),
                "median_price_ratio": median_ratio,
                "mad_log_price_ratio": mad_log,
                "p95_ratio_deviation": _percentile(deviations, 0.95),
                "ratio_regime_changes": len(changes),
                "median_return_difference": _percentile(returns, 0.50),
                "p95_return_difference": _percentile(returns, 0.95),
                "maximum_return_difference": max(returns, default=0.0),
                "stable_price_level_offset": str(stable and abs(median_ratio - 1.0) > CLOSE_REL_THRESHOLD).lower(),
                "regime_status": status,
            }
        )
    return output


def price_ratio_regime_changes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for symbol, symbol_rows in sorted(_group_by_symbol(rows).items()):
        output.extend(_symbol_regime_changes([row for row in symbol_rows if _present_both(row)]))
    return output


def _symbol_regime_changes(symbol_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(symbol_rows, key=lambda row: row["session_date"])
    changes = []
    previous = None
    for row in ordered:
        ratio = _price_ratio(row)
        if ratio is None:
            continue
        if previous is not None:
            ratio_change = abs(ratio - previous["ratio"])
            if ratio_change > RATIO_REGIME_JUMP_THRESHOLD:
                changes.append(
                    {
                        "canonical_symbol": row["canonical_symbol"],
                        "session_date": row["session_date"],
                        "previous_session_date": previous["session_date"],
                        "previous_price_ratio": previous["ratio"],
                        "price_ratio": ratio,
                        "ratio_change": ratio_change,
                        "return_abs_diff": _float(row.get("return_abs_diff")) or 0.0,
                        "classification_hint": "POSSIBLE_CORPORATE_ACTION" if (_float(row.get("return_abs_diff")) or 0.0) > 0.02 else "POSSIBLE_ADJUSTMENT_TRANSITION",
                    }
                )
        previous = {"session_date": row["session_date"], "ratio": ratio}
    return changes


def stooq_only_boundary_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stooq_only = [row for row in rows if row["classification"] == "STOOQ_ONLY"]
    counts = Counter(row["session_date"] for row in stooq_only)
    sessions = sorted(counts)
    alpaca_min = min(row["session_date"] for row in rows if row["alpaca_present"] == "true")
    output = []
    for session in sessions:
        boundary = session < alpaca_min and counts[session] == len({row["canonical_symbol"] for row in stooq_only if row["session_date"] == session})
        output.append(
            {
                "session_date": session,
                "row_count": counts[session],
                "classification": "STOOQ_ONLY_ARCHIVE_BOUNDARY" if boundary else "STOOQ_ONLY_REVIEW_REQUIRED",
                "explanation": "shared Stooq-only session immediately before Alpaca archive begins" if boundary else "not a clean archive-start boundary",
            }
        )
    return output


def alpaca_only_extension_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_symbol(rows)
    latest_stooq_by_symbol = {
        symbol: max((row["session_date"] for row in symbol_rows if row["stooq_present"] == "true"), default="")
        for symbol, symbol_rows in grouped.items()
    }
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row["classification"] != "ALPACA_ONLY":
            continue
        latest = latest_stooq_by_symbol.get(row["canonical_symbol"], "")
        if latest and row["session_date"] > latest:
            cls = "ALPACA_ONLY_RECENT_EXTENSION"
            explanation = "Alpaca row is after the latest local Stooq session for that symbol"
        elif not latest:
            cls = "MISSING_HISTORICAL_STOOQ_COVERAGE"
            explanation = "No Stooq rows exist for the symbol in this overlap file"
        else:
            cls = "SYMBOL_SPECIFIC_EARLY_STOOQ_ENDPOINT"
            explanation = "Alpaca row occurs before or within symbol-specific local Stooq coverage gaps"
        counts[(row["session_date"], cls + "|" + explanation)] += 1
    return [
        {
            "session_date": session,
            "row_count": count,
            "classification": packed.split("|", 1)[0],
            "explanation": packed.split("|", 1)[1],
        }
        for (session, packed), count in sorted(counts.items())
    ]


def revised_reconciliation(
    rows: Sequence[Mapping[str, Any]],
    stats_by_symbol: Mapping[str, Mapping[str, Any]],
    stooq_boundary: Sequence[Mapping[str, Any]],
    alpaca_extension: Sequence[Mapping[str, Any]],
    regime_changes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stooq_boundary_sessions = {
        row["session_date"] for row in stooq_boundary if row["classification"] == "STOOQ_ONLY_ARCHIVE_BOUNDARY"
    }
    alpaca_extension_sessions = {
        row["session_date"] for row in alpaca_extension if row["classification"] == "ALPACA_ONLY_RECENT_EXTENSION"
    }
    regime_change_keys = {(row["canonical_symbol"], row["session_date"]): row for row in regime_changes}
    output = []
    for row in rows:
        rewritten = dict(row)
        rewritten["original_classification"] = row["classification"]
        triggers = large_row_triggers(row) if row["classification"] == "LARGE_UNEXPLAINED_DIFFERENCE" else base_triggers(row)
        ratio = _price_ratio(row)
        stats = stats_by_symbol.get(row["canonical_symbol"], {})
        median_ratio = _float(stats.get("median_price_ratio"))
        ratio_dev = abs(ratio - median_ratio) if ratio is not None and median_ratio is not None else None
        revised = revised_classification(row, stats, triggers, stooq_boundary_sessions, alpaca_extension_sessions, regime_change_keys)
        if (row["canonical_symbol"], row["session_date"]) in regime_change_keys:
            triggers = sorted(set(triggers + ["adjustment-ratio instability"]))
        rewritten["classification"] = revised
        rewritten["trigger_reasons"] = "|".join(triggers)
        rewritten["price_ratio"] = _csv_value(ratio)
        rewritten["price_ratio_deviation"] = _csv_value(ratio_dev)
        rewritten["symbol_median_price_ratio"] = _csv_value(median_ratio)
        rewritten["symbol_ratio_regime"] = stats.get("regime_status", "")
        output.append(rewritten)
    return output


def base_triggers(row: Mapping[str, Any]) -> list[str]:
    if row["alpaca_present"] != "true" or row["stooq_present"] != "true":
        return [row.get("missing_source_indicator") or "missing source"]
    triggers = []
    if (_float(row.get("return_abs_diff")) or 0.0) > RETURN_THRESHOLD:
        triggers.append("return threshold")
    if (_float(row.get("volume_rel_diff")) or 0.0) > VOLUME_REL_THRESHOLD:
        triggers.append("volume threshold")
    if (_float(row.get("close_rel_diff")) or 0.0) > CLOSE_REL_THRESHOLD:
        triggers.append("close relative threshold")
    return triggers or ["within thresholds"]


def revised_classification(
    row: Mapping[str, Any],
    stats: Mapping[str, Any],
    triggers: Sequence[str],
    stooq_boundary_sessions: set[str],
    alpaca_extension_sessions: set[str],
    regime_change_keys: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    if row["classification"] == "STOOQ_ONLY":
        return "STOOQ_ONLY_ARCHIVE_BOUNDARY" if row["session_date"] in stooq_boundary_sessions else "UNEXPLAINED_REVIEW_REQUIRED"
    if row["classification"] == "ALPACA_ONLY":
        return "ALPACA_ONLY_RECENT_EXTENSION" if row["session_date"] in alpaca_extension_sessions else "UNEXPLAINED_REVIEW_REQUIRED"
    key = (row["canonical_symbol"], row["session_date"])
    return_diff = _float(row.get("return_abs_diff")) or 0.0
    close_rel = _float(row.get("close_rel_diff")) or 0.0
    volume_rel = _float(row.get("volume_rel_diff")) or 0.0
    if row["classification"] == "POSSIBLE_CORPORATE_ACTION" or (key in regime_change_keys and return_diff > 0.02):
        return "POSSIBLE_CORPORATE_ACTION"
    if key in regime_change_keys:
        return "POSSIBLE_ADJUSTMENT_TRANSITION"
    if return_diff > GENUINE_RETURN_THRESHOLD:
        return "GENUINE_LARGE_RETURN_DISAGREEMENT"
    if str(stats.get("stable_price_level_offset")) == "true" and close_rel > CLOSE_REL_THRESHOLD and return_diff <= STABLE_RATIO_RETURN_P95_MAX:
        return "STABLE_PRICE_LEVEL_ADJUSTMENT_DIFFERENCE"
    ohlc = any(trigger.startswith(prefix) for prefix in ("open ", "high ", "low ") for trigger in triggers)
    close = any(trigger.startswith("close ") for trigger in triggers)
    if ohlc and not close and return_diff <= RETURN_THRESHOLD:
        return "OHLC_INTRADAY_VENDOR_DIFFERENCE"
    if volume_rel > VOLUME_REL_THRESHOLD and close_rel <= CLOSE_REL_THRESHOLD and return_diff <= RETURN_THRESHOLD:
        return "VOLUME_DEFINITION_DIFFERENCE"
    if close_rel <= CLOSE_REL_THRESHOLD and return_diff <= RETURN_THRESHOLD:
        return "MATCH" if row["classification"] == "MATCH" else "SMALL_VENDOR_DIFFERENCE"
    if row["classification"] == "POSSIBLE_ADJUSTMENT_DIFFERENCE":
        return "STABLE_PRICE_LEVEL_ADJUSTMENT_DIFFERENCE" if return_diff <= STABLE_RATIO_RETURN_P95_MAX else "POSSIBLE_ADJUSTMENT_TRANSITION"
    return "UNEXPLAINED_REVIEW_REQUIRED"


def trigger_summaries(rows: Sequence[Mapping[str, Any]], *, original_large_only: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_counts: Counter[str] = Counter()
    by_symbol: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if original_large_only and row.get("original_classification") != "LARGE_UNEXPLAINED_DIFFERENCE":
            continue
        triggers = [trigger for trigger in str(row.get("trigger_reasons", "")).split("|") if trigger]
        if not triggers:
            triggers = ["other"]
        for trigger in triggers:
            global_counts[trigger] += 1
            by_symbol[(row["canonical_symbol"], trigger)] += 1
    return (
        [{"trigger": trigger, "row_count": count} for trigger, count in sorted(global_counts.items())],
        [{"canonical_symbol": symbol, "trigger": trigger, "row_count": count} for (symbol, trigger), count in sorted(by_symbol.items())],
    )


def extreme_return_review(rows: Sequence[Mapping[str, Any]], revised_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    revised_by_key = {(row["canonical_symbol"], row["session_date"]): row for row in revised_rows}
    output = []
    for symbol, symbol_rows in sorted(_group_by_symbol(rows).items()):
        ordered = [row for row in sorted(symbol_rows, key=lambda item: item["session_date"]) if _present_both(row)]
        previous = None
        for row in ordered:
            return_diff = _float(row.get("return_abs_diff")) or 0.0
            if previous is not None and return_diff > EXTREME_RETURN_THRESHOLD:
                current_ratio = _price_ratio(row)
                previous_ratio = _price_ratio(previous)
                output.append(
                    {
                        "canonical_symbol": symbol,
                        "session_date": row["session_date"],
                        "previous_session_date": previous["session_date"],
                        "stooq_previous_close": previous["stooq_close"],
                        "stooq_current_close": row["stooq_close"],
                        "alpaca_previous_close": previous["alpaca_close"],
                        "alpaca_current_close": row["alpaca_close"],
                        "stooq_return": row["stooq_return"],
                        "alpaca_return": row["alpaca_return"],
                        "return_difference": return_diff,
                        "price_ratio_change": _csv_value(abs(current_ratio - previous_ratio) if current_ratio is not None and previous_ratio is not None else None),
                        "classification_trigger": revised_by_key[(symbol, row["session_date"])]["trigger_reasons"],
                        "diagnosis": diagnose_extreme_return(row, previous, revised_by_key[(symbol, row["session_date"])]),
                    }
                )
            previous = row
    max_row = max(output, key=lambda row: float(row["return_difference"]), default=None)
    return output, {
        "extreme_return_threshold": EXTREME_RETURN_THRESHOLD,
        "row_count": len(output),
        "maximum_return_difference_row": max_row,
        "diagnosis_counts": dict(sorted(Counter(row["diagnosis"] for row in output).items())),
    }


def diagnose_extreme_return(row: Mapping[str, Any], previous: Mapping[str, Any], revised: Mapping[str, Any]) -> str:
    stooq_prev = _float(previous.get("stooq_close"))
    alpaca_prev = _float(previous.get("alpaca_close"))
    if stooq_prev is None or alpaca_prev is None:
        return "missing/misaligned previous session"
    if abs(stooq_prev) < 1e-9 or abs(alpaca_prev) < 1e-9:
        return "near-zero denominator"
    ratio_change = _float(revised.get("price_ratio_deviation")) or 0.0
    if revised["classification"] == "POSSIBLE_CORPORATE_ACTION":
        return "corporate action"
    if revised["classification"] == "POSSIBLE_ADJUSTMENT_TRANSITION":
        return "adjustment transition"
    if ratio_change > RATIO_REGIME_JUMP_THRESHOLD:
        return "adjustment transition"
    if revised["classification"] == "GENUINE_LARGE_RETURN_DISAGREEMENT":
        return "genuine unexplained provider disagreement"
    return "stable offset or vendor rounding"


def worst_symbol_context(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"DD", "AIV", "UIS", "AD", "ADTN", "SPGI", "HON", "BLK", "BRK-A", "SO", "FDX", "BDN"}
    output = []
    by_symbol = _group_by_symbol(rows)
    for symbol in sorted(wanted):
        ordered = sorted(by_symbol.get(symbol, []), key=lambda row: row["session_date"])
        if not ordered:
            continue
        ranked = sorted(
            [row for row in ordered if _present_both(row)],
            key=lambda row: max(_float(row.get("return_abs_diff")) or 0.0, _float(row.get("close_rel_diff")) or 0.0, _float(row.get("volume_rel_diff")) or 0.0),
            reverse=True,
        )
        target_sessions = {row["session_date"] for row in ranked[:3]}
        for index, row in enumerate(ordered):
            if row["session_date"] in target_sessions:
                for neighbor in ordered[max(0, index - 1): min(len(ordered), index + 2)]:
                    output.append(
                        {
                            "canonical_symbol": symbol,
                            "session_date": neighbor["session_date"],
                            "context_role": "target" if neighbor["session_date"] == row["session_date"] else "neighbor",
                            "alpaca_close": neighbor.get("alpaca_close", ""),
                            "stooq_close": neighbor.get("stooq_close", ""),
                            "alpaca_return": neighbor.get("alpaca_return", ""),
                            "stooq_return": neighbor.get("stooq_return", ""),
                            "return_abs_diff": neighbor.get("return_abs_diff", ""),
                            "close_rel_diff": neighbor.get("close_rel_diff", ""),
                            "volume_rel_diff": neighbor.get("volume_rel_diff", ""),
                            "price_ratio": neighbor.get("price_ratio", ""),
                            "revised_classification": neighbor.get("classification", ""),
                            "trigger_reasons": neighbor.get("trigger_reasons", ""),
                        }
                    )
    dedup = {(row["canonical_symbol"], row["session_date"], row["context_role"]): row for row in output}
    return [dedup[key] for key in sorted(dedup)]


def revised_summary(
    original_rows: Sequence[Mapping[str, Any]],
    revised_rows: Sequence[Mapping[str, Any]],
    symbol_stats: Sequence[Mapping[str, Any]],
    trigger_counts: Sequence[Mapping[str, Any]],
    stooq_boundary: Sequence[Mapping[str, Any]],
    alpaca_extension: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    revised_counts = Counter(row["classification"] for row in revised_rows)
    original_large = [row for row in revised_rows if row["original_classification"] == "LARGE_UNEXPLAINED_DIFFERENCE"]
    only_ohlc = [
        row for row in original_large
        if any(token.startswith(("open ", "high ", "low ")) for token in row["trigger_reasons"].split("|"))
        and not any(token.startswith("close ") or token == "return threshold" for token in row["trigger_reasons"].split("|"))
    ]
    return {
        "schema_version": "alpaca_daily_reclassification_diagnostics.v1",
        "row_count": len(revised_rows),
        "original_classification_counts": dict(sorted(Counter(row["classification"] for row in original_rows).items())),
        "revised_classification_counts": dict(sorted(revised_counts.items())),
        "original_large_rows": len(original_large),
        "original_large_close_trigger_rows": sum(1 for row in original_large if "close relative threshold" in row["trigger_reasons"] or "close absolute threshold" in row["trigger_reasons"]),
        "original_large_only_open_high_low_trigger_rows": len(only_ohlc),
        "original_large_return_trigger_rows": sum(1 for row in original_large if "return threshold" in row["trigger_reasons"]),
        "original_large_volume_trigger_rows": sum(1 for row in original_large if "volume threshold" in row["trigger_reasons"]),
        "stable_price_level_adjustment_rows": revised_counts.get("STABLE_PRICE_LEVEL_ADJUSTMENT_DIFFERENCE", 0),
        "genuine_large_return_disagreement_rows": revised_counts.get("GENUINE_LARGE_RETURN_DISAGREEMENT", 0),
        "trigger_counts": list(trigger_counts),
        "price_ratio_regime_counts": dict(sorted(Counter(row["regime_status"] for row in symbol_stats).items())),
        "unstable_price_ratio_symbols": sorted(row["canonical_symbol"] for row in symbol_stats if row["regime_status"] in {"sudden_ratio_changes", "genuinely_unstable_ratios"}),
        "stable_price_level_symbols": sorted(row["canonical_symbol"] for row in symbol_stats if row["stable_price_level_offset"] == "true"),
        "stooq_only_boundary_distribution": stooq_boundary,
        "alpaca_only_extension_distribution": alpaca_extension,
        "api_requests_attempted": 0,
        "source_archives_modified": False,
        "canonical_market_data_modified": False,
    }


def revised_compatibility_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    counts = summary["revised_classification_counts"]
    genuine = int(counts.get("GENUINE_LARGE_RETURN_DISAGREEMENT", 0))
    transitions = int(counts.get("POSSIBLE_ADJUSTMENT_TRANSITION", 0))
    corp = int(counts.get("POSSIBLE_CORPORATE_ACTION", 0))
    if genuine == 0 and transitions == 0 and corp == 0:
        decision = "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE_WITH_CONTROLS"
    elif genuine < 500 and corp <= 10:
        decision = "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE_WITH_CONTROLS"
    else:
        decision = "FULL_UNIVERSE_REVIEW_REQUIRED"
    return {
        "decision": decision,
        "canonical_daily_market_dataset_v2_approved_for_construction": decision in {
            "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE",
            "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE_WITH_CONTROLS",
        },
        "evidence": {
            "genuine_large_return_disagreement_rows": genuine,
            "possible_adjustment_transition_rows": transitions,
            "possible_corporate_action_rows": corp,
            "stable_price_level_adjustment_rows": int(counts.get("STABLE_PRICE_LEVEL_ADJUSTMENT_DIFFERENCE", 0)),
            "stooq_only_archive_boundary_rows": int(counts.get("STOOQ_ONLY_ARCHIVE_BOUNDARY", 0)),
            "alpaca_only_recent_extension_rows": int(counts.get("ALPACA_ONLY_RECENT_EXTENSION", 0)),
        },
        "provider_splicing_controls_required": [
            "retain provider provenance on every row",
            "add provider_transition_flag around the splice boundary",
            "compute returns within provider-adjusted close series before splice validation",
            "use provider-local rolling volume percentile and z-score",
            "avoid raw cross-provider volume ratios near transition",
            "quarantine symbols with genuine large return disagreement pending review",
        ],
    }


def render_revised_summary(summary: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    lines = [
        "# Revised Provider Reconciliation Summary",
        "",
        f"- Original large rows: {summary['original_large_rows']}",
        f"- Revised decision: {decision['decision']}",
        f"- Genuine large return disagreement rows: {summary['genuine_large_return_disagreement_rows']}",
        f"- Stable price-level adjustment rows: {summary['stable_price_level_adjustment_rows']}",
        "",
        "## Revised Classification Counts",
        "",
    ]
    for key, value in summary["revised_classification_counts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _present_both(row: Mapping[str, Any]) -> bool:
    return row.get("alpaca_present") == "true" and row.get("stooq_present") == "true"


def _price_ratio(row: Mapping[str, Any]) -> float | None:
    alpaca = _float(row.get("alpaca_close"))
    stooq = _float(row.get("stooq_close"))
    if alpaca is None or stooq in (None, 0.0):
        return None
    return alpaca / stooq


def _group_by_symbol(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_symbol"])].append(row)
    return grouped


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _percentile(values: Sequence[float | None], quantile: float) -> float:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose and reclassify Alpaca/Stooq reconciliation differences.")
    parser.add_argument("--reconciliation-path", type=Path, default=DEFAULT_RECONCILIATION)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_reclassification_diagnostics(
        reconciliation_path=args.reconciliation_path,
        report_root=args.report_root,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "report_root": result["report_root"],
        "dry_run": result["dry_run"],
        "original_large_rows": result["original_large_rows"],
        "decision": result["compatibility_decision"]["decision"],
        "api_requests_attempted": result["api_requests_attempted"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
