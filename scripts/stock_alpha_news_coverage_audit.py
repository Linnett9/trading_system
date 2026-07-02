from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources import load_validated_rss_registry
from core.research.ml.stock_level.stock_alpha_news_contract import REQUIRED_NEWS_CONTRACT_COLUMNS


def build_stock_alpha_news_coverage_audit(
    *,
    universe_path: str | Path,
    registry_path: str | Path,
    raw_news_path: str | Path,
    sec_report_paths: Sequence[str | Path] | None = None,
    reports_root: str | Path | None = None,
) -> dict[str, Any]:
    universe_symbols, feeds, registry = load_validated_rss_registry(universe_path, registry_path)
    rows = CsvRowRepository().read(Path(raw_news_path))
    row_symbols = [
        str(row.get("symbol", "")).strip().upper()
        for row in rows
        if str(row.get("symbol", "")).strip()
    ]
    row_symbol_set = set(row_symbols)
    universe_symbol_set = set(universe_symbols)
    rows_by_symbol = _sorted_counter(Counter(row_symbols))
    rows_by_provider = _sorted_counter(Counter(str(row.get("provider", "")).strip() or "unknown" for row in rows))
    rows_by_source = _sorted_counter(Counter(str(row.get("source", "")).strip() or "unknown" for row in rows))
    sec_summary = _sec_report_summary(sec_report_paths or [])
    sec_symbols = set(sec_summary["rows_by_symbol"])
    rss_symbols = row_symbol_set & universe_symbol_set
    combined_symbols = (rss_symbols | sec_symbols) & universe_symbol_set

    timestamps = [_parse_timestamp(str(row.get("published_at_utc", "")).strip()) for row in rows]
    valid_timestamps = [value for value in timestamps if value is not None]
    invalid_timestamp_count = len(timestamps) - len(valid_timestamps)
    rows_by_date = _sorted_counter(Counter(value.date().isoformat() for value in valid_timestamps))
    rows_by_month = _sorted_counter(Counter(value.date().isoformat()[:7] for value in valid_timestamps))
    now = datetime.now(timezone.utc)
    future_timestamp_count = sum(1 for value in valid_timestamps if value > now)

    urls = [str(row.get("provider_url", "")).strip() for row in rows if str(row.get("provider_url", "")).strip()]
    headlines = [
        str(row.get("headline", "")).strip().lower()
        for row in rows
        if str(row.get("headline", "")).strip()
    ]
    missing_required_fields = {
        column: sum(1 for row in rows if not str(row.get(column, "")).strip())
        for column in REQUIRED_NEWS_CONTRACT_COLUMNS
    }
    missing_required_fields = {k: v for k, v in missing_required_fields.items() if v}

    verified_feed_symbols = set(registry.get("verified_rss_feed_symbols", []) or [])
    enabled_feed_symbols = set(feeds)
    verified_feed_symbols_without_rows = sorted(verified_feed_symbols - row_symbol_set)
    symbols_with_rows_without_verified_feed = sorted((row_symbol_set - verified_feed_symbols) & universe_symbol_set)
    weak_less_than_5 = sorted(symbol for symbol, count in rows_by_symbol.items() if count < 5)
    weak_less_than_10 = sorted(symbol for symbol, count in rows_by_symbol.items() if count < 10)
    unsafe_reasons = _unsafe_reasons(
        universe_symbol_count=len(universe_symbols),
        raw_symbol_coverage_count=len(row_symbol_set & universe_symbol_set),
        rows=rows,
        valid_timestamps=valid_timestamps,
        weak_less_than_10=weak_less_than_10,
        rows_by_provider=rows_by_provider,
    )

    return {
        "audit_type": "stock_alpha_news_coverage_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "universe_symbol_count": len(universe_symbols),
        "raw_row_count": len(rows),
        "rss_raw_row_count": len(rows),
        "raw_symbol_coverage_count": len(row_symbol_set & universe_symbol_set),
        "rss_symbol_coverage_count": len(rss_symbols),
        "raw_symbol_coverage_rate": (len(row_symbol_set & universe_symbol_set) / len(universe_symbols)) if universe_symbols else 0.0,
        "sec_dry_run_row_count": sec_summary["row_count"],
        "sec_symbol_coverage_count": len(sec_symbols & universe_symbol_set),
        "combined_official_symbol_coverage_count": len(combined_symbols),
        "combined_official_symbol_coverage_rate": (len(combined_symbols) / len(universe_symbols)) if universe_symbols else 0.0,
        "symbols_covered_by_rss_only": sorted(rss_symbols - sec_symbols),
        "symbols_covered_by_sec_only": sorted((sec_symbols - rss_symbols) & universe_symbol_set),
        "symbols_covered_by_both": sorted(rss_symbols & sec_symbols),
        "symbols_with_no_official_rows": sorted(universe_symbol_set - combined_symbols),
        "rows_by_symbol_by_provider": {
            "company_press_release_rss": rows_by_symbol,
            "sec_company_filings": sec_summary["rows_by_symbol"],
        },
        "rows_by_month_by_provider": {
            "company_press_release_rss": rows_by_month,
            "sec_company_filings": sec_summary["rows_by_month"],
        },
        "forms_by_symbol": sec_summary["forms_by_symbol"],
        "forms_by_type": sec_summary["forms_by_type"],
        "official_provider_mix": {
            "company_press_release_rss": len(rss_symbols),
            "sec_company_filings": len(sec_symbols & universe_symbol_set),
        },
        "verified_rss_feed_count": int(registry.get("classification_counts", {}).get("verified_rss_feed", 0)),
        "enabled_feed_symbol_count": len(enabled_feed_symbols),
        "verified_feed_symbols_without_rows": verified_feed_symbols_without_rows,
        "symbols_with_rows_without_verified_feed": symbols_with_rows_without_verified_feed,
        "rows_by_symbol": rows_by_symbol,
        "rows_by_date": rows_by_date,
        "rows_by_month": rows_by_month,
        "rows_by_provider": rows_by_provider,
        "rows_by_source_name_or_feed": rows_by_source,
        "symbols_with_1_row": sorted(symbol for symbol, count in rows_by_symbol.items() if count == 1),
        "symbols_with_2_rows": sorted(symbol for symbol, count in rows_by_symbol.items() if count == 2),
        "symbols_with_less_than_5_rows": weak_less_than_5,
        "symbols_with_less_than_10_rows": weak_less_than_10,
        "top_symbols_by_rows": _top_items(rows_by_symbol, reverse=True),
        "bottom_symbols_by_rows": _top_items(rows_by_symbol, reverse=False),
        "published_at_min": min((value.isoformat().replace("+00:00", "Z") for value in valid_timestamps), default=""),
        "published_at_max": max((value.isoformat().replace("+00:00", "Z") for value in valid_timestamps), default=""),
        "missing_required_fields": missing_required_fields,
        "duplicate_provider_url_count": len(urls) - len(set(urls)),
        "duplicate_provider_url_rate": ((len(urls) - len(set(urls))) / len(rows)) if rows else 0.0,
        "duplicate_headline_count": len(headlines) - len(set(headlines)),
        "duplicate_headline_rate": ((len(headlines) - len(set(headlines))) / len(rows)) if rows else 0.0,
        "future_timestamp_count": future_timestamp_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "provider_error_summary_from_reports_if_available": _provider_error_summary(reports_root),
        "sec_reports_included": [str(path) for path in sec_report_paths or []],
        "known_error_feed_symbols": list(registry.get("known_error_feed_symbols", []) or []),
        "disabled_pending_review_count": int(registry.get("classification_counts", {}).get("disabled_pending_review", 0)),
        "safe_for_feature_generation": not unsafe_reasons,
        "unsafe_reasons": unsafe_reasons,
        "recommended_next_steps": _recommended_next_steps(unsafe_reasons),
    }


def write_stock_alpha_news_coverage_audit(
    *,
    universe_path: str | Path,
    registry_path: str | Path,
    raw_news_path: str | Path,
    output_path: str | Path,
    sec_report_paths: Sequence[str | Path] | None = None,
    reports_root: str | Path | None = None,
) -> Path:
    payload = build_stock_alpha_news_coverage_audit(
        universe_path=universe_path,
        registry_path=registry_path,
        raw_news_path=raw_news_path,
        sec_report_paths=sec_report_paths,
        reports_root=reports_root,
    )
    output = Path(output_path)
    ResearchArtifactWriter().write_json(output, payload)
    return output


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _top_items(values: Mapping[str, int], *, reverse: bool) -> list[dict[str, Any]]:
    return [
        {"symbol": symbol, "row_count": count}
        for symbol, count in sorted(values.items(), key=lambda item: (item[1], item[0]), reverse=reverse)[:20]
    ]


def _unsafe_reasons(
    *,
    universe_symbol_count: int,
    raw_symbol_coverage_count: int,
    rows: list[dict[str, str]],
    valid_timestamps: list[datetime],
    weak_less_than_10: list[str],
    rows_by_provider: Mapping[str, int],
) -> list[str]:
    reasons: list[str] = []
    coverage_rate = raw_symbol_coverage_count / universe_symbol_count if universe_symbol_count else 0.0
    if coverage_rate < 0.80:
        reasons.append("coverage below 80% of canonical universe")
    if len(rows) < 10_000:
        reasons.append("raw row count below 10000-row feature-generation floor")
    if valid_timestamps:
        history_days = (max(valid_timestamps) - min(valid_timestamps)).days
        if history_days < 365:
            reasons.append("history window below 365 days")
    else:
        reasons.append("no valid published_at_utc timestamps")
    if len(weak_less_than_10) > 0:
        reasons.append("one or more covered symbols have fewer than 10 rows")
    if set(rows_by_provider) <= {"company_press_release_rss"}:
        reasons.append("RSS-only source concentration")
    reasons.append("provider/contract ingest gate has not approved this dataset for features")
    return reasons


def _recommended_next_steps(unsafe_reasons: list[str]) -> list[str]:
    steps = [
        "Keep news_analysis_transformer disabled.",
        "Expand official issuer coverage with SEC company filings before feature generation.",
        "Review repeated feed errors and symbols with fewer than 10 rows.",
    ]
    if unsafe_reasons:
        steps.append("Rerun this audit after adding official SEC rows and broader historical coverage.")
    return steps


def _provider_error_summary(reports_root: str | Path | None) -> dict[str, Any]:
    if not reports_root:
        return {}
    root = Path(reports_root)
    if not root.exists():
        return {}
    symbols_with_feed_errors: Counter[str] = Counter()
    providers_failed: Counter[str] = Counter()
    for path in root.rglob("stock_alpha_news_free_source_collect.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for symbol in payload.get("symbols_with_feed_errors", []) or []:
            symbols_with_feed_errors[str(symbol)] += 1
        for provider in (payload.get("providers_failed", {}) or {}):
            providers_failed[str(provider)] += 1
    return {
        "symbols_with_feed_errors": _sorted_counter(symbols_with_feed_errors),
        "providers_failed": _sorted_counter(providers_failed),
    }


def _sec_report_summary(paths: Sequence[str | Path]) -> dict[str, Any]:
    rows_by_symbol: Counter[str] = Counter()
    forms_by_type: Counter[str] = Counter()
    forms_by_symbol: dict[str, dict[str, int]] = {}
    rows_by_month: Counter[str] = Counter()
    row_count = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("provider_row_counts", {}).get("sec_company_filings", 0) == 0:
            continue
        symbol_counts = {
            str(symbol).strip().upper(): int(count)
            for symbol, count in (payload.get("rows_by_symbol", {}) or {}).items()
            if str(symbol).strip()
        }
        for symbol, count in symbol_counts.items():
            rows_by_symbol[symbol] += count
            row_count += count
        for form_type, count in (payload.get("rows_by_form_type", {}) or {}).items():
            forms_by_type[str(form_type)] += int(count)
        # The dry-run report does not persist row-level SEC data; retain an
        # explicit empty per-symbol form map rather than guessing allocation.
        for symbol in symbol_counts:
            forms_by_symbol.setdefault(symbol, {})
    return {
        "row_count": row_count,
        "rows_by_symbol": _sorted_counter(rows_by_symbol),
        "rows_by_month": _sorted_counter(rows_by_month),
        "forms_by_symbol": {symbol: forms_by_symbol[symbol] for symbol in sorted(forms_by_symbol)},
        "forms_by_type": _sorted_counter(forms_by_type),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit stock-alpha raw news coverage without generating features.")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--raw-news", required=True)
    parser.add_argument("--sec-report", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args(argv)
    path = write_stock_alpha_news_coverage_audit(
        universe_path=args.universe,
        registry_path=args.registry,
        raw_news_path=args.raw_news,
        sec_report_paths=args.sec_report,
        output_path=args.output,
        reports_root=args.reports_root,
    )
    print(f"Wrote stock-alpha news coverage audit: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
