from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

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
    sec_event_row_paths: Sequence[str | Path] | None = None,
    sec_artifact_selection: str = "all",
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
    all_sec_event_row_paths = list(sec_event_row_paths or [])
    selected_sec_event_row_paths = _select_sec_event_row_paths(all_sec_event_row_paths, sec_artifact_selection)
    sec_artifact_diagnostics = _sec_artifact_selection_diagnostics(
        all_paths=all_sec_event_row_paths,
        selected_paths=selected_sec_event_row_paths,
        selection=sec_artifact_selection,
    )
    sec_summary = _sec_report_summary(sec_report_paths or [], selected_sec_event_row_paths)
    sec_symbols = set(sec_summary["rows_by_symbol"])
    rss_symbols = row_symbol_set & universe_symbol_set
    combined_symbols = (rss_symbols | sec_symbols) & universe_symbol_set
    provider_errors = _provider_error_summary(reports_root)
    sec_cap_starvation = _sec_cap_starvation_diagnostics(reports_root)
    cap_starved_symbols = set(sec_cap_starvation["symbols"])
    official_rows_by_symbol = _official_rows_by_symbol(
        universe_symbols=universe_symbols,
        rss_rows_by_symbol=rows_by_symbol,
        sec_rows_by_symbol=sec_summary["rows_by_symbol"],
    )
    row_depth = _official_row_depth_diagnostics(
        universe_symbols=universe_symbols,
        official_rows_by_symbol=official_rows_by_symbol,
    )
    classifications = _official_source_classifications(
        registry=registry,
    )

    timestamps = [_parse_timestamp(str(row.get("published_at_utc", "")).strip()) for row in rows]
    valid_timestamps = [value for value in timestamps if value is not None]
    invalid_timestamp_count = len(timestamps) - len(valid_timestamps)
    rows_by_date = _sorted_counter(Counter(value.date().isoformat() for value in valid_timestamps))
    rows_by_month = _sorted_counter(Counter(value.date().isoformat()[:7] for value in valid_timestamps))
    official_timestamp_bounds = [
        value for value in valid_timestamps
    ]
    for value in (sec_summary["published_at_min"], sec_summary["published_at_max"]):
        parsed = _parse_timestamp(str(value))
        if parsed is not None:
            official_timestamp_bounds.append(parsed)
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
    weak_less_than_10 = row_depth["symbols_with_1_to_9_valid_official_rows"]
    unsafe_reasons = _unsafe_reasons(
        universe_symbol_count=len(universe_symbols),
        official_symbol_coverage_count=len(combined_symbols),
        official_row_count=sum(official_rows_by_symbol.values()),
        valid_timestamps=official_timestamp_bounds,
        weak_less_than_10=weak_less_than_10,
        rows_by_provider={
            **rows_by_provider,
            **({"sec_company_filings": sec_summary["row_count"]} if sec_summary["row_count"] else {}),
        },
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
        "symbols_with_no_official_rows": row_depth["symbols_with_zero_official_rows"],
        "symbols_with_zero_official_rows": row_depth["symbols_with_zero_official_rows"],
        "symbols_with_1_to_9_valid_official_rows": row_depth["symbols_with_1_to_9_valid_official_rows"],
        "symbols_with_10_plus_valid_official_rows": row_depth["symbols_with_10_plus_valid_official_rows"],
        "zero_row_symbol_count": row_depth["zero_row_symbol_count"],
        "thin_symbol_count_under_10": row_depth["thin_symbol_count_under_10"],
        "covered_symbol_count_10_plus": row_depth["covered_symbol_count_10_plus"],
        "audited_exception_symbols": classifications["audited_exception_symbols"],
        "blocked_etf_or_fund_symbols": classifications["blocked_etf_or_fund_symbols"],
        "known_rss_failure_symbols": classifications["known_rss_failure_symbols"],
        "audited_exception_count": len(classifications["audited_exception_symbols"]),
        "blocked_fund_count": len(classifications["blocked_etf_or_fund_symbols"]),
        "known_rss_failure_count": len(classifications["known_rss_failure_symbols"]),
        "sec_cap_starvation_diagnostics": sec_cap_starvation["diagnostics"],
        "sec_cap_starved_symbols": sorted(cap_starved_symbols),
        "unresolved_sec_cap_starved_symbols": sorted(cap_starved_symbols - combined_symbols),
        "sec_cap_starved_count": len(cap_starved_symbols),
        "unresolved_sec_cap_starved_count": len(cap_starved_symbols - combined_symbols),
        "rows_by_symbol_by_provider": {
            "company_press_release_rss": rows_by_symbol,
            "sec_company_filings": sec_summary["rows_by_symbol"],
        },
        "valid_official_rows_by_symbol": official_rows_by_symbol,
        "rows_by_month_by_provider": {
            "company_press_release_rss": rows_by_month,
            "sec_company_filings": sec_summary["rows_by_month"],
        },
        "forms_by_symbol": sec_summary["forms_by_symbol"],
        "forms_by_type": sec_summary["forms_by_type"],
        "sec_published_at_min": sec_summary["published_at_min"],
        "sec_published_at_max": sec_summary["published_at_max"],
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
        "provider_error_summary_from_reports_if_available": provider_errors,
        "sec_reports_included": [str(path) for path in sec_report_paths or []],
        "sec_artifact_selection": sec_artifact_selection,
        "sec_event_rows_included": [str(path) for path in selected_sec_event_row_paths],
        **sec_artifact_diagnostics,
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
    sec_event_row_paths: Sequence[str | Path] | None = None,
    sec_artifact_selection: str = "all",
    reports_root: str | Path | None = None,
) -> Path:
    payload = build_stock_alpha_news_coverage_audit(
        universe_path=universe_path,
        registry_path=registry_path,
        raw_news_path=raw_news_path,
        sec_report_paths=sec_report_paths,
        sec_event_row_paths=sec_event_row_paths,
        sec_artifact_selection=sec_artifact_selection,
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


def _official_rows_by_symbol(
    *,
    universe_symbols: Sequence[str],
    rss_rows_by_symbol: Mapping[str, int],
    sec_rows_by_symbol: Mapping[str, int],
) -> dict[str, int]:
    universe = {str(symbol).strip().upper() for symbol in universe_symbols if str(symbol).strip()}
    counter: Counter[str] = Counter()
    for source in (rss_rows_by_symbol, sec_rows_by_symbol):
        for symbol, count in source.items():
            normalized = str(symbol).strip().upper()
            if normalized in universe:
                counter[normalized] += int(count)
    return _sorted_counter(counter)


def _official_row_depth_diagnostics(
    *,
    universe_symbols: Sequence[str],
    official_rows_by_symbol: Mapping[str, int],
) -> dict[str, Any]:
    universe = sorted({str(symbol).strip().upper() for symbol in universe_symbols if str(symbol).strip()})
    zero = [symbol for symbol in universe if int(official_rows_by_symbol.get(symbol, 0)) == 0]
    thin = [
        symbol
        for symbol in universe
        if 1 <= int(official_rows_by_symbol.get(symbol, 0)) < 10
    ]
    covered_10_plus = [
        symbol
        for symbol in universe
        if int(official_rows_by_symbol.get(symbol, 0)) >= 10
    ]
    return {
        "symbols_with_zero_official_rows": zero,
        "symbols_with_1_to_9_valid_official_rows": thin,
        "symbols_with_10_plus_valid_official_rows": covered_10_plus,
        "zero_row_symbol_count": len(zero),
        "thin_symbol_count_under_10": len(thin),
        "covered_symbol_count_10_plus": len(covered_10_plus),
    }


def _official_source_classifications(
    *,
    registry: Mapping[str, Any],
) -> dict[str, list[str]]:
    audited_exceptions = _registry_symbols("config/news_source_registry.stock_alpha_official_exceptions.yaml", "exceptions")
    blocked_funds = _registry_symbols("config/news_source_registry.stock_alpha_etf_funds.yaml", "funds")
    known_rss_failures = [
        str(symbol).strip().upper()
        for symbol in (registry.get("known_error_feed_symbols", []) or [])
        if str(symbol).strip()
    ]
    return {
        "audited_exception_symbols": audited_exceptions,
        "blocked_etf_or_fund_symbols": blocked_funds,
        "known_rss_failure_symbols": sorted(set(known_rss_failures)),
    }


def _registry_symbols(path: str | Path, key: str) -> list[str]:
    payload = _read_yaml(Path(path))
    values = payload.get(key, {}) if isinstance(payload, Mapping) else {}
    if isinstance(values, Mapping):
        return sorted(str(symbol).strip().upper() for symbol in values if str(symbol).strip())
    if isinstance(values, list):
        return sorted(str(symbol).strip().upper() for symbol in values if str(symbol).strip())
    return []


def _unsafe_reasons(
    *,
    universe_symbol_count: int,
    official_symbol_coverage_count: int,
    official_row_count: int,
    valid_timestamps: list[datetime],
    weak_less_than_10: list[str],
    rows_by_provider: Mapping[str, int],
) -> list[str]:
    reasons: list[str] = []
    coverage_rate = official_symbol_coverage_count / universe_symbol_count if universe_symbol_count else 0.0
    if coverage_rate < 0.80:
        reasons.append("coverage below 80% of canonical universe")
    if official_row_count < 10_000:
        reasons.append("official row count below 10000-row feature-generation floor")
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


def _sec_report_summary(paths: Sequence[str | Path], event_row_paths: Sequence[str | Path]) -> dict[str, Any]:
    rows_by_symbol: Counter[str] = Counter()
    forms_by_type: Counter[str] = Counter()
    forms_by_symbol_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows_by_month: Counter[str] = Counter()
    row_count = 0
    event_rows = _read_sec_event_rows(event_row_paths)
    if not event_rows:
        for report_path in paths:
            report_payload = _read_json(Path(report_path))
            artifact = str(report_payload.get("sec_company_filings_event_rows_path", "")).strip()
            if artifact:
                event_rows.extend(_read_sec_event_rows([artifact]))
    for row in event_rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        form_type = str(row.get("form_type", "")).strip()
        timestamp = _parse_timestamp(str(row.get("published_at_utc", "")).strip())
        if symbol:
            rows_by_symbol[symbol] += 1
            row_count += 1
        if form_type:
            forms_by_type[form_type] += 1
            if symbol:
                forms_by_symbol_counts[symbol][form_type] += 1
        if timestamp is not None:
            rows_by_month[timestamp.date().isoformat()[:7]] += 1
    if event_rows:
        return {
            "row_count": row_count,
            "rows_by_symbol": _sorted_counter(rows_by_symbol),
            "rows_by_month": _sorted_counter(rows_by_month),
            "forms_by_symbol": {
                symbol: _sorted_counter(counter)
                for symbol, counter in sorted(forms_by_symbol_counts.items())
            },
            "forms_by_type": _sorted_counter(forms_by_type),
            "published_at_min": min(
                (str(row.get("published_at_utc", "")).strip() for row in event_rows if _parse_timestamp(str(row.get("published_at_utc", "")).strip()) is not None),
                default="",
            ),
            "published_at_max": max(
                (str(row.get("published_at_utc", "")).strip() for row in event_rows if _parse_timestamp(str(row.get("published_at_utc", "")).strip()) is not None),
                default="",
            ),
        }
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
        for symbol in symbol_counts:
            forms_by_symbol_counts.setdefault(symbol, Counter())
    return {
        "row_count": row_count,
        "rows_by_symbol": _sorted_counter(rows_by_symbol),
        "rows_by_month": _sorted_counter(rows_by_month),
        "forms_by_symbol": {
            symbol: _sorted_counter(counter)
            for symbol, counter in sorted(forms_by_symbol_counts.items())
        },
        "forms_by_type": _sorted_counter(forms_by_type),
        "published_at_min": "",
        "published_at_max": "",
    }


def _select_sec_event_row_paths(paths: Sequence[str | Path], selection: str) -> list[str | Path]:
    if selection == "all":
        return list(paths)
    if selection not in {"prefer_12mo", "prefer_36mo", "merge_36mo_pilots", "merge_36mo_parts"}:
        raise ValueError("sec_artifact_selection must be 'all', 'prefer_12mo', 'prefer_36mo', 'merge_36mo_pilots', or 'merge_36mo_parts'")
    by_batch, other_paths = _group_sec_event_row_paths(paths)
    if selection == "prefer_36mo":
        return _prefer_36mo_sec_event_row_paths(by_batch, other_paths)
    if selection == "merge_36mo_pilots":
        return _merge_36mo_sec_event_row_paths(by_batch, other_paths, marker="_36mo_pilot")
    if selection == "merge_36mo_parts":
        return _merge_36mo_sec_event_row_paths(by_batch, other_paths, marker="_36mo_part_")
    return _prefer_12mo_sec_event_row_paths(by_batch, other_paths)


def _group_sec_event_row_paths(paths: Sequence[str | Path]) -> tuple[dict[str, list[str | Path]], list[str | Path]]:
    by_batch: dict[str, list[str | Path]] = defaultdict(list)
    other_paths: list[str | Path] = []
    for path in paths:
        path_text = str(path)
        marker = "stock_alpha_news_collect_sec_company_filings_batch_"
        if marker not in path_text:
            other_paths.append(path)
            continue
        batch = path_text.split(marker, 1)[1][:2]
        by_batch[batch].append(path)
    return by_batch, other_paths


def _prefer_12mo_sec_event_row_paths(
    by_batch: Mapping[str, list[str | Path]],
    other_paths: Sequence[str | Path],
) -> list[str | Path]:
    selected: list[str | Path] = []
    for batch in sorted(by_batch):
        batch_paths = by_batch[batch]
        preferred = [path for path in batch_paths if "_12mo_dry_run/" in str(path)]
        selected.extend(preferred or batch_paths)
    selected.extend(
        path for path in other_paths
        if not _is_data_news_path(path) and not _is_36mo_path(path)
    )
    return selected


def _prefer_36mo_sec_event_row_paths(
    by_batch: Mapping[str, list[str | Path]],
    other_paths: Sequence[str | Path],
) -> list[str | Path]:
    selected: list[str | Path] = []
    for batch in sorted(by_batch):
        batch_paths = [path for path in by_batch[batch] if not _is_data_news_path(path)]
        preferred_36mo = [path for path in batch_paths if "_36mo" in str(path)]
        preferred_12mo = [path for path in batch_paths if "_12mo_dry_run/" in str(path)]
        selected.extend(preferred_36mo or preferred_12mo or batch_paths)
    selected.extend(path for path in other_paths if not _is_data_news_path(path))
    return selected


def _merge_36mo_sec_event_row_paths(
    by_batch: Mapping[str, list[str | Path]],
    other_paths: Sequence[str | Path],
    *,
    marker: str,
) -> list[str | Path]:
    baseline = [
        path for path in _prefer_12mo_sec_event_row_paths(by_batch, other_paths)
        if not _is_data_news_path(path)
    ]
    baseline_text = {str(path) for path in baseline}
    overlay: list[str | Path] = []
    for batch in sorted(by_batch):
        overlay.extend(
            path for path in by_batch[batch]
            if marker in str(path) and not _is_data_news_path(path) and str(path) not in baseline_text
        )
    overlay.extend(
        path for path in sorted(other_paths, key=str)
        if marker in str(path) and not _is_data_news_path(path) and str(path) not in baseline_text
    )
    return [*baseline, *overlay]


def _is_data_news_path(path: str | Path) -> bool:
    path_text = str(path).replace("\\", "/")
    return path_text.startswith("data/news/") or "/data/news/" in path_text


def _is_36mo_path(path: str | Path) -> bool:
    return "_36mo" in str(path)


def _sec_artifact_selection_diagnostics(
    *,
    all_paths: Sequence[str | Path],
    selected_paths: Sequence[str | Path],
    selection: str,
) -> dict[str, Any]:
    if selection not in {"merge_36mo_pilots", "merge_36mo_parts"}:
        return {
            "sec_artifact_selection_mode": selection,
            "selected_sec_artifacts": [str(path) for path in selected_paths],
        }
    by_batch, other_paths = _group_sec_event_row_paths(all_paths)
    baseline_paths = [
        path for path in _prefer_12mo_sec_event_row_paths(by_batch, other_paths)
        if not _is_data_news_path(path)
    ]
    baseline_rows = _read_sec_event_rows(baseline_paths)
    selected_rows = _read_sec_event_rows(selected_paths)
    baseline_keys = {_sec_event_key(row) for row in baseline_rows}
    selected_keys = {_sec_event_key(row) for row in selected_rows}
    baseline_counts = _row_counts_by_symbol(baseline_rows)
    selected_counts = _row_counts_by_symbol(selected_rows)
    all_symbols = set(baseline_counts) | set(selected_counts)
    added_artifacts = [str(path) for path in selected_paths if str(path) not in {str(item) for item in baseline_paths}]
    return {
        "sec_artifact_selection_mode": selection,
        "baseline_12mo_sec_row_count": len(baseline_rows),
        "selected_sec_row_count": len(selected_rows),
        "additional_sec_event_key_count": len(selected_keys - baseline_keys),
        "removed_sec_event_key_count": len(baseline_keys - selected_keys),
        "symbols_with_row_count_increase": sorted(
            symbol for symbol in all_symbols
            if selected_counts.get(symbol, 0) > baseline_counts.get(symbol, 0)
        ),
        "symbols_with_row_count_decrease": sorted(
            symbol for symbol in all_symbols
            if selected_counts.get(symbol, 0) < baseline_counts.get(symbol, 0)
        ),
        "selected_sec_artifacts": [str(path) for path in selected_paths],
        "baseline_sec_artifacts": [str(path) for path in baseline_paths],
        "added_sec_artifacts": added_artifacts,
    }


def _row_counts_by_symbol(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol:
            counter[symbol] += 1
    return _sorted_counter(counter)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sec_cap_starvation_diagnostics(reports_root: str | Path | None) -> dict[str, Any]:
    if reports_root is None or not Path(reports_root).exists():
        return {"symbols": [], "diagnostics": []}
    diagnostics: list[dict[str, Any]] = []
    starved_symbols: set[str] = set()
    for path in sorted(Path(reports_root).rglob("stock_alpha_news_free_source_collect.json")):
        payload = _read_json(path)
        configured = {
            str(symbol).strip().upper()
            for symbol in (payload.get("only_symbols", []) or [])
            if str(symbol).strip()
        }
        for batch in payload.get("provider_batch_diagnostics", []) or []:
            if batch.get("provider") != "sec_company_filings":
                continue
            attempted = {
                str(symbol).strip().upper()
                for symbol in (batch.get("sec_company_filings_attempted_symbols", []) or [])
                if str(symbol).strip()
            }
            requested_limit = int(batch.get("requested_limit", 0) or 0)
            response_row_count = int(batch.get("response_row_count", 0) or 0)
            starved = sorted(configured - attempted)
            if not starved or requested_limit <= 0 or response_row_count < requested_limit:
                continue
            starved_symbols.update(starved)
            diagnostics.append({
                "report_path": str(path),
                "provider": "sec_company_filings",
                "requested_limit": requested_limit,
                "response_row_count": response_row_count,
                "configured_symbol_count": len(configured),
                "attempted_symbol_count": len(attempted),
                "cap_starved_symbols": starved,
            })
    return {"symbols": sorted(starved_symbols), "diagnostics": diagnostics}


def _read_sec_event_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        row = dict(value)
                        event_key = _sec_event_key(row)
                        if event_key not in seen:
                            seen.add(event_key)
                            rows.append(row)
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _sec_event_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        "sec_company_filings",
        str(row.get("symbol", "")).strip().upper(),
        str(row.get("source_url") or row.get("filing_url") or row.get("primary_document_url") or "").strip(),
        str(row.get("published_at_utc", "")).strip(),
        str(row.get("accession_number", "")).strip(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit stock-alpha raw news coverage without generating features.")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--raw-news", required=True)
    parser.add_argument("--sec-report", action="append", default=[])
    parser.add_argument("--sec-event-rows", action="append", default=[])
    parser.add_argument("--include-sec-event-rows", action="store_true")
    parser.add_argument("--sec-artifact-selection", choices=["all", "prefer_12mo", "prefer_36mo", "merge_36mo_pilots", "merge_36mo_parts"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args(argv)
    path = write_stock_alpha_news_coverage_audit(
        universe_path=args.universe,
        registry_path=args.registry,
        raw_news_path=args.raw_news,
        sec_report_paths=args.sec_report,
        sec_event_row_paths=[
            *args.sec_event_rows,
            *(
                [
                    str(path)
                    for path in Path(args.reports_root).rglob("sec_company_filings_event_rows.jsonl")
                ]
                if args.include_sec_event_rows and Path(args.reports_root).exists()
                else []
            ),
        ],
        sec_artifact_selection=args.sec_artifact_selection,
        output_path=args.output,
        reports_root=args.reports_root,
    )
    print(f"Wrote stock-alpha news coverage audit: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
