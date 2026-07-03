from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


CLASSIFICATIONS = (
    "success_with_event_rows",
    "success_missing_event_rows",
    "success_zero_rows_clean",
    "timeout_failure",
    "provider_failure",
    "rate_limited",
    "missing_report",
    "unattempted_symbols",
    "unsafe_output_path",
)


def build_sec_event_row_coverage_summary(
    *,
    reports_root: str | Path,
    window_months: int,
    include_retries: bool = False,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    reports_root_path = Path(reports_root)
    config_dir_path = Path(config_dir) if config_dir else None
    reports = _report_paths(reports_root_path, window_months, include_retries)
    items = [_classify_report(path, reports_root_path, window_months) for path in reports]

    if config_dir_path is not None:
        reported_families = {item["family_name"] for item in items}
        for config_path in _config_paths(config_dir_path, window_months, include_retries):
            family_name = config_path.name.removeprefix("config.").removesuffix(".yaml")
            if family_name in reported_families:
                continue
            items.append(_missing_report_item(config_path, reports_root_path, family_name))

    items.sort(key=lambda item: (item["family_name"], item["report_path"]))
    counts = Counter(item["classification"] for item in items)
    timeout_symbols = _symbols_for(items, "timeout_failure")
    provider_failure_symbols = _symbols_for(items, "provider_failure")
    rate_limited_symbols = _symbols_for(items, "rate_limited")
    unattempted_symbols = sorted({symbol for item in items for symbol in item["unattempted_symbols"]})
    missing_event_rows = [
        item for item in items
        if item["classification"] == "success_missing_event_rows"
    ]
    rerun_items = [
        item for item in items
        if item["classification"] in {
            "success_missing_event_rows",
            "timeout_failure",
            "provider_failure",
            "rate_limited",
            "missing_report",
            "unattempted_symbols",
            "unsafe_output_path",
        }
    ]

    return {
        "summary_type": "stock_alpha_sec_event_row_coverage",
        "reports_root": str(reports_root_path),
        "config_dir": str(config_dir_path) if config_dir_path else "",
        "window_months": window_months,
        "include_retries": include_retries,
        "classification_counts": {name: counts.get(name, 0) for name in CLASSIFICATIONS},
        "successful_reports": counts.get("success_with_event_rows", 0) + counts.get("success_zero_rows_clean", 0),
        "successful_reports_missing_event_rows": len(missing_event_rows),
        "timeout_symbols": timeout_symbols,
        "provider_failure_symbols": provider_failure_symbols,
        "rate_limited_symbols": rate_limited_symbols,
        "unattempted_symbols": unattempted_symbols,
        "total_event_rows_found": sum(item["event_row_count"] for item in items),
        "total_provider_rows_reported": sum(item["provider_row_count"] for item in items),
        "touches_data_news": any(item["touches_data_news"] for item in items),
        "mismatch_count": sum(
            1 for item in items
            if item["event_rows_exists"] and item["provider_row_count"] != item["event_row_count"]
        ),
        "reports_that_should_be_rerun": [item["family_name"] for item in rerun_items],
        "symbols_that_should_be_one_symbol_retried": sorted(
            set(timeout_symbols) | set(provider_failure_symbols) | set(rate_limited_symbols) | set(unattempted_symbols)
        ),
        "items": items,
    }


def _report_paths(reports_root: Path, window_months: int, include_retries: bool) -> list[Path]:
    if not reports_root.exists():
        return []
    patterns = [f"stock_alpha_news_collect_sec_company_filings_{window_months}mo_part_*_dry_run/dev/stock_alpha_news_free_source_collect.json"]
    if include_retries:
        patterns.append(f"stock_alpha_news_collect_sec_company_filings_{window_months}mo_retry_*_dry_run/dev/stock_alpha_news_free_source_collect.json")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(reports_root.glob(pattern))
    return sorted(paths)


def _config_paths(config_dir: Path, window_months: int, include_retries: bool) -> list[Path]:
    patterns = [f"config.stock_alpha_news_collect_sec_company_filings_{window_months}mo_part_*_dry_run.yaml"]
    if include_retries:
        patterns.append(f"config.stock_alpha_news_collect_sec_company_filings_{window_months}mo_retry_*_dry_run.yaml")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(config_dir.glob(pattern))
    return sorted(paths)


def _classify_report(path: Path, reports_root: Path, window_months: int) -> dict[str, Any]:
    payload = _read_json(path)
    family_name = path.parents[1].name
    requested = _requested_symbols(payload)
    attempted = _attempted_symbols(payload)
    returned = sorted(_rows_by_symbol(payload))
    zero_return = sorted(set(requested) - set(returned))
    unattempted = sorted(set(requested) - set(attempted))
    output_path = str(payload.get("output_path", "") or "")
    event_rows_path = str(payload.get("sec_company_filings_event_rows_path", "") or "")
    event_path = Path(event_rows_path) if event_rows_path else None
    event_rows_exists = bool(event_path and event_path.exists())
    event_row_count = _jsonl_count(event_path) if event_path and event_path.exists() else 0
    provider_failures = dict(payload.get("providers_failed", {}) or {})
    rate_limited = bool(payload.get("providers_rate_limited") or [])
    timeout = _has_timeout(payload)
    provider_row_count = int((payload.get("provider_row_counts", {}) or {}).get("sec_company_filings", 0) or 0)
    output_written = bool(payload.get("output_written"))
    output_under_reports = _is_reports_path(output_path, reports_root) and (
        not event_rows_path or _is_reports_path(event_rows_path, reports_root)
    )
    touches_data_news = "data/news" in output_path or "data/news" in event_rows_path

    if touches_data_news or not output_under_reports:
        classification = "unsafe_output_path"
    elif rate_limited:
        classification = "rate_limited"
    elif timeout:
        classification = "timeout_failure"
    elif provider_failures:
        classification = "provider_failure"
    elif unattempted:
        classification = "unattempted_symbols"
    elif provider_row_count > 0 and not event_rows_exists:
        classification = "success_missing_event_rows"
    elif provider_row_count == 0:
        classification = "success_zero_rows_clean"
    else:
        classification = "success_with_event_rows"

    return {
        "classification": classification,
        "report_path": str(path),
        "config_path": _config_path_for_family(family_name),
        "family_name": family_name,
        "window_months": window_months,
        "requested_symbols": requested,
        "attempted_symbols": attempted,
        "returned_symbols": returned,
        "zero_return_symbols": zero_return,
        "unattempted_symbols": unattempted,
        "provider_row_count": provider_row_count,
        "event_rows_path": event_rows_path,
        "event_rows_exists": event_rows_exists,
        "event_row_count": event_row_count,
        "provider_failures": provider_failures,
        "timeout": timeout,
        "rate_limited": rate_limited,
        "output_written": output_written,
        "output_path": output_path,
        "output_under_reports": output_under_reports,
        "touches_data_news": touches_data_news,
    }


def _missing_report_item(config_path: Path, reports_root: Path, family_name: str) -> dict[str, Any]:
    payload = _read_yaml(config_path)
    collect = dict(payload.get("ml", {}).get("stock_alpha_news_collect", {}) or {})
    requested = [str(symbol).strip().upper() for symbol in collect.get("only_symbols", []) or [] if str(symbol).strip()]
    return {
        "classification": "missing_report",
        "report_path": str(reports_root / family_name / "dev" / "stock_alpha_news_free_source_collect.json"),
        "config_path": str(config_path),
        "family_name": family_name,
        "window_months": int(str(collect.get("source_window", "0mo")).removesuffix("mo") or 0),
        "requested_symbols": requested,
        "attempted_symbols": [],
        "returned_symbols": [],
        "zero_return_symbols": requested,
        "unattempted_symbols": requested,
        "provider_row_count": 0,
        "event_rows_path": "",
        "event_rows_exists": False,
        "event_row_count": 0,
        "provider_failures": {},
        "timeout": False,
        "rate_limited": False,
        "output_written": False,
        "output_path": "",
        "output_under_reports": False,
        "touches_data_news": False,
    }


def _symbols_for(items: Sequence[Mapping[str, Any]], classification: str) -> list[str]:
    return sorted({
        symbol
        for item in items
        if item["classification"] == classification
        for symbol in item["requested_symbols"]
    })


def _requested_symbols(payload: Mapping[str, Any]) -> list[str]:
    return sorted({
        str(symbol).strip().upper()
        for symbol in (payload.get("only_symbols", []) or payload.get("symbols", []) or [])
        if str(symbol).strip()
    })


def _attempted_symbols(payload: Mapping[str, Any]) -> list[str]:
    attempted: set[str] = set()
    for batch in payload.get("provider_batch_diagnostics", []) or []:
        attempted.update(
            str(symbol).strip().upper()
            for symbol in (batch.get("sec_company_filings_attempted_symbols", []) or batch.get("symbols", []) or [])
            if str(symbol).strip()
        )
    return sorted(attempted)


def _rows_by_symbol(payload: Mapping[str, Any]) -> set[str]:
    return {
        str(symbol).strip().upper()
        for symbol in (payload.get("rows_by_symbol", {}) or {})
        if str(symbol).strip()
    }


def _has_timeout(payload: Mapping[str, Any]) -> bool:
    failure_text = json.dumps(payload.get("providers_failed", {}) or {})
    if "TimeoutError" in failure_text or "timed out" in failure_text.lower():
        return True
    for batch in payload.get("provider_batch_diagnostics", []) or []:
        if str(batch.get("error_type", "")) == "TimeoutError" or "timed out" in json.dumps(batch).lower():
            return True
    return False


def _jsonl_count(path: Path | None) -> int:
    if path is None:
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _config_path_for_family(family_name: str) -> str:
    return f"config/config.{family_name}.yaml"


def _is_reports_path(value: str, reports_root: Path) -> bool:
    if not value:
        return False
    if value.startswith("reports/"):
        return True
    try:
        Path(value).resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, AttributeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit SEC event-row coverage for stock-alpha collector reports.")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--config-dir")
    parser.add_argument("--window-months", type=int, choices=[36, 60, 120], required=True)
    parser.add_argument("--include-retries", action="store_true")
    parser.add_argument("--summary-output")
    args = parser.parse_args(argv)

    summary = build_sec_event_row_coverage_summary(
        reports_root=args.reports_root,
        config_dir=args.config_dir,
        window_months=args.window_months,
        include_retries=args.include_retries,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_output:
        Path(args.summary_output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
