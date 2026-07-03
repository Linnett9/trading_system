from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SYMBOLS = ("AEM", "ASML")


def build_thin_symbol_audit(
    *,
    reports_root: str | Path,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
) -> dict[str, Any]:
    root = Path(reports_root)
    normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    coverage = _latest_json(root, "stock_alpha_news_coverage_audit*.json")
    preflight = _latest_json(root, "stock_alpha_news_contract_ingest_preflight*.json")
    reports = _collector_reports(root)
    rows_by_provider = _rows_by_provider(coverage)
    official_rows = _official_rows(coverage, preflight)

    return {
        "summary_type": "stock_alpha_thin_symbol_audit",
        "reports_root": str(root),
        "coverage_audit_path": str(coverage["path"]) if coverage else "",
        "preflight_path": str(preflight["path"]) if preflight else "",
        "symbols": {
            symbol: _symbol_summary(
                symbol,
                official_rows=official_rows,
                rows_by_provider=rows_by_provider,
                reports=reports,
            )
            for symbol in normalized_symbols
        },
    }


def _symbol_summary(
    symbol: str,
    *,
    official_rows: Mapping[str, int],
    rows_by_provider: Mapping[str, Mapping[str, int]],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matching_reports = [
        report for report in reports
        if symbol in set(report.get("requested_symbols", [])) | set(report.get("returned_symbols", []))
    ]
    sec_reports = [report for report in matching_reports if "sec_company_filings" in str(report.get("family_name", ""))]
    has_36mo = any("36mo" in str(report.get("family_name", "")) for report in sec_reports)
    has_120mo = any("120mo" in str(report.get("family_name", "")) for report in sec_reports)
    sec_row_count = int(rows_by_provider.get("sec_company_filings", {}).get(symbol, 0))
    rss_row_count = int(rows_by_provider.get("company_press_release_rss", {}).get(symbol, 0))
    current_official_count = int(official_rows.get(symbol, sec_row_count + rss_row_count))
    provider_failures = [
        report for report in sec_reports
        if report.get("provider_failures") or report.get("timeout") or report.get("rate_limited")
    ]
    returned_zero_sec_rows = any(
        report.get("provider_row_count", 0) == 0
        and symbol in report.get("requested_symbols", [])
        and not report.get("provider_failures")
        and not report.get("timeout")
        for report in sec_reports
    )

    return {
        "current_official_row_count": current_official_count,
        "rss_row_count": rss_row_count,
        "sec_row_count": sec_row_count,
        "has_36mo_artifact": has_36mo,
        "has_120mo_artifact": has_120mo,
        "returned_zero_sec_rows": returned_zero_sec_rows,
        "provider_failure_or_timeout": bool(provider_failures),
        "provider_failure_report_count": len(provider_failures),
        "recommended_action": _recommended_action(
            current_official_count=current_official_count,
            has_120mo=has_120mo,
            returned_zero_sec_rows=returned_zero_sec_rows,
            has_provider_failure=bool(provider_failures),
        ),
    }


def _recommended_action(
    *,
    current_official_count: int,
    has_120mo: bool,
    returned_zero_sec_rows: bool,
    has_provider_failure: bool,
) -> str:
    if current_official_count >= 10:
        return "already_sufficient"
    if has_provider_failure or not has_120mo:
        return "retry_120mo_single_symbol"
    if returned_zero_sec_rows:
        return "mapping_or_form_investigation"
    return "audited_exception_candidate"


def _collector_reports(root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(root.rglob("stock_alpha_news_free_source_collect.json")):
        payload = _read_json(path)
        family = path.parents[1].name if len(path.parents) > 1 else path.parent.name
        reports.append(
            {
                "family_name": family,
                "path": str(path),
                "requested_symbols": _requested_symbols(payload),
                "returned_symbols": sorted(_rows_by_symbol(payload)),
                "provider_row_count": int((payload.get("provider_row_counts", {}) or {}).get("sec_company_filings", 0) or 0),
                "provider_failures": dict(payload.get("providers_failed", {}) or {}),
                "timeout": _has_timeout(payload),
                "rate_limited": bool(payload.get("providers_rate_limited") or []),
            }
        )
    return reports


def _rows_by_provider(coverage: Mapping[str, Any] | None) -> dict[str, dict[str, int]]:
    payload = coverage["payload"] if coverage else {}
    providers = dict(payload.get("rows_by_symbol_by_provider", {}) or {})
    return {
        str(provider): {str(symbol).upper(): int(count) for symbol, count in dict(rows or {}).items()}
        for provider, rows in providers.items()
    }


def _official_rows(coverage: Mapping[str, Any] | None, preflight: Mapping[str, Any] | None) -> dict[str, int]:
    for source in (coverage, preflight):
        payload = source["payload"] if source else {}
        rows = payload.get("valid_official_rows_by_symbol") or payload.get("official_rows_by_symbol")
        if rows:
            return {str(symbol).upper(): int(count) for symbol, count in dict(rows).items()}
    return {}


def _latest_json(root: Path, pattern: str) -> dict[str, Any] | None:
    paths = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        payload = _read_json(path)
        if payload:
            return {"path": path, "payload": payload}
    return None


def _requested_symbols(payload: Mapping[str, Any]) -> list[str]:
    return sorted({
        str(symbol).strip().upper()
        for symbol in (payload.get("only_symbols", []) or payload.get("symbols", []) or [])
        if str(symbol).strip()
    })


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit already-generated news coverage for thin symbols.")
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--summary-output")
    args = parser.parse_args(argv)

    summary = build_thin_symbol_audit(
        reports_root=args.reports_root,
        symbols=[symbol.strip() for symbol in args.symbols.split(",")],
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_output:
        Path(args.summary_output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
