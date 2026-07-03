from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_COLUMNS = (
    "event_id",
    "event_key",
    "symbol",
    "provider",
    "source_type",
    "event_timestamp",
    "available_at_timestamp",
    "form_type",
    "title",
    "summary_or_text",
    "url_or_accession",
    "is_sec_filing",
    "is_rss_item",
    "event_year",
    "event_month",
    "event_day_of_week",
)


def build_report_only_news_transformer_feature_dataset(
    *,
    feature_gate: Mapping[str, Any],
    coverage_audit: Mapping[str, Any],
    contract_preflight: Mapping[str, Any],
    reports_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    if feature_gate.get("approved") is not True:
        raise ValueError("feature-generation gate is not approved")
    reports_root_path = Path(reports_root)
    output_dir_path = Path(output_dir)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("feature dataset output_dir must be under reports/")

    event_paths = _selected_sec_event_row_paths(coverage_audit, contract_preflight)
    input_rows = [_feature_row(row) for row in _read_jsonl_rows(event_paths)]
    input_duplicate_event_key_count = _duplicate_count(row["event_key"] for row in input_rows)
    rows = _dedupe_rows(input_rows)
    rows.sort(key=lambda row: (row["available_at_timestamp"], row["symbol"], row["event_key"]))
    duplicate_event_key_count = _duplicate_count(row["event_key"] for row in rows)
    duplicate_symbol_time_title_count = _duplicate_count(
        (row["symbol"], row["available_at_timestamp"], row["title"]) for row in rows
    )
    future_timestamp_count = sum(1 for row in rows if _is_future(row["available_at_timestamp"]))
    unresolved_timeouts = list(contract_preflight.get("unresolved_provider_timeout_symbols", []) or [])
    blocking_reasons: list[str] = []
    if duplicate_event_key_count:
        blocking_reasons.append("duplicate event keys detected")
    if future_timestamp_count:
        blocking_reasons.append("future event timestamps detected")
    if unresolved_timeouts:
        blocking_reasons.append("unresolved provider timeout artifacts remain")
    if not event_paths:
        blocking_reasons.append("no selected SEC event-row artifacts available")

    output_dir_path.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir_path / "news_transformer_event_features.csv"
    report_path = output_dir_path / "news_transformer_feature_dataset_report.json"
    leakage_path = output_dir_path / "news_transformer_leakage_audit.json"
    _write_csv(dataset_path, rows)
    labels_attached = False
    warnings = [
        "labels_attached=false; no clean project price-label utility was attached",
    ]
    if input_duplicate_event_key_count:
        warnings.append("overlapping selected SEC event rows were deduplicated by event_key")
    rss_rows = int((contract_preflight.get("rows_checked_by_provider", {}) or {}).get("company_press_release_rss", 0) or 0)
    if rss_rows:
        warnings.append("RSS rows are summarized in audit/preflight but no reusable selected RSS event-row path is exposed")
    report = {
        "feature_dataset_status": "ready_for_label_attachment" if not blocking_reasons else "blocked",
        "rows": len(rows),
        "symbols": sorted({row["symbol"] for row in rows}),
        "date_min": min((row["available_at_timestamp"] for row in rows), default=""),
        "date_max": max((row["available_at_timestamp"] for row in rows), default=""),
        "duplicate_event_key_count": duplicate_event_key_count,
        "input_duplicate_event_key_count": input_duplicate_event_key_count,
        "duplicate_symbol_available_title_count": duplicate_symbol_time_title_count,
        "future_timestamp_count": future_timestamp_count,
        "leakage_violation_count": 0,
        "labels_attached": labels_attached,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "next_allowed_step": "attach_price_return_labels" if not blocking_reasons else "resolve_feature_dataset_blockers",
        "dataset_path": str(dataset_path),
        "leakage_audit_path": str(leakage_path),
        "selected_sec_event_rows_included": [str(path) for path in event_paths],
    }
    leakage = {
        key: report[key]
        for key in (
            "feature_dataset_status",
            "rows",
            "symbols",
            "date_min",
            "date_max",
            "duplicate_event_key_count",
            "future_timestamp_count",
            "leakage_violation_count",
            "labels_attached",
            "blocking_reasons",
            "warnings",
            "next_allowed_step",
        )
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    leakage_path.write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _selected_sec_event_row_paths(
    coverage_audit: Mapping[str, Any],
    contract_preflight: Mapping[str, Any],
) -> list[Path]:
    values = contract_preflight.get("sec_event_rows_included") or coverage_audit.get("sec_event_rows_included") or []
    return [Path(value) for value in values if str(value).strip() and "data/news" not in str(value)]


def _read_jsonl_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _feature_row(row: Mapping[str, Any]) -> dict[str, str]:
    symbol = str(row.get("symbol", "")).strip().upper()
    provider = str(row.get("provider", "")).strip()
    source_url = str(row.get("source_url") or row.get("filing_url") or row.get("primary_document_url") or "").strip()
    timestamp = str(row.get("published_at_utc") or row.get("accepted_datetime") or "").strip()
    available_at = str(row.get("accepted_datetime") or row.get("published_at_utc") or "").strip()
    title = str(row.get("headline_or_title") or row.get("title") or "").strip()
    event_key = "|".join((provider, symbol, source_url, timestamp))
    parsed = _parse_timestamp(available_at)
    source_type = str(row.get("source_type", "")).strip()
    return {
        "event_id": hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:24],
        "event_key": event_key,
        "symbol": symbol,
        "provider": provider,
        "source_type": source_type,
        "event_timestamp": timestamp,
        "available_at_timestamp": available_at,
        "form_type": str(row.get("form_type", "")).strip(),
        "title": title,
        "summary_or_text": str(row.get("summary") or row.get("text") or "").strip(),
        "url_or_accession": str(row.get("accession_number") or source_url).strip(),
        "is_sec_filing": str(provider == "sec_company_filings" or source_type == "sec_filing").lower(),
        "is_rss_item": str(provider == "company_press_release_rss" or source_type == "rss").lower(),
        "event_year": str(parsed.year if parsed else ""),
        "event_month": str(parsed.month if parsed else ""),
        "event_day_of_week": str(parsed.weekday() if parsed else ""),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _dedupe_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda item: (item["available_at_timestamp"], item["symbol"], item["event_key"])):
        deduped.setdefault(str(row["event_key"]), dict(row))
    return list(deduped.values())


def _duplicate_count(values: Sequence[Any]) -> int:
    materialized = list(values)
    return len(materialized) - len(set(materialized))


def _is_future(value: str) -> bool:
    parsed = _parse_timestamp(value)
    return bool(parsed and parsed > datetime.now(timezone.utc))


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build report-only news transformer event feature dataset.")
    parser.add_argument("--feature-gate", required=True)
    parser.add_argument("--coverage-audit", required=True)
    parser.add_argument("--contract-preflight", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    report = build_report_only_news_transformer_feature_dataset(
        feature_gate=_read_json(args.feature_gate),
        coverage_audit=_read_json(args.coverage_audit),
        contract_preflight=_read_json(args.contract_preflight),
        reports_root=args.reports_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
