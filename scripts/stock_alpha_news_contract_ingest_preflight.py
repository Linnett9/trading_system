from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter


COMMON_REQUIRED_FIELDS = (
    "symbol",
    "provider",
    "source_type",
    "headline_or_title",
    "source_url",
    "published_at_utc",
    "collected_at_utc",
)


def build_stock_alpha_news_contract_ingest_preflight(
    *,
    rss_raw_news_path: str | Path | None = None,
    sec_report_paths: Sequence[str | Path] | None = None,
    sec_event_row_paths: Sequence[str | Path] | None = None,
    sec_artifact_selection: str = "all",
    min_symbol_coverage: int = 300,
) -> dict[str, Any]:
    rows = []
    if rss_raw_news_path:
        rows.extend(_rss_common_rows(CsvRowRepository().read(Path(rss_raw_news_path))))
    sec_report_summaries = [_read_json(Path(path)) for path in sec_report_paths or []]
    event_row_paths = list(sec_event_row_paths or [])
    for report in sec_report_summaries:
        artifact = str(report.get("sec_company_filings_event_rows_path", "")).strip()
        if artifact:
            event_row_paths.append(artifact)
    selected_event_row_paths = _select_sec_event_row_paths(event_row_paths, sec_artifact_selection)
    sec_common_rows, sec_aggregate_report_count = _sec_report_common_rows(
        sec_report_summaries,
        sec_event_rows=_read_sec_event_rows(selected_event_row_paths),
    )
    rows.extend(sec_common_rows)

    providers_checked = sorted({str(row.get("provider", "")) for row in rows if row.get("provider")})
    rows_checked = Counter(str(row.get("provider", "")) for row in rows)
    missing_by_provider: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    invalid_by_provider: Counter[str] = Counter()
    valid_by_provider: Counter[str] = Counter()
    invalid_timestamp_count = 0
    future_timestamp_count = 0
    now = datetime.now(timezone.utc)
    event_keys = []
    symbols = set()

    for row in rows:
        provider = str(row.get("provider", ""))
        symbols.add(str(row.get("symbol", "")).upper())
        missing = [field for field in COMMON_REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if provider == "sec_company_filings":
            missing.extend(
                field
                for field in ("cik", "form_type", "accession_number", "filing_url")
                if not str(row.get(field, "")).strip()
            )
        for field in missing:
            missing_by_provider[provider][field] += 1
        timestamp = _parse_timestamp(str(row.get("published_at_utc", "")))
        if timestamp is None:
            invalid_timestamp_count += 1
        elif timestamp > now:
            future_timestamp_count += 1
        key = "|".join(str(row.get(field, "")).strip() for field in ("provider", "symbol", "source_url", "published_at_utc"))
        event_keys.append(key)
        if missing or timestamp is None or (timestamp is not None and timestamp > now):
            invalid_by_provider[provider] += 1
        else:
            valid_by_provider[provider] += 1

    duplicate_event_key_count = len(event_keys) - len(set(event_keys))
    unsafe_reasons = []
    if len(symbols) < min_symbol_coverage:
        unsafe_reasons.append("symbol coverage below contract ingest threshold")
    if invalid_timestamp_count:
        unsafe_reasons.append("one or more rows have invalid published_at_utc timestamps")
    if future_timestamp_count:
        unsafe_reasons.append("one or more rows have future published_at_utc timestamps")
    if duplicate_event_key_count:
        unsafe_reasons.append("duplicate provider/symbol/url/timestamp event keys detected")
    if any(invalid_by_provider.values()):
        unsafe_reasons.append("one or more provider rows failed common schema validation")
    unsafe_reasons.append("contract ingest preflight is report-only and has not approved feature generation")

    return {
        "preflight_type": "stock_alpha_news_contract_ingest_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sec_artifact_selection": sec_artifact_selection,
        "sec_event_rows_included": [str(path) for path in selected_event_row_paths],
        "sec_aggregate_reports_checked": sec_aggregate_report_count,
        "providers_checked": providers_checked,
        "rows_checked_by_provider": dict(sorted(rows_checked.items())),
        "valid_rows_by_provider": dict(sorted(valid_by_provider.items())),
        "invalid_rows_by_provider": dict(sorted(invalid_by_provider.items())),
        "missing_required_fields_by_provider": {
            provider: dict(sorted(fields.items()))
            for provider, fields in sorted(missing_by_provider.items())
        },
        "invalid_timestamp_count": invalid_timestamp_count,
        "future_timestamp_count": future_timestamp_count,
        "duplicate_event_key_count": duplicate_event_key_count,
        "symbols_checked": sorted(symbols),
        "symbol_count": len(symbols),
        "safe_for_feature_generation": False if unsafe_reasons else True,
        "unsafe_reasons": unsafe_reasons,
    }


def write_stock_alpha_news_contract_ingest_preflight(
    *,
    output_path: str | Path,
    rss_raw_news_path: str | Path | None = None,
    sec_report_paths: Sequence[str | Path] | None = None,
    sec_event_row_paths: Sequence[str | Path] | None = None,
    sec_artifact_selection: str = "all",
) -> Path:
    payload = build_stock_alpha_news_contract_ingest_preflight(
        rss_raw_news_path=rss_raw_news_path,
        sec_report_paths=sec_report_paths,
        sec_event_row_paths=sec_event_row_paths,
        sec_artifact_selection=sec_artifact_selection,
    )
    output = Path(output_path)
    ResearchArtifactWriter().write_json(output, payload)
    return output


def _rss_common_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        result.append(
            {
                "symbol": row.get("symbol", ""),
                "provider": row.get("provider", "company_press_release_rss"),
                "source_type": "rss_item",
                "headline_or_title": row.get("headline", ""),
                "source_url": row.get("provider_url", ""),
                "published_at_utc": row.get("published_at_utc", ""),
                "collected_at_utc": row.get("ingested_at", ""),
            }
        )
    return result


def _sec_report_common_rows(
    reports: Sequence[Mapping[str, Any]],
    *,
    sec_event_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    result = []
    aggregate_report_count = 0
    for row in sec_event_rows:
        result.append(_sec_event_common_row(row))
    for report in reports:
        if report.get("provider_row_counts", {}).get("sec_company_filings", 0) <= 0:
            continue
        event_rows = report.get("event_rows", []) or []
        if not event_rows:
            aggregate_report_count += 1
            continue
        for row in event_rows:
            result.append(_sec_event_common_row(row))
    return result, aggregate_report_count


def _sec_event_common_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "symbol": str(row.get("symbol", "")),
        "provider": "sec_company_filings",
        "source_type": str(row.get("source_type", "sec_filing")),
        "headline_or_title": str(row.get("headline_or_title") or row.get("headline") or row.get("form_type", "")),
        "source_url": str(row.get("source_url") or row.get("filing_url") or row.get("primary_document_url") or ""),
        "published_at_utc": str(row.get("published_at_utc", "")),
        "collected_at_utc": str(row.get("collected_at_utc", "")),
        "cik": str(row.get("cik", "")),
        "form_type": str(row.get("form_type", "")),
        "accession_number": str(row.get("accession_number", "")),
        "filing_url": str(row.get("filing_url", "")),
    }


def _read_sec_event_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        rows.append(dict(value))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _select_sec_event_row_paths(paths: Sequence[str | Path], selection: str) -> list[str | Path]:
    if selection == "all":
        return list(paths)
    if selection != "prefer_12mo":
        raise ValueError("sec_artifact_selection must be 'all' or 'prefer_12mo'")
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
    selected: list[str | Path] = []
    for batch in sorted(by_batch):
        batch_paths = by_batch[batch]
        preferred = [path for path in batch_paths if "_12mo_dry_run/" in str(path)]
        selected.extend(preferred or batch_paths)
    selected.extend(other_paths)
    return selected


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    payload.setdefault("json_path", str(path))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight stock-alpha news provider rows for common event-schema ingest.")
    parser.add_argument("--rss-raw-news")
    parser.add_argument("--sec-report", action="append", default=[])
    parser.add_argument("--sec-event-rows", action="append", default=[])
    parser.add_argument("--include-sec-event-rows", action="store_true")
    parser.add_argument("--sec-artifact-selection", choices=["all", "prefer_12mo"], default="all")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = write_stock_alpha_news_contract_ingest_preflight(
        output_path=args.output,
        rss_raw_news_path=args.rss_raw_news,
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
    )
    print(f"Wrote stock-alpha news contract ingest preflight: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
