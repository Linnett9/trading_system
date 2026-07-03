from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_FILENAME = "news_transformer_official_text_enrichment_plan.json"
MANIFEST_FILENAME = "news_transformer_official_text_enrichment_manifest.csv"
REQUIRED_LABEL_COLUMNS = (
    "label_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_drawdown_20d",
    "reduce_exposure_label",
)
BODY_TEXT_COLUMNS = ("summary", "text", "body_or_summary", "official_text", "primary_document_text")


def build_official_text_enrichment_plan_report_only(
    *,
    labeled_dataset_path: str | Path,
    text_baseline_report_path: str | Path,
    sec_event_rows_root: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    text_baseline_report = _read_json(Path(text_baseline_report_path))
    _validate_text_baseline_report(text_baseline_report)

    labeled_rows = _read_csv(Path(labeled_dataset_path))
    sec_cache_rows = _read_sec_cache_rows(Path(sec_event_rows_root))
    cache_by_accession = _cache_by_accession(sec_cache_rows)
    manifest = _manifest_rows(labeled_rows, cache_by_accession)
    report = _report(labeled_rows, sec_cache_rows, cache_by_accession, manifest, text_baseline_report)
    _write_outputs(output_dir_path, report, manifest)
    return report


def _validate_text_baseline_report(report: Mapping[str, Any]) -> None:
    comparison = report.get("text_vs_metadata_baseline", {}) or {}
    guardrails = {
        "mode": report.get("mode") == "news_transformer_text_baseline_report_only",
        "research_only": report.get("research_only") is True,
        "trading_impact": report.get("trading_impact") == "none",
        "transformer_training_started": report.get("transformer_training_started") is False,
        "leakage_violation_count": int(report.get("leakage_violation_count", 0)) == 0,
        "next_allowed_step": report.get("next_allowed_step") == "review_news_transformer_text_baseline_report",
        "text_did_not_beat_metadata": comparison.get("beats_metadata_on_balanced_accuracy") is False,
    }
    failed = [name for name, passed in guardrails.items() if not passed]
    if failed:
        raise ValueError(f"text baseline report is not approved for official text enrichment planning: {', '.join(failed)}")


def _read_sec_cache_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/sec_company_filings_event_rows.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source_cache_path"] = str(path)
                rows.append(row)
    return rows


def _cache_by_accession(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    cache: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        accession = _normal_accession(str(row.get("accession_number", "")))
        if accession and accession not in cache:
            cache[accession] = row
    return cache


def _manifest_rows(
    labeled_rows: Sequence[Mapping[str, str]],
    cache_by_accession: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in labeled_rows:
        if str(row.get("provider", "")) != "sec_company_filings":
            continue
        accession = _normal_accession(str(row.get("url_or_accession", "")))
        cached = cache_by_accession.get(accession, {})
        body_text = _body_text(cached)
        rows.append(
            {
                "event_key": str(row.get("event_key", "")),
                "symbol": str(row.get("symbol", "")),
                "form_type": str(row.get("form_type", "")),
                "available_at_timestamp": str(row.get("available_at_timestamp", "")),
                "label_date": str(row.get("label_date", "")),
                "accession_number": str(cached.get("accession_number") or row.get("url_or_accession", "")),
                "primary_document_url": str(cached.get("primary_document_url", "")),
                "filing_url": str(cached.get("filing_url") or cached.get("source_url", "")),
                "cache_match": str(bool(cached)).lower(),
                "body_text_available": str(bool(body_text)).lower(),
                "body_text_length": str(len(body_text)),
                "source_cache_path": str(cached.get("_source_cache_path", "")),
            }
        )
    return rows


def _report(
    labeled_rows: Sequence[Mapping[str, str]],
    sec_cache_rows: Sequence[Mapping[str, Any]],
    cache_by_accession: Mapping[str, Mapping[str, Any]],
    manifest: Sequence[Mapping[str, str]],
    text_baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    form_counts = Counter(row.get("form_type", "") for row in manifest)
    cache_form_counts = Counter(str(row.get("form_type", "")) for row in sec_cache_rows)
    matched_rows = [row for row in manifest if row["cache_match"] == "true"]
    body_text_rows = [row for row in manifest if row["body_text_available"] == "true"]
    primary_url_rows = [row for row in matched_rows if row.get("primary_document_url")]
    warnings = [
        "All labelled SEC accessions map to local official SEC cache rows with primary document URLs."
        if len(matched_rows) == len(manifest)
        else "Some labelled SEC accessions do not map to local SEC cache rows."
    ]
    if not body_text_rows:
        warnings.append("No filing body text is cached locally; do not train text/transformer models on titles alone.")
    return {
        "mode": "news_transformer_official_text_enrichment_plan_report_only",
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "model_training_started": False,
        "transformer_training_started": False,
        "rows_total": len(labeled_rows),
        "rows_with_required_labels": sum(1 for row in labeled_rows if _has_required_labels(row)),
        "sec_rows_in_dataset": len(manifest),
        "unique_dataset_accessions": len({_normal_accession(row["accession_number"]) for row in manifest if row.get("accession_number")}),
        "dataset_form_counts": dict(sorted(form_counts.items())),
        "sec_event_cache_rows": len(sec_cache_rows),
        "unique_sec_cache_accessions": len(cache_by_accession),
        "sec_cache_form_counts": dict(sorted(cache_form_counts.items())),
        "matched_by_accession_count": len(matched_rows),
        "matched_by_accession_rate": _safe_div(len(matched_rows), len(manifest)),
        "matched_rows_with_primary_document_url": len(primary_url_rows),
        "local_body_text_available_count": len(body_text_rows),
        "local_body_text_available_rate": _safe_div(len(body_text_rows), len(manifest)),
        "unmatched_accession_count": len(manifest) - len(matched_rows),
        "unmatched_accession_sample": [row["accession_number"] for row in manifest if row["cache_match"] != "true"][:20],
        "text_baseline_balanced_accuracy": (
            text_baseline_report.get("selected_model_average_test_metrics", {}) or {}
        ).get("balanced_accuracy"),
        "metadata_baseline_balanced_accuracy": (
            text_baseline_report.get("metadata_baseline_average_test_metrics", {}) or {}
        ).get("balanced_accuracy"),
        "text_vs_metadata_baseline": text_baseline_report.get("text_vs_metadata_baseline", {}),
        "blocking_reasons": ["official_sec_body_text_not_cached"] if not body_text_rows else [],
        "warnings": warnings,
        "recommended_official_text_fields": [
            "primary_document_url",
            "primary_document_text",
            "filing_url",
            "accession_number",
            "form_type",
        ],
        "next_allowed_step": "cache_official_sec_primary_document_text_report_only",
        "training_allowed": False,
    }


def _body_text(row: Mapping[str, Any]) -> str:
    for column in BODY_TEXT_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def _has_required_labels(row: Mapping[str, str]) -> bool:
    return all(row.get(column) for column in REQUIRED_LABEL_COLUMNS)


def _normal_accession(value: str) -> str:
    return value.strip().replace("-", "")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(output_dir: Path, report: Mapping[str, Any], manifest: Sequence[Mapping[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / MANIFEST_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "event_key",
            "symbol",
            "form_type",
            "available_at_timestamp",
            "label_date",
            "accession_number",
            "primary_document_url",
            "filing_url",
            "cache_match",
            "body_text_available",
            "body_text_length",
            "source_cache_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a report-only official SEC text enrichment plan for news transformer rows.")
    parser.add_argument("--labeled-dataset", required=True)
    parser.add_argument("--text-baseline-report", required=True)
    parser.add_argument("--sec-event-rows-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args(argv)

    report = build_official_text_enrichment_plan_report_only(
        labeled_dataset_path=args.labeled_dataset,
        text_baseline_report_path=args.text_baseline_report,
        sec_event_rows_root=args.sec_event_rows_root,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
