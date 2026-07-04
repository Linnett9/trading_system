from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ENRICHMENT_REPORT_FILENAME = "news_transformer_official_text_enrichment_plan.json"
ENRICHMENT_MANIFEST_FILENAME = "news_transformer_official_text_enrichment_manifest.csv"
CACHE_MANIFEST_FILENAME = "sec_primary_document_text_cache_manifest.json"
CACHE_SUMMARY_FILENAME = "sec_primary_document_text_cache_summary.json"
DEFAULT_USER_AGENT = "stock-alpha-research/1.0 research-contact@example.invalid"
NEXT_ALLOWED_STEP = "build_enriched_official_text_dataset_report_only"
MIN_DOCUMENT_TEXT_LENGTH = 200

FetchText = Callable[[str, str, int], str]


@dataclass(frozen=True)
class DocumentRequest:
    accession_number: str
    primary_document_url: str
    event_keys: tuple[str, ...]
    symbols: tuple[str, ...]
    form_types: tuple[str, ...]

    @property
    def cache_filename(self) -> str:
        accession = _safe_slug(self.accession_number or "missing-accession")
        url_hash = hashlib.sha256(self.primary_document_url.encode("utf-8")).hexdigest()[:16]
        return f"{accession}_{url_hash}.txt"


def cache_sec_primary_text_report_only(
    *,
    enrichment_plan_dir: str | Path | None,
    output_dir: str | Path,
    reports_root: str | Path,
    input_manifest: str | Path | None = None,
    max_documents: int | None = None,
    sleep_seconds: float = 0.0,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_seconds: int = 30,
    overwrite: bool = False,
    fetch_text: FetchText = None,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")
    _validate_runtime_options(max_documents=max_documents, sleep_seconds=sleep_seconds, timeout_seconds=timeout_seconds)

    documents = _load_documents(enrichment_plan_dir=enrichment_plan_dir, input_manifest=input_manifest)
    selected_documents = documents[:max_documents] if max_documents is not None else documents

    fetcher = fetch_text or standard_library_sec_text_get
    documents_dir = output_dir_path / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    attempted_documents = 0
    cached_documents = 0
    skipped_existing_documents = 0
    failed_documents = 0
    empty_text_documents = 0
    rate_limit_or_timeout_failures = 0
    text_lengths: list[int] = []

    for index, document in enumerate(selected_documents):
        cache_path = documents_dir / document.cache_filename
        entry = _base_manifest_entry(document, cache_path)
        if cache_path.exists() and not overwrite:
            text_length = len(cache_path.read_text(encoding="utf-8"))
            entry.update({"status": "skipped_existing", "text_length": text_length})
            text_lengths.append(text_length)
            skipped_existing_documents += 1
            manifest.append(entry)
            continue

        attempted_documents += 1
        try:
            raw_text = fetcher(document.primary_document_url, user_agent, timeout_seconds)
            extracted_text = extract_sec_document_text(raw_text)
            text_length = len(extracted_text)
            entry["text_length"] = text_length
            if text_length < MIN_DOCUMENT_TEXT_LENGTH:
                empty_text_documents += 1
                entry["status"] = "empty_text"
                entry["error"] = f"extracted text shorter than {MIN_DOCUMENT_TEXT_LENGTH} characters"
            else:
                cache_path.write_text(extracted_text + "\n", encoding="utf-8")
                text_lengths.append(text_length)
                cached_documents += 1
                entry["status"] = "cached"
        except Exception as exc:
            failed_documents += 1
            if _is_rate_limit_or_timeout(exc):
                rate_limit_or_timeout_failures += 1
            entry["status"] = "failed"
            entry["error_type"] = type(exc).__name__
            entry["error"] = str(exc)
        manifest.append(entry)

        if sleep_seconds > 0.0 and index < len(selected_documents) - 1:
            time.sleep(sleep_seconds)

    summary = _summary(
        requested_documents=len(selected_documents),
        unique_document_urls=len(documents),
        attempted_documents=attempted_documents,
        cached_documents=cached_documents,
        skipped_existing_documents=skipped_existing_documents,
        failed_documents=failed_documents,
        empty_text_documents=empty_text_documents,
        rate_limit_or_timeout_failures=rate_limit_or_timeout_failures,
        cache_dir=documents_dir,
        text_lengths=text_lengths,
    )
    _write_json(output_dir_path / CACHE_MANIFEST_FILENAME, {"documents": manifest})
    _write_json(output_dir_path / CACHE_SUMMARY_FILENAME, summary)
    return summary


def _load_documents(
    *,
    enrichment_plan_dir: str | Path | None,
    input_manifest: str | Path | None,
) -> list[DocumentRequest]:
    if input_manifest is not None:
        return _read_retry_manifest(Path(input_manifest))
    if enrichment_plan_dir is None:
        raise ValueError("enrichment_plan_dir is required when input_manifest is not provided")
    plan_dir = Path(enrichment_plan_dir)
    enrichment_report = _read_json(plan_dir / ENRICHMENT_REPORT_FILENAME)
    _validate_enrichment_report(enrichment_report)
    rows = _read_csv(plan_dir / ENRICHMENT_MANIFEST_FILENAME)
    return _deduplicated_documents(rows)


def _read_retry_manifest(path: Path) -> list[DocumentRequest]:
    documents: list[DocumentRequest] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed retry manifest JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"retry manifest line {line_number} must be a JSON object")
            if row.get("retry_priority") == "do_not_retry":
                continue
            document = _retry_manifest_document(row, line_number=line_number)
            identity = str(row.get("document_id") or f"{document.accession_number}|{document.primary_document_url}")
            if identity in seen:
                continue
            seen.add(identity)
            documents.append(document)
    return documents


def _retry_manifest_document(row: Mapping[str, Any], *, line_number: int) -> DocumentRequest:
    url = str(row.get("document_url") or row.get("primary_document_url") or "").strip()
    if not url:
        raise ValueError(f"retry manifest line {line_number} is missing document_url")
    _validate_sec_url(url)
    accession = str(row.get("accession") or row.get("accession_number") or "").strip()
    document_id = str(row.get("document_id") or "").strip()
    if not accession and not document_id:
        raise ValueError(f"retry manifest line {line_number} is missing stable document identity")
    if not accession and "|" in document_id:
        accession = document_id.split("|", 1)[0]
    if not accession:
        raise ValueError(f"retry manifest line {line_number} is missing accession")
    return DocumentRequest(
        accession_number=accession,
        primary_document_url=url,
        event_keys=(),
        symbols=_single_value_tuple(row.get("symbol")),
        form_types=_single_value_tuple(row.get("form_type")),
    )


def standard_library_sec_text_get(url: str, user_agent: str, timeout_seconds: int) -> str:
    _validate_sec_url(url)
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - URL is restricted to sec.gov.
        return response.read().decode("utf-8", errors="replace")


def extract_sec_document_text(raw_document: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw_document)
    parser.close()
    text = parser.text()
    if not text:
        text = raw_document
    return _normalize_text(html.unescape(text))


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "head", "noscript"}:
            self._hidden_depth += 1
        elif tag.lower() in {"br", "p", "div", "tr", "li", "table"} and self._hidden_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "head", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag.lower() in {"p", "div", "tr", "li", "table"} and self._hidden_depth == 0:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


def _validate_runtime_options(*, max_documents: int | None, sleep_seconds: float, timeout_seconds: int) -> None:
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be at least 1 when provided")
    if sleep_seconds < 0.0 or sleep_seconds > 60.0:
        raise ValueError("sleep_seconds must be between 0 and 60")
    if timeout_seconds < 1 or timeout_seconds > 120:
        raise ValueError("timeout_seconds must be between 1 and 120")


def _validate_enrichment_report(report: Mapping[str, Any]) -> None:
    guardrails = {
        "mode": report.get("mode") == "news_transformer_official_text_enrichment_plan_report_only",
        "research_only": report.get("research_only") is True,
        "training_allowed": report.get("training_allowed") is False,
        "model_training_started": report.get("model_training_started") is False,
        "transformer_training_started": report.get("transformer_training_started") is False,
        "next_allowed_step": report.get("next_allowed_step") == "cache_official_sec_primary_document_text_report_only",
    }
    failed = [name for name, passed in guardrails.items() if not passed]
    if failed:
        raise ValueError(f"official text enrichment plan is not approved for SEC text caching: {', '.join(failed)}")


def _deduplicated_documents(rows: Sequence[Mapping[str, str]]) -> list[DocumentRequest]:
    grouped: dict[str, dict[str, set[str] | str]] = {}
    for row in rows:
        url = str(row.get("primary_document_url", "")).strip()
        if not url:
            continue
        _validate_sec_url(url)
        bucket = grouped.setdefault(
            url,
            {
                "accession_number": str(row.get("accession_number", "")).strip(),
                "event_keys": set(),
                "symbols": set(),
                "form_types": set(),
            },
        )
        for key, column in (("event_keys", "event_key"), ("symbols", "symbol"), ("form_types", "form_type")):
            value = str(row.get(column, "")).strip()
            if value:
                cast_set = bucket[key]
                assert isinstance(cast_set, set)
                cast_set.add(value)
    documents: list[DocumentRequest] = []
    for url, values in grouped.items():
        documents.append(
            DocumentRequest(
                accession_number=str(values["accession_number"]),
                primary_document_url=url,
                event_keys=tuple(sorted(values["event_keys"])),
                symbols=tuple(sorted(values["symbols"])),
                form_types=tuple(sorted(values["form_types"])),
            )
        )
    return documents


def _base_manifest_entry(document: DocumentRequest, cache_path: Path) -> dict[str, Any]:
    return {
        "accession_number": document.accession_number,
        "primary_document_url": document.primary_document_url,
        "cache_path": str(cache_path),
        "event_keys": list(document.event_keys),
        "symbols": list(document.symbols),
        "form_types": list(document.form_types),
        "status": "pending",
        "text_length": 0,
    }


def _summary(
    *,
    requested_documents: int,
    unique_document_urls: int,
    attempted_documents: int,
    cached_documents: int,
    skipped_existing_documents: int,
    failed_documents: int,
    empty_text_documents: int,
    rate_limit_or_timeout_failures: int,
    cache_dir: Path,
    text_lengths: Sequence[int],
) -> dict[str, Any]:
    blocking_reasons = []
    warnings = []
    if failed_documents:
        blocking_reasons.append("document_fetch_failures")
    if empty_text_documents:
        blocking_reasons.append("empty_or_too_short_document_text")
    if requested_documents == 0:
        blocking_reasons.append("no_documents_requested")
    if requested_documents < unique_document_urls:
        warnings.append("max_documents limited this run; full cache has not been attempted")
    return {
        "mode": "sec_primary_document_text_cache_report_only",
        "research_only": True,
        "trading_impact": "none",
        "model_training_started": False,
        "transformer_training_started": False,
        "requested_documents": requested_documents,
        "unique_document_urls": unique_document_urls,
        "attempted_documents": attempted_documents,
        "cached_documents": cached_documents,
        "skipped_existing_documents": skipped_existing_documents,
        "failed_documents": failed_documents,
        "empty_text_documents": empty_text_documents,
        "cache_dir": str(cache_dir),
        "document_text_min_length": min(text_lengths) if text_lengths else 0,
        "document_text_median_length": median(text_lengths) if text_lengths else 0,
        "document_text_max_length": max(text_lengths) if text_lengths else 0,
        "rate_limit_or_timeout_failures": rate_limit_or_timeout_failures,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "next_allowed_step": NEXT_ALLOWED_STEP if not blocking_reasons else "resolve_sec_primary_document_text_cache_failures",
    }


def _validate_sec_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"www.sec.gov", "sec.gov"}:
        raise ValueError("primary_document_url must be an https://www.sec.gov URL")


def _is_rate_limit_or_timeout(exc: Exception) -> bool:
    message = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, HTTPError) and exc.code in {403, 408, 429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, URLError):
        return "timed out" in message or "timeout" in message
    return "rate limit" in message or "timed out" in message or "timeout" in message


def _normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())[:120]


def _single_value_tuple(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    return (text,) if text else ()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache official SEC primary-document text under reports/.")
    parser.add_argument("--enrichment-plan-dir")
    parser.add_argument("--input-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.overwrite and (args.resume or args.skip_existing):
        parser.error("--overwrite cannot be combined with --resume or --skip-existing")

    summary = cache_sec_primary_text_report_only(
        enrichment_plan_dir=args.enrichment_plan_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
        input_manifest=args.input_manifest,
        max_documents=args.max_documents,
        sleep_seconds=args.sleep_seconds,
        user_agent=args.user_agent,
        timeout_seconds=args.timeout_seconds,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
