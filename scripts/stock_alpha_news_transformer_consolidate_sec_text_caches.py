from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


CACHE_MANIFEST_FILENAME = "sec_primary_document_text_cache_manifest.json"
CACHE_SUMMARY_FILENAME = "sec_primary_document_text_cache_summary.json"


def consolidate_sec_text_caches(
    *,
    primary_cache_dir: str | Path,
    retry_cache_dirs: Sequence[str | Path],
    output_dir: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    source_dirs = [Path(primary_cache_dir), *[Path(path) for path in retry_cache_dirs]]
    output_documents_dir = output_path / "documents"
    rows_by_id: dict[str, dict[str, Any]] = {}
    state = _initial_state(source_dirs)

    for source_index, source_dir in enumerate(source_dirs):
        manifest_path = source_dir / CACHE_MANIFEST_FILENAME
        manifest = _read_manifest(manifest_path)
        state["source_manifest_rows"] += len(manifest)
        state["source_document_files"] += _count_text_files(source_dir)
        manifest_cache_paths = {str(row.get("cache_path", "")) for row in manifest if isinstance(row, Mapping)}
        state["orphan_document_file_count"] += _count_orphan_files(source_dir, manifest_cache_paths)

        for row in manifest:
            candidate = _normalized_row(row, source_dir=source_dir, manifest_path=manifest_path, source_index=source_index)
            if candidate is None:
                state["invalid_manifest_row_count"] += 1
                continue
            if candidate["status"] in {"cached", "skipped_existing"}:
                state["source_success_rows"] += 1
                if not candidate["source_file"].exists():
                    state["missing_success_file_count"] += 1
                    continue
                candidate["content_sha256"] = _sha256(candidate["source_file"])
                candidate["text_length"] = candidate["source_file"].stat().st_size
            else:
                state["source_failed_rows"] += 1

            existing = rows_by_id.get(candidate["document_id"])
            if existing is None:
                rows_by_id[candidate["document_id"]] = candidate
                continue
            _merge_candidate(existing, candidate, state)

    output_rows = []
    for document_id in sorted(rows_by_id):
        row = rows_by_id[document_id]
        if row["status"] in {"cached", "skipped_existing"}:
            output_rows.append(_copy_success(row, output_documents_dir, state))
        else:
            output_rows.append(_manifest_output_row(row, None))

    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path / CACHE_MANIFEST_FILENAME, {"documents": output_rows})
    summary = _summary(state, source_dirs, output_rows, output_documents_dir)
    _atomic_write_json(output_path / CACHE_SUMMARY_FILENAME, summary)
    return summary


def _initial_state(source_dirs: Sequence[Path]) -> dict[str, Any]:
    return {
        "source_cache_directories": [str(path) for path in source_dirs],
        "source_manifest_rows": 0,
        "source_success_rows": 0,
        "source_failed_rows": 0,
        "source_document_files": 0,
        "newly_recovered_document_ids": set(),
        "duplicate_document_count": 0,
        "identical_duplicate_file_count": 0,
        "conflicting_document_count": 0,
        "missing_success_file_count": 0,
        "orphan_document_file_count": 0,
        "invalid_manifest_row_count": 0,
        "warnings": [],
    }


def _read_manifest(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    return documents if isinstance(documents, list) else []


def _normalized_row(row: Any, *, source_dir: Path, manifest_path: Path, source_index: int) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    url = str(row.get("primary_document_url") or "").strip()
    accession = str(row.get("accession_number") or "").strip()
    document_id = str(row.get("document_id") or "").strip() or (f"{accession}|{url}" if accession and url else "")
    status = str(row.get("status") or "").strip()
    if not document_id or not url or not accession or not status:
        return None
    cache_path = Path(str(row.get("cache_path") or ""))
    source_file = cache_path if cache_path.is_absolute() else Path.cwd() / cache_path
    if not source_file.exists():
        source_file = source_dir / "documents" / cache_path.name
    normalized = dict(row)
    normalized.update(
        {
            "document_id": document_id,
            "primary_document_url": url,
            "accession_number": accession,
            "status": status,
            "source_cache": str(source_dir),
            "source_manifest": str(manifest_path),
            "source_file": source_file,
            "source_index": source_index,
            "recovered_from_retry": source_index > 0 and status in {"cached", "skipped_existing"},
            "prior_failure_sources": [],
        }
    )
    return normalized


def _merge_candidate(existing: dict[str, Any], candidate: dict[str, Any], state: dict[str, Any]) -> None:
    state["duplicate_document_count"] += 1
    existing_success = existing["status"] in {"cached", "skipped_existing"}
    candidate_success = candidate["status"] in {"cached", "skipped_existing"}
    if existing_success and candidate_success:
        if existing.get("content_sha256") == candidate.get("content_sha256"):
            state["identical_duplicate_file_count"] += 1
            if candidate["source_index"] > existing["source_index"]:
                existing["recovered_from_retry"] = existing["recovered_from_retry"] or candidate["recovered_from_retry"]
            return
        state["conflicting_document_count"] += 1
        existing.setdefault("conflict_sources", []).append(candidate["source_cache"])
        return
    if candidate_success and not existing_success:
        if candidate["source_index"] > 0:
            state["newly_recovered_document_ids"].add(candidate["document_id"])
        candidate["prior_failure_sources"] = existing.get("prior_failure_sources", []) + [existing["source_cache"]]
        existing.clear()
        existing.update(candidate)
        return
    if existing_success and not candidate_success:
        existing.setdefault("prior_failure_sources", []).append(candidate["source_cache"])
        return
    existing.setdefault("duplicate_failure_sources", []).append(candidate["source_cache"])


def _copy_success(row: Mapping[str, Any], output_documents_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    source_file = Path(row["source_file"])
    destination = output_documents_dir / source_file.name
    output_documents_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != row["content_sha256"]:
            state["conflicting_document_count"] += 1
            return _manifest_output_row(row, destination)
    else:
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source_file, tmp)
        os.replace(tmp, destination)
    return _manifest_output_row(row, destination)


def _manifest_output_row(row: Mapping[str, Any], cache_path: Path | None) -> dict[str, Any]:
    hidden = {"source_file", "source_index", "conflict_sources"}
    output = {key: value for key, value in row.items() if key not in hidden}
    if cache_path is not None:
        output["cache_path"] = str(cache_path)
    output["source_cache"] = row["source_cache"]
    output["source_manifest"] = row["source_manifest"]
    output["recovered_from_retry"] = bool(row.get("recovered_from_retry"))
    output["content_sha256"] = row.get("content_sha256")
    return output


def _summary(state: Mapping[str, Any], source_dirs: Sequence[Path], rows: Sequence[Mapping[str, Any]], output_documents_dir: Path) -> dict[str, Any]:
    statuses = Counter(row.get("status") for row in rows)
    success_count = statuses.get("cached", 0) + statuses.get("skipped_existing", 0)
    failed_count = len(rows) - success_count
    blocking_reasons = []
    if state["conflicting_document_count"]:
        blocking_reasons.append("conflicting_document_content")
    if state["missing_success_file_count"]:
        blocking_reasons.append("missing_success_files")
    return {
        "mode": "sec_primary_document_text_cache_consolidation_report_only",
        "research_only": True,
        "source_cache_directories": [str(path) for path in source_dirs],
        "source_manifest_rows": state["source_manifest_rows"],
        "source_success_rows": state["source_success_rows"],
        "source_failed_rows": state["source_failed_rows"],
        "source_document_files": state["source_document_files"],
        "unique_document_count": len(rows),
        "consolidated_success_count": success_count,
        "consolidated_failed_count": failed_count,
        "newly_recovered_document_count": len(state["newly_recovered_document_ids"]),
        "duplicate_document_count": state["duplicate_document_count"],
        "identical_duplicate_file_count": state["identical_duplicate_file_count"],
        "conflicting_document_count": state["conflicting_document_count"],
        "missing_success_file_count": state["missing_success_file_count"],
        "orphan_document_file_count": state["orphan_document_file_count"],
        "invalid_manifest_row_count": state["invalid_manifest_row_count"],
        "output_document_file_count": _count_text_files(output_documents_dir.parent),
        "output_manifest_row_count": len(rows),
        "blocking_reasons": blocking_reasons,
        "warnings": state["warnings"],
        "next_allowed_step": "audit_consolidated_sec_primary_text_cache" if not blocking_reasons else "resolve_sec_cache_consolidation_conflicts",
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }


def _count_text_files(cache_dir: Path) -> int:
    documents_dir = cache_dir / "documents"
    return len(list(documents_dir.glob("*.txt"))) if documents_dir.exists() else 0


def _count_orphan_files(cache_dir: Path, manifest_cache_paths: set[str]) -> int:
    documents_dir = cache_dir / "documents"
    if not documents_dir.exists():
        return 0
    manifest_names = {Path(path).name for path in manifest_cache_paths if path}
    return sum(1 for path in documents_dir.glob("*.txt") if path.name not in manifest_names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate SEC primary text caches offline.")
    parser.add_argument("--primary-cache-dir", required=True)
    parser.add_argument("--retry-cache-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args(argv)
    summary = consolidate_sec_text_caches(
        primary_cache_dir=args.primary_cache_dir,
        retry_cache_dirs=args.retry_cache_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
