"""Explicit smoke composition for tiny stock-alpha news corpus reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.corpus_dry_run import write_corpus_dry_run
from core.research.ml.stock_level.news_sources.corpus_readiness_audit import (
    write_corpus_readiness_audit,
)
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
    write_corpus_sample_selection,
)


CORPUS_COMPOSITION_SMOKE_SCHEMA_VERSION = "stock_alpha_news.corpus_composition_smoke.v1"


@dataclass(frozen=True)
class CorpusCompositionSmokePaths:
    """Scratch artifacts written by the explicit composition smoke helper."""

    report_json_path: Path
    summary_markdown_path: Path
    sample_selection_dir: Path
    readiness_dir: Path
    corpus_dir: Path


def write_corpus_composition_smoke_report(
    rows: Sequence[Mapping[str, Any]],
    report_dir: str | Path,
    *,
    sample_size: int,
) -> tuple[dict[str, Any], CorpusCompositionSmokePaths]:
    """Write a deterministic scratch bundle by composing Phase 5, 3, and 4."""

    report_root = Path(report_dir)
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")

    paths = CorpusCompositionSmokePaths(
        report_json_path=report_root / "composition_smoke_report.json",
        summary_markdown_path=report_root / "composition_smoke_summary.md",
        sample_selection_dir=report_root / "sample_selection",
        readiness_dir=report_root / "readiness",
        corpus_dir=report_root / "corpus",
    )
    _ensure_paths_under_report_dir(report_root, paths)

    sample_paths = write_corpus_sample_selection(
        rows,
        paths.sample_selection_dir,
        sample_size=sample_size,
    )
    selected_rows = _read_json(sample_paths.sample_rows_json_path)
    if not isinstance(selected_rows, list):
        raise ValueError("sample selector output must be a list")

    readiness_paths = write_corpus_readiness_audit(
        selected_rows,
        paths.readiness_dir,
        rows_are_canonical=True,
    )
    corpus_paths = write_corpus_dry_run(
        selected_rows,
        paths.corpus_dir,
        rows_are_canonical=True,
    )

    sample_audit = _read_json(sample_paths.audit_json_path)
    readiness_audit = _read_json(readiness_paths.audit_json_path)
    corpus_manifest = _read_json(corpus_paths.manifest_json_path)
    report = _composition_report(
        input_row_count=len(rows),
        sample_audit=sample_audit,
        readiness_audit=readiness_audit,
        corpus_manifest=corpus_manifest,
    )
    report["sample_output_files"] = dict(sample_audit.get("output_files", {}) or {})
    report["readiness_output_files"] = dict(readiness_audit.get("output_files", {}) or {})
    report["corpus_output_files"] = dict(corpus_manifest.get("output_files", {}) or {})
    report["output_files"] = {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }

    writer = ResearchArtifactWriter()
    writer.write_json(paths.report_json_path, report)
    writer.write_markdown(paths.summary_markdown_path, _markdown(report))
    return report, paths


def _composition_report(
    *,
    input_row_count: int,
    sample_audit: Mapping[str, Any],
    readiness_audit: Mapping[str, Any],
    corpus_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selected_row_count = int(sample_audit.get("selected_row_count", 0) or 0)
    corpus_row_count = int(corpus_manifest.get("corpus_row_count", 0) or 0)
    blockers = list(readiness_audit.get("blockers", []) or [])
    if selected_row_count == 0:
        blockers.append("no_sample_rows_selected")
    warnings = list(readiness_audit.get("warnings", []) or [])
    return {
        "schema_version": CORPUS_COMPOSITION_SMOKE_SCHEMA_VERSION,
        "artifact_type": "sample_corpus_composition_smoke",
        "input_row_count": input_row_count,
        "selected_row_count": selected_row_count,
        "corpus_row_count": corpus_row_count,
        "skipped_row_count": int(corpus_manifest.get("skipped_row_count", 0) or 0),
        "sample_excluded_row_count": int(sample_audit.get("excluded_row_count", 0) or 0),
        "sample_skip_reasons": dict(sample_audit.get("skip_reasons", {}) or {}),
        "corpus_skip_reasons": dict(corpus_manifest.get("skip_reasons", {}) or {}),
        "sample_selection_recommendation": (
            "READY_FOR_COMPOSITION_SMOKE" if selected_row_count > 0 else "NO_SAMPLE_ROWS_SELECTED"
        ),
        "readiness_recommendation": readiness_audit.get("recommendation"),
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "sample_output_files": {},
        "readiness_output_files": {},
        "corpus_output_files": {},
        "output_files": {},
        "safety_flags": {
            "provider_collection_invoked": False,
            "network_invoked": False,
            "canonical_ingest_invoked": False,
            "historical_backfill_invoked": False,
            "corpus_assembly_invoked": False,
            "feature_generation_invoked": False,
            "model_training_invoked": False,
            "model_inference_invoked": False,
            "trading_impact": "none",
            "protected_active_backfill_path_rejected": True,
        },
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_paths_under_report_dir(
    report_root: Path,
    paths: CorpusCompositionSmokePaths,
) -> None:
    root = report_root.resolve(strict=False)
    for path in (
        paths.report_json_path,
        paths.summary_markdown_path,
        paths.sample_selection_dir,
        paths.readiness_dir,
        paths.corpus_dir,
    ):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("composition smoke outputs must stay under report_dir") from exc


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _markdown(report: Mapping[str, Any]) -> str:
    safety = dict(report.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# Corpus Composition Smoke",
            "",
            f"- Schema version: {report['schema_version']}",
            f"- Artifact type: {report['artifact_type']}",
            f"- Input rows: {report['input_row_count']}",
            f"- Selected rows: {report['selected_row_count']}",
            f"- Corpus rows: {report['corpus_row_count']}",
            f"- Skipped corpus rows: {report['skipped_row_count']}",
            f"- Sample excluded rows: {report['sample_excluded_row_count']}",
            f"- Sample skip reasons: {report['sample_skip_reasons']}",
            f"- Corpus skip reasons: {report['corpus_skip_reasons']}",
            f"- Sample selection recommendation: {report['sample_selection_recommendation']}",
            f"- Readiness recommendation: {report['readiness_recommendation']}",
            f"- Blockers: {report['blockers']}",
            f"- Warnings: {report['warnings']}",
            f"- Provider collection invoked: {safety['provider_collection_invoked']}",
            f"- Historical backfill invoked: {safety['historical_backfill_invoked']}",
            f"- Corpus assembly invoked: {safety['corpus_assembly_invoked']}",
            f"- Feature generation invoked: {safety['feature_generation_invoked']}",
            f"- Model training invoked: {safety['model_training_invoked']}",
            f"- Model inference invoked: {safety['model_inference_invoked']}",
            f"- Trading impact: {safety['trading_impact']}",
            "",
        ]
    )
