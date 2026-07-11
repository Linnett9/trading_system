"""Explicit manual-fixture workflow for tiny stock-alpha news corpus smoke reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.corpus_composition_smoke import (
    write_corpus_composition_smoke_report,
)
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
    load_rows_from_path,
)


MANUAL_FIXTURE_WORKFLOW_SCHEMA_VERSION = "stock_alpha_news.manual_fixture_workflow.v1"


@dataclass(frozen=True)
class ManualFixtureWorkflowPaths:
    """Scratch artifacts written by the explicit manual-fixture workflow."""

    workflow_report_json_path: Path
    workflow_summary_markdown_path: Path
    composition_dir: Path


def write_manual_fixture_workflow_report(
    fixture_path: str | Path,
    report_dir: str | Path,
    *,
    sample_size: int,
) -> tuple[dict[str, Any], ManualFixtureWorkflowPaths]:
    """Run the explicit fixture-to-composition smoke workflow."""

    fixture = Path(fixture_path)
    report_root = Path(report_dir)
    if _contains_protected_path(fixture):
        raise ValueError("fixture_path must not reference the protected active backfill path")
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")

    rows = load_rows_from_path(fixture)
    paths = ManualFixtureWorkflowPaths(
        workflow_report_json_path=report_root / "manual_fixture_workflow_report.json",
        workflow_summary_markdown_path=report_root / "manual_fixture_workflow_summary.md",
        composition_dir=report_root / "composition",
    )
    _ensure_paths_under_report_dir(report_root, paths)

    composition_report, composition_paths = write_corpus_composition_smoke_report(
        rows,
        paths.composition_dir,
        sample_size=sample_size,
    )
    report = _workflow_report(
        fixture_path=fixture,
        sample_size=sample_size,
        composition_report=composition_report,
        composition_report_path=composition_paths.report_json_path,
    )
    report["output_files"] = {
        "workflow_report_json": str(paths.workflow_report_json_path),
        "workflow_summary_markdown": str(paths.workflow_summary_markdown_path),
        "composition_report_json": str(composition_paths.report_json_path),
        "composition_summary_markdown": str(composition_paths.summary_markdown_path),
    }

    writer = ResearchArtifactWriter()
    writer.write_json(paths.workflow_report_json_path, report)
    writer.write_markdown(paths.workflow_summary_markdown_path, _markdown(report))
    return report, paths


def _workflow_report(
    *,
    fixture_path: Path,
    sample_size: int,
    composition_report: Mapping[str, Any],
    composition_report_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": MANUAL_FIXTURE_WORKFLOW_SCHEMA_VERSION,
        "artifact_type": "manual_fixture_composition_smoke_workflow",
        "fixture_path": str(fixture_path),
        "fixture_format": _fixture_format(fixture_path),
        "sample_size": sample_size,
        "input_row_count": int(composition_report.get("input_row_count", 0) or 0),
        "selected_row_count": int(composition_report.get("selected_row_count", 0) or 0),
        "corpus_row_count": int(composition_report.get("corpus_row_count", 0) or 0),
        "skipped_row_count": int(composition_report.get("skipped_row_count", 0) or 0),
        "sample_excluded_row_count": int(composition_report.get("sample_excluded_row_count", 0) or 0),
        "composition_report_path": str(composition_report_path),
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
        "blockers": list(composition_report.get("blockers", []) or []),
        "warnings": list(composition_report.get("warnings", []) or []),
    }


def _fixture_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def _ensure_paths_under_report_dir(
    report_root: Path,
    paths: ManualFixtureWorkflowPaths,
) -> None:
    root = report_root.resolve(strict=False)
    for path in (
        paths.workflow_report_json_path,
        paths.workflow_summary_markdown_path,
        paths.composition_dir,
    ):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("manual fixture workflow outputs must stay under report_dir") from exc


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _markdown(report: Mapping[str, Any]) -> str:
    safety = dict(report.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# Manual Fixture Workflow",
            "",
            f"- Schema version: {report['schema_version']}",
            f"- Artifact type: {report['artifact_type']}",
            f"- Fixture path: {report['fixture_path']}",
            f"- Fixture format: {report['fixture_format']}",
            f"- Sample size: {report['sample_size']}",
            f"- Input rows: {report['input_row_count']}",
            f"- Selected rows: {report['selected_row_count']}",
            f"- Corpus rows: {report['corpus_row_count']}",
            f"- Skipped corpus rows: {report['skipped_row_count']}",
            f"- Sample excluded rows: {report['sample_excluded_row_count']}",
            f"- Composition report: {report['composition_report_path']}",
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
