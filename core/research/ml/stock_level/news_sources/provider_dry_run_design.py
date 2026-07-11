"""Passive disabled provider dry-run design reports for stock-alpha news."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.provider_candidate_planning import (
    NEEDS_MAPPING_REVIEW,
    NEEDS_TERMS_REVIEW,
    NEEDS_TIMESTAMP_REVIEW,
    READY_FOR_DISABLED_DRY_RUN_DESIGN,
    READY_FOR_MANUAL_FIXTURE_ONLY,
    build_provider_candidate_planning_report,
)


PROVIDER_DRY_RUN_DESIGN_SCHEMA_VERSION = "stock_alpha_news.provider_dry_run_design.v1"
PROTECTED_ACTIVE_BACKFILL_PATH = (
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
    "stock_alpha_news_historical_backfill_alpaca_benzinga_full/dev"
)

DESIGN_ONLY = "DESIGN_ONLY"
READY_FOR_MOCKED_ADAPTER_TEST = "READY_FOR_MOCKED_ADAPTER_TEST"
READY_FOR_DISABLED_SCRATCH_DRY_RUN = "READY_FOR_DISABLED_SCRATCH_DRY_RUN"
BLOCKED_PENDING_TERMS_REVIEW = "BLOCKED_PENDING_TERMS_REVIEW"
BLOCKED_PENDING_TIMESTAMP_REVIEW = "BLOCKED_PENDING_TIMESTAMP_REVIEW"
BLOCKED_PENDING_MAPPING_REVIEW = "BLOCKED_PENDING_MAPPING_REVIEW"
NOT_READY = "NOT_READY"

DEFAULT_MAX_SYMBOLS = 3
DEFAULT_MAX_ROWS = 25
DEFAULT_MAX_REQUESTS = 3
DEFAULT_ALLOWED_OUTPUT_ROOT = "/private/tmp"


@dataclass(frozen=True)
class ProviderDryRunDesignPaths:
    """Scratch artifacts written by the passive dry-run design helper."""

    report_json_path: Path
    summary_markdown_path: Path


def build_provider_dry_run_design_report(
    *,
    planning_report: Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    allowed_output_root: str | Path = DEFAULT_ALLOWED_OUTPUT_ROOT,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> dict[str, Any]:
    """Design a future disabled dry-run without collecting provider data."""

    planning = dict(planning_report or build_provider_candidate_planning_report(candidates))
    candidate = _select_candidate(list(planning.get("ranked_candidates", []) or []))
    status = _dry_run_status(candidate)
    blockers = _design_blockers(candidate, status)
    warnings = _design_warnings(candidate)
    return {
        "schema_version": PROVIDER_DRY_RUN_DESIGN_SCHEMA_VERSION,
        "artifact_type": "provider_dry_run_design_report",
        "selected_candidate_id": _text(candidate.get("candidate_id")),
        "selected_provider_family": _text(candidate.get("provider_family")),
        "selected_candidate_readiness": _text(candidate.get("readiness")),
        "dry_run_status": status,
        "recommended_next_action": _recommended_next_action(candidate, status),
        "is_collection_enabled": False,
        "network_allowed": False,
        "requires_api_key": bool(candidate.get("requires_api_key", False)),
        "max_symbols": int(max_symbols),
        "max_rows": int(max_rows),
        "max_requests": int(max_requests),
        "allowed_output_root": str(allowed_output_root),
        "forbidden_paths": [PROTECTED_ACTIVE_BACKFILL_PATH],
        "required_guards": _required_guards(),
        "required_audits": _required_audits(),
        "required_test_fixtures": _required_test_fixtures(candidate),
        "blockers": blockers,
        "warnings": warnings,
        "candidate_blockers": list(candidate.get("blockers", []) or []),
        "candidate_warnings": list(candidate.get("warnings", []) or []),
        "planning_recommended_candidate_id": planning.get("recommended_candidate_id"),
        "safety_flags": {
            "provider_collection_invoked": False,
            "provider_object_instantiated": False,
            "network_invoked": False,
            "download_invoked": False,
            "api_keys_read": False,
            "config_read": False,
            "canonical_ingest_invoked": False,
            "historical_backfill_invoked": False,
            "active_backfill_path_read": False,
            "corpus_assembly_invoked": False,
            "feature_generation_invoked": False,
            "model_training_invoked": False,
            "model_inference_invoked": False,
            "trading_impact": "none",
        },
        "output_files": {},
    }


def write_provider_dry_run_design_report(
    report_dir: str | Path,
    *,
    planning_report: Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    allowed_output_root: str | Path = DEFAULT_ALLOWED_OUTPUT_ROOT,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> ProviderDryRunDesignPaths:
    """Write a passive disabled dry-run design report to a scratch directory."""

    report_root = Path(report_dir)
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")
    paths = ProviderDryRunDesignPaths(
        report_json_path=report_root / "provider_dry_run_design_report.json",
        summary_markdown_path=report_root / "provider_dry_run_design_summary.md",
    )
    _ensure_paths_under_report_dir(report_root, paths)
    report = build_provider_dry_run_design_report(
        planning_report=planning_report,
        candidates=candidates,
        allowed_output_root=allowed_output_root,
        max_symbols=max_symbols,
        max_rows=max_rows,
        max_requests=max_requests,
    )
    report["output_files"] = {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    writer = ResearchArtifactWriter()
    writer.write_json(paths.report_json_path, report)
    writer.write_markdown(paths.summary_markdown_path, _markdown(report))
    return paths


def _select_candidate(candidates: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    for candidate in candidates:
        if candidate.get("readiness") == READY_FOR_DISABLED_DRY_RUN_DESIGN:
            return candidate
    for candidate in candidates:
        if candidate.get("readiness") == READY_FOR_MANUAL_FIXTURE_ONLY:
            return candidate
    return candidates[0] if candidates else {}


def _dry_run_status(candidate: Mapping[str, Any]) -> str:
    readiness = _text(candidate.get("readiness"))
    blockers = set(candidate.get("blockers", []) or [])
    if readiness == READY_FOR_DISABLED_DRY_RUN_DESIGN:
        return READY_FOR_DISABLED_SCRATCH_DRY_RUN
    if readiness == READY_FOR_MANUAL_FIXTURE_ONLY:
        return DESIGN_ONLY
    if readiness == NEEDS_TERMS_REVIEW or "terms_or_license_review_required" in blockers:
        return BLOCKED_PENDING_TERMS_REVIEW
    if readiness == NEEDS_TIMESTAMP_REVIEW or {
        "timestamp_quality_review_required",
        "point_in_time_review_required",
    } & blockers:
        return BLOCKED_PENDING_TIMESTAMP_REVIEW
    if readiness == NEEDS_MAPPING_REVIEW or "symbol_mapping_review_required" in blockers:
        return BLOCKED_PENDING_MAPPING_REVIEW
    return NOT_READY


def _design_blockers(candidate: Mapping[str, Any], status: str) -> list[str]:
    blockers = list(candidate.get("blockers", []) or [])
    if status == DESIGN_ONLY:
        blockers.append("real_provider_dry_run_not_ready_use_manual_fixture_only")
    if status.startswith("BLOCKED"):
        blockers.append("phase9_candidate_blockers_must_be_resolved_before_provider_dry_run")
    if status == NOT_READY:
        blockers.append("candidate_not_ready_for_disabled_provider_dry_run_design")
    return sorted(set(blockers))


def _design_warnings(candidate: Mapping[str, Any]) -> list[str]:
    warnings = list(candidate.get("warnings", []) or [])
    if bool(candidate.get("requires_network", False)):
        warnings.append("future_provider_dry_run_requires_network_but_network_remains_disabled_by_default")
    if bool(candidate.get("requires_api_key", False)):
        warnings.append("future_provider_dry_run_requires_api_key_but_keys_must_not_be_read_in_design_phase")
    warnings.append("design_report_is_not_permission_to_collect")
    return sorted(set(warnings))


def _recommended_next_action(candidate: Mapping[str, Any], status: str) -> str:
    if status == READY_FOR_DISABLED_SCRATCH_DRY_RUN:
        return "Draft mocked-adapter tests and a disabled scratch-only provider dry-run gate; do not collect data."
    if status == DESIGN_ONLY:
        return "Continue manual fixture validation; real-provider dry-run remains blocked."
    if status == BLOCKED_PENDING_TERMS_REVIEW:
        return "Resolve terms/licensing review before any provider dry-run design advances."
    if status == BLOCKED_PENDING_TIMESTAMP_REVIEW:
        return "Resolve timestamp and point-in-time semantics before any provider dry-run design advances."
    if status == BLOCKED_PENDING_MAPPING_REVIEW:
        return "Resolve symbol mapping and relevance rules before any provider dry-run design advances."
    return _text(candidate.get("recommended_next_action")) or "Resolve planning blockers before dry-run design advances."


def _required_guards() -> list[str]:
    return [
        "explicit_enable_flag_required",
        "network_disabled_by_default",
        "scratch_output_directory_required",
        "protected_active_backfill_path_rejected",
        "max_request_cap_enforced",
        "max_row_cap_enforced",
        "max_symbol_cap_enforced",
        "no_feature_generation",
        "no_model_training_or_inference",
        "no_replay_or_trading",
        "full_json_and_markdown_audit_output_required",
    ]


def _required_audits() -> list[str]:
    return [
        "provider_request_plan_audit",
        "provider_response_shape_audit",
        "timestamp_semantics_audit",
        "symbol_mapping_audit",
        "terms_review_record",
        "scratch_output_manifest",
    ]


def _required_test_fixtures(candidate: Mapping[str, Any]) -> list[str]:
    fixtures = [
        "manual_fixture_workflow_tiny.jsonl",
        "mock_provider_success_response",
        "mock_provider_empty_response",
        "mock_provider_rate_limit_or_entitlement_response",
    ]
    if bool(candidate.get("requires_api_key", False)):
        fixtures.append("mock_missing_api_key_diagnostic")
    return fixtures


def _ensure_paths_under_report_dir(report_root: Path, paths: ProviderDryRunDesignPaths) -> None:
    root = report_root.resolve(strict=False)
    for path in (paths.report_json_path, paths.summary_markdown_path):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("provider dry-run design outputs must stay under report_dir") from exc


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _text(value: Any) -> str:
    return str(value or "").strip()


def _markdown(report: Mapping[str, Any]) -> str:
    safety = dict(report.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# Provider Dry-Run Design",
            "",
            f"- Schema version: {report['schema_version']}",
            f"- Artifact type: {report['artifact_type']}",
            f"- Selected candidate: {report['selected_candidate_id']}",
            f"- Selected provider family: {report['selected_provider_family']}",
            f"- Candidate readiness: {report['selected_candidate_readiness']}",
            f"- Dry-run status: {report['dry_run_status']}",
            f"- Recommended next action: {report['recommended_next_action']}",
            f"- Collection enabled: {report['is_collection_enabled']}",
            f"- Network allowed: {report['network_allowed']}",
            f"- Requires API key: {report['requires_api_key']}",
            f"- Max symbols: {report['max_symbols']}",
            f"- Max rows: {report['max_rows']}",
            f"- Max requests: {report['max_requests']}",
            f"- Allowed output root: {report['allowed_output_root']}",
            f"- Forbidden paths: {report['forbidden_paths']}",
            f"- Required guards: {report['required_guards']}",
            f"- Required audits: {report['required_audits']}",
            f"- Required test fixtures: {report['required_test_fixtures']}",
            f"- Blockers: {report['blockers']}",
            f"- Warnings: {report['warnings']}",
            f"- Provider collection invoked: {safety['provider_collection_invoked']}",
            f"- Provider object instantiated: {safety['provider_object_instantiated']}",
            f"- Network invoked: {safety['network_invoked']}",
            f"- API keys read: {safety['api_keys_read']}",
            f"- Historical backfill invoked: {safety['historical_backfill_invoked']}",
            f"- Feature generation invoked: {safety['feature_generation_invoked']}",
            f"- Model training invoked: {safety['model_training_invoked']}",
            f"- Model inference invoked: {safety['model_inference_invoked']}",
            f"- Trading impact: {safety['trading_impact']}",
            "",
        ]
    )
