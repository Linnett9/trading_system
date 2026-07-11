"""Passive provider/source candidate planning reports for stock-alpha news."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.registry import news_source_planning_registry


PROVIDER_CANDIDATE_PLANNING_SCHEMA_VERSION = "stock_alpha_news.provider_candidate_planning.v1"
PROTECTED_ACTIVE_BACKFILL_PATH = (
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
    "stock_alpha_news_historical_backfill_alpaca_benzinga_full/dev"
)

READY_FOR_DISABLED_DRY_RUN_DESIGN = "READY_FOR_DISABLED_DRY_RUN_DESIGN"
READY_FOR_MANUAL_FIXTURE_ONLY = "READY_FOR_MANUAL_FIXTURE_ONLY"
NEEDS_TERMS_REVIEW = "NEEDS_TERMS_REVIEW"
NEEDS_MAPPING_REVIEW = "NEEDS_MAPPING_REVIEW"
NEEDS_TIMESTAMP_REVIEW = "NEEDS_TIMESTAMP_REVIEW"
NOT_READY = "NOT_READY"
DO_NOT_USE_YET = "DO_NOT_USE_YET"


@dataclass(frozen=True)
class ProviderCandidatePlanningPaths:
    """Scratch artifacts written by the passive provider planning helper."""

    report_json_path: Path
    summary_markdown_path: Path


def build_provider_candidate_planning_report(
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank static provider/source candidates without creating provider objects."""

    candidate_rows = list(candidates) if candidates is not None else default_provider_candidates()
    ranked = sorted(
        [_candidate_report(candidate) for candidate in candidate_rows],
        key=_rank_key,
    )
    recommended = _recommended_candidate(ranked)
    return {
        "schema_version": PROVIDER_CANDIDATE_PLANNING_SCHEMA_VERSION,
        "artifact_type": "provider_candidate_planning_report",
        "candidate_count": len(ranked),
        "ranked_candidates": ranked,
        "recommended_candidate_id": recommended.get("candidate_id") if recommended else None,
        "recommended_next_action": recommended.get("recommended_next_action") if recommended else None,
        "blockers": _top_level_messages(ranked, "blockers"),
        "warnings": _top_level_messages(ranked, "warnings"),
        "safety_flags": {
            "provider_collection_invoked": False,
            "provider_object_instantiated": False,
            "network_invoked": False,
            "download_invoked": False,
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


def write_provider_candidate_planning_report(
    report_dir: str | Path,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> ProviderCandidatePlanningPaths:
    """Write a passive planning report to a caller-supplied scratch directory."""

    report_root = Path(report_dir)
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")
    paths = ProviderCandidatePlanningPaths(
        report_json_path=report_root / "provider_candidate_planning_report.json",
        summary_markdown_path=report_root / "provider_candidate_planning_summary.md",
    )
    _ensure_paths_under_report_dir(report_root, paths)
    report = build_provider_candidate_planning_report(candidates)
    report["output_files"] = {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    writer = ResearchArtifactWriter()
    writer.write_json(paths.report_json_path, report)
    writer.write_markdown(paths.summary_markdown_path, _markdown(report))
    return paths


def default_provider_candidates() -> list[dict[str, Any]]:
    """Return static planning candidates derived from local disabled metadata."""

    registry = news_source_planning_registry()
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "manual_fixture_source",
            "provider_family": "manual_fixture",
            "source_type": "manual_fixture",
            "implementation_status": "fixture_workflow_available",
            "requires_api_key": False,
            "requires_network": False,
            "historical_depth_expectation": "tiny_fixture_only",
            "symbol_mapping_risk": "low",
            "timestamp_quality_expectation": "fixture_controlled",
            "text_quality_expectation": "fixture_controlled",
            "rate_limit_risk": "low",
            "license_or_terms_risk": "low",
            "deduplication_risk": "low",
            "relevance_noise_risk": "low",
            "point_in_time_risk": "low",
            "integration_risk": "low",
            "recommended_next_action": "Use manual fixtures to refine disabled dry-run output expectations.",
        }
    ]
    for name, metadata in sorted(registry.items()):
        candidates.append(_candidate_from_registry(name, metadata))
    return candidates


def _candidate_from_registry(name: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    source_category = str(metadata.get("source_category", "unknown"))
    limitations = tuple(str(value) for value in metadata.get("known_limitations", ()) or ())
    terms_status = str(metadata.get("terms_or_licensing_review_status", "unknown"))
    timestamp_semantics = str(metadata.get("timestamp_semantics", ""))
    return {
        "candidate_id": name,
        "provider_family": name,
        "source_type": source_category,
        "implementation_status": str(metadata.get("current_implementation_status", "unknown")),
        "requires_api_key": any("key" in value.lower() or "entitlement" in value.lower() for value in limitations),
        "requires_network": True,
        "historical_depth_expectation": str(metadata.get("historical_support_status", "unknown")),
        "symbol_mapping_risk": "high" if any("symbol" in value.lower() for value in limitations) else "medium",
        "timestamp_quality_expectation": timestamp_semantics,
        "text_quality_expectation": ",".join(str(value) for value in metadata.get("expected_text_fields", ()) or ()),
        "rate_limit_risk": "medium" if any("rate" in value.lower() for value in limitations) else "unknown",
        "license_or_terms_risk": "high" if "required" in terms_status else "medium",
        "deduplication_risk": "medium",
        "relevance_noise_risk": "high" if "web_news" in str(metadata.get("intended_event_coverage", ())) else "medium",
        "point_in_time_risk": "medium" if "not guaranteed" not in timestamp_semantics.lower() else "high",
        "integration_risk": "medium",
        "recommended_next_action": "Design a disabled dry-run plan only after reviewing terms, mapping, and timestamps.",
    }


def _candidate_report(candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): _json_ready(value) for key, value in candidate.items()}
    blockers, warnings = _candidate_messages(payload)
    readiness = _readiness(payload, blockers, warnings)
    score = _score(payload, blockers, warnings, readiness)
    payload.update(
        {
            "readiness": readiness,
            "score": score,
            "blockers": blockers,
            "warnings": warnings,
            "safety_flags": {
                "provider_collection_invoked": False,
                "provider_object_instantiated": False,
                "network_invoked": False,
                "download_invoked": False,
                "historical_backfill_invoked": False,
                "feature_generation_invoked": False,
                "model_training_invoked": False,
                "model_inference_invoked": False,
                "trading_impact": "none",
            },
        }
    )
    return payload


def _candidate_messages(candidate: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    status = _text(candidate.get("implementation_status")).lower()
    if status in {"", "not_implemented", "unknown"}:
        blockers.append("implementation_not_ready")
    if _bool(candidate.get("requires_network")):
        warnings.append("requires_network_for_future_collection")
    if _bool(candidate.get("requires_api_key")):
        warnings.append("requires_api_key_or_entitlement")
    _risk_message(
        candidate,
        "license_or_terms_risk",
        high_blocker="terms_or_license_review_required",
        medium_warning="terms_or_license_review_recommended",
        blockers=blockers,
        warnings=warnings,
    )
    _risk_message(
        candidate,
        "symbol_mapping_risk",
        high_blocker="symbol_mapping_review_required",
        medium_warning="symbol_mapping_review_recommended",
        blockers=blockers,
        warnings=warnings,
    )
    _risk_message(
        candidate,
        "point_in_time_risk",
        high_blocker="point_in_time_review_required",
        medium_warning="point_in_time_review_recommended",
        blockers=blockers,
        warnings=warnings,
    )
    timestamp = _text(candidate.get("timestamp_quality_expectation")).lower()
    if any(token in timestamp for token in ("poor", "unknown", "not guaranteed", "seen date")):
        blockers.append("timestamp_quality_review_required")
    elif timestamp and "fixture" not in timestamp and "published" not in timestamp and "accepted" not in timestamp:
        warnings.append("timestamp_semantics_review_recommended")
    for field, warning in (
        ("rate_limit_risk", "rate_limit_review_recommended"),
        ("deduplication_risk", "deduplication_review_recommended"),
        ("relevance_noise_risk", "relevance_noise_review_recommended"),
        ("integration_risk", "integration_review_recommended"),
    ):
        if _risk(candidate.get(field)) in {"medium", "high", "unknown"}:
            warnings.append(warning)
    return sorted(set(blockers)), sorted(set(warnings))


def _risk_message(
    candidate: Mapping[str, Any],
    field: str,
    *,
    high_blocker: str,
    medium_warning: str,
    blockers: list[str],
    warnings: list[str],
) -> None:
    risk = _risk(candidate.get(field))
    if risk in {"high", "unknown"}:
        blockers.append(high_blocker)
    elif risk == "medium":
        warnings.append(medium_warning)


def _readiness(candidate: Mapping[str, Any], blockers: list[str], warnings: list[str]) -> str:
    source_type = _text(candidate.get("source_type")).lower()
    if _text(candidate.get("recommended_next_action")).lower().startswith("do not use"):
        return DO_NOT_USE_YET
    if blockers:
        if "terms_or_license_review_required" in blockers:
            return NEEDS_TERMS_REVIEW
        if "symbol_mapping_review_required" in blockers:
            return NEEDS_MAPPING_REVIEW
        if "timestamp_quality_review_required" in blockers or "point_in_time_review_required" in blockers:
            return NEEDS_TIMESTAMP_REVIEW
        return NOT_READY
    if source_type == "manual_fixture" or _text(candidate.get("implementation_status")) == "fixture_workflow_available":
        return READY_FOR_MANUAL_FIXTURE_ONLY
    if warnings:
        if "terms_or_license_review_recommended" in warnings:
            return NEEDS_TERMS_REVIEW
        if "symbol_mapping_review_recommended" in warnings:
            return NEEDS_MAPPING_REVIEW
        if "timestamp_semantics_review_recommended" in warnings or "point_in_time_review_recommended" in warnings:
            return NEEDS_TIMESTAMP_REVIEW
    return READY_FOR_DISABLED_DRY_RUN_DESIGN


def _score(candidate: Mapping[str, Any], blockers: list[str], warnings: list[str], readiness: str) -> int:
    score = 100
    score -= 25 * len(blockers)
    score -= 5 * len(warnings)
    for field in (
        "rate_limit_risk",
        "license_or_terms_risk",
        "deduplication_risk",
        "relevance_noise_risk",
        "point_in_time_risk",
        "integration_risk",
        "symbol_mapping_risk",
    ):
        risk = _risk(candidate.get(field))
        if risk == "high":
            score -= 15
        elif risk == "unknown":
            score -= 10
        elif risk == "medium":
            score -= 5
    if _bool(candidate.get("requires_network")):
        score -= 10
    if _bool(candidate.get("requires_api_key")):
        score -= 5
    if readiness == READY_FOR_MANUAL_FIXTURE_ONLY:
        score += 5
    if readiness == READY_FOR_DISABLED_DRY_RUN_DESIGN:
        score += 10
    if readiness == DO_NOT_USE_YET:
        score -= 50
    return max(0, score)


def _rank_key(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    return (len(candidate.get("blockers", []) or []), -int(candidate.get("score", 0) or 0), _text(candidate.get("candidate_id")))


def _recommended_candidate(ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in ranked:
        if candidate.get("readiness") != DO_NOT_USE_YET:
            return candidate
    return ranked[0] if ranked else None


def _top_level_messages(ranked: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    values = []
    for candidate in ranked:
        candidate_id = _text(candidate.get("candidate_id"))
        for message in candidate.get(field, []) or []:
            values.append(f"{candidate_id}:{message}")
    return sorted(values)


def _risk(value: Any) -> str:
    text = _text(value).lower()
    if text in {"low", "medium", "high"}:
        return text
    return "unknown"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _ensure_paths_under_report_dir(
    report_root: Path,
    paths: ProviderCandidatePlanningPaths,
) -> None:
    root = report_root.resolve(strict=False)
    for path in (paths.report_json_path, paths.summary_markdown_path):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("provider candidate planning outputs must stay under report_dir") from exc


def _markdown(report: Mapping[str, Any]) -> str:
    safety = dict(report.get("safety_flags", {}) or {})
    lines = [
        "# Provider Candidate Planning",
        "",
        f"- Schema version: {report['schema_version']}",
        f"- Artifact type: {report['artifact_type']}",
        f"- Candidate count: {report['candidate_count']}",
        f"- Recommended candidate: {report['recommended_candidate_id']}",
        f"- Recommended next action: {report['recommended_next_action']}",
        f"- Blockers: {report['blockers']}",
        f"- Warnings: {report['warnings']}",
        f"- Provider collection invoked: {safety['provider_collection_invoked']}",
        f"- Provider object instantiated: {safety['provider_object_instantiated']}",
        f"- Network invoked: {safety['network_invoked']}",
        f"- Download invoked: {safety['download_invoked']}",
        f"- Historical backfill invoked: {safety['historical_backfill_invoked']}",
        f"- Feature generation invoked: {safety['feature_generation_invoked']}",
        f"- Model training invoked: {safety['model_training_invoked']}",
        f"- Model inference invoked: {safety['model_inference_invoked']}",
        f"- Trading impact: {safety['trading_impact']}",
        "",
        "## Ranked Candidates",
        "",
    ]
    for candidate in report.get("ranked_candidates", []) or []:
        lines.extend(
            [
                f"- {candidate['candidate_id']}: {candidate['readiness']} "
                f"(score={candidate['score']}, blockers={candidate['blockers']}, warnings={candidate['warnings']})",
            ]
        )
    lines.append("")
    return "\n".join(lines)
