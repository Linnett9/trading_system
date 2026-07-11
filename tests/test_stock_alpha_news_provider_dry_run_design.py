from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.news_sources.provider_candidate_planning import (
    build_provider_candidate_planning_report,
)
from core.research.ml.stock_level.news_sources.provider_dry_run_design import (
    BLOCKED_PENDING_TERMS_REVIEW,
    BLOCKED_PENDING_TIMESTAMP_REVIEW,
    DESIGN_ONLY,
    PROVIDER_DRY_RUN_DESIGN_SCHEMA_VERSION,
    PROTECTED_ACTIVE_BACKFILL_PATH,
    READY_FOR_DISABLED_SCRATCH_DRY_RUN,
    build_provider_dry_run_design_report,
    write_provider_dry_run_design_report,
)


def test_ready_low_risk_candidate_produces_disabled_scratch_dry_run_design() -> None:
    planning = build_provider_candidate_planning_report([_candidate("company_rss_candidate")])

    report = build_provider_dry_run_design_report(planning_report=planning)

    assert report["schema_version"] == PROVIDER_DRY_RUN_DESIGN_SCHEMA_VERSION
    assert report["artifact_type"] == "provider_dry_run_design_report"
    assert report["selected_candidate_id"] == "company_rss_candidate"
    assert report["dry_run_status"] == READY_FOR_DISABLED_SCRATCH_DRY_RUN
    assert report["is_collection_enabled"] is False
    assert report["network_allowed"] is False
    assert report["max_symbols"] == 3
    assert report["max_rows"] == 25
    assert report["max_requests"] == 3
    assert "explicit_enable_flag_required" in report["required_guards"]
    assert "network_disabled_by_default" in report["required_guards"]
    assert "scratch_output_directory_required" in report["required_guards"]
    assert "protected_active_backfill_path_rejected" in report["required_guards"]
    assert "max_request_cap_enforced" in report["required_guards"]
    assert "max_row_cap_enforced" in report["required_guards"]
    assert "max_symbol_cap_enforced" in report["required_guards"]
    assert "no_feature_generation" in report["required_guards"]
    assert "no_model_training_or_inference" in report["required_guards"]
    assert "no_replay_or_trading" in report["required_guards"]


def test_unresolved_terms_candidate_remains_blocked() -> None:
    planning = build_provider_candidate_planning_report(
        [_candidate("paid_provider_candidate", license_or_terms_risk="high")]
    )

    report = build_provider_dry_run_design_report(planning_report=planning)

    assert report["selected_candidate_id"] == "paid_provider_candidate"
    assert report["dry_run_status"] == BLOCKED_PENDING_TERMS_REVIEW
    assert "terms_or_license_review_required" in report["blockers"]
    assert "phase9_candidate_blockers_must_be_resolved_before_provider_dry_run" in report["blockers"]
    assert report["is_collection_enabled"] is False
    assert report["network_allowed"] is False


def test_poor_timestamp_candidate_remains_blocked() -> None:
    planning = build_provider_candidate_planning_report(
        [
            _candidate(
                "web_news_index_candidate",
                point_in_time_risk="high",
                timestamp_quality_expectation="seen date, not guaranteed original publication",
            )
        ]
    )

    report = build_provider_dry_run_design_report(planning_report=planning)

    assert report["selected_candidate_id"] == "web_news_index_candidate"
    assert report["dry_run_status"] == BLOCKED_PENDING_TIMESTAMP_REVIEW
    assert "timestamp_quality_review_required" in report["blockers"]
    assert "point_in_time_review_required" in report["blockers"]


def test_only_manual_fixture_candidate_marks_real_provider_dry_run_not_ready() -> None:
    planning = build_provider_candidate_planning_report(
        [
            _candidate(
                "manual_fixture_source",
                source_type="manual_fixture",
                implementation_status="fixture_workflow_available",
                requires_network=False,
            )
        ]
    )

    report = build_provider_dry_run_design_report(planning_report=planning)

    assert report["selected_candidate_id"] == "manual_fixture_source"
    assert report["dry_run_status"] == DESIGN_ONLY
    assert "real_provider_dry_run_not_ready_use_manual_fixture_only" in report["blockers"]
    assert report["recommended_next_action"] == "Continue manual fixture validation; real-provider dry-run remains blocked."


def test_output_files_are_written_only_under_supplied_tmp_path(tmp_path: Path) -> None:
    report_dir = tmp_path / "dry-run-design"

    paths = write_provider_dry_run_design_report(report_dir, candidates=[_candidate("company_rss_candidate")])

    assert paths.report_json_path.parent == report_dir
    assert paths.summary_markdown_path.parent == report_dir
    assert paths.report_json_path.name == "provider_dry_run_design_report.json"
    assert paths.summary_markdown_path.name == "provider_dry_run_design_summary.md"
    paths.report_json_path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))
    paths.summary_markdown_path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))

    report = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
    assert report["output_files"] == {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    assert "Dry-run status: READY_FOR_DISABLED_SCRATCH_DRY_RUN" in paths.summary_markdown_path.read_text(
        encoding="utf-8"
    )


def test_protected_active_backfill_output_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="protected active backfill"):
        write_provider_dry_run_design_report(
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "dry-run-design",
            candidates=[_candidate("company_rss_candidate")],
        )


def test_no_provider_object_api_keys_network_config_backfill_or_model_paths_are_needed() -> None:
    report = build_provider_dry_run_design_report(candidates=[_candidate("company_rss_candidate")])

    assert report["safety_flags"] == {
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
    }
    assert report["is_collection_enabled"] is False
    assert report["network_allowed"] is False


def _candidate(
    candidate_id: str,
    *,
    provider_family: str = "candidate_family",
    source_type: str = "official_company_rss",
    implementation_status: str = "adapter_available",
    requires_api_key: bool = False,
    requires_network: bool = True,
    historical_depth_expectation: str = "unknown_until_disabled_dry_run",
    symbol_mapping_risk: str = "low",
    timestamp_quality_expectation: str = "published timestamp expected",
    text_quality_expectation: str = "headline and summary expected",
    rate_limit_risk: str = "low",
    license_or_terms_risk: str = "low",
    deduplication_risk: str = "low",
    relevance_noise_risk: str = "low",
    point_in_time_risk: str = "low",
    integration_risk: str = "low",
    recommended_next_action: str = "Design disabled dry-run only.",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "provider_family": provider_family,
        "source_type": source_type,
        "implementation_status": implementation_status,
        "requires_api_key": requires_api_key,
        "requires_network": requires_network,
        "historical_depth_expectation": historical_depth_expectation,
        "symbol_mapping_risk": symbol_mapping_risk,
        "timestamp_quality_expectation": timestamp_quality_expectation,
        "text_quality_expectation": text_quality_expectation,
        "rate_limit_risk": rate_limit_risk,
        "license_or_terms_risk": license_or_terms_risk,
        "deduplication_risk": deduplication_risk,
        "relevance_noise_risk": relevance_noise_risk,
        "point_in_time_risk": point_in_time_risk,
        "integration_risk": integration_risk,
        "recommended_next_action": recommended_next_action,
    }
