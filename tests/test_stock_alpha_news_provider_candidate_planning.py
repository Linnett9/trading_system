from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.stock_level.news_sources.provider_candidate_planning import (
    NEEDS_TERMS_REVIEW,
    NEEDS_TIMESTAMP_REVIEW,
    PROVIDER_CANDIDATE_PLANNING_SCHEMA_VERSION,
    READY_FOR_DISABLED_DRY_RUN_DESIGN,
    build_provider_candidate_planning_report,
    default_provider_candidates,
    write_provider_candidate_planning_report,
)
from core.research.ml.stock_level.news_sources.registry import news_source_planning_registry


def test_static_candidates_produce_deterministic_ranked_report() -> None:
    report = build_provider_candidate_planning_report(
        [
            _candidate("gdelt_candidate", requires_network=True, relevance_noise_risk="high"),
            _candidate("company_rss_candidate", requires_network=True),
            _candidate("manual_fixture_source", source_type="manual_fixture", implementation_status="fixture_workflow_available"),
        ]
    )

    assert report["schema_version"] == PROVIDER_CANDIDATE_PLANNING_SCHEMA_VERSION
    assert report["artifact_type"] == "provider_candidate_planning_report"
    assert report["candidate_count"] == 3
    assert [candidate["candidate_id"] for candidate in report["ranked_candidates"]] == [
        "company_rss_candidate",
        "manual_fixture_source",
        "gdelt_candidate",
    ]
    assert report["recommended_candidate_id"] == "company_rss_candidate"
    assert report["recommended_next_action"] == "Design disabled dry-run only."
    assert report["ranked_candidates"][0]["readiness"] == READY_FOR_DISABLED_DRY_RUN_DESIGN
    assert report["safety_flags"]["provider_collection_invoked"] is False
    assert report["safety_flags"]["provider_object_instantiated"] is False
    assert report["safety_flags"]["network_invoked"] is False


def test_high_risk_candidate_is_not_incorrectly_marked_ready() -> None:
    report = build_provider_candidate_planning_report(
        [
            _candidate(
                "risky_free_api",
                implementation_status="unknown",
                license_or_terms_risk="high",
                symbol_mapping_risk="high",
                point_in_time_risk="high",
                timestamp_quality_expectation="unknown",
            )
        ]
    )

    candidate = report["ranked_candidates"][0]
    assert candidate["readiness"] != READY_FOR_DISABLED_DRY_RUN_DESIGN
    assert "implementation_not_ready" in candidate["blockers"]
    assert "terms_or_license_review_required" in candidate["blockers"]
    assert "symbol_mapping_review_required" in candidate["blockers"]
    assert "point_in_time_review_required" in candidate["blockers"]
    assert "timestamp_quality_review_required" in candidate["blockers"]


def test_terms_review_candidate_gets_terms_recommendation() -> None:
    report = build_provider_candidate_planning_report(
        [
            _candidate(
                "paid_provider_candidate",
                license_or_terms_risk="high",
                recommended_next_action="Review terms before disabled dry-run design.",
            )
        ]
    )

    candidate = report["ranked_candidates"][0]
    assert candidate["readiness"] == NEEDS_TERMS_REVIEW
    assert "terms_or_license_review_required" in candidate["blockers"]
    assert report["recommended_next_action"] == "Review terms before disabled dry-run design."


def test_poor_timestamp_candidate_gets_timestamp_recommendation() -> None:
    report = build_provider_candidate_planning_report(
        [
            _candidate(
                "web_news_index_candidate",
                point_in_time_risk="high",
                timestamp_quality_expectation="seen date, not guaranteed original publication",
            )
        ]
    )

    candidate = report["ranked_candidates"][0]
    assert candidate["readiness"] == NEEDS_TIMESTAMP_REVIEW
    assert "point_in_time_review_required" in candidate["blockers"]
    assert "timestamp_quality_review_required" in candidate["blockers"]


def test_report_writes_only_under_supplied_tmp_path(tmp_path: Path) -> None:
    report_dir = tmp_path / "provider-planning"

    paths = write_provider_candidate_planning_report(report_dir, candidates=[_candidate("company_rss_candidate")])

    assert paths.report_json_path.parent == report_dir
    assert paths.summary_markdown_path.parent == report_dir
    assert paths.report_json_path.name == "provider_candidate_planning_report.json"
    assert paths.summary_markdown_path.name == "provider_candidate_planning_summary.md"
    paths.report_json_path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))
    paths.summary_markdown_path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))

    report = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
    assert report["output_files"] == {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    assert "Recommended candidate: company_rss_candidate" in paths.summary_markdown_path.read_text(encoding="utf-8")


def test_no_provider_config_network_download_backfill_features_models_or_brokers_are_needed() -> None:
    report = build_provider_candidate_planning_report([_candidate("manual_fixture_source", requires_network=False)])

    assert report["safety_flags"] == {
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
    }
    assert report["ranked_candidates"][0]["safety_flags"]["provider_object_instantiated"] is False
    assert report["ranked_candidates"][0]["safety_flags"]["network_invoked"] is False


def test_existing_phase1_registry_compatibility_is_not_broken() -> None:
    registry = news_source_planning_registry()
    candidates = default_provider_candidates()

    assert {"alpaca_benzinga", "sec_edgar", "company_ir_or_rss", "alpha_vantage", "gdelt"} <= set(registry)
    assert all(plan["collection_enabled"] is False for plan in registry.values())
    assert all(plan["canonical_ingest_enabled"] is False for plan in registry.values())
    assert all(plan["feature_generation_enabled"] is False for plan in registry.values())
    assert {"manual_fixture_source", "alpaca_benzinga", "sec_edgar", "company_ir_or_rss"} <= {
        candidate["candidate_id"] for candidate in candidates
    }


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
