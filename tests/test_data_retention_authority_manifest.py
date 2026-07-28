from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.research.ml.reference.data_retention_authority import (
    RetentionManifestValidationError,
    load_retention_manifest,
    retention_manifest_hash,
    stable_manifest_serialization,
    validate_retention_manifest,
)


MANIFEST_PATH = Path("config/data_retention_authority_manifest.v1.json")


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _family(payload: dict, family_id: str) -> dict:
    for row in payload["families"]:
        if row["family_id"] == family_id:
            return row
    raise AssertionError(f"missing family {family_id}")


def test_manifest_loads_and_represents_required_ticket37a_families():
    payload = load_retention_manifest(MANIFEST_PATH)
    family_ids = {row["family_id"] for row in payload["families"]}

    assert {
        "alpaca_raw_5m_archive",
        "alpaca_converted_chunks",
        "alpaca_final_5m_store",
        "legacy_raw_5m_text",
        "canonical_daily_v2",
        "seven_row_canonical_daily_candidate",
        "stooq_daily_raw_store",
        "stooq_daily_processed_store",
        "market_parquet_daily_store",
        "legacy_processed_symbol_trees",
        "adjusted_price_csv_collection",
        "news_canonical_exports_and_backups",
        "sec_fundamentals_raw_evidence",
        "ml_feature_banks",
        "frozen_experiment_inputs",
        "reproducibility_critical_reports",
        "regenerable_report_runs",
        "model_binaries_finbert",
        "universe_registry_files",
        "pit_universe_authority_data",
        "test_fixtures",
    } <= family_ids
    assert payload["cleanup_executed"] is False


def test_every_family_has_retention_classification_and_deterministic_path_pattern():
    payload = load_retention_manifest(MANIFEST_PATH)
    for family in payload["families"]:
        assert family["retention_classification"]
        assert "\\" not in family["path_pattern"]
        assert not Path(family["path_pattern"].split(";")[0]).is_absolute()


def test_unknown_classifications_fail_closed_and_unknown_values_rejected():
    payload = _manifest()
    unknown = _family(payload, "legacy_raw_5m_text")
    assert unknown["retention_classification"] == "UNKNOWN_DO_NOT_DELETE"
    assert unknown["cleanup_eligibility"] == "NOT_ELIGIBLE"
    assert unknown["cleanup_confidence"] == "FAIL_CLOSED"

    broken = copy.deepcopy(payload)
    _family(broken, "legacy_raw_5m_text")["retention_classification"] = "DELETE_ME"
    with pytest.raises(RetentionManifestValidationError, match="unknown retention_classification"):
        validate_retention_manifest(broken)


def test_cleanup_eligibility_is_explicit_and_not_implied_by_rebuildability():
    payload = _manifest()
    rebuilt = _family(payload, "canonical_daily_v2")
    assert rebuilt["rebuildability"] == "REBUILDABLE_WITH_PREREQUISITES"
    assert rebuilt["cleanup_eligibility"] == "NOT_ELIGIBLE"

    missing = copy.deepcopy(payload)
    del _family(missing, "canonical_daily_v2")["cleanup_eligibility"]
    with pytest.raises(RetentionManifestValidationError, match="missing required fields"):
        validate_retention_manifest(missing)


def test_raw_authority_cannot_be_marked_cleanup_proposal_eligible():
    payload = _manifest()
    broken = copy.deepcopy(payload)
    _family(broken, "alpaca_raw_5m_archive")["cleanup_eligibility"] = "FUTURE_PROPOSAL_ALLOWED"
    with pytest.raises(RetentionManifestValidationError, match="raw authority"):
        validate_retention_manifest(broken)


def test_frozen_experiment_inputs_and_acceptance_reports_require_retention():
    payload = _manifest()
    frozen = _family(payload, "frozen_experiment_inputs")
    acceptance = _family(payload, "reproducibility_critical_reports")
    assert frozen["cleanup_eligibility"] == "NOT_ELIGIBLE"
    assert acceptance["cleanup_eligibility"] == "NOT_ELIGIBLE"

    broken = copy.deepcopy(payload)
    _family(broken, "frozen_experiment_inputs")["cleanup_eligibility"] = "REVIEW_ONLY"
    with pytest.raises(RetentionManifestValidationError, match="frozen evidence"):
        validate_retention_manifest(broken)


def test_pit_and_knowledge_time_records_are_not_collapsed_without_evidence():
    payload = _manifest()
    pit = _family(payload, "pit_universe_authority_data")
    assert pit["cleanup_eligibility"] == "NOT_ELIGIBLE"
    assert pit["pit_or_knowledge_time_significance"] == "PIT_CRITICAL"

    broken = copy.deepcopy(payload)
    _family(broken, "pit_universe_authority_data")["cleanup_eligibility"] = "REVIEW_ONLY"
    with pytest.raises(RetentionManifestValidationError, match="PIT/knowledge-time"):
        validate_retention_manifest(broken)


def test_canonical_authorities_are_unambiguous_by_domain():
    payload = load_retention_manifest(MANIFEST_PATH)
    family_ids = {row["family_id"] for row in payload["families"]}
    for domain, rule in payload["authority_domains"].items():
        assert rule["canonical_family_id"] in family_ids, domain
        assert rule["intended_authority"]
        assert rule["fallback_policy"]
        assert rule["conflict_policy"]

    broken = copy.deepcopy(payload)
    broken["authority_domains"]["daily_prices"]["canonical_family_id"] = "missing_family"
    with pytest.raises(RetentionManifestValidationError, match="unknown canonical family"):
        validate_retention_manifest(broken)


def test_manifest_serialization_and_hash_are_stable():
    left = load_retention_manifest(MANIFEST_PATH)
    right = load_retention_manifest(MANIFEST_PATH)

    assert stable_manifest_serialization(left) == stable_manifest_serialization(right)
    assert retention_manifest_hash(left) == retention_manifest_hash(right)


def test_semantic_overlaps_and_exact_duplicates_remain_review_only():
    payload = load_retention_manifest(MANIFEST_PATH)

    assert payload["exact_duplicate_reviews"]
    assert all("paths" in row and len(row["paths"]) >= 2 for row in payload["exact_duplicate_reviews"])
    assert all(row["consolidation_allowed"] is False for row in payload["semantic_overlap_reviews"])
    assert all(row["proof_required"] for row in payload["semantic_overlap_reviews"])
