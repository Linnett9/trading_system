from __future__ import annotations

from copy import deepcopy

import pytest

from core.research.ml.reference.historical_reporting_entities import (
    CONTRACT_VERSION, NON_COMPANY_POLICY, logical_sha256, resolve,
    validate_intervals, validate_override,
)


def row(**changes):
    value = {
        "asset_id": "asset_a", "canonical_symbol": "AAA",
        "security_type": "common_stock",
        "reporting_entity_id": "CIK0000000001", "cik": "0000000001",
        "mapping_status": "resolved_company", "mapping_quality": "certified",
        "effective_start_date": "2000-01-01", "effective_end_date": "",
        "knowledge_available_timestamp": "2000-01-01T00:00:00Z",
        "source_identity": "source-hash", "source_record_identity": "record-1",
        "evidence_type": "sec_submission_with_symbol_interval",
        "contract_version": CONTRACT_VERSION,
    }
    value.update(changes)
    return value


def test_current_static_mapping_cannot_authorize_history():
    current = row(effective_start_date="", knowledge_available_timestamp="2026-01-01T00:00:00Z")
    assert validate_intervals([current])["status"] == "FAILED"
    assert resolve([current], "asset_a", "2017-01-01T20:00:00Z") is None


@pytest.mark.parametrize("changes,code", [
    ({"asset_id": ""}, "SELECTOR_ASSET_ID_UNRESOLVED"),
    ({"effective_start_date": ""}, "HISTORICAL_EFFECTIVE_INTERVAL_UNRESOLVED"),
    ({"cik": ""}, "SEC_ENTITY_EVIDENCE_UNRESOLVED"),
])
def test_required_company_identity_fields_fail_closed(changes, code):
    result = validate_intervals([row(**changes)])
    assert code in {item["code"] for item in result["errors"]}


def test_future_known_mapping_is_not_resolved_earlier():
    assert resolve([row(knowledge_available_timestamp="2020-01-01T00:00:00Z")],
                   "asset_a", "2019-12-31T23:59:59Z") is None


def test_non_company_assets_have_no_fake_cik_and_all_contract_symbols_are_explicit():
    assert len(NON_COMPANY_POLICY) == 11
    rows = [
        row(asset_id=f"asset_{symbol}", canonical_symbol=symbol, reporting_entity_id="",
            cik="", mapping_status="resolved_non_company_asset", evidence_type="certified_fund_classification",
            source_record_identity=symbol)
        for symbol in NON_COMPANY_POLICY
    ]
    assert validate_intervals(rows)["status"] == "PASS"


def test_ticker_change_intervals_are_deterministic_and_non_overlapping():
    rows = [
        row(canonical_symbol="OLD", effective_end_date="2010-01-01", source_record_identity="old"),
        row(canonical_symbol="NEW", effective_start_date="2010-01-01", source_record_identity="new"),
    ]
    assert validate_intervals(rows)["overlapping_interval_count"] == 0
    assert resolve(rows, "asset_a", "2009-01-01T00:00:00Z")["canonical_symbol"] == "OLD"
    assert resolve(rows, "asset_a", "2010-01-01T00:00:00Z")["canonical_symbol"] == "NEW"


def test_symbol_reuse_does_not_merge_unrelated_assets():
    rows = [row(), row(asset_id="asset_b", reporting_entity_id="CIK0000000002",
                       cik="0000000002", source_record_identity="record-2")]
    assert resolve(rows, "asset_a", "2020-01-01T00:00:00Z")["cik"] == "0000000001"
    assert resolve(rows, "asset_b", "2020-01-01T00:00:00Z")["cik"] == "0000000002"


def test_share_classes_require_explicit_distinct_security_evidence():
    rows = [
        row(asset_id="brka", canonical_symbol="BRK-A", source_record_identity="class-a"),
        row(asset_id="brkb", canonical_symbol="BRK-B", source_record_identity="class-b",
            evidence_type="certified_share_class_relationship"),
    ]
    assert validate_intervals(rows)["status"] == "PASS"
    assert rows[0]["reporting_entity_id"] == rows[1]["reporting_entity_id"]


def test_merger_boundary_and_amendment_knowledge_are_temporal():
    rows = [
        row(effective_end_date="2020-06-01", source_record_identity="pre-merger"),
        row(reporting_entity_id="CIK0000000002", cik="0000000002",
            effective_start_date="2020-06-01", knowledge_available_timestamp="2020-06-01T12:00:00Z",
            source_record_identity="post-merger"),
    ]
    assert resolve(rows, "asset_a", "2020-06-01T11:00:00Z") is None
    assert resolve(rows, "asset_a", "2020-06-01T13:00:00Z")["cik"] == "0000000002"


def test_overlap_duplicate_and_ambiguity_fail():
    first = row(effective_end_date="2021-01-01")
    second = row(effective_start_date="2020-01-01", source_record_identity="record-2")
    result = validate_intervals([first, second, deepcopy(first)])
    assert result["overlapping_interval_count"] > 0
    assert result["duplicate_interval_count"] == 1
    ambiguous = validate_intervals([row(mapping_status="ambiguous_entity")])
    assert ambiguous["ambiguous_active_mapping_count"] == 1


def test_manual_overrides_require_evidence_review_and_affect_identity():
    override = {
        "asset_id": "a", "canonical_symbol": "A", "effective_start_date": "2000-01-01",
        "knowledge_available_timestamp": "2000-01-01T00:00:00Z", "reason": "rename",
        "evidence_reference": "accession:1", "review_status": "approved",
    }
    validate_override(override)
    with pytest.raises(ValueError, match="MANUAL_OVERRIDE_NOT_APPROVED"):
        validate_override({**override, "review_status": "pending"})
    assert logical_sha256([row()]) != logical_sha256([row(source_identity="changed")])


def test_selector_resolution_is_by_asset_and_decision_time_not_row_order():
    rows = [row(asset_id="b", source_record_identity="b"), row()]
    assert resolve(list(reversed(rows)), "asset_a", "2020-01-01T00:00:00Z")["asset_id"] == "asset_a"
