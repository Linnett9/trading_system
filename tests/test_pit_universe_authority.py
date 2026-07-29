from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.reference.canonical_assets import canonical_asset_id
from core.research.ml.reference.pit_universe_authority import (
    QUERY_RESULT_SCHEMA_VERSION,
    STATIC_UNCERTIFIED_STATUS,
    PointInTimeUniverseAuthority,
    load_pit_universe_authority,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pit_universe_authority"
UNIVERSE_ID = "synthetic_pit_v1"


def test_pre_ipo_exclusion_and_listing_date_inclusion():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    before = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-06-14")
    assert "perm_ipo_002" not in before["eligible_asset_ids"]
    assert _row(before["exclusions"], "perm_ipo_002")["reason_code"] == "PRE_LISTING"

    on_listing = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-06-15")
    assert "perm_ipo_002" in on_listing["eligible_asset_ids"]
    assert _row(on_listing["eligible_assets"], "perm_ipo_002")["symbol"] == "IPOC"


def test_delisting_boundary_is_last_eligible_date():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    last_day = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-09-30")
    assert "perm_delisted_003" in last_day["eligible_asset_ids"]

    after = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-10-01")
    assert "perm_delisted_003" not in after["eligible_asset_ids"]
    assert _row(after["exclusions"], "perm_delisted_003")["reason_code"] == "POST_DELISTING"


def test_ticker_change_preserves_permanent_asset_id():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    before = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-05-31")
    after = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-06-01")

    assert "perm_ticker_004" in before["eligible_asset_ids"]
    assert "perm_ticker_004" in after["eligible_asset_ids"]
    assert _row(before["eligible_assets"], "perm_ticker_004")["symbol"] == "OLDT"
    assert _row(after["eligible_assets"], "perm_ticker_004")["symbol"] == "NEWT"


def test_merger_predecessor_and_successor_are_distinct_assets():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    before_close = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-08-31")
    assert "perm_merger_pred_005" in before_close["eligible_asset_ids"]
    assert _row(before_close["exclusions"], "perm_merger_succ_006")["reason_code"] == "PRE_LISTING"

    after_close = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-09-01")
    assert "perm_merger_succ_006" in after_close["eligible_asset_ids"]
    assert "perm_merger_pred_005" not in after_close["eligible_asset_ids"]
    assert _row(after_close["exclusions"], "perm_merger_pred_005")["reason_code"] == "TERMINAL_EVENT:merger_predecessor_terminal"


def test_spin_off_parent_and_child_eligibility_dates():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    before = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-14")
    assert "perm_spin_parent_007" in before["eligible_asset_ids"]
    assert _row(before["exclusions"], "perm_spin_child_008")["reason_code"] == "PRE_LISTING"

    on_spin = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-15")
    assert "perm_spin_parent_007" in on_spin["eligible_asset_ids"]
    assert "perm_spin_child_008" in on_spin["eligible_asset_ids"]


def test_bankruptcy_terminal_event_excludes_from_effective_date():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    before = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-08-14")
    assert "perm_bankrupt_009" in before["eligible_asset_ids"]

    on_terminal = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-08-15")
    assert "perm_bankrupt_009" not in on_terminal["eligible_asset_ids"]
    assert _row(on_terminal["exclusions"], "perm_bankrupt_009")["reason_code"] == "TERMINAL_EVENT:bankruptcy"


def test_unknown_listing_and_delisting_status_are_reported_unresolved():
    authority = load_pit_universe_authority(FIXTURE_ROOT)
    result = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-01")

    assert _row(result["unresolved_assets"], "perm_unknown_listing_010")["reason_code"] == "UNKNOWN_LISTING_DATE"
    assert _row(result["unresolved_assets"], "perm_unknown_delisting_011")["reason_code"] == "UNKNOWN_DELISTING_STATUS"
    assert "perm_unknown_listing_010" not in result["eligible_asset_ids"]
    assert "perm_unknown_delisting_011" not in result["eligible_asset_ids"]


def test_conflicting_membership_sources_are_reported_not_included():
    authority = load_pit_universe_authority(FIXTURE_ROOT)
    result = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-01")

    conflict = _row(result["conflicts"], "perm_conflict_012")
    assert conflict["reason_code"] == "CONFLICTING_MEMBERSHIP"
    assert result["coverage_status"] == "CONFLICTING_AUTHORITY"
    assert "perm_conflict_012" not in result["eligible_asset_ids"]


def test_authority_version_and_knowledge_cutoff_reproduce_correction_state():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    original = authority.query(
        universe_id=UNIVERSE_ID,
        decision_timestamp="2020-05-15",
        authority_version="synthetic_v1",
    )
    assert "perm_correction_013" not in original["eligible_asset_ids"]
    assert _row(original["unresolved_assets"], "perm_correction_013")["reason_code"] == "NO_MEMBERSHIP_AUTHORITY"

    corrected = authority.query(
        universe_id=UNIVERSE_ID,
        decision_timestamp="2020-05-15",
        authority_version="synthetic_v2",
    )
    assert "perm_correction_013" in corrected["eligible_asset_ids"]
    assert _row(corrected["eligible_assets"], "perm_correction_013")["reason_code"] == "CORRECTED_EARLIER_MEMBERSHIP_START"

    cutoff_before_correction = authority.query(
        universe_id=UNIVERSE_ID,
        decision_timestamp="2020-05-15",
        authority_version="latest",
        knowledge_cutoff="2020-06-01T00:00:00+00:00",
    )
    assert cutoff_before_correction["authority_version"] == "synthetic_v1"
    assert "perm_correction_013" not in cutoff_before_correction["eligible_asset_ids"]


def test_static_universe_adapter_does_not_silently_create_eligibility():
    authority = PointInTimeUniverseAuthority.from_static_universe(
        universe_id="legacy_static",
        symbols=["AAPL", "MSFT"],
        authority_version="legacy_static_universe.v1",
        recorded_at="2026-07-28T00:00:00+00:00",
    )

    result = authority.query(universe_id="legacy_static", decision_timestamp="2020-01-02")

    assert result["coverage_status"] == STATIC_UNCERTIFIED_STATUS
    assert result["eligible_asset_ids"] == []
    assert {row["symbol"] for row in result["unresolved_assets"]} == {"AAPL", "MSFT"}
    assert {row["reason_code"] for row in result["unresolved_assets"]} == {STATIC_UNCERTIFIED_STATUS}


def test_fixture_static_current_registry_asset_is_uncertified():
    authority = load_pit_universe_authority(FIXTURE_ROOT)
    result = authority.query(universe_id="legacy_static", decision_timestamp="2020-01-02")

    assert result["coverage_status"] == STATIC_UNCERTIFIED_STATUS
    assert result["eligible_asset_ids"] == []
    row = next(row for row in result["unresolved_assets"] if row["symbol"] == "STATC")
    assert row["reason_code"] == STATIC_UNCERTIFIED_STATUS


def test_provider_alias_changes_independently_from_permanent_asset_id():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    old_alias = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-06-30")
    new_alias = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-01")

    assert "perm_provider_alias_014" in old_alias["eligible_asset_ids"]
    assert "perm_provider_alias_014" in new_alias["eligible_asset_ids"]
    assert _row(old_alias["eligible_assets"], "perm_provider_alias_014")["provider_symbols"] == {"alpaca": "PALP1"}
    assert _row(new_alias["eligible_assets"], "perm_provider_alias_014")["provider_symbols"] == {"alpaca": "PALP2"}


def test_result_ordering_and_serialization_are_deterministic():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    left = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-01")
    right = authority.query(universe_id=UNIVERSE_ID, decision_timestamp="2020-07-01")

    assert left["schema_version"] == QUERY_RESULT_SCHEMA_VERSION
    assert left["eligible_asset_ids"] == sorted(left["eligible_asset_ids"])
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_validation_artifact_documents_lineage_and_failure_policy(tmp_path):
    authority = load_pit_universe_authority(FIXTURE_ROOT)
    output = tmp_path / "pit_validation.json"

    authority.write_validation_artifact(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "pit_universe_authority_validation.v1"
    assert payload["record_counts"]["security_master"] == 14
    assert payload["failure_policy"]["static_universe_adapter"] == STATIC_UNCERTIFIED_STATUS
    assert payload["failure_policy"]["price_row_presence"] == "does_not_prove_membership"
    assert payload["precedence_policy"][0]["authority_type"] == "manual_verified_security_master"


def test_permanent_asset_identity_is_not_ticker_symbol():
    authority = load_pit_universe_authority(FIXTURE_ROOT)

    for record in authority.security_master:
        assert record.asset_id != record.canonical_symbol
    assert canonical_asset_id("AAPL") == canonical_asset_id("aapl")


def _row(rows: list[dict], asset_id: str) -> dict:
    for row in rows:
        if row["asset_id"] == asset_id:
            return row
    raise AssertionError(f"missing row for {asset_id}")
