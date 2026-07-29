from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.reference.canonical_assets import (
    CanonicalAsset,
    ProviderAlias,
    canonical_asset_id,
)
from core.research.ml.reference.historical_identity_authority import (
    AMBIGUOUS,
    CONFLICTING_AUTHORITY,
    OUTSIDE_EFFECTIVE_WINDOW,
    PRODUCTION_SYMBOL_RESOLUTION_ENABLED_DEFAULT,
    RESOLVED,
    UNKNOWN_AT_KNOWLEDGE_CUTOFF,
    HistoricalIdentityAuthority,
    enrich_pit_universe_result,
    load_historical_identity_authority,
    stable_resolution_serialization,
)
from core.research.ml.reference.pit_universe_authority import load_pit_universe_authority


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "historical_identity_authority"
FIXTURE_PATH = FIXTURE_ROOT / "fixture.json"
PIT_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pit_universe_authority"


def _authority() -> HistoricalIdentityAuthority:
    return load_historical_identity_authority(FIXTURE_PATH)


def test_permanent_identity_survives_ticker_changes_and_symbol_lookup_changes():
    authority = _authority()

    old_symbol = authority.resolve_symbol("OLDX", timestamp="2020-05-31T12:00:00Z")
    new_symbol = authority.resolve_symbol("NEWX", timestamp="2020-06-01T12:00:00Z")
    before = authority.symbol_for_asset("perm_rename_alpha", timestamp="2020-05-31T12:00:00Z")
    after = authority.symbol_for_asset("perm_rename_alpha", timestamp="2020-06-01T12:00:00Z")

    assert old_symbol["state"] == RESOLVED
    assert new_symbol["state"] == RESOLVED
    assert old_symbol["permanent_asset_id"] == "perm_rename_alpha"
    assert new_symbol["permanent_asset_id"] == "perm_rename_alpha"
    assert before["symbol"] == "OLDX"
    assert after["symbol"] == "NEWX"


def test_company_name_change_does_not_change_symbol_or_identity():
    authority = _authority()

    old_name = authority.company_name_for_asset("perm_name_only", timestamp="2020-06-30T12:00:00Z")
    new_name = authority.company_name_for_asset("perm_name_only", timestamp="2020-07-01T12:00:00Z")
    symbol = authority.resolve_symbol("NAME", timestamp="2020-07-01T12:00:00Z")

    assert old_name["symbol"] == "Old Name Industries"
    assert new_name["symbol"] == "New Name Industries"
    assert symbol["permanent_asset_id"] == "perm_name_only"


def test_provider_aliases_are_effective_dated_and_do_not_redefine_identity():
    authority = _authority()

    old_alias = authority.resolve_provider_alias("alpaca", "PALP1", timestamp="2020-06-30T12:00:00Z")
    new_alias = authority.resolve_provider_alias("alpaca", "PALP2", timestamp="2020-07-01T12:00:00Z")
    expired = authority.resolve_provider_alias("alpaca", "PALP1", timestamp="2020-07-01T12:00:00Z")
    canonical = authority.resolve_symbol("PAL", timestamp="2020-07-01T12:00:00Z")

    assert old_alias["permanent_asset_id"] == "perm_provider_alias"
    assert new_alias["permanent_asset_id"] == "perm_provider_alias"
    assert expired["state"] == OUTSIDE_EFFECTIVE_WINDOW
    assert canonical["permanent_asset_id"] == "perm_provider_alias"


def test_exchange_suffix_variation_resolves_as_provider_alias():
    authority = _authority()

    result = authority.resolve_provider_alias("stooq", "SUF.US", timestamp="2020-03-01T12:00:00Z")

    assert result["state"] == RESOLVED
    assert result["permanent_asset_id"] == "perm_suffix"


def test_ticker_reuse_does_not_merge_distinct_securities():
    authority = _authority()

    old_owner = authority.resolve_symbol("RUSE", timestamp="2020-03-01T12:00:00Z")
    inactive_gap = authority.resolve_symbol("RUSE", timestamp="2020-05-01T12:00:00Z")
    new_owner = authority.resolve_symbol("RUSE", timestamp="2020-07-01T12:00:00Z")

    assert old_owner["permanent_asset_id"] == "perm_reuse_old"
    assert inactive_gap["state"] == OUTSIDE_EFFECTIVE_WINDOW
    assert new_owner["permanent_asset_id"] == "perm_reuse_new"


def test_merger_predecessor_successor_histories_remain_separate():
    authority = _authority()

    predecessor_before = authority.symbol_for_asset("perm_merger_pred", timestamp="2020-08-30T12:00:00Z")
    predecessor_after = authority.symbol_for_asset("perm_merger_pred", timestamp="2020-09-01T12:00:00Z")
    acquirer_after = authority.symbol_for_asset("perm_merger_acquirer", timestamp="2020-09-01T12:00:00Z")
    successor = authority.successors("perm_merger_pred")
    acquirer_terminal = authority.terminal_events("perm_merger_acquirer", timestamp="2020-09-01T12:00:00Z")

    assert predecessor_before["state"] == RESOLVED
    assert predecessor_after["state"] == OUTSIDE_EFFECTIVE_WINDOW
    assert acquirer_after["state"] == RESOLVED
    assert successor["successor_asset_ids"] == ["perm_merger_acquirer"]
    assert acquirer_terminal["state"] != RESOLVED


def test_merger_creating_new_successor_uses_distinct_permanent_asset():
    authority = _authority()

    successor = authority.successors("perm_merger_new_left")
    predecessors = authority.predecessors("perm_merger_new_successor")

    assert successor["successor_asset_ids"] == ["perm_merger_new_successor"]
    assert set(predecessors["predecessor_asset_ids"]) == {
        "perm_merger_new_left",
        "perm_merger_new_right",
    }
    assert "new_successor_security_created" in predecessors["events"][0]["relationship_semantics"]


def test_acquisition_variants_are_explicit_and_do_not_compute_returns():
    authority = _authority()

    cash = authority.terminal_events("perm_cash_target", timestamp="2020-05-21T12:00:00Z")
    stock = authority.terminal_events("perm_stock_unresolved_target", timestamp="2020-10-02T12:00:00Z")

    assert cash["terminal_event_types"] == ["cash_acquisition"]
    assert cash["events"][0]["terminal_economics"] == "known_cash_consideration_not_return"
    assert stock["terminal_event_types"] == ["acquisition"]
    assert "stock_ratio" in stock["events"][0]["unresolved_fields"]


def test_spin_off_child_unavailable_before_effective_date():
    authority = _authority()

    before = authority.symbol_for_asset("perm_spin_child", timestamp="2020-07-14T12:00:00Z")
    after = authority.symbol_for_asset("perm_spin_child", timestamp="2020-07-15T12:00:00Z")
    parent_successors = authority.successors("perm_spin_parent")

    assert before["state"] == OUTSIDE_EFFECTIVE_WINDOW
    assert after["state"] == RESOLVED
    assert parent_successors["successor_asset_ids"] == ["perm_spin_child"]


def test_bankruptcy_liquidation_delisting_and_relisting_states_are_explicit():
    authority = _authority()

    bankruptcy = authority.terminal_events("perm_bankrupt_delist", timestamp="2020-08-16T12:00:00Z")
    delisting = authority.terminal_events("perm_bankrupt_delist", timestamp="2020-08-21T12:00:00Z")
    liquidation = authority.terminal_events("perm_liquidation_unknown", timestamp="2020-11-02T12:00:00Z")
    relisted_after = authority.symbol_for_asset("perm_relisted", timestamp="2020-05-02T12:00:00Z")

    assert bankruptcy["terminal_event_types"] == ["bankruptcy"]
    assert delisting["terminal_event_types"] == ["bankruptcy", "delisting"]
    assert liquidation["events"][0]["terminal_economics"] == "unknown"
    assert relisted_after["permanent_asset_id"] == "perm_relisted"


def test_ambiguous_and_conflicting_aliases_fail_closed():
    authority = _authority()

    ambiguous = authority.resolve_provider_alias("news", "AMBIG", timestamp="2020-02-01T12:00:00Z")
    conflicting = authority.resolve_provider_alias("alpaca", "CNFL", timestamp="2020-02-01T12:00:00Z")

    assert ambiguous["state"] == AMBIGUOUS
    assert set(ambiguous["candidate_asset_ids"]) == {"perm_ambig_left", "perm_ambig_right"}
    assert conflicting["state"] == CONFLICTING_AUTHORITY
    assert set(conflicting["candidate_asset_ids"]) == {"perm_conflict_left", "perm_conflict_right"}


def test_recorded_at_cutoff_is_enforced_for_retrospective_correction():
    authority = _authority()

    before_cutoff = authority.resolve_symbol(
        "CORR",
        timestamp="2020-01-15T12:00:00Z",
        knowledge_cutoff="2020-02-01T00:00:00Z",
    )
    after_cutoff = authority.resolve_symbol(
        "CORR",
        timestamp="2020-01-15T12:00:00Z",
        knowledge_cutoff="2020-04-01T00:00:00Z",
    )

    assert before_cutoff["state"] == UNKNOWN_AT_KNOWLEDGE_CUTOFF
    assert before_cutoff["authority_version"] == "histid_v1"
    assert after_cutoff["state"] == RESOLVED
    assert after_cutoff["authority_version"] == "histid_v2"


def test_serialization_authority_versions_and_validation_are_deterministic():
    authority = _authority()
    left = authority.resolve_symbol("NEWX", timestamp="2020-06-02T12:00:00Z")
    right = authority.resolve_symbol("NEWX", timestamp="2020-06-02T12:00:00Z")
    validation = authority.validation_artifact()

    assert stable_resolution_serialization(left) == stable_resolution_serialization(right)
    assert left["resolution_id"] == right["resolution_id"]
    assert authority.authority_versions_at() == ("histid_v1", "histid_v2")
    assert validation["production_selection_enabled"] is False
    assert validation["failure_policy"]["merger_prices"] == "never_concatenate_automatically"


def test_ticket37_pit_result_can_be_enriched_without_changing_original_query():
    identity = _authority()
    pit = load_pit_universe_authority(PIT_FIXTURE_ROOT)
    original = pit.query(universe_id="synthetic_pit_v1", decision_timestamp="2020-06-01")
    original_json = json.dumps(original, sort_keys=True)

    enriched = enrich_pit_universe_result(original, identity)
    ticker_row = next(row for row in enriched["eligible_assets"] if row["asset_id"] == "perm_ticker_004")

    assert json.dumps(original, sort_keys=True) == original_json
    assert ticker_row["permanent_asset_id"] == "perm_ticker_004"
    assert ticker_row["historically_valid_symbol"] == "NEWT"
    assert ticker_row["historical_identity_resolution_state"] == RESOLVED
    assert enriched["historical_identity_enrichment"]["production_selection_enabled"] is False


def test_canonical_registry_adapter_is_static_low_precedence_compatibility_only():
    asset = CanonicalAsset(
        asset_id=canonical_asset_id("AAPL"),
        canonical_symbol="AAPL",
        security_name=None,
        security_type="UNKNOWN",
        share_class=None,
        exchange="XNAS",
        currency="USD",
        country="US",
        cik=None,
        sector=None,
        industry=None,
        valid_from="1900-01-01",
        valid_to="",
        is_active=True,
        collection_universe_514=True,
        registry_version="fixture_registry",
    )
    alias = ProviderAlias(
        asset_id=asset.asset_id,
        provider="alpaca",
        provider_symbol="AAPL",
        valid_from="1900-01-01",
        valid_to="",
        is_primary=True,
        mapping_reason="identity",
        source="fixture",
        registry_version="fixture_registry",
    )

    authority = HistoricalIdentityAuthority.from_canonical_registry([asset], [alias])
    result = authority.resolve_symbol("AAPL", timestamp="2020-01-01T12:00:00Z")
    validation = authority.validation_artifact()

    assert result["state"] == RESOLVED
    assert result["permanent_asset_id"] == asset.asset_id
    assert result["matched_records"][0]["authority_type"] == "canonical_registry_default"
    assert validation["production_selection_enabled"] is PRODUCTION_SYMBOL_RESOLUTION_ENABLED_DEFAULT
