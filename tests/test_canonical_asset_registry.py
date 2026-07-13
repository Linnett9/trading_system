from __future__ import annotations

from pathlib import Path

import pytest

from core.research.ml.reference.canonical_assets import (
    CanonicalAsset,
    ProviderAlias,
    alpaca_provider_symbol,
    ambiguous_aliases,
    build_and_audit,
    build_dataset_manifest,
    build_registry_from_universe,
    canonical_asset_id,
    daily_spine_row_id,
    duplicate_provider_aliases,
    registry_content_hash,
    validate_feature_family_row,
    validate_registry,
)


def test_deterministic_asset_ids():
    assert canonical_asset_id("aapl") == canonical_asset_id("AAPL")
    assert canonical_asset_id("AAPL") != canonical_asset_id("MSFT")


def test_deterministic_registry_ordering_and_hashing(tmp_path):
    universe = tmp_path / "universe.txt"
    universe.write_text("MSFT\nAAPL\n", encoding="utf-8")
    left_assets, left_aliases, _ = build_registry_from_universe(universe)
    right_assets, right_aliases, _ = build_registry_from_universe(universe)
    assert [row.canonical_symbol for row in left_assets] == ["AAPL", "MSFT"]
    assert registry_content_hash(left_assets, left_aliases) == registry_content_hash(right_assets, right_aliases)


def test_brk_a_to_alpaca_dot_mapping():
    assert alpaca_provider_symbol("BRK-A") == "BRK.A"


def test_brk_b_to_alpaca_dot_mapping():
    assert alpaca_provider_symbol("BRK-B") == "BRK.B"


def test_duplicate_active_canonical_symbol_rejection():
    asset = _asset("AAPL")
    duplicate = CanonicalAsset(**{**asset.__dict__, "asset_id": canonical_asset_id("AAPL") + "x"})
    with pytest.raises(ValueError, match="duplicate active canonical symbols"):
        validate_registry([asset, duplicate], [])


def test_duplicate_overlapping_provider_alias_rejection():
    left = _alias("asset_a", "alpaca", "ABC", "1900-01-01", "")
    right = _alias("asset_b", "alpaca", "ABC", "2000-01-01", "")
    with pytest.raises(ValueError, match="duplicate overlapping provider aliases"):
        validate_registry([_asset("AAA"), _asset("BBB")], [left, right])


def test_historical_non_overlapping_aliases_allowed():
    left = _alias("asset_a", "alpaca", "OLD", "1900-01-01", "1999-12-31")
    right = _alias("asset_b", "alpaca", "OLD", "2000-01-01", "")
    assert duplicate_provider_aliases([left, right]) == []


def test_ambiguous_mappings_reported_rather_than_guessed():
    left = _alias("asset_a", "news", "ABC", "1900-01-01", "")
    right = _alias("asset_b", "news", "ABC", "1900-01-01", "")
    assert ambiguous_aliases([left, right]) == [{"provider": "news", "provider_symbol": "ABC"}]


def test_missing_cik_allowed_but_reported(tmp_path):
    universe = tmp_path / "universe.txt"
    universe.write_text("AAPL\n", encoding="utf-8")
    report_dir = tmp_path / "reports"
    audit = build_and_audit(
        dry_run=True,
        universe_path=universe,
        registry_output=tmp_path / "assets.csv",
        alias_output=tmp_path / "aliases.csv",
        parquet_output=tmp_path / "assets.parquet",
        report_dir=report_dir,
        repo_root=tmp_path,
    )
    assert audit["missing_sec_ciks"] == ["AAPL"]


def test_collection_membership_does_not_imply_selector_eligibility(tmp_path):
    universe = tmp_path / "universe.txt"
    universe.write_text("AAPL\n", encoding="utf-8")
    audit = build_and_audit(
        dry_run=True,
        universe_path=universe,
        registry_output=tmp_path / "assets.csv",
        alias_output=tmp_path / "aliases.csv",
        parquet_output=tmp_path / "assets.parquet",
        report_dir=tmp_path / "reports",
        repo_root=tmp_path,
    )
    assert audit["selector_eligibility_inferred"] is False


def test_daily_spine_row_ids_are_independent_of_row_ordering():
    row_a = daily_spine_row_id(
        asset_id="asset_1",
        decision_timestamp="2026-01-02T21:00:00+00:00",
        target_horizon_sessions=10,
        universe_version="u1",
        daily_price_dataset_version="d1",
        target_definition_version="t1",
    )
    row_b = daily_spine_row_id(
        target_definition_version="t1",
        daily_price_dataset_version="d1",
        universe_version="u1",
        target_horizon_sessions=10,
        decision_timestamp="2026-01-02T21:00:00+00:00",
        asset_id="asset_1",
    )
    assert row_a == row_b


def test_pit_feature_validation_rejects_availability_after_decision_time():
    with pytest.raises(ValueError, match="feature_available_timestamp"):
        validate_feature_family_row(
            {
                "row_id": "row_1",
                "asset_id": "asset_1",
                "decision_timestamp": "2026-01-02T21:00:00+00:00",
                "feature_available_timestamp": "2026-01-03T00:00:00+00:00",
                "feature_family": "sec_fundamentals",
                "feature_version": "v1",
                "source_dataset_version": "s1",
            }
        )


def test_manifest_status_blocked_when_required_inputs_absent(tmp_path):
    manifest = build_dataset_manifest(
        dataset_type="canonical_daily_spine",
        row_grain="one asset at one selector decision timestamp",
        primary_keys=("row_id",),
        required_source_paths=(tmp_path / "missing.parquet",),
    )
    assert manifest.status == "BLOCKED"
    assert manifest.warnings == (f"missing_required_source:{tmp_path / 'missing.parquet'}",)


def test_dry_run_writes_no_registry_outputs(tmp_path):
    universe = tmp_path / "universe.txt"
    asset_output = tmp_path / "assets.csv"
    alias_output = tmp_path / "aliases.csv"
    parquet_output = tmp_path / "assets.parquet"
    universe.write_text("AAPL\n", encoding="utf-8")
    build_and_audit(
        dry_run=True,
        universe_path=universe,
        registry_output=asset_output,
        alias_output=alias_output,
        parquet_output=parquet_output,
        report_dir=tmp_path / "reports",
        repo_root=tmp_path,
    )
    assert not asset_output.exists()
    assert not alias_output.exists()
    assert not parquet_output.exists()


def test_existing_unrelated_files_remain_unchanged(tmp_path):
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    universe = tmp_path / "universe.txt"
    universe.write_text("AAPL\n", encoding="utf-8")
    build_and_audit(
        dry_run=True,
        universe_path=universe,
        registry_output=tmp_path / "assets.csv",
        alias_output=tmp_path / "aliases.csv",
        parquet_output=tmp_path / "assets.parquet",
        report_dir=tmp_path / "reports",
        repo_root=tmp_path,
    )
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def _asset(symbol: str) -> CanonicalAsset:
    return CanonicalAsset(
        asset_id=canonical_asset_id(symbol),
        canonical_symbol=symbol,
        security_name=None,
        security_type="UNKNOWN",
        share_class=None,
        exchange=None,
        currency="USD",
        country="US",
        cik=None,
        sector=None,
        industry=None,
        valid_from="1900-01-01",
        valid_to="",
        is_active=True,
        collection_universe_514=True,
        registry_version="test",
    )


def _alias(asset_id: str, provider: str, provider_symbol: str, valid_from: str, valid_to: str) -> ProviderAlias:
    return ProviderAlias(
        asset_id=asset_id,
        provider=provider,
        provider_symbol=provider_symbol,
        valid_from=valid_from,
        valid_to=valid_to,
        is_primary=True,
        mapping_reason="test",
        source="test",
        registry_version="test",
    )

