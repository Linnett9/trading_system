from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from core.research.ml.reference.ticket_62_pit_authority_population import (
    AUTHORITY_KNOWLEDGE_TIME,
    AUTHORITY_VERSION,
    CLASSIFICATION,
    DEFAULT_UNIVERSE_ID,
    REQUIRED_ARTIFACTS,
    EligibilityRuleConfig,
    build_ticket_62_pit_authority,
    file_sha256,
    load_ticket62_selector_adapter,
)


def test_ticket62_builder_emits_required_artifacts_and_rules_based_adapter(tmp_path: Path) -> None:
    _write_tiny_sources(
        tmp_path,
        {
            "AAA": [
                _market_row("asset_aaa", "AAA", "2020-01-02", close=10.0, volume=1_000_000),
                _market_row("asset_aaa", "AAA", "2020-01-03", close=10.5, volume=1_000_000),
                _market_row("asset_aaa", "AAA", "2020-01-06", close=11.0, volume=1_000_000),
            ],
            "BBB": [
                _market_row("asset_bbb", "BBB", "2020-01-02", close=2.0, volume=1_000_000),
                _market_row("asset_bbb", "BBB", "2020-01-03", close=2.1, volume=1_000_000),
                _market_row("asset_bbb", "BBB", "2020-01-06", close=2.2, volume=1_000_000),
            ],
        },
    )

    result = build_ticket_62_pit_authority(
        repo_root=tmp_path,
        output_root="authority",
        rules=EligibilityRuleConfig(
            min_observed_sessions=2,
            min_model_close=5.0,
            min_trailing_dollar_volume=0.0,
            trailing_dollar_volume_sessions=2,
        ),
    )

    root = tmp_path / "authority"
    assert result["classification"] == CLASSIFICATION
    assert result["missing_required_artifacts"] == []
    assert result["model_training_executed"] is False
    assert result["promotion_usable"] is False
    assert all((root / name).exists() for name in REQUIRED_ARTIFACTS)
    validation = json.loads((root / "pit_authority_validation.json").read_text(encoding="utf-8"))
    assert "pit_authority_validation.json" not in validation["artifact_hashes"]
    assert validation["artifact_hash_policy"]["pit_authority_validation.json"].startswith("self-referential")
    assert all(file_sha256(root / name) == digest for name, digest in validation["artifact_hashes"].items())
    coverage = json.loads((root / "coverage_summary.json").read_text(encoding="utf-8"))
    assert coverage["symbols_requested"] == 2
    assert coverage["symbols_populated"] == 2
    assert coverage["identities_externally_corroborated"] == 0
    assert coverage["internal_reconstructed_identities"] == 2
    assert coverage["static_symbol_fallback_identities"] == 2
    assert coverage["cik_coverage_count"] == 0
    assert coverage["reconstructed_eligibility_security_count"] == 2
    security_rows = pq.read_table(root / "security_master.parquet").to_pylist()
    assert {row["identity_status"] for row in security_rows} == {"internal_reconstructed"}
    assert {row["current_symbol_identity_status"] for row in security_rows} == {
        "static_symbol_fallback_uncertified"
    }
    assert all(row["external_identity_corroborated"] is False for row in security_rows)
    inventory = json.loads((root / "source_inventory.json").read_text(encoding="utf-8"))["sources"]
    required_source_fields = {
        "source_name",
        "provider",
        "local_path",
        "source_type",
        "coverage",
        "event_timestamp_support",
        "knowledge_time_support",
        "revision_behaviour",
        "licence_usage_status",
        "redistribution_restrictions",
        "trust_tier",
        "supported_authority_domains",
    }
    assert all(required_source_fields <= set(source) for source in inventory)
    adjusted_source = next(source for source in inventory if source["source_name"] == "adjusted_price_manifest")
    assert adjusted_source["authority_population_usable"] is False

    adapter = load_ticket62_selector_adapter(root)
    first_day = adapter.resolve_selector_row(
        {"asset_id": "asset_aaa", "canonical_symbol": "AAA", "decision_timestamp": "2020-01-02"}
    )
    second_day = adapter.resolve_selector_row(
        {"asset_id": "asset_aaa", "canonical_symbol": "AAA", "decision_timestamp": "2020-01-03"}
    )
    low_price = adapter.resolve_selector_row(
        {"asset_id": "asset_bbb", "canonical_symbol": "BBB", "decision_timestamp": "2020-01-06"}
    )

    assert first_day["universe_eligible"] is False
    assert "seasoning_lt_2_sessions" in first_day["reason_codes"]
    assert second_day["universe_eligible"] is True
    assert low_price["universe_eligible"] is False
    assert "price_lt_5" in low_price["reason_codes"]


def test_ticket62_builder_is_deterministic_and_does_not_use_future_data(tmp_path: Path) -> None:
    rows = [
        _market_row("asset_aaa", "AAA", "2020-01-02", close=10.0, volume=1_000_000),
        _market_row("asset_aaa", "AAA", "2020-01-03", close=10.0, volume=1_000_000),
        _market_row("asset_aaa", "AAA", "2020-01-06", close=10.0, volume=1_000_000),
    ]
    _write_tiny_sources(tmp_path, {"AAA": rows})
    rules = EligibilityRuleConfig(
        min_observed_sessions=2,
        min_model_close=5.0,
        min_trailing_dollar_volume=0.0,
        trailing_dollar_volume_sessions=2,
    )

    first = build_ticket_62_pit_authority(repo_root=tmp_path, output_root="authority_first", rules=rules)
    early_before = load_ticket62_selector_adapter(tmp_path / "authority_first").resolve_selector_row(
        {"asset_id": "asset_aaa", "canonical_symbol": "AAA", "decision_timestamp": "2020-01-03"}
    )

    rows.append(_market_row("asset_aaa", "AAA", "2030-01-02", close=1000.0, volume=100_000_000))
    _write_tiny_sources(tmp_path, {"AAA": rows})
    second = build_ticket_62_pit_authority(repo_root=tmp_path, output_root="authority_second", rules=rules)
    early_after = load_ticket62_selector_adapter(tmp_path / "authority_second").resolve_selector_row(
        {"asset_id": "asset_aaa", "canonical_symbol": "AAA", "decision_timestamp": "2020-01-03"}
    )

    assert first["classification"] == second["classification"]
    assert early_before["universe_eligible"] == early_after["universe_eligible"]
    assert early_before["reason_codes"] == early_after["reason_codes"]
    assert early_before["permanent_security_id"] == early_after["permanent_security_id"]


def test_selector_adapter_handles_symbol_change_reuse_ipo_delisting_and_relisting(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    _write_authority_root(
        root,
        security=[
            _security("sec_change", "issuer_change", "NEW", "2020-01-01", "2020-12-31", asset_id="asset_change"),
            _security("sec_reuse_old", "issuer_old", "RUSE", "2020-01-01", "2020-03-31"),
            _security("sec_reuse_new", "issuer_new", "RUSE", "2020-06-01", "2020-12-31"),
            _security("sec_ipo", "issuer_ipo", "IPO", "2020-06-15", "2020-12-31"),
            _security("sec_delist", "issuer_delist", "DLST", "2020-01-01", "2020-09-30"),
            _security("sec_relist", "issuer_relist", "RLIST", "2020-01-01", "2020-12-31"),
        ],
        symbols=[
            _symbol("sec_change", "OLD", "2020-01-01", "2020-05-31"),
            _symbol("sec_change", "NEW", "2020-06-01", "2020-12-31"),
            _symbol("sec_reuse_old", "RUSE", "2020-01-01", "2020-03-31"),
            _symbol("sec_reuse_new", "RUSE", "2020-06-01", "2020-12-31"),
            _symbol("sec_ipo", "IPO", "2020-06-15", "2020-12-31"),
            _symbol("sec_delist", "DLST", "2020-01-01", "2020-09-30"),
            _symbol("sec_relist", "RLIST", "2020-01-01", "2020-02-01"),
            _symbol("sec_relist", "RLIST", "2020-05-01", "2020-12-31"),
        ],
        memberships=[
            _membership("sec_change", "2020-01-01", "2020-12-31"),
            _membership("sec_reuse_old", "2020-01-01", "2020-03-31"),
            _membership("sec_reuse_new", "2020-06-01", "2020-12-31"),
            _membership("sec_ipo", "2020-06-15", "2020-12-31"),
            _membership("sec_delist", "2020-01-01", "2020-09-30"),
            _membership("sec_relist", "2020-01-01", "2020-02-01"),
            _membership("sec_relist", "2020-05-01", "2020-12-31"),
        ],
        eligibility=[
            _eligibility("sec_change", "2020-01-01", "2020-12-31"),
            _eligibility("sec_reuse_old", "2020-01-01", "2020-03-31"),
            _eligibility("sec_reuse_new", "2020-06-01", "2020-12-31"),
            _eligibility("sec_ipo", "2020-06-15", "2020-12-31"),
            _eligibility("sec_delist", "2020-01-01", "2020-09-30"),
            _eligibility("sec_relist", "2020-01-01", "2020-02-01"),
            _eligibility("sec_relist", "2020-05-01", "2020-12-31"),
        ],
    )
    adapter = load_ticket62_selector_adapter(root)

    old_symbol = adapter.resolve_selector_row({"asset_id": "asset_change", "decision_timestamp": "2020-05-31"})
    new_symbol = adapter.resolve_selector_row({"asset_id": "asset_change", "decision_timestamp": "2020-06-01"})
    reuse_old = adapter.resolve_selector_row({"symbol": "RUSE", "decision_timestamp": "2020-03-01"})
    reuse_gap = adapter.resolve_selector_row({"symbol": "RUSE", "decision_timestamp": "2020-05-01"})
    reuse_new = adapter.resolve_selector_row({"symbol": "RUSE", "decision_timestamp": "2020-07-01"})

    assert old_symbol["historical_symbol"] == "OLD"
    assert new_symbol["historical_symbol"] == "NEW"
    assert old_symbol["permanent_security_id"] == new_symbol["permanent_security_id"]
    assert reuse_old["permanent_security_id"] == "sec_reuse_old"
    assert reuse_gap["identity_resolution_state"] == "UNRESOLVED"
    assert reuse_new["permanent_security_id"] == "sec_reuse_new"

    assert adapter.resolve_selector_row({"symbol": "IPO", "decision_timestamp": "2020-06-14"})["universe_eligible"] is False
    assert adapter.resolve_selector_row({"symbol": "IPO", "decision_timestamp": "2020-06-15"})["universe_eligible"] is True
    assert adapter.resolve_selector_row({"symbol": "DLST", "decision_timestamp": "2020-09-30"})["universe_eligible"] is True
    assert adapter.resolve_selector_row({"symbol": "DLST", "decision_timestamp": "2020-10-01"})["listing_state"] == "post_last_observed"
    assert adapter.resolve_selector_row({"symbol": "RLIST", "decision_timestamp": "2020-03-01"})["identity_resolution_state"] == "UNRESOLVED"
    assert adapter.resolve_selector_row({"symbol": "RLIST", "decision_timestamp": "2020-05-02"})["universe_eligible"] is True


def test_selector_adapter_handles_merger_spinoff_bankruptcy_liquidation_and_exchange_transfer(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    _write_authority_root(
        root,
        security=[
            _security("sec_merger_pred", "issuer_pred", "MPRD", "2020-01-01", "2020-12-31"),
            _security("sec_merger_succ", "issuer_succ", "MSUC", "2020-09-01", "2020-12-31"),
            _security("sec_spin_parent", "issuer_spin_parent", "SPINP", "2020-01-01", "2020-12-31"),
            _security("sec_spin_child", "issuer_spin_child", "SPINC", "2020-07-15", "2020-12-31"),
            _security("sec_bankrupt", "issuer_bankrupt", "BKPT", "2020-01-01", "2020-12-31"),
            _security("sec_liquidation", "issuer_liq", "LIQ", "2020-01-01", "2020-12-31"),
            _security("sec_xfer", "issuer_xfer", "XFER", "2020-01-01", "2020-12-31"),
        ],
        symbols=[
            _symbol("sec_merger_pred", "MPRD", "2020-01-01", "2020-12-31"),
            _symbol("sec_merger_succ", "MSUC", "2020-09-01", "2020-12-31"),
            _symbol("sec_spin_parent", "SPINP", "2020-01-01", "2020-12-31"),
            _symbol("sec_spin_child", "SPINC", "2020-07-15", "2020-12-31"),
            _symbol("sec_bankrupt", "BKPT", "2020-01-01", "2020-12-31"),
            _symbol("sec_liquidation", "LIQ", "2020-01-01", "2020-12-31"),
            _symbol("sec_xfer", "XFER", "2020-01-01", "2020-05-31", exchange="XNYS"),
            _symbol("sec_xfer", "XFER", "2020-06-01", "2020-12-31", exchange="XNAS"),
        ],
        memberships=[
            _membership(security_id, "2020-01-01", "2020-12-31")
            for security_id in ("sec_merger_pred", "sec_spin_parent", "sec_bankrupt", "sec_liquidation", "sec_xfer")
        ]
        + [
            _membership("sec_merger_succ", "2020-09-01", "2020-12-31"),
            _membership("sec_spin_child", "2020-07-15", "2020-12-31"),
        ],
        eligibility=[
            _eligibility(security_id, "2020-01-01", "2020-12-31")
            for security_id in ("sec_merger_pred", "sec_spin_parent", "sec_bankrupt", "sec_liquidation", "sec_xfer")
        ]
        + [
            _eligibility("sec_merger_succ", "2020-09-01", "2020-12-31"),
            _eligibility("sec_spin_child", "2020-07-15", "2020-12-31"),
        ],
        events=[
            _event("merger", "sec_merger_pred", "2020-09-01", "MERGED_TERMINAL", successor="sec_merger_succ"),
            _event("spinoff", "sec_spin_parent|sec_spin_child", "2020-07-15", "SPINOFF_CHILD_LISTED"),
            _event("bankruptcy", "sec_bankrupt", "2020-08-15", "BANKRUPT"),
            _event("liquidation", "sec_liquidation", "2020-11-01", "LIQUIDATED"),
            _event("exchange_transfer", "sec_xfer", "2020-06-01", "LISTED_TRANSFERRED"),
        ],
    )
    adapter = load_ticket62_selector_adapter(root)

    assert adapter.resolve_selector_row({"symbol": "MPRD", "decision_timestamp": "2020-08-31"})["universe_eligible"] is True
    assert adapter.resolve_selector_row({"symbol": "MPRD", "decision_timestamp": "2020-09-01"})["universe_eligible"] is False
    assert adapter.resolve_selector_row({"symbol": "MSUC", "decision_timestamp": "2020-09-01"})["universe_eligible"] is True
    assert adapter.resolve_selector_row({"symbol": "SPINC", "decision_timestamp": "2020-07-14"})["universe_eligible"] is False
    assert adapter.resolve_selector_row({"symbol": "SPINC", "decision_timestamp": "2020-07-15"})["universe_eligible"] is True
    assert adapter.resolve_selector_row({"symbol": "BKPT", "decision_timestamp": "2020-08-15"})["universe_eligible"] is False
    assert adapter.resolve_selector_row({"symbol": "LIQ", "decision_timestamp": "2020-11-01"})["universe_eligible"] is False
    assert adapter.resolve_selector_row({"symbol": "XFER", "decision_timestamp": "2020-05-31"})["exchange"] == "XNYS"
    assert adapter.resolve_selector_row({"symbol": "XFER", "decision_timestamp": "2020-06-01"})["exchange"] == "XNAS"


def test_selector_adapter_fails_closed_for_conflicts_knowledge_cutoff_and_current_symbol_fallback(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    _write_authority_root(
        root,
        security=[
            _security("sec_ambig_left", "issuer_left", "AMBIG", "2020-01-01", "2020-12-31"),
            _security("sec_ambig_right", "issuer_right", "AMBIG", "2020-01-01", "2020-12-31"),
            _security("sec_known_late", "issuer_late", "LATE", "2020-01-01", "2020-12-31", knowledge_time="2020-06-01T00:00:00Z"),
            _security("sec_fallback", "issuer_fallback", "FALL", "2020-01-01", "2020-12-31"),
        ],
        symbols=[
            _symbol("sec_ambig_left", "AMBIG", "2020-01-01", "2020-12-31"),
            _symbol("sec_ambig_right", "AMBIG", "2020-01-01", "2020-12-31"),
            _symbol("sec_known_late", "LATE", "2020-01-01", "2020-12-31", knowledge_time="2020-06-01T00:00:00Z"),
            _symbol("sec_fallback", "FALL", "2020-01-01", "2020-12-31"),
        ],
        memberships=[
            _membership("sec_known_late", "2020-01-01", "2020-12-31", knowledge_time="2020-06-01T00:00:00Z"),
            _membership("sec_fallback", "2020-01-01", "2020-12-31"),
        ],
        eligibility=[
            _eligibility("sec_known_late", "2020-01-01", "2020-12-31", knowledge_time="2020-06-01T00:00:00Z"),
            _eligibility("sec_fallback", "2020-01-01", "2020-12-31"),
        ],
        validation={"test_coverage": ["revision_after_decision", "unknown_knowledge_time", "conflicting_sources"]},
    )
    adapter = load_ticket62_selector_adapter(root)

    conflict = adapter.resolve_selector_row({"symbol": "AMBIG", "decision_timestamp": "2020-02-01"})
    unknown = adapter.resolve_selector_row(
        {"symbol": "LATE", "decision_timestamp": "2020-02-01"},
        knowledge_cutoff="2020-01-31T00:00:00Z",
    )
    fallback = adapter.resolve_selector_row({"symbol": "FALL", "decision_timestamp": "2020-02-01"})

    assert conflict["identity_resolution_state"] == "CONFLICTING_AUTHORITY"
    assert unknown["identity_resolution_state"] == "UNKNOWN_AT_KNOWLEDGE_CUTOFF"
    assert fallback["identity_resolution_state"] == "CURRENT_SYMBOL_FALLBACK_UNCERTIFIED"
    assert fallback["promotion_usable"] is False


def _write_tiny_sources(root: Path, market_rows_by_symbol: dict[str, list[dict]]) -> None:
    assets = []
    aliases = []
    for symbol in sorted(market_rows_by_symbol):
        asset_id = f"asset_{symbol.lower()}"
        assets.append(
            {
                "asset_id": asset_id,
                "canonical_symbol": symbol,
                "security_name": "",
                "security_type": "UNKNOWN",
                "share_class": "",
                "exchange": "",
                "currency": "USD",
                "country": "US",
                "cik": "",
                "sector": "",
                "industry": "",
                "valid_from": "1900-01-01",
                "valid_to": "",
                "is_active": "true",
                "collection_universe_514": "true",
                "registry_version": "tiny_registry_v1",
            }
        )
        aliases.append(
            {
                "asset_id": asset_id,
                "provider": "canonical",
                "provider_symbol": symbol,
                "valid_from": "1900-01-01",
                "valid_to": "",
                "is_primary": "true",
                "mapping_reason": "identity",
                "source": "tiny",
                "registry_version": "tiny_registry_v1",
            }
        )
    _write_csv(root / "data/reference/assets/canonical_asset_registry.csv", assets)
    _write_csv(root / "data/reference/assets/provider_symbol_aliases.csv", aliases)
    (root / "config/universes").mkdir(parents=True, exist_ok=True)
    (root / "config/universes/alpaca_514_symbols.txt").write_text("\n".join(sorted(market_rows_by_symbol)) + "\n", encoding="utf-8")
    for symbol, rows in market_rows_by_symbol.items():
        years: dict[str, list[dict]] = {}
        for row in rows:
            years.setdefault(row["session_date"][:4], []).append(row)
        for year, year_rows in years.items():
            _write_parquet(
                root / f"data/processed/market_data/canonical_daily_v2/full/symbol={symbol}/year={year}/bars.parquet",
                year_rows,
            )
    _write_json(
        root / "reports/data_lineage/canonical_daily_v2/build_manifest.json",
        {
            "schema_version": "canonical_daily_v2.partitioned.v1",
            "status": "COMPLETE",
            "date_min": "2020-01-02",
            "date_max": "2020-01-06",
            "dataset_logical_partition_hash": "tiny_partition_hash",
        },
    )
    _write_csv(root / "reports/data_lineage/canonical_daily_v2/eligibility_summary.csv", [{"symbol": "AAA"}])
    _write_json(
        root / "reports/data_sources/sec_edgar/submissions_bulk/run=20260723T172705Z/manifest.json",
        {"provider": "SEC EDGAR", "artifact_sha256": "sec_hash", "registered_at_utc": "2026-07-23T00:00:00Z"},
    )
    _write_json(root / "data/reference/adjusted_prices/manifest.json", {"source": "unused"})
    _write_yaml(root / "config/news_source_registry.stock_alpha_etf_funds.yaml", {"funds": {}})
    _write_yaml(root / "data/reference/universes/us_liquid_500.yaml", {"symbols": sorted(market_rows_by_symbol)})


def _write_authority_root(
    root: Path,
    *,
    security: list[dict],
    symbols: list[dict],
    memberships: list[dict],
    eligibility: list[dict],
    events: list[dict] | None = None,
    validation: dict | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_parquet(root / "security_master.parquet", security)
    _write_parquet(root / "symbol_history.parquet", symbols)
    _write_parquet(root / "universe_membership.parquet", memberships)
    _write_parquet(root / "eligibility_reconstruction.parquet", eligibility)
    _write_parquet(root / "corporate_events.parquet", events or [_event("observed_listing", "none", "1900-01-01", "OBSERVED")])
    _write_json(root / "pit_authority_validation.json", validation or {"classification": CLASSIFICATION})


def _market_row(asset_id: str, symbol: str, session_date: str, *, close: float, volume: float) -> dict:
    return {
        "asset_id": asset_id,
        "canonical_symbol": symbol,
        "session_date": session_date,
        "model_close": close,
        "raw_close": close,
        "raw_volume": volume,
        "selector_eligible": True,
        "eligibility_reason": "eligible",
        "quarantine_flag": False,
        "quarantine_reason": "",
        "source_provider": "tiny",
        "source_path": "tiny",
        "compatibility_tier": "TIER_A_NATIVE_COMPATIBLE",
    }


def _security(
    security_id: str,
    issuer_id: str,
    symbol: str,
    start: str,
    end: str,
    *,
    asset_id: str = "",
    knowledge_time: str = AUTHORITY_KNOWLEDGE_TIME,
) -> dict:
    return {
        "authority_version": AUTHORITY_VERSION,
        "permanent_issuer_id": issuer_id,
        "permanent_security_id": security_id,
        "permanent_asset_id": security_id,
        "canonical_registry_asset_id": asset_id,
        "current_symbol": symbol,
        "exchange": "",
        "effective_start": start,
        "effective_end": end,
        "status": "OBSERVED_ACTIVE",
        "knowledge_time": knowledge_time,
        "source": "fixture",
    }


def _symbol(
    security_id: str,
    symbol: str,
    start: str,
    end: str,
    *,
    exchange: str = "",
    knowledge_time: str = AUTHORITY_KNOWLEDGE_TIME,
) -> dict:
    return {
        "authority_version": AUTHORITY_VERSION,
        "permanent_security_id": security_id,
        "permanent_asset_id": security_id,
        "symbol": symbol,
        "normalized_symbol": symbol,
        "exchange": exchange,
        "alias_type": "canonical_symbol",
        "provider": "canonical",
        "effective_from": start,
        "effective_to": end,
        "knowledge_time": knowledge_time,
        "authority_status": "VERIFIED_HISTORICAL_SYMBOL_HISTORY",
    }


def _membership(security_id: str, start: str, end: str, *, knowledge_time: str = AUTHORITY_KNOWLEDGE_TIME) -> dict:
    return {
        "authority_version": AUTHORITY_VERSION,
        "universe_id": DEFAULT_UNIVERSE_ID,
        "permanent_security_id": security_id,
        "permanent_asset_id": security_id,
        "symbol_at_time": "",
        "effective_from": start,
        "effective_to": end,
        "knowledge_time": knowledge_time,
        "membership_state": "included",
        "inclusion_reason": "fixture_included",
        "exclusion_reason": "",
        "source_snapshot": "fixture",
        "source_version": AUTHORITY_VERSION,
        "unresolved_conflict_state": "RESOLVED_RULE_EVALUATION",
    }


def _eligibility(security_id: str, start: str, end: str, *, knowledge_time: str = AUTHORITY_KNOWLEDGE_TIME) -> dict:
    return {
        "authority_version": AUTHORITY_VERSION,
        "permanent_security_id": security_id,
        "permanent_asset_id": security_id,
        "effective_from": start,
        "effective_to": end,
        "knowledge_time": knowledge_time,
        "eligibility_state": "included",
        "reason_codes": "rules_passed",
        "source_snapshot": "fixture",
        "source_version": AUTHORITY_VERSION,
        "unresolved_conflict_state": "RESOLVED_RULE_EVALUATION",
    }


def _event(event_type: str, affected: str, effective: str, post_state: str, *, successor: str = "") -> dict:
    return {
        "authority_version": AUTHORITY_VERSION,
        "event_id": f"{event_type}_{affected}_{effective}".replace("|", "_"),
        "event_type": event_type,
        "affected_security_ids": affected,
        "predecessor_security_ids": affected.split("|")[0],
        "successor_security_ids": successor,
        "effective_time": effective,
        "event_time": effective,
        "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
        "first_known_time": AUTHORITY_KNOWLEDGE_TIME,
        "post_event_state": post_state,
        "source": "fixture",
        "source_version": AUTHORITY_VERSION,
        "unresolved_state": "",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_dump(payload), encoding="utf-8")


def yaml_dump(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=True)
