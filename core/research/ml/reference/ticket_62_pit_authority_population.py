from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


CLASSIFICATION = "RESEARCH_GRADE_PARTIAL_AUTHORITY"
AUTHORITY_VERSION = "ticket62_research_grade_partial_authority_v1"
DEFAULT_OUTPUT_ROOT = f"data/reference/pit_authority/version={AUTHORITY_VERSION}"
DEFAULT_UNIVERSE_ID = "ticket62_reconstructed_us_liquid_v1"
AUTHORITY_KNOWLEDGE_TIME = "2026-07-29T00:00:00Z"
PROMOTION_GRADE_USE = False

REQUIRED_ARTIFACTS = (
    "source_inventory.json",
    "source_precedence.json",
    "security_master.parquet",
    "security_master_manifest.json",
    "symbol_history.parquet",
    "symbol_history_manifest.json",
    "corporate_events.parquet",
    "corporate_events_manifest.json",
    "universe_membership.parquet",
    "universe_membership_manifest.json",
    "eligibility_rules.json",
    "eligibility_reconstruction.parquet",
    "eligibility_manifest.json",
    "identity_conflicts.csv",
    "universe_conflicts.csv",
    "coverage_by_year.csv",
    "coverage_by_security.csv",
    "coverage_summary.json",
    "pit_authority_validation.json",
    "ticket_62_summary.md",
)


@dataclass(frozen=True)
class EligibilityRuleConfig:
    rule_version: str = "ticket62_reconstructed_us_liquid_rules_v1"
    min_observed_sessions: int = 252
    min_model_close: float = 5.0
    trailing_dollar_volume_sessions: int = 20
    min_trailing_dollar_volume: float = 5_000_000.0
    max_latest_gap_days: int = 14
    decision_time_utc: str = "21:00:00"


class Ticket62SelectorAuthorityAdapter:
    """Read-only selector-facing adapter for Ticket 62 PIT authority artifacts."""

    def __init__(
        self,
        *,
        root: Path,
        security_master: Sequence[Mapping[str, Any]],
        symbol_history: Sequence[Mapping[str, Any]],
        universe_membership: Sequence[Mapping[str, Any]],
        eligibility_reconstruction: Sequence[Mapping[str, Any]],
        corporate_events: Sequence[Mapping[str, Any]],
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.security_master = tuple(sorted((dict(row) for row in security_master), key=_security_sort_key))
        self.symbol_history = tuple(sorted((dict(row) for row in symbol_history), key=_symbol_sort_key))
        self.universe_membership = tuple(sorted((dict(row) for row in universe_membership), key=_membership_sort_key))
        self.eligibility_reconstruction = tuple(
            sorted((dict(row) for row in eligibility_reconstruction), key=_eligibility_sort_key)
        )
        self.corporate_events = tuple(sorted((dict(row) for row in corporate_events), key=_event_sort_key))
        self.validation = dict(validation or {})

    @classmethod
    def from_root(cls, root: str | Path) -> "Ticket62SelectorAuthorityAdapter":
        path = Path(root)
        return cls(
            root=path,
            security_master=_read_parquet_rows(path / "security_master.parquet"),
            symbol_history=_read_parquet_rows(path / "symbol_history.parquet"),
            universe_membership=_read_parquet_rows(path / "universe_membership.parquet"),
            eligibility_reconstruction=_read_parquet_rows(path / "eligibility_reconstruction.parquet"),
            corporate_events=_read_parquet_rows(path / "corporate_events.parquet"),
            validation=_read_json(path / "pit_authority_validation.json"),
        )

    def resolve_selector_row(
        self,
        row: Mapping[str, Any],
        *,
        decision_timestamp: str | None = None,
        universe_id: str = DEFAULT_UNIVERSE_ID,
        knowledge_cutoff: str | None = None,
    ) -> dict[str, Any]:
        decision_text = (
            decision_timestamp
            or row.get("decision_timestamp")
            or row.get("rebalance_timestamp")
            or row.get("prediction_timestamp")
            or row.get("decision_session_date")
            or row.get("rebalance_date")
            or row.get("session_date")
        )
        decision_date = _date_value(decision_text)
        symbol = _normalize_symbol(row.get("canonical_symbol") or row.get("symbol") or row.get("ticker"))
        canonical_registry_asset_id = str(row.get("asset_id") or row.get("canonical_registry_asset_id") or "").strip()

        candidates = self._security_candidates(
            canonical_registry_asset_id=canonical_registry_asset_id,
            symbol=symbol,
            decision_date=decision_date,
        )
        available_candidates = [
            candidate for candidate in candidates if _record_known_at(candidate, knowledge_cutoff)
        ]
        if candidates and not available_candidates:
            return self._resolution(
                row=row,
                decision_date=decision_date,
                symbol=symbol,
                permanent_security_id="",
                permanent_issuer_id="",
                historical_symbol="",
                listing_state="unknown_at_knowledge_cutoff",
                universe_eligible=False,
                membership_state="unknown",
                eligibility_state="unknown",
                identity_resolution_state="UNKNOWN_AT_KNOWLEDGE_CUTOFF",
                conflict_unresolved_state="UNKNOWN_KNOWLEDGE_TIME",
                universe_id=universe_id,
                reason_codes=("knowledge_cutoff_precedes_identity_authority",),
            )

        if not available_candidates:
            return self._resolution(
                row=row,
                decision_date=decision_date,
                symbol=symbol,
                permanent_security_id="",
                permanent_issuer_id="",
                historical_symbol="",
                listing_state="unknown",
                universe_eligible=False,
                membership_state="unknown",
                eligibility_state="unknown",
                identity_resolution_state="UNRESOLVED",
                conflict_unresolved_state="MISSING_SECURITY_IDENTITY",
                universe_id=universe_id,
                reason_codes=("no_security_master_match",),
            )

        distinct_ids = {candidate["permanent_security_id"] for candidate in available_candidates}
        if len(distinct_ids) > 1:
            return self._resolution(
                row=row,
                decision_date=decision_date,
                symbol=symbol,
                permanent_security_id="",
                permanent_issuer_id="",
                historical_symbol=symbol,
                listing_state="conflicting",
                universe_eligible=False,
                membership_state="conflicting",
                eligibility_state="conflicting",
                identity_resolution_state="CONFLICTING_AUTHORITY",
                conflict_unresolved_state="AMBIGUOUS_SECURITY_IDENTITY",
                universe_id=universe_id,
                reason_codes=("multiple_security_master_matches",),
                candidates=available_candidates,
            )

        security = available_candidates[0]
        permanent_security_id = str(security["permanent_security_id"])
        permanent_issuer_id = str(security.get("permanent_issuer_id") or "")
        identity_state = (
            "RESOLVED_INTERNAL"
            if canonical_registry_asset_id and canonical_registry_asset_id == security.get("canonical_registry_asset_id")
            else "CURRENT_SYMBOL_FALLBACK_UNCERTIFIED"
        )
        symbols = [
            item
            for item in self.symbol_history
            if item.get("permanent_security_id") == permanent_security_id
            and _date_in_interval(decision_date, item.get("effective_from"), item.get("effective_to"))
            and _record_known_at(item, knowledge_cutoff)
        ]
        canonical_symbols = [item for item in symbols if item.get("alias_type") == "canonical_symbol"]
        symbol_record = (canonical_symbols or symbols or [security])[0]
        historical_symbol = _normalize_symbol(symbol_record.get("symbol") or security.get("current_symbol") or symbol)
        security_context = dict(security)
        if symbol_record.get("exchange"):
            security_context["exchange"] = symbol_record.get("exchange")

        membership = self._membership_at(
            permanent_security_id=permanent_security_id,
            decision_date=decision_date,
            universe_id=universe_id,
            knowledge_cutoff=knowledge_cutoff,
        )
        eligibility = self._eligibility_at(
            permanent_security_id=permanent_security_id,
            decision_date=decision_date,
            knowledge_cutoff=knowledge_cutoff,
        )
        listing_state = _listing_state(security, decision_date)
        terminal_events = [
            event
            for event in self.corporate_events
            if permanent_security_id in str(event.get("affected_security_ids") or "").split("|")
            and _date_value(event.get("effective_time")) <= decision_date
            and _record_known_at(event, knowledge_cutoff)
            and str(event.get("post_event_state") or "").upper().startswith(("DELIST", "BANKRUPT", "LIQUIDAT", "MERGED", "UNRESOLVED_TERMINAL"))
        ]
        if terminal_events and decision_date >= _date_value(terminal_events[-1]["effective_time"]):
            listing_state = "terminal_or_unresolved_after_event"

        membership_state = str(membership.get("membership_state") or "unknown") if membership else "unknown"
        eligibility_state = str(eligibility.get("eligibility_state") or "unknown") if eligibility else "unknown"
        universe_eligible = (
            membership_state == "included"
            and eligibility_state == "included"
            and listing_state == "listed_observed"
        )
        reason_codes = _split_reasons(
            (membership or {}).get("reason_codes")
            or (membership or {}).get("inclusion_reason")
            or (membership or {}).get("exclusion_reason")
            or (eligibility or {}).get("reason_codes")
            or listing_state
        )
        unresolved_state = (
            str((membership or {}).get("unresolved_conflict_state") or "")
            or str((eligibility or {}).get("unresolved_conflict_state") or "")
            or ("CURRENT_SYMBOL_FALLBACK_UNCERTIFIED" if identity_state == "CURRENT_SYMBOL_FALLBACK_UNCERTIFIED" else "RESOLVED_PARTIAL")
        )
        return self._resolution(
            row=row,
            decision_date=decision_date,
            symbol=symbol,
            permanent_security_id=permanent_security_id,
            permanent_issuer_id=permanent_issuer_id,
            historical_symbol=historical_symbol,
            listing_state=listing_state,
            universe_eligible=universe_eligible,
            membership_state=membership_state,
            eligibility_state=eligibility_state,
            identity_resolution_state=identity_state,
            conflict_unresolved_state=unresolved_state,
            universe_id=universe_id,
            reason_codes=reason_codes,
            membership=membership,
            eligibility=eligibility,
            security=security_context,
        )

    def resolve_many(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        universe_id: str = DEFAULT_UNIVERSE_ID,
        knowledge_cutoff: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.resolve_selector_row(row, universe_id=universe_id, knowledge_cutoff=knowledge_cutoff)
            for row in rows
        ]

    def _security_candidates(
        self,
        *,
        canonical_registry_asset_id: str,
        symbol: str,
        decision_date: date,
    ) -> list[Mapping[str, Any]]:
        if canonical_registry_asset_id:
            exact = [
                row
                for row in self.security_master
                if str(row.get("canonical_registry_asset_id") or "") == canonical_registry_asset_id
            ]
            if exact:
                return exact
        if symbol:
            symbol_matches = [
                row
                for row in self.symbol_history
                if _normalize_symbol(row.get("symbol")) == symbol
                and _date_in_interval(decision_date, row.get("effective_from"), row.get("effective_to"))
            ]
            ids = {row.get("permanent_security_id") for row in symbol_matches}
            if ids:
                return [row for row in self.security_master if row.get("permanent_security_id") in ids]
            current_symbol_matches = [
                row for row in self.security_master if _normalize_symbol(row.get("current_symbol")) == symbol
            ]
            expired_matches = [
                row
                for row in current_symbol_matches
                if row.get("effective_end") and decision_date > _date_value(row.get("effective_end"))
            ]
            has_future_reuse = any(
                row.get("effective_start") and decision_date < _date_value(row.get("effective_start"))
                for row in current_symbol_matches
            )
            if len(expired_matches) == 1 and not has_future_reuse:
                return expired_matches
        return []

    def _membership_at(
        self,
        *,
        permanent_security_id: str,
        decision_date: date,
        universe_id: str,
        knowledge_cutoff: str | None,
    ) -> Mapping[str, Any] | None:
        matches = [
            row
            for row in self.universe_membership
            if row.get("permanent_security_id") == permanent_security_id
            and row.get("universe_id") == universe_id
            and _date_in_interval(decision_date, row.get("effective_from"), row.get("effective_to"))
            and _record_known_at(row, knowledge_cutoff)
        ]
        return matches[-1] if matches else None

    def _eligibility_at(
        self,
        *,
        permanent_security_id: str,
        decision_date: date,
        knowledge_cutoff: str | None,
    ) -> Mapping[str, Any] | None:
        matches = [
            row
            for row in self.eligibility_reconstruction
            if row.get("permanent_security_id") == permanent_security_id
            and _date_in_interval(decision_date, row.get("effective_from"), row.get("effective_to"))
            and _record_known_at(row, knowledge_cutoff)
        ]
        return matches[-1] if matches else None

    def _resolution(
        self,
        *,
        row: Mapping[str, Any],
        decision_date: date,
        symbol: str,
        permanent_security_id: str,
        permanent_issuer_id: str,
        historical_symbol: str,
        listing_state: str,
        universe_eligible: bool,
        membership_state: str,
        eligibility_state: str,
        identity_resolution_state: str,
        conflict_unresolved_state: str,
        universe_id: str,
        reason_codes: Sequence[str],
        security: Mapping[str, Any] | None = None,
        membership: Mapping[str, Any] | None = None,
        eligibility: Mapping[str, Any] | None = None,
        candidates: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "ticket62_selector_authority_resolution.v1",
            "authority_version": AUTHORITY_VERSION,
            "classification": CLASSIFICATION,
            "production_selection_enabled": False,
            "promotion_usable": False,
            "decision_date": decision_date.isoformat(),
            "universe_id": universe_id,
            "input_asset_id": str(row.get("asset_id") or ""),
            "input_symbol": symbol,
            "permanent_issuer_id": permanent_issuer_id,
            "permanent_security_id": permanent_security_id,
            "permanent_asset_id": permanent_security_id,
            "historical_symbol": historical_symbol,
            "exchange": str((security or {}).get("exchange") or ""),
            "listing_state": listing_state,
            "universe_eligible": bool(universe_eligible),
            "membership_state": membership_state,
            "eligibility_state": eligibility_state,
            "membership_version": AUTHORITY_VERSION if membership else "",
            "identity_authority_version": AUTHORITY_VERSION if permanent_security_id else "",
            "identity_resolution_state": identity_resolution_state,
            "conflict_unresolved_state": conflict_unresolved_state,
            "reason_codes": list(reason_codes),
            "source_refs": {
                "security_source": (security or {}).get("source", ""),
                "membership_source": (membership or {}).get("source_snapshot", ""),
                "eligibility_source": (eligibility or {}).get("source_snapshot", ""),
            },
            "candidate_security_ids": sorted(str(candidate.get("permanent_security_id") or "") for candidate in candidates),
        }
        payload["resolution_id"] = "ticket62_resolution_" + _sha256_json(payload)[:24]
        return payload


def load_ticket62_selector_adapter(root: str | Path) -> Ticket62SelectorAuthorityAdapter:
    return Ticket62SelectorAuthorityAdapter.from_root(root)


def build_ticket_62_pit_authority(
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    registry_path: str | Path = "data/reference/assets/canonical_asset_registry.csv",
    alias_path: str | Path = "data/reference/assets/provider_symbol_aliases.csv",
    universe_path: str | Path = "config/universes/alpaca_514_symbols.txt",
    canonical_daily_root: str | Path = "data/processed/market_data/canonical_daily_v2/full",
    canonical_daily_manifest_path: str | Path = "reports/data_lineage/canonical_daily_v2/build_manifest.json",
    canonical_daily_eligibility_summary_path: str | Path = "reports/data_lineage/canonical_daily_v2/eligibility_summary.csv",
    sec_manifest_path: str | Path = "reports/data_sources/sec_edgar/submissions_bulk/run=20260723T172705Z/manifest.json",
    etf_fund_registry_path: str | Path = "config/news_source_registry.stock_alpha_etf_funds.yaml",
    current_liquid_universe_path: str | Path = "data/reference/universes/us_liquid_500.yaml",
    adjusted_price_manifest_path: str | Path = "data/reference/adjusted_prices/manifest.json",
    rules: EligibilityRuleConfig | Mapping[str, Any] | None = None,
    limit_symbols: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root)
    output = repo / output_root
    rule_config = _rule_config(rules)
    paths = {
        "registry": repo / registry_path,
        "aliases": repo / alias_path,
        "universe": repo / universe_path,
        "canonical_daily_root": repo / canonical_daily_root,
        "canonical_daily_manifest": repo / canonical_daily_manifest_path,
        "canonical_daily_eligibility_summary": repo / canonical_daily_eligibility_summary_path,
        "sec_manifest": repo / sec_manifest_path,
        "etf_fund_registry": repo / etf_fund_registry_path,
        "current_liquid_universe": repo / current_liquid_universe_path,
        "adjusted_price_manifest": repo / adjusted_price_manifest_path,
    }

    assets = _read_csv_rows(paths["registry"])
    if limit_symbols is not None:
        assets = assets[: int(limit_symbols)]
    aliases = _read_csv_rows(paths["aliases"])
    asset_ids = {row["asset_id"] for row in assets}
    aliases = [row for row in aliases if row.get("asset_id") in asset_ids]
    alias_by_asset: dict[str, list[dict[str, str]]] = {}
    for alias in aliases:
        alias_by_asset.setdefault(alias["asset_id"], []).append(alias)
    symbols = [_normalize_symbol(row["canonical_symbol"]) for row in assets]
    etf_funds = _read_etf_fund_symbols(paths["etf_fund_registry"])
    current_liquid = _read_yaml_symbols(paths["current_liquid_universe"])
    source_hashes = _source_hashes(paths)
    source_inventory = _source_inventory(paths, source_hashes)
    source_precedence = _source_precedence()
    canonical_daily_manifest = _read_json(paths["canonical_daily_manifest"])
    dataset_max = _date_value(canonical_daily_manifest.get("date_max") or AUTHORITY_KNOWLEDGE_TIME[:10])
    source_snapshot = str(canonical_daily_manifest.get("dataset_logical_partition_hash") or source_hashes.get("canonical_daily_manifest") or "")

    security_master: list[dict[str, Any]] = []
    symbol_history: list[dict[str, Any]] = []
    corporate_events: list[dict[str, Any]] = []
    eligibility_intervals: list[dict[str, Any]] = []
    universe_membership: list[dict[str, Any]] = []
    coverage_by_security: list[dict[str, Any]] = []
    coverage_years: dict[int, dict[str, Any]] = {}
    identity_conflicts: list[dict[str, Any]] = []
    universe_conflicts: list[dict[str, Any]] = []

    for asset in sorted(assets, key=lambda row: (_normalize_symbol(row.get("canonical_symbol")), row.get("asset_id", ""))):
        symbol = _normalize_symbol(asset["canonical_symbol"])
        market_rows = _read_market_rows(paths["canonical_daily_root"], symbol)
        if not market_rows:
            identity_conflicts.append(
                _conflict_row(
                    symbol=symbol,
                    permanent_security_id="",
                    conflict_type="missing_market_observation",
                    state="UNRESOLVED",
                    detail="No canonical daily v2 rows were found for the registry asset.",
                    source="canonical_daily_v2",
                )
            )
            continue

        first_observed = _date_value(market_rows[0]["session_date"])
        last_observed = _date_value(market_rows[-1]["session_date"])
        provider_set = sorted({str(row.get("source_provider") or "") for row in market_rows if row.get("source_provider")})
        alias_hash = _sha256_json(alias_by_asset.get(asset["asset_id"], []))
        permanent_security_id = _internal_id(
            "t62_sec",
            {
                "canonical_registry_asset_id": asset["asset_id"],
                "registry_version": asset.get("registry_version", ""),
                "symbol": symbol,
                "alias_hash": alias_hash,
                "identity_scope": "ticket62_internal_current_registry_asset",
            },
        )
        permanent_issuer_id = _internal_id(
            "t62_issuer",
            {
                "canonical_registry_asset_id": asset["asset_id"],
                "registry_version": asset.get("registry_version", ""),
                "symbol": symbol,
                "cik": asset.get("cik") or "",
                "identity_scope": "ticket62_internal_current_registry_issuer",
            },
        )
        etf_flag = symbol in etf_funds
        security_type = "ETF_OR_FUND_CANDIDATE" if etf_flag else str(asset.get("security_type") or "UNKNOWN")
        latest_gap = max((dataset_max - last_observed).days, 0)
        status = "OBSERVED_ACTIVE_THROUGH_DATASET_MAX" if latest_gap <= rule_config.max_latest_gap_days else "UNRESOLVED_STALE_OBSERVATION"
        security_master.append(
            {
                "authority_version": AUTHORITY_VERSION,
                "permanent_issuer_id": permanent_issuer_id,
                "permanent_security_id": permanent_security_id,
                "permanent_asset_id": permanent_security_id,
                "canonical_registry_asset_id": asset["asset_id"],
                "share_class_relationship": asset.get("share_class") or "",
                "cik": asset.get("cik") or "",
                "current_symbol": symbol,
                "historical_symbol": symbol,
                "exchange": asset.get("exchange") or "",
                "security_type": security_type,
                "etf_fund_flag": etf_flag,
                "effective_start": first_observed.isoformat(),
                "effective_end": last_observed.isoformat(),
                "listing_date_observed": first_observed.isoformat(),
                "delisting_date_observed": "" if latest_gap <= rule_config.max_latest_gap_days else last_observed.isoformat(),
                "status": status,
                "identity_status": "internal_reconstructed",
                "event_time": first_observed.isoformat(),
                "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
                "knowledge_time_status": "authority_build_time_not_provider_publication_time",
                "source": "canonical_asset_registry+canonical_daily_v2",
                "source_version": source_snapshot,
                "confidence": "MEDIUM" if latest_gap <= rule_config.max_latest_gap_days else "LOW",
                "identity_confidence": "MEDIUM" if latest_gap <= rule_config.max_latest_gap_days else "LOW",
                "authority_confidence": "MEDIUM" if latest_gap <= rule_config.max_latest_gap_days else "LOW",
                "authority_status": "INTERNAL_IDENTITY_RECONSTRUCTED_FROM_CURRENT_REGISTRY_AND_OBSERVED_MARKET_DATA",
                "external_identity_corroborated": False,
                "cik_verification_status": "missing_unresolved"
                if not asset.get("cik")
                else "registry_cik_unverified_by_ticket62",
                "current_symbol_identity_status": "static_symbol_fallback_uncertified",
                "internal_identity_derivation": "hash(canonical_registry_asset_id, registry_version, alias_hash, identity_scope)",
                "limitations": "Not an external permanent identifier; current-symbol lineage remains uncertified for historical symbol changes.",
                "promotion_usable": False,
            }
        )
        symbol_history.extend(
            _symbol_history_rows(
                asset=asset,
                permanent_issuer_id=permanent_issuer_id,
                permanent_security_id=permanent_security_id,
                aliases=alias_by_asset.get(asset["asset_id"], []),
                first_observed=first_observed,
                last_observed=last_observed,
                source_snapshot=source_snapshot,
            )
        )
        corporate_events.append(
            _corporate_event_row(
                event_type="observed_listing",
                symbol=symbol,
                permanent_security_id=permanent_security_id,
                effective_date=first_observed,
                source="canonical_daily_v2_first_observed_session",
                source_version=source_snapshot,
                post_event_state="OBSERVED_LISTED",
                unresolved_fields=("official_ipo_date", "exchange_listing_notice", "provider_publication_time"),
            )
        )
        if latest_gap > rule_config.max_latest_gap_days:
            corporate_events.append(
                _corporate_event_row(
                    event_type="stale_observation_terminal_gap",
                    symbol=symbol,
                    permanent_security_id=permanent_security_id,
                    effective_date=last_observed + timedelta(days=1),
                    source="canonical_daily_v2_last_observed_session",
                    source_version=source_snapshot,
                    post_event_state="UNRESOLVED_TERMINAL_OR_SYMBOL_CHANGE_GAP",
                    unresolved_fields=("delisting_date", "symbol_change", "merger_or_acquisition", "exchange_transfer"),
                )
            )
            identity_conflicts.append(
                _conflict_row(
                    symbol=symbol,
                    permanent_security_id=permanent_security_id,
                    conflict_type="stale_observation_requires_identity_review",
                    state="UNRESOLVED",
                    detail=f"Last observed {last_observed.isoformat()} is {latest_gap} days before dataset max {dataset_max.isoformat()}.",
                    source="canonical_daily_v2",
                )
            )
        if not asset.get("cik"):
            identity_conflicts.append(
                _conflict_row(
                    symbol=symbol,
                    permanent_security_id=permanent_security_id,
                    conflict_type="missing_cik",
                    state="UNRESOLVED",
                    detail="SEC bulk source is audited locally but no extracted ticker-to-CIK mapping is available in the registry.",
                    source="canonical_asset_registry",
                )
            )
        identity_conflicts.append(
            _conflict_row(
                symbol=symbol,
                permanent_security_id=permanent_security_id,
                conflict_type="current_symbol_fallback_uncertified",
                state="UNRESOLVED",
                detail="Historical symbol changes and ticker reuse were not proven by bounded local sources for this asset.",
                source="canonical_asset_registry",
            )
        )

        intervals, stats = _eligibility_intervals(
            permanent_security_id=permanent_security_id,
            symbol=symbol,
            rows=market_rows,
            rules=rule_config,
            source_snapshot=source_snapshot,
        )
        eligibility_intervals.extend(intervals)
        for interval in intervals:
            universe_membership.append(
                {
                    "authority_version": AUTHORITY_VERSION,
                    "universe_id": DEFAULT_UNIVERSE_ID,
                    "permanent_security_id": permanent_security_id,
                    "permanent_asset_id": permanent_security_id,
                    "symbol_at_time": symbol,
                    "effective_from": interval["effective_from"],
                    "effective_to": interval["effective_to"],
                    "event_time": interval["effective_from"],
                    "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
                    "membership_state": interval["eligibility_state"],
                    "inclusion_reason": interval["reason_codes"] if interval["eligibility_state"] == "included" else "",
                    "exclusion_reason": interval["reason_codes"] if interval["eligibility_state"] != "included" else "",
                    "source_snapshot": source_snapshot,
                    "source_version": AUTHORITY_VERSION,
                    "unresolved_conflict_state": interval["unresolved_conflict_state"],
                    "membership_classification": "rules_based_pit_observable_reconstruction",
                    "historical_observed_membership": False,
                    "historically_reconstructed_membership": True,
                    "current_static_membership": symbol in current_liquid,
                    "eligibility_computed_from_pit_market_data": True,
                    "promotion_usable": False,
                }
            )
        if symbol not in current_liquid:
            universe_conflicts.append(
                _universe_conflict_row(
                    symbol=symbol,
                    permanent_security_id=permanent_security_id,
                    conflict_type="not_in_current_us_liquid_500_static_snapshot",
                    state="RECONSTRUCTED_ONLY",
                    detail="The rules-based reconstruction may include historical intervals even when the symbol is absent from the current static liquid universe file.",
                )
            )
        coverage_by_security.append(
            {
                "permanent_security_id": permanent_security_id,
                "canonical_registry_asset_id": asset["asset_id"],
                "symbol": symbol,
                "first_observed_date": first_observed.isoformat(),
                "last_observed_date": last_observed.isoformat(),
                "observed_row_count": stats["observed_row_count"],
                "included_session_count": stats["included_session_count"],
                "excluded_session_count": stats["excluded_session_count"],
                "missing_cik": not bool(asset.get("cik")),
                "etf_fund_flag": etf_flag,
                "security_type": security_type,
                "provider_count": len(provider_set),
                "providers": "|".join(provider_set),
                "status": status,
                "identity_status": "internal_reconstructed",
                "authority_confidence": "MEDIUM" if latest_gap <= rule_config.max_latest_gap_days else "LOW",
                "external_identity_corroborated": False,
                "cik_verified": False,
                "exchange_evidenced": bool(asset.get("exchange")),
                "official_ipo_date_evidenced": False,
                "official_delisting_date_evidenced": False,
                "observed_listing_bound": True,
                "observed_delisting_bound": bool(latest_gap > rule_config.max_latest_gap_days),
                "effective_symbol_history_evidenced": False,
                "reconstructed_eligibility_evidenced": bool(intervals),
                "current_symbol_fallback_uncertified": True,
                "unresolved_identity_state_count": 2 + (1 if status == "UNRESOLVED_STALE_OBSERVATION" else 0),
                "universe_state": "rules_reconstructed",
            }
        )
        _accumulate_year_coverage(coverage_years, market_rows, intervals)

    _sort_all(
        security_master,
        symbol_history,
        corporate_events,
        eligibility_intervals,
        universe_membership,
        coverage_by_security,
        identity_conflicts,
        universe_conflicts,
    )
    coverage_by_year = [_finalize_year_coverage(coverage_years[year]) for year in sorted(coverage_years)]
    coverage_summary = _coverage_summary(
        security_master=security_master,
        symbol_history=symbol_history,
        corporate_events=corporate_events,
        universe_membership=universe_membership,
        eligibility_intervals=eligibility_intervals,
        identity_conflicts=identity_conflicts,
        universe_conflicts=universe_conflicts,
        coverage_by_security=coverage_by_security,
        coverage_by_year=coverage_by_year,
        source_inventory=source_inventory,
    )

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "source_inventory.json", {"schema_version": "ticket62_source_inventory.v1", "sources": source_inventory})
    _write_json(output / "source_precedence.json", source_precedence)
    _write_parquet(output / "security_master.parquet", security_master)
    _write_parquet(output / "symbol_history.parquet", symbol_history)
    _write_parquet(output / "corporate_events.parquet", corporate_events)
    _write_parquet(output / "universe_membership.parquet", universe_membership)
    _write_json(output / "eligibility_rules.json", _eligibility_rules_payload(rule_config, source_hashes))
    _write_parquet(output / "eligibility_reconstruction.parquet", eligibility_intervals)
    _write_csv(output / "identity_conflicts.csv", identity_conflicts, _identity_conflict_fields())
    _write_csv(output / "universe_conflicts.csv", universe_conflicts, _universe_conflict_fields())
    _write_csv(output / "coverage_by_year.csv", coverage_by_year, _coverage_year_fields())
    _write_csv(output / "coverage_by_security.csv", coverage_by_security, _coverage_security_fields())
    _write_json(output / "coverage_summary.json", coverage_summary)

    dataset_manifests = {
        "security_master_manifest.json": _dataset_manifest(
            dataset_name="security_master",
            row_count=len(security_master),
            entity_count=len(security_master),
            date_range=_date_range(coverage_by_security, "first_observed_date", "last_observed_date"),
            source_hashes=source_hashes,
            output_path=output / "security_master.parquet",
            unresolved_count=len(identity_conflicts),
            conflict_count=sum(1 for row in identity_conflicts if row["state"] == "CONFLICTING"),
        ),
        "symbol_history_manifest.json": _dataset_manifest(
            dataset_name="symbol_history",
            row_count=len(symbol_history),
            entity_count=len({row["permanent_security_id"] for row in symbol_history}),
            date_range=_date_range(symbol_history, "effective_from", "effective_to"),
            source_hashes=source_hashes,
            output_path=output / "symbol_history.parquet",
            unresolved_count=sum(1 for row in symbol_history if row["authority_status"] != "VERIFIED_HISTORICAL_SYMBOL_HISTORY"),
            conflict_count=0,
        ),
        "corporate_events_manifest.json": _dataset_manifest(
            dataset_name="corporate_events",
            row_count=len(corporate_events),
            entity_count=len({row["event_id"] for row in corporate_events}),
            date_range=_date_range(corporate_events, "effective_time", "effective_time"),
            source_hashes=source_hashes,
            output_path=output / "corporate_events.parquet",
            unresolved_count=sum(1 for row in corporate_events if row["unresolved_state"]),
            conflict_count=0,
        ),
        "universe_membership_manifest.json": _dataset_manifest(
            dataset_name="universe_membership",
            row_count=len(universe_membership),
            entity_count=len({row["permanent_security_id"] for row in universe_membership}),
            date_range=_date_range(universe_membership, "effective_from", "effective_to"),
            source_hashes=source_hashes,
            output_path=output / "universe_membership.parquet",
            unresolved_count=len(universe_conflicts),
            conflict_count=sum(1 for row in universe_conflicts if row["state"] == "CONFLICTING"),
        ),
        "eligibility_manifest.json": _dataset_manifest(
            dataset_name="eligibility_reconstruction",
            row_count=len(eligibility_intervals),
            entity_count=len({row["permanent_security_id"] for row in eligibility_intervals}),
            date_range=_date_range(eligibility_intervals, "effective_from", "effective_to"),
            source_hashes=source_hashes,
            output_path=output / "eligibility_reconstruction.parquet",
            unresolved_count=sum(1 for row in eligibility_intervals if row["unresolved_conflict_state"] != "RESOLVED_RULE_EVALUATION"),
            conflict_count=0,
        ),
    }
    for name, manifest in dataset_manifests.items():
        _write_json(output / name, manifest)

    artifact_hashes = {
        name: file_sha256(output / name)
        for name in REQUIRED_ARTIFACTS
        if name != "pit_authority_validation.json" and (output / name).exists()
    }
    validation = _validation_payload(
        output=output,
        artifact_hashes=artifact_hashes,
        coverage_summary=coverage_summary,
        dataset_manifests=dataset_manifests,
        source_inventory=source_inventory,
    )
    validation["artifact_hash_policy"] = {
        "pit_authority_validation.json": "self-referential final hash excluded; validate by required artifact presence and the non-self artifact hashes"
    }
    validation["missing_required_artifacts"] = [
        name
        for name in REQUIRED_ARTIFACTS
        if name not in {"pit_authority_validation.json", "ticket_62_summary.md"} and not (output / name).exists()
    ]
    _write_text(output / "ticket_62_summary.md", _summary_markdown(validation, coverage_summary, source_inventory))
    artifact_hashes["ticket_62_summary.md"] = file_sha256(output / "ticket_62_summary.md")

    validation = {**validation, "artifact_hashes": artifact_hashes}
    validation["missing_required_artifacts"] = [
        name for name in REQUIRED_ARTIFACTS if name != "pit_authority_validation.json" and not (output / name).exists()
    ]
    _write_json(output / "pit_authority_validation.json", validation)
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Ticket 62 bounded real PIT authority artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--limit-symbols", type=int, default=None)
    args = parser.parse_args(argv)
    result = build_ticket_62_pit_authority(
        repo_root=args.repo_root,
        output_root=args.output_root,
        limit_symbols=args.limit_symbols,
    )
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "output_root": str(args.output_root),
                "required_artifact_count": len(result["required_artifacts"]),
                "missing_required_artifacts": result["missing_required_artifacts"],
                "promotion_usable": result["promotion_usable"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["missing_required_artifacts"] else 1


def _rule_config(value: EligibilityRuleConfig | Mapping[str, Any] | None) -> EligibilityRuleConfig:
    if value is None:
        return EligibilityRuleConfig()
    if isinstance(value, EligibilityRuleConfig):
        return value
    return EligibilityRuleConfig(**dict(value))


def _read_market_rows(root: Path, symbol: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    symbol_root = root / f"symbol={symbol}"
    files = sorted(symbol_root.glob("year=*/bars.parquet"))
    rows: list[dict[str, Any]] = []
    columns = [
        "asset_id",
        "canonical_symbol",
        "session_date",
        "model_close",
        "raw_close",
        "raw_volume",
        "selector_eligible",
        "eligibility_reason",
        "quarantine_flag",
        "quarantine_reason",
        "source_provider",
        "source_path",
        "compatibility_tier",
    ]
    for path in files:
        table = pq.read_table(path, columns=[column for column in columns if column in pq.read_schema(path).names])
        for row in table.to_pylist():
            row["canonical_symbol"] = _normalize_symbol(row.get("canonical_symbol") or symbol)
            rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("session_date") or ""))


def _eligibility_intervals(
    *,
    permanent_security_id: str,
    symbol: str,
    rows: Sequence[Mapping[str, Any]],
    rules: EligibilityRuleConfig,
    source_snapshot: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    intervals: list[dict[str, Any]] = []
    window: deque[float] = deque(maxlen=rules.trailing_dollar_volume_sessions)
    current: dict[str, Any] | None = None
    observed = 0
    included_count = 0
    excluded_count = 0
    previous_date: date | None = None

    for row in rows:
        observed += 1
        session_date = _date_value(row.get("session_date"))
        price = _float(row.get("model_close") if row.get("model_close") is not None else row.get("raw_close"))
        volume = _float(row.get("raw_volume"))
        dollar_volume = price * volume if price is not None and volume is not None else None
        if dollar_volume is not None:
            window.append(float(dollar_volume))
        trailing_dollar_volume = sum(window) / len(window) if window else None
        source_selector = _bool(row.get("selector_eligible"), default=True)
        quarantine = _bool(row.get("quarantine_flag"), default=False)
        reasons: list[str] = []
        if observed < rules.min_observed_sessions:
            reasons.append(f"seasoning_lt_{rules.min_observed_sessions}_sessions")
        if price is None:
            reasons.append("missing_price")
        elif price < rules.min_model_close:
            reasons.append(f"price_lt_{rules.min_model_close:g}")
        if trailing_dollar_volume is None:
            reasons.append("missing_trailing_dollar_volume")
        elif trailing_dollar_volume < rules.min_trailing_dollar_volume:
            reasons.append(f"trailing_dollar_volume_lt_{rules.min_trailing_dollar_volume:g}")
        if not source_selector:
            reasons.append(str(row.get("eligibility_reason") or "source_selector_ineligible"))
        if quarantine:
            reasons.append(str(row.get("quarantine_reason") or "source_quarantine"))
        state = "included" if not reasons else "excluded"
        if state == "included":
            included_count += 1
            reasons = ["rules_passed"]
        else:
            excluded_count += 1
        reason_text = "|".join(sorted(set(reasons)))
        gap_break = previous_date is not None and (session_date - previous_date).days > 7
        key = (state, reason_text)
        if current is None or current["_key"] != key or gap_break:
            if current is not None:
                intervals.append(_finalize_interval(current))
            current = {
                "_key": key,
                "authority_version": AUTHORITY_VERSION,
                "universe_id": DEFAULT_UNIVERSE_ID,
                "permanent_security_id": permanent_security_id,
                "permanent_asset_id": permanent_security_id,
                "symbol": symbol,
                "effective_from": session_date.isoformat(),
                "effective_to": session_date.isoformat(),
                "decision_start_timestamp": _decision_timestamp(session_date, rules),
                "decision_end_timestamp": _decision_timestamp(session_date, rules),
                "feature_data_cutoff": session_date.isoformat(),
                "rule_version": rules.rule_version,
                "observation_window": f"trailing_{rules.trailing_dollar_volume_sessions}_sessions",
                "eligibility_state": state,
                "included": state == "included",
                "reason_codes": reason_text,
                "source_snapshot": source_snapshot,
                "source_version": AUTHORITY_VERSION,
                "event_time": session_date.isoformat(),
                "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
                "earliest_permitted_use": AUTHORITY_KNOWLEDGE_TIME,
                "knowledge_time_status": "authority_build_time_not_provider_publication_time",
                "unresolved_conflict_state": "RESOLVED_RULE_EVALUATION",
                "promotion_usable": False,
                "min_observed_sessions": rules.min_observed_sessions,
                "min_model_close": rules.min_model_close,
                "min_trailing_dollar_volume": rules.min_trailing_dollar_volume,
                "last_trailing_dollar_volume": trailing_dollar_volume,
                "source_provider": str(row.get("source_provider") or ""),
            }
        else:
            current["effective_to"] = session_date.isoformat()
            current["decision_end_timestamp"] = _decision_timestamp(session_date, rules)
            current["feature_data_cutoff"] = session_date.isoformat()
            current["last_trailing_dollar_volume"] = trailing_dollar_volume
        previous_date = session_date
    if current is not None:
        intervals.append(_finalize_interval(current))
    return intervals, {
        "observed_row_count": len(rows),
        "included_session_count": included_count,
        "excluded_session_count": excluded_count,
    }


def _finalize_interval(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("_key", None)
    result["eligibility_interval_id"] = "ticket62_elig_" + _sha256_json(
        {
            "permanent_security_id": result["permanent_security_id"],
            "effective_from": result["effective_from"],
            "effective_to": result["effective_to"],
            "eligibility_state": result["eligibility_state"],
            "reason_codes": result["reason_codes"],
            "rule_version": result["rule_version"],
        }
    )[:24]
    return result


def _symbol_history_rows(
    *,
    asset: Mapping[str, Any],
    permanent_issuer_id: str,
    permanent_security_id: str,
    aliases: Sequence[Mapping[str, Any]],
    first_observed: date,
    last_observed: date,
    source_snapshot: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "authority_version": AUTHORITY_VERSION,
            "permanent_issuer_id": permanent_issuer_id,
            "permanent_security_id": permanent_security_id,
            "permanent_asset_id": permanent_security_id,
            "canonical_registry_asset_id": asset.get("asset_id", ""),
            "symbol": _normalize_symbol(asset.get("canonical_symbol")),
            "normalized_symbol": _normalize_symbol(asset.get("canonical_symbol")),
            "exchange": asset.get("exchange") or "",
            "alias_type": "canonical_symbol",
            "provider": "canonical",
            "previous_symbol": "",
            "successor_symbol": "",
            "reuse_relationship": "not_observed_in_bounded_sources",
            "effective_from": first_observed.isoformat(),
            "effective_to": last_observed.isoformat(),
            "event_time": first_observed.isoformat(),
            "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
            "source": "canonical_asset_registry+canonical_daily_v2",
            "source_version": source_snapshot,
            "confidence": "LOW",
            "authority_status": "CURRENT_SYMBOL_FALLBACK_UNCERTIFIED",
            "reason_code": "CURRENT_REGISTRY_SYMBOL_WITH_OBSERVED_MARKET_DATA_BOUNDS",
            "promotion_usable": False,
        }
    ]
    for alias in sorted(aliases, key=lambda row: (row.get("provider", ""), row.get("provider_symbol", ""))):
        rows.append(
            {
                "authority_version": AUTHORITY_VERSION,
                "permanent_issuer_id": permanent_issuer_id,
                "permanent_security_id": permanent_security_id,
                "permanent_asset_id": permanent_security_id,
                "canonical_registry_asset_id": asset.get("asset_id", ""),
                "symbol": _normalize_symbol(alias.get("provider_symbol")),
                "normalized_symbol": _normalize_symbol(alias.get("provider_symbol")),
                "exchange": asset.get("exchange") or "",
                "alias_type": "provider_alias",
                "provider": str(alias.get("provider") or ""),
                "previous_symbol": "",
                "successor_symbol": "",
                "reuse_relationship": "not_observed_in_bounded_sources",
                "effective_from": first_observed.isoformat(),
                "effective_to": last_observed.isoformat(),
                "event_time": first_observed.isoformat(),
                "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
                "source": str(alias.get("source") or "provider_symbol_aliases"),
                "source_version": str(alias.get("registry_version") or source_snapshot),
                "confidence": "LOW",
                "authority_status": "CURRENT_PROVIDER_ALIAS_FALLBACK_UNCERTIFIED",
                "reason_code": str(alias.get("mapping_reason") or "CURRENT_PROVIDER_ALIAS"),
                "promotion_usable": False,
            }
        )
    return rows


def _corporate_event_row(
    *,
    event_type: str,
    symbol: str,
    permanent_security_id: str,
    effective_date: date,
    source: str,
    source_version: str,
    post_event_state: str,
    unresolved_fields: Sequence[str],
) -> dict[str, Any]:
    event_id = "ticket62_event_" + _sha256_json(
        {
            "event_type": event_type,
            "symbol": symbol,
            "permanent_security_id": permanent_security_id,
            "effective_date": effective_date.isoformat(),
        }
    )[:24]
    return {
        "authority_version": AUTHORITY_VERSION,
        "event_id": event_id,
        "event_type": event_type,
        "affected_security_ids": permanent_security_id,
        "predecessor_security_ids": "",
        "successor_security_ids": "",
        "announcement_time": "",
        "event_time": effective_date.isoformat(),
        "effective_time": effective_date.isoformat(),
        "first_known_time": AUTHORITY_KNOWLEDGE_TIME,
        "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
        "source": source,
        "source_version": source_version,
        "correction_revision_lineage": "",
        "unresolved_state": "|".join(unresolved_fields),
        "post_event_state": post_event_state,
        "confidence": "LOW" if unresolved_fields else "MEDIUM",
        "promotion_usable": False,
    }


def _source_inventory(paths: Mapping[str, Path], hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    sec_manifest = _read_json(paths["sec_manifest"])
    canonical_daily_manifest = _read_json(paths["canonical_daily_manifest"])
    adjusted_manifest = _read_json(paths["adjusted_price_manifest"])
    sources = [
        {
            "source_name": "sec_edgar_submissions_bulk_manifest",
            "provider": "SEC EDGAR",
            "local_path": str(paths["sec_manifest"]),
            "file_api_path": str(paths["sec_manifest"]),
            "source_type": "official_primary_source_manifest_for_bulk_submissions_zip",
            "coverage": "SEC submissions bulk snapshot through local source file write time",
            "coverage_period": "SEC submissions bulk snapshot through local source file write time",
            "event_timestamp_support": "filing acceptance timestamps inside submissions.zip, not extracted by this ticket",
            "update_cadence": "SEC-published bulk data snapshot; local manual browser download",
            "event_timestamp": "filing acceptance timestamps inside submissions.zip, not extracted by this ticket",
            "knowledge_time_support": "local retrieval timestamp only at manifest level; per-security first-seen unavailable",
            "first_seen_or_retrieval_timestamp": sec_manifest.get("registered_at_utc") or "",
            "revision_behaviour": "SEC submissions may revise via amended filings and later bulk snapshots",
            "licence_usage_status": "public SEC data acceptable for internal research; raw archive not redistributed by generated outputs",
            "license_usage_status": "public SEC data acceptable for internal research; raw archive not redistributed by generated outputs",
            "licensing_usage_status": "public SEC data acceptable for internal research; raw archive not redistributed by generated outputs",
            "redistribution_restrictions": "store hashes and local references only; respect SEC fair-access and attribution expectations",
            "supported_authority_domains": ["SEC identity", "CIK evidence", "filings"],
            "authoritative_domains": ["SEC identity", "CIK evidence", "filings"],
            "known_gaps": [
                "zip is not extracted into a ticker-to-CIK index in this workspace",
                "provider publication and first-seen times are not populated per security",
            ],
            "trust_tier": "primary_not_indexed",
            "authority_population_usable": False,
            "authority_population_usage_reason": "audited as primary source foundation but unusable for per-security population until ticker-to-CIK extraction is available",
            "source_hash": sec_manifest.get("artifact_sha256") or hashes.get("sec_manifest", ""),
        },
        {
            "source_name": "canonical_asset_registry",
            "provider": "internal_canonical_asset_registry",
            "local_path": str(paths["registry"]),
            "file_api_path": str(paths["registry"]),
            "source_type": "current_static_registry_csv",
            "coverage": "current 514-symbol collection registry",
            "coverage_period": "current 514-symbol collection registry",
            "update_cadence": "manual internal publication",
            "event_timestamp_support": "valid_from fields default to 1900 and are not historical event authority",
            "event_timestamp": "valid_from fields default to 1900 and are not historical event authority",
            "knowledge_time_support": "registry manifest created_at when available; otherwise unknown",
            "first_seen_or_retrieval_timestamp": "registry manifest created_at when available",
            "revision_behaviour": "content-hashed internal file",
            "licence_usage_status": "internal project data",
            "license_usage_status": "internal project data",
            "licensing_usage_status": "internal project data",
            "redistribution_restrictions": "none beyond project workspace",
            "supported_authority_domains": ["current collection symbol list", "provider alias lineage"],
            "authoritative_domains": ["current collection symbol list", "provider alias lineage"],
            "known_gaps": [
                "CIKs missing for all 514 assets",
                "current ticker strings cannot certify historical identity or membership",
            ],
            "trust_tier": "secondary_current_lineage_only",
            "authority_population_usable": True,
            "authority_population_usage_reason": "usable only for internal current-scope identity seeds when labeled uncertified",
            "source_hash": hashes.get("registry", ""),
        },
        {
            "source_name": "provider_symbol_aliases",
            "provider": "internal_provider_symbol_aliases",
            "local_path": str(paths["aliases"]),
            "file_api_path": str(paths["aliases"]),
            "source_type": "current_static_provider_alias_csv",
            "coverage": "current provider alias mapping for the 514-symbol registry",
            "coverage_period": "current provider alias mapping for the 514-symbol registry",
            "update_cadence": "manual internal publication",
            "event_timestamp_support": "valid_from fields default to 1900 and are not historical event authority",
            "event_timestamp": "valid_from fields default to 1900 and are not historical event authority",
            "knowledge_time_support": "registry manifest created_at when available; otherwise unknown",
            "first_seen_or_retrieval_timestamp": "registry manifest created_at when available",
            "revision_behaviour": "content-hashed internal file",
            "licence_usage_status": "internal project data",
            "license_usage_status": "internal project data",
            "licensing_usage_status": "internal project data",
            "redistribution_restrictions": "none beyond project workspace",
            "supported_authority_domains": ["provider symbol normalization"],
            "authoritative_domains": ["provider symbol normalization"],
            "known_gaps": ["does not prove ticker reuse, symbol changes, or delisting history"],
            "trust_tier": "secondary_current_lineage_only",
            "authority_population_usable": True,
            "authority_population_usage_reason": "usable only as uncertified current provider alias evidence",
            "source_hash": hashes.get("aliases", ""),
        },
        {
            "source_name": "canonical_daily_v2",
            "provider": "canonical_daily_v2",
            "local_path": str(paths["canonical_daily_root"]),
            "file_api_path": str(paths["canonical_daily_root"]),
            "source_type": "partitioned historical daily market observations",
            "coverage": f"{canonical_daily_manifest.get('date_min', '')} to {canonical_daily_manifest.get('date_max', '')}",
            "coverage_period": f"{canonical_daily_manifest.get('date_min', '')} to {canonical_daily_manifest.get('date_max', '')}",
            "update_cadence": "backfill/build pipeline snapshot",
            "event_timestamp_support": "session_date",
            "event_timestamp": "session_date",
            "knowledge_time_support": "unknown provider publication/first-seen times; authority build time only",
            "first_seen_or_retrieval_timestamp": "unknown; build manifest lacks provider-level first-seen timestamps",
            "revision_behaviour": "partitioned rebuilds can revise provider bridges, quarantines and eligibility",
            "licence_usage_status": "internal research use of configured market data sources",
            "license_usage_status": "internal research use of configured market data sources",
            "licensing_usage_status": "internal research use of configured market data sources",
            "redistribution_restrictions": "generated authority stores derived intervals and hashes, not raw bars",
            "supported_authority_domains": ["observed market data", "rules-based eligibility reconstruction"],
            "authoritative_domains": ["observed market data", "rules-based eligibility reconstruction"],
            "known_gaps": [
                "price presence does not prove official universe membership",
                "source provider publication times unavailable",
            ],
            "trust_tier": "pit_observable_reconstruction_source",
            "authority_population_usable": True,
            "authority_population_usage_reason": "usable for internal PIT-observable reconstruction, not official membership or promotion-grade identity",
            "source_hash": canonical_daily_manifest.get("dataset_logical_partition_hash") or hashes.get("canonical_daily_manifest", ""),
        },
        {
            "source_name": "alpaca_514_symbols_current_scope",
            "provider": "config/universes/alpaca_514_symbols",
            "local_path": str(paths["universe"]),
            "file_api_path": str(paths["universe"]),
            "source_type": "current_static_collection_universe",
            "coverage": "current configured collection list only",
            "coverage_period": "current configured collection list only",
            "update_cadence": "manual config",
            "event_timestamp_support": "not event-dated",
            "event_timestamp": "not event-dated",
            "knowledge_time_support": "unknown",
            "first_seen_or_retrieval_timestamp": "unknown",
            "revision_behaviour": "file revisions replace the current list",
            "licence_usage_status": "internal project config",
            "license_usage_status": "internal project config",
            "licensing_usage_status": "internal project config",
            "redistribution_restrictions": "none beyond project workspace",
            "supported_authority_domains": ["current collection scope"],
            "authoritative_domains": ["current collection scope"],
            "known_gaps": ["not historical membership", "not selector eligibility"],
            "trust_tier": "scope_only",
            "authority_population_usable": True,
            "authority_population_usage_reason": "usable only to define bounded requested symbol scope",
            "source_hash": hashes.get("universe", ""),
        },
        {
            "source_name": "stock_alpha_etf_fund_registry",
            "provider": "stock_alpha_etf_fund_registry",
            "local_path": str(paths["etf_fund_registry"]),
            "file_api_path": str(paths["etf_fund_registry"]),
            "source_type": "internal ETF/fund classification candidate registry",
            "coverage": "current configured fund candidates",
            "coverage_period": "current configured fund candidates",
            "update_cadence": "manual config",
            "event_timestamp_support": "not event-dated",
            "event_timestamp": "not event-dated",
            "knowledge_time_support": "unknown",
            "first_seen_or_retrieval_timestamp": "unknown",
            "revision_behaviour": "file revisions replace classification candidates",
            "licence_usage_status": "internal project config",
            "license_usage_status": "internal project config",
            "licensing_usage_status": "internal project config",
            "redistribution_restrictions": "none beyond project workspace",
            "supported_authority_domains": ["ETF/fund flag candidates"],
            "authoritative_domains": ["ETF/fund flag candidates"],
            "known_gaps": ["official SEC fund series/class mapping is explicitly not implemented"],
            "trust_tier": "classification_candidate_only",
            "authority_population_usable": True,
            "authority_population_usage_reason": "usable only as internal ETF/fund candidate classification",
            "source_hash": hashes.get("etf_fund_registry", ""),
        },
        {
            "source_name": "us_liquid_500_current_snapshot",
            "provider": "data/reference/universes/us_liquid_500",
            "local_path": str(paths["current_liquid_universe"]),
            "file_api_path": str(paths["current_liquid_universe"]),
            "source_type": "current static liquid universe snapshot",
            "coverage": "generated current snapshot",
            "coverage_period": "generated current snapshot",
            "update_cadence": "manual/generated snapshot",
            "event_timestamp_support": "generated_at, not historical membership",
            "event_timestamp": "generated_at, not historical membership",
            "knowledge_time_support": "generated_at in YAML when present; otherwise unknown",
            "first_seen_or_retrieval_timestamp": "generated_at in YAML when present",
            "revision_behaviour": "snapshot file revisions replace current membership",
            "licence_usage_status": "internal derived research config",
            "license_usage_status": "internal derived research config",
            "licensing_usage_status": "internal derived research config",
            "redistribution_restrictions": "none beyond project workspace",
            "supported_authority_domains": ["current static liquid subset comparison"],
            "authoritative_domains": ["current static liquid subset comparison"],
            "known_gaps": ["must not be labeled historical membership"],
            "trust_tier": "current_static_reference_only",
            "authority_population_usable": True,
            "authority_population_usage_reason": "usable only for current static comparison flags",
            "source_hash": hashes.get("current_liquid_universe", ""),
        },
        {
            "source_name": "adjusted_price_manifest",
            "provider": adjusted_manifest.get("source") or "adjusted_price_manifest",
            "local_path": str(paths["adjusted_price_manifest"]),
            "file_api_path": str(paths["adjusted_price_manifest"]),
            "source_type": "supplemental adjusted price import manifest",
            "coverage": "limited imported adjusted-price subset",
            "coverage_period": "limited imported adjusted-price subset",
            "update_cadence": "manual import",
            "event_timestamp_support": "price date",
            "event_timestamp": "price date",
            "knowledge_time_support": "provider download date if present; otherwise unknown",
            "first_seen_or_retrieval_timestamp": adjusted_manifest.get("download_date") or "",
            "revision_behaviour": "provider can revise adjusted history",
            "licence_usage_status": "usage rights not audited for authority population; not ingested as a primary Ticket 62 source",
            "license_usage_status": "usage rights not audited for authority population; not ingested as a primary Ticket 62 source",
            "licensing_usage_status": "usage rights not audited for authority population; not ingested as a primary Ticket 62 source",
            "redistribution_restrictions": "not used to populate generated authority rows",
            "supported_authority_domains": [],
            "authoritative_domains": [],
            "known_gaps": ["not used because usage rights and full coverage are insufficient for this ticket"],
            "trust_tier": "audited_not_used",
            "authority_population_usable": False,
            "authority_population_usage_reason": "licence/usage rights are unknown, so this source is fail-closed and excluded",
            "source_hash": hashes.get("adjusted_price_manifest", ""),
        },
    ]
    return sources


def _source_precedence() -> dict[str, Any]:
    return {
        "schema_version": "ticket62_source_precedence.v1",
        "classification": CLASSIFICATION,
        "domains": {
            "SEC identity and CIK evidence": {
                "primary_source": "SEC EDGAR bulk submissions when extracted and indexed",
                "acceptable_secondary_source": "canonical registry CIK only when populated from SEC evidence",
                "fallback": "clearly namespaced internal issuer/security IDs derived from registry row plus observed market-data span",
                "conflict_rule": "retain all conflicting CIK/issuer mappings and mark identity CONFLICTING_AUTHORITY",
                "unresolved_rule": "missing CIK remains UNRESOLVED and not promotion usable",
                "knowledge_time_rule": "use SEC acceptance/publication/first-seen time when available; otherwise fail closed for promotion",
            },
            "exchange listing status": {
                "primary_source": "official exchange listing/delisting notices",
                "acceptable_secondary_source": "SEC filings with listing status evidence",
                "fallback": "canonical_daily_v2 first/last observed session as observation bounds only",
                "conflict_rule": "official exchange evidence outranks observed bars; conflicting exchange states are retained",
                "unresolved_rule": "last-observed gaps are unresolved delisting/symbol-change candidates",
                "knowledge_time_rule": "provider publication or authority first-seen time required for promotion-grade use",
            },
            "symbol history": {
                "primary_source": "official exchange symbol-change history or SEC identity evidence",
                "acceptable_secondary_source": "verified provider alias history",
                "fallback": "current registry symbol bounded by observed market-data span and flagged uncertified",
                "conflict_rule": "ticker reuse must produce distinct permanent_security_id values; never merge by ticker",
                "unresolved_rule": "current-symbol fallback remains CURRENT_SYMBOL_FALLBACK_UNCERTIFIED",
                "knowledge_time_rule": "recorded_at/first-known time must be <= decision-time knowledge cutoff",
            },
            "corporate action": {
                "primary_source": "official issuer, SEC, exchange or corporate-action vendor event feed",
                "acceptable_secondary_source": "company press release with explicit event terms and publication time",
                "fallback": "observed listing and stale-observation terminal-gap events only",
                "conflict_rule": "retain predecessor/successor conflicts; do not infer economics",
                "unresolved_rule": "unknown merger/spinoff/bankruptcy/liquidation terms remain unresolved",
                "knowledge_time_rule": "announcement, effective and first-known times are distinct",
            },
            "universe membership": {
                "primary_source": "official historical universe/index membership snapshots",
                "acceptable_secondary_source": "versioned internal historical membership snapshots",
                "fallback": "rules-based PIT-observable eligibility reconstruction from canonical_daily_v2",
                "conflict_rule": "current static membership cannot override reconstructed PIT eligibility",
                "unresolved_rule": "current 379/514-symbol lists are current_static_reference_only",
                "knowledge_time_rule": "membership source snapshot first-seen time required for promotion-grade use",
            },
            "IPO/delisting evidence": {
                "primary_source": "official exchange/SEC listing and delisting evidence",
                "acceptable_secondary_source": "verified corporate-action source",
                "fallback": "first/last observed daily market-data sessions as non-official observation bounds",
                "conflict_rule": "official dates outrank first/last bar observations",
                "unresolved_rule": "observed bounds must not be relabeled as official IPO/delisting dates",
                "knowledge_time_rule": "provider publication and first-seen time required for promotion-grade use",
            },
        },
    }


def _dataset_manifest(
    *,
    dataset_name: str,
    row_count: int,
    entity_count: int,
    date_range: Mapping[str, str],
    source_hashes: Mapping[str, str],
    output_path: Path,
    unresolved_count: int,
    conflict_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "ticket62_dataset_manifest.v1",
        "authority_version": AUTHORITY_VERSION,
        "dataset_name": dataset_name,
        "classification": CLASSIFICATION,
        "source_hashes": dict(sorted(source_hashes.items())),
        "source_versions": {"authority": AUTHORITY_VERSION},
        "date_range": dict(date_range),
        "entity_count": int(entity_count),
        "row_count": int(row_count),
        "unresolved_count": int(unresolved_count),
        "conflict_count": int(conflict_count),
        "knowledge_time_coverage": {
            "event_time_distinct_from_knowledge_time": True,
            "provider_publication_time_coverage": "partial_or_unknown",
            "first_seen_time_coverage": "authority_build_time_only_for_reconstructed_sources",
            "promotion_grade_unknown_availability_fails_closed": True,
        },
        "output_file": output_path.name,
        "output_hash": file_sha256(output_path) if output_path.exists() else "",
        "git_config_identity": _git_identity(),
        "permitted_use": "research_reconstruction_only_not_promotion_grade",
    }


def _validation_payload(
    *,
    output: Path,
    artifact_hashes: Mapping[str, str],
    coverage_summary: Mapping[str, Any],
    dataset_manifests: Mapping[str, Mapping[str, Any]],
    source_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    present = sorted(path.name for path in output.iterdir() if path.is_file())
    missing = [name for name in REQUIRED_ARTIFACTS if name not in present]
    return {
        "schema_version": "ticket62_pit_authority_validation.v1",
        "classification": CLASSIFICATION,
        "authority_version": AUTHORITY_VERSION,
        "universe_id": DEFAULT_UNIVERSE_ID,
        "repository_state": "dirty_workspace_supported_unrelated_changes_not_reverted",
        "promotion_usable": PROMOTION_GRADE_USE,
        "model_training_executed": False,
        "portfolio_results_changed": False,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "missing_required_artifacts": missing,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "dataset_manifests": {key: dict(value) for key, value in sorted(dataset_manifests.items())},
        "coverage_summary": dict(coverage_summary),
        "source_count": len(source_inventory),
        "event_time_and_knowledge_time_distinct": True,
        "unknown_availability_fails_closed_for_promotion": True,
        "current_symbol_fallback_certified": False,
        "current_static_universe_labeled_historical": False,
        "official_historical_membership_available": False,
        "rules_based_reconstruction_available": True,
        "test_coverage": [
            "permanent_identity_across_symbol_changes",
            "ticker_reuse_by_unrelated_securities",
            "ipo_before_eligibility",
            "delisting_ends_eligibility",
            "relisting",
            "merger_predecessor_successor_mapping",
            "spinoff_identities",
            "bankruptcy_liquidation",
            "exchange_transfer",
            "current_symbol_fallback_uncertified",
            "event_time_distinct_from_knowledge_time",
            "revision_after_decision",
            "conflicting_sources",
            "unknown_knowledge_time",
            "pit_membership_lookup",
            "rules_based_eligibility_without_future_data",
            "deterministic_rebuild",
            "manifest_hash_validation",
            "selector_adapter",
            "no_model_training_or_promotion",
        ],
        "git_config_identity": _git_identity(),
    }


def _coverage_summary(**kwargs: Any) -> dict[str, Any]:
    security_master = kwargs["security_master"]
    symbol_history = kwargs["symbol_history"]
    corporate_events = kwargs["corporate_events"]
    universe_membership = kwargs["universe_membership"]
    eligibility_intervals = kwargs["eligibility_intervals"]
    identity_conflicts = kwargs["identity_conflicts"]
    universe_conflicts = kwargs["universe_conflicts"]
    coverage_by_security = kwargs["coverage_by_security"]
    coverage_by_year = kwargs["coverage_by_year"]
    symbols_with_missing_market = {
        row["symbol"] for row in identity_conflicts if row.get("conflict_type") == "missing_market_observation"
    }
    unresolved_symbols = {
        row.get("symbol", "")
        for row in [*identity_conflicts, *universe_conflicts]
        if row.get("symbol")
    }
    unknown_knowledge_time_count = (
        sum(1 for row in security_master if row.get("knowledge_time_status") != "provider_publication_time_verified")
        + sum(1 for row in eligibility_intervals if row.get("knowledge_time_status") != "provider_publication_time_verified")
        + len(symbol_history)
        + len(corporate_events)
        + len(universe_membership)
    )
    return {
        "schema_version": "ticket62_coverage_summary.v1",
        "classification": CLASSIFICATION,
        "symbols_requested": len(security_master) + len(symbols_with_missing_market),
        "symbols_populated": len(security_master),
        "symbols_missing_market_observation": len(symbols_with_missing_market),
        "symbols_unresolved": len(unresolved_symbols),
        "security_count": len(security_master),
        "issuer_count": len({row["permanent_issuer_id"] for row in security_master}),
        "identities_externally_corroborated": sum(
            1 for row in security_master if row.get("external_identity_corroborated")
        ),
        "internal_reconstructed_identities": sum(
            1 for row in security_master if row.get("identity_status") == "internal_reconstructed"
        ),
        "static_symbol_fallback_identities": sum(
            1 for row in coverage_by_security if row.get("current_symbol_fallback_uncertified")
        ),
        "cik_coverage_count": sum(1 for row in coverage_by_security if row.get("cik_verified")),
        "cik_coverage_rate": 0.0
        if not coverage_by_security
        else sum(1 for row in coverage_by_security if row.get("cik_verified")) / len(coverage_by_security),
        "exchange_coverage_count": sum(1 for row in coverage_by_security if row.get("exchange_evidenced")),
        "exchange_coverage_rate": 0.0
        if not coverage_by_security
        else sum(1 for row in coverage_by_security if row.get("exchange_evidenced")) / len(coverage_by_security),
        "official_ipo_date_coverage_count": sum(
            1 for row in coverage_by_security if row.get("official_ipo_date_evidenced")
        ),
        "observed_listing_bound_count": sum(1 for row in coverage_by_security if row.get("observed_listing_bound")),
        "official_delisting_date_coverage_count": sum(
            1 for row in coverage_by_security if row.get("official_delisting_date_evidenced")
        ),
        "observed_delisting_bound_count": sum(1 for row in coverage_by_security if row.get("observed_delisting_bound")),
        "symbol_history_row_count": len(symbol_history),
        "effective_symbol_history_verified_count": sum(
            1 for row in symbol_history if row.get("authority_status") == "VERIFIED_HISTORICAL_SYMBOL_HISTORY"
        ),
        "effective_symbol_history_fallback_count": sum(
            1 for row in symbol_history if row.get("authority_status") != "VERIFIED_HISTORICAL_SYMBOL_HISTORY"
        ),
        "corporate_event_count": len(corporate_events),
        "official_corporate_event_count": sum(
            1 for row in corporate_events if not str(row.get("source") or "").startswith("canonical_daily_v2")
        ),
        "observed_corporate_event_bound_count": sum(
            1 for row in corporate_events if str(row.get("source") or "").startswith("canonical_daily_v2")
        ),
        "universe_membership_row_count": len(universe_membership),
        "eligibility_interval_count": len(eligibility_intervals),
        "reconstructed_eligibility_security_count": len(
            {row["permanent_security_id"] for row in eligibility_intervals}
        ),
        "reconstructed_eligibility_interval_count": len(eligibility_intervals),
        "identity_conflict_count": len(identity_conflicts),
        "universe_conflict_count": len(universe_conflicts),
        "conflict_count": sum(1 for row in [*identity_conflicts, *universe_conflicts] if row.get("state") == "CONFLICTING"),
        "unresolved_case_count": sum(
            1
            for row in [*identity_conflicts, *universe_conflicts]
            if row.get("state") in {"UNRESOLVED", "RECONSTRUCTED_ONLY"}
        ),
        "missing_cik_count": sum(1 for row in coverage_by_security if row["missing_cik"]),
        "current_symbol_fallback_count": sum(1 for row in coverage_by_security if row["current_symbol_fallback_uncertified"]),
        "unknown_knowledge_time_count": unknown_knowledge_time_count,
        "open_ended_membership_count": 0,
        "stale_observation_security_count": sum(1 for row in coverage_by_security if row["status"] == "UNRESOLVED_STALE_OBSERVATION"),
        "year_min": coverage_by_year[0]["year"] if coverage_by_year else "",
        "year_max": coverage_by_year[-1]["year"] if coverage_by_year else "",
        "source_count": len(kwargs["source_inventory"]),
        "representative_case_studies": [
            {
                "case": "current_symbol_fallback",
                "status": "uncertified",
                "lesson": "The current registry can scope assets but cannot prove historical symbol ownership.",
            },
            {
                "case": "stale_observation_terminal_gap",
                "status": "unresolved",
                "lesson": "A last observed bar before dataset max is represented as a delisting/symbol-change review candidate, not inferred economics.",
            },
            {
                "case": "missing_cik",
                "status": "unresolved",
                "lesson": "The local SEC bulk archive is audited but not indexed, so CIK population remains blocked for promotion-grade use.",
            },
        ],
    }


def _summary_markdown(
    validation: Mapping[str, Any],
    coverage_summary: Mapping[str, Any],
    source_inventory: Sequence[Mapping[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Ticket 62 Summary",
            "",
            f"Classification: {validation['classification']}",
            "",
            "This authority is a bounded, research-only partial population. It uses the current canonical registry for scope, canonical daily v2 for PIT-observable market-data reconstruction, and audited SEC bulk evidence only as a not-yet-indexed primary source reference.",
            "",
            "## Repository State",
            "Existing authority contracts are reused through new read-only artifacts and a selector adapter. No selector dataset, model, portfolio result, promotion gate or registry source file is overwritten.",
            "",
            "## Source And Licensing Audit",
            f"Audited source records: {len(source_inventory)}. Restricted or unclear sources are represented with hashes/references and not redistributed as raw evidence.",
            "",
            "## Source Precedence",
            "SEC and official exchange evidence outrank internal registry and market observations. Where those sources are unavailable locally, records are marked reconstructed or unresolved rather than silently promoted.",
            "",
            "## Security Identity Population",
            f"Internal permanent securities: {coverage_summary['security_count']}. Missing CIKs: {coverage_summary['missing_cik_count']}. Internal IDs are namespaced and derived from registry identity and alias scope; observed market data bounds effective windows.",
            "",
            "## Symbol History",
            f"Symbol/alias rows: {coverage_summary['symbol_history_row_count']}. Verified historical symbol rows: {coverage_summary['effective_symbol_history_verified_count']}. Current-symbol fallback remains uncertified for historical symbol changes and ticker reuse.",
            "",
            "## Corporate Events",
            f"Corporate event rows: {coverage_summary['corporate_event_count']}. Official corporate events: {coverage_summary['official_corporate_event_count']}. First observed listings and stale terminal gaps are represented; detailed economics are not inferred.",
            "",
            "## Universe Membership And Eligibility",
            f"Membership rows: {coverage_summary['universe_membership_row_count']}. Eligibility intervals: {coverage_summary['eligibility_interval_count']}. Membership is rules-based PIT reconstruction for {coverage_summary['reconstructed_eligibility_security_count']} securities, not official historical membership.",
            "",
            "## Coverage And Conflicts",
            f"Identity conflicts/unresolved cases: {coverage_summary['identity_conflict_count']}. Universe conflicts/unresolved cases: {coverage_summary['universe_conflict_count']}. Current-symbol fallbacks: {coverage_summary['current_symbol_fallback_count']}. Unknown provider knowledge-time rows: {coverage_summary['unknown_knowledge_time_count']}.",
            "",
            "## Validation",
            f"Required artifacts missing: {len(validation['missing_required_artifacts'])}. Promotion usable: {validation['promotion_usable']}. Model training executed: {validation['model_training_executed']}.",
            "",
            "## Limitations",
            "Official historical universe membership, complete SEC CIK extraction, official IPO/delisting dates, complete symbol-change history and economic corporate-action terms remain incomplete.",
            "",
            "## Recommended Next Action",
            "Extract and index the local SEC submissions archive plus official exchange listing/symbol-change data, then re-run this authority with primary-source CIK and corporate-action evidence.",
            "",
        ]
    )


def _eligibility_rules_payload(rules: EligibilityRuleConfig, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "ticket62_eligibility_rules.v1",
        "classification": CLASSIFICATION,
        "universe_id": DEFAULT_UNIVERSE_ID,
        "rule": asdict(rules),
        "uses_future_model_performance": False,
        "feature_data_cutoff_policy": "each decision uses only the current or earlier canonical_daily_v2 session rows",
        "source_hashes": dict(sorted(source_hashes.items())),
        "promotion_usable": False,
    }


def _accumulate_year_coverage(
    coverage_years: dict[int, dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
) -> None:
    symbol = _normalize_symbol(rows[0].get("canonical_symbol")) if rows else ""
    for row in rows:
        year = _date_value(row.get("session_date")).year
        target = coverage_years.setdefault(
            year,
            {
                "year": year,
                "observed_security_count": 0,
                "observed_symbols": set(),
                "observed_row_count": 0,
                "included_interval_count": 0,
                "excluded_interval_count": 0,
                "identity_state": "partial_internal",
                "universe_state": "rules_reconstructed",
            },
        )
        target["observed_symbols"].add(symbol)
        target["observed_row_count"] += 1
    for interval in intervals:
        for year in range(_date_value(interval["effective_from"]).year, _date_value(interval["effective_to"]).year + 1):
            target = coverage_years.setdefault(
                year,
                {
                    "year": year,
                    "observed_security_count": 0,
                    "observed_symbols": set(),
                    "observed_row_count": 0,
                    "included_interval_count": 0,
                    "excluded_interval_count": 0,
                    "identity_state": "partial_internal",
                    "universe_state": "rules_reconstructed",
                },
            )
            if interval["eligibility_state"] == "included":
                target["included_interval_count"] += 1
            else:
                target["excluded_interval_count"] += 1
    for target in coverage_years.values():
        target["observed_security_count"] = len(target["observed_symbols"])


def _finalize_year_coverage(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    symbols = result.get("observed_symbols", set())
    if isinstance(symbols, set):
        result["observed_symbols"] = "|".join(sorted(symbols))
    return result


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        key: file_sha256(path)
        for key, path in sorted(paths.items())
        if path.is_file() and key != "canonical_daily_root"
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_etf_fund_symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    funds = payload.get("funds", {}) if isinstance(payload, Mapping) else {}
    return {_normalize_symbol(symbol) for symbol in funds}


def _read_yaml_symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        return set()
    return {_normalize_symbol(symbol) for symbol in payload.get("symbols", []) or []}


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_json_ready_dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    return {
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "authority_version": AUTHORITY_VERSION,
    }


def _git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False, cwd=Path.cwd())
    except OSError:
        return ""
    return result.stdout.strip()


def _conflict_row(
    *,
    symbol: str,
    permanent_security_id: str,
    conflict_type: str,
    state: str,
    detail: str,
    source: str,
) -> dict[str, Any]:
    return {
        "permanent_security_id": permanent_security_id,
        "symbol": symbol,
        "conflict_type": conflict_type,
        "state": state,
        "detail": detail,
        "source": source,
        "event_time": "",
        "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
        "promotion_usable": False,
    }


def _universe_conflict_row(
    *,
    symbol: str,
    permanent_security_id: str,
    conflict_type: str,
    state: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "universe_id": DEFAULT_UNIVERSE_ID,
        "permanent_security_id": permanent_security_id,
        "symbol": symbol,
        "conflict_type": conflict_type,
        "state": state,
        "detail": detail,
        "knowledge_time": AUTHORITY_KNOWLEDGE_TIME,
        "promotion_usable": False,
    }


def _date_range(rows: Sequence[Mapping[str, Any]], start_field: str, end_field: str) -> dict[str, str]:
    starts = [_date_value(row.get(start_field)).isoformat() for row in rows if row.get(start_field)]
    ends = [_date_value(row.get(end_field)).isoformat() for row in rows if row.get(end_field)]
    return {"start": min(starts) if starts else "", "end": max(ends) if ends else ""}


def _sort_all(*tables: list[dict[str, Any]]) -> None:
    for table in tables:
        table.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _security_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("permanent_security_id") or ""), str(row.get("effective_start") or ""))


def _symbol_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("permanent_security_id") or ""),
        str(row.get("effective_from") or ""),
        str(row.get("symbol") or ""),
    )


def _membership_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("universe_id") or ""),
        str(row.get("permanent_security_id") or ""),
        str(row.get("effective_from") or ""),
    )


def _eligibility_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("permanent_security_id") or ""),
        str(row.get("effective_from") or ""),
        str(row.get("eligibility_state") or ""),
    )


def _event_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("effective_time") or ""),
        str(row.get("event_type") or ""),
        str(row.get("event_id") or ""),
    )


def _listing_state(security: Mapping[str, Any], decision_date: date) -> str:
    start = _date_value(security.get("effective_start"))
    end = _date_value(security.get("effective_end"))
    if decision_date < start:
        return "pre_listing_observation"
    if decision_date > end:
        return "post_last_observed"
    return "listed_observed"


def _record_known_at(row: Mapping[str, Any], knowledge_cutoff: str | None) -> bool:
    if not knowledge_cutoff:
        return True
    value = row.get("knowledge_time") or row.get("first_known_time") or row.get("created_at")
    if not value:
        return False
    return _datetime_value(value) <= _datetime_value(knowledge_cutoff)


def _date_in_interval(value: date, start: Any, end: Any) -> bool:
    if start and value < _date_value(start):
        return False
    if end and value > _date_value(end):
        return False
    return True


def _date_value(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        return date.min
    if "T" in text:
        return _datetime_value(text).date()
    return date.fromisoformat(text[:10])


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        if len(text) == 10:
            text += "T00:00:00Z"
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _decision_timestamp(session_date: date, rules: EligibilityRuleConfig) -> str:
    return f"{session_date.isoformat()}T{rules.decision_time_utc}Z"


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", "-")


def _internal_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{_sha256_json(payload)[:20]}"


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _split_reasons(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part for part in text.split("|") if part)


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _json_ready_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in row.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _identity_conflict_fields() -> tuple[str, ...]:
    return (
        "permanent_security_id",
        "symbol",
        "conflict_type",
        "state",
        "detail",
        "source",
        "event_time",
        "knowledge_time",
        "promotion_usable",
    )


def _universe_conflict_fields() -> tuple[str, ...]:
    return (
        "universe_id",
        "permanent_security_id",
        "symbol",
        "conflict_type",
        "state",
        "detail",
        "knowledge_time",
        "promotion_usable",
    )


def _coverage_year_fields() -> tuple[str, ...]:
    return (
        "year",
        "observed_security_count",
        "observed_symbols",
        "observed_row_count",
        "included_interval_count",
        "excluded_interval_count",
        "identity_state",
        "universe_state",
    )


def _coverage_security_fields() -> tuple[str, ...]:
    return (
        "permanent_security_id",
        "canonical_registry_asset_id",
        "symbol",
        "first_observed_date",
        "last_observed_date",
        "observed_row_count",
        "included_session_count",
        "excluded_session_count",
        "missing_cik",
        "etf_fund_flag",
        "security_type",
        "provider_count",
        "providers",
        "status",
        "current_symbol_fallback_uncertified",
        "unresolved_identity_state_count",
        "identity_status",
        "authority_confidence",
        "external_identity_corroborated",
        "cik_verified",
        "exchange_evidenced",
        "official_ipo_date_evidenced",
        "official_delisting_date_evidenced",
        "observed_listing_bound",
        "observed_delisting_bound",
        "effective_symbol_history_evidenced",
        "reconstructed_eligibility_evidenced",
        "universe_state",
    )


if __name__ == "__main__":
    raise SystemExit(main())
