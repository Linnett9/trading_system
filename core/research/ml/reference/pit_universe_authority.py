from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SECURITY_MASTER_SCHEMA_VERSION = "pit_security_master.v1"
MEMBERSHIP_SCHEMA_VERSION = "pit_universe_membership.v1"
SYMBOL_HISTORY_SCHEMA_VERSION = "pit_symbol_history.v1"
CORPORATE_ACTION_SCHEMA_VERSION = "pit_corporate_action.v1"
STATIC_UNIVERSE_SCHEMA_VERSION = "pit_static_universe_adapter.v1"
QUERY_RESULT_SCHEMA_VERSION = "pit_universe_query_result.v1"
VALIDATION_ARTIFACT_SCHEMA_VERSION = "pit_universe_authority_validation.v1"

LATEST_AUTHORITY_VERSION = "latest"
STATIC_UNCERTIFIED_STATUS = "STATIC_UNCERTIFIED"

MEMBERSHIP_INCLUDED = "included"
MEMBERSHIP_EXCLUDED = "excluded"
MEMBERSHIP_UNKNOWN = "unknown"

TERMINAL_EVENT_TYPES = frozenset(
    {
        "bankruptcy",
        "liquidation",
        "terminal_delisting",
        "merger_predecessor_terminal",
    }
)
TERMINAL_SECURITY_STATUSES = frozenset({"DELISTED", "BANKRUPT", "LIQUIDATED", "MERGED"})
UNRESOLVED_SECURITY_STATUSES = frozenset({"UNKNOWN", "UNKNOWN_LISTING_DATE", "UNKNOWN_DELISTING_STATUS", "UNVERIFIED", STATIC_UNCERTIFIED_STATUS})

PRECEDENCE_POLICY = (
    {
        "authority_type": "manual_verified_security_master",
        "rank": 100,
        "description": "Verified effective-dated security-master records outrank every inferred or static source.",
    },
    {
        "authority_type": "authoritative_membership",
        "rank": 90,
        "description": "Effective-dated universe membership records determine inclusion when security-master gates pass.",
    },
    {
        "authority_type": "corporate_action",
        "rank": 80,
        "description": "Confirmed terminal events can make an asset ineligible from the event date.",
    },
    {
        "authority_type": "canonical_registry_default",
        "rank": 40,
        "description": "Current registry defaults are lineage only and do not prove historical membership.",
    },
    {
        "authority_type": "provider_alias",
        "rank": 30,
        "description": "Provider aliases resolve symbols but do not prove universe membership.",
    },
    {
        "authority_type": "price_availability",
        "rank": 20,
        "description": "Price observations do not prove authoritative universe membership or ineligibility.",
    },
    {
        "authority_type": "inferred_observation",
        "rank": 10,
        "description": "Inferred observations are diagnostic and cannot silently create eligibility.",
    },
    {
        "authority_type": "static_universe_adapter",
        "rank": 0,
        "description": "Legacy static universes are exposed as unresolved/uncertified, never certified PIT membership.",
    },
)
PRECEDENCE_RANK = {row["authority_type"]: int(row["rank"]) for row in PRECEDENCE_POLICY}


@dataclass(frozen=True)
class SecurityMasterRecord:
    asset_id: str
    canonical_symbol: str
    security_name: str
    asset_type: str
    listing_date: str
    delisting_date: str
    status: str
    exchange: str
    authority_source: str
    authority_type: str
    authority_version: str
    recorded_at: str
    effective_from: str
    effective_to: str
    verification_status: str
    confidence: str
    reason_code: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class MembershipRecord:
    universe_id: str
    asset_id: str
    membership_start: str
    membership_end: str
    state: str
    reason_code: str
    authority_source: str
    authority_type: str
    authority_version: str
    recorded_at: str
    effective_from: str
    effective_to: str
    verification_status: str
    confidence: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class SymbolHistoryRecord:
    asset_id: str
    symbol: str
    symbol_kind: str
    provider: str
    valid_from: str
    valid_to: str
    authority_source: str
    authority_type: str
    authority_version: str
    recorded_at: str
    effective_from: str
    effective_to: str
    verification_status: str
    confidence: str
    reason_code: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class CorporateActionRecord:
    event_id: str
    event_type: str
    asset_id: str
    related_asset_id: str
    effective_date: str
    post_event_state: str
    authority_source: str
    authority_type: str
    authority_version: str
    recorded_at: str
    effective_from: str
    effective_to: str
    verification_status: str
    confidence: str
    reason_code: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class StaticUniverseAsset:
    universe_id: str
    asset_id: str
    symbol: str
    source: str
    authority_version: str
    recorded_at: str
    lineage: Mapping[str, Any]


class PointInTimeUniverseAuthority:
    """Deterministic PIT universe authority for historical research gates.

    This class is intentionally independent from model training and portfolio
    optimisation. It exposes legacy static universes only as unresolved,
    uncertified candidates so current membership cannot silently become
    historical membership.
    """

    def __init__(
        self,
        *,
        security_master: Sequence[SecurityMasterRecord] = (),
        memberships: Sequence[MembershipRecord] = (),
        symbol_history: Sequence[SymbolHistoryRecord] = (),
        corporate_actions: Sequence[CorporateActionRecord] = (),
        static_assets: Sequence[StaticUniverseAsset] = (),
        authority_versions: Sequence[str] = (),
    ) -> None:
        self.security_master = tuple(sorted(security_master, key=_record_sort_key))
        self.memberships = tuple(sorted(memberships, key=_record_sort_key))
        self.symbol_history = tuple(sorted(symbol_history, key=_record_sort_key))
        self.corporate_actions = tuple(sorted(corporate_actions, key=_record_sort_key))
        self.static_assets = tuple(sorted(static_assets, key=_record_sort_key))
        self.authority_versions = tuple(authority_versions) or _discover_versions(
            [self.security_master, self.memberships, self.symbol_history, self.corporate_actions, self.static_assets]
        )
        self._version_rank = {version: index for index, version in enumerate(self.authority_versions)}
        self._validate_versions()
        self._validate_identity()

    @classmethod
    def from_csv_directory(cls, path: Path | str) -> "PointInTimeUniverseAuthority":
        root = Path(path)
        manifest = _read_json(root / "manifest.json")
        return cls(
            security_master=[_security_from_row(row) for row in _read_csv(root / "security_master.csv")],
            memberships=[_membership_from_row(row) for row in _read_csv(root / "membership.csv")],
            symbol_history=[_symbol_from_row(row) for row in _read_csv(root / "symbol_history.csv")],
            corporate_actions=[_corporate_action_from_row(row) for row in _read_csv(root / "corporate_actions.csv")],
            static_assets=[_static_from_row(row) for row in _read_csv(root / "static_universe.csv")],
            authority_versions=tuple(str(version) for version in manifest.get("authority_versions", ())),
        )

    @classmethod
    def from_static_universe(
        cls,
        *,
        universe_id: str,
        symbols: Iterable[str],
        authority_version: str = "legacy_static_universe.v1",
        recorded_at: str = "1970-01-01T00:00:00+00:00",
        source: str = "legacy_static_universe_adapter",
    ) -> "PointInTimeUniverseAuthority":
        static_assets = [
            StaticUniverseAsset(
                universe_id=universe_id,
                asset_id=_static_asset_id(universe_id, symbol),
                symbol=normalize_symbol(symbol),
                source=source,
                authority_version=authority_version,
                recorded_at=recorded_at,
                lineage={
                    "classification": STATIC_UNCERTIFIED_STATUS,
                    "survivorship_certification": "not_certified",
                    "historical_membership": "not_authoritative",
                },
            )
            for symbol in sorted({normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)})
        ]
        return cls(static_assets=static_assets, authority_versions=(authority_version,))

    def query(
        self,
        *,
        universe_id: str,
        decision_timestamp: str,
        authority_version: str = LATEST_AUTHORITY_VERSION,
        knowledge_cutoff: str | None = None,
    ) -> dict[str, Any]:
        decision_date = _decision_date(decision_timestamp)
        cutoff_dt = _parse_datetime(knowledge_cutoff) if knowledge_cutoff else None
        resolved_version = self._resolve_authority_version(authority_version, cutoff_dt)
        version_rank = self._rank_for_version(resolved_version)

        security = _by_asset(
            self._available_records(self.security_master, version_rank=version_rank, cutoff_dt=cutoff_dt)
        )
        memberships = [
            row
            for row in self._available_records(self.memberships, version_rank=version_rank, cutoff_dt=cutoff_dt)
            if row.universe_id == universe_id
        ]
        symbols = self._available_records(self.symbol_history, version_rank=version_rank, cutoff_dt=cutoff_dt)
        actions = self._available_records(self.corporate_actions, version_rank=version_rank, cutoff_dt=cutoff_dt)
        static_assets = [
            row
            for row in self._available_records(self.static_assets, version_rank=version_rank, cutoff_dt=cutoff_dt)
            if row.universe_id == universe_id
        ]
        candidate_ids = sorted({row.asset_id for row in memberships} | {row.asset_id for row in static_assets})

        eligible: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []

        for asset_id in candidate_ids:
            static = [row for row in static_assets if row.asset_id == asset_id]
            security_choice, security_conflict = _select_security(security.get(asset_id, ()))
            membership_choice, membership_conflict = _select_membership(
                memberships,
                asset_id=asset_id,
                decision_date=decision_date,
            )
            terminal_action = _select_terminal_action(actions, asset_id=asset_id, decision_date=decision_date)
            symbol = _symbol_at(symbols, asset_id=asset_id, decision_date=decision_date, provider="canonical")
            provider_symbols = _provider_symbols_at(symbols, asset_id=asset_id, decision_date=decision_date)

            if security_conflict:
                conflicts.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="conflicting_authority",
                        reason_code="CONFLICTING_SECURITY_MASTER",
                        sources=security_conflict,
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if membership_conflict:
                conflicts.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="conflicting_authority",
                        reason_code="CONFLICTING_MEMBERSHIP",
                        sources=membership_conflict,
                        symbol=symbol or (security_choice.canonical_symbol if security_choice else ""),
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if static and not security_choice and not membership_choice:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code=STATIC_UNCERTIFIED_STATUS,
                        sources=static,
                        symbol=static[0].symbol,
                        provider_symbols={},
                    )
                )
                continue

            if not security_choice:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code="MISSING_SECURITY_MASTER",
                        sources=[membership_choice] if membership_choice else static,
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            symbol = symbol or security_choice.canonical_symbol
            listing_date = _known_date(security_choice.listing_date)
            delisting_date = _known_date(security_choice.delisting_date)
            status = security_choice.status.upper()

            if not listing_date:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code="UNKNOWN_LISTING_DATE",
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if decision_date < listing_date:
                exclusions.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="known_ineligible",
                        reason_code="PRE_LISTING",
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if terminal_action:
                exclusions.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="known_ineligible",
                        reason_code=f"TERMINAL_EVENT:{terminal_action.event_type}",
                        sources=[security_choice, membership_choice, terminal_action],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if delisting_date and decision_date > delisting_date:
                exclusions.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="known_ineligible",
                        reason_code="POST_DELISTING",
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if status in TERMINAL_SECURITY_STATUSES and not delisting_date:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code="TERMINAL_STATUS_WITH_UNKNOWN_DATE",
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if status in UNRESOLVED_SECURITY_STATUSES:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code=status,
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if not membership_choice:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code="NO_MEMBERSHIP_AUTHORITY",
                        sources=[security_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if membership_choice.state == MEMBERSHIP_EXCLUDED:
                exclusions.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="known_ineligible",
                        reason_code=membership_choice.reason_code or "MEMBERSHIP_EXCLUDED",
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            if membership_choice.state == MEMBERSHIP_UNKNOWN:
                unresolved.append(
                    _diagnostic(
                        asset_id=asset_id,
                        state="unresolved",
                        reason_code=membership_choice.reason_code or "MEMBERSHIP_UNKNOWN",
                        sources=[security_choice, membership_choice],
                        symbol=symbol,
                        provider_symbols=provider_symbols,
                    )
                )
                continue

            eligible.append(
                _diagnostic(
                    asset_id=asset_id,
                    state="known_eligible",
                    reason_code=membership_choice.reason_code or "MEMBERSHIP_INCLUDED",
                    sources=[security_choice, membership_choice],
                    symbol=symbol,
                    provider_symbols=provider_symbols,
                )
            )

        result = {
            "schema_version": QUERY_RESULT_SCHEMA_VERSION,
            "universe_id": universe_id,
            "decision_timestamp": _normalize_decision_timestamp(decision_timestamp),
            "decision_date": decision_date.isoformat(),
            "authority_version": resolved_version,
            "requested_authority_version": authority_version,
            "knowledge_cutoff": _normalize_optional_timestamp(knowledge_cutoff),
            "coverage_status": _coverage_status(eligible, exclusions, unresolved, conflicts, static_assets),
            "eligible_asset_ids": [row["asset_id"] for row in sorted(eligible, key=_diagnostic_sort_key)],
            "eligible_assets": sorted(eligible, key=_diagnostic_sort_key),
            "exclusions": sorted(exclusions, key=_diagnostic_sort_key),
            "unresolved_assets": sorted(unresolved, key=_diagnostic_sort_key),
            "conflicts": sorted(conflicts, key=_diagnostic_sort_key),
            "lineage": {
                "authority_versions": list(self.authority_versions),
                "precedence_policy": list(PRECEDENCE_POLICY),
                "record_counts": self.record_counts(),
                "classification": "point_in_time_authority" if not static_assets else "contains_static_uncertified_adapter",
            },
        }
        return result

    def record_counts(self) -> dict[str, int]:
        return {
            "security_master": len(self.security_master),
            "membership": len(self.memberships),
            "symbol_history": len(self.symbol_history),
            "corporate_actions": len(self.corporate_actions),
            "static_assets": len(self.static_assets),
        }

    def validation_artifact(self) -> dict[str, Any]:
        universe_ids = sorted({row.universe_id for row in self.memberships} | {row.universe_id for row in self.static_assets})
        return {
            "schema_version": VALIDATION_ARTIFACT_SCHEMA_VERSION,
            "authority_versions": list(self.authority_versions),
            "record_counts": self.record_counts(),
            "universe_ids": universe_ids,
            "precedence_policy": list(PRECEDENCE_POLICY),
            "failure_policy": {
                "pre_listing_assets": "known_ineligible",
                "post_delisting_assets": "known_ineligible",
                "terminal_events": "known_ineligible_from_effective_date",
                "unresolved_membership": "reported_not_eligible",
                "authority_conflicts": "reported_not_eligible",
                "current_active_status": "does_not_prove_historical_eligibility",
                "price_row_presence": "does_not_prove_membership",
                "price_row_absence": "does_not_prove_ineligibility",
                "static_universe_adapter": STATIC_UNCERTIFIED_STATUS,
            },
        }

    def write_validation_artifact(self, path: Path | str) -> None:
        write_json(Path(path), self.validation_artifact())

    def _available_records(self, records: Sequence[Any], *, version_rank: int, cutoff_dt: datetime | None) -> list[Any]:
        available: list[Any] = []
        for record in records:
            if self._rank_for_version(record.authority_version) > version_rank:
                continue
            if cutoff_dt is not None and _parse_datetime(record.recorded_at) > cutoff_dt:
                continue
            available.append(record)
        return available

    def _resolve_authority_version(self, authority_version: str, cutoff_dt: datetime | None) -> str:
        if authority_version != LATEST_AUTHORITY_VERSION:
            if authority_version not in self._version_rank:
                raise ValueError(f"unknown authority_version: {authority_version}")
            return authority_version
        if cutoff_dt is None:
            if not self.authority_versions:
                raise ValueError("no authority versions available")
            return self.authority_versions[-1]
        known_versions: list[str] = []
        for version in self.authority_versions:
            rank = self._rank_for_version(version)
            records = self._available_records(
                [*self.security_master, *self.memberships, *self.symbol_history, *self.corporate_actions, *self.static_assets],
                version_rank=rank,
                cutoff_dt=cutoff_dt,
            )
            if any(record.authority_version == version for record in records):
                known_versions.append(version)
        if not known_versions:
            raise ValueError("no authority version available at knowledge_cutoff")
        return known_versions[-1]

    def _rank_for_version(self, version: str) -> int:
        try:
            return self._version_rank[version]
        except KeyError as exc:
            raise ValueError(f"unknown authority_version: {version}") from exc

    def _validate_versions(self) -> None:
        if not self.authority_versions:
            raise ValueError("at least one authority version is required")
        seen = set()
        duplicates = []
        for version in self.authority_versions:
            if version in seen:
                duplicates.append(version)
            seen.add(version)
        if duplicates:
            raise ValueError("duplicate authority versions: " + ", ".join(sorted(duplicates)))
        for record in [*self.security_master, *self.memberships, *self.symbol_history, *self.corporate_actions, *self.static_assets]:
            if record.authority_version not in self._version_rank:
                raise ValueError(f"record references undeclared authority_version: {record.authority_version}")

    def _validate_identity(self) -> None:
        bad = [
            record.asset_id
            for record in self.security_master
            if normalize_symbol(record.asset_id) == normalize_symbol(record.canonical_symbol)
        ]
        if bad:
            raise ValueError("asset_id must not be the ticker symbol: " + ", ".join(sorted(set(bad))))


def load_pit_universe_authority(path: Path | str) -> PointInTimeUniverseAuthority:
    return PointInTimeUniverseAuthority.from_csv_directory(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _security_from_row(row: Mapping[str, Any]) -> SecurityMasterRecord:
    return SecurityMasterRecord(
        asset_id=_text(row, "asset_id"),
        canonical_symbol=normalize_symbol(row.get("canonical_symbol")),
        security_name=_text(row, "security_name"),
        asset_type=_text(row, "asset_type"),
        listing_date=_text(row, "listing_date"),
        delisting_date=_text(row, "delisting_date"),
        status=_text(row, "status").upper(),
        exchange=_text(row, "exchange"),
        authority_source=_text(row, "authority_source"),
        authority_type=_text(row, "authority_type"),
        authority_version=_text(row, "authority_version"),
        recorded_at=_text(row, "recorded_at"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        verification_status=_text(row, "verification_status"),
        confidence=_text(row, "confidence"),
        reason_code=_text(row, "reason_code"),
        lineage=_lineage(row.get("lineage")),
    )


def _membership_from_row(row: Mapping[str, Any]) -> MembershipRecord:
    return MembershipRecord(
        universe_id=_text(row, "universe_id"),
        asset_id=_text(row, "asset_id"),
        membership_start=_text(row, "membership_start"),
        membership_end=_text(row, "membership_end"),
        state=_text(row, "state").lower(),
        reason_code=_text(row, "reason_code"),
        authority_source=_text(row, "authority_source"),
        authority_type=_text(row, "authority_type"),
        authority_version=_text(row, "authority_version"),
        recorded_at=_text(row, "recorded_at"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        verification_status=_text(row, "verification_status"),
        confidence=_text(row, "confidence"),
        lineage=_lineage(row.get("lineage")),
    )


def _symbol_from_row(row: Mapping[str, Any]) -> SymbolHistoryRecord:
    return SymbolHistoryRecord(
        asset_id=_text(row, "asset_id"),
        symbol=normalize_symbol(row.get("symbol")),
        symbol_kind=_text(row, "symbol_kind").lower(),
        provider=_text(row, "provider").lower(),
        valid_from=_text(row, "valid_from"),
        valid_to=_text(row, "valid_to"),
        authority_source=_text(row, "authority_source"),
        authority_type=_text(row, "authority_type"),
        authority_version=_text(row, "authority_version"),
        recorded_at=_text(row, "recorded_at"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        verification_status=_text(row, "verification_status"),
        confidence=_text(row, "confidence"),
        reason_code=_text(row, "reason_code"),
        lineage=_lineage(row.get("lineage")),
    )


def _corporate_action_from_row(row: Mapping[str, Any]) -> CorporateActionRecord:
    return CorporateActionRecord(
        event_id=_text(row, "event_id"),
        event_type=_text(row, "event_type").lower(),
        asset_id=_text(row, "asset_id"),
        related_asset_id=_text(row, "related_asset_id"),
        effective_date=_text(row, "effective_date"),
        post_event_state=_text(row, "post_event_state").lower(),
        authority_source=_text(row, "authority_source"),
        authority_type=_text(row, "authority_type"),
        authority_version=_text(row, "authority_version"),
        recorded_at=_text(row, "recorded_at"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        verification_status=_text(row, "verification_status"),
        confidence=_text(row, "confidence"),
        reason_code=_text(row, "reason_code"),
        lineage=_lineage(row.get("lineage")),
    )


def _static_from_row(row: Mapping[str, Any]) -> StaticUniverseAsset:
    symbol = normalize_symbol(row.get("symbol"))
    universe_id = _text(row, "universe_id")
    return StaticUniverseAsset(
        universe_id=universe_id,
        asset_id=_text(row, "asset_id") or _static_asset_id(universe_id, symbol),
        symbol=symbol,
        source=_text(row, "source"),
        authority_version=_text(row, "authority_version"),
        recorded_at=_text(row, "recorded_at"),
        lineage=_lineage(row.get("lineage")),
    )


def _select_security(records: Sequence[SecurityMasterRecord]) -> tuple[SecurityMasterRecord | None, list[Any]]:
    if not records:
        return None, []
    top = _top_precedence(records)
    latest = _latest_recorded(top)
    keys = {
        (
            row.canonical_symbol,
            row.listing_date,
            row.delisting_date,
            row.status,
            row.verification_status,
        )
        for row in latest
    }
    if len(keys) > 1:
        return None, list(latest)
    return latest[0], []


def _select_membership(
    records: Sequence[MembershipRecord],
    *,
    asset_id: str,
    decision_date: date,
) -> tuple[MembershipRecord | None, list[Any]]:
    applicable = [
        row
        for row in records
        if row.asset_id == asset_id and _date_in_interval(decision_date, row.membership_start, row.membership_end)
    ]
    if not applicable:
        return None, []
    top = _top_precedence(applicable)
    states = {row.state for row in top}
    if len(states) > 1:
        return None, list(top)
    latest = _latest_recorded(top)
    return latest[0], []


def _select_terminal_action(
    records: Sequence[CorporateActionRecord],
    *,
    asset_id: str,
    decision_date: date,
) -> CorporateActionRecord | None:
    applicable = [
        row
        for row in records
        if row.asset_id == asset_id
        and row.event_type in TERMINAL_EVENT_TYPES
        and _known_date(row.effective_date) is not None
        and decision_date >= _known_date(row.effective_date)
    ]
    if not applicable:
        return None
    return _latest_recorded(_top_precedence(applicable))[0]


def _symbol_at(records: Sequence[SymbolHistoryRecord], *, asset_id: str, decision_date: date, provider: str) -> str:
    matches = [
        row
        for row in records
        if row.asset_id == asset_id and row.provider == provider and _date_in_interval(decision_date, row.valid_from, row.valid_to)
    ]
    if not matches:
        return ""
    return _latest_recorded(_top_precedence(matches))[0].symbol


def _provider_symbols_at(records: Sequence[SymbolHistoryRecord], *, asset_id: str, decision_date: date) -> dict[str, str]:
    providers = sorted({row.provider for row in records if row.asset_id == asset_id and row.provider != "canonical"})
    result: dict[str, str] = {}
    for provider in providers:
        symbol = _symbol_at(records, asset_id=asset_id, decision_date=decision_date, provider=provider)
        if symbol:
            result[provider] = symbol
    return result


def _top_precedence(records: Sequence[Any]) -> list[Any]:
    best_rank = max(_precedence_rank(row) for row in records)
    return [row for row in records if _precedence_rank(row) == best_rank]


def _latest_recorded(records: Sequence[Any]) -> list[Any]:
    best_recorded_at = max(_parse_datetime(row.recorded_at) for row in records)
    latest = [row for row in records if _parse_datetime(row.recorded_at) == best_recorded_at]
    return sorted(latest, key=_record_sort_key)


def _precedence_rank(record: Any) -> int:
    return PRECEDENCE_RANK.get(record.authority_type, -1)


def _diagnostic(
    *,
    asset_id: str,
    state: str,
    reason_code: str,
    sources: Sequence[Any | None],
    symbol: str,
    provider_symbols: Mapping[str, str],
) -> dict[str, Any]:
    clean_sources = [source for source in sources if source is not None]
    return {
        "asset_id": asset_id,
        "symbol": symbol,
        "provider_symbols": dict(sorted(provider_symbols.items())),
        "state": state,
        "reason_code": reason_code,
        "sources": [_source_summary(source) for source in clean_sources],
        "lineage": {
            "source_count": len(clean_sources),
            "source_lineage": [_lineage_mapping(source) for source in clean_sources],
        },
    }


def _source_summary(record: Any) -> dict[str, Any]:
    result = {
        "record_type": record.__class__.__name__,
        "authority_source": getattr(record, "authority_source", getattr(record, "source", "")),
        "authority_type": getattr(record, "authority_type", "static_universe_adapter"),
        "authority_version": getattr(record, "authority_version", ""),
        "recorded_at": getattr(record, "recorded_at", ""),
        "reason_code": getattr(record, "reason_code", ""),
        "precedence_rank": _precedence_rank(record) if hasattr(record, "authority_type") else PRECEDENCE_RANK["static_universe_adapter"],
    }
    for field in ("state", "status", "event_type", "membership_start", "membership_end", "listing_date", "delisting_date", "effective_date"):
        if hasattr(record, field):
            result[field] = getattr(record, field)
    return result


def _lineage_mapping(record: Any) -> Mapping[str, Any]:
    return getattr(record, "lineage", {})


def _coverage_status(
    eligible: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    unresolved: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    static_assets: Sequence[StaticUniverseAsset],
) -> str:
    if conflicts:
        return "CONFLICTING_AUTHORITY"
    if unresolved:
        if static_assets and not eligible and not exclusions:
            return STATIC_UNCERTIFIED_STATUS
        return "PARTIAL_UNRESOLVED"
    if not eligible and not exclusions:
        return "EMPTY"
    return "COMPLETE"


def _by_asset(records: Sequence[Any]) -> dict[str, tuple[Any, ...]]:
    result: dict[str, list[Any]] = {}
    for record in records:
        result.setdefault(record.asset_id, []).append(record)
    return {asset_id: tuple(rows) for asset_id, rows in result.items()}


def _date_in_interval(value: date, start: str, end: str) -> bool:
    start_date = _known_date(start)
    end_date = _known_date(end)
    if start_date and value < start_date:
        return False
    if end_date and value > end_date:
        return False
    return True


def _known_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "none", "null", "na", "n/a"}:
        return None
    if "T" in text:
        return _parse_datetime(text).date()
    return date.fromisoformat(text[:10])


def _decision_date(value: str) -> date:
    parsed = _known_date(value)
    if parsed is None:
        raise ValueError("decision_timestamp must contain a date")
    return parsed


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    if len(text) == 10:
        return datetime.combine(date.fromisoformat(text), time.min, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_decision_timestamp(value: str) -> str:
    parsed = _parse_datetime(value)
    if "T" not in value:
        return parsed.date().isoformat()
    return parsed.isoformat()


def _normalize_optional_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    return _parse_datetime(value).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _lineage(value: Any) -> Mapping[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"raw": payload}


def _text(row: Mapping[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def _discover_versions(record_groups: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    versions: list[str] = []
    for group in record_groups:
        for record in group:
            version = record.authority_version
            if version and version not in versions:
                versions.append(version)
    return tuple(versions)


def _static_asset_id(universe_id: str, symbol: str) -> str:
    payload = f"static_universe:{universe_id}:{normalize_symbol(symbol)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"static_asset_{digest}"


def _record_sort_key(record: Any) -> tuple[Any, ...]:
    return (
        getattr(record, "universe_id", ""),
        getattr(record, "asset_id", ""),
        getattr(record, "authority_version", ""),
        getattr(record, "recorded_at", ""),
        getattr(record, "effective_from", ""),
        getattr(record, "membership_start", ""),
        getattr(record, "symbol", ""),
        getattr(record, "event_id", ""),
        getattr(record, "source", ""),
    )


def _diagnostic_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("asset_id", "")), str(row.get("symbol", "")), str(row.get("reason_code", "")))
