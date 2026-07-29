from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HISTORICAL_IDENTITY_AUTHORITY_SCHEMA_VERSION = "historical_identity_authority.v1"
HISTORICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION = "historical_identity_resolution.v1"
PIT_IDENTITY_ENRICHMENT_SCHEMA_VERSION = "pit_universe_identity_enrichment.v1"

RESOLVED = "RESOLVED"
UNRESOLVED = "UNRESOLVED"
AMBIGUOUS = "AMBIGUOUS"
CONFLICTING_AUTHORITY = "CONFLICTING_AUTHORITY"
OUTSIDE_EFFECTIVE_WINDOW = "OUTSIDE_EFFECTIVE_WINDOW"
UNKNOWN_AT_KNOWLEDGE_CUTOFF = "UNKNOWN_AT_KNOWLEDGE_CUTOFF"

LATEST_AUTHORITY_VERSION = "latest"
PRODUCTION_SYMBOL_RESOLUTION_ENABLED_DEFAULT = False

PRECEDENCE_POLICY = (
    ("manual_verified_identity", 100),
    ("authoritative_symbol_history", 90),
    ("authoritative_corporate_action_lineage", 80),
    ("verified_provider_alias", 70),
    ("canonical_registry_default", 40),
    ("inferred_mapping", 20),
    ("current_static_symbol_assumption", 0),
)
PRECEDENCE_RANK = dict(PRECEDENCE_POLICY)

TERMINAL_EVENT_TYPES = frozenset(
    {"bankruptcy", "liquidation", "delisting", "cash_acquisition", "acquisition", "merger"}
)


@dataclass(frozen=True)
class PermanentAssetIdentity:
    permanent_asset_id: str
    asset_kind: str
    legal_entity_hint: str
    current_symbol_hint: str
    created_at: str
    authority_version: str
    source: str
    verification_state: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class HistoricalSymbolRecord:
    permanent_asset_id: str
    symbol: str
    normalized_symbol: str
    provider: str
    venue: str
    effective_from: str
    effective_to: str
    recorded_at: str
    authority_version: str
    authority_type: str
    source: str
    verification_state: str
    reason_code: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderAliasRecord:
    permanent_asset_id: str
    alias: str
    normalized_alias: str
    provider: str
    venue: str
    effective_from: str
    effective_to: str
    recorded_at: str
    authority_version: str
    authority_type: str
    source: str
    verification_state: str
    reason_code: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class CompanyNameRecord:
    permanent_asset_id: str
    name: str
    effective_from: str
    effective_to: str
    recorded_at: str
    authority_version: str
    authority_type: str
    source: str
    verification_state: str
    reason_code: str
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class CorporateActionLineageEvent:
    event_id: str
    event_type: str
    predecessor_asset_ids: tuple[str, ...]
    successor_asset_ids: tuple[str, ...]
    effective_timestamp: str
    recorded_at: str
    source: str
    authority_version: str
    authority_type: str
    verification_state: str
    confidence: str
    relationship_semantics: str
    terminal_economics: str
    unresolved_fields: tuple[str, ...]
    notes: str
    lineage: Mapping[str, Any]


class HistoricalIdentityAuthority:
    """Research-only historical identity and corporate-action lineage authority.

    The authority is additive and disabled by default for production selection.
    It resolves historical identity using event-time effective windows and
    optional knowledge-time cutoffs, and it refuses to choose among ambiguous
    mappings silently.
    """

    def __init__(
        self,
        *,
        permanent_assets: Sequence[PermanentAssetIdentity] = (),
        symbol_history: Sequence[HistoricalSymbolRecord] = (),
        provider_aliases: Sequence[ProviderAliasRecord] = (),
        company_names: Sequence[CompanyNameRecord] = (),
        corporate_actions: Sequence[CorporateActionLineageEvent] = (),
        authority_versions: Sequence[str] = (),
        production_selection_enabled: bool = PRODUCTION_SYMBOL_RESOLUTION_ENABLED_DEFAULT,
    ) -> None:
        self.permanent_assets = tuple(sorted(permanent_assets, key=_record_sort_key))
        self.symbol_history = tuple(sorted(symbol_history, key=_record_sort_key))
        self.provider_aliases = tuple(sorted(provider_aliases, key=_record_sort_key))
        self.company_names = tuple(sorted(company_names, key=_record_sort_key))
        self.corporate_actions = tuple(sorted(corporate_actions, key=_event_sort_key))
        self.authority_versions = tuple(authority_versions) or _discover_versions(
            [
                self.permanent_assets,
                self.symbol_history,
                self.provider_aliases,
                self.company_names,
                self.corporate_actions,
            ]
        )
        self.production_selection_enabled = bool(production_selection_enabled)
        self._version_rank = {version: index for index, version in enumerate(self.authority_versions)}
        self._validate()

    @classmethod
    def from_json(cls, path: Path | str) -> "HistoricalIdentityAuthority":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HistoricalIdentityAuthority":
        return cls(
            permanent_assets=[_asset(row) for row in payload.get("permanent_assets", ())],
            symbol_history=[_symbol(row) for row in payload.get("symbol_history", ())],
            provider_aliases=[_alias(row) for row in payload.get("provider_aliases", ())],
            company_names=[_name(row) for row in payload.get("company_names", ())],
            corporate_actions=[_event(row) for row in payload.get("corporate_actions", ())],
            authority_versions=tuple(str(version) for version in payload.get("authority_versions", ())),
            production_selection_enabled=bool(
                payload.get("production_selection_enabled", PRODUCTION_SYMBOL_RESOLUTION_ENABLED_DEFAULT)
            ),
        )

    @classmethod
    def from_canonical_registry(
        cls,
        assets: Sequence[Any],
        aliases: Sequence[Any] = (),
        *,
        authority_version: str = "canonical_registry_static.v1",
        recorded_at: str = "1970-01-01T00:00:00Z",
    ) -> "HistoricalIdentityAuthority":
        permanent_assets = [
            PermanentAssetIdentity(
                permanent_asset_id=str(asset.asset_id),
                asset_kind=str(getattr(asset, "security_type", "UNKNOWN") or "UNKNOWN"),
                legal_entity_hint="",
                current_symbol_hint=normalize_symbol(getattr(asset, "canonical_symbol", "")),
                created_at=recorded_at,
                authority_version=authority_version,
                source="canonical_asset_registry",
                verification_state="STATIC_UNCERTIFIED",
                lineage={"registry_version": getattr(asset, "registry_version", "")},
            )
            for asset in assets
        ]
        symbol_history = [
            HistoricalSymbolRecord(
                permanent_asset_id=str(asset.asset_id),
                symbol=normalize_symbol(getattr(asset, "canonical_symbol", "")),
                normalized_symbol=normalize_symbol(getattr(asset, "canonical_symbol", "")),
                provider="canonical",
                venue=str(getattr(asset, "exchange", "") or ""),
                effective_from=str(getattr(asset, "valid_from", "") or "1900-01-01T00:00:00Z"),
                effective_to=str(getattr(asset, "valid_to", "") or ""),
                recorded_at=recorded_at,
                authority_version=authority_version,
                authority_type="canonical_registry_default",
                source="canonical_asset_registry",
                verification_state="STATIC_UNCERTIFIED",
                reason_code="CURRENT_STATIC_REGISTRY_DEFAULT",
                lineage={"registry_version": getattr(asset, "registry_version", "")},
            )
            for asset in assets
        ]
        provider_aliases = [
            ProviderAliasRecord(
                permanent_asset_id=str(alias.asset_id),
                alias=str(alias.provider_symbol),
                normalized_alias=normalize_provider_alias(alias.provider_symbol),
                provider=str(alias.provider).lower(),
                venue="",
                effective_from=str(getattr(alias, "valid_from", "") or "1900-01-01T00:00:00Z"),
                effective_to=str(getattr(alias, "valid_to", "") or ""),
                recorded_at=recorded_at,
                authority_version=authority_version,
                authority_type="canonical_registry_default",
                source=str(getattr(alias, "source", "canonical_asset_registry")),
                verification_state="STATIC_UNCERTIFIED",
                reason_code=str(getattr(alias, "mapping_reason", "CURRENT_STATIC_ALIAS")),
                lineage={"registry_version": getattr(alias, "registry_version", "")},
            )
            for alias in aliases
        ]
        return cls(
            permanent_assets=permanent_assets,
            symbol_history=symbol_history,
            provider_aliases=provider_aliases,
            authority_versions=(authority_version,),
        )

    def resolve_symbol(
        self,
        symbol: str,
        *,
        timestamp: str,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
        venue: str | None = None,
    ) -> dict[str, Any]:
        query = normalize_symbol(symbol)
        available, hidden = self._visible_and_hidden(
            self.symbol_history,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda row: row.normalized_symbol == query and _venue_matches(row.venue, venue),
        )
        return self._resolve_records(
            available=available,
            hidden_by_cutoff=hidden,
            timestamp=timestamp,
            query_type="symbol",
            query_value=symbol,
            provider="canonical",
            venue=venue,
            symbol=query,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
        )

    def resolve_provider_alias(
        self,
        provider: str,
        alias: str,
        *,
        timestamp: str,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
        venue: str | None = None,
    ) -> dict[str, Any]:
        provider_key = str(provider or "").strip().lower()
        alias_key = normalize_provider_alias(alias)
        available, hidden = self._visible_and_hidden(
            self.provider_aliases,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda row: row.provider == provider_key
            and row.normalized_alias == alias_key
            and _venue_matches(row.venue, venue),
        )
        return self._resolve_records(
            available=available,
            hidden_by_cutoff=hidden,
            timestamp=timestamp,
            query_type="provider_alias",
            query_value=alias,
            provider=provider_key,
            venue=venue,
            symbol=alias_key,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
        )

    def symbol_for_asset(
        self,
        permanent_asset_id: str,
        *,
        timestamp: str,
        provider: str = "canonical",
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        provider_key = str(provider or "canonical").strip().lower()
        available, hidden = self._visible_and_hidden(
            self.symbol_history,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda row: row.permanent_asset_id == permanent_asset_id and row.provider == provider_key,
        )
        return self._resolve_asset_attribute_records(
            available=available,
            hidden_by_cutoff=hidden,
            timestamp=timestamp,
            query_type="asset_symbol",
            permanent_asset_id=permanent_asset_id,
            provider=provider_key,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            value_field="symbol",
        )

    def provider_alias_for_asset(
        self,
        permanent_asset_id: str,
        *,
        provider: str,
        timestamp: str,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        provider_key = str(provider or "").strip().lower()
        available, hidden = self._visible_and_hidden(
            self.provider_aliases,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda row: row.permanent_asset_id == permanent_asset_id and row.provider == provider_key,
        )
        return self._resolve_asset_attribute_records(
            available=available,
            hidden_by_cutoff=hidden,
            timestamp=timestamp,
            query_type="asset_provider_alias",
            permanent_asset_id=permanent_asset_id,
            provider=provider_key,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            value_field="alias",
        )

    def company_name_for_asset(
        self,
        permanent_asset_id: str,
        *,
        timestamp: str,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        available, hidden = self._visible_and_hidden(
            self.company_names,
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda row: row.permanent_asset_id == permanent_asset_id,
        )
        return self._resolve_asset_attribute_records(
            available=available,
            hidden_by_cutoff=hidden,
            timestamp=timestamp,
            query_type="asset_company_name",
            permanent_asset_id=permanent_asset_id,
            provider="company_name",
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            value_field="name",
        )

    def predecessors(
        self,
        permanent_asset_id: str,
        *,
        timestamp: str | None = None,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        events = self._lineage_events(
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda event: permanent_asset_id in event.successor_asset_ids,
            timestamp=timestamp,
        )
        return self._lineage_result("predecessors", permanent_asset_id, events)

    def successors(
        self,
        permanent_asset_id: str,
        *,
        timestamp: str | None = None,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        events = self._lineage_events(
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda event: permanent_asset_id in event.predecessor_asset_ids,
            timestamp=timestamp,
        )
        return self._lineage_result("successors", permanent_asset_id, events)

    def lineage_traversal(
        self,
        permanent_asset_id: str,
        *,
        direction: str = "successors",
        max_depth: int = 4,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        visited = {permanent_asset_id}
        frontier = [(permanent_asset_id, 0)]
        events: list[CorporateActionLineageEvent] = []
        while frontier:
            asset_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            if direction == "predecessors":
                result = self.predecessors(
                    asset_id,
                    knowledge_cutoff=knowledge_cutoff,
                    authority_version=authority_version,
                )
                next_ids = set(result["predecessor_asset_ids"])
            else:
                result = self.successors(
                    asset_id,
                    knowledge_cutoff=knowledge_cutoff,
                    authority_version=authority_version,
                )
                next_ids = set(result["successor_asset_ids"])
            events.extend(_event_from_dict(row) for row in result["events"])
            for next_id in sorted(next_ids - visited):
                visited.add(next_id)
                frontier.append((next_id, depth + 1))
        unique_events = {event.event_id: event for event in events}
        return {
            "schema_version": HISTORICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
            "query_type": "lineage_traversal",
            "state": RESOLVED if events else UNRESOLVED,
            "permanent_asset_id": permanent_asset_id,
            "direction": direction,
            "asset_ids": sorted(visited),
            "events": [_event_dict(event) for event in sorted(unique_events.values(), key=_event_sort_key)],
            "production_selection_enabled": self.production_selection_enabled,
        }

    def terminal_events(
        self,
        permanent_asset_id: str,
        *,
        timestamp: str | None = None,
        knowledge_cutoff: str | None = None,
        authority_version: str = LATEST_AUTHORITY_VERSION,
    ) -> dict[str, Any]:
        events = self._lineage_events(
            authority_version=authority_version,
            knowledge_cutoff=knowledge_cutoff,
            predicate=lambda event: (
                permanent_asset_id in event.predecessor_asset_ids
                and _is_terminal_for_asset(event, permanent_asset_id)
            ),
            timestamp=timestamp,
        )
        return {
            "schema_version": HISTORICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
            "query_type": "terminal_events",
            "state": RESOLVED if events else UNRESOLVED,
            "permanent_asset_id": permanent_asset_id,
            "events": [_event_dict(event) for event in events],
            "terminal_event_types": [event.event_type for event in events],
            "terminal_economics": [event.terminal_economics for event in events],
            "production_selection_enabled": self.production_selection_enabled,
        }

    def authority_versions_at(
        self,
        *,
        knowledge_cutoff: str | None = None,
    ) -> tuple[str, ...]:
        if knowledge_cutoff is None:
            return self.authority_versions
        cutoff_dt = _parse_datetime(knowledge_cutoff)
        result: list[str] = []
        for version in self.authority_versions:
            records = [
                *self.permanent_assets,
                *self.symbol_history,
                *self.provider_aliases,
                *self.company_names,
                *self.corporate_actions,
            ]
            if any(record.authority_version == version and _parse_datetime(_recorded_at(record)) <= cutoff_dt for record in records):
                result.append(version)
        return tuple(result)

    def validation_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORICAL_IDENTITY_AUTHORITY_SCHEMA_VERSION,
            "authority_versions": list(self.authority_versions),
            "record_counts": {
                "permanent_assets": len(self.permanent_assets),
                "symbol_history": len(self.symbol_history),
                "provider_aliases": len(self.provider_aliases),
                "company_names": len(self.company_names),
                "corporate_actions": len(self.corporate_actions),
            },
            "precedence_policy": [
                {"authority_type": authority_type, "rank": rank}
                for authority_type, rank in PRECEDENCE_POLICY
            ],
            "production_selection_enabled": self.production_selection_enabled,
            "failure_policy": {
                "ambiguous_symbol": AMBIGUOUS,
                "conflicting_alias": CONFLICTING_AUTHORITY,
                "unknown_at_knowledge_cutoff": UNKNOWN_AT_KNOWLEDGE_CUTOFF,
                "outside_effective_window": OUTSIDE_EFFECTIVE_WINDOW,
                "current_ticker_assumption": "lowest_precedence_static_fallback",
                "merger_prices": "never_concatenate_automatically",
            },
            "authority_hash": self.authority_hash(),
        }

    def authority_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORICAL_IDENTITY_AUTHORITY_SCHEMA_VERSION,
            "authority_versions": list(self.authority_versions),
            "production_selection_enabled": self.production_selection_enabled,
            "permanent_assets": [_dataclass_dict(row) for row in self.permanent_assets],
            "symbol_history": [_dataclass_dict(row) for row in self.symbol_history],
            "provider_aliases": [_dataclass_dict(row) for row in self.provider_aliases],
            "company_names": [_dataclass_dict(row) for row in self.company_names],
            "corporate_actions": [_event_dict(row) for row in self.corporate_actions],
        }

    def _resolve_records(
        self,
        *,
        available: Sequence[Any],
        hidden_by_cutoff: Sequence[Any],
        timestamp: str,
        query_type: str,
        query_value: str,
        provider: str,
        venue: str | None,
        symbol: str,
        authority_version: str,
        knowledge_cutoff: str | None,
    ) -> dict[str, Any]:
        event_dt = _parse_datetime(timestamp)
        effective = [row for row in available if _effective_at(row, event_dt)]
        hidden_effective = [row for row in hidden_by_cutoff if _effective_at(row, event_dt)]
        if not effective and hidden_effective:
            return _resolution_record(
                state=UNKNOWN_AT_KNOWLEDGE_CUTOFF,
                query_type=query_type,
                query_value=query_value,
                permanent_asset_id=None,
                symbol=symbol,
                provider=provider,
                venue=venue,
                timestamp=timestamp,
                knowledge_cutoff=knowledge_cutoff,
                authority_version=self._resolved_authority_version(authority_version, knowledge_cutoff),
                records=(),
                candidate_records=hidden_effective,
                reasons=("matching record exists but was recorded after the knowledge cutoff",),
                production_selection_enabled=self.production_selection_enabled,
            )
        if not effective and available:
            return _resolution_record(
                state=OUTSIDE_EFFECTIVE_WINDOW,
                query_type=query_type,
                query_value=query_value,
                permanent_asset_id=None,
                symbol=symbol,
                provider=provider,
                venue=venue,
                timestamp=timestamp,
                knowledge_cutoff=knowledge_cutoff,
                authority_version=self._resolved_authority_version(authority_version, knowledge_cutoff),
                records=(),
                candidate_records=available,
                reasons=("records exist for the query but not at the event timestamp",),
                production_selection_enabled=self.production_selection_enabled,
            )
        if not effective:
            return _resolution_record(
                state=UNRESOLVED,
                query_type=query_type,
                query_value=query_value,
                permanent_asset_id=None,
                symbol=symbol,
                provider=provider,
                venue=venue,
                timestamp=timestamp,
                knowledge_cutoff=knowledge_cutoff,
                authority_version=self._resolved_authority_version(authority_version, knowledge_cutoff),
                records=(),
                candidate_records=(),
                reasons=("no matching historical identity record",),
                production_selection_enabled=self.production_selection_enabled,
            )
        selected_state, selected, reasons = _select_records(effective)
        permanent_asset_id = selected[0].permanent_asset_id if selected_state == RESOLVED else None
        return _resolution_record(
            state=selected_state,
            query_type=query_type,
            query_value=query_value,
            permanent_asset_id=permanent_asset_id,
            symbol=getattr(selected[0], "symbol", getattr(selected[0], "alias", symbol)) if selected else symbol,
            provider=provider,
            venue=venue,
            timestamp=timestamp,
            knowledge_cutoff=knowledge_cutoff,
            authority_version=self._resolved_authority_version(authority_version, knowledge_cutoff),
            records=selected,
            candidate_records=effective,
            reasons=reasons,
            production_selection_enabled=self.production_selection_enabled,
        )

    def _resolve_asset_attribute_records(
        self,
        *,
        available: Sequence[Any],
        hidden_by_cutoff: Sequence[Any],
        timestamp: str,
        query_type: str,
        permanent_asset_id: str,
        provider: str,
        authority_version: str,
        knowledge_cutoff: str | None,
        value_field: str,
    ) -> dict[str, Any]:
        event_dt = _parse_datetime(timestamp)
        effective = [row for row in available if _effective_at(row, event_dt)]
        hidden_effective = [row for row in hidden_by_cutoff if _effective_at(row, event_dt)]
        if not effective and hidden_effective:
            state = UNKNOWN_AT_KNOWLEDGE_CUTOFF
            selected: Sequence[Any] = ()
            reasons = ("matching asset record exists but was recorded after the knowledge cutoff",)
        elif not effective and available:
            state = OUTSIDE_EFFECTIVE_WINDOW
            selected = ()
            reasons = ("asset records exist but not at the event timestamp",)
        elif not effective:
            state = UNRESOLVED
            selected = ()
            reasons = ("no matching asset record",)
        else:
            state, selected, reasons = _select_records(effective, distinct_field=value_field)
        value = getattr(selected[0], value_field) if state == RESOLVED and selected else ""
        return _resolution_record(
            state=state,
            query_type=query_type,
            query_value=permanent_asset_id,
            permanent_asset_id=permanent_asset_id if state == RESOLVED else None,
            symbol=value,
            provider=provider,
            venue=None,
            timestamp=timestamp,
            knowledge_cutoff=knowledge_cutoff,
            authority_version=self._resolved_authority_version(authority_version, knowledge_cutoff),
            records=selected,
            candidate_records=effective or hidden_effective or available,
            reasons=reasons,
            production_selection_enabled=self.production_selection_enabled,
        )

    def _visible_and_hidden(
        self,
        records: Sequence[Any],
        *,
        authority_version: str,
        knowledge_cutoff: str | None,
        predicate: Any,
    ) -> tuple[list[Any], list[Any]]:
        scan_version = self.authority_versions[-1] if authority_version == LATEST_AUTHORITY_VERSION else self._resolved_authority_version(authority_version, knowledge_cutoff)
        version_rank = self._rank_for_version(scan_version)
        cutoff_dt = _parse_datetime(knowledge_cutoff) if knowledge_cutoff else None
        visible: list[Any] = []
        hidden: list[Any] = []
        for record in records:
            if self._rank_for_version(record.authority_version) > version_rank or not predicate(record):
                continue
            if cutoff_dt is not None and _parse_datetime(record.recorded_at) > cutoff_dt:
                hidden.append(record)
            else:
                visible.append(record)
        return visible, hidden

    def _lineage_events(
        self,
        *,
        authority_version: str,
        knowledge_cutoff: str | None,
        predicate: Any,
        timestamp: str | None,
    ) -> list[CorporateActionLineageEvent]:
        version = self._resolved_authority_version(authority_version, knowledge_cutoff)
        rank = self._rank_for_version(version)
        cutoff_dt = _parse_datetime(knowledge_cutoff) if knowledge_cutoff else None
        event_dt = _parse_datetime(timestamp) if timestamp else None
        events = []
        for event in self.corporate_actions:
            if self._rank_for_version(event.authority_version) > rank:
                continue
            if cutoff_dt and _parse_datetime(event.recorded_at) > cutoff_dt:
                continue
            if event_dt and _parse_datetime(event.effective_timestamp) > event_dt:
                continue
            if predicate(event):
                events.append(event)
        return sorted(events, key=_event_sort_key)

    def _lineage_result(
        self,
        query_type: str,
        permanent_asset_id: str,
        events: Sequence[CorporateActionLineageEvent],
    ) -> dict[str, Any]:
        predecessor_ids = sorted({asset for event in events for asset in event.predecessor_asset_ids})
        successor_ids = sorted({asset for event in events for asset in event.successor_asset_ids})
        return {
            "schema_version": HISTORICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
            "query_type": query_type,
            "state": RESOLVED if events else UNRESOLVED,
            "permanent_asset_id": permanent_asset_id,
            "predecessor_asset_ids": predecessor_ids,
            "successor_asset_ids": successor_ids,
            "events": [_event_dict(event) for event in events],
            "production_selection_enabled": self.production_selection_enabled,
        }

    def _resolved_authority_version(self, authority_version: str, knowledge_cutoff: str | None) -> str:
        if authority_version != LATEST_AUTHORITY_VERSION:
            if authority_version not in self._version_rank:
                raise ValueError(f"unknown authority_version: {authority_version}")
            return authority_version
        versions = self.authority_versions_at(knowledge_cutoff=knowledge_cutoff)
        if not versions:
            raise ValueError("no authority versions available at knowledge cutoff")
        return versions[-1]

    def _rank_for_version(self, authority_version: str) -> int:
        try:
            return self._version_rank[authority_version]
        except KeyError as exc:
            raise ValueError(f"unknown authority_version: {authority_version}") from exc

    def _validate(self) -> None:
        if not self.authority_versions:
            raise ValueError("historical identity authority requires at least one authority version")
        duplicate_versions = _duplicates(self.authority_versions)
        if duplicate_versions:
            raise ValueError("duplicate authority versions: " + ", ".join(duplicate_versions))
        asset_ids = [asset.permanent_asset_id for asset in self.permanent_assets]
        duplicate_assets = _duplicates(asset_ids)
        if duplicate_assets:
            raise ValueError("duplicate permanent asset ids: " + ", ".join(duplicate_assets))
        for record in [
            *self.permanent_assets,
            *self.symbol_history,
            *self.provider_aliases,
            *self.company_names,
            *self.corporate_actions,
        ]:
            if record.authority_version not in self._version_rank:
                raise ValueError(f"record references unknown authority_version: {record.authority_version}")
        ticker_like_ids = [
            asset.permanent_asset_id
            for asset in self.permanent_assets
            if asset.permanent_asset_id == normalize_symbol(asset.current_symbol_hint)
        ]
        if ticker_like_ids:
            raise ValueError("permanent_asset_id must not be the current ticker symbol: " + ", ".join(ticker_like_ids))


def load_historical_identity_authority(path: Path | str) -> HistoricalIdentityAuthority:
    return HistoricalIdentityAuthority.from_json(path)


def enrich_pit_universe_result(
    pit_result: Mapping[str, Any],
    authority: HistoricalIdentityAuthority,
    *,
    provider: str = "canonical",
    knowledge_cutoff: str | None = None,
) -> dict[str, Any]:
    decision_timestamp = str(pit_result.get("decision_timestamp") or pit_result.get("decision_date") or "")
    cutoff = knowledge_cutoff if knowledge_cutoff is not None else pit_result.get("knowledge_cutoff")

    def enrich_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for row in rows:
            asset_id = str(row.get("asset_id") or row.get("permanent_asset_id") or "")
            resolution = authority.symbol_for_asset(
                asset_id,
                timestamp=decision_timestamp,
                provider=provider,
                knowledge_cutoff=str(cutoff) if cutoff else None,
            )
            payload = dict(row)
            payload["permanent_asset_id"] = asset_id
            payload["historically_valid_symbol"] = resolution.get("symbol") or row.get("symbol") or ""
            payload["historical_identity_resolution_state"] = resolution["state"]
            payload["historical_identity_authority_version"] = resolution["authority_version"]
            payload["lineage_reference"] = resolution["lineage_references"]
            return_row = payload
            enriched.append(return_row)
        return enriched

    enriched = dict(pit_result)
    for key in ("eligible_assets", "exclusions", "unresolved_assets", "conflicts"):
        enriched[key] = enrich_rows(pit_result.get(key, ()))
    enriched["historical_identity_enrichment"] = {
        "schema_version": PIT_IDENTITY_ENRICHMENT_SCHEMA_VERSION,
        "provider": provider,
        "knowledge_cutoff": str(cutoff) if cutoff else None,
        "authority_hash": authority.authority_hash(),
        "production_selection_enabled": authority.production_selection_enabled,
    }
    return enriched


def stable_resolution_serialization(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", "-").replace(".", "-")


def normalize_provider_alias(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", "-")


def _asset(row: Mapping[str, Any]) -> PermanentAssetIdentity:
    return PermanentAssetIdentity(
        permanent_asset_id=_text(row, "permanent_asset_id"),
        asset_kind=_text(row, "asset_kind", "common_stock"),
        legal_entity_hint=_text(row, "legal_entity_hint"),
        current_symbol_hint=normalize_symbol(row.get("current_symbol_hint")),
        created_at=_text(row, "created_at", "1970-01-01T00:00:00Z"),
        authority_version=_text(row, "authority_version"),
        source=_text(row, "source", "synthetic_fixture"),
        verification_state=_text(row, "verification_state", "VERIFIED"),
        lineage=_mapping(row.get("lineage")),
    )


def _symbol(row: Mapping[str, Any]) -> HistoricalSymbolRecord:
    symbol = _text(row, "symbol")
    return HistoricalSymbolRecord(
        permanent_asset_id=_text(row, "permanent_asset_id"),
        symbol=symbol,
        normalized_symbol=_text(row, "normalized_symbol", normalize_symbol(symbol)),
        provider=_text(row, "provider", "canonical").lower(),
        venue=_text(row, "venue"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        recorded_at=_text(row, "recorded_at"),
        authority_version=_text(row, "authority_version"),
        authority_type=_text(row, "authority_type", "authoritative_symbol_history"),
        source=_text(row, "source", "synthetic_fixture"),
        verification_state=_text(row, "verification_state", "VERIFIED"),
        reason_code=_text(row, "reason_code", "HISTORICAL_SYMBOL"),
        lineage=_mapping(row.get("lineage")),
    )


def _alias(row: Mapping[str, Any]) -> ProviderAliasRecord:
    alias = _text(row, "alias")
    return ProviderAliasRecord(
        permanent_asset_id=_text(row, "permanent_asset_id"),
        alias=alias,
        normalized_alias=_text(row, "normalized_alias", normalize_provider_alias(alias)),
        provider=_text(row, "provider").lower(),
        venue=_text(row, "venue"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        recorded_at=_text(row, "recorded_at"),
        authority_version=_text(row, "authority_version"),
        authority_type=_text(row, "authority_type", "verified_provider_alias"),
        source=_text(row, "source", "synthetic_fixture"),
        verification_state=_text(row, "verification_state", "VERIFIED"),
        reason_code=_text(row, "reason_code", "PROVIDER_ALIAS"),
        lineage=_mapping(row.get("lineage")),
    )


def _name(row: Mapping[str, Any]) -> CompanyNameRecord:
    return CompanyNameRecord(
        permanent_asset_id=_text(row, "permanent_asset_id"),
        name=_text(row, "name"),
        effective_from=_text(row, "effective_from"),
        effective_to=_text(row, "effective_to"),
        recorded_at=_text(row, "recorded_at"),
        authority_version=_text(row, "authority_version"),
        authority_type=_text(row, "authority_type", "manual_verified_identity"),
        source=_text(row, "source", "synthetic_fixture"),
        verification_state=_text(row, "verification_state", "VERIFIED"),
        reason_code=_text(row, "reason_code", "COMPANY_NAME_HISTORY"),
        lineage=_mapping(row.get("lineage")),
    )


def _event(row: Mapping[str, Any]) -> CorporateActionLineageEvent:
    return CorporateActionLineageEvent(
        event_id=_text(row, "event_id"),
        event_type=_text(row, "event_type").lower(),
        predecessor_asset_ids=tuple(str(value) for value in row.get("predecessor_asset_ids", ())),
        successor_asset_ids=tuple(str(value) for value in row.get("successor_asset_ids", ())),
        effective_timestamp=_text(row, "effective_timestamp"),
        recorded_at=_text(row, "recorded_at"),
        source=_text(row, "source", "synthetic_fixture"),
        authority_version=_text(row, "authority_version"),
        authority_type=_text(row, "authority_type", "authoritative_corporate_action_lineage"),
        verification_state=_text(row, "verification_state", "VERIFIED"),
        confidence=_text(row, "confidence", "HIGH"),
        relationship_semantics=_text(row, "relationship_semantics"),
        terminal_economics=_text(row, "terminal_economics", "not_applicable"),
        unresolved_fields=tuple(str(value) for value in row.get("unresolved_fields", ())),
        notes=_text(row, "notes"),
        lineage=_mapping(row.get("lineage")),
    )


def _event_from_dict(row: Mapping[str, Any]) -> CorporateActionLineageEvent:
    return _event(row)


def _resolution_record(
    *,
    state: str,
    query_type: str,
    query_value: str,
    permanent_asset_id: str | None,
    symbol: str,
    provider: str,
    venue: str | None,
    timestamp: str,
    knowledge_cutoff: str | None,
    authority_version: str,
    records: Sequence[Any],
    candidate_records: Sequence[Any],
    reasons: Sequence[str],
    production_selection_enabled: bool,
) -> dict[str, Any]:
    candidate_asset_ids = sorted({row.permanent_asset_id for row in candidate_records})
    payload = {
        "schema_version": HISTORICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
        "state": state,
        "query_type": query_type,
        "query_value": query_value,
        "permanent_asset_id": permanent_asset_id,
        "symbol": symbol,
        "provider": provider,
        "venue": venue,
        "effective_timestamp": _timestamp_text(timestamp),
        "knowledge_cutoff": _timestamp_text(knowledge_cutoff) if knowledge_cutoff else None,
        "authority_version": authority_version,
        "candidate_asset_ids": candidate_asset_ids,
        "matched_records": [_record_summary(row) for row in records],
        "candidate_records": [_record_summary(row) for row in candidate_records],
        "reasons": list(reasons),
        "lineage_references": sorted(
            {
                str(getattr(row, "reason_code", "") or getattr(row, "event_id", ""))
                for row in records
                if str(getattr(row, "reason_code", "") or getattr(row, "event_id", "")).strip()
            }
        ),
        "production_selection_enabled": production_selection_enabled,
    }
    payload["resolution_id"] = "histid_" + hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "resolution_id"},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return payload


def _select_records(
    records: Sequence[Any],
    *,
    distinct_field: str = "permanent_asset_id",
) -> tuple[str, list[Any], tuple[str, ...]]:
    if any(str(row.verification_state).upper() == "CONFLICTING" for row in records):
        return CONFLICTING_AUTHORITY, sorted(records, key=_record_sort_key), (
            "one or more records are marked CONFLICTING",
        )
    best_rank = max(PRECEDENCE_RANK.get(row.authority_type, -1) for row in records)
    top = [row for row in records if PRECEDENCE_RANK.get(row.authority_type, -1) == best_rank]
    values = {getattr(row, distinct_field) for row in top}
    asset_ids = {row.permanent_asset_id for row in top}
    if len(asset_ids) > 1:
        return AMBIGUOUS, sorted(top, key=_record_sort_key), (
            "multiple permanent asset IDs share the highest-precedence mapping",
        )
    if len(values) > 1:
        return CONFLICTING_AUTHORITY, sorted(top, key=_record_sort_key), (
            f"multiple {distinct_field} values share the highest-precedence mapping",
        )
    latest_recorded_at = max(_parse_datetime(row.recorded_at) for row in top)
    latest = [row for row in top if _parse_datetime(row.recorded_at) == latest_recorded_at]
    return RESOLVED, sorted(latest, key=_record_sort_key)[:1], ("resolved by highest-precedence effective record",)


def _record_summary(row: Any) -> dict[str, Any]:
    payload = {
        "record_type": row.__class__.__name__,
        "permanent_asset_id": getattr(row, "permanent_asset_id", ""),
        "authority_type": getattr(row, "authority_type", ""),
        "authority_version": getattr(row, "authority_version", ""),
        "recorded_at": _recorded_at(row),
        "effective_from": getattr(row, "effective_from", getattr(row, "effective_timestamp", "")),
        "effective_to": getattr(row, "effective_to", ""),
        "verification_state": getattr(row, "verification_state", ""),
        "source": getattr(row, "source", ""),
        "reason_code": getattr(row, "reason_code", getattr(row, "event_id", "")),
    }
    for field in ("symbol", "alias", "name", "provider", "venue"):
        if hasattr(row, field):
            payload[field] = getattr(row, field)
    return payload


def _recorded_at(row: Any) -> str:
    return str(getattr(row, "recorded_at", getattr(row, "created_at", "")) or "")


def _event_dict(event: CorporateActionLineageEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["predecessor_asset_ids"] = list(event.predecessor_asset_ids)
    payload["successor_asset_ids"] = list(event.successor_asset_ids)
    payload["unresolved_fields"] = list(event.unresolved_fields)
    return payload


def _dataclass_dict(row: Any) -> dict[str, Any]:
    payload = asdict(row)
    return dict(payload)


def _effective_at(row: Any, timestamp: datetime) -> bool:
    start = _parse_datetime(getattr(row, "effective_from", getattr(row, "effective_timestamp", "")))
    end_text = getattr(row, "effective_to", "")
    if timestamp < start:
        return False
    if end_text and timestamp >= _parse_datetime(end_text):
        return False
    return True


def _is_terminal_for_asset(event: CorporateActionLineageEvent, permanent_asset_id: str) -> bool:
    if event.event_type not in TERMINAL_EVENT_TYPES:
        return False
    if event.event_type == "merger" and permanent_asset_id in event.successor_asset_ids:
        return False
    return permanent_asset_id in event.predecessor_asset_ids


def _venue_matches(record_venue: str, query_venue: str | None) -> bool:
    return query_venue is None or str(record_venue or "").upper() == str(query_venue or "").upper()


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    if len(text) == 10:
        text = text + "T00:00:00Z"
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: Any) -> str:
    parsed = _parse_datetime(value)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    return str(row.get(key, default) or "").strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _record_sort_key(row: Any) -> tuple[Any, ...]:
    return (
        getattr(row, "permanent_asset_id", ""),
        getattr(row, "provider", ""),
        getattr(row, "normalized_symbol", getattr(row, "normalized_alias", "")),
        getattr(row, "effective_from", ""),
        getattr(row, "effective_to", ""),
        getattr(row, "authority_version", ""),
        getattr(row, "recorded_at", ""),
        getattr(row, "reason_code", ""),
    )


def _event_sort_key(event: CorporateActionLineageEvent) -> tuple[Any, ...]:
    return (
        event.effective_timestamp,
        event.event_type,
        event.event_id,
        event.predecessor_asset_ids,
        event.successor_asset_ids,
    )


def _discover_versions(record_groups: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    versions: list[str] = []
    for group in record_groups:
        for record in group:
            if record.authority_version and record.authority_version not in versions:
                versions.append(record.authority_version)
    return tuple(versions)


def _duplicates(values: Sequence[str] | set[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
