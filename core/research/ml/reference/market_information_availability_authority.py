from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence

from infrastructure.data.market_sessions import (
    AFTER_HOURS_SESSION,
    CLOSED_SESSION,
    CONFLICTING_CALENDAR_EVIDENCE_STATUS,
    HALTED_SESSION,
    OUTSIDE_AUTHORITY_RANGE_STATUS,
    PRE_MARKET_SESSION,
    REGULAR_SESSION,
    exchange_session_context,
)
from infrastructure.data.calendar_authority import calendar_contract_payload


MARKET_INFORMATION_AVAILABILITY_AUTHORITY_VERSION = (
    "market_information_availability_authority.v1"
)
AVAILABLE = "AVAILABLE"
NOT_YET_AVAILABLE = "NOT_YET_AVAILABLE"
OUTSIDE_SESSION_POLICY = "OUTSIDE_SESSION_POLICY"
REVISED_AFTER_DECISION = "REVISED_AFTER_DECISION"
UNKNOWN_AVAILABILITY = "UNKNOWN_AVAILABILITY"
CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"

TIME_FIELDS = (
    "source_event_timestamp",
    "provider_published_timestamp",
    "provider_received_timestamp",
    "first_seen_timestamp",
    "revision_timestamp",
    "ingestion_timestamp",
    "earliest_permitted_use",
)
DEFAULT_AVAILABILITY_BASIS_FIELDS = (
    "provider_published_timestamp",
    "provider_received_timestamp",
    "first_seen_timestamp",
    "ingestion_timestamp",
    "revision_timestamp",
    "earliest_permitted_use",
)
ALL_TRADING_SESSIONS = (
    PRE_MARKET_SESSION,
    REGULAR_SESSION,
    AFTER_HOURS_SESSION,
)
ANY_DECISION_SESSION = (
    PRE_MARKET_SESSION,
    REGULAR_SESSION,
    AFTER_HOURS_SESSION,
    CLOSED_SESSION,
    HALTED_SESSION,
)


@dataclass(frozen=True)
class AvailabilitySessionPolicy:
    policy_id: str = "us_equity_all_trading_sessions_v1"
    allowed_sessions: tuple[str, ...] = ALL_TRADING_SESSIONS
    require_trading_day: bool = True
    allow_halted: bool = False
    description: str = (
        "Decision timestamp must fall in a represented US equity trading "
        "session unless a surface-specific legacy policy says otherwise."
    )


@dataclass(frozen=True)
class MarketInformationEvent:
    source_kind: str
    decision_timestamp: Any
    source_event_timestamp: Any = None
    provider_published_timestamp: Any = None
    provider_received_timestamp: Any = None
    first_seen_timestamp: Any = None
    revision_timestamp: Any = None
    ingestion_timestamp: Any = None
    earliest_permitted_use: Any = None
    exchange: str = "XNYS"
    timezone: str = "America/New_York"
    trading_session: str = ""
    source_version: str = ""
    correction_lineage: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None
    required_knowledge_fields: tuple[str, ...] = ()
    availability_basis_fields: tuple[str, ...] = DEFAULT_AVAILABILITY_BASIS_FIELDS
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AvailabilityResolution:
    authority_version: str
    status: str
    available: bool
    usable_for_promotion: bool
    decision_timestamp: str | None
    earliest_permitted_use: str | None
    source_kind: str
    source_version: str
    exchange: str
    timezone: str
    trading_session: str
    session_date: str | None
    session_policy: Mapping[str, Any]
    calendar_identity: str
    calendar_authority_version: str
    calendar_authority_version_identity: str
    calendar_source_status: str
    calendar_base_status: str
    calendar_package: str
    calendar_package_version: str
    calendar_schedule_hash: str
    calendar_fallback_used: bool
    calendar_closure_reason: str
    source_event_timestamp: str | None
    provider_published_timestamp: str | None
    provider_received_timestamp: str | None
    first_seen_timestamp: str | None
    revision_timestamp: str | None
    ingestion_timestamp: str | None
    correction_lineage: Any
    reason_codes: tuple[str, ...]
    evidence_conflicts: tuple[str, ...]
    resolution_id: str = ""

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        payload["evidence_conflicts"] = list(self.evidence_conflicts)
        return payload


class MarketInformationAvailabilityAuthority:
    """Single PIT authority for market-information availability.

    Event time and knowledge time are intentionally separate. A period end,
    filing date, or article topic date is never enough by itself to prove that
    information was available at a decision timestamp.
    """

    def __init__(
        self,
        *,
        authority_version: str = MARKET_INFORMATION_AVAILABILITY_AUTHORITY_VERSION,
        session_policy: AvailabilitySessionPolicy | None = None,
        market_halts: Sequence[Any] = (),
        promotion_grade: bool = True,
    ) -> None:
        self.authority_version = authority_version
        self.session_policy = session_policy or AvailabilitySessionPolicy()
        self.market_halts = tuple(market_halts)
        self.promotion_grade = bool(promotion_grade)

    def evaluate(
        self,
        event: MarketInformationEvent | Mapping[str, Any],
        *,
        session_policy: AvailabilitySessionPolicy | None = None,
        promotion_grade: bool | None = None,
    ) -> dict[str, Any]:
        resolved_event = event if isinstance(event, MarketInformationEvent) else event_from_mapping(event)
        policy = session_policy or self.session_policy
        promotion = self.promotion_grade if promotion_grade is None else bool(promotion_grade)
        parsed, parse_errors = _parse_event_times(resolved_event)
        decision = parsed.get("decision_timestamp")
        if decision is None:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=UNKNOWN_AVAILABILITY,
                reason_codes=(*parse_errors, "decision_timestamp_missing_or_invalid"),
                promotion_grade=promotion,
            )
        context = exchange_session_context(
            decision,
            exchange=resolved_event.exchange,
            market_halts=self.market_halts,
        )
        session_reasons = _session_policy_reasons(context, policy)
        if session_reasons:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=OUTSIDE_SESSION_POLICY,
                reason_codes=session_reasons,
                promotion_grade=promotion,
                context=context,
            )
        if parse_errors:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=UNKNOWN_AVAILABILITY,
                reason_codes=parse_errors,
                promotion_grade=promotion,
                context=context,
            )
        conflicts = _evidence_conflicts(parsed, resolved_event.availability_basis_fields)
        if conflicts:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=CONFLICTING_EVIDENCE,
                reason_codes=("conflicting_availability_evidence",),
                evidence_conflicts=conflicts,
                promotion_grade=promotion,
                context=context,
            )
        missing_required = tuple(
            field
            for field in resolved_event.required_knowledge_fields
            if parsed.get(field) is None
        )
        if missing_required:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=UNKNOWN_AVAILABILITY,
                reason_codes=tuple(f"{field}_missing" for field in missing_required),
                promotion_grade=promotion,
                context=context,
            )
        basis_fields = resolved_event.availability_basis_fields or DEFAULT_AVAILABILITY_BASIS_FIELDS
        knowledge_times = [
            parsed[field]
            for field in basis_fields
            if field in TIME_FIELDS and parsed.get(field) is not None
        ]
        if not knowledge_times:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=UNKNOWN_AVAILABILITY,
                reason_codes=("no_knowledge_timestamp",),
                promotion_grade=promotion,
                context=context,
            )
        revision = parsed.get("revision_timestamp")
        if revision is not None and revision > decision:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=REVISED_AFTER_DECISION,
                reason_codes=("revision_timestamp_after_decision",),
                promotion_grade=promotion,
                context=context,
            )
        parsed = {**parsed, "earliest_permitted_use": max(knowledge_times)}
        if parsed["earliest_permitted_use"] > decision:
            return self._resolution(
                event=resolved_event,
                policy=policy,
                parsed=parsed,
                status=NOT_YET_AVAILABLE,
                reason_codes=("earliest_permitted_use_after_decision",),
                promotion_grade=promotion,
                context=context,
            )
        return self._resolution(
            event=resolved_event,
            policy=policy,
            parsed=parsed,
            status=AVAILABLE,
            reason_codes=("earliest_permitted_use_lte_decision",),
            promotion_grade=promotion,
            context=context,
        )

    def validation_artifact(self) -> dict[str, Any]:
        return {
            "authority_version": self.authority_version,
            "contract": "market_information_availability_authority",
            "event_time_and_knowledge_time_distinct": True,
            "unknown_fails_closed_for_promotion_grade": self.promotion_grade,
            "calendar_source": "infrastructure.data.market_sessions",
            "calendar_policy": "route session semantics through infrastructure.data.calendar_authority with explicit fallback status",
            "calendar_contract": calendar_contract_payload(),
            "supported_statuses": [
                AVAILABLE,
                NOT_YET_AVAILABLE,
                OUTSIDE_SESSION_POLICY,
                REVISED_AFTER_DECISION,
                UNKNOWN_AVAILABILITY,
                CONFLICTING_EVIDENCE,
            ],
            "default_session_policy": _session_policy_payload(self.session_policy),
        }

    def _resolution(
        self,
        *,
        event: MarketInformationEvent,
        policy: AvailabilitySessionPolicy,
        parsed: Mapping[str, datetime | None],
        status: str,
        reason_codes: Sequence[str],
        promotion_grade: bool,
        evidence_conflicts: Sequence[str] = (),
        context: Any | None = None,
    ) -> dict[str, Any]:
        decision = parsed.get("decision_timestamp")
        if context is None and decision is not None:
            context = exchange_session_context(
                decision,
                exchange=event.exchange,
                market_halts=self.market_halts,
            )
        trading_session = (
            context.trading_session
            if context is not None
            else event.trading_session or ""
        )
        timezone_name = context.timezone if context is not None else event.timezone
        exchange = context.exchange if context is not None else event.exchange
        calendar_identity = context.calendar_identity if context is not None else ""
        calendar_authority_version = (
            context.calendar_authority_version if context is not None else ""
        )
        calendar_authority_version_identity = (
            context.calendar_authority_version_identity if context is not None else ""
        )
        calendar_source_status = context.calendar_source_status if context is not None else ""
        calendar_base_status = context.calendar_base_status if context is not None else ""
        calendar_package = context.calendar_package if context is not None else ""
        calendar_package_version = context.calendar_package_version if context is not None else ""
        calendar_schedule_hash = context.calendar_schedule_hash if context is not None else ""
        calendar_fallback_used = context.calendar_fallback_used if context is not None else False
        calendar_closure_reason = context.calendar_closure_reason if context is not None else ""
        session_date = context.session_date.isoformat() if context is not None else None
        resolution = AvailabilityResolution(
            authority_version=self.authority_version,
            status=status,
            available=status == AVAILABLE,
            usable_for_promotion=status == AVAILABLE if promotion_grade else status == AVAILABLE,
            decision_timestamp=_format_timestamp(decision),
            earliest_permitted_use=_format_timestamp(parsed.get("earliest_permitted_use")),
            source_kind=event.source_kind,
            source_version=event.source_version,
            exchange=exchange,
            timezone=timezone_name,
            trading_session=trading_session,
            session_date=session_date,
            session_policy=_session_policy_payload(policy),
            calendar_identity=calendar_identity,
            calendar_authority_version=calendar_authority_version,
            calendar_authority_version_identity=calendar_authority_version_identity,
            calendar_source_status=calendar_source_status,
            calendar_base_status=calendar_base_status,
            calendar_package=calendar_package,
            calendar_package_version=calendar_package_version,
            calendar_schedule_hash=calendar_schedule_hash,
            calendar_fallback_used=calendar_fallback_used,
            calendar_closure_reason=calendar_closure_reason,
            source_event_timestamp=_format_timestamp(parsed.get("source_event_timestamp")),
            provider_published_timestamp=_format_timestamp(parsed.get("provider_published_timestamp")),
            provider_received_timestamp=_format_timestamp(parsed.get("provider_received_timestamp")),
            first_seen_timestamp=_format_timestamp(parsed.get("first_seen_timestamp")),
            revision_timestamp=_format_timestamp(parsed.get("revision_timestamp")),
            ingestion_timestamp=_format_timestamp(parsed.get("ingestion_timestamp")),
            correction_lineage=_json_ready(event.correction_lineage or []),
            reason_codes=tuple(str(value) for value in reason_codes),
            evidence_conflicts=tuple(str(value) for value in evidence_conflicts),
        )
        payload = resolution.payload()
        payload["resolution_id"] = "mia_" + hashlib.sha256(
            json.dumps(
                {key: value for key, value in payload.items() if key != "resolution_id"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:24]
        return payload


def event_from_mapping(row: Mapping[str, Any]) -> MarketInformationEvent:
    values = dict(row)
    source_kind = str(values.pop("source_kind", values.pop("surface", "unknown")) or "unknown")
    decision = values.pop("decision_timestamp", None)
    return MarketInformationEvent(
        source_kind=source_kind,
        decision_timestamp=decision,
        source_event_timestamp=values.pop("source_event_timestamp", None),
        provider_published_timestamp=values.pop("provider_published_timestamp", None),
        provider_received_timestamp=values.pop("provider_received_timestamp", None),
        first_seen_timestamp=values.pop("first_seen_timestamp", None),
        revision_timestamp=values.pop("revision_timestamp", None),
        ingestion_timestamp=values.pop("ingestion_timestamp", None),
        earliest_permitted_use=values.pop("earliest_permitted_use", None),
        exchange=str(values.pop("exchange", "XNYS") or "XNYS"),
        timezone=str(values.pop("timezone", "America/New_York") or "America/New_York"),
        trading_session=str(values.pop("trading_session", "") or ""),
        source_version=str(values.pop("source_version", "") or ""),
        correction_lineage=values.pop("correction_lineage", None),
        required_knowledge_fields=tuple(values.pop("required_knowledge_fields", ()) or ()),
        availability_basis_fields=tuple(
            values.pop("availability_basis_fields", DEFAULT_AVAILABILITY_BASIS_FIELDS)
            or DEFAULT_AVAILABILITY_BASIS_FIELDS
        ),
        metadata=values,
    )


def default_market_information_authority() -> MarketInformationAvailabilityAuthority:
    return MarketInformationAvailabilityAuthority()


def any_timestamp_session_policy(policy_id: str = "legacy_any_timestamp_pit_policy_v1") -> AvailabilitySessionPolicy:
    return AvailabilitySessionPolicy(
        policy_id=policy_id,
        allowed_sessions=ANY_DECISION_SESSION,
        require_trading_day=False,
        allow_halted=True,
        description=(
            "Legacy date-based research rows may use midnight UTC decision "
            "cutoffs; availability still requires knowledge timestamps."
        ),
    )


def sec_filing_event_from_row(
    row: Mapping[str, Any],
    *,
    decision_timestamp: Any,
    source_kind: str = "sec_filing",
) -> MarketInformationEvent:
    accepted = (
        row.get("acceptance_timestamp")
        or row.get("accepted_datetime")
        or row.get("acceptanceDateTime")
    )
    filing_date = row.get("filing_timestamp") or row.get("filing_date") or row.get("filed")
    provider_published = accepted or filing_date or row.get("published_at_utc")
    return MarketInformationEvent(
        source_kind=source_kind,
        decision_timestamp=decision_timestamp,
        source_event_timestamp=row.get("report_date") or row.get("period_end") or filing_date,
        provider_published_timestamp=provider_published,
        provider_received_timestamp=row.get("provider_received_timestamp"),
        first_seen_timestamp=row.get("provider_first_seen_timestamp"),
        revision_timestamp=row.get("revision_timestamp"),
        ingestion_timestamp=row.get("ingestion_timestamp"),
        earliest_permitted_use=row.get("earliest_permitted_use") or row.get("available_timestamp"),
        exchange=str(row.get("exchange") or "XNYS"),
        source_version=str(row.get("source_version") or row.get("form_type") or row.get("source_document_id") or ""),
        correction_lineage=row.get("correction_lineage") or _correction_from_row(row),
        required_knowledge_fields=("provider_published_timestamp",),
        availability_basis_fields=(
            "provider_published_timestamp",
            "provider_received_timestamp",
            "earliest_permitted_use",
            "revision_timestamp",
        ),
        metadata={"source_row": _compact_row(row)},
    )


def daily_price_feature_event(
    *,
    observation_timestamp: Any,
    decision_timestamp: Any,
    exchange: str = "XNYS",
    source_version: str = "daily_price_feature_cutoff_v1",
) -> MarketInformationEvent:
    return MarketInformationEvent(
        source_kind="daily_price_feature",
        decision_timestamp=decision_timestamp,
        source_event_timestamp=observation_timestamp,
        provider_published_timestamp=observation_timestamp,
        earliest_permitted_use=observation_timestamp,
        exchange=exchange,
        source_version=source_version,
        required_knowledge_fields=("provider_published_timestamp",),
        availability_basis_fields=("provider_published_timestamp", "earliest_permitted_use"),
    )


def fundamental_fact_event_from_row(
    row: Mapping[str, Any],
    *,
    decision_timestamp: Any,
) -> MarketInformationEvent:
    return sec_filing_event_from_row(
        {
            **dict(row),
            "filing_date": row.get("filing_timestamp") or row.get("available_timestamp"),
            "accepted_datetime": row.get("acceptance_timestamp"),
            "source_version": row.get("source_document_id") or row.get("filing_accession"),
        },
        decision_timestamp=decision_timestamp,
        source_kind="fundamental_feature_source",
    )


def availability_result_is_available(result: Mapping[str, Any]) -> bool:
    return str(result.get("status") or "") == AVAILABLE


def _parse_event_times(event: MarketInformationEvent) -> tuple[dict[str, datetime | None], tuple[str, ...]]:
    parsed: dict[str, datetime | None] = {}
    errors = []
    for field_name in ("decision_timestamp", *TIME_FIELDS):
        if field_name == "decision_timestamp":
            raw = event.decision_timestamp
            date_policy = "start"
        else:
            raw = getattr(event, field_name)
            date_policy = "end"
        try:
            parsed[field_name] = _coerce_timestamp(raw, date_policy=date_policy)
        except ValueError:
            parsed[field_name] = None
            errors.append(f"{field_name}_malformed")
    return parsed, tuple(errors)


def _coerce_timestamp(value: Any, *, date_policy: str) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(
            value,
            time(23, 59, 59) if date_policy == "end" else time(0, 0, 0),
            tzinfo=timezone.utc,
        )
    else:
        text = str(value).strip()
        if not text:
            return None
        if len(text) == 10:
            parsed = datetime.fromisoformat(text).replace(
                hour=23 if date_policy == "end" else 0,
                minute=59 if date_policy == "end" else 0,
                second=59 if date_policy == "end" else 0,
                tzinfo=timezone.utc,
            )
        elif len(text) >= 14 and text[:14].isdigit():
            parsed = datetime.strptime(text[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _evidence_conflicts(
    parsed: Mapping[str, datetime | None],
    basis_fields: Sequence[str],
) -> tuple[str, ...]:
    conflicts = []
    published = parsed.get("provider_published_timestamp")
    for field_name in (
        "provider_received_timestamp",
        "first_seen_timestamp",
        "ingestion_timestamp",
        "earliest_permitted_use",
    ):
        value = parsed.get(field_name)
        if (
            published is not None
            and value is not None
            and field_name in basis_fields
            and value < published
        ):
            conflicts.append(f"{field_name}_before_provider_published_timestamp")
    source_event = parsed.get("source_event_timestamp")
    if (
        source_event is not None
        and published is not None
        and source_event > published
        and "source_event_timestamp" in basis_fields
    ):
        conflicts.append("source_event_timestamp_after_provider_published_timestamp")
    return tuple(conflicts)


def _session_policy_reasons(context: Any, policy: AvailabilitySessionPolicy) -> tuple[str, ...]:
    reasons = []
    if getattr(context, "calendar_source_status", "") == OUTSIDE_AUTHORITY_RANGE_STATUS:
        reasons.append("calendar_authority_outside_range")
    if getattr(context, "calendar_base_status", "") == OUTSIDE_AUTHORITY_RANGE_STATUS:
        reasons.append("calendar_authority_outside_range")
    if getattr(context, "calendar_source_status", "") == CONFLICTING_CALENDAR_EVIDENCE_STATUS:
        reasons.append("calendar_authority_conflicting_evidence")
    if getattr(context, "calendar_base_status", "") == CONFLICTING_CALENDAR_EVIDENCE_STATUS:
        reasons.append("calendar_authority_conflicting_evidence")
    if policy.require_trading_day and not context.is_trading_day:
        reasons.append("decision_timestamp_not_trading_day")
    if context.halted and not policy.allow_halted:
        reasons.append("decision_timestamp_in_represented_market_halt")
    if context.trading_session not in set(policy.allowed_sessions):
        reasons.append(f"decision_session_{context.trading_session}_not_allowed")
    return tuple(dict.fromkeys(reasons))


def _session_policy_payload(policy: AvailabilitySessionPolicy) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "allowed_sessions": list(policy.allowed_sessions),
        "require_trading_day": policy.require_trading_day,
        "allow_halted": policy.allow_halted,
        "description": policy.description,
    }


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _correction_from_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    lineage = []
    if row.get("is_amendment") or str(row.get("form_type") or "").endswith("/A"):
        lineage.append(
            {
                "kind": "amendment",
                "source_document_id": row.get("source_document_id") or row.get("filing_accession") or "",
                "amends_document_id": row.get("amends_document_id") or "",
            }
        )
    return lineage


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "article_id",
        "symbol",
        "provider",
        "source",
        "source_type",
        "form_type",
        "filing_accession",
        "accession_number",
        "source_document_id",
        "published_at_source",
        "timestamp_precision",
    )
    return {key: row.get(key) for key in keys if key in row}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (date, datetime)):
        return str(value)
    return value
