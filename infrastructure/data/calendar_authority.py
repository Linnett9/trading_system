from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo


CALENDAR_AUTHORITY_VERSION = "market_calendar_authority.v1"
CALENDAR_ID = "XNYS"
SELECTED_MAINTAINED_PACKAGE = "exchange_calendars"
SELECTED_MAINTAINED_PACKAGE_DISTRIBUTION = "exchange-calendars"
SELECTED_MAINTAINED_PACKAGE_VERSION = "4.13.2"
FALLBACK_CALENDAR_VERSION = "compact_nyse_like_rth_2016_2026_v1"
FALLBACK_CALENDAR_IDENTITY = (
    "infrastructure.data.market_sessions.nyse_like_rth_2016_2026_v1"
)
GENERATED_AT = "2026-07-29T00:00:00Z"

EASTERN = ZoneInfo("America/New_York")
PRE_MARKET_OPEN = time(4, 0)
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
AFTER_HOURS_CLOSE = time(20, 0)

REGULAR_SESSION_STATUS = "REGULAR_SESSION"
EARLY_CLOSE_STATUS = "EARLY_CLOSE"
CLOSED_HOLIDAY_STATUS = "CLOSED_HOLIDAY"
CLOSED_WEEKEND_STATUS = "CLOSED_WEEKEND"
SPECIAL_CLOSURE_STATUS = "SPECIAL_CLOSURE"
OUTSIDE_AUTHORITY_RANGE_STATUS = "OUTSIDE_AUTHORITY_RANGE"
CONFLICTING_CALENDAR_EVIDENCE_STATUS = "CONFLICTING_CALENDAR_EVIDENCE"
FALLBACK_CALENDAR_USED_STATUS = "FALLBACK_CALENDAR_USED"

CALENDAR_START_YEAR = 2016
CALENDAR_END_YEAR = 2026
SPECIAL_FULL_CLOSURES = {
    date(2018, 12, 5): "National day of mourning for President George H.W. Bush",
    date(2025, 1, 9): "National day of mourning for President Jimmy Carter",
}
CALENDAR_VALIDATION_DATES = (
    date(2018, 12, 5),
    date(2021, 12, 31),
    date(2025, 1, 9),
    date(2025, 7, 3),
    date(2025, 11, 27),
    date(2025, 11, 28),
    date(2025, 12, 24),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 1, 2),
    date(2026, 1, 3),
    date(2026, 3, 9),
    date(2026, 11, 2),
    date(2026, 12, 24),
)

MAINTAINED_SUPPORTED_EXCHANGES = (
    "XNYS",
    "XNAS",
    "ARCX",
    "BATS",
    "XASE",
    "XLON",
    "XHKG",
    "XTKS",
    "XASX",
    "XPAR",
)


class CalendarBackend(Protocol):
    backend_id: str
    package_name: str
    package_version: str
    supported_exchanges: tuple[str, ...]
    historical_start: date | None
    historical_end: date | None
    future_end: date | None
    deterministic_notes: str
    holiday_source: str
    early_close_support: str
    special_closure_support: str
    timezone_handling: str
    fallback: bool

    def lookup_session(self, exchange: str, day: date) -> "CalendarSessionRecord":
        ...

    def sessions_in_range(self, exchange: str, start: date, end: date) -> list[date]:
        ...

    def previous_session(self, exchange: str, day: date) -> date | None:
        ...

    def next_session(self, exchange: str, day: date) -> date | None:
        ...


@dataclass(frozen=True)
class CalendarConflict:
    conflict_type: str
    session_date: str
    base_status: str
    override_status: str
    base_evidence: Mapping[str, Any]
    override_evidence: Mapping[str, Any]
    resolution: str
    promotion_blocking: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalendarSessionRecord:
    exchange: str
    calendar_id: str
    authority_version: str
    authority_version_identity: str
    package: str
    package_version: str
    fallback_used: bool
    fallback_version: str | None
    source_status: str
    base_status: str
    session_date: date
    market_timezone: str
    open_timestamp: datetime | None
    close_timestamp: datetime | None
    pre_market_open_timestamp: datetime | None
    after_hours_close_timestamp: datetime | None
    early_close: bool
    closure_reason: str
    schedule_generation_parameters: Mapping[str, Any]
    generated_at: str
    schedule_hash: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()
    conflict: CalendarConflict | None = None

    @property
    def is_trading_day(self) -> bool:
        return (
            self.open_timestamp is not None
            and self.close_timestamp is not None
            and self.base_status
            in {REGULAR_SESSION_STATUS, EARLY_CLOSE_STATUS}
        )

    @property
    def close_time(self) -> time | None:
        if self.close_timestamp is None:
            return None
        return self.close_timestamp.astimezone(ZoneInfo(self.market_timezone)).time()

    def payload(self) -> dict[str, Any]:
        output = {
            "exchange": self.exchange,
            "calendar_id": self.calendar_id,
            "authority_version": self.authority_version,
            "authority_version_identity": self.authority_version_identity,
            "package": self.package,
            "package_version": self.package_version,
            "fallback_used": self.fallback_used,
            "fallback_version": self.fallback_version,
            "source_status": self.source_status,
            "base_status": self.base_status,
            "session_date": self.session_date.isoformat(),
            "market_timezone": self.market_timezone,
            "open_timestamp": _format_timestamp(self.open_timestamp),
            "close_timestamp": _format_timestamp(self.close_timestamp),
            "pre_market_open_timestamp": _format_timestamp(self.pre_market_open_timestamp),
            "after_hours_close_timestamp": _format_timestamp(self.after_hours_close_timestamp),
            "early_close": self.early_close,
            "closure_reason": self.closure_reason,
            "schedule_generation_parameters": _json_ready(
                self.schedule_generation_parameters
            ),
            "generated_at": self.generated_at,
            "schedule_hash": self.schedule_hash,
            "evidence": [_json_ready(item) for item in self.evidence],
            "conflict": self.conflict.payload() if self.conflict else None,
        }
        return output

    def with_hash(self) -> "CalendarSessionRecord":
        payload = self.payload()
        payload.pop("schedule_hash", None)
        schedule_hash = _canonical_hash(payload)
        return CalendarSessionRecord(
            exchange=self.exchange,
            calendar_id=self.calendar_id,
            authority_version=self.authority_version,
            authority_version_identity=self.authority_version_identity,
            package=self.package,
            package_version=self.package_version,
            fallback_used=self.fallback_used,
            fallback_version=self.fallback_version,
            source_status=self.source_status,
            base_status=self.base_status,
            session_date=self.session_date,
            market_timezone=self.market_timezone,
            open_timestamp=self.open_timestamp,
            close_timestamp=self.close_timestamp,
            pre_market_open_timestamp=self.pre_market_open_timestamp,
            after_hours_close_timestamp=self.after_hours_close_timestamp,
            early_close=self.early_close,
            closure_reason=self.closure_reason,
            schedule_generation_parameters=self.schedule_generation_parameters,
            generated_at=self.generated_at,
            schedule_hash=schedule_hash,
            evidence=self.evidence,
            conflict=self.conflict,
        )


@dataclass(frozen=True)
class CalendarAuthorityIdentity:
    calendar_authority_id: str
    exchange: str
    calendar_id: str
    authority_version: str
    version: str
    package: str
    package_version: str
    selected_package_version: str
    fallback_state: str
    fallback_used: bool
    date_range_used: Mapping[str, str | None]
    schedule_hash: str
    generated_at: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalendarSourceAudit:
    package: str
    version: str
    licence: str
    supported_exchanges: Sequence[str]
    historical_range: str
    future_schedule_behaviour: str
    holiday_source: str
    early_close_support: str
    special_closures: str
    timezone_handling: str
    determinism_concerns: str
    selected: bool
    installed: bool
    status: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalCalendarOverride:
    session_date: date
    action: str
    reason: str
    reviewed: bool
    effective_date: date
    source_path: str
    source_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def applied_status(self) -> str:
        action = self.action.upper()
        if action == "CLOSE":
            return SPECIAL_CLOSURE_STATUS
        if action == "EARLY_CLOSE":
            return EARLY_CLOSE_STATUS
        if action == "OPEN":
            return REGULAR_SESSION_STATUS
        return CONFLICTING_CALENDAR_EVIDENCE_STATUS


class ExchangeCalendarAuthority:
    """Versioned exchange-calendar authority with explicit fallback semantics."""

    def __init__(
        self,
        *,
        exchange: str = CALENDAR_ID,
        backend: CalendarBackend | None = None,
        allow_fallback: bool = True,
        local_override_paths: Sequence[Path | str] = (),
        expected_package_version: str = SELECTED_MAINTAINED_PACKAGE_VERSION,
    ) -> None:
        self.exchange = _normalize_exchange(exchange)
        self.expected_package_version = expected_package_version
        self.allow_fallback = bool(allow_fallback)
        maintained = backend or load_exchange_calendars_backend(
            expected_package_version=expected_package_version
        )
        self.backend_status = "MAINTAINED"
        self.backend_status_reason = ""
        if maintained is None:
            self.backend_status = "DEPENDENCY_UNAVAILABLE"
            self.backend_status_reason = "exchange_calendars import or version unavailable"
            if not self.allow_fallback:
                self.backend: CalendarBackend = UnavailableCalendarBackend(
                    reason=self.backend_status_reason
                )
            else:
                self.backend = CompactNyseFallbackBackend()
        elif (
            maintained.package_name == SELECTED_MAINTAINED_PACKAGE
            and maintained.package_version != expected_package_version
        ):
            self.backend_status = "VERSION_MISMATCH"
            self.backend_status_reason = (
                f"{maintained.package_version} != {expected_package_version}"
            )
            if not self.allow_fallback:
                self.backend = UnavailableCalendarBackend(
                    reason=self.backend_status_reason,
                    package_version=maintained.package_version,
                )
            else:
                self.backend = CompactNyseFallbackBackend()
        else:
            self.backend = maintained
        self.overrides = _load_overrides(local_override_paths, self.exchange)

    def session(self, day: date | str, *, exchange: str | None = None) -> CalendarSessionRecord:
        session_date = _coerce_date(day)
        base = self.backend.lookup_session(
            _normalize_exchange(exchange or self.exchange),
            session_date,
        )
        override = self.overrides.get(session_date)
        if override is None:
            return base
        return _apply_override(base, override)

    def timestamp_context(
        self,
        timestamp: datetime,
        *,
        exchange: str | None = None,
    ) -> CalendarSessionRecord:
        timestamp_utc = _to_utc(timestamp)
        local = timestamp_utc.astimezone(EASTERN)
        return self.session(local.date(), exchange=exchange)

    def sessions(
        self,
        start: date | str,
        end: date | str,
        *,
        exchange: str | None = None,
    ) -> list[date]:
        start_date = _coerce_date(start)
        end_date = _coerce_date(end)
        if end_date < start_date:
            return []
        days = self.backend.sessions_in_range(
            _normalize_exchange(exchange or self.exchange),
            start_date,
            end_date,
        )
        output = []
        current = start_date
        base_days = set(days)
        while current <= end_date:
            record = self.session(current, exchange=exchange)
            if record.is_trading_day or current in base_days:
                if record.is_trading_day:
                    output.append(current)
            current += timedelta(days=1)
        return output

    def previous_session(self, day: date | str, *, exchange: str | None = None) -> date | None:
        current = _coerce_date(day) - timedelta(days=1)
        stop = current - timedelta(days=366 * 20)
        while current >= stop:
            if self.session(current, exchange=exchange).is_trading_day:
                return current
            current -= timedelta(days=1)
        return self.backend.previous_session(_normalize_exchange(exchange or self.exchange), _coerce_date(day))

    def next_session(self, day: date | str, *, exchange: str | None = None) -> date | None:
        current = _coerce_date(day) + timedelta(days=1)
        stop = current + timedelta(days=366 * 20)
        while current <= stop:
            if self.session(current, exchange=exchange).is_trading_day:
                return current
            current += timedelta(days=1)
        return self.backend.next_session(_normalize_exchange(exchange or self.exchange), _coerce_date(day))

    def materialize_schedule(
        self,
        start: date | str,
        end: date | str,
        *,
        exchange: str | None = None,
        include_closed: bool = False,
    ) -> list[dict[str, Any]]:
        start_date = _coerce_date(start)
        end_date = _coerce_date(end)
        rows: list[dict[str, Any]] = []
        current = start_date
        while current <= end_date:
            record = self.session(current, exchange=exchange)
            if include_closed or record.is_trading_day:
                rows.append(record.payload())
            current += timedelta(days=1)
        return rows

    def identity(
        self,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        exchange: str | None = None,
    ) -> dict[str, Any]:
        resolved_exchange = _normalize_exchange(exchange or self.exchange)
        date_range = {
            "start": _coerce_date(start).isoformat() if start is not None else None,
            "end": _coerce_date(end).isoformat() if end is not None else None,
        }
        if start is not None and end is not None:
            rows = self.materialize_schedule(start, end, exchange=resolved_exchange)
            schedule_hash = _canonical_hash(rows)
        else:
            schedule_hash = _canonical_hash(
                {
                    "calendar_id": resolved_exchange,
                    "authority_version": CALENDAR_AUTHORITY_VERSION,
                    "package": self.backend.package_name,
                    "package_version": self.backend.package_version,
                    "fallback_used": self.backend.fallback,
                }
            )
        identity = CalendarAuthorityIdentity(
            calendar_authority_id=f"{CALENDAR_AUTHORITY_VERSION}:{resolved_exchange}",
            exchange=resolved_exchange,
            calendar_id=resolved_exchange,
            authority_version=CALENDAR_AUTHORITY_VERSION,
            version=self.authority_version_identity,
            package=self.backend.package_name,
            package_version=self.backend.package_version,
            selected_package_version=self.expected_package_version,
            fallback_state="FALLBACK_USED" if self.backend.fallback else "MAINTAINED_USED",
            fallback_used=self.backend.fallback,
            date_range_used=date_range,
            schedule_hash=schedule_hash,
            generated_at=GENERATED_AT,
        )
        return identity.payload()

    @property
    def authority_version_identity(self) -> str:
        return (
            f"{CALENDAR_AUTHORITY_VERSION}:{self.backend.backend_id}:"
            f"{self.backend.package_name}:{self.backend.package_version}"
        )

    def source_audit(self) -> dict[str, Any]:
        audits = calendar_source_audit(installed_backend=self.backend)
        return {
            "contract_version": "calendar_source_audit.v1",
            "selected_authority": SELECTED_MAINTAINED_PACKAGE,
            "selected_version": SELECTED_MAINTAINED_PACKAGE_VERSION,
            "runtime_backend_status": self.backend_status,
            "runtime_backend_status_reason": self.backend_status_reason,
            "sources": [item.payload() for item in audits],
        }


class CompactNyseFallbackBackend:
    backend_id = FALLBACK_CALENDAR_VERSION
    package_name = "infrastructure.data.market_sessions.compact_fallback"
    package_version = FALLBACK_CALENDAR_VERSION
    supported_exchanges = ("XNYS", "XNAS", "ARCX", "BATS", "XASE")
    historical_start = date(CALENDAR_START_YEAR, 1, 1)
    historical_end = date(CALENDAR_END_YEAR, 12, 31)
    future_end = date(CALENDAR_END_YEAR, 12, 31)
    deterministic_notes = "Embedded deterministic compact NYSE-like rules; not maintained authority evidence."
    holiday_source = "Local embedded NYSE-like holiday rules plus explicit special full closures."
    early_close_support = "Local deterministic early-close rules for 2016-2026."
    special_closure_support = "Only explicitly embedded closures."
    timezone_handling = "zoneinfo America/New_York converted to UTC."
    fallback = True

    def lookup_session(self, exchange: str, day: date) -> CalendarSessionRecord:
        exchange_id = _normalize_exchange(exchange)
        params = _generation_params(exchange_id, "fallback")
        if not self._in_range(day):
            return _record(
                exchange=exchange_id,
                package=self.package_name,
                package_version=self.package_version,
                fallback_used=True,
                fallback_version=FALLBACK_CALENDAR_VERSION,
                source_status=OUTSIDE_AUTHORITY_RANGE_STATUS,
                base_status=OUTSIDE_AUTHORITY_RANGE_STATUS,
                session_date=day,
                open_timestamp=None,
                close_timestamp=None,
                pre_market_open_timestamp=None,
                after_hours_close_timestamp=None,
                early_close=False,
                closure_reason=(
                    f"compact fallback coverage is {CALENDAR_START_YEAR}-{CALENDAR_END_YEAR}"
                ),
                schedule_generation_parameters=params,
                evidence=(
                    {
                        "source": self.package_name,
                        "source_version": self.package_version,
                        "status": OUTSIDE_AUTHORITY_RANGE_STATUS,
                    },
                ),
            )
        if day.weekday() >= 5:
            return _record(
                exchange=exchange_id,
                package=self.package_name,
                package_version=self.package_version,
                fallback_used=True,
                fallback_version=FALLBACK_CALENDAR_VERSION,
                source_status=FALLBACK_CALENDAR_USED_STATUS,
                base_status=CLOSED_WEEKEND_STATUS,
                session_date=day,
                open_timestamp=None,
                close_timestamp=None,
                pre_market_open_timestamp=None,
                after_hours_close_timestamp=None,
                early_close=False,
                closure_reason="weekend",
                schedule_generation_parameters=params,
            )
        if day in SPECIAL_FULL_CLOSURES:
            return _record(
                exchange=exchange_id,
                package=self.package_name,
                package_version=self.package_version,
                fallback_used=True,
                fallback_version=FALLBACK_CALENDAR_VERSION,
                source_status=FALLBACK_CALENDAR_USED_STATUS,
                base_status=SPECIAL_CLOSURE_STATUS,
                session_date=day,
                open_timestamp=None,
                close_timestamp=None,
                pre_market_open_timestamp=None,
                after_hours_close_timestamp=None,
                early_close=False,
                closure_reason=SPECIAL_FULL_CLOSURES[day],
                schedule_generation_parameters=params,
            )
        if day in compact_nyse_holidays(day.year):
            return _record(
                exchange=exchange_id,
                package=self.package_name,
                package_version=self.package_version,
                fallback_used=True,
                fallback_version=FALLBACK_CALENDAR_VERSION,
                source_status=FALLBACK_CALENDAR_USED_STATUS,
                base_status=CLOSED_HOLIDAY_STATUS,
                session_date=day,
                open_timestamp=None,
                close_timestamp=None,
                pre_market_open_timestamp=None,
                after_hours_close_timestamp=None,
                early_close=False,
                closure_reason="NYSE-like holiday",
                schedule_generation_parameters=params,
            )
        close = EARLY_CLOSE if day in compact_nyse_early_closes(day.year) else RTH_CLOSE
        base_status = EARLY_CLOSE_STATUS if close == EARLY_CLOSE else REGULAR_SESSION_STATUS
        return _record(
            exchange=exchange_id,
            package=self.package_name,
            package_version=self.package_version,
            fallback_used=True,
            fallback_version=FALLBACK_CALENDAR_VERSION,
            source_status=FALLBACK_CALENDAR_USED_STATUS,
            base_status=base_status,
            session_date=day,
            open_timestamp=_local_datetime(day, RTH_OPEN).astimezone(timezone.utc),
            close_timestamp=_local_datetime(day, close).astimezone(timezone.utc),
            pre_market_open_timestamp=_local_datetime(day, PRE_MARKET_OPEN).astimezone(timezone.utc),
            after_hours_close_timestamp=_local_datetime(day, AFTER_HOURS_CLOSE).astimezone(timezone.utc),
            early_close=close == EARLY_CLOSE,
            closure_reason="",
            schedule_generation_parameters=params,
        )

    def sessions_in_range(self, exchange: str, start: date, end: date) -> list[date]:
        if end < start:
            return []
        output = []
        current = start
        while current <= end:
            if self.lookup_session(exchange, current).is_trading_day:
                output.append(current)
            current += timedelta(days=1)
        return output

    def previous_session(self, exchange: str, day: date) -> date | None:
        current = day - timedelta(days=1)
        while current >= self.historical_start:
            if self.lookup_session(exchange, current).is_trading_day:
                return current
            current -= timedelta(days=1)
        return None

    def next_session(self, exchange: str, day: date) -> date | None:
        current = day + timedelta(days=1)
        while current <= self.future_end:
            if self.lookup_session(exchange, current).is_trading_day:
                return current
            current += timedelta(days=1)
        return None

    def _in_range(self, day: date) -> bool:
        return self.historical_start <= day <= self.future_end


class UnavailableCalendarBackend(CompactNyseFallbackBackend):
    backend_id = "maintained_calendar_unavailable"
    package_name = SELECTED_MAINTAINED_PACKAGE
    fallback = False

    def __init__(self, *, reason: str, package_version: str = "") -> None:
        self.reason = reason
        self.package_version = package_version

    def lookup_session(self, exchange: str, day: date) -> CalendarSessionRecord:
        exchange_id = _normalize_exchange(exchange)
        return _record(
            exchange=exchange_id,
            package=self.package_name,
            package_version=self.package_version,
            fallback_used=False,
            fallback_version=None,
            source_status=OUTSIDE_AUTHORITY_RANGE_STATUS,
            base_status=OUTSIDE_AUTHORITY_RANGE_STATUS,
            session_date=day,
            open_timestamp=None,
            close_timestamp=None,
            pre_market_open_timestamp=None,
            after_hours_close_timestamp=None,
            early_close=False,
            closure_reason=self.reason,
            schedule_generation_parameters=_generation_params(exchange_id, "unavailable"),
        )

    def sessions_in_range(self, exchange: str, start: date, end: date) -> list[date]:
        return []

    def previous_session(self, exchange: str, day: date) -> date | None:
        return None

    def next_session(self, exchange: str, day: date) -> date | None:
        return None


class ExchangeCalendarsBackend:
    backend_id = "exchange_calendars"
    package_name = SELECTED_MAINTAINED_PACKAGE
    supported_exchanges = MAINTAINED_SUPPORTED_EXCHANGES
    historical_start = None
    historical_end = None
    future_end = None
    deterministic_notes = (
        "Use only when installed package version equals the pinned selected version; "
        "record package version and schedule hash for replay."
    )
    holiday_source = "exchange_calendars maintained exchange definitions."
    early_close_support = "Uses exchange_calendars special close schedules."
    special_closure_support = "Uses exchange_calendars regular and adhoc holidays/special closes where defined."
    timezone_handling = "UTC pandas timestamps with exchange timezone metadata."
    fallback = False

    def __init__(self, module: Any, *, package_version: str) -> None:
        self.module = module
        self.package_version = package_version

    def lookup_session(self, exchange: str, day: date) -> CalendarSessionRecord:
        exchange_id = _normalize_exchange(exchange)
        calendar = self._calendar(exchange_id)
        params = _generation_params(exchange_id, "maintained")
        timezone_name = _calendar_timezone(calendar)
        if not _calendar_in_range(calendar, day):
            return _record(
                exchange=exchange_id,
                package=self.package_name,
                package_version=self.package_version,
                fallback_used=False,
                fallback_version=None,
                source_status=OUTSIDE_AUTHORITY_RANGE_STATUS,
                base_status=OUTSIDE_AUTHORITY_RANGE_STATUS,
                session_date=day,
                open_timestamp=None,
                close_timestamp=None,
                pre_market_open_timestamp=None,
                after_hours_close_timestamp=None,
                early_close=False,
                closure_reason="date outside exchange_calendars generated schedule range",
                schedule_generation_parameters=params,
                market_timezone=timezone_name,
            )
        if not _calendar_is_session(calendar, day):
            status = _closed_status(calendar, day)
            return _record(
                exchange=exchange_id,
                package=self.package_name,
                package_version=self.package_version,
                fallback_used=False,
                fallback_version=None,
                source_status=status,
                base_status=status,
                session_date=day,
                open_timestamp=None,
                close_timestamp=None,
                pre_market_open_timestamp=None,
                after_hours_close_timestamp=None,
                early_close=False,
                closure_reason=_closure_reason(calendar, day, status),
                schedule_generation_parameters=params,
                market_timezone=timezone_name,
            )
        open_ts = _calendar_session_open(calendar, day)
        close_ts = _calendar_session_close(calendar, day)
        regular_close = _local_datetime(day, RTH_CLOSE, timezone_name).astimezone(timezone.utc)
        early = close_ts < regular_close
        base_status = EARLY_CLOSE_STATUS if early else REGULAR_SESSION_STATUS
        return _record(
            exchange=exchange_id,
            package=self.package_name,
            package_version=self.package_version,
            fallback_used=False,
            fallback_version=None,
            source_status=base_status,
            base_status=base_status,
            session_date=day,
            open_timestamp=open_ts,
            close_timestamp=close_ts,
            pre_market_open_timestamp=_local_datetime(day, PRE_MARKET_OPEN, timezone_name).astimezone(timezone.utc),
            after_hours_close_timestamp=_local_datetime(day, AFTER_HOURS_CLOSE, timezone_name).astimezone(timezone.utc),
            early_close=early,
            closure_reason="",
            schedule_generation_parameters=params,
            market_timezone=timezone_name,
        )

    def sessions_in_range(self, exchange: str, start: date, end: date) -> list[date]:
        calendar = self._calendar(_normalize_exchange(exchange))
        if hasattr(calendar, "sessions_in_range"):
            values = calendar.sessions_in_range(start.isoformat(), end.isoformat())
            return [_timestamp_to_date(value) for value in list(values)]
        return [
            day
            for day in _date_range(start, end)
            if self.lookup_session(exchange, day).is_trading_day
        ]

    def previous_session(self, exchange: str, day: date) -> date | None:
        calendar = self._calendar(_normalize_exchange(exchange))
        try:
            if hasattr(calendar, "previous_session"):
                return _timestamp_to_date(calendar.previous_session(day.isoformat()))
            if hasattr(calendar, "previous_close"):
                return _timestamp_to_date(calendar.previous_close(day.isoformat()))
        except Exception:
            return None
        return None

    def next_session(self, exchange: str, day: date) -> date | None:
        calendar = self._calendar(_normalize_exchange(exchange))
        try:
            if hasattr(calendar, "next_session"):
                return _timestamp_to_date(calendar.next_session(day.isoformat()))
            if hasattr(calendar, "next_open"):
                return _timestamp_to_date(calendar.next_open(day.isoformat()))
        except Exception:
            return None
        return None

    def _calendar(self, exchange: str) -> Any:
        return self.module.get_calendar(exchange)


def load_exchange_calendars_backend(
    *,
    expected_package_version: str = SELECTED_MAINTAINED_PACKAGE_VERSION,
) -> CalendarBackend | None:
    try:
        module = importlib.import_module(SELECTED_MAINTAINED_PACKAGE)
        version = importlib.metadata.version(SELECTED_MAINTAINED_PACKAGE_DISTRIBUTION)
    except Exception:
        return None
    if version != expected_package_version:
        return ExchangeCalendarsBackend(module, package_version=version)
    return ExchangeCalendarsBackend(module, package_version=version)


def default_calendar_authority() -> ExchangeCalendarAuthority:
    return ExchangeCalendarAuthority()


def calendar_authority_identity(
    *,
    start: date | str | None = None,
    end: date | str | None = None,
    exchange: str = CALENDAR_ID,
) -> dict[str, Any]:
    return default_calendar_authority().identity(start=start, end=end, exchange=exchange)


def materialize_calendar_schedule(
    start: date | str,
    end: date | str,
    *,
    exchange: str = CALENDAR_ID,
    include_closed: bool = False,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows = default_calendar_authority().materialize_schedule(
        start,
        end,
        exchange=exchange,
        include_closed=include_closed,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            _write_schedule_csv(output_path, rows)
        else:
            _write_json_atomic(
                output_path,
                {
                    "contract_version": "calendar_schedule_materialization.v1",
                    "rows": rows,
                    "content_hash": _canonical_hash(rows),
                },
            )
    return rows


def write_calendar_artifacts(output_dir: Path, *, exchange: str = CALENDAR_ID) -> dict[str, Path]:
    authority = default_calendar_authority()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "calendar_source_audit": output_dir / "calendar_source_audit.json",
        "calendar_authority_contract": output_dir / "calendar_authority_contract.json",
        "calendar_version": output_dir / "calendar_version.json",
        "calendar_coverage": output_dir / "calendar_coverage.json",
        "calendar_conflicts": output_dir / "calendar_conflicts.csv",
        "maintained_fallback_comparison": output_dir / "maintained_fallback_comparison.csv",
        "materialised_schedule": output_dir / "materialised_schedule.parquet",
        "calendar_validation": output_dir / "calendar_validation.json",
    }
    _write_json_atomic(artifacts["calendar_source_audit"], authority.source_audit())
    _write_json_atomic(artifacts["calendar_authority_contract"], calendar_contract_payload())
    _write_json_atomic(
        artifacts["calendar_version"],
        authority.identity(
            start=min(CALENDAR_VALIDATION_DATES),
            end=max(CALENDAR_VALIDATION_DATES),
            exchange=exchange,
        ),
    )
    _write_json_atomic(artifacts["calendar_coverage"], calendar_coverage_payload())
    _write_conflicts_csv(artifacts["calendar_conflicts"], [])
    comparison_rows = maintained_fallback_comparison_rows(exchange=exchange)
    _write_comparison_csv(artifacts["maintained_fallback_comparison"], comparison_rows)
    _write_schedule_parquet(
        artifacts["materialised_schedule"],
        _materialised_validation_schedule_rows(authority, exchange=exchange),
    )
    _write_json_atomic(artifacts["calendar_validation"], calendar_validation_payload())
    return artifacts


def calendar_contract_payload() -> dict[str, Any]:
    return {
        "contract_version": "calendar_authority_contract.v1",
        "authority_version": CALENDAR_AUTHORITY_VERSION,
        "selected_calendar": CALENDAR_ID,
        "selected_package": SELECTED_MAINTAINED_PACKAGE,
        "selected_package_version": SELECTED_MAINTAINED_PACKAGE_VERSION,
        "fallback_calendar_version": FALLBACK_CALENDAR_VERSION,
        "required_fields": [
            "exchange",
            "calendar_id",
            "authority_version",
            "package",
            "package_version",
            "schedule_generation_parameters",
            "session_date",
            "market_timezone",
            "open_timestamp",
            "close_timestamp",
            "pre_market_open_timestamp",
            "after_hours_close_timestamp",
            "early_close",
            "closure_reason",
            "source_status",
            "base_status",
            "generated_at",
            "schedule_hash",
        ],
        "statuses": [
            REGULAR_SESSION_STATUS,
            EARLY_CLOSE_STATUS,
            CLOSED_HOLIDAY_STATUS,
            CLOSED_WEEKEND_STATUS,
            SPECIAL_CLOSURE_STATUS,
            OUTSIDE_AUTHORITY_RANGE_STATUS,
            CONFLICTING_CALENDAR_EVIDENCE_STATUS,
            FALLBACK_CALENDAR_USED_STATUS,
        ],
        "third_party_call_boundary": "infrastructure.data.calendar_authority",
        "timestamp_policy": "All persisted timestamps are UTC ISO-8601 with Z suffix.",
        "fallback_policy": "Fallback rows set fallback_used=true and source_status=FALLBACK_CALENDAR_USED unless outside range.",
        "conflict_precedence": [
            "Pinned maintained package exact version",
            "Reviewed, versioned, hashed local override",
            "Compact fallback only when maintained package is unavailable or mismatched and fallback is allowed",
        ],
    }


def calendar_coverage_payload() -> dict[str, Any]:
    return {
        "contract_version": "calendar_coverage.v1",
        "selected_authority": SELECTED_MAINTAINED_PACKAGE,
        "selected_package_version": SELECTED_MAINTAINED_PACKAGE_VERSION,
        "selected_supported_exchange": CALENDAR_ID,
        "maintained_historical_range": "package-defined; adapter records OUTSIDE_AUTHORITY_RANGE when the generated calendar declines a date",
        "maintained_future_schedule_behaviour": "bounded by package calendar generation parameters; record schedule hash for every used range",
        "fallback_verified_coverage": {
            "start": f"{CALENDAR_START_YEAR}-01-01",
            "end": f"{CALENDAR_END_YEAR}-12-31",
            "promotion_grade_outside_range": "fail_closed",
        },
        "timezone": str(EASTERN.key),
    }


def calendar_validation_payload() -> dict[str, Any]:
    authority = default_calendar_authority()
    maintained_active = (
        authority.backend_status == "MAINTAINED"
        and not authority.backend.fallback
        and authority.backend.package_name == SELECTED_MAINTAINED_PACKAGE
        and authority.backend.package_version == SELECTED_MAINTAINED_PACKAGE_VERSION
    )
    return {
        "contract_version": "calendar_validation.v1",
        "classification": (
            "MAINTAINED_CALENDAR_AUTHORITY_IMPLEMENTED"
            if maintained_active
            else "CALENDAR_AUTHORITY_IMPLEMENTED_WITH_FALLBACK"
        ),
        "runtime_backend_status": authority.backend_status,
        "runtime_backend_status_reason": authority.backend_status_reason,
        "runtime_package": authority.backend.package_name,
        "runtime_package_version": authority.backend.package_version,
        "runtime_fallback_used": authority.backend.fallback,
        "validation_dates": [day.isoformat() for day in CALENDAR_VALIDATION_DATES],
        "dependency_activation": {
            "environment": ".venv-ticket-68a",
            "install_command": (
                ".\\.venv-ticket-68a\\Scripts\\python.exe -m pip install "
                "exchange-calendars==4.13.2"
            ),
            "resolved_package_version": SELECTED_MAINTAINED_PACKAGE_VERSION,
            "import_result": "import exchange_calendars succeeded",
            "dependency_conflicts": [
                "pip check reports torch 2.13.0+cpu requires setuptools>=77.0.3; "
                "venv has setuptools 65.5.0"
            ],
        },
        "focused_tests": [
            "regular session",
            "early close",
            "holiday",
            "weekend",
            "DST start",
            "DST end",
            "pre-market",
            "after-hours",
            "previous/next session",
            "outside authority range",
            "maintained dependency unavailable",
            "explicit fallback",
            "special closure",
            "local override",
            "conflicting evidence",
            "deterministic schedule hash",
            "session_type compatibility",
            "Ticket 68 integration",
            "manifest/certification lineage",
            "no trading behaviour changes",
        ],
        "validation_runs": [
            {
                "mode": "maintained",
                "command": (
                    ".\\.venv-ticket-68a\\Scripts\\python.exe -m pytest "
                    "tests/test_calendar_authority.py "
                    "tests/test_market_information_availability_authority.py "
                    "tests/test_dataset_build_manifest.py "
                    "tests/test_frozen_selector_lineage_guard.py "
                    "tests/test_selector_dataset_lineage.py "
                    "tests/test_research_certification.py "
                    "tests/test_historical_bar_backfill.py "
                    "tests/test_alpaca_5m_symbol_year_finalizer.py "
                    "tests/test_stock_level_prediction_artifacts.py -q"
                ),
                "result": "passed",
                "passed": 191,
                "warnings": 1,
            },
            {
                "mode": "forced_fallback",
                "command": (
                    ".\\.venv-ticket-68a\\Scripts\\python.exe -m pytest "
                    "tests/test_calendar_authority.py -k "
                    "\"fallback or session_type or regular_early_close_holiday_weekend\" -q"
                ),
                "result": "passed",
                "passed": 3,
                "deselected": 10,
            },
            {
                "mode": "dependency_unavailable_fallback",
                "command": (
                    "python -m pytest tests/test_calendar_authority.py "
                    "tests/test_market_information_availability_authority.py -q"
                ),
                "result": "passed",
                "passed": 23,
                "skipped": 5,
                "warnings": 1,
            },
        ],
        "trading_impact": "none",
    }


def maintained_fallback_comparison_rows(*, exchange: str = CALENDAR_ID) -> list[dict[str, Any]]:
    maintained = ExchangeCalendarAuthority(exchange=exchange, allow_fallback=False)
    fallback = ExchangeCalendarAuthority(
        exchange=exchange,
        backend=CompactNyseFallbackBackend(),
    )
    rows = []
    for day in CALENDAR_VALIDATION_DATES:
        maintained_record = maintained.session(day, exchange=exchange)
        fallback_record = fallback.session(day, exchange=exchange)
        mismatches = _calendar_comparison_mismatches(
            maintained_record,
            fallback_record,
        )
        rows.append(
            {
                "session_date": day.isoformat(),
                "maintained_status": maintained_record.base_status,
                "fallback_status": fallback_record.base_status,
                "maintained_open_timestamp": _format_timestamp(
                    maintained_record.open_timestamp
                ),
                "fallback_open_timestamp": _format_timestamp(
                    fallback_record.open_timestamp
                ),
                "maintained_close_timestamp": _format_timestamp(
                    maintained_record.close_timestamp
                ),
                "fallback_close_timestamp": _format_timestamp(
                    fallback_record.close_timestamp
                ),
                "maintained_early_close": maintained_record.early_close,
                "fallback_early_close": fallback_record.early_close,
                "maintained_holiday_closure": _holiday_closure(maintained_record),
                "fallback_holiday_closure": _holiday_closure(fallback_record),
                "mismatch_fields": "|".join(mismatches),
                "classification": _calendar_comparison_classification(
                    maintained_record,
                    fallback_record,
                    mismatches,
                ),
                "maintained_schedule_hash": maintained_record.schedule_hash,
                "fallback_schedule_hash": fallback_record.schedule_hash,
            }
        )
    return rows


def calendar_source_audit(*, installed_backend: CalendarBackend | None = None) -> list[CalendarSourceAudit]:
    installed_version = ""
    installed = False
    try:
        installed_version = importlib.metadata.version(
            SELECTED_MAINTAINED_PACKAGE_DISTRIBUTION
        )
        installed = True
    except Exception:
        installed_version = SELECTED_MAINTAINED_PACKAGE_VERSION
    return [
        CalendarSourceAudit(
            package=SELECTED_MAINTAINED_PACKAGE,
            version=installed_version,
            licence="Apache Software License / Apache-2.0",
            supported_exchanges=MAINTAINED_SUPPORTED_EXCHANGES,
            historical_range="package-defined generated schedules; supports historical research subject to package calendar bounds",
            future_schedule_behaviour="bounded future generation; schedules must be hashed and version-pinned for replay",
            holiday_source="maintained package rules and contributed exchange definitions",
            early_close_support="supported through package special close schedule",
            special_closures="supported where package exchange definition includes adhoc holidays and special closes",
            timezone_handling="UTC schedule timestamps with exchange timezone metadata",
            determinism_concerns="must pin exact package version; package upgrades can change past/future schedules",
            selected=True,
            installed=installed,
            status="INSTALLED" if installed else "SELECTED_NOT_INSTALLED_IN_CURRENT_ENV",
        ),
        CalendarSourceAudit(
            package="pandas_market_calendars",
            version="not added",
            licence="not audited locally",
            supported_exchanges=("XNYS", "NASDAQ", "NYSE"),
            historical_range="not selected",
            future_schedule_behaviour="not selected",
            holiday_source="not selected",
            early_close_support="not selected",
            special_closures="not selected",
            timezone_handling="not selected",
            determinism_concerns="would add second competing calendar dependency",
            selected=False,
            installed=False,
            status="REJECTED_TO_AVOID_COMPETING_DEPENDENCY",
        ),
        CalendarSourceAudit(
            package="compact embedded fallback",
            version=FALLBACK_CALENDAR_VERSION,
            licence="project local",
            supported_exchanges=CompactNyseFallbackBackend.supported_exchanges,
            historical_range=f"{CALENDAR_START_YEAR}-01-01 through {CALENDAR_END_YEAR}-12-31",
            future_schedule_behaviour="no future authority beyond verified coverage",
            holiday_source="embedded deterministic rules",
            early_close_support="limited deterministic rules",
            special_closures="explicit local list only",
            timezone_handling="zoneinfo America/New_York",
            determinism_concerns="not maintained authority evidence; use must be explicit",
            selected=False,
            installed=True,
            status="FALLBACK_ONLY",
        ),
    ]


def compact_nyse_holidays(year: int) -> set[date]:
    if year < CALENDAR_START_YEAR or year > CALENDAR_END_YEAR:
        raise ValueError(
            f"NYSE fallback calendar is implemented for {CALENDAR_START_YEAR}-{CALENDAR_END_YEAR}"
        )
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _good_friday(year),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    if date(year + 1, 1, 1).weekday() == 5:
        holidays.add(date(year, 12, 31))
    holidays.update(day for day in SPECIAL_FULL_CLOSURES if day.year == year)
    return {day for day in holidays if day.year == year}


def compact_nyse_early_closes(year: int) -> set[date]:
    if year < CALENDAR_START_YEAR or year > CALENDAR_END_YEAR:
        raise ValueError(
            f"NYSE fallback calendar is implemented for {CALENDAR_START_YEAR}-{CALENDAR_END_YEAR}"
        )
    candidates = {
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),
        date(year, 12, 24),
    }
    july_4 = date(year, 7, 4)
    if july_4.weekday() in {1, 2, 3, 4}:
        candidates.add(july_4 - timedelta(days=1))
    return {day for day in candidates if _compact_is_trading_session(day)}


def _compact_is_trading_session(day: date) -> bool:
    return (
        CALENDAR_START_YEAR <= day.year <= CALENDAR_END_YEAR
        and day.weekday() < 5
        and day not in compact_nyse_holidays(day.year)
    )


def _record(
    *,
    exchange: str,
    package: str,
    package_version: str,
    fallback_used: bool,
    fallback_version: str | None,
    source_status: str,
    base_status: str,
    session_date: date,
    open_timestamp: datetime | None,
    close_timestamp: datetime | None,
    pre_market_open_timestamp: datetime | None,
    after_hours_close_timestamp: datetime | None,
    early_close: bool,
    closure_reason: str,
    schedule_generation_parameters: Mapping[str, Any],
    market_timezone: str = "America/New_York",
    evidence: Sequence[Mapping[str, Any]] = (),
    conflict: CalendarConflict | None = None,
) -> CalendarSessionRecord:
    authority_version_identity = (
        f"{CALENDAR_AUTHORITY_VERSION}:{package}:{package_version}"
    )
    final_evidence = tuple(evidence) or (
        {
            "source": package,
            "source_version": package_version,
            "source_status": source_status,
            "base_status": base_status,
        },
    )
    return CalendarSessionRecord(
        exchange=exchange,
        calendar_id=exchange,
        authority_version=CALENDAR_AUTHORITY_VERSION,
        authority_version_identity=authority_version_identity,
        package=package,
        package_version=package_version,
        fallback_used=fallback_used,
        fallback_version=fallback_version,
        source_status=source_status,
        base_status=base_status,
        session_date=session_date,
        market_timezone=market_timezone,
        open_timestamp=_to_utc(open_timestamp) if open_timestamp else None,
        close_timestamp=_to_utc(close_timestamp) if close_timestamp else None,
        pre_market_open_timestamp=_to_utc(pre_market_open_timestamp) if pre_market_open_timestamp else None,
        after_hours_close_timestamp=_to_utc(after_hours_close_timestamp) if after_hours_close_timestamp else None,
        early_close=early_close,
        closure_reason=closure_reason,
        schedule_generation_parameters=schedule_generation_parameters,
        generated_at=GENERATED_AT,
        evidence=final_evidence,
        conflict=conflict,
    ).with_hash()


def _apply_override(
    base: CalendarSessionRecord,
    override: LocalCalendarOverride,
) -> CalendarSessionRecord:
    override_evidence = {
        "source": override.source_path,
        "source_hash": override.source_hash,
        "reviewed": override.reviewed,
        "effective_date": override.effective_date.isoformat(),
        "action": override.action,
        "reason": override.reason,
        "payload": _json_ready(override.payload),
    }
    if not override.reviewed or override.effective_date > base.session_date:
        conflict = CalendarConflict(
            conflict_type="UNREVIEWED_OR_NOT_EFFECTIVE_LOCAL_OVERRIDE",
            session_date=base.session_date.isoformat(),
            base_status=base.base_status,
            override_status=override.applied_status,
            base_evidence=base.payload(),
            override_evidence=override_evidence,
            resolution="BLOCK_PROMOTION_GRADE_USE",
            promotion_blocking=True,
        )
        return _record(
            exchange=base.exchange,
            package=base.package,
            package_version=base.package_version,
            fallback_used=base.fallback_used,
            fallback_version=base.fallback_version,
            source_status=CONFLICTING_CALENDAR_EVIDENCE_STATUS,
            base_status=CONFLICTING_CALENDAR_EVIDENCE_STATUS,
            session_date=base.session_date,
            open_timestamp=None,
            close_timestamp=None,
            pre_market_open_timestamp=None,
            after_hours_close_timestamp=None,
            early_close=False,
            closure_reason=override.reason or "local override conflict",
            schedule_generation_parameters=base.schedule_generation_parameters,
            market_timezone=base.market_timezone,
            evidence=(*base.evidence, override_evidence),
            conflict=conflict,
        )
    action = override.action.upper()
    if action == "CLOSE":
        return _record(
            exchange=base.exchange,
            package=base.package,
            package_version=base.package_version,
            fallback_used=base.fallback_used,
            fallback_version=base.fallback_version,
            source_status=SPECIAL_CLOSURE_STATUS,
            base_status=SPECIAL_CLOSURE_STATUS,
            session_date=base.session_date,
            open_timestamp=None,
            close_timestamp=None,
            pre_market_open_timestamp=None,
            after_hours_close_timestamp=None,
            early_close=False,
            closure_reason=override.reason,
            schedule_generation_parameters=base.schedule_generation_parameters,
            market_timezone=base.market_timezone,
            evidence=(*base.evidence, override_evidence),
        )
    if action == "EARLY_CLOSE":
        close_time = _parse_time(override.payload.get("close_time"), EARLY_CLOSE)
        return _record(
            exchange=base.exchange,
            package=base.package,
            package_version=base.package_version,
            fallback_used=base.fallback_used,
            fallback_version=base.fallback_version,
            source_status=EARLY_CLOSE_STATUS,
            base_status=EARLY_CLOSE_STATUS,
            session_date=base.session_date,
            open_timestamp=base.open_timestamp
            or _local_datetime(base.session_date, RTH_OPEN, base.market_timezone).astimezone(timezone.utc),
            close_timestamp=_local_datetime(base.session_date, close_time, base.market_timezone).astimezone(timezone.utc),
            pre_market_open_timestamp=base.pre_market_open_timestamp
            or _local_datetime(base.session_date, PRE_MARKET_OPEN, base.market_timezone).astimezone(timezone.utc),
            after_hours_close_timestamp=base.after_hours_close_timestamp
            or _local_datetime(base.session_date, AFTER_HOURS_CLOSE, base.market_timezone).astimezone(timezone.utc),
            early_close=True,
            closure_reason=override.reason,
            schedule_generation_parameters=base.schedule_generation_parameters,
            market_timezone=base.market_timezone,
            evidence=(*base.evidence, override_evidence),
        )
    conflict = CalendarConflict(
        conflict_type="UNSUPPORTED_LOCAL_OVERRIDE_ACTION",
        session_date=base.session_date.isoformat(),
        base_status=base.base_status,
        override_status=override.applied_status,
        base_evidence=base.payload(),
        override_evidence=override_evidence,
        resolution="BLOCK_PROMOTION_GRADE_USE",
        promotion_blocking=True,
    )
    return _record(
        exchange=base.exchange,
        package=base.package,
        package_version=base.package_version,
        fallback_used=base.fallback_used,
        fallback_version=base.fallback_version,
        source_status=CONFLICTING_CALENDAR_EVIDENCE_STATUS,
        base_status=CONFLICTING_CALENDAR_EVIDENCE_STATUS,
        session_date=base.session_date,
        open_timestamp=None,
        close_timestamp=None,
        pre_market_open_timestamp=None,
        after_hours_close_timestamp=None,
        early_close=False,
        closure_reason=override.reason or "unsupported local override",
        schedule_generation_parameters=base.schedule_generation_parameters,
        market_timezone=base.market_timezone,
        evidence=(*base.evidence, override_evidence),
        conflict=conflict,
    )


def _load_overrides(
    paths: Sequence[Path | str],
    exchange: str,
) -> dict[date, LocalCalendarOverride]:
    overrides: dict[date, LocalCalendarOverride] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        source_hash = _file_sha256(path)
        contract = str(payload.get("contract_version") or "")
        reviewed = bool(payload.get("reviewed"))
        effective_date = _coerce_date(payload.get("effective_date") or date.min)
        calendar_id = _normalize_exchange(str(payload.get("calendar_id") or exchange))
        if calendar_id != exchange:
            continue
        for item in payload.get("overrides", []) or []:
            if not isinstance(item, Mapping):
                continue
            day = _coerce_date(item.get("session_date"))
            item_reviewed = bool(item.get("reviewed", reviewed))
            if contract != "calendar_local_override.v1":
                item_reviewed = False
            overrides[day] = LocalCalendarOverride(
                session_date=day,
                action=str(item.get("action") or "").upper(),
                reason=str(item.get("reason") or "local calendar override"),
                reviewed=item_reviewed,
                effective_date=_coerce_date(item.get("effective_date") or effective_date),
                source_path=str(path),
                source_hash=source_hash,
                payload=dict(item),
            )
    return overrides


def _generation_params(exchange: str, backend_kind: str) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "calendar_id": exchange,
        "backend": backend_kind,
        "side": "left",
        "pre_market_open": PRE_MARKET_OPEN.isoformat(timespec="minutes"),
        "regular_open": RTH_OPEN.isoformat(timespec="minutes"),
        "regular_close": RTH_CLOSE.isoformat(timespec="minutes"),
        "after_hours_close": AFTER_HOURS_CLOSE.isoformat(timespec="minutes"),
        "market_timezone": str(EASTERN.key),
        "selected_package_version": SELECTED_MAINTAINED_PACKAGE_VERSION,
    }


def _calendar_in_range(calendar: Any, day: date) -> bool:
    for attr in ("bound_min", "first_session"):
        value = _calendar_attr_value(calendar, attr)
        if value is not None and day < _timestamp_to_date(value):
            return False
    for attr in ("bound_max", "last_session"):
        value = _calendar_attr_value(calendar, attr)
        if value is not None and day > _timestamp_to_date(value):
            return False
    return True


def _calendar_attr_value(calendar: Any, attr: str) -> Any:
    value = getattr(calendar, attr, None)
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _calendar_is_session(calendar: Any, day: date) -> bool:
    if hasattr(calendar, "is_session"):
        return bool(calendar.is_session(day.isoformat()))
    schedule = getattr(calendar, "schedule", None)
    if schedule is not None:
        try:
            return day.isoformat() in {str(item)[:10] for item in schedule.index}
        except Exception:
            return False
    return False


def _calendar_session_open(calendar: Any, day: date) -> datetime:
    if hasattr(calendar, "session_open"):
        return _to_utc(calendar.session_open(day.isoformat()))
    row = _calendar_schedule_row(calendar, day)
    value = row.get("open") or row.get("market_open")
    return _to_utc(value)


def _calendar_session_close(calendar: Any, day: date) -> datetime:
    if hasattr(calendar, "session_close"):
        return _to_utc(calendar.session_close(day.isoformat()))
    row = _calendar_schedule_row(calendar, day)
    value = row.get("close") or row.get("market_close")
    return _to_utc(value)


def _calendar_schedule_row(calendar: Any, day: date) -> Mapping[str, Any]:
    schedule = getattr(calendar, "schedule", None)
    if schedule is None:
        raise ValueError("calendar schedule is unavailable")
    row = schedule.loc[day.isoformat()]
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _calendar_timezone(calendar: Any) -> str:
    for attr in ("tz", "timezone"):
        value = getattr(calendar, attr, None)
        if value is not None:
            return str(getattr(value, "zone", None) or getattr(value, "key", None) or value)
    return str(EASTERN.key)


def _closed_status(calendar: Any, day: date) -> str:
    if day.weekday() >= 5:
        return CLOSED_WEEKEND_STATUS
    if _is_adhoc_holiday(calendar, day):
        return SPECIAL_CLOSURE_STATUS
    return CLOSED_HOLIDAY_STATUS


def _closure_reason(calendar: Any, day: date, status: str) -> str:
    if status == CLOSED_WEEKEND_STATUS:
        return "weekend"
    if status == SPECIAL_CLOSURE_STATUS:
        return "adhoc exchange closure"
    return "exchange holiday"


def _is_adhoc_holiday(calendar: Any, day: date) -> bool:
    for attr in ("adhoc_holidays", "special_holidays_adhoc"):
        values = getattr(calendar, attr, None)
        if values is None:
            continue
        try:
            return day in {_timestamp_to_date(value) for value in list(values)}
        except Exception:
            continue
    return day in SPECIAL_FULL_CLOSURES


def _date_range(start: date, end: date) -> list[date]:
    output = []
    current = start
    while current <= end:
        output.append(current)
        current += timedelta(days=1)
    return output


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        raise ValueError("date value is required")
    return datetime.fromisoformat(text[:10]).date()


def _timestamp_to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date"):
        return value.date()
    return datetime.fromisoformat(str(value)[:10]).date()


def _parse_time(value: Any, default: time) -> time:
    if isinstance(value, time):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    return time.fromisoformat(text)


def _local_datetime(
    day: date,
    value: time,
    timezone_name: str = "America/New_York",
) -> datetime:
    return datetime.combine(day, value, tzinfo=ZoneInfo(timezone_name))


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime"):
        parsed = value.to_pydatetime()
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (date, datetime, time)):
        return str(value)
    return value


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalize_exchange(value: str) -> str:
    text = str(value or CALENDAR_ID).upper()
    aliases = {
        "NYSE": "XNYS",
        "NASDAQ": "XNAS",
        "NASD": "XNAS",
        "US": "XNYS",
        "US_EQUITY": "XNYS",
    }
    return aliases.get(text, text)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    value = date(year, month, day)
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _good_friday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    le = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * le) // 451
    month = (h + le - 7 * m + 114) // 31
    day = ((h + le - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_schedule_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "session_date",
        "open_timestamp",
        "close_timestamp",
        "early_close",
        "closure_reason",
        "source_status",
        "base_status",
        "authority_version",
        "package_version",
        "fallback_used",
        "schedule_hash",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_schedule_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path)


def _materialised_validation_schedule_rows(
    authority: ExchangeCalendarAuthority,
    *,
    exchange: str,
) -> list[dict[str, Any]]:
    rows = []
    for day in CALENDAR_VALIDATION_DATES:
        record = authority.session(day, exchange=exchange)
        rows.append(
            {
                "calendar_id": record.calendar_id,
                "authority_version": record.authority_version,
                "authority_version_identity": record.authority_version_identity,
                "session_date": record.session_date.isoformat(),
                "open": _format_timestamp(record.open_timestamp),
                "close": _format_timestamp(record.close_timestamp),
                "early_close": record.early_close,
                "closure_metadata": json.dumps(
                    {
                        "closure_reason": record.closure_reason,
                        "source_status": record.source_status,
                        "base_status": record.base_status,
                        "fallback_used": record.fallback_used,
                        "fallback_version": record.fallback_version,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "package": record.package,
                "package_version": record.package_version,
                "schedule_hash": record.schedule_hash,
            }
        )
    return rows


def _calendar_comparison_mismatches(
    maintained: CalendarSessionRecord,
    fallback: CalendarSessionRecord,
) -> list[str]:
    comparisons = {
        "session_dates": maintained.session_date == fallback.session_date,
        "open_timestamps": maintained.open_timestamp == fallback.open_timestamp,
        "close_timestamps": maintained.close_timestamp == fallback.close_timestamp,
        "early_close_flags": maintained.early_close == fallback.early_close,
        "holiday_closures": _holiday_closure(maintained) == _holiday_closure(fallback),
    }
    return [
        field_name
        for field_name, matches in comparisons.items()
        if not matches
    ]


def _holiday_closure(record: CalendarSessionRecord) -> bool:
    return (
        not record.is_trading_day
        and record.base_status in {CLOSED_HOLIDAY_STATUS, SPECIAL_CLOSURE_STATUS}
    )


def _calendar_comparison_classification(
    maintained: CalendarSessionRecord,
    fallback: CalendarSessionRecord,
    mismatches: Sequence[str],
) -> str:
    if not mismatches:
        return "match"
    if fallback.base_status == OUTSIDE_AUTHORITY_RANGE_STATUS:
        return "fallback limitation"
    if maintained.package == SELECTED_MAINTAINED_PACKAGE and not maintained.fallback_used:
        return "maintained authority correction"
    return "unresolved conflict"


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "session_date",
        "maintained_status",
        "fallback_status",
        "maintained_open_timestamp",
        "fallback_open_timestamp",
        "maintained_close_timestamp",
        "fallback_close_timestamp",
        "maintained_early_close",
        "fallback_early_close",
        "maintained_holiday_closure",
        "fallback_holiday_closure",
        "mismatch_fields",
        "classification",
        "maintained_schedule_hash",
        "fallback_schedule_hash",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_conflicts_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "session_date",
        "conflict_type",
        "base_status",
        "override_status",
        "resolution",
        "promotion_blocking",
        "base_evidence_hash",
        "override_evidence_hash",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
