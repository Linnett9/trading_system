from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Sequence

from infrastructure.data.calendar_authority import (
    AFTER_HOURS_CLOSE,
    CALENDAR_END_YEAR,
    CALENDAR_START_YEAR,
    CLOSED_HOLIDAY_STATUS,
    CLOSED_WEEKEND_STATUS,
    CONFLICTING_CALENDAR_EVIDENCE_STATUS,
    EARLY_CLOSE,
    EARLY_CLOSE_STATUS,
    EASTERN,
    FALLBACK_CALENDAR_IDENTITY,
    FALLBACK_CALENDAR_USED_STATUS,
    OUTSIDE_AUTHORITY_RANGE_STATUS,
    PRE_MARKET_OPEN,
    REGULAR_SESSION_STATUS,
    RTH_CLOSE,
    RTH_OPEN,
    SPECIAL_CLOSURE_STATUS,
    SPECIAL_FULL_CLOSURES,
    compact_nyse_early_closes,
    compact_nyse_holidays,
    default_calendar_authority,
)


PRE_MARKET_SESSION = "pre_market"
REGULAR_SESSION = "regular"
RTH_SESSION = "rth"
AFTER_HOURS_SESSION = "after_hours"
CLOSED_SESSION = "closed"
HALTED_SESSION = "halted"


@dataclass(frozen=True)
class MarketSessionContext:
    exchange: str
    timezone: str
    timestamp_utc: datetime
    local_timestamp: datetime
    session_date: date
    trading_session: str
    is_trading_day: bool
    is_early_close: bool
    regular_open: datetime | None
    regular_close: datetime | None
    pre_market_open: datetime | None
    after_hours_close: datetime | None
    halted: bool = False
    halt_reason: str = ""
    calendar_identity: str = FALLBACK_CALENDAR_IDENTITY
    calendar_authority_version: str = ""
    calendar_authority_version_identity: str = ""
    calendar_source_status: str = ""
    calendar_base_status: str = ""
    calendar_package: str = ""
    calendar_package_version: str = ""
    calendar_schedule_hash: str = ""
    calendar_fallback_used: bool = False
    calendar_closure_reason: str = ""


def session_type(timestamp: datetime) -> str:
    local = _to_eastern(timestamp)
    close = rth_close_for_date(local.date())
    if local.time() < RTH_OPEN:
        return PRE_MARKET_SESSION
    if close is not None and RTH_OPEN <= local.time() < close:
        return RTH_SESSION
    return AFTER_HOURS_SESSION


def is_rth_timestamp(timestamp: datetime) -> bool:
    return session_type(timestamp) == RTH_SESSION


def exchange_session_context(
    timestamp: datetime,
    *,
    exchange: str = "XNYS",
    market_halts: Sequence[MappingLikeHalt] = (),
) -> MarketSessionContext:
    """Return DST-aware US equity session context for an instant."""

    timestamp_utc = _to_utc(timestamp)
    local = timestamp_utc.astimezone(EASTERN)
    record = default_calendar_authority().session(local.date(), exchange=exchange)
    trading_session = CLOSED_SESSION
    if (
        record.is_trading_day
        and record.pre_market_open_timestamp
        and record.open_timestamp
        and record.close_timestamp
        and record.after_hours_close_timestamp
    ):
        pre_open = record.pre_market_open_timestamp.astimezone(EASTERN)
        regular_open = record.open_timestamp.astimezone(EASTERN)
        regular_close = record.close_timestamp.astimezone(EASTERN)
        extended_close = record.after_hours_close_timestamp.astimezone(EASTERN)
        if pre_open <= local < regular_open:
            trading_session = PRE_MARKET_SESSION
        elif regular_open <= local < regular_close:
            trading_session = REGULAR_SESSION
        elif regular_close <= local < extended_close:
            trading_session = AFTER_HOURS_SESSION
    halted, halt_reason = _halt_state(timestamp_utc, market_halts)
    if halted and trading_session != CLOSED_SESSION:
        trading_session = HALTED_SESSION
    return MarketSessionContext(
        exchange=record.exchange,
        timezone=record.market_timezone,
        timestamp_utc=timestamp_utc,
        local_timestamp=local,
        session_date=record.session_date,
        trading_session=trading_session,
        is_trading_day=record.is_trading_day,
        is_early_close=record.early_close,
        regular_open=record.open_timestamp,
        regular_close=record.close_timestamp,
        pre_market_open=record.pre_market_open_timestamp,
        after_hours_close=record.after_hours_close_timestamp,
        halted=halted,
        halt_reason=halt_reason,
        calendar_identity=record.authority_version_identity,
        calendar_authority_version=record.authority_version,
        calendar_authority_version_identity=record.authority_version_identity,
        calendar_source_status=record.source_status,
        calendar_base_status=record.base_status,
        calendar_package=record.package,
        calendar_package_version=record.package_version,
        calendar_schedule_hash=record.schedule_hash,
        calendar_fallback_used=record.fallback_used,
        calendar_closure_reason=record.closure_reason,
    )


def is_trading_session(day: date) -> bool:
    return default_calendar_authority().session(day).is_trading_day


def trading_sessions(start: date, end: date) -> list[date]:
    return default_calendar_authority().sessions(start, end)


def previous_trading_session(day: date) -> date:
    previous = default_calendar_authority().previous_session(day)
    if previous is None:
        raise ValueError(f"No previous trading session available before {day}")
    return previous


def next_trading_session(day: date) -> date:
    next_day = default_calendar_authority().next_session(day)
    if next_day is None:
        raise ValueError(f"No next trading session available after {day}")
    return next_day


def rth_close_for_date(day: date) -> time | None:
    return default_calendar_authority().session(day).close_time


def expected_rth_timestamps(start: datetime, end: datetime, *, step: timedelta) -> list[datetime]:
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    local_day = _to_eastern(start_utc).date()
    last_day = _to_eastern(end_utc).date()
    output: list[datetime] = []
    while local_day <= last_day:
        close = rth_close_for_date(local_day)
        if close is not None:
            current = datetime.combine(local_day, RTH_OPEN, tzinfo=EASTERN)
            session_end = datetime.combine(local_day, close, tzinfo=EASTERN)
            while current < session_end:
                current_utc = current.astimezone(timezone.utc)
                if start_utc <= current_utc <= end_utc:
                    output.append(current_utc)
                current += step
        local_day += timedelta(days=1)
    return output


def nyse_holidays(year: int) -> set[date]:
    return compact_nyse_holidays(year)


def nyse_early_closes(year: int) -> set[date]:
    return compact_nyse_early_closes(year)


def calendar_status_for_date(day: date) -> dict[str, Any]:
    return default_calendar_authority().session(day).payload()


def _to_eastern(value: datetime) -> datetime:
    return _to_utc(value).astimezone(EASTERN)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


MappingLikeHalt = tuple[datetime, datetime] | dict[str, object]


def _halt_state(timestamp_utc: datetime, market_halts: Sequence[MappingLikeHalt]) -> tuple[bool, str]:
    for halt in market_halts:
        if isinstance(halt, dict):
            start = halt.get("start") or halt.get("start_timestamp")
            end = halt.get("end") or halt.get("end_timestamp")
            reason = str(halt.get("reason") or halt.get("halt_reason") or "represented_market_halt")
        else:
            start, end = halt
            reason = "represented_market_halt"
        start_ts = _coerce_halt_timestamp(start)
        end_ts = _coerce_halt_timestamp(end)
        if start_ts and end_ts and start_ts <= timestamp_utc < end_ts:
            return True, reason
    return False, ""


def _coerce_halt_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _to_utc(value)
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _to_utc(parsed)


__all__ = [
    "AFTER_HOURS_CLOSE",
    "AFTER_HOURS_SESSION",
    "CALENDAR_END_YEAR",
    "CALENDAR_START_YEAR",
    "CLOSED_HOLIDAY_STATUS",
    "CLOSED_SESSION",
    "CLOSED_WEEKEND_STATUS",
    "CONFLICTING_CALENDAR_EVIDENCE_STATUS",
    "EARLY_CLOSE",
    "EARLY_CLOSE_STATUS",
    "EASTERN",
    "FALLBACK_CALENDAR_USED_STATUS",
    "HALTED_SESSION",
    "OUTSIDE_AUTHORITY_RANGE_STATUS",
    "PRE_MARKET_OPEN",
    "PRE_MARKET_SESSION",
    "REGULAR_SESSION",
    "REGULAR_SESSION_STATUS",
    "RTH_CLOSE",
    "RTH_OPEN",
    "RTH_SESSION",
    "SPECIAL_CLOSURE_STATUS",
    "SPECIAL_FULL_CLOSURES",
    "MarketSessionContext",
    "calendar_status_for_date",
    "exchange_session_context",
    "expected_rth_timestamps",
    "is_rth_timestamp",
    "is_trading_session",
    "next_trading_session",
    "nyse_early_closes",
    "nyse_holidays",
    "previous_trading_session",
    "rth_close_for_date",
    "session_type",
    "trading_sessions",
]
