from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

CALENDAR_START_YEAR = 2016
CALENDAR_END_YEAR = 2026
SPECIAL_FULL_CLOSURES = {
    date(2018, 12, 5),  # National day of mourning for President George H.W. Bush.
    date(2025, 1, 9),  # National day of mourning for President Jimmy Carter.
}


def session_type(timestamp: datetime) -> str:
    local = _to_eastern(timestamp)
    close = rth_close_for_date(local.date())
    if local.time() < RTH_OPEN:
        return "pre_market"
    if close is not None and RTH_OPEN <= local.time() < close:
        return "rth"
    return "after_hours"


def is_rth_timestamp(timestamp: datetime) -> bool:
    return session_type(timestamp) == "rth"


def is_trading_session(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def trading_sessions(start: date, end: date) -> list[date]:
    if end < start:
        return []
    current = start
    output: list[date] = []
    while current <= end:
        if is_trading_session(current):
            output.append(current)
        current += timedelta(days=1)
    return output


def previous_trading_session(day: date) -> date:
    current = day - timedelta(days=1)
    while not is_trading_session(current):
        current -= timedelta(days=1)
    return current


def next_trading_session(day: date) -> date:
    current = day + timedelta(days=1)
    while not is_trading_session(current):
        current += timedelta(days=1)
    return current


def rth_close_for_date(day: date) -> time | None:
    if not is_trading_session(day):
        return None
    if day in nyse_early_closes(day.year):
        return EARLY_CLOSE
    return RTH_CLOSE


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
    if year < CALENDAR_START_YEAR or year > CALENDAR_END_YEAR:
        raise ValueError(f"NYSE calendar is implemented for {CALENDAR_START_YEAR}-{CALENDAR_END_YEAR}")
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day.
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday / Presidents Day.
        _good_friday(year),
        _last_weekday(year, 5, 0),  # Memorial Day.
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),  # Labor Day.
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving.
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    if date(year + 1, 1, 1).weekday() == 5:
        holidays.add(date(year, 12, 31))
    holidays.update(day for day in SPECIAL_FULL_CLOSURES if day.year == year)
    return {day for day in holidays if day.year == year}


def nyse_early_closes(year: int) -> set[date]:
    if year < CALENDAR_START_YEAR or year > CALENDAR_END_YEAR:
        raise ValueError(f"NYSE calendar is implemented for {CALENDAR_START_YEAR}-{CALENDAR_END_YEAR}")
    candidates = {
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),  # Day after Thanksgiving.
        date(year, 12, 24),  # Christmas Eve when it is a trading day.
    }
    july_4 = date(year, 7, 4)
    if july_4.weekday() in {1, 2, 3, 4}:  # Tue-Fri holiday has prior early close.
        candidates.add(july_4 - timedelta(days=1))
    return {day for day in candidates if is_trading_session(day)}


def _to_eastern(value: datetime) -> datetime:
    return _to_utc(value).astimezone(EASTERN)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
    # Anonymous Gregorian computus for Easter Sunday, minus two days.
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
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)
