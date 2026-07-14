from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from infrastructure.data.market_sessions import (
    EASTERN,
    next_trading_session,
    rth_close_for_date,
    trading_sessions,
)


DECISION_GRID_VERSION = "stock_level_daily_decision_grid_v1"
EXCHANGE_CALENDAR_IDENTITY = (
    "infrastructure.data.market_sessions.nyse_like_rth_2016_2026_v1"
)


@dataclass(frozen=True)
class DecisionGrid:
    dates: list[str]
    row_metadata_by_date: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def resolve_decision_grid(
    *,
    expanded_rows: list[dict[str, str]],
    artifact_rows: list[dict[str, str]],
    symbols: list[str],
    prepared_symbol_data: dict[str, dict[str, Any]],
    market_data: dict[str, Any],
    frequency: str = "source",
    start_date: str | None = None,
    end_date: str | None = None,
    max_sessions: int | None = None,
    min_history_sessions: int = 1,
) -> DecisionGrid:
    source_dates = _artifact_dates(artifact_rows) or _expanded_dates(expanded_rows)
    if str(frequency).lower() != "daily":
        return _source_grid(source_dates)
    if not source_dates and not start_date and not end_date:
        return _daily_grid([], {}, frequency="daily", min_history_sessions=min_history_sessions)
    start = date.fromisoformat(start_date or source_dates[0])
    end = date.fromisoformat(end_date or source_dates[-1])
    market_close_dates = set(market_data.get("close_dates", []))
    close_index_by_symbol = {
        symbol: prepared_symbol_data.get(symbol, {}).get("close_index_by_date", {})
        for symbol in symbols
    }
    sessions = [
        session.isoformat()
        for session in trading_sessions(start, end)
        if _eligible_session(
            session.isoformat(),
            symbols=symbols,
            close_index_by_symbol=close_index_by_symbol,
            market_close_dates=market_close_dates,
            min_history_sessions=min_history_sessions,
        )
    ]
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    context = _context_asof_by_date(sessions, expanded_rows)
    return _daily_grid(
        sessions,
        context,
        frequency="daily",
        min_history_sessions=min_history_sessions,
    )


def decision_timing_metadata(session: str) -> dict[str, Any]:
    day = date.fromisoformat(session)
    close = rth_close_for_date(day)
    if close is None:
        raise ValueError(f"Not an exchange trading session: {session}")
    close_utc = datetime.combine(day, close, tzinfo=EASTERN).astimezone(timezone.utc)
    decision = close_utc + timedelta(minutes=5)
    first_actionable = next_trading_session(day).isoformat()
    return {
        "decision_session_date": session,
        "feature_data_cutoff_timestamp": close_utc.isoformat().replace("+00:00", "Z"),
        "decision_timestamp": decision.isoformat().replace("+00:00", "Z"),
        "first_actionable_session": first_actionable,
        "decision_grid_version": DECISION_GRID_VERSION,
        "exchange_calendar_identity": EXCHANGE_CALENDAR_IDENTITY,
    }


def _daily_grid(
    dates: list[str],
    context_by_date: dict[str, dict[str, Any]],
    *,
    frequency: str,
    min_history_sessions: int,
) -> DecisionGrid:
    metadata_by_date = {}
    for session in dates:
        metadata_by_date[session] = {
            **decision_timing_metadata(session),
            "decision_frequency": frequency,
            "target_horizon_trading_days": 10,
            "overlapping_targets": frequency == "daily",
            "required_purge_horizon_trading_days": 10,
            **context_by_date.get(session, _empty_context_metadata()),
        }
    identity_payload = {
        "version": DECISION_GRID_VERSION,
        "frequency": frequency,
        "exchange_calendar_identity": EXCHANGE_CALENDAR_IDENTITY,
        "dates": dates,
        "min_history_sessions": min_history_sessions,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for metadata in metadata_by_date.values():
        metadata["decision_grid_identity"] = identity
    return DecisionGrid(
        dates=dates,
        row_metadata_by_date=metadata_by_date,
        audit={
            "decision_frequency": frequency,
            "decision_grid_version": DECISION_GRID_VERSION,
            "decision_grid_identity": identity,
            "exchange_calendar_identity": EXCHANGE_CALENDAR_IDENTITY,
            "decision_date_count": len(dates),
            "first_decision_date": dates[0] if dates else None,
            "last_decision_date": dates[-1] if dates else None,
            "target_horizon_trading_days": 10,
            "overlapping_targets": frequency == "daily",
            "required_purge_horizon_trading_days": 10,
            "minimum_history_sessions": min_history_sessions,
        },
    )


def _source_grid(dates: list[str]) -> DecisionGrid:
    metadata = {session: _source_timing_metadata(session) for session in dates}
    for index, session in enumerate(dates[:-1]):
        metadata[session]["first_actionable_session"] = dates[index + 1]
    payload = {
        "version": "stock_level_source_decision_grid_v1",
        "frequency": "source",
        "dates": dates,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for session_metadata in metadata.values():
        session_metadata.update(
            {
                "decision_grid_version": "stock_level_source_decision_grid_v1",
                "decision_grid_identity": identity,
                "decision_frequency": "source",
                "target_horizon_trading_days": 10,
                "overlapping_targets": False,
                "required_purge_horizon_trading_days": 10,
                **_empty_context_metadata(),
            }
        )
    return DecisionGrid(
        dates=dates,
        row_metadata_by_date=metadata,
        audit={
            "decision_frequency": "source",
            "decision_grid_version": "stock_level_source_decision_grid_v1",
            "decision_grid_identity": identity,
            "exchange_calendar_identity": EXCHANGE_CALENDAR_IDENTITY,
            "decision_date_count": len(dates),
            "first_decision_date": dates[0] if dates else None,
            "last_decision_date": dates[-1] if dates else None,
            "target_horizon_trading_days": 10,
            "overlapping_targets": False,
            "required_purge_horizon_trading_days": 10,
        },
    )


def _source_timing_metadata(session: str) -> dict[str, Any]:
    return {
        "decision_session_date": session,
        "feature_data_cutoff_timestamp": session,
        "decision_timestamp": session,
        "first_actionable_session": "",
        "decision_grid_version": "stock_level_source_decision_grid_v1",
        "exchange_calendar_identity": EXCHANGE_CALENDAR_IDENTITY,
    }


def _eligible_session(
    session: str,
    *,
    symbols: list[str],
    close_index_by_symbol: dict[str, dict[str, int]],
    market_close_dates: set[str],
    min_history_sessions: int,
) -> bool:
    if session not in market_close_dates:
        return False
    for symbol in symbols:
        index = close_index_by_symbol.get(symbol, {}).get(session)
        if index is not None and index >= min_history_sessions:
            return True
    return False


def _context_asof_by_date(
    sessions: list[str],
    expanded_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    sources = []
    for row in expanded_rows:
        source_date = row.get("rebalance_date") or row.get("feature_date")
        if source_date:
            sources.append((source_date, row))
    sources.sort(key=lambda item: item[0])
    output = {}
    index = 0
    last: tuple[str, dict[str, str]] | None = None
    for session in sessions:
        while index < len(sources) and sources[index][0] <= session:
            last = sources[index]
            index += 1
        if last is None:
            output[session] = _empty_context_metadata()
            continue
        source_date, row = last
        output[session] = {
            "context_source_timestamp": source_date,
            "context_age_calendar_days": (
                date.fromisoformat(session) - date.fromisoformat(source_date)
            ).days,
            "context_asof_join_direction": "backward",
            "context_asof_join_policy": "latest_source_timestamp_lte_decision_session",
            **{column: row.get(column, "") for column in _context_columns()},
        }
    return output


def _empty_context_metadata() -> dict[str, Any]:
    return {
        "context_source_timestamp": "",
        "context_age_calendar_days": "",
        "context_asof_join_direction": "none",
        "context_asof_join_policy": "latest_source_timestamp_lte_decision_session",
    }


def _context_columns() -> tuple[str, ...]:
    from core.research.ml.stock_level.prediction_artifacts.types import CONTEXT_COLUMNS

    return CONTEXT_COLUMNS


def _artifact_dates(rows: list[dict[str, str]]) -> list[str]:
    return sorted({
        str(row.get("rebalance_date") or row.get("date") or "")
        for row in rows
        if row.get("rebalance_date") or row.get("date")
    })


def _expanded_dates(rows: list[dict[str, str]]) -> list[str]:
    return sorted({
        str(row.get("rebalance_date") or row.get("feature_date") or "")
        for row in rows
        if row.get("rebalance_date") or row.get("feature_date")
    })
