from __future__ import annotations

import inspect

from core.research.ml.audits import data_adjustment_validation
from core.research.ml.audits.benchmark_relative_validation import (
    build_benchmark_relative_validation,
)
from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)
from core.research.ml.audits.data_adjustment_validation import (
    build_clean_data_replay,
    build_data_adjustment_audit,
    build_independent_period_validation,
    detect_split_like_jumps,
)


def test_split_like_jump_detection_flags_common_split_ratio():
    rows = [
        {"date": "2024-01-01", "close": 100.0},
        {"date": "2024-01-02", "close": 50.0},
    ]

    events = detect_split_like_jumps("TEST", rows, suspicious_daily_return_abs=0.40)

    assert len(events) == 1
    assert events[0]["event_type"] == "split_like_jump"
    assert events[0]["split_like_factor"] == 2.0


def test_suspicious_rows_are_excluded_in_clean_replay():
    champion = _champion_payload([0.50, 0.04, 0.04])
    selected = _selected_optimizer_payload([0.40, 0.03, 0.03])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )
    adjustment = {
        "suspicious_rebalance_dates": ["2024-01-01"],
        "adjusted_price_status": "known_adjusted",
    }

    replay = build_clean_data_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjustment_audit=adjustment,
        closes_by_symbol=_closes(),
        validation_config={
            "max_top_5_date_profit_share": 1.0,
            "max_drawdown_worse_than_spy": 1.0,
        },
    )

    exact = replay["candidates"]["exact_champion_replay"]
    clean_rows = replay["clean_canonical_replay"]["candidates"][
        "exact_champion_replay"
    ]["rows"]

    assert exact["excluded_period_count"] == 1
    assert exact["clean_canonical_return"] < exact["raw_canonical_return"]
    assert [
        row["rebalance_date"] for row in clean_rows
        if row["included_in_canonical"]
    ] == ["2024-01-03", "2024-01-05"]


def test_unknown_adjusted_status_blocks_promotion():
    canonical = build_canonical_replay(
        selected_optimizer=_selected_optimizer_payload([0.08] * 6),
        champion_audit=_champion_payload([0.08] * 6),
    )

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=_closes(period_count=6, benchmark_return=0.01),
        validation_config={
            "max_top_5_date_profit_share": 1.0,
            "max_drawdown_worse_than_spy": 1.0,
            "allow_unknown_adjusted_price_status": False,
        },
        external_reports={
            "data_adjustment_audit": {
                "adjusted_price_status": "unknown",
                "candidate_dependencies": {},
                "promotion_gate": {
                    "adjusted_price_status_acceptable": False,
                },
            }
        },
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["gates"]["adjusted_price_status"] is False
    assert "adjusted_price_status" in exact["failed_gates"]
    assert exact["promotion_candidate_status"] == "blocked"


def test_too_few_independent_periods_blocks_promotion():
    canonical = build_canonical_replay(
        selected_optimizer=_selected_optimizer_payload([0.08] * 6),
        champion_audit=_champion_payload([0.08] * 6),
    )
    independent = build_independent_period_validation(
        canonical_replay=canonical,
        validation_config={"min_independent_periods": 36},
    )

    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report={"flagged_rebalance_dates": []},
        closes_by_symbol=_closes(period_count=6, benchmark_return=0.01),
        validation_config={
            "max_top_5_date_profit_share": 1.0,
            "max_drawdown_worse_than_spy": 1.0,
            "min_independent_periods": 36,
        },
        external_reports={
            "independent_period_validation": independent,
        },
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert independent["independent_canonical_period_count"] == 6
    assert independent["gate"]["passed"] is False
    assert exact["gates"]["minimum_independent_periods"] is False
    assert exact["promotion_candidate_status"] == "blocked"


def test_data_adjustment_audit_tracks_candidate_dependencies():
    champion = _champion_payload([0.50, 0.04, 0.04])
    selected = _selected_optimizer_payload([0.40, 0.03, 0.03])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    audit = build_data_adjustment_audit(
        symbol_rows_by_symbol={
            "AAA": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 50.0},
            ],
        },
        canonical_replay=canonical,
        champion_audit=champion,
        audit_config={
            "inspect_symbols": ["AAA"],
            "suspicious_daily_return_abs": 0.40,
            "allow_unknown_adjusted_price_status": False,
        },
    )

    dependency = audit["candidate_dependencies"]["exact_champion_replay"]
    assert audit["adjusted_price_status"] == "appears_unadjusted"
    assert dependency["suspicious_dependency_count"] == 1
    assert dependency["suspicious_rebalance_dates"] == ["2024-01-01"]


def test_data_adjustment_validation_has_no_operational_imports_or_references():
    source = inspect.getsource(data_adjustment_validation)

    assert "broker" not in source
    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "order_execution" not in source


def _champion_payload(returns: list[float]) -> dict:
    rows = []
    for index, value in enumerate(returns):
        start_day = 1 + index * 2
        end_day = start_day + 1
        rows.append({
            "rebalance_date": f"2024-01-{start_day:02d}",
            "outcome_end_date": f"2024-01-{end_day:02d}",
            "period_return": value,
            "exposure_target": 1.0,
            "selected_symbols": ["AAA", "BBB"],
            "target_weights": {"AAA": 0.5, "BBB": 0.5},
            "symbol_return_anomalies": [],
        })
    return {
        "exact_champion_replay": {
            "period_rows": rows,
        }
    }


def _selected_optimizer_payload(returns: list[float]) -> dict:
    rows = []
    for index, value in enumerate(returns):
        start_day = 1 + index * 2
        rows.append({
            "rebalance_date": f"2024-01-{start_day:02d}",
            "period_return": value,
            "exposure": 1.0,
            "turnover": 0.0,
            "cost": 0.0,
            "net_return": value,
        })
    return {"rows": rows}


def _closes(
    *,
    period_count: int = 3,
    benchmark_return: float = 0.01,
) -> dict[str, dict[str, float]]:
    closes: dict[str, dict[str, float]] = {
        "SPY": {},
        "QQQ": {},
        "AAA": {},
        "BBB": {},
    }
    for index in range(period_count):
        start_day = 1 + index * 2
        end_day = start_day + 1
        start = f"2024-01-{start_day:02d}"
        end = f"2024-01-{end_day:02d}"
        for symbol in closes:
            period_return = 0.02 if symbol in {"AAA", "BBB"} else benchmark_return
            closes[symbol][start] = 100.0
            closes[symbol][end] = 100.0 * (1.0 + period_return)
    return closes


def _candidate(payload: dict, name: str) -> dict:
    return next(
        row for row in payload["candidates"]
        if row["candidate_name"] == name
    )
