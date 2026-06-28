from __future__ import annotations

import inspect

from core.research.ml import adjusted_data_comparison
from core.research.ml.adjusted_data_comparison import (
    build_adjusted_data_comparison,
    build_adjusted_price_replay,
    detect_split_like_adjustment_ratio,
)
from core.research.ml.benchmark_relative_validation import (
    build_benchmark_relative_validation,
)
from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)
from infrastructure.data.adjusted_price_csv_data_feed import (
    LocalAdjustedPriceCsvDataFeed,
)


def test_raw_vs_adjusted_comparison_detects_split_distortion():
    canonical = build_canonical_replay(
        selected_optimizer=_selected_optimizer_payload([0.40]),
        champion_audit=_champion_payload([0.50]),
    )

    comparison = build_adjusted_data_comparison(
        raw_rows_by_symbol={
            "AAA": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 50.0},
            ],
        },
        adjusted_rows_by_symbol={
            "AAA": [
                {"date": "2024-01-01", "adjusted_close": 50.0},
                {"date": "2024-01-02", "adjusted_close": 50.0},
            ],
        },
        canonical_replay=canonical,
        comparison_config={
            "inspect_symbols": ["AAA"],
            "suspicious_daily_return_abs": 0.40,
        },
    )

    distorted = [
        row for row in comparison["rows"]
        if row["split_like_distortion"]
    ]
    dependency = comparison["candidate_dependencies"]["exact_champion_replay"]

    assert comparison["adjusted_source"]["available_status"] == "available"
    assert len(distorted) == 1
    assert distorted[0]["raw_close"] == 50.0
    assert distorted[0]["adjusted_close"] == 50.0
    assert distorted[0]["adjustment_ratio_split_like_factor"] == 2.0
    assert dependency["raw_adjusted_distortion_dependency_count"] == 1


def test_split_like_ratio_detection_works():
    assert detect_split_like_adjustment_ratio(0.5, 1.0) == 2.0
    assert detect_split_like_adjustment_ratio(1.0, 1.02) is None


def test_adjusted_csv_feed_preserves_raw_source_separately(tmp_path):
    data_dir = tmp_path / "adjusted"
    data_dir.mkdir()
    (data_dir / "AAA.csv").write_text(
        "date,raw_close,adjusted_close\n"
        "2024-01-01,100,50\n"
        "2024-01-02,50,50\n",
        encoding="utf-8",
    )

    points = LocalAdjustedPriceCsvDataFeed(str(data_dir)).get_adjusted_prices("AAA")
    comparison = build_adjusted_data_comparison(
        raw_rows_by_symbol={
            "AAA": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 50.0},
            ],
        },
        adjusted_rows_by_symbol={"AAA": points},
        canonical_replay={},
        comparison_config={"inspect_symbols": ["AAA"]},
    )

    assert points[0].raw_close == 100.0
    assert points[0].adjusted_close == 50.0
    assert comparison["raw_source"]["preserved_separately"] is True
    assert comparison["rows"][0]["raw_close"] == 100.0
    assert comparison["rows"][0]["adjusted_close"] == 50.0


def test_adjusted_price_replay_uses_adjusted_closes_when_available():
    champion = _champion_payload([0.50])
    selected = _selected_optimizer_payload([0.40])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )
    comparison = build_adjusted_data_comparison(
        raw_rows_by_symbol={
            "AAA": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 50.0},
            ],
            "BBB": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 110.0},
            ],
            "SPY": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 101.0},
            ],
            "QQQ": [
                {"date": "2024-01-01", "close": 100.0},
                {"date": "2024-01-02", "close": 101.0},
            ],
        },
        adjusted_rows_by_symbol={
            "AAA": [
                {"date": "2024-01-01", "adjusted_close": 50.0},
                {"date": "2024-01-02", "adjusted_close": 50.0},
            ],
            "BBB": [
                {"date": "2024-01-01", "adjusted_close": 100.0},
                {"date": "2024-01-02", "adjusted_close": 110.0},
            ],
            "SPY": [
                {"date": "2024-01-01", "adjusted_close": 100.0},
                {"date": "2024-01-02", "adjusted_close": 101.0},
            ],
            "QQQ": [
                {"date": "2024-01-01", "adjusted_close": 100.0},
                {"date": "2024-01-02", "adjusted_close": 101.0},
            ],
        },
        canonical_replay=canonical,
        comparison_config={"inspect_symbols": ["AAA"]},
    )

    replay = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=comparison,
        adjusted_closes_by_symbol={
            "AAA": {"2024-01-01": 50.0, "2024-01-02": 50.0},
            "BBB": {"2024-01-01": 100.0, "2024-01-02": 110.0},
            "SPY": {"2024-01-01": 100.0, "2024-01-02": 101.0},
            "QQQ": {"2024-01-01": 100.0, "2024-01-02": 101.0},
        },
        validation_config={
            "max_top_5_date_profit_share": 1.0,
            "max_drawdown_worse_than_spy": 1.0,
            "min_independent_periods": 1,
        },
    )

    exact = replay["candidates"]["exact_champion_replay"]

    assert replay["adjusted_source_available"] is True
    assert round(exact["adjusted_canonical_return"], 6) == 0.05


def test_missing_adjusted_symbol_blocks_adjusted_period_fail_closed():
    champion = _champion_payload([0.25])
    selected = _selected_optimizer_payload([0.25])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    replay = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={
            "AAA": {"2024-01-01": 100.0, "2024-01-02": 200.0},
        },
        validation_config={"min_independent_periods": 1},
    )
    exact = replay["candidates"]["exact_champion_replay"]

    assert exact["adjusted_canonical_return"] is None
    assert exact["adjusted_price_replay_verdict"] == "blocked"
    assert exact["adjusted_full_symbol_coverage"] is False
    assert exact["invalid_adjusted_period_count"] == 1
    assert exact["valid_adjusted_period_count"] == 0
    assert exact["missing_adjusted_symbols"] == ["BBB"]
    assert exact["fail_closed_reason"] == "missing_adjusted_prices_for_selected_symbols"


def test_missing_adjusted_symbol_does_not_increase_return_by_dropping_loser():
    champion = _champion_payload([-0.25])
    selected = _selected_optimizer_payload([-0.25])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    replay = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={
            "AAA": {"2024-01-01": 100.0, "2024-01-02": 200.0},
        },
        validation_config={"min_independent_periods": 1},
    )
    exact = replay["candidates"]["exact_champion_replay"]

    assert exact["coverage_valid_adjusted_canonical_return"] is None
    assert exact["adjusted_canonical_return"] is None
    assert replay["adjusted_canonical_replay"]["candidates"][
        "exact_champion_replay"
    ]["canonical_continuous_equity"]["row_count"] == 0


def test_optimizer_positive_exposure_empty_selection_fails_closed():
    selected = {
        "rows": [
            {
                "rebalance_date": "2024-01-01",
                "period_return": 0.25,
                "exposure": 0.75,
                "turnover": 0.0,
                "cost": 0.0,
                "net_return": 0.1875,
            }
        ]
    }
    champion = {
        "exact_champion_replay": {
            "period_rows": [
                {
                    "rebalance_date": "2024-01-01",
                    "outcome_end_date": "2024-01-02",
                    "period_return": 0.0,
                    "exposure_target": 0.0,
                    "selected_symbols": [],
                    "target_weights": {},
                    "symbol_return_anomalies": [],
                }
            ]
        }
    }
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    replay = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={},
        validation_config={"min_independent_periods": 1},
    )
    optimizer = replay["candidates"][
        "selected_bayesian_optimizer_diagnostic_policy"
    ]

    assert optimizer["adjusted_canonical_return"] is None
    assert optimizer["coverage_valid_adjusted_canonical_return"] is None
    assert optimizer["empty_selection_with_positive_exposure_count"] == 1
    assert optimizer["affected_dates"] == ["2024-01-01"]
    assert optimizer["empty_selection_resolution"] == "invalidated"
    assert optimizer["fail_closed_reason"] == "empty_selection_with_positive_exposure"
    assert "empty_selection_with_positive_exposure" in optimizer["failed_gates"]
    assert replay["adjusted_canonical_replay"]["candidates"][
        "selected_bayesian_optimizer_diagnostic_policy"
    ]["canonical_continuous_equity"]["row_count"] == 0


def test_full_adjusted_coverage_preserves_selected_symbol_composition():
    champion = _champion_payload([0.10])
    selected = _selected_optimizer_payload([0.10])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    replay = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={
            "AAA": {"2024-01-01": 100.0, "2024-01-02": 120.0},
            "BBB": {"2024-01-01": 100.0, "2024-01-02": 80.0},
            "SPY": {"2024-01-01": 100.0, "2024-01-02": 101.0},
            "QQQ": {"2024-01-01": 100.0, "2024-01-02": 101.0},
        },
        validation_config={"min_independent_periods": 1},
    )
    exact = replay["candidates"]["exact_champion_replay"]
    adjusted_row = replay["adjusted_canonical_replay"]["candidates"][
        "exact_champion_replay"
    ]["rows"][0]

    assert exact["adjusted_full_symbol_coverage"] is True
    assert exact["adjusted_coverage_ratio"] == 1.0
    assert adjusted_row["selected_symbols"] == ["AAA", "BBB"]
    assert round(adjusted_row["period_return"], 12) == 0.0
    assert round(exact["coverage_valid_adjusted_canonical_return"], 12) == 0.0


def test_adjusted_replay_raw_fallback_is_off_by_default():
    replay = build_adjusted_price_replay(
        canonical_replay={},
        champion_audit={},
        selected_optimizer={},
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={},
        validation_config={},
    )

    assert replay["coverage_rules"]["allow_raw_fallback"] is False
    assert replay["coverage_rules"]["missing_symbol_policy"] == "fail_closed"
    assert replay["coverage_rules"]["require_full_adjusted_coverage"] is True


def test_fallback_raw_only_works_when_explicitly_enabled():
    champion = _champion_payload([0.25])
    selected = _selected_optimizer_payload([0.25])
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    blocked = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={
            "AAA": {"2024-01-01": 100.0, "2024-01-02": 200.0},
        },
        raw_closes_by_symbol={
            "BBB": {"2024-01-01": 100.0, "2024-01-02": 50.0},
        },
        validation_config={"min_independent_periods": 1},
    )
    fallback = build_adjusted_price_replay(
        canonical_replay=canonical,
        champion_audit=champion,
        selected_optimizer=selected,
        adjusted_comparison=_available_adjusted_comparison(),
        adjusted_closes_by_symbol={
            "AAA": {"2024-01-01": 100.0, "2024-01-02": 200.0},
        },
        raw_closes_by_symbol={
            "BBB": {"2024-01-01": 100.0, "2024-01-02": 50.0},
        },
        validation_config={
            "min_independent_periods": 1,
            "adjusted_replay": {"missing_symbol_policy": "fallback_raw"},
        },
    )

    blocked_exact = blocked["candidates"]["exact_champion_replay"]
    fallback_exact = fallback["candidates"]["exact_champion_replay"]
    adjusted_row = fallback["adjusted_canonical_replay"]["candidates"][
        "exact_champion_replay"
    ]["rows"][0]

    assert blocked_exact["adjusted_canonical_return"] is None
    assert fallback["coverage_rules"]["missing_symbol_policy"] == "fallback_raw"
    assert fallback_exact["coverage_valid_adjusted_canonical_return"] == 0.25
    assert fallback_exact["adjusted_canonical_return"] == 0.25
    assert fallback_exact["adjusted_full_symbol_coverage"] is False
    assert fallback_exact["raw_fallback_symbols"] == ["BBB"]
    assert adjusted_row["selected_symbols"] == ["AAA", "BBB"]


def test_missing_adjusted_data_blocks_promotion():
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
        },
        external_reports={
            "adjusted_data_comparison": {
                "adjusted_source": {
                    "available_status": "missing",
                    "acceptable": False,
                },
                "candidate_dependencies": {},
            }
        },
    )
    exact = _candidate(payload, "exact_champion_replay")

    assert exact["gates"]["adjusted_source_available"] is False
    assert "adjusted_source_available" in exact["failed_gates"]
    assert exact["promotion_candidate_status"] == "blocked"


def test_adjusted_data_comparison_has_no_operational_imports_or_references():
    source = inspect.getsource(adjusted_data_comparison)

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
    return {"exact_champion_replay": {"period_rows": rows}}


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
    period_count: int,
    benchmark_return: float,
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


def _available_adjusted_comparison() -> dict:
    return {
        "adjusted_source": {
            "available_status": "available",
            "acceptable": True,
        },
        "candidate_dependencies": {},
    }
