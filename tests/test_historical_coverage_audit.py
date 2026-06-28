from __future__ import annotations

import inspect

from core.research.ml import historical_coverage_audit
from core.research.ml.historical_coverage_audit import (
    build_historical_coverage_audit,
)


def test_historical_coverage_identifies_prediction_artifact_limitation():
    payload = build_historical_coverage_audit(
        raw_price_ranges=[
            _range("SPY", "2005-01-03", "2026-06-18", row_count=5000),
        ],
        adjusted_price_ranges=[
            _range("SPY", "1993-01-29", "2026-06-25", row_count=8000),
        ],
        source_prediction_ranges=[
            _prediction_range("model_a", "2021-01-04", "2026-04-20"),
        ],
        meta_prediction_range=_prediction_range(
            "meta",
            "2024-07-15",
            "2026-04-20",
        ),
        canonical_replay=_canonical_replay(
            start="2024-07-15",
            count=11,
            diagnostic_count=104,
        ),
        adjusted_price_replay=_adjusted_replay(valid_independent=9),
        config={"min_independent_periods": 36},
    )

    assert payload["historical_bottleneck"]["limiting_layer"] == (
        "meta_or_canonical_artifacts"
    )
    assert "prediction_artifacts_start_after_price_history" in payload[
        "historical_bottleneck"
    ]["reasons"]
    assert "meta_or_canonical_artifacts_start_after_source_predictions" in payload[
        "historical_bottleneck"
    ]["reasons"]
    assert payload["recommendations"]["regenerate_base_artifacts"] is True
    assert payload["full_model_rerun_required"] is True


def test_historical_coverage_identifies_price_data_limitation():
    payload = build_historical_coverage_audit(
        raw_price_ranges=[],
        adjusted_price_ranges=[],
        source_prediction_ranges=[
            _prediction_range("model_a", "2021-01-04", "2026-04-20"),
        ],
        meta_prediction_range=_prediction_range(
            "meta",
            "2021-01-04",
            "2026-04-20",
        ),
        canonical_replay=_canonical_replay(
            start="2024-07-15",
            count=11,
            diagnostic_count=104,
        ),
        adjusted_price_replay=_adjusted_replay(valid_independent=9),
        config={"min_independent_periods": 36},
    )

    assert payload["historical_bottleneck"]["limiting_layer"] == "price_data"
    assert "adjusted_prices_missing" in payload["historical_bottleneck"]["reasons"]
    assert payload["recommendations"]["regenerate_base_artifacts"] is False


def test_historical_coverage_reports_required_history_for_targets():
    payload = build_historical_coverage_audit(
        raw_price_ranges=[_range("SPY", "2005-01-03", "2026-06-18")],
        adjusted_price_ranges=[_range("SPY", "1993-01-29", "2026-06-25")],
        source_prediction_ranges=[
            _prediction_range("model_a", "2021-01-04", "2026-04-20"),
        ],
        meta_prediction_range=_prediction_range(
            "meta",
            "2024-07-15",
            "2026-04-20",
        ),
        canonical_replay=_canonical_replay(
            start="2024-07-15",
            count=11,
            diagnostic_count=104,
        ),
        adjusted_price_replay=_adjusted_replay(valid_independent=9),
        config={"min_independent_periods": 36},
    )

    assert payload["history_required_for_targets"]["36"][
        "estimated_required_start_date"
    ]
    assert payload["possible_leakage_safe_non_overlap_windows"][
        "exact_champion_replay"
    ] == 11


def test_historical_coverage_audit_has_no_operational_imports():
    source = inspect.getsource(historical_coverage_audit)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _range(
    symbol: str,
    start: str,
    end: str,
    *,
    row_count: int = 100,
) -> dict:
    return {
        "symbol": symbol,
        "available": True,
        "row_count": row_count,
        "earliest_date": start,
        "latest_date": end,
    }


def _prediction_range(name: str, start: str, end: str) -> dict:
    return {
        "name": name,
        "available": True,
        "row_count": 10,
        "unique_rebalance_dates": 10,
        "earliest_date": start,
        "latest_date": end,
    }


def _canonical_replay(
    *,
    start: str,
    count: int,
    diagnostic_count: int,
) -> dict:
    rows = []
    for index in range(count):
        month = 7 + index
        year = 2024 + ((month - 1) // 12)
        month = ((month - 1) % 12) + 1
        rows.append({
            "rebalance_date": f"{year}-{month:02d}-15",
            "outcome_end_date": f"{year}-{month:02d}-28",
            "included_in_canonical": True,
        })
    rows[0]["rebalance_date"] = start
    return {
        "candidates": {
            "exact_champion_replay": {
                "rows": rows,
                "canonical_continuous_equity": {"row_count": count},
                "diagnostic_period_grid": {"row_count": diagnostic_count},
            },
            "selected_bayesian_optimizer_diagnostic_policy": {
                "rows": rows,
                "canonical_continuous_equity": {"row_count": count},
                "diagnostic_period_grid": {"row_count": diagnostic_count},
            },
        }
    }


def _adjusted_replay(*, valid_independent: int) -> dict:
    return {
        "candidates": {
            "exact_champion_replay": {
                "valid_adjusted_independent_period_count": valid_independent,
                "valid_adjusted_period_count": 84,
                "invalid_adjusted_period_count": 20,
            }
        }
    }
