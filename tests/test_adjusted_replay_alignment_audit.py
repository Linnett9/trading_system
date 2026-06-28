from __future__ import annotations

import inspect

from core.research.ml.audits import adjusted_replay_alignment_audit
from core.research.ml.audits.adjusted_replay_alignment_audit import (
    build_adjusted_replay_alignment_audit,
)
from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)


def test_identical_raw_and_adjusted_prices_produce_identical_alignment():
    canonical = _canonical(period_return=0.10)

    audit = build_adjusted_replay_alignment_audit(
        canonical_replay=canonical,
        adjusted_price_replay={"adjusted_canonical_replay": canonical},
        raw_closes_by_symbol=_closes(start=100.0, end=110.0),
        adjusted_closes_by_symbol=_closes(start=100.0, end=110.0),
    )

    assert audit["alignment"]["aligned_correctly"] is True
    assert audit["alignment"]["explanation_verdict"] == "aligned_no_material_adjusted_delta"
    assert audit["alignment"]["large_return_delta_row_count"] == 0
    assert all(row["return_delta"] == 0.0 for row in audit["rows"])


def test_split_adjusted_prices_change_returns_only_through_adjustment_ratio():
    raw = _canonical(period_return=-0.50)
    adjusted = _canonical(period_return=0.0)

    audit = build_adjusted_replay_alignment_audit(
        canonical_replay=raw,
        adjusted_price_replay={"adjusted_canonical_replay": adjusted},
        raw_closes_by_symbol=_closes(start=100.0, end=50.0),
        adjusted_closes_by_symbol=_closes(start=50.0, end=50.0),
        audit_config={"return_delta_abs_threshold": 0.05},
    )
    row = _row(audit, "exact_champion_replay", "AAA")

    assert row["raw_return"] == -0.5
    assert row["adjusted_return"] == 0.0
    assert row["return_delta"] == 0.5
    assert row["adjustment_ratio_start"] == 0.5
    assert row["adjustment_ratio_end"] == 1.0
    assert row["adjusted_return_matches_ratio"] is True
    assert row["adjustment_ratio_jump"] is True
    assert row["candidate_net_return_delta_above_threshold"] is True
    assert row["unexplained_adjusted_delta"] is False
    assert audit["alignment"]["explanation_verdict"] == (
        "aligned_large_deltas_explained_by_adjustment_ratios"
    )


def test_missing_adjusted_rows_are_reported_not_silently_filled():
    canonical = _canonical(period_return=0.10)
    adjusted = {"candidates": {}}

    audit = build_adjusted_replay_alignment_audit(
        canonical_replay=canonical,
        adjusted_price_replay={"adjusted_canonical_replay": adjusted},
        raw_closes_by_symbol=_closes(start=100.0, end=110.0),
        adjusted_closes_by_symbol={"AAA": {"2024-01-01": 100.0}},
    )
    row = _row(audit, "exact_champion_replay", "AAA")

    assert row["missing_adjusted_prices"] is True
    assert row["date_misalignment"] is True
    assert row["adjusted_return"] is None
    assert "missing_adjusted_prices" in audit["red_flags"]
    assert audit["alignment"]["aligned_correctly"] is False


def test_replay_dates_symbols_exposures_and_non_overlap_flags_are_identical():
    canonical = _canonical(period_return=0.10)

    audit = build_adjusted_replay_alignment_audit(
        canonical_replay=canonical,
        adjusted_price_replay={"adjusted_canonical_replay": canonical},
        raw_closes_by_symbol=_closes(start=100.0, end=110.0),
        adjusted_closes_by_symbol=_closes(start=100.0, end=110.0),
    )
    checks = audit["alignment"]["checks"]

    assert checks["same_rebalance_dates"] is True
    assert checks["same_selected_symbols"] is True
    assert checks["same_exposure_path"] is True
    assert checks["same_label_windows"] is True
    assert checks["same_non_overlap_periods"] is True


def test_adjusted_replay_alignment_has_no_operational_imports_or_references():
    source = inspect.getsource(adjusted_replay_alignment_audit)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _canonical(period_return: float) -> dict:
    champion = {
        "exact_champion_replay": {
            "period_rows": [
                {
                    "rebalance_date": "2024-01-01",
                    "outcome_end_date": "2024-01-10",
                    "period_return": period_return,
                    "exposure_target": 1.0,
                    "selected_symbols": ["AAA"],
                    "target_weights": {"AAA": 1.0},
                    "symbol_return_anomalies": [],
                }
            ]
        }
    }
    selected = {
        "rows": [
            {
                "rebalance_date": "2024-01-01",
                "period_return": period_return,
                "exposure": 1.0,
                "turnover": 0.0,
                "cost": 0.0,
                "net_return": period_return,
            }
        ]
    }
    return build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )


def _closes(*, start: float, end: float) -> dict[str, dict[str, float]]:
    return {"AAA": {"2024-01-01": start, "2024-01-10": end}}


def _row(audit: dict, candidate: str, symbol: str) -> dict:
    return next(
        row for row in audit["rows"]
        if row["candidate"] == candidate and row["symbol"] == symbol
    )
