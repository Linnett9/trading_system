from __future__ import annotations

import json

from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)
from core.research.ml.data_anomaly_quarantine import (
    build_anomaly_quarantine_report,
)
from core.research.ml.audits.profit_concentration_audit import (
    build_profit_concentration_audit,
)
from core.research.ml.stock_level.trading_research_leaderboard import (
    write_trading_research_leaderboard,
)


def test_canonical_replay_keeps_non_overlapping_dates_only():
    canonical = build_canonical_replay(
        selected_optimizer=_selected_optimizer_payload(),
        champion_audit=_champion_audit_payload(),
    )

    rows = canonical["candidates"]["selected_bayesian_optimizer_diagnostic_policy"][
        "rows"
    ]
    kept_dates = [
        row["rebalance_date"] for row in rows if row["included_in_canonical"]
    ]

    assert kept_dates == ["2024-01-01", "2024-01-15", "2024-02-01"]
    assert canonical["candidates"]["selected_bayesian_optimizer_diagnostic_policy"][
        "canonical_continuous_equity"
    ]["row_count"] == 3


def test_selected_optimizer_diagnostic_period_grid_matches_saved_path():
    selected = _selected_optimizer_payload()
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=_champion_audit_payload(),
    )

    reported = selected["rows"][-1]["equity"] - 1.0
    replayed = canonical["candidates"][
        "selected_bayesian_optimizer_diagnostic_policy"
    ]["diagnostic_period_grid"]["total_return"]

    assert abs(reported - replayed) < 1e-12


def test_optimizer_positive_exposure_empty_selection_is_invalidated():
    selected = {"rows": [_selected_row("2024-03-03", 0.20, 0.75)]}
    champion = {
        "exact_champion_replay": {
            "period_rows": [
                {
                    "rebalance_date": "2024-03-03",
                    "outcome_end_date": "2024-04-01",
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
    candidate = canonical["candidates"][
        "selected_bayesian_optimizer_diagnostic_policy"
    ]
    row = candidate["rows"][0]

    assert row["replay_valid"] is False
    assert row["replay_invalid_reason"] == "empty_selection_with_positive_exposure"
    assert row["included_in_canonical"] is False
    assert row["exclusion_reason"] == "empty_selection_with_positive_exposure"
    assert row["net_return"] == 0.0
    assert candidate["canonical_continuous_equity"]["row_count"] == 0
    assert candidate["empty_selection_with_positive_exposure_count"] == 1
    assert candidate["empty_selection_resolution"] == "invalidated"


def test_anomaly_exclusion_changes_results_deterministically():
    champion = _champion_audit_payload()
    selected = _selected_optimizer_payload()
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )

    first = build_anomaly_quarantine_report(
        champion_audit=champion,
        canonical_replay=canonical,
        selected_optimizer=selected,
        exclude_flagged=True,
    )
    second = build_anomaly_quarantine_report(
        champion_audit=champion,
        canonical_replay=canonical,
        selected_optimizer=selected,
        exclude_flagged=True,
    )

    baseline = canonical["candidates"][
        "selected_bayesian_optimizer_diagnostic_policy"
    ]["canonical_continuous_equity"]["total_return"]
    excluded = first["exclusion_preview"][
        "selected_bayesian_optimizer_diagnostic_policy"
    ]["canonical_continuous_return"]

    assert first["flagged_rebalance_dates"] == ["2024-01-01"]
    assert first["exclusion_preview"] == second["exclusion_preview"]
    assert excluded != baseline


def test_profit_concentration_removes_top_contributor():
    champion = _champion_audit_payload()
    selected = _selected_optimizer_payload()
    canonical = build_canonical_replay(
        selected_optimizer=selected,
        champion_audit=champion,
    )
    anomaly = build_anomaly_quarantine_report(
        champion_audit=champion,
        canonical_replay=canonical,
        selected_optimizer=selected,
    )

    concentration = build_profit_concentration_audit(
        canonical_replay=canonical,
        anomaly_report=anomaly,
    )
    scenario = next(
        row for row in concentration["candidates"]["exact_champion_replay"][
            "scenarios"
        ]
        if row["scenario_name"] == "remove_best_symbol"
    )

    assert scenario["excluded_symbols"]
    removed = scenario["excluded_symbols"][0]
    assert scenario["removed_contributor_remaining_contribution"][removed] == 0.0


def test_trading_leaderboard_prefers_canonical_return_when_available(tmp_path):
    classification_path = tmp_path / "leaderboard.json"
    allocation_path = tmp_path / "allocation_policy_comparison.json"
    optimizer_path = tmp_path / "allocation_optimizer_results.json"
    auxiliary_path = tmp_path / "meta_auxiliary_metrics.json"
    canonical_path = tmp_path / "canonical_continuous_equity_replay.json"
    concentration_path = tmp_path / "profit_concentration_audit.json"
    classification_path.write_text(json.dumps({"leaderboard": []}), encoding="utf-8")
    allocation_path.write_text(
        json.dumps({
            "policies": [
                {
                    "policy_name": "diagnostic_only_big_return",
                    "available": True,
                    "total_return": 999.0,
                    "max_drawdown": 0.1,
                }
            ],
            "baselines": [],
        }),
        encoding="utf-8",
    )
    optimizer_path.write_text(
        json.dumps({
            "objective_mode": "robustness_adjusted_canonical_score",
            "selected_policy": {
                "candidate_id": "optimizer",
                "selected_by_robustness_objective": True,
                "holdout_objective_metrics": {
                    "max_allowed_anomaly_dependency_ratio": 0.25,
                    "turnover_penalty": 0.0,
                    "cost_stress_penalty": 0.0,
                    "robustness_weights": {
                        "drawdown": 0.5,
                        "turnover": 0.25,
                        "concentration": 0.25,
                        "anomaly_dependency": 0.5,
                        "cost_stress": 1.0,
                    },
                },
                "frozen_holdout_metrics": {
                    "policy_name": "selected_bayesian_optimizer_diagnostic_policy",
                    "total_return": 10.0,
                    "max_drawdown": 0.8,
                },
            }
        }),
        encoding="utf-8",
    )
    auxiliary_path.write_text(json.dumps({}), encoding="utf-8")
    canonical_path.write_text(
        json.dumps({
            "candidates": {
                "selected_bayesian_optimizer_diagnostic_policy": {
                    "diagnostic_period_grid": {"total_return": 10.0},
                    "canonical_continuous_equity": {
                        "total_return": 0.10,
                        "canonical_tradable_total_return": 0.10,
                        "max_drawdown": 0.05,
                    },
                },
                "exact_champion_replay": {
                    "diagnostic_period_grid": {"total_return": 1.0},
                    "canonical_continuous_equity": {
                        "total_return": 0.20,
                        "canonical_tradable_total_return": 0.20,
                        "max_drawdown": 0.10,
                    },
                },
            }
        }),
        encoding="utf-8",
    )
    concentration_path.write_text(
        json.dumps({
            "candidates": {
                "selected_bayesian_optimizer_diagnostic_policy": {
                    "profit_concentration": {
                        "top_5_date_positive_return_share": 0.9
                    },
                    "scenarios": [
                        {
                            "scenario_name": "remove_anomaly_dates",
                            "summary": {"total_return": 0.05},
                        }
                    ],
                },
                "exact_champion_replay": {
                    "profit_concentration": {
                        "top_5_date_positive_return_share": 0.4
                    },
                    "scenarios": [
                        {
                            "scenario_name": "remove_anomaly_dates",
                            "summary": {"total_return": 0.15},
                        }
                    ],
                },
            }
        }),
        encoding="utf-8",
    )

    paths = write_trading_research_leaderboard(
        tmp_path,
        classification_path,
        allocation_path,
        optimizer_path,
        auxiliary_path,
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert [row["entity_name"] for row in payload["leaderboard"]] == [
        "exact_champion_replay",
        "selected_bayesian_optimizer_diagnostic_policy",
        "diagnostic_only_big_return",
    ]
    assert payload["leaderboard"][1]["diagnostic_period_grid_return"] == 10.0
    assert payload["leaderboard"][1]["canonical_continuous_return"] == 0.10
    assert payload["leaderboard"][1]["anomaly_dependency_ratio"] == 0.5
    assert abs(
        payload["leaderboard"][1]["robustness_adjusted_score"] - (-0.325)
    ) < 1e-12


def _champion_audit_payload() -> dict:
    return {
        "stooq_adjustment_audit": {
            "adjusted_status": "unknown_from_repo_metadata",
            "data_path": "data/processed/stooq_parquet",
            "price_column_used": "close",
        },
        "exact_champion_replay": {
            "period_rows": [
                {
                    "rebalance_date": "2024-01-01",
                    "outcome_end_date": "2024-01-10",
                    "period_return": 0.10,
                    "exposure_target": 0.9,
                    "selected_symbols": ["AXTI", "AAA"],
                    "target_weights": {"AXTI": 0.7, "AAA": 0.3},
                    "symbol_return_anomalies": [
                        {
                            "symbol": "AXTI",
                            "start_close": 1.0,
                            "end_close": 2.5,
                            "return": 1.5,
                        }
                    ],
                },
                {
                    "rebalance_date": "2024-01-05",
                    "outcome_end_date": "2024-01-15",
                    "period_return": 0.20,
                    "exposure_target": 0.9,
                    "selected_symbols": ["BBB"],
                    "target_weights": {"BBB": 1.0},
                    "symbol_return_anomalies": [],
                },
                {
                    "rebalance_date": "2024-01-15",
                    "outcome_end_date": "2024-01-25",
                    "period_return": -0.05,
                    "exposure_target": 0.9,
                    "selected_symbols": ["AXTI"],
                    "target_weights": {"AXTI": 1.0},
                    "symbol_return_anomalies": [],
                },
                {
                    "rebalance_date": "2024-02-01",
                    "outcome_end_date": "2024-02-10",
                    "period_return": 0.05,
                    "exposure_target": 0.9,
                    "selected_symbols": ["CCC"],
                    "target_weights": {"CCC": 1.0},
                    "symbol_return_anomalies": [],
                },
            ]
        },
    }


def _selected_optimizer_payload() -> dict:
    rows = [
        _selected_row("2024-01-01", 0.10, 1.0),
        _selected_row("2024-01-05", 0.20, 1.0),
        _selected_row("2024-01-15", -0.05, 0.5),
        _selected_row("2024-02-01", 0.05, 1.0),
    ]
    equity = 1.0
    for row in rows:
        equity *= 1.0 + row["net_return"]
        row["equity"] = equity
    return {"rows": rows}


def _selected_row(date: str, period_return: float, exposure: float) -> dict:
    return {
        "rebalance_date": date,
        "period_return": period_return,
        "exposure": exposure,
        "turnover": 0.0,
        "cost": 0.0,
        "net_return": period_return * exposure,
    }
