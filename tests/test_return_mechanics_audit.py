from __future__ import annotations

import json
import math
from pathlib import Path

from core.research.ml.champion_baseline_audit import (
    exact_champion_replay_from_equity,
    write_champion_baseline_audit,
)
from core.research.ml.return_mechanics_audit import write_return_mechanics_audit
from core.research.dual_momentum.models import DualMomentumSelection
from core.services.portfolio_engine import EquityPoint


def test_return_mechanics_audit_writes_outputs_without_modifying_sources(tmp_path):
    config, output_dir = _write_audit_fixture(tmp_path)
    shadow_path = output_dir / "allocation_shadow_overlay.json"
    before = shadow_path.read_text(encoding="utf-8")

    paths = write_return_mechanics_audit(config)

    assert paths.csv_path.exists()
    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert shadow_path.read_text(encoding="utf-8") == before
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["production_validated"] is False


def test_return_mechanics_audit_aggregates_rebalance_date_before_compounding(tmp_path):
    config, output_dir = _write_audit_fixture(tmp_path)

    paths = write_return_mechanics_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    champion = _candidate(payload, "champion_baseline")

    assert champion["number_of_periods"] == 2
    assert math.isclose(champion["total_return"], 0.32)
    assert payload["mechanics"]["rows_aggregated_before_compounding"] is True
    assert (
        payload["mechanics"]["aggregation_method"]
        == "mean_by_rebalance_date_for_return_and_exposure"
    )


def test_return_mechanics_audit_detects_out_of_range_exposure(tmp_path):
    config, _ = _write_audit_fixture(tmp_path)

    paths = write_return_mechanics_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    binary = _candidate(payload, "binary_exposure_overlay")

    assert binary["exposure_sanity_checks"]["exposure_above_one"] is True
    assert binary["exposure_sanity_checks"]["out_of_range_exposure_dates"] == [
        "2024-01-01"
    ]
    assert "exposure_outside_0_1" in binary["red_flags"]


def test_return_mechanics_audit_does_not_use_actual_columns_as_forecasts(tmp_path):
    config, _ = _write_audit_fixture(tmp_path)

    paths = write_return_mechanics_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["leakage_check"]["actual_columns_used_as_forecasts"] == []
    assert payload["leakage_check"]["actual_columns_are_evaluation_only"] is True
    assert all(
        not row.get("forecast_inputs_use_actual_columns", False)
        for row in payload["candidates"]
    )


def test_return_mechanics_audit_capped_sensitivity_runs_on_fake_dataset(tmp_path):
    config, _ = _write_audit_fixture(tmp_path)

    paths = write_return_mechanics_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    return_only = _candidate(payload, "return_only_allocation")

    uncapped = return_only["total_return"]
    capped = return_only["capped_return_sensitivity"]["cap_-50pct_+50pct"][
        "total_return"
    ]
    assert capped < uncapped


def test_return_mechanics_audit_flags_champion_as_full_exposure_diagnostic(tmp_path):
    config, _ = _write_audit_fixture(tmp_path)

    paths = write_return_mechanics_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    audit = payload["champion_baseline_audit"]
    assert audit["champion_baseline_equals_always_full_exposure"] is True
    assert audit["intended_by_current_allocation_code"] is True
    assert audit["represents_full_frozen_champion_yaml_replay"] is False
    assert audit["champion_config_id"] == "ranked_top5_monthly_exposure90_v1"


def test_champion_audit_does_not_label_full_exposure_diagnostic_as_exact(tmp_path):
    config, _ = _write_audit_fixture(tmp_path)
    config["ml"]["stooq_parquet_dir"] = str(tmp_path / "missing_parquet")
    write_return_mechanics_audit(config)

    paths = write_champion_baseline_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    rows = {row["baseline_name"]: row for row in payload["baseline_rows"]}

    assert rows["champion_full_exposure_diagnostic"][
        "is_exact_champion_replay"
    ] is False
    assert rows["exact_champion_replay"]["semantic_type"] == "exact_champion_replay"
    assert payload["baseline_semantics"][
        "current_champion_baseline_is_exact_champion_replay"
    ] is False


def test_exact_champion_replay_uses_target_exposure_when_available():
    periods = [
        {"rebalance_date": "2024-01-01", "outcome_end_date": "2024-01-08"},
        {"rebalance_date": "2024-01-08", "outcome_end_date": "2024-01-15"},
    ]
    equity_curve = [
        EquityPoint(datetime_from("2024-01-01"), 100.0),
        EquityPoint(datetime_from("2024-01-08"), 109.0),
        EquityPoint(datetime_from("2024-01-15"), 119.9),
    ]
    selections = [
        DualMomentumSelection(
            timestamp=datetime_from("2024-01-01"),
            symbols=["AAA", "BBB"],
            scores={"AAA": 1.0, "BBB": 0.9},
            risk_on=True,
            exposure_target=0.9,
            target_weights={"AAA": 0.5, "BBB": 0.5},
        )
    ]

    replay = exact_champion_replay_from_equity(
        periods=periods,
        equity_curve=equity_curve,
        selections=selections,
        champion_config={"overrides": {"target_exposure": 0.9}},
    )

    assert replay["available"] is True
    assert replay["summary"]["target_exposure"] == 0.9
    assert replay["period_rows"][0]["exposure_target"] == 0.9


def test_exact_champion_replay_marks_cost_turnover_attribution_status():
    replay = exact_champion_replay_from_equity(
        periods=[{"rebalance_date": "2024-01-01", "outcome_end_date": "2024-01-08"}],
        equity_curve=[
            EquityPoint(datetime_from("2024-01-01"), 100.0),
            EquityPoint(datetime_from("2024-01-08"), 101.0),
        ],
        selections=[],
        champion_config={"overrides": {"target_exposure": 0.9}},
    )

    assert replay["summary"]["costs"] is None
    assert "handled_inside_dual_momentum_backtester" in replay["summary"][
        "cost_turnover_status"
    ]


def test_champion_audit_does_not_import_live_or_broker_code_paths():
    import inspect
    import core.research.ml.champion_baseline_audit as audit

    source = inspect.getsource(audit)

    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "broker" not in source


def datetime_from(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)


def _candidate(payload: dict, name: str) -> dict:
    return next(row for row in payload["candidates"] if row["candidate_name"] == name)


def _write_audit_fixture(tmp_path: Path) -> tuple[dict, Path]:
    output_dir = tmp_path / "reports" / "ml" / "benchmark" / "regime_transformer_meta_ensemble_v1"
    cache_dir = tmp_path / "cache" / "ml" / "benchmark"
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    champion_path = tmp_path / "configs" / "champions" / "ranked_top5_monthly_exposure90_v1.yaml"
    champion_path.parent.mkdir(parents=True)
    champion_path.write_text(
        "\n".join([
            "champion_id: ranked_top5_monthly_exposure90_v1",
            "frozen: true",
            "do_not_mutate: true",
            "overrides:",
            "  target_exposure: 0.90",
            "",
        ]),
        encoding="utf-8",
    )

    champion_rows = [
        {"date": "2024-01-01", "baseline_return": 0.10, "exposure": 1.0},
        {"date": "2024-01-01", "baseline_return": 0.30, "exposure": 1.0},
        {"date": "2024-01-08", "baseline_return": 0.10, "exposure": 1.0},
    ]
    shadow = {
        "mode": "allocation_shadow_overlay_v2_research_only",
        "policies": {
            "binary_exposure_overlay": _shadow_payload(
                "binary_exposure_overlay",
                [
                    {"date": "2024-01-01", "baseline_return": 0.10, "exposure": 1.2},
                    {"date": "2024-01-08", "baseline_return": 0.10, "exposure": 0.8},
                ],
                required=["predicted_probability"],
            ),
            "return_only_allocation": _shadow_payload(
                "return_only_allocation",
                [
                    {"date": "2024-01-01", "baseline_return": 1.00, "exposure": 1.0},
                    {"date": "2024-01-08", "baseline_return": 0.10, "exposure": 1.0},
                ],
                required=[
                    "predicted_forward_return_10d|predicted_forward_return_5d"
                ],
            ),
        },
        "baselines": {
            "champion_baseline": _shadow_payload(
                "champion_baseline",
                champion_rows,
                kind="diagnostic_baseline",
                required=[],
            ),
            "always_full_exposure": _shadow_payload(
                "always_full_exposure",
                champion_rows,
                kind="diagnostic_baseline",
                required=[],
            ),
        },
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    _write_json(output_dir / "allocation_shadow_overlay.json", shadow)
    comparison = {
        "mode": "allocation_policy_comparison_v2_research_only",
        "transaction_cost_bps": 0.0,
        "policies": [
            _comparison_row(
                "binary_exposure_overlay",
                required=["predicted_probability"],
            ),
            _comparison_row(
                "return_only_allocation",
                total_return=1.20,
                required=[
                    "predicted_forward_return_10d|predicted_forward_return_5d"
                ],
            ),
        ],
        "baselines": [
            _comparison_row("champion_baseline", total_return=0.32),
            _comparison_row("always_full_exposure", total_return=0.32),
        ],
    }
    _write_json(output_dir / "allocation_policy_comparison.json", comparison)
    _write_json(output_dir / "allocation_policy_grid_search.json", {})
    _write_json(output_dir / "allocation_optimizer_results.json", {})
    _write_json(output_dir / "meta_dataset_audit.json", {
        "row_count": 3,
        "source_dataset_hash": "hash",
    })
    _write_json(cache_dir / "expanded_rebalance_dataset_audit.json", {
        "row_count": 3,
        "variant_count": 1,
        "universe_paths": ["data/reference/universes/us_liquid_500.yaml"],
        "variants": [{"available_symbols": 379}],
    })
    (cache_dir / "meta_ensemble_dataset.csv").write_text(
        "\n".join([
            "feature_id,rebalance_date,split,actual_forward_return_10d,champion_return_next_period",
            "a,2024-01-01,holdout,0.10,0.10",
            "b,2024-01-01,holdout,0.30,0.30",
            "c,2024-01-08,holdout,0.10,0.10",
            "",
        ]),
        encoding="utf-8",
    )
    (cache_dir / "expanded_rebalance_dataset.csv").write_text(
        "\n".join([
            "rebalance_date,selected_symbols,champion_return_next_period",
            '2024-01-01,"A,B",0.10',
            '2024-01-08,"A,C",0.10',
            "",
        ]),
        encoding="utf-8",
    )
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "\n".join([
            "feature_id,rebalance_date,meta_predicted_forward_return_10d,actual_forward_return_10d",
            "a,2024-01-01,0.01,0.10",
            "",
        ]),
        encoding="utf-8",
    )
    config = {
        "ml": {
            "output_dir": str(output_dir),
            "meta_dataset_path": str(cache_dir / "meta_ensemble_dataset.csv"),
            "expanded_rebalance_dataset_path": str(
                cache_dir / "expanded_rebalance_dataset.csv"
            ),
            "expanded_rebalance_audit_path": str(
                cache_dir / "expanded_rebalance_dataset_audit.json"
            ),
            "allocation_transaction_cost_bps": 0.0,
        },
        "cache": {"ml_dir": str(cache_dir)},
        "reports": {"ml_dir": str(output_dir.parent)},
        "research_profile": {
            "name": "benchmark",
            "universe": "us_liquid_500",
            "max_symbols": 500,
        },
        "research": {
            "dual_momentum": {
                "champion_config_path": str(champion_path),
            }
        },
    }
    return config, output_dir


def _shadow_payload(
    name: str,
    rows: list[dict],
    *,
    kind: str = "allocation_policy",
    required: list[str],
) -> dict:
    return {
        "policy_name": name,
        "policy_kind": kind,
        "available": True,
        "required_prediction_columns": required,
        "forecast_source": "probability_only" if required else "none",
        "transaction_cost_bps": 0.0,
        "rows": rows,
    }


def _comparison_row(
    name: str,
    *,
    total_return: float = 0.0,
    required: list[str] | None = None,
) -> dict:
    return {
        "policy_name": name,
        "available": True,
        "required_prediction_columns": required or [],
        "total_return": total_return,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "turnover": 0.0,
        "estimated_transaction_costs": 0.0,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
