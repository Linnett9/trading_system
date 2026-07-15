from types import SimpleNamespace

import pytest

from core.research.ml.selector_evaluation import (
    COST_SCENARIOS_BPS,
    build_date_panel,
    daily_top_k,
    matched_comparison,
    orchestrate_date_panel,
    percentile_weighted_top_k,
    portfolio_metrics,
    rank_hysteresis,
    regime_summaries,
    staggered_cohorts,
    traded_notional_cost,
    verify_evaluation_inputs,
)


def _rows(date="2024-01-01", scores=(4, 3, 2, 1)):
    return [
        {"decision_date": date, "row_id": f"{date}:{index}", "symbol": chr(65 + index), "score": score,
         "selector_artifact_link_hash": f"link:{date}"}
        for index, score in enumerate(scores)
    ]


def test_date_panel_resolution_is_deterministic_and_records_overlap():
    first = build_date_panel(["2024-01-01", "2024-01-04"], ["2024-01-02", "2024-01-05"], panel_id="x")
    second = build_date_panel(["2024-01-01", "2024-01-04"], ["2024-01-02", "2024-01-05"], panel_id="x")
    assert first == second
    assert first["dates"] == ["2024-01-02", "2024-01-05"]
    assert first["overlapping_pair_count"] == 1
    assert first["inferential_independence"] is False


def test_daily_and_percentile_policies_are_tie_deterministic():
    rows = _rows(scores=(2, 2, 1))
    assert daily_top_k(rows, score_field="score", k=2) == {"A": .5, "B": .5}
    assert percentile_weighted_top_k(rows, score_field="score", k=2) == {"A": 2 / 3, "B": 1 / 3}


def test_staggered_cohorts_preserve_ancestry_and_expire_at_horizon():
    rows = []
    for day in range(1, 5):
        rows.extend(_rows(f"2024-01-0{day}"))
    result = staggered_cohorts(rows, score_field="score", k=2, horizon=3)
    assert result[0]["active_cohort_count"] == 1
    assert sum(result[0]["aggregate_holdings"].values()) == pytest.approx(1 / 3)
    assert result[3]["expired_cohorts"] == ["cohort:2024-01-01"]
    assert result[3]["active_cohort_count"] == 3
    assert result[3]["active_cohorts"][-1]["selector_artifact_link_hash"] == "link:2024-01-04"


def test_hysteresis_retains_names_inside_exit_band_and_fills_vacancy():
    first = _rows("2024-01-01")
    second = _rows("2024-01-02", scores=(4, 2, 3, 1))
    result = rank_hysteresis({"2024-01-01": first, "2024-01-02": second}, score_field="score", enter_rank=2, exit_rank=3)
    assert result[0]["held"] == ["A", "B"]
    assert result[1]["held"] == ["A", "B"]
    assert result[1]["avoided_trades"] == 1
    assert result[1]["opportunity_cost_symbols"] == ["C"]


@pytest.mark.parametrize("bps", COST_SCENARIOS_BPS)
def test_cost_is_applied_to_traded_notional_and_capacity_is_unverified(bps):
    result = traded_notional_cost({"A": 1}, {"B": 1}, cost_bps=bps, portfolio_value=100)
    assert result["gross_turnover"] == 2
    assert result["traded_notional"] == 200
    assert result["cost"] == pytest.approx(200 * bps / 10_000)
    assert {row["status"] for row in result["capacity"].values()} == {"UNVERIFIED"}


def test_metrics_regimes_and_matched_comparison_are_explicit():
    periods = [
        {"net_return": .1, "gross_return": .11, "turnover": .2, "cost": .01, "holdings": {"A": 1}, "benchmark_return": .02},
        {"net_return": -.05, "gross_return": -.04, "turnover": .4, "cost": .01, "holdings": {"B": 1}, "benchmark_return": 0},
    ]
    metrics = portfolio_metrics(periods)
    assert metrics["maximum_drawdown"] == pytest.approx(-.05)
    assert metrics["total_cost"] == .02
    regimes = regime_summaries([{"net_return": .1, "market_trend_state": "UP"}])
    assert regimes["descriptive_only"] is True and regimes["fitted_regime_model"] is False
    left = [{"decision_date": "d", "row_id": "a", "net_return": .1}, {"decision_date": "d", "row_id": "b", "net_return": .2}]
    right = [{"decision_date": "d", "row_id": "a", "net_return": .05}]
    comparison = matched_comparison(left, right)
    assert comparison["shared_asset_row_count"] == 1
    assert comparison["population_mismatch"] is True
    assert comparison["ordinary_independent_significance_reported"] is False


def test_orchestrator_stops_new_batches_at_threshold_and_resumes(tmp_path):
    calls = []
    def runner(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="out", stderr="", returncode=1 if command[1] == "bad" else 0)
    def command(date, model, _owner):
        return [date, model]

    result = orchestrate_date_panel(dates=["d1", "d2", "d3"], models=["bad"], output_root=tmp_path, command_builder=command, failure_threshold=1, runner=runner)
    assert result["status"] == "failed_threshold"
    assert calls == [["d1", "bad"]]

    calls.clear()
    good_root = tmp_path / "good"
    first = orchestrate_date_panel(dates=["d1"], models=["good"], output_root=good_root, command_builder=command, runner=runner)
    second = orchestrate_date_panel(dates=["d1"], models=["good"], output_root=good_root, command_builder=command, runner=runner)
    assert first["status"] == "complete" and second["results"][0]["status"] == "skipped_complete"
    assert calls == [["d1", "good"]]


def test_orchestrator_rejects_duplicate_ownership(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        orchestrate_date_panel(dates=["d", "d"], models=["m"], output_root=tmp_path, command_builder=lambda *_: [])


def test_promotion_rejects_unverified_selector_ancestry(monkeypatch, tmp_path):
    monkeypatch.setattr("core.research.ml.selector_evaluation.read_artifact_link", lambda _path: {"verification_status": "INSUFFICIENT_EVIDENCE"})
    with pytest.raises(ValueError, match="VERIFIED_STRICT_OOS"):
        verify_evaluation_inputs([tmp_path / "manifest.json"], promotion_mode=True)
    assert verify_evaluation_inputs([tmp_path / "manifest.json"], promotion_mode=False)[0]["verification_status"] == "INSUFFICIENT_EVIDENCE"
