from datetime import datetime
from types import SimpleNamespace

import pytest

from core.rebalance.model_triggered import evaluate_model_triggered_rebalance, validate_rebalance_policy


def _config(**gate_overrides):
    gates = {"minimum_net_improvement": 0.03, "maximum_turnover": 2.0, "maximum_replacements": 2, "minimum_confidence": 0.5}
    gates.update(gate_overrides)
    return {"ml": {"rebalance_policy": "model_triggered", "model_triggered_rebalance": {"paper_only": True, "allow_paper_submit": False, "evaluate_every_run": True, "candidate": {"source": "dual_momentum"}, "gates": gates, "costs": {"transaction_cost_bps": 10, "slippage_bps": 5}, "safety": {"allow_live_trading": False, "allow_champion_mutation": False}}}}


def _decision(current=0.2):
    return SimpleNamespace(timestamp=datetime(2026, 7, 1), exposure_target=1.0, target_weights={"AAA": .5, "BBB": .5}, current_positions={}, orders=[SimpleNamespace(symbol="CCC", current_weight=current)], model_context={"scores": {"AAA": 1.0, "BBB": .8, "CCC": .1}})


def test_fixed_schedule_is_backward_compatible():
    assert validate_rebalance_policy({}) == "fixed_schedule"
    assert evaluate_model_triggered_rebalance({}, _decision(), submit_requested=False) is None


def test_unknown_policy_fails_closed():
    with pytest.raises(ValueError, match="Unknown"):
        validate_rebalance_policy({"ml": {"rebalance_policy": "surprise"}})


def test_live_mode_is_refused():
    config = _config(); config["trading"] = {"mode": "live"}
    with pytest.raises(RuntimeError, match="unavailable in live"):
        validate_rebalance_policy(config)


def test_below_threshold_and_high_turnover_are_no_trade():
    low = evaluate_model_triggered_rebalance(_config(minimum_net_improvement=2), _decision(), submit_requested=False)
    high = evaluate_model_triggered_rebalance(_config(maximum_turnover=.1), _decision(), submit_requested=False)
    assert low.decision == "NO_TRADE"
    assert high.decision == "NO_TRADE"


def test_passing_gates_recommends_but_does_not_allow_submit_by_default():
    result = evaluate_model_triggered_rebalance(_config(), _decision(), submit_requested=True)
    assert result.decision == "PAPER_REBALANCE_RECOMMENDED"
    assert result.paper_submit_allowed is False
    assert result.evaluate_every_run is True
    assert result.fixed_schedule_gate_bypassed_for_candidate_evaluation is True


def test_submission_status_requires_allowed_evaluation():
    config = _config()
    config["ml"]["model_triggered_rebalance"]["allow_paper_submit"] = True
    result = evaluate_model_triggered_rebalance(config, _decision(), submit_requested=True)
    assert result.paper_submit_allowed is True
    submitted = result.with_submission_result(True)
    assert submitted.decision == "PAPER_REBALANCE_SUBMITTED"
    assert submitted.orders_submitted is True


def test_recommended_dry_run_does_not_claim_submission_permission():
    config = _config()
    config["ml"]["model_triggered_rebalance"]["allow_paper_submit"] = True
    result = evaluate_model_triggered_rebalance(
        config, _decision(), submit_requested=False,
    )
    assert result.decision == "PAPER_REBALANCE_RECOMMENDED"
    assert result.paper_submit_requested is False
    assert result.paper_submit_allowed is False


def test_full_current_portfolio_weights_are_used_when_no_order_is_generated():
    decision = _decision()
    decision.model_context["current_weights"] = {"AAA": 0.5, "CCC": 0.5}
    decision.orders = []

    result = evaluate_model_triggered_rebalance(
        _config(maximum_turnover=0.5),
        decision,
        submit_requested=False,
    )

    assert result.current_holdings == {"AAA": 0.5, "CCC": 0.5}
    assert result.turnover == pytest.approx(0.5)
    assert result.estimated_transaction_cost == pytest.approx(0.001)
    assert result.estimated_slippage == pytest.approx(0.0005)
    assert result.decision == "PAPER_REBALANCE_RECOMMENDED"


def test_typical_single_position_replacement_passes_trial_turnover_gate():
    incumbent = {
        "AAA": 0.18,
        "BBB": 0.18,
        "CCC": 0.18,
        "DDD": 0.18,
        "EEE": 0.18,
    }
    candidate = {
        "AAA": 0.18,
        "BBB": 0.18,
        "DDD": 0.18,
        "EEE": 0.18,
        "FFF": 0.18,
    }
    decision = SimpleNamespace(
        timestamp=datetime(2026, 7, 1),
        exposure_target=1.0,
        target_weights=candidate,
        current_positions={},
        orders=[],
        model_context={
            "current_weights": incumbent,
            "scores": {
                "AAA": 0.5,
                "BBB": 0.5,
                "CCC": 0.0,
                "DDD": 0.5,
                "EEE": 0.5,
                "FFF": 1.0,
            },
        },
    )

    result = evaluate_model_triggered_rebalance(
        _config(maximum_turnover=0.25),
        decision,
        submit_requested=False,
    )

    assert result.replacement_count == 1
    assert result.turnover == pytest.approx(0.18)
    assert result.risk_gate_passed is True
    assert result.decision == "PAPER_REBALANCE_RECOMMENDED"
