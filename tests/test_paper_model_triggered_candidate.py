from datetime import datetime
from types import SimpleNamespace

from application.services.paper_service import create_paper_decision


class _Feed:
    pass


def test_model_triggered_paper_candidate_uses_daily_evaluation(monkeypatch):
    captured = {}
    candle = SimpleNamespace(timestamp=datetime(2026, 7, 1), close=100.0)
    monkeypatch.setattr(
        "application.services.paper_service.active_dual_momentum_config",
        lambda config: {"symbols": ["AAA"], "regime_symbol": "AAA", "rebalance_frequency": "monthly"},
    )
    monkeypatch.setattr(
        "application.services.paper_service.load_candles",
        lambda symbol, config, feed: [candle],
    )
    monkeypatch.setattr(
        "application.services.paper_service.latest_prices",
        lambda candles: {"AAA": 100.0},
    )
    monkeypatch.setattr(
        "application.services.paper_service.latest_data_freshness",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "application.services.paper_service.data_quality_report",
        lambda *args, **kwargs: {},
    )
    decision = SimpleNamespace(selected_symbols=[], model_context={})
    engine = SimpleNamespace(create_decision=lambda *args, **kwargs: decision)
    monkeypatch.setattr("application.services.paper_service.build_paper_engine", lambda config: engine)
    monkeypatch.setattr("application.services.paper_service._sync_broker_state_before_decision", lambda *args: None)

    def build(config, dual_config):
        captured.update(dual_config)
        return SimpleNamespace(run=lambda candles: SimpleNamespace())

    config = {
        "backtest": {"symbols": ["AAA"]},
        "risk": {"paper": {}},
        "ml": {
            "rebalance_policy": "model_triggered",
            "model_triggered_rebalance": {"evaluate_every_run": True},
        },
    }
    result = create_paper_decision(config, _Feed(), build)
    assert captured["rebalance_frequency"] == "daily"
    assert result.model_context["candidate_source"] == "dual_momentum"
    assert result.model_context["fixed_schedule_gate_bypassed_for_candidate_evaluation"] is True


def test_fixed_schedule_preserves_configured_frequency(monkeypatch):
    from core.research.portfolio_utils import rebalance_key

    first = datetime(2026, 7, 1)
    second = datetime(2026, 7, 2)
    assert rebalance_key(first, "monthly") == rebalance_key(second, "monthly")
    assert rebalance_key(first, "daily") != rebalance_key(second, "daily")
