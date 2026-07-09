from __future__ import annotations

from types import SimpleNamespace

import application.cli_dispatch as cli_dispatch
import application.cli_runtime as cli_runtime


def test_stock_alpha_ensemble_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-ensemble")


def test_stock_alpha_ensemble_portfolio_sweep_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-ensemble-portfolio-sweep")


def test_stock_alpha_experiment_preflight_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-experiment-preflight")


def test_stock_alpha_news_features_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-news-features")


def test_dual_momentum_stock_score_comparison_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-dual-momentum-stock-score-comparison")


def test_dual_momentum_stock_score_comparison_dispatch_accepts_none_feed(monkeypatch):
    captured = {}

    class DualMomentumCommands:
        @staticmethod
        def run_stock_ml_dual_momentum_comparison(config, feed, strategy_names=None):
            captured["config"] = config
            captured["feed"] = feed
            captured["strategy_names"] = strategy_names

    monkeypatch.setattr(
        cli_dispatch,
        "import_module",
        lambda name: DualMomentumCommands,
    )

    cli_dispatch.dispatch(
        SimpleNamespace(
            mode="ml-dual-momentum-stock-score-comparison",
            strategies=["elastic_net_oos"],
        ),
        {"ml": {}},
        None,
    )

    assert captured == {
        "config": {"ml": {}},
        "feed": None,
        "strategy_names": ["elastic_net_oos"],
    }


def test_feed_required_mode_still_builds_feed(monkeypatch):
    args = SimpleNamespace(
        mode="dual-momentum",
        config="config/config.yaml",
        profile=None,
        log_level="info",
    )
    feed = object()
    captured = {}

    monkeypatch.setattr(cli_runtime, "parse_args", lambda: args)
    monkeypatch.setattr(cli_runtime, "load_config", lambda *args, **kwargs: {"ml": {}})
    monkeypatch.setattr(cli_runtime, "apply_research_profile", lambda config, profile: config)
    monkeypatch.setattr(cli_runtime, "apply_runtime_overrides", lambda config, parsed: config)
    monkeypatch.setattr(cli_runtime, "build_feed", lambda config: feed)

    def fake_dispatch(parsed, config, received_feed):
        captured["mode"] = parsed.mode
        captured["feed"] = received_feed
        captured["config_path"] = config["config_path"]

    monkeypatch.setattr(cli_runtime, "dispatch", fake_dispatch)

    cli_runtime.run_cli()

    assert captured == {
        "mode": "dual-momentum",
        "feed": feed,
        "config_path": "config/config.yaml",
    }


def _assert_feedless(monkeypatch, mode):
    args = SimpleNamespace(
        mode=mode,
        config="config/config.yaml",
        profile=None,
        log_level="info",
    )
    captured = {}

    monkeypatch.setattr(cli_runtime, "parse_args", lambda: args)
    monkeypatch.setattr(cli_runtime, "load_config", lambda *args, **kwargs: {"ml": {}})
    monkeypatch.setattr(cli_runtime, "apply_research_profile", lambda config, profile: config)
    monkeypatch.setattr(cli_runtime, "apply_runtime_overrides", lambda config, parsed: config)
    monkeypatch.setattr(
        cli_runtime,
        "build_feed",
        lambda config: (_ for _ in ()).throw(AssertionError("feed should not build")),
    )

    def fake_dispatch(parsed, config, feed):
        captured["mode"] = parsed.mode
        captured["feed"] = feed
        captured["config_path"] = config["config_path"]

    monkeypatch.setattr(cli_runtime, "dispatch", fake_dispatch)

    cli_runtime.run_cli()

    assert captured == {"mode": mode, "feed": None, "config_path": "config/config.yaml"}
