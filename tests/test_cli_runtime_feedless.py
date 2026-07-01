from __future__ import annotations

from types import SimpleNamespace

import application.cli_runtime as cli_runtime


def test_stock_alpha_ensemble_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-ensemble")


def test_stock_alpha_ensemble_portfolio_sweep_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-ensemble-portfolio-sweep")


def test_stock_alpha_experiment_preflight_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-experiment-preflight")


def test_stock_alpha_news_features_mode_is_feedless(monkeypatch):
    _assert_feedless(monkeypatch, "ml-stock-alpha-news-features")


def _assert_feedless(monkeypatch, mode):
    args = SimpleNamespace(mode=mode, config="config/config.yaml", profile=None)
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
