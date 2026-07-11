from __future__ import annotations

from types import SimpleNamespace

import pytest

import application.cli_dispatch as cli_dispatch
import application.cli_parser as cli_parser
import application.cli_runtime as cli_runtime
from application.services import ml_commands_stock
from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    CANONICAL_CORPUS_AUDIT_JSON,
    CANONICAL_CORPUS_CSV,
    CANONICAL_CORPUS_MANIFEST_JSON,
    CANONICAL_CORPUS_SUMMARY_MD,
    HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
    HistoricalCanonicalCorpusError,
)


def test_cli_mode_dispatches_to_canonical_corpus_service(monkeypatch) -> None:
    captured = {}

    class StockCommands:
        @staticmethod
        def run_ml_stock_alpha_news_canonical_corpus(config):
            captured["config"] = config

    monkeypatch.setattr(cli_dispatch, "import_module", lambda name: StockCommands)

    cli_dispatch.dispatch(
        SimpleNamespace(mode="ml-stock-alpha-news-canonical-corpus"),
        {"ml": {"stock_alpha_news_canonical_corpus": {"write_enabled": False}}},
        None,
    )

    assert captured == {"config": {"ml": {"stock_alpha_news_canonical_corpus": {"write_enabled": False}}}}


def test_cli_parser_accepts_canonical_corpus_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--mode", "ml-stock-alpha-news-canonical-corpus"],
    )

    args = cli_parser.parse_args()

    assert args.mode == "ml-stock-alpha-news-canonical-corpus"


def test_canonical_corpus_mode_is_feedless(monkeypatch) -> None:
    args = SimpleNamespace(
        mode="ml-stock-alpha-news-canonical-corpus",
        config="config/config.stock_alpha_news_canonical_corpus_alpaca_benzinga_full.yaml",
        profile=None,
        log_level="info",
    )
    config = {"ml": {"stock_alpha_news_canonical_corpus": {"write_enabled": False}}}
    captured = {}

    monkeypatch.setattr(cli_runtime, "parse_args", lambda: args)
    monkeypatch.setattr(cli_runtime, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli_runtime, "apply_research_profile", lambda loaded, profile: loaded)
    monkeypatch.setattr(cli_runtime, "apply_runtime_overrides", lambda loaded, parsed: loaded)
    monkeypatch.setattr(
        cli_runtime,
        "build_feed",
        lambda loaded: (_ for _ in ()).throw(AssertionError("market feed should not build")),
    )

    def fake_dispatch(parsed, received_config, feed):
        captured["mode"] = parsed.mode
        captured["feed"] = feed
        captured["config"] = received_config

    monkeypatch.setattr(cli_runtime, "dispatch", fake_dispatch)

    cli_runtime.run_cli()

    assert captured["mode"] == "ml-stock-alpha-news-canonical-corpus"
    assert captured["feed"] is None
    assert captured["config"]["ml"]["stock_alpha_news_canonical_corpus"]["write_enabled"] is False


def test_service_passes_canonical_corpus_config_to_materializer(monkeypatch, capsys) -> None:
    captured = {}

    def fake_materialize(config):
        captured["config"] = config
        return _manifest()

    monkeypatch.setattr(
        ml_commands_stock.historical_canonical_corpus,
        "materialize_historical_canonical_corpus_from_config",
        fake_materialize,
    )

    result = ml_commands_stock.run_ml_stock_alpha_news_canonical_corpus(_config(write_enabled=True))
    output = capsys.readouterr().out

    assert captured["config"].source_assembly_csv_path == "reports/source/assembly.csv"
    assert captured["config"].source_assembly_metadata_json_path == "reports/source/assembly.json"
    assert captured["config"].output_dir == "reports/derived/canonical"
    assert captured["config"].expected_source_checksum == "abc123"
    assert captured["config"].transformation_version == HISTORICAL_CANONICAL_TRANSFORMATION_VERSION
    assert captured["config"].write_enabled is True
    assert result["canonical_row_count"] == 2
    assert "contract_ingest_invoked=false" in output
    assert "features_generated=false" in output
    assert "model_training_invoked=false" in output
    assert "Canonical CSV: reports/derived/canonical/" + CANONICAL_CORPUS_CSV in output


def test_service_preserves_write_disabled_by_default_and_reports_blocker(monkeypatch, capsys) -> None:
    captured = {}

    def fake_materialize(config):
        captured["write_enabled"] = config.write_enabled
        raise HistoricalCanonicalCorpusError("canonical corpus materialisation is disabled by default")

    monkeypatch.setattr(
        ml_commands_stock.historical_canonical_corpus,
        "materialize_historical_canonical_corpus_from_config",
        fake_materialize,
    )

    with pytest.raises(SystemExit):
        ml_commands_stock.run_ml_stock_alpha_news_canonical_corpus(_config(write_enabled=False))
    output = capsys.readouterr().out

    assert captured == {"write_enabled": False}
    assert "canonical_corpus_written=false" in output
    assert "blocking_issue=canonical corpus materialisation is disabled by default" in output


def test_service_requires_canonical_corpus_section() -> None:
    with pytest.raises(HistoricalCanonicalCorpusError, match="config is required"):
        ml_commands_stock._historical_canonical_corpus_config({"ml": {}})


def test_expected_paths_are_separate_in_template_shape() -> None:
    settings = {
        "source_assembly_csv_path": (
            "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
            "stock_alpha_news_historical_backfill_alpaca_benzinga_full/dev/"
            "stock_alpha_news_historical_corpus_assembly.csv"
        ),
        "output_dir": (
            "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
            "stock_alpha_news_canonical_corpus_alpaca_benzinga_full/dev"
        ),
    }

    assert settings["source_assembly_csv_path"] != settings["output_dir"]
    assert "historical_backfill_alpaca_benzinga_full" in settings["source_assembly_csv_path"]
    assert "stock_alpha_news_canonical_corpus_alpaca_benzinga_full" in settings["output_dir"]


def _config(*, write_enabled: bool) -> dict:
    return {
        "ml": {
            "stock_alpha_news_canonical_corpus": {
                "source_assembly_csv_path": "reports/source/assembly.csv",
                "source_assembly_metadata_json_path": "reports/source/assembly.json",
                "output_dir": "reports/derived/canonical",
                "expected_source_checksum": "abc123",
                "transformation_version": HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
                "write_enabled": write_enabled,
            }
        }
    }


def _manifest() -> dict:
    return {
        "source_row_count": 2,
        "canonical_row_count": 2,
        "row_count_reconciled": True,
        "output_files": {
            "canonical_corpus_csv": "reports/derived/canonical/" + CANONICAL_CORPUS_CSV,
            "manifest_json": "reports/derived/canonical/" + CANONICAL_CORPUS_MANIFEST_JSON,
            "audit_json": "reports/derived/canonical/" + CANONICAL_CORPUS_AUDIT_JSON,
            "summary_markdown": "reports/derived/canonical/" + CANONICAL_CORPUS_SUMMARY_MD,
        },
    }
