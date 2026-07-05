from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import core.research.ml.stock_level.news_risk_overlay_research as research
from core.research.ml.stock_level.news_risk_overlay_research import (
    _accounting_audit,
    _assert_score_direction_contract,
    _bar_sets_equal,
    _chunks,
    _cost_scenario_comparison,
    _event_category_analysis,
    _load_daily_price_bars,
    _parallel_config,
    _portfolio_comparison,
    _run_open_trade_replay,
    _score_direction_audit,
    _variant_multiplier,
    format_news_risk_overlay_summary,
    inspect_stock_alpha_news_risk_overlay_results,
    write_stock_alpha_news_risk_overlay_research,
)
from core.research.ml.stock_level.news_risk_overlay import NewsRiskOverlayConfig


def test_news_risk_overlay_research_writes_end_to_end_outputs(tmp_path: Path) -> None:
    price_path = tmp_path / "price_candidates.csv"
    news_path = tmp_path / "news_features.csv"
    _write_csv(
        price_path,
        [
            _price_row("2024-01-01", "AAPL", -0.06, 0.90, -0.07),
            _price_row("2024-01-02", "MSFT", 0.03, 0.80, -0.01),
            _price_row("2024-01-03", "AAPL", -0.08, 0.70, -0.09),
            _price_row("2024-01-04", "MSFT", 0.04, 0.60, -0.01),
            _price_row("2024-01-05", "AAPL", -0.02, 0.50, -0.02),
            _price_row("2024-01-06", "MSFT", 0.05, 0.40, -0.01),
            _price_row("2024-01-07", "AAPL", -0.09, 0.30, -0.10),
            _price_row("2024-01-08", "MSFT", 0.06, 0.20, -0.01),
        ],
    )
    _write_csv(
        news_path,
        [
            {"symbol": "AAPL", "available_at_timestamp": "2023-12-31T12:00:00+00:00", "sentiment": "-0.8", "event_count": "2"},
            {"symbol": "MSFT", "available_at_timestamp": "2023-12-31T12:00:00+00:00", "sentiment": "0.2", "event_count": "1"},
        ],
    )

    result = write_stock_alpha_news_risk_overlay_research(
        {
            "ml": {
                "stock_alpha_news_risk_overlay_price_candidates_path": str(price_path),
                "stock_alpha_news_risk_overlay_news_features_path": str(news_path),
                "stock_alpha_news_risk_overlay_output_dir": str(tmp_path / "research-results"),
                "stock_alpha_news_risk_overlay_walk_forward_folds": 2,
                "stock_alpha_news_risk_overlay_epochs": 25,
                "stock_alpha_news_risk_overlay_min_coverage_ratio": 1.0,
                "stock_alpha_news_risk_overlay_portfolio_top_n": 1,
            }
        }
    )

    assert result.dataset_csv_path.exists()
    assert result.metrics_json_path.exists()
    assert result.portfolio_json_path.exists()
    assert result.accounting_json_path.exists()
    assert result.accounting_audit_json_path.exists()
    assert result.equity_curve_csv_path.exists()
    assert result.drawdown_curve_csv_path.exists()
    assert result.shadow_csv_path.exists()
    coverage = json.loads(result.coverage_json_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_json_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))
    assert coverage["row_coverage_ratio"] == 1.0
    assert metrics["price_plus_news"]["oos_rows"] > 0
    assert manifest["transformer_trained"] is False
    assert manifest["paper_orders_enabled"] is False
    portfolio = json.loads(result.portfolio_json_path.read_text(encoding="utf-8"))
    assert "total_return_decimal" in portfolio["price_only"]
    assert "wealth_multiple" in portfolio["price_only"]
    assert portfolio["transaction_cost_bps"] == 0.0
    assert portfolio["slippage_bps"] == 0.0
    audit = json.loads(result.accounting_audit_json_path.read_text(encoding="utf-8"))
    assert audit["return_arithmetic_or_compounded"] == "compounded"
    assert audit["is_full_marked_to_market_portfolio_backtest"] is False
    assert audit["replacement_candidates_selected"] is False


def test_news_risk_overlay_research_fails_without_label_sources(tmp_path: Path) -> None:
    price_path = tmp_path / "price_candidates.csv"
    news_path = tmp_path / "news_features.csv"
    _write_csv(
        price_path,
        [
            {
                "rebalance_date": "2024-01-01",
                "symbol": "AAPL",
                "stock_level_predicted_forward_return_10d_elastic_net": "0.1",
            }
        ],
    )
    _write_csv(
        news_path,
        [
            {
                "symbol": "AAPL",
                "available_at_timestamp": "2023-12-31T12:00:00+00:00",
                "sentiment": "-0.8",
            }
        ],
    )

    with pytest.raises(ValueError, match="adverse-outcome label source"):
        write_stock_alpha_news_risk_overlay_research(
            {
                "ml": {
                    "stock_alpha_news_risk_overlay_price_candidates_path": str(price_path),
                    "stock_alpha_news_risk_overlay_news_features_path": str(news_path),
                    "stock_alpha_news_risk_overlay_output_dir": str(tmp_path / "research-results"),
                }
            }
        )


def test_portfolio_accounting_compounds_equity_and_distinguishes_trade_returns() -> None:
    rows = [
        {
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "1.0",
            "actual_forward_return_10d": "0.10",
            "actual_max_adverse_excursion": "-0.02",
            "news_position_multiplier": "1.0",
        },
        {
            "decision_timestamp": "2024-01-02T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "1.0",
            "actual_forward_return_10d": "-0.10",
            "actual_future_drawdown": "0.12",
            "news_position_multiplier": "0.0",
        },
    ]

    report = _portfolio_comparison(
        rows,
        price_score_column="score",
        return_column="actual_forward_return_10d",
        top_n=1,
        starting_equity=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert report["price_only"]["starting_equity"] == 100.0
    assert report["price_only"]["ending_equity"] == pytest.approx(99.0)
    assert report["price_only"]["wealth_multiple"] == pytest.approx(0.99)
    assert report["price_only"]["total_return_decimal"] == pytest.approx(-0.01)
    assert report["price_only"]["total_return_percent"] == pytest.approx(-1.0)
    assert report["price_plus_news"]["ending_equity"] == pytest.approx(110.0)
    assert report["price_plus_news"]["total_return_decimal"] == pytest.approx(0.10)
    assert report["news_overlay_lowered_drawdown"] is True
    assert report["price_only"]["maximum_drawdown"] == pytest.approx(-0.10)
    assert report["price_only"]["maximum_adverse_excursion"] == pytest.approx(-0.12)


def test_accounting_audit_explains_wealth_multiple_and_approximation() -> None:
    report = _portfolio_comparison(
        [
            {
                "decision_timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "score": "1.0",
                "actual_forward_return_10d": "0.20",
                "news_position_multiplier": "1.0",
            }
        ],
        price_score_column="score",
        return_column="actual_forward_return_10d",
        top_n=1,
        starting_equity=1.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    audit = _accounting_audit(report)

    assert audit["price_only"]["ending_equity"] == pytest.approx(1.2)
    assert audit["price_only"]["wealth_multiple"] == pytest.approx(1.2)
    assert audit["price_only"]["total_return_percent"] == pytest.approx(20.0)
    assert "wealth multiple" in audit["plain_english_answer"]["answer"]
    assert audit["overlapping_trades_represented"] is False
    assert audit["unused_cash_represented"] == "partially"


def test_news_risk_overlay_research_fails_before_writing_when_disk_space_is_insufficient(
    tmp_path: Path,
) -> None:
    price_path = tmp_path / "price_candidates.csv"
    news_path = tmp_path / "news_features.csv"
    output_dir = tmp_path / "research-results"
    _write_csv(price_path, [_price_row("2024-01-01", "AAPL", -0.06, 0.90, -0.07)])
    _write_csv(
        news_path,
        [
            {
                "symbol": "AAPL",
                "available_at_timestamp": "2023-12-31T12:00:00+00:00",
                "sentiment": "-0.8",
            }
        ],
    )

    with pytest.raises(ValueError, match="insufficient disk space"):
        write_stock_alpha_news_risk_overlay_research(
            {
                "ml": {
                    "stock_alpha_news_risk_overlay_price_candidates_path": str(price_path),
                    "stock_alpha_news_risk_overlay_news_features_path": str(news_path),
                    "stock_alpha_news_risk_overlay_output_dir": str(output_dir),
                    "stock_alpha_news_risk_overlay_min_free_bytes": 10**18,
                }
            }
        )
    assert not output_dir.exists()


def test_open_trade_replay_preserves_cash_and_marks_open_positions() -> None:
    rows = [
        _candidate("2024-01-01", "AAA", 1.0, "ALLOW"),
        _candidate("2024-01-01", "BBB", 0.9, "ALLOW"),
    ]
    bars = {
        "AAA": _bars("AAA", [("2024-01-02", 10, 10, 10, 11), ("2024-01-03", 11, 11, 11, 11)]),
        "BBB": _bars("BBB", [("2024-01-02", 20, 20, 20, 20), ("2024-01-03", 20, 20, 20, 20)]),
    }

    result = _run_open_trade_replay(
        rows,
        bars_by_symbol=bars,
        price_score_column="score",
        variant="price_only",
        variant_settings={"use_news": False},
        replay_config=_replay_config(max_positions=1, max_holding_bars=2, max_position_weight=0.50),
    )

    assert len(result["ledger"]) == 1
    first_mark = next(row for row in result["daily_equity"] if row["date"] == "2024-01-02")
    assert first_mark["cash"] == pytest.approx(0.5)
    assert first_mark["total_equity"] == pytest.approx(1.05)
    assert first_mark["concurrent_positions"] == 1


def test_open_trade_replay_uses_conservative_stop_before_target() -> None:
    rows = [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")]
    bars = {
        "AAA": _bars("AAA", [("2024-01-02", 10, 12, 8, 11)]),
    }

    result = _run_open_trade_replay(
        rows,
        bars_by_symbol=bars,
        price_score_column="score",
        variant="price_only",
        variant_settings={"use_news": False},
        replay_config=_replay_config(max_holding_bars=5, stop_loss_pct=0.10, profit_target_pct=0.10),
    )

    assert result["ledger"][0]["exit_reason"] == "stop_hit_conservative_before_target"
    assert result["ledger"][0]["exit_price"] == pytest.approx(9.0)


def test_replacement_variant_uses_next_point_in_time_candidate() -> None:
    rows = [
        _candidate("2024-01-01", "AAA", 1.0, "BLOCK", forward_return=-0.10),
        _candidate("2024-01-01", "BBB", 0.9, "ALLOW", forward_return=0.05),
    ]
    bars = {
        "AAA": _bars("AAA", [("2024-01-02", 10, 10, 10, 10)]),
        "BBB": _bars("BBB", [("2024-01-02", 20, 20, 20, 21)]),
    }

    result = _run_open_trade_replay(
        rows,
        bars_by_symbol=bars,
        price_score_column="score",
        variant="news_replacement",
        variant_settings={"use_news": True, "replace_blocked": True, "strict_gate": True},
        replay_config=_replay_config(max_positions=1, max_holding_bars=1),
    )

    assert len(result["ledger"]) == 1
    assert result["ledger"][0]["symbol"] == "BBB"
    blocked = [row for row in result["action_events"] if row["blocked"]]
    assert blocked[0]["symbol"] == "AAA"


def test_score_direction_audit_documents_probability_contract() -> None:
    config = NewsRiskOverlayConfig(adverse_return_threshold=-0.05, reduce_threshold=0.5, block_threshold=0.7)
    rows = [
        {
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAA",
            "news_risk_label": 1,
            "price_plus_news_risk_probability": 0.8,
            "actual_forward_return_10d": -0.06,
        }
    ]

    audit = _score_direction_audit(rows=rows, config=config, target_column="news_risk_label")
    _assert_score_direction_contract(audit, rows)

    assert audit["higher_model_probability_means"] == "higher probability of label 1, therefore higher intended risk"
    assert audit["probabilities_inverted_anywhere"] is False


def test_score_direction_audit_rejects_positive_downside_threshold() -> None:
    config = NewsRiskOverlayConfig(adverse_return_threshold=0.05, reduce_threshold=0.5, block_threshold=0.7)
    audit = _score_direction_audit(rows=[], config=config, target_column="news_risk_label")

    with pytest.raises(ValueError, match="adverse threshold"):
        _assert_score_direction_contract(audit, [])


def test_inverted_gate_changes_actions_without_changing_probability() -> None:
    replay_config = _replay_config(reduce_multiplier=0.25)

    multiplier, blocked = _variant_multiplier(
        "BLOCK",
        {"use_news": True, "inverted": True, "strict_gate": True},
        replay_config,
    )

    assert multiplier == 1.0
    assert blocked is False


def test_excluded_event_types_cannot_be_contrarian_allowed() -> None:
    report = _event_category_analysis(
        [
            {
                "decision_timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "AAA",
                "event_category": "fraud",
                "actual_forward_return_10d": "0.20",
                "actual_max_adverse_excursion": "-0.02",
            }
        ],
        [],
    )

    assert report["fraud_allegation"]["policy"] == "EXCLUDED"
    assert report["fraud_allegation"]["contrarian_suitability"] == "excluded_by_policy"


def test_cost_scenarios_reduce_net_returns() -> None:
    rows = [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")]
    bars = {"AAA": _bars("AAA", [("2024-01-02", 10, 10, 10, 11)])}

    report = _cost_scenario_comparison(
        rows,
        bars_by_symbol=bars,
        price_score_column="score",
        base_replay_config=_replay_config(max_holding_bars=1),
    )

    zero = report["scenarios"]["0_bps_round_trip"]["variants"]["price_only"]["ending_equity"]
    costly = report["scenarios"]["20_bps_round_trip"]["variants"]["price_only"]["ending_equity"]
    assert report["zero_costs_recorded_as"] == 0.0
    assert costly < zero


def test_parallel_one_worker_mode_matches_sorted_bar_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_loader(symbol: str, processed_root: Path) -> dict[str, object]:
        del processed_root
        return {
            "symbol": symbol,
            "status": "OK",
            "rows": _bars(symbol, [("2024-01-02", 10, 11, 9, 10), ("2024-01-01", 9, 10, 8, 9)]),
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(research, "_load_daily_price_bar_file", fake_loader)
    sequential, _ = _load_daily_price_bars(
        ["BBB", "AAA"],
        tmp_path,
        parallel_config=_parallel_config({"news_risk_parallel_enabled": False}),
        parallel_report={},
    )
    single_worker, _ = _load_daily_price_bars(
        ["BBB", "AAA"],
        tmp_path,
        parallel_config=_parallel_config({"news_risk_parallel_enabled": True, "news_risk_max_workers": 1}),
        parallel_report={},
    )

    assert _bar_sets_equal(sequential, single_worker)
    assert list(single_worker) == ["AAA", "BBB"]
    assert [row["date"] for row in single_worker["AAA"]] == ["2024-01-01", "2024-01-02"]


def test_parallel_multi_worker_bar_loading_keeps_stable_sorted_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_loader(symbol: str, processed_root: Path) -> dict[str, object]:
        del processed_root
        return {
            "symbol": symbol,
            "status": "OK",
            "rows": _bars(symbol, [("2024-01-02", 10, 11, 9, 10), ("2024-01-01", 9, 10, 8, 9)]),
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(research, "_load_daily_price_bar_file", fake_loader)
    bars, report = _load_daily_price_bars(
        ["CCC", "AAA", "BBB"],
        tmp_path,
        parallel_config=_parallel_config(
            {
                "news_risk_parallel_enabled": True,
                "news_risk_max_workers": 2,
                "news_risk_parallel_min_items": 1,
                "news_risk_parallel_chunk_size": 2,
            }
        ),
        parallel_report={},
    )

    assert list(bars) == ["AAA", "BBB", "CCC"]
    assert report["loaded_symbol_count"] == 3


def test_parallel_worker_failure_names_symbol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_loader(symbol: str, processed_root: Path) -> dict[str, object]:
        del processed_root
        return {
            "symbol": symbol,
            "status": "FAILED",
            "rows": [],
            "error": "boom",
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(research, "_load_daily_price_bar_file", fake_loader)

    with pytest.raises(ValueError, match="AAA"):
        _load_daily_price_bars(
            ["AAA"],
            tmp_path,
            parallel_config=_parallel_config({"news_risk_parallel_enabled": False}),
            parallel_report={},
        )


def test_parallel_chunking_respects_configured_limit() -> None:
    assert list(_chunks(["A", "B", "C", "D", "E"], 2)) == [["A", "B"], ["C", "D"], ["E"]]


def test_parallel_cost_scenarios_match_sequential_risk_metrics() -> None:
    rows = [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")]
    bars = {"AAA": _bars("AAA", [("2024-01-02", 10, 10, 10, 11)])}
    sequential = _cost_scenario_comparison(
        rows,
        bars_by_symbol=bars,
        price_score_column="score",
        base_replay_config=_replay_config(max_holding_bars=1),
        parallel_config=_parallel_config({"news_risk_parallel_enabled": False}),
        parallel_report={},
    )
    parallel = _cost_scenario_comparison(
        rows,
        bars_by_symbol=bars,
        price_score_column="score",
        base_replay_config=_replay_config(max_holding_bars=1),
        parallel_config=_parallel_config(
            {
                "news_risk_parallel_enabled": True,
                "news_risk_max_workers": 2,
                "news_risk_parallel_min_items": 1,
            }
        ),
        parallel_report={},
    )

    assert parallel["scenarios"] == sequential["scenarios"]


def test_news_risk_executive_summary_is_concise_and_warns(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    _write_json(
        output / "risk_metrics.json",
        {
            "price_only": {
                "ending_equity": 1.2,
                "total_return_percent": 20.0,
                "CAGR": 0.1,
                "maximum_drawdown": -0.2,
                "Sharpe_ratio": 1.0,
                "Calmar_ratio": 0.5,
                "CVaR_5pct": -0.03,
                "number_of_trades": 3,
                "total_costs": 0.0,
            },
            "news_contrarian_rerank": {
                "ending_equity": 1.3,
                "total_return_percent": 30.0,
                "CAGR": 0.12,
                "maximum_drawdown": -0.1,
                "Sharpe_ratio": 1.2,
                "Calmar_ratio": 0.8,
                "CVaR_5pct": -0.02,
                "number_of_trades": 3,
                "total_costs": 0.0,
            },
        },
    )
    _write_json(output / "coverage_report.json", {"row_coverage_ratio": 0.5})
    _write_json(
        output / "news_score_direction_report.json",
        {"answers": {"relationship_supports_inversion": True}},
    )
    _write_json(
        output / "cost_scenario_comparison.json",
        {
            "scenarios": {
                "0_bps_round_trip": {
                    "round_trip_bps": 0.0,
                    "variants": {
                        "price_only": {"ending_equity": 1.2, "maximum_drawdown": -0.2},
                        "news_contrarian_rerank": {"ending_equity": 1.3, "maximum_drawdown": -0.1},
                    },
                }
            }
        },
    )
    _write_json(
        output / "event_category_analysis.json",
        {"general_negative_sentiment_or_uncategorized": {"count": 10}},
    )
    _write_json(output / "extreme_event_memory_report.json", {"point_in_time_archive_rows": 1})
    _write_json(output / "portfolio_comparison.json", {"paper_orders_enabled": False, "live_orders_enabled": False})
    _write_json(output / "replay_data_audit.json", {"adjusted_status": "not explicit"})
    _write_csv(
        output / "news_score_deciles.csv",
        [
            {"decile": "1", "candidate_count": "10", "average_forward_return": "0.1", "executed_trade_count": "3"},
            {"decile": "2", "candidate_count": "10", "average_forward_return": "0.1", "executed_trade_count": "3"},
            {"decile": "3", "candidate_count": "10", "average_forward_return": "0.1", "executed_trade_count": "3"},
        ],
    )

    inspection = inspect_stock_alpha_news_risk_overlay_results(
        {"ml": {"stock_alpha_news_risk_overlay_output_dir": str(output)}}
    )
    text = format_news_risk_overlay_summary(
        inspection.summary,
        inspection.artifact_status,
        mode="summary",
    )

    assert len(text.splitlines()) <= 40
    assert "Strategy comparison" in text
    assert "WARNINGS" in text
    assert "mostly uncategorized events" in text


def test_news_risk_summary_artifact_list_mode(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    inspection = inspect_stock_alpha_news_risk_overlay_results(
        {"ml": {"stock_alpha_news_risk_overlay_output_dir": str(output)}}
    )

    text = format_news_risk_overlay_summary(
        inspection.summary,
        inspection.artifact_status,
        mode="artifact-list",
    )

    assert "STOCK-ALPHA NEWS RISK OVERLAY ARTIFACTS" in text
    assert "risk_metrics" in text


def _price_row(
    date: str,
    symbol: str,
    forward_return: float,
    score: float,
    adverse: float,
) -> dict[str, str]:
    return {
        "rebalance_date": date,
        "symbol": symbol,
        "actual_forward_return_10d": str(forward_return),
        "actual_max_adverse_excursion": str(adverse),
        "stock_level_predicted_forward_return_10d_elastic_net": str(score),
        "predicted_momentum_120d": str(score / 2.0),
    }


def _candidate(
    date: str,
    symbol: str,
    score: float,
    action: str,
    *,
    forward_return: float = 0.0,
) -> dict[str, str]:
    return {
        "decision_timestamp": f"{date}T00:00:00+00:00",
        "rebalance_date": date,
        "symbol": symbol,
        "score": str(score),
        "news_action": action,
        "news_coverage_status": "COVERED",
        "price_plus_news_risk_probability": "0.8" if action == "BLOCK" else "0.2",
        "actual_forward_return_10d": str(forward_return),
    }


def _bars(symbol: str, rows: list[tuple[str, float, float, float, float]]) -> list[dict[str, object]]:
    return [
        {
            "date": item[0],
            "timestamp": item[0],
            "open": item[1],
            "high": item[2],
            "low": item[3],
            "close": item[4],
            "volume": 1000.0,
            "symbol": symbol,
        }
        for item in rows
    ]


def _replay_config(**overrides: object) -> dict[str, object]:
    config = {
        "starting_equity": 1.0,
        "top_n": 25,
        "max_positions": 25,
        "max_position_weight": 1.0,
        "max_holding_bars": 1,
        "entry_slippage_bps": 0.0,
        "exit_slippage_bps": 0.0,
        "commission_bps": 0.0,
        "stop_loss_pct": None,
        "profit_target_pct": None,
        "reduce_multiplier": 0.5,
    }
    config.update(overrides)
    return config


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
