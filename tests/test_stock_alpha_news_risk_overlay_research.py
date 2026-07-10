from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import core.research.ml.stock_level.news_risk_overlay_research as research
from core.research.ml.stock_level.news_risk_overlay_research import (
    _accounting_audit,
    _assert_score_direction_contract,
    _assign_candidate_ids,
    _bar_sets_equal,
    _chunks,
    _cost_scenario_comparison,
    _event_category_analysis,
    _load_daily_price_bars,
    _missing_news_bias,
    _news_score_decile_diagnostics,
    _parallel_config,
    _parallel_determinism_status,
    _portfolio_comparison,
    _run_open_trade_replay,
    _score_direction_audit,
    _stable_hash,
    _text_model_readiness,
    _variant_multiplier,
    format_news_risk_overlay_summary,
    inspect_stock_alpha_news_risk_overlay_results,
    write_stock_alpha_news_risk_overlay_research,
)
from core.research.ml.stock_level.news_risk_overlay import NewsRiskOverlayConfig
from core.research.ml.stock_level.news_sources import (
    catastrophic_news_taxonomy_report,
    classify_catastrophic_news_event,
)
from core.research.ml.stock_level.news_transformer import (
    NewsSequenceExample,
    build_news_transformer_readiness_report,
    build_news_transformer_training_plan,
    validate_news_sequence_schema,
    validate_no_random_split,
)


def test_news_risk_overlay_research_writes_end_to_end_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    bars = {
        "AAPL": _bars(
            "AAPL",
            [
                ("2024-01-01", 100.0, 101.0, 99.0, 100.5),
                ("2024-01-02", 100.5, 102.0, 100.0, 101.0),
                ("2024-01-03", 101.0, 102.0, 100.0, 101.5),
                ("2024-01-04", 101.5, 103.0, 101.0, 102.0),
                ("2024-01-05", 102.0, 103.0, 101.0, 102.5),
                ("2024-01-08", 102.5, 104.0, 102.0, 103.0),
                ("2024-01-09", 103.0, 104.0, 102.0, 103.5),
                ("2024-01-10", 103.5, 105.0, 103.0, 104.0),
            ],
        ),
        "MSFT": _bars(
            "MSFT",
            [
                ("2024-01-01", 200.0, 202.0, 199.0, 201.0),
                ("2024-01-02", 201.0, 203.0, 200.0, 202.0),
                ("2024-01-03", 202.0, 204.0, 201.0, 203.0),
                ("2024-01-04", 203.0, 205.0, 202.0, 204.0),
                ("2024-01-05", 204.0, 206.0, 203.0, 205.0),
                ("2024-01-08", 205.0, 207.0, 204.0, 206.0),
                ("2024-01-09", 206.0, 208.0, 205.0, 207.0),
                ("2024-01-10", 207.0, 209.0, 206.0, 208.0),
            ],
        ),
    }

    def fake_bar_loader(symbol: str, _processed_root: Path) -> dict[str, object]:
        return {"status": "OK", "rows": bars[symbol], "elapsed_seconds": 0.0}

    monkeypatch.setattr(research, "_load_daily_price_bar_file", fake_bar_loader)

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
    assert result.corrected_news_score_deciles_csv_path.exists()
    assert result.decile_join_audit_json_path.exists()
    assert result.decile_trade_reconciliation_json_path.exists()
    assert result.chronological_split_manifest_json_path.exists()
    assert result.contrarian_grid_selection_json_path.exists()
    assert result.contrarian_frozen_config_json_path.exists()
    assert result.contrarian_holdout_report_json_path.exists()
    assert result.contrarian_walk_forward_summary_json_path.exists()
    assert result.contrarian_placebo_summary_json_path.exists()
    assert result.universe_survivorship_audit_json_path.exists()
    assert result.corporate_action_audit_json_path.exists()
    assert result.missing_news_bias_report_json_path.exists()
    assert result.text_model_readiness_json_path.exists()
    assert result.catastrophic_news_audit_json_path.exists()
    assert result.catastrophic_news_candidates_csv_path.exists()
    assert result.catastrophic_news_veto_report_json_path.exists()
    assert result.catastrophic_veto_candidate_attribution_json_path.exists()
    assert result.catastrophic_veto_trade_attribution_csv_path.exists()
    assert result.catastrophic_veto_strategy_comparison_json_path.exists()
    assert result.catastrophic_veto_policy_json_path.exists()
    assert result.catastrophic_veto_filtered_strategy_report_json_path.exists()
    assert result.catastrophic_veto_removed_trades_csv_path.exists()
    assert result.catastrophic_veto_removed_symbols_csv_path.exists()
    assert result.catastrophic_veto_full_replay_report_json_path.exists()
    assert result.catastrophic_veto_full_replay_trade_ledger_csv_path.exists()
    assert result.catastrophic_veto_full_replay_equity_csv_path.exists()
    assert result.catastrophic_veto_filtered_candidates_csv_path.exists()
    assert result.catastrophic_veto_blocked_candidates_csv_path.exists()
    assert result.catastrophic_veto_replay_seam_report_json_path.exists()
    assert result.catastrophic_news_evidence_quality_report_json_path.exists()
    assert result.catastrophic_news_evidence_quality_by_field_csv_path.exists()
    assert result.catastrophic_news_evidence_quality_by_symbol_csv_path.exists()
    assert result.catastrophic_veto_policy_mode_comparison_json_path.exists()
    assert result.catastrophic_veto_policy_mode_counts_csv_path.exists()
    assert result.news_evidence_lineage_report_json_path.exists()
    assert result.news_evidence_lineage_by_stage_csv_path.exists()
    assert result.news_evidence_missing_field_examples_csv_path.exists()
    assert result.news_evidence_readiness_report_json_path.exists()
    assert result.news_event_taxonomy_report_json_path.exists()
    assert result.news_event_taxonomy_counts_csv_path.exists()
    assert result.news_event_taxonomy_examples_csv_path.exists()
    assert result.news_duplicate_grouping_report_json_path.exists()
    assert result.news_duplicate_grouping_examples_csv_path.exists()
    assert result.news_point_in_time_text_safety_report_json_path.exists()
    assert result.news_point_in_time_text_safety_examples_csv_path.exists()
    assert result.news_text_keyword_baseline_report_json_path.exists()
    assert result.news_text_keyword_baseline_scores_csv_path.exists()
    assert result.catastrophic_veto_bounceback_report_json_path.exists()
    assert result.catastrophic_veto_bounceback_by_category_csv_path.exists()
    assert result.catastrophic_veto_bounceback_examples_csv_path.exists()
    assert result.catastrophic_veto_extreme_only_policy_proposal_json_path.exists()
    assert result.catastrophic_veto_policy_variant_comparison_json_path.exists()
    assert result.catastrophic_veto_policy_variant_counts_csv_path.exists()
    assert result.catastrophic_veto_policy_variant_metrics_csv_path.exists()
    assert result.catastrophic_veto_policy_variant_removed_trades_csv_path.exists()
    assert result.catastrophic_veto_policy_variant_bounceback_csv_path.exists()
    assert result.catastrophic_veto_policy_frontier_report_json_path.exists()
    assert result.catastrophic_veto_policy_frontier_csv_path.exists()
    assert result.catastrophic_veto_policy_variant_examples_csv_path.exists()
    assert result.catastrophic_veto_loser_bounceback_casebook_json_path.exists()
    assert result.catastrophic_veto_loser_bounceback_cases_csv_path.exists()
    assert result.catastrophic_veto_loser_bounceback_feature_diff_csv_path.exists()
    assert result.catastrophic_veto_loser_bounceback_keyword_diff_csv_path.exists()
    assert result.catastrophic_veto_taxonomy_improvement_plan_json_path.exists()
    coverage = json.loads(result.coverage_json_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_json_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))
    assert coverage["row_coverage_ratio"] == 1.0
    assert metrics["price_plus_news"]["oos_rows"] > 0
    assert metrics["paper_trading_enabled"] is False
    assert metrics["live_trading_enabled"] is False
    assert metrics["paper_orders_enabled"] is False
    assert manifest["mode"] == "ml-stock-alpha-news-risk-overlay-research"
    assert manifest["research_only"] is True
    assert manifest["trading_impact"] == "none"
    assert manifest["price_candidates_path"] == str(price_path)
    assert manifest["news_features_path"] == str(news_path)
    assert manifest["output_dir"] == str(tmp_path / "research-results")
    assert manifest["full_joined_row_count"] == 8
    assert manifest["dataset_csv_row_count"] == 8
    assert manifest["shadow_csv_row_count"] > 0
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
    decile_reconciliation = json.loads(result.decile_trade_reconciliation_json_path.read_text(encoding="utf-8"))
    holdout = json.loads(result.contrarian_holdout_report_json_path.read_text(encoding="utf-8"))
    readiness = json.loads(result.text_model_readiness_json_path.read_text(encoding="utf-8"))
    assert decile_reconciliation["one_trade_no_more_than_one_decile"] is True
    assert holdout["validation_label"] == "PSEUDO_HOLDOUT"
    assert holdout["holdout_type"] == "PSEUDO_HOLDOUT"
    assert holdout["evaluation_result"] == "PASSED_WITH_WARNINGS"
    assert holdout["validation_passed"] is False
    assert holdout["selection_round_trip_cost_bps"] == 10.0
    assert readiness["transformer_trained"] is False
    risk_metrics = json.loads(result.replay_risk_metrics_json_path.read_text(encoding="utf-8"))
    assert "price_only" in risk_metrics
    assert "news_contrarian_rerank" in risk_metrics
    assert "news_contrarian_rerank_catastrophic_veto_confirmed_only" in risk_metrics
    assert "news_contrarian_rerank_catastrophic_veto_manual_review" in risk_metrics


def test_news_risk_overlay_research_path_builder_preserves_artifact_path_contract(tmp_path: Path) -> None:
    paths = research.build_news_risk_research_paths(tmp_path / "overlay")

    expected = {
        "dataset_csv_path": "stock_alpha_news_risk_overlay_dataset.csv",
        "coverage_json_path": "coverage_report.json",
        "leakage_json_path": "leakage_report.json",
        "metrics_json_path": "logistic_regression_metrics.json",
        "portfolio_json_path": "price_vs_news_portfolio_report.json",
        "chronological_split_manifest_json_path": "chronological_split_manifest.json",
        "contrarian_grid_selection_json_path": "contrarian_grid_selection.json",
        "catastrophic_news_audit_json_path": "catastrophic_news_audit.json",
        "catastrophic_veto_policy_mode_comparison_json_path": "catastrophic_veto_policy_mode_comparison.json",
        "news_evidence_lineage_report_json_path": "news_evidence_lineage_report.json",
        "news_event_taxonomy_report_json_path": "news_event_taxonomy_report.json",
        "news_point_in_time_text_safety_report_json_path": "news_point_in_time_text_safety_report.json",
        "shadow_csv_path": "shadow_decision_log.csv",
        "manifest_json_path": "model_manifest.json",
        "markdown_path": "README.md",
    }

    assert paths.output_dir == tmp_path / "overlay"
    for attribute, filename in expected.items():
        assert getattr(paths, attribute) == paths.output_dir / filename


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


def test_news_risk_overlay_research_raises_on_timestamp_leakage_if_join_reports_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_path = tmp_path / "price_candidates.csv"
    news_path = tmp_path / "news_features.csv"
    _write_csv(
        price_path,
        [_price_row("2024-01-01", "AAPL", -0.06, 0.90, -0.07)],
    )
    _write_csv(
        news_path,
        [
            {
                "symbol": "AAPL",
                "available_at_timestamp": "2024-01-01T01:00:00+00:00",
                "sentiment": "-0.8",
            }
        ],
    )

    def fake_join(price_rows, _news_rows, _overlay_config):
        joined = [dict(price_rows[0], decision_timestamp="2024-01-01T00:00:00+00:00", news_coverage_status="COVERED")]
        return joined, {
            "leakage_violation_count": 1,
            "future_news_rows_rejected": 0,
            "symbol_coverage": {"AAPL": 1},
            "date_coverage": {"2024-01-01": 1},
        }

    monkeypatch.setattr(research, "join_news_to_stock_alpha_observations", fake_join)

    with pytest.raises(ValueError, match="timestamp leakage detected"):
        write_stock_alpha_news_risk_overlay_research(
            {
                "ml": {
                    "stock_alpha_news_risk_overlay_price_candidates_path": str(price_path),
                    "stock_alpha_news_risk_overlay_news_features_path": str(news_path),
                    "stock_alpha_news_risk_overlay_output_dir": str(tmp_path / "research-results"),
                }
            }
        )


def test_decile_attribution_uses_candidate_id_and_reconciles_unique_trades() -> None:
    rows = [
        {
            "decision_timestamp": f"2024-01-{index + 1:02d}T00:00:00+00:00",
            "symbol": "AAPL",
            "score": str(index / 10),
            "price_plus_news_risk_probability": str(index / 10),
            "actual_forward_return_10d": str((index - 5) / 100),
        }
        for index in range(10)
    ]
    _assign_candidate_ids(rows, "score")
    ledger = [
        {
            "trade_id": "trade-low",
            "candidate_id": rows[0]["candidate_id"],
            "strategy_variant": "price_only",
            "net_return": "0.01",
        },
        {
            "trade_id": "trade-high",
            "candidate_id": rows[-1]["candidate_id"],
            "strategy_variant": "news_contrarian_rerank",
            "net_return": "-0.02",
        },
    ]

    deciles, _direction, join_audit, reconciliation = _news_score_decile_diagnostics(
        rows,
        ledger,
        price_score_column="score",
    )

    assert sum(row["unique_trade_count"] for row in deciles) == 2
    assert join_audit["schema_name"] == "news_risk_decile_join_audit"
    assert join_audit["join_keys"] == ["candidate_id", "strategy_variant"]
    assert join_audit["candidate_multiple_decile_count"] == 0
    assert join_audit["duplicate_candidate_id_count"] == 0
    assert join_audit["duplicate_trade_id_count"] == 0
    assert join_audit["strategy_variant_mismatch_count"] == 0
    assert reconciliation["one_candidate_exactly_one_decile"] is True
    assert reconciliation["one_trade_no_more_than_one_decile"] is True
    assert reconciliation["total_unique_matched_trades"] == 2
    assert reconciliation["schema_name"] == "news_risk_decile_trade_reconciliation"
    assert reconciliation["by_strategy_variant"]["price_only"]["unique_matched_trade_ids"] == 1
    assert reconciliation["by_strategy_variant"]["news_contrarian_rerank"]["unique_matched_trade_ids"] == 1


def test_decile_reconciliation_flags_duplicates_and_missing_scores() -> None:
    rows = [
        {
            "candidate_id": "candidate-a",
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "1.0",
            "price_plus_news_risk_probability": "0.5",
            "actual_forward_return_10d": "0.01",
        },
        {
            "candidate_id": "candidate-a",
            "decision_timestamp": "2024-01-02T00:00:00+00:00",
            "symbol": "MSFT",
            "score": "0.8",
            "price_plus_news_risk_probability": "0.0",
            "actual_forward_return_10d": "-0.02",
        },
        {
            "candidate_id": "candidate-missing",
            "decision_timestamp": "2024-01-03T00:00:00+00:00",
            "symbol": "GOOG",
            "score": "0.7",
            "actual_forward_return_10d": "0.03",
        },
    ]
    ledger = [
        {
            "trade_id": "duplicate-trade",
            "candidate_id": "candidate-a",
            "strategy_variant": "news_contrarian_rerank",
            "net_return": "0.01",
        },
        {
            "trade_id": "duplicate-trade",
            "candidate_id": "candidate-a",
            "strategy_variant": "news_contrarian_rerank",
            "net_return": "0.02",
        },
        {
            "trade_id": "unmatched-trade",
            "candidate_id": "unknown-candidate",
            "strategy_variant": "news_contrarian_rerank",
            "net_return": "-0.01",
        },
        {
            "trade_id": "wrong-variant",
            "candidate_id": "candidate-a",
            "strategy_variant": "diagnostic_variant",
            "net_return": "0.05",
        },
    ]

    _deciles, _direction, join_audit, reconciliation = _news_score_decile_diagnostics(
        rows,
        ledger,
        price_score_column="score",
    )

    assert join_audit["status"] == "FAILED"
    assert join_audit["duplicate_candidate_id_count"] == 1
    assert join_audit["duplicate_trade_id_count"] == 1
    assert join_audit["missing_news_score_count"] == 1
    assert join_audit["neutral_news_score_count"] == 1
    assert join_audit["cross_strategy_candidate_trade_pairs_excluded"] == 1
    assert join_audit["strategy_variant_mismatch_count"] == 0
    assert join_audit["strategy_variant_mismatch_is_error"] is False
    assert join_audit["unmatched_trade_rows"] == 1
    assert reconciliation["by_strategy_variant"]["news_contrarian_rerank"]["unmatched_trade_rows"] == 1


def test_decile_reconciliation_counts_all_missing_scores_without_neutralising_them() -> None:
    rows = [
        {
            "candidate_id": "candidate-missing",
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "1.0",
            "actual_forward_return_10d": "0.01",
        }
    ]
    ledger = [
        {
            "trade_id": "trade-missing-score",
            "candidate_id": "candidate-missing",
            "strategy_variant": "news_contrarian_rerank",
            "net_return": "0.02",
        }
    ]

    deciles, _direction, join_audit, reconciliation = _news_score_decile_diagnostics(
        rows,
        ledger,
        price_score_column="score",
    )

    assert deciles == []
    assert join_audit["schema_name"] == "news_risk_decile_join_audit"
    assert join_audit["status"] == "PASSED_WITH_WARNINGS"
    assert join_audit["missing_news_score_count"] == 1
    assert join_audit["neutral_news_score_count"] == 0
    assert join_audit["unmatched_candidate_rows"] == 1
    assert join_audit["unmatched_trade_rows"] == 1
    assert reconciliation["schema_name"] == "news_risk_decile_trade_reconciliation"
    assert reconciliation["by_strategy_variant"]["news_contrarian_rerank"]["unmatched_trade_rows"] == 1


def test_decile_reconciliation_adds_missing_trade_id_as_provenance_only() -> None:
    rows = [
        {
            "candidate_id": "candidate-a",
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "1.0",
            "price_plus_news_risk_probability": "0.5",
            "actual_forward_return_10d": "0.01",
        }
    ]
    ledger = [
        {
            "candidate_id": "candidate-a",
            "strategy_variant": "news_contrarian_rerank",
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "entry_date": "2024-01-02",
            "exit_date": "2024-01-03",
            "net_return": "0.02",
        }
    ]

    _deciles, _direction, join_audit, reconciliation = _news_score_decile_diagnostics(
        rows,
        ledger,
        price_score_column="score",
    )

    assert ledger[0]["trade_id"]
    assert ledger[0]["model_version"] == "news-risk-overlay-research-v1"
    assert join_audit["matched_trade_rows"] == 1
    assert reconciliation["by_strategy_variant"]["news_contrarian_rerank"]["unique_matched_trade_ids"] == 1


def test_decile_attribution_requires_candidate_id() -> None:
    rows = [
        {
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "1.0",
            "price_plus_news_risk_probability": "0.5",
            "actual_forward_return_10d": "0.01",
        }
    ]

    with pytest.raises(ValueError, match="candidate_id"):
        _news_score_decile_diagnostics(rows, [], price_score_column="score")


def test_frozen_config_hash_ignores_generation_timestamp() -> None:
    first = {"configuration_id": "a", "contrarian_weight": 0.25, "generated_timestamp": "one"}
    second = {"configuration_id": "a", "contrarian_weight": 0.25, "generated_timestamp": "two"}

    assert _stable_hash(first) == _stable_hash(second)


def test_frozen_config_hash_helper_ignores_only_declared_volatile_fields() -> None:
    first = {
        "configuration_id": "a",
        "contrarian_weight": 0.25,
        "generated_timestamp": "one",
        "hash_excluded_fields": ["generated_timestamp"],
    }
    second = {
        "configuration_id": "a",
        "contrarian_weight": 0.25,
        "generated_timestamp": "two",
        "hash_excluded_fields": ["generated_timestamp"],
    }
    changed = {
        "configuration_id": "a",
        "contrarian_weight": 0.50,
        "generated_timestamp": "two",
        "hash_excluded_fields": ["generated_timestamp"],
    }

    assert research._frozen_config_hash(first) == research._frozen_config_hash(second)
    assert research._frozen_config_hash(first) != research._frozen_config_hash(changed)


def test_grid_selection_prefers_eligible_cost_aware_metrics() -> None:
    winner = research._select_contrarian_grid_configuration(
        [
            {
                "configuration_id": "a",
                "eligible": True,
                "median_validation_calmar": 0.7,
                "median_excess_return": 0.20,
                "median_excess_sharpe": 0.1,
                "contrarian_weight": 0.10,
            },
            {
                "configuration_id": "b",
                "eligible": True,
                "median_validation_calmar": 1.1,
                "median_excess_return": 0.05,
                "median_excess_sharpe": 0.2,
                "contrarian_weight": 0.25,
            },
            {
                "configuration_id": "c",
                "eligible": False,
                "median_validation_calmar": 9.0,
                "median_excess_return": 9.0,
                "median_excess_sharpe": 9.0,
                "contrarian_weight": 1.0,
            },
        ]
    )

    assert winner["configuration_id"] == "b"


def test_retuning_refusal_blocks_final_evaluation_overrides() -> None:
    gate = research._retuning_refusal_gate(
        {"parameter_overrides_allowed_for_final_evaluation": False},
        override_requested=True,
    )

    assert gate["status"] == "REFUSED"
    assert gate["final_evaluation_valid"] is False
    assert gate["overrides_allowed_for_final_evaluation"] is False


def test_text_model_readiness_is_not_ready_without_enabling_bert_or_transformers() -> None:
    report = research._not_ready_text_model_report(
        {
            "schema_name": "stock_alpha_news_text_model_readiness",
            "ready_for_text_model": True,
            "available_text_columns": ["headline"],
            "transformer_trained": True,
        }
    )

    assert report["status"] == "NOT_READY"
    assert report["ready_for_text_model"] is False
    assert report["finbert_readiness"] == "NOT_READY"
    assert report["bert_readiness"] == "NOT_READY"
    assert report["numeric_transformer_readiness"] == "NOT_READY"
    assert report["transformer_trained"] is False
    assert report["bert_enabled"] is False
    assert report["finbert_enabled"] is False
    assert report["transformer_training_enabled"] is False
    assert report["recommended_next_baseline"] == "structured_event_taxonomy_or_simple_text_baseline"
    assert "FinBERT deferred" in report["warnings"]


def test_contrarian_selection_uses_one_predeclared_cost_assumption() -> None:
    rows = [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")]
    replay = {
        "risk_metrics": {
            "price_only": {"total_return_decimal": 0.10, "Sharpe_ratio": 1.0, "Calmar_ratio": 0.5},
            "news_contrarian_rerank": {"total_return_decimal": 0.12, "Sharpe_ratio": 1.1, "Calmar_ratio": 0.6},
        }
    }
    periods = {
        "periods": {
            "development": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            "parameter_validation": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
        }
    }

    grid, _folds, selection = research._contrarian_grid_reports(
        rows,
        replay,
        "score",
        periods,
        {"stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 7.5},
    )
    frozen = research._frozen_contrarian_config(selection, {"stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 7.5}, periods)

    assert {row["selection_round_trip_cost_bps"] for row in grid} == {7.5}
    assert selection["selection_round_trip_cost_bps"] == 7.5
    assert selection["experiment_registry_metadata"]["research_only"] is True
    assert selection["experiment_registry_metadata"]["news_originated_entries_enabled"] is False
    assert frozen["selection_round_trip_cost_bps"] == 7.5
    assert frozen["exact_formula"] == "contrarian_score = price_score + contrarian_weight * transformed_news_score"
    assert frozen["hash_excluded_fields"] == ["generated_timestamp"]
    assert frozen["experiment_registry_metadata"]["status"] == "DEVELOPMENT_ONLY"
    assert frozen["validation_label"] == "PSEUDO_HOLDOUT"
    assert frozen["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert frozen["validation_passed"] is False
    assert frozen["paper_orders_enabled"] is False
    assert frozen["live_orders_enabled"] is False


def test_chronological_split_manifest_preserves_whole_decision_dates() -> None:
    rows = [
        _candidate("2024-01-01", "AAA", 1.0, "ALLOW"),
        _candidate("2024-01-01", "BBB", 0.9, "ALLOW"),
        _candidate("2024-01-02", "CCC", 0.8, "ALLOW"),
        _candidate("2024-01-03", "DDD", 0.7, "ALLOW"),
        _candidate("2024-01-04", "EEE", 0.6, "ALLOW"),
        _candidate("2024-01-05", "FFF", 0.5, "ALLOW"),
    ]
    ledger = [
        {"decision_timestamp": "2024-01-01T00:00:00+00:00", "strategy_variant": "price_only"},
        {"decision_timestamp": "2024-01-05T00:00:00+00:00", "strategy_variant": "news_contrarian_rerank"},
    ]

    manifest = research._chronological_periods(rows, ledger)
    periods = manifest["periods"]
    date_membership = {}
    for period_name, payload in periods.items():
        start = payload["start_date"]
        end = payload["end_date"]
        for row in rows:
            date_key = row["decision_timestamp"][:10]
            if start and end and start <= date_key <= end:
                date_membership.setdefault(date_key, set()).add(period_name)

    assert set(periods) == {"development", "parameter_validation", "final_holdout"}
    assert all(len(names) == 1 for names in date_membership.values())
    assert periods["development"]["candidate_count"] >= 2
    assert manifest["decision_date_integrity_check"]["passed"] is True
    assert manifest["holdout_type"] == "PSEUDO_HOLDOUT"
    assert manifest["validation_label"] == "PSEUDO_HOLDOUT"
    assert manifest["is_final_validation"] is False
    assert manifest["validation_passed"] is False
    assert manifest["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert periods["final_holdout"]["previously_inspected"] is True


def test_walk_forward_validation_artifact_is_chronological_and_non_final() -> None:
    rows = [
        _candidate("2024-01-01", "AAA", 1.0, "ALLOW"),
        _candidate("2024-01-02", "BBB", 0.9, "ALLOW"),
        _candidate("2024-01-03", "CCC", 0.8, "ALLOW"),
        _candidate("2024-01-04", "DDD", 0.7, "ALLOW"),
        _candidate("2024-01-05", "EEE", 0.6, "ALLOW"),
    ]
    periods = research._chronological_periods(rows, [])
    frozen = {"configuration_id": "config-a", "immutable_configuration_hash": "hash-a"}
    selection = {"selected_configuration_id": "config-a"}

    report, folds = research._walk_forward_validation_artifacts(
        {"risk_metrics": {"price_only": {}, "news_contrarian_rerank": {}}},
        periods,
        selection,
        frozen,
        {"stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 10.0},
    )

    assert report["status"] == "NOT_IMPLEMENTED"
    assert report["validation_passed"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert report["configuration_id"] == "config-a"
    assert report["frozen_configuration_hash"] == "hash-a"
    assert report["selection_round_trip_cost_bps"] == 10.0
    assert all(row["random_split_used"] is False for row in folds)
    assert all(row["same_decision_date_crosses_folds"] is False for row in folds)
    assert next(row for row in folds if row["period_name"] == "final_holdout")["used_for_parameter_selection"] is False
    assert all(row["uses_frozen_configuration"] is True for row in folds)


def test_placebo_permutation_artifact_is_deterministic_and_not_used_for_selection() -> None:
    report, rows = research._placebo_permutation_artifacts(
        [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")],
        {},
        {"stock_alpha_news_risk_overlay_seed": 123},
    )

    assert report["status"] == "UNAVAILABLE_INPUT"
    assert report["deterministic_seed"] == 123
    assert report["validation_passed"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert {row["check_name"] for row in rows} == {
        "news_score_permuted_by_decision_date",
        "news_score_permuted_globally",
        "news_score_sign_flipped",
        "random_decile_assignment_fixed_seed",
    }
    assert all(row["status"] == "UNAVAILABLE_INPUT" for row in rows)
    assert all(row["used_for_configuration_selection"] is False for row in rows)


def test_matched_control_and_concentration_artifacts_block_final_validation() -> None:
    exposure = research._matched_control_artifact("exposure_matched_controls")
    concentration = research._concentration_fragility_artifact({"contrarian_trade_ledger": [{"net_return": "0.10"}]})

    assert exposure["status"] == "NOT_IMPLEMENTED"
    assert exposure["blocks_final_validation"] is True
    assert exposure["validation_passed"] is False
    assert concentration["validation_passed"] is False
    assert concentration["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_catastrophic_veto_parked_status_is_diagnostic_only() -> None:
    parked = research._catastrophic_veto_parked_status(
        {"status": "AVAILABLE", "replay_impact_status": "BROAD_RETURN_DRAG"},
        {"frontier_status": "NO_USABLE_POLICY"},
        {"status": "AVAILABLE", "severe_loser_case_count": 3, "strong_bounceback_case_count": 2},
    )

    assert parked["status"] == "PARKED_DIAGNOSTIC_ONLY"
    assert parked["used_in_current_strategy"] is False
    assert parked["paper_trading_enabled"] is False
    assert parked["live_trading_enabled"] is False
    assert parked["validation_label"] == "PSEUDO_HOLDOUT"
    assert parked["validation_passed"] is False
    assert parked["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_contrarian_chronological_validation_plan_uses_complete_dates_and_pseudo_holdout() -> None:
    rows = [
        _candidate("2024-01-01", "AAA", 1.0, "ALLOW"),
        _candidate("2024-01-01", "BBB", 1.0, "ALLOW"),
        _candidate("2024-01-02", "CCC", 1.0, "ALLOW"),
        _candidate("2024-01-03", "DDD", 1.0, "ALLOW"),
    ]
    replay = {
        "trade_ledger": [
            {"strategy_variant": "news_contrarian_rerank", "entry_date": "2024-01-03", "net_pnl": "10"}
        ]
    }
    periods = {
        "periods": {
            "development": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            "parameter_validation": {"start_date": "2024-01-02", "end_date": "2024-01-02"},
            "final_holdout": {"start_date": "2024-01-03", "end_date": "2024-01-03"},
        }
    }

    plan, period_rows = research._contrarian_chronological_validation_plan(rows, replay, periods)
    pseudo_holdout = next(row for row in period_rows if row["period_name"] == "pseudo_holdout")
    future_holdout = next(row for row in period_rows if row["period_name"] == "future_final_holdout")

    assert plan["split_method"] == "chronological_by_complete_decision_date"
    assert plan["complete_decision_dates_only"] is True
    assert pseudo_holdout["decision_date_count"] == 1
    assert pseudo_holdout["trade_count_if_available"] == 1
    assert pseudo_holdout["contamination_status"] == "PSEUDO_HOLDOUT_PREVIOUSLY_INSPECTED"
    assert pseudo_holdout["used_for_final_validation"] is False
    assert future_holdout["start_date"] == "NOT_YET_DEFINED"
    assert plan["validation_passed"] is False
    assert plan["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_contrarian_validation_scaffolds_do_not_emit_fake_metrics() -> None:
    placebo, placebo_rows = research._contrarian_placebo_permutation_report(
        {"stock_alpha_news_risk_overlay_seed": 99}
    )
    matched, matched_rows = research._contrarian_matched_control_report()

    assert placebo["status"] == "UNAVAILABLE_INPUT"
    assert placebo["deterministic_seed"] == 99
    assert all(row["status"] == "UNAVAILABLE_INPUT" for row in placebo_rows)
    assert all(row["return"] == "UNAVAILABLE_INPUT" for row in placebo_rows)
    assert matched["status"] == "NOT_IMPLEMENTED"
    assert all(row["status"] == "NOT_IMPLEMENTED" for row in matched_rows)
    assert all(row["return"] == "UNAVAILABLE_INPUT" for row in matched_rows)
    assert placebo["validation_passed"] is False
    assert matched["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_profit_concentration_uses_trade_ledger_and_top_trade_removal_is_deterministic() -> None:
    replay = {
        "trade_ledger": [
            {"trade_id": "T2", "strategy_variant": "news_contrarian_rerank", "symbol": "BBB", "entry_date": "2024-01-02", "net_pnl": "20", "net_return": "0.20"},
            {"trade_id": "T1", "strategy_variant": "news_contrarian_rerank", "symbol": "AAA", "entry_date": "2024-01-01", "net_pnl": "10", "net_return": "0.10"},
            {"trade_id": "PX", "strategy_variant": "price_only", "symbol": "AAA", "entry_date": "2024-01-01", "net_pnl": "999", "net_return": "9.99"},
        ]
    }

    report, symbol_rows, year_rows, top_rows = research._contrarian_profit_concentration_artifacts(replay)

    assert report["status"] == "IMPLEMENTED"
    assert report["trade_count"] == 2
    assert report["total_net_pnl"] == 30.0
    assert report["largest_winner"] == "T2"
    assert symbol_rows[0]["symbol"] == "BBB"
    assert year_rows[0]["year"] == "2024"
    assert top_rows[0]["removed_net_pnl"] == 20.0
    assert top_rows[0]["deterministic_sort"] == "net_pnl_desc_trade_id"
    assert report["validation_passed"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_year_regime_report_identifies_positive_and_negative_years() -> None:
    replay = {
        "trade_ledger": [
            {"trade_id": "A1", "strategy_variant": "news_contrarian_rerank", "symbol": "AAA", "entry_date": "2021-01-02", "net_pnl": "10", "net_return": "0.10"},
            {"trade_id": "A2", "strategy_variant": "news_contrarian_rerank", "symbol": "AAA", "entry_date": "2021-02-02", "net_pnl": "5", "net_return": "0.05"},
            {"trade_id": "B1", "strategy_variant": "news_contrarian_rerank", "symbol": "BBB", "entry_date": "2022-01-02", "net_pnl": "-7", "net_return": "-0.07"},
            {"trade_id": "B2", "strategy_variant": "news_contrarian_rerank", "symbol": "BBB", "entry_date": "2022-02-02", "net_pnl": "-3", "net_return": "-0.03"},
            {"trade_id": "PX", "strategy_variant": "price_only", "symbol": "ZZZ", "entry_date": "2022-01-02", "net_pnl": "999"},
        ]
    }

    report, rows, examples = research._contrarian_year_regime_artifacts(replay)
    by_year = {row["year"]: row for row in rows}

    assert report["status"] == "AVAILABLE"
    assert report["metric_basis"] == "LEDGER_LEVEL_APPROXIMATION"
    assert by_year["2021"]["net_pnl"] == 15.0
    assert by_year["2022"]["net_pnl"] == -10.0
    assert by_year["2022"]["regime_status"] == "negative_year"
    assert report["year_2022_status"] == "negative_year"
    assert examples
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_symbol_and_year_ablation_are_deterministic_and_ledger_level() -> None:
    replay = {
        "trade_ledger": [
            {"trade_id": "A1", "strategy_variant": "news_contrarian_rerank", "symbol": "AAA", "entry_date": "2021-01-02", "net_pnl": "20", "net_return": "0.20"},
            {"trade_id": "A2", "strategy_variant": "news_contrarian_rerank", "symbol": "AAA", "entry_date": "2021-02-02", "net_pnl": "10", "net_return": "0.10"},
            {"trade_id": "B1", "strategy_variant": "news_contrarian_rerank", "symbol": "BBB", "entry_date": "2022-01-02", "net_pnl": "-5", "net_return": "-0.05"},
            {"trade_id": "C1", "strategy_variant": "news_contrarian_rerank", "symbol": "CCC", "entry_date": "2023-01-02", "net_pnl": "15", "net_return": "0.15"},
        ]
    }

    first_report, first_symbols, first_years = research._contrarian_symbol_year_ablation_artifacts(replay)
    second_report, second_symbols, second_years = research._contrarian_symbol_year_ablation_artifacts(replay)
    symbol_by_name = {row["ablation_name"]: row for row in first_symbols}
    year_by_name = {row["ablation_name"]: row for row in first_years}

    assert first_symbols == second_symbols
    assert first_years == second_years
    assert first_report["metric_basis"] == "LEDGER_LEVEL_APPROXIMATION"
    assert symbol_by_name["without_top_1_symbol"]["removed_group"] == "AAA"
    assert symbol_by_name["without_top_1_symbol"]["removed_trade_count"] == 2
    assert "LEDGER_LEVEL_APPROXIMATION" in symbol_by_name["without_top_1_symbol"]["warnings"]
    assert year_by_name["without_top_1_year"]["removed_group"] == "2021"
    assert year_by_name["without_negative_years"]["removed_group"] == "2022"
    assert first_report["validation_passed"] is False
    assert second_report["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_walk_forward_folds_use_complete_dates_and_do_not_fake_metrics() -> None:
    rows = [
        _candidate("2024-01-01", "AAA", 1.0, "ALLOW"),
        _candidate("2024-01-01", "BBB", 1.0, "ALLOW"),
        _candidate("2024-01-02", "CCC", 1.0, "ALLOW"),
        _candidate("2024-01-03", "DDD", 1.0, "ALLOW"),
    ]
    periods = research._chronological_periods(rows, [])
    report, folds = research._walk_forward_validation_artifacts(
        {"risk_metrics": {"price_only": {}, "news_contrarian_rerank": {}}},
        periods,
        {"selected_configuration_id": "cfg"},
        {"configuration_id": "cfg", "immutable_configuration_hash": "hash"},
        {},
    )

    assert report["status"] == "NOT_IMPLEMENTED"
    assert all(row["random_split_used"] is False for row in folds)
    assert all(row["same_decision_date_crosses_folds"] is False for row in folds)
    assert all(row["metric_status"] == "NOT_IMPLEMENTED" for row in folds)
    assert all(row["wealth"] == "UNAVAILABLE_INPUT" for row in folds)
    assert all(row["test_start"] == row["start_date"] for row in folds)
    assert report["validation_passed"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_cost_slippage_robustness_preserves_existing_table_and_marks_extra_costs() -> None:
    report, rows = research._contrarian_cost_slippage_robustness(
        {
            "scenarios": {
                "10_bps_round_trip": {
                    "variants": {
                        "news_contrarian_rerank": {"ending_equity": 1.2, "total_return_decimal": 0.2},
                        "price_only": {"ending_equity": 1.1, "total_return_decimal": 0.1},
                    }
                }
            }
        }
    )
    row_by_cost = {row["cost_bps"]: row for row in rows}

    assert report["status"] == "PARTIAL_EXISTING_COST_TABLE"
    assert report["computed_cost_bps"] == [10]
    assert 100 in report["not_computed_cost_bps"]
    assert row_by_cost[10]["cost_robustness_status"] == "COMPUTED_EXISTING_COST_TABLE"
    assert row_by_cost[10]["metric_status"] == "COMPUTED_FROM_EXISTING_COST_TABLE"
    assert row_by_cost[10]["beats_price_only"] is True
    assert row_by_cost[100]["cost_robustness_status"] == "NOT_COMPUTED"
    assert row_by_cost[100]["metric_status"] == "NOT_COMPUTED"
    assert row_by_cost[100]["return"] == "UNAVAILABLE_INPUT"
    assert report["validation_passed"] is False


def test_contrarian_data_validity_audit_blocks_final_validation() -> None:
    audit = research._contrarian_data_validity_audit(
        {"missing_bar_count": 2},
        {"status": "INSUFFICIENT_DATA"},
    )

    assert audit["status"] == "BLOCKING"
    assert audit["checks"]["missing_price_bars"]["status"] == "FAILED"
    assert audit["checks"]["survivorship_bias"]["blocks_final_validation"] is True
    assert audit["checks"]["survivorship_bias"]["evidence_available"] is False
    assert audit["checks"]["survivorship_bias"]["required_input_files"] == [
        "point_in_time_universe_membership",
        "delisted_symbol_reference",
    ]
    assert "survivorship_bias" in audit["blocking_checks"]
    assert "missing_news_bias" in audit["blocking_checks"]
    assert audit["validation_label"] == "PSEUDO_HOLDOUT"
    assert audit["validation_passed"] is False
    assert audit["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_intraday_5min_expansion_plan_is_planning_only_and_trading_disabled() -> None:
    plan = research._intraday_5min_expansion_plan({})

    assert plan["status"] == "PLANNING_ONLY"
    assert plan["target_machine"] == "Dell PC"
    assert plan["recommended_data_frequency"] == "5min"
    assert plan["required_years"] == "TO_BE_CONFIRMED"
    assert plan["parquet_conversion_required"] is True
    assert "run a small subset first" in plan["pipeline_steps"]
    assert plan["validation_passed"] is False
    assert plan["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert plan["paper_trading_enabled"] is False
    assert plan["live_trading_enabled"] is False


def test_experiment_registry_appends_development_only_entries(tmp_path: Path) -> None:
    path = tmp_path / "experiment_registry.jsonl"
    replay = {
        "risk_metrics": {
            "price_only": {"total_return_decimal": 0.10, "Sharpe_ratio": 1.0},
            "news_contrarian_rerank": {"total_return_decimal": 0.20, "Sharpe_ratio": 1.2},
        }
    }
    kwargs = {
        "rows": [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")],
        "replay": replay,
        "validation": {},
        "coverage": {"row_coverage_ratio": 0.429},
        "event_category_analysis": {"general_negative_sentiment_or_uncategorized": {"count": 10}},
        "decile_reconciliation": {"status": "PASSED"},
        "config": {"stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 10.0},
    }

    first = research._append_experiment_registry_entry(path, **kwargs)
    second = research._append_experiment_registry_entry(path, **kwargs)
    rows = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(rows) == 2
    assert first["status"] == "DEVELOPMENT_ONLY"
    assert second["validation_label"] == "PSEUDO_HOLDOUT"
    assert first["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert first["is_final_validation"] is False
    assert first["validation_passed"] is False
    assert first["hypothesis"] == "Among price-model-approved candidates, downside-news pressure may improve ranking."
    assert first["decile_reconciliation_status"] == "PASSED"
    assert first["grid_selection_status"] == "UNKNOWN"
    assert first["pseudo_holdout_gate_status"] == "UNKNOWN"
    assert "frozen_config_hash" in first
    assert first["holdout_type"] == "PSEUDO_HOLDOUT"
    assert first["selected_transaction_cost_bps"] == 10.0
    assert "bad news = buy" not in json.dumps(first)


def test_contrarian_selection_cost_controls_winning_configuration() -> None:
    rows = [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")]
    replay = {"risk_metrics": {"price_only": {}, "news_contrarian_rerank": {}}}
    periods = {"periods": {"parameter_validation": {"start_date": "2024-01-01", "end_date": "2024-01-01"}}}

    _grid, _folds, selection = research._contrarian_grid_reports(
        rows,
        replay,
        "score",
        periods,
        {
            "stock_alpha_news_risk_overlay_contrarian_weight": 0.10,
            "stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 10.0,
            "stock_alpha_news_risk_overlay_contrarian_grid_weights": [0.10, 0.25],
            "stock_alpha_news_risk_overlay_contrarian_grid_cost_metrics": {
                "w0.10_raw_probability_no_adjustment": {
                    "0": {"median_validation_calmar": 2.0, "median_excess_return": 0.20, "median_excess_sharpe": 0.3},
                    "10": {"median_validation_calmar": 0.4, "median_excess_return": 0.01, "median_excess_sharpe": 0.1},
                },
                "w0.25_raw_probability_no_adjustment": {
                    "0": {"median_validation_calmar": 1.0, "median_excess_return": 0.10, "median_excess_sharpe": 0.2},
                    "10": {"median_validation_calmar": 1.5, "median_excess_return": 0.08, "median_excess_sharpe": 0.4},
                },
            },
        },
    )

    assert selection["selection_round_trip_cost_bps"] == 10.0
    assert selection["selected_configuration_id"] == "w0.25_raw_probability_no_adjustment"


def test_selection_cost_metrics_can_be_sourced_from_cost_scenarios() -> None:
    metrics = research._selection_cost_metrics_from_scenarios(
        {"stock_alpha_news_risk_overlay_contrarian_weight": 0.25},
        {
            "scenarios": {
                "10_bps_round_trip": {
                    "round_trip_bps": 10.0,
                    "variants": {
                        "price_only": {"total_return_decimal": 0.10, "Sharpe_ratio": 1.0, "Calmar_ratio": 0.5},
                        "news_contrarian_rerank": {"total_return_decimal": 0.18, "Sharpe_ratio": 1.3, "Calmar_ratio": 0.9},
                    },
                }
            }
        },
    )

    selected = metrics["w0.25_raw_probability_no_adjustment"]["10"]
    assert selected["median_validation_calmar"] == 0.9
    assert selected["median_excess_return"] == pytest.approx(0.08)


def test_pseudo_holdout_status_never_reports_validation_passed() -> None:
    report = research._holdout_report(
        {
            "risk_metrics": {
                "price_only": {"total_return_decimal": 0.10, "Sharpe_ratio": 1.0},
                "news_contrarian_rerank": {"total_return_decimal": 0.30, "Sharpe_ratio": 2.0},
            }
        },
        {"periods": {"final_untouched_holdout": {"start_date": "2024-01-01", "end_date": "2024-01-31"}}},
        {"immutable_configuration_hash": "abc", "selection_round_trip_cost_bps": 10.0},
        {},
    )

    assert report["validation_label"] == "PSEUDO_HOLDOUT"
    assert report["holdout_type"] == "PSEUDO_HOLDOUT"
    assert report["evaluation_result"] == "PASSED_WITH_WARNINGS"
    assert report["is_final_validation"] is False
    assert report["validation_passed"] is False
    assert report["validation_gate"]["status"] == "BLOCKED_PSEUDO_HOLDOUT"
    assert report["validation_gate"]["validation_passed"] is False
    assert report["validation_gate"]["retuning_gate_passed"] is True
    assert report["validation_gate"]["final_validation_blocked_by_pseudo_holdout"] is True
    assert report["validation_gate"]["final_evaluation_valid"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert "VALIDATION_PASSED" not in json.dumps(report)


def test_validation_stage_placeholders_are_not_metric_outputs() -> None:
    placeholders = research._validation_stage_placeholders()

    assert placeholders["walk_forward"]["status"] == "NOT_IMPLEMENTED"
    assert placeholders["walk_forward"]["implemented"] is False
    assert placeholders["walk_forward"]["blocks_final_validation"] is True
    assert placeholders["walk_forward"]["metric_output_allowed"] is False
    assert placeholders["walk_forward"]["reason"]
    assert "warnings" in placeholders["walk_forward"]
    assert placeholders["placebo_permutation"]["status"] == "NOT_IMPLEMENTED"
    assert placeholders["exposure_matched_controls"]["blocks_final_validation"] is True
    assert placeholders["trade_count_matched_controls"]["metric_output_allowed"] is False
    assert placeholders["text_model_readiness"]["status"] == "NOT_READY"
    assert placeholders["text_model_readiness"]["bert_enabled"] is False
    assert placeholders["text_model_readiness"]["finbert_enabled"] is False


def test_validation_summary_hides_scaffold_metrics_and_preserves_decile_warning() -> None:
    text = "\n".join(
        [
            "- holdout excess return: 1.087",
            "- walk-forward positive folds: 100.0%",
            "- placebo p-value: 0.143",
            "- decile_join_audit.status: FAILED",
            "- warning: identical executed-trade counts across multiple deciles",
            "- status: VALIDATION_PASSED",
        ]
    )

    sanitized = research._sanitize_validation_summary_text(text)

    assert "holdout excess return: 1.087" not in sanitized
    assert "walk-forward positive folds: 100.0%" not in sanitized
    assert "placebo p-value: 0.143" not in sanitized
    assert "holdout validation metric: PSEUDO_HOLDOUT_ONLY" in sanitized
    assert "walk-forward validation metric: NOT_IMPLEMENTED" in sanitized
    assert "placebo validation metric: NOT_IMPLEMENTED" in sanitized
    assert "identical executed-trade counts across multiple deciles" in sanitized
    assert "VALIDATION_PASSED" not in sanitized


def test_summary_adds_workflow_readiness_lines_and_blocks_final_validation() -> None:
    sanitized = research._sanitize_validation_summary_text("STOCK-ALPHA NEWS RISK OVERLAY\n")

    assert "decile trade reconciliation: PASSED" in sanitized
    assert "validation readiness: DEVELOPMENT_ONLY / NOT_FINAL_VALIDATION | FinBERT: NOT_READY | gaps: OPEN" in sanitized
    assert "paper/live trading enabled: False / False" in sanitized
    assert "walk-forward positive folds" not in sanitized
    assert "placebo p-value" not in sanitized


def test_expanded_workflow_readiness_lines_are_available_for_detailed_output() -> None:
    lines = research._expanded_workflow_readiness_summary_lines()

    assert "- workflow map: PRESENT" in lines
    assert "- validation dependency graph: BLOCKED" in lines
    assert "- validation readiness: DEVELOPMENT_ONLY / NOT_FINAL_VALIDATION" in lines
    assert "- gap analysis: OPEN_GAPS_BLOCK_FINAL_VALIDATION" in lines
    assert "- FinBERT readiness: NOT_READY" in lines
    assert "- paper/live trading enabled: False / False" in lines


def test_summary_keeps_genuine_decile_warning_and_does_not_claim_passed() -> None:
    sanitized = research._sanitize_validation_summary_text(
        "- decile_join_audit.status: FAILED\n- warning: identical executed-trade counts across multiple deciles"
    )

    assert "identical executed-trade counts across multiple deciles" in sanitized
    assert "decile trade reconciliation: PASSED" not in sanitized


def test_summary_removes_stale_decile_warning_when_audit_passes() -> None:
    sanitized = research._sanitize_validation_summary_text(
        "- decile trade reconciliation: PASSED\n- warning: identical executed-trade counts across multiple deciles"
    )

    assert "decile trade reconciliation: PASSED" in sanitized
    assert "identical executed-trade counts across multiple deciles" not in sanitized


def test_validation_readiness_is_inserted_before_warnings_block() -> None:
    sanitized = research._sanitize_validation_summary_text("SUMMARY\nWARNINGS\n- mostly uncategorized events")
    lines = sanitized.splitlines()
    readiness_index = next(index for index, line in enumerate(lines) if "validation readiness:" in line)
    warnings_index = next(index for index, line in enumerate(lines) if line == "WARNINGS")

    assert readiness_index < warnings_index
    assert "mostly uncategorized events" in sanitized


def test_report_warning_text_cleanup_removes_malformed_substrings() -> None:
    cleaned = research._clean_report_text(
        "untoucheddata isNOT_READY audithas not ademonstrably not anuntouched"
    )

    for malformed in ("untoucheddata", "isNOT_READY", "audithas", "not ademonstrably", "not anuntouched"):
        assert malformed not in cleaned
    assert "untouched data" in cleaned
    assert "is NOT_READY" in cleaned
    assert "audit has" in cleaned
    assert "not a demonstrably" in cleaned
    assert "not an untouched" in cleaned


def test_decile_summary_reports_passed_reconciliation_without_stale_identical_count_warning() -> None:
    lines = research._decile_reconciliation_summary_lines(
        {
            "status": "PASSED",
            "matched_trade_rows": 5747,
            "eligible_trade_rows": 5747,
            "trades_assigned_to_multiple_deciles": 0,
            "deciles_receiving_full_ledger_count": 0,
            "identical_decile_metric_diagnostic": {"matched_executed_trade_count": False},
            "warnings": [],
        },
        {"status": "PASSED", "warnings": []},
    )
    text = "\n".join(lines)

    assert "decile trade reconciliation: PASSED" in text
    assert "identical executed-trade counts across multiple deciles" not in text
    assert "matched_trade_rows / eligible_trade_rows: 5747 / 5747" in text


def test_decile_summary_keeps_identical_count_warning_when_audit_does_not_clear_it() -> None:
    lines = research._decile_reconciliation_summary_lines(
        {
            "status": "PASSED_WITH_WARNINGS",
            "matched_trade_rows": 10,
            "eligible_trade_rows": 10,
            "trades_assigned_to_multiple_deciles": 0,
            "deciles_receiving_full_ledger_count": 0,
            "identical_decile_metric_diagnostic": {"matched_executed_trade_count": True},
            "warnings": [],
        },
        {"status": "PASSED_WITH_WARNINGS", "warnings": []},
    )

    assert "warning: identical executed-trade counts across multiple deciles" in "\n".join(lines)


def test_missing_news_bias_separates_covered_and_uncovered_candidates() -> None:
    rows = [
        {
            "decision_timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "AAPL",
            "score": "0.8",
            "news_coverage_status": "COVERED",
            "actual_forward_return_10d": "0.02",
            "price_plus_news_risk_probability": "0.4",
        },
        {
            "decision_timestamp": "2024-01-02T00:00:00+00:00",
            "symbol": "MSFT",
            "score": "0.2",
            "news_coverage_status": "MISSING",
            "news_missing_coverage": "1",
            "actual_forward_return_10d": "-0.03",
        },
    ]

    report, table = _missing_news_bias(rows, "score")

    assert report["covered_candidate_count"] == 1
    assert report["uncovered_candidate_count"] == 1
    assert {row["coverage_group"] for row in table} == {"covered", "uncovered"}


def test_text_model_readiness_defers_transformer_training() -> None:
    report = _text_model_readiness([{"headline": "Earnings warning", "symbol": "AAPL"}])

    assert report["ready_for_text_model"] is True
    assert report["transformer_trained"] is False
    assert "deferred" in report["blocked_reason"]


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


def test_parallel_report_semantics_distinguish_disabled_and_equivalent_modes() -> None:
    disabled = research._parallel_report_skeleton(_parallel_config({"news_risk_parallel_enabled": False}))
    parallel = research._parallel_report_skeleton(
        _parallel_config(
            {
                "news_risk_parallel_enabled": True,
                "news_risk_max_workers": 2,
            }
        )
    )
    parallel["phases_parallelised"] = ["bar_loading"]

    assert disabled["worker_count_semantics"] == "unused because parallel mode is disabled"
    assert _parallel_determinism_status(disabled) == "NOT_ENABLED"
    assert _parallel_determinism_status(parallel) == "DETERMINISTIC_EQUIVALENCE_PASSED"


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
    assert "PSEUDO_HOLDOUT" in text
    assert "NOT_FINAL_VALIDATION" in text
    assert "NOT_IMPLEMENTED" in text
    assert "FinBERT" in text
    assert "NOT_READY" in text
    assert "news transformer scaffold: PRESENT / DISABLED" in text
    assert "news transformer readiness: NOT_READY" in text
    assert "news transformer training/inference enabled: False / False" in text
    assert "used in strategy/replay: False / False" in text
    assert "paper/live trading enabled: False / False" in text
    assert "lowest max drawdown" not in text.lower()


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


def test_grid_selection_artifact_uses_helper_and_selection_cost_fields() -> None:
    rows = [_candidate("2024-01-01", "AAA", 1.0, "ALLOW")]
    replay = {
        "risk_metrics": {
            "price_only": {"total_return_decimal": 0.10, "Sharpe_ratio": 1.0, "Calmar_ratio": 0.5},
            "news_contrarian_rerank": {"total_return_decimal": 0.20, "Sharpe_ratio": 1.2, "Calmar_ratio": 0.8},
        }
    }
    periods = {"periods": {"parameter_validation": {"start_date": "2024-01-01", "end_date": "2024-01-01"}}}

    grid, _folds, selection = research._contrarian_grid_reports(
        rows,
        replay,
        "score",
        periods,
        {
            "stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 10.0,
            "stock_alpha_news_risk_overlay_contrarian_weight": 0.25,
        },
    )

    assert selection["selected_config"] == research._select_contrarian_grid_configuration(grid)
    assert selection["selected_configuration_id"] == selection["selected_config"]["configuration_id"]
    assert selection["selection_round_trip_cost_bps"] == 10.0
    assert selection["selected_config_metric_at_selection_cost"] == selection["selected_config"]["median_validation_calmar"]
    assert selection["used_holdout_for_selection"] is False
    assert selection["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert selection["validation_passed"] is False


def test_frozen_config_artifact_hash_matches_helper_and_keeps_trading_disabled() -> None:
    periods = {
        "periods": {
            "development": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            "parameter_validation": {"start_date": "2024-01-02", "end_date": "2024-01-02"},
        }
    }
    selection = {
        "selection_policy": "highest median validation Calmar among eligible chronological folds",
        "experiment_registry_metadata": {"status": "DEVELOPMENT_ONLY"},
        "selected_configuration": {
            "configuration_id": "w0.25_raw_probability_no_adjustment",
            "news_transformation": "raw_probability",
            "contrarian_weight": 0.25,
            "missing_news_treatment": "no_adjustment",
            "median_validation_calmar": 0.9,
        },
    }

    frozen = research._frozen_contrarian_config(
        selection,
        {"stock_alpha_news_risk_overlay_selection_round_trip_cost_bps": 10.0},
        periods,
    )

    assert frozen["immutable_configuration_hash"] == research._frozen_config_hash(frozen)
    assert frozen["production_signal"] is False
    assert frozen["paper_trading_enabled"] is False
    assert frozen["live_trading_enabled"] is False
    assert frozen["news_originated_entries_enabled"] is False
    assert frozen["used_holdout_for_selection"] is False
    assert frozen["eligibility_constraints"]["status"] == "SCAFFOLD_CURRENT_FORMULA_ONLY"


def test_artifact_validation_report_is_artifact_presence_only(tmp_path: Path) -> None:
    class Paths:
        pass

    paths = Paths()
    required = (
        "chronological_split_manifest_json_path",
        "experiment_registry_jsonl_path",
        "contrarian_grid_selection_json_path",
        "contrarian_frozen_config_json_path",
        "contrarian_holdout_report_json_path",
        "contrarian_chronological_validation_plan_json_path",
        "contrarian_chronological_periods_csv_path",
        "contrarian_walk_forward_validation_report_json_path",
        "contrarian_placebo_permutation_report_json_path",
        "contrarian_placebo_permutation_results_csv_path",
        "contrarian_matched_control_report_json_path",
        "contrarian_matched_control_results_csv_path",
        "contrarian_profit_concentration_report_json_path",
        "contrarian_trade_fragility_by_symbol_csv_path",
        "contrarian_trade_fragility_by_year_csv_path",
        "contrarian_top_trade_removal_csv_path",
        "contrarian_year_regime_report_json_path",
        "contrarian_year_regime_results_csv_path",
        "contrarian_year_regime_examples_csv_path",
        "contrarian_symbol_year_ablation_report_json_path",
        "contrarian_without_top_symbols_csv_path",
        "contrarian_without_top_years_csv_path",
        "contrarian_cost_slippage_robustness_report_json_path",
        "contrarian_cost_slippage_robustness_csv_path",
        "contrarian_data_validity_audit_json_path",
        "intraday_5min_expansion_plan_json_path",
        "text_model_readiness_json_path",
        "validation_stage_placeholders_json_path",
        "decile_join_audit_json_path",
        "decile_trade_reconciliation_json_path",
        "corrected_news_score_deciles_csv_path",
        "news_validation_workflow_map_json_path",
        "validation_dependency_graph_json_path",
        "validation_readiness_dashboard_json_path",
        "artifact_lineage_report_json_path",
        "news_validation_gap_analysis_json_path",
        "news_transformer_readiness_json_path",
        "news_transformer_training_plan_json_path",
        "catastrophic_news_audit_json_path",
        "catastrophic_news_candidates_csv_path",
        "catastrophic_news_veto_report_json_path",
        "catastrophic_veto_candidate_attribution_json_path",
        "catastrophic_veto_trade_attribution_csv_path",
        "catastrophic_veto_strategy_comparison_json_path",
        "catastrophic_veto_policy_json_path",
        "catastrophic_veto_filtered_strategy_report_json_path",
        "catastrophic_veto_removed_trades_csv_path",
        "catastrophic_veto_removed_symbols_csv_path",
        "catastrophic_veto_full_replay_report_json_path",
        "catastrophic_veto_full_replay_trade_ledger_csv_path",
        "catastrophic_veto_full_replay_equity_csv_path",
        "catastrophic_veto_filtered_candidates_csv_path",
        "catastrophic_veto_blocked_candidates_csv_path",
        "catastrophic_veto_replay_seam_report_json_path",
        "catastrophic_veto_bounceback_report_json_path",
        "catastrophic_veto_bounceback_by_category_csv_path",
        "catastrophic_veto_bounceback_examples_csv_path",
        "catastrophic_veto_extreme_only_policy_proposal_json_path",
        "catastrophic_veto_policy_variant_comparison_json_path",
        "catastrophic_veto_policy_variant_counts_csv_path",
        "catastrophic_veto_policy_variant_metrics_csv_path",
        "catastrophic_veto_policy_variant_removed_trades_csv_path",
        "catastrophic_veto_policy_variant_bounceback_csv_path",
        "catastrophic_veto_policy_frontier_report_json_path",
        "catastrophic_veto_policy_frontier_csv_path",
        "catastrophic_veto_policy_variant_examples_csv_path",
        "catastrophic_veto_loser_bounceback_casebook_json_path",
        "catastrophic_veto_loser_bounceback_cases_csv_path",
        "catastrophic_veto_loser_bounceback_feature_diff_csv_path",
        "catastrophic_veto_loser_bounceback_keyword_diff_csv_path",
        "catastrophic_veto_taxonomy_improvement_plan_json_path",
        "catastrophic_veto_parked_status_json_path",
        "catastrophic_news_evidence_quality_report_json_path",
        "catastrophic_news_evidence_quality_by_field_csv_path",
        "catastrophic_news_evidence_quality_by_symbol_csv_path",
        "catastrophic_veto_policy_mode_comparison_json_path",
        "catastrophic_veto_policy_mode_counts_csv_path",
        "news_evidence_lineage_report_json_path",
        "news_evidence_lineage_by_stage_csv_path",
        "news_evidence_missing_field_examples_csv_path",
        "news_evidence_readiness_report_json_path",
        "news_event_taxonomy_report_json_path",
        "news_event_taxonomy_counts_csv_path",
        "news_event_taxonomy_examples_csv_path",
        "news_duplicate_grouping_report_json_path",
        "news_duplicate_grouping_examples_csv_path",
        "news_point_in_time_text_safety_report_json_path",
        "news_point_in_time_text_safety_examples_csv_path",
        "news_text_keyword_baseline_report_json_path",
        "news_text_keyword_baseline_scores_csv_path",
        "walk_forward_validation_report_json_path",
        "walk_forward_fold_results_csv_path",
        "placebo_permutation_report_json_path",
        "placebo_permutation_results_csv_path",
        "exposure_matched_controls_json_path",
        "trade_count_matched_controls_json_path",
        "concentration_fragility_report_json_path",
    )
    for name in required:
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        setattr(paths, name, path)

    report = research._artifact_validation_report(paths)

    assert report["status"] == "PASSED"
    assert report["status_scope"] == "ARTIFACT_PRESENCE_ONLY"
    assert report["artifact_presence_status"] == "PASSED"
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert report["validation_passed"] is False
    assert report["is_final_validation"] is False
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False
    artifact_keys = {row["artifact_key"] for row in report["artifacts"]}
    assert "news_validation_workflow_map_json_path" in artifact_keys
    assert "validation_dependency_graph_json_path" in artifact_keys
    assert "validation_readiness_dashboard_json_path" in artifact_keys
    assert "artifact_lineage_report_json_path" in artifact_keys
    assert "news_validation_gap_analysis_json_path" in artifact_keys
    assert "news_transformer_readiness_json_path" in artifact_keys
    assert "news_transformer_training_plan_json_path" in artifact_keys
    assert "catastrophic_news_audit_json_path" in artifact_keys
    assert "catastrophic_news_candidates_csv_path" in artifact_keys
    assert "catastrophic_news_veto_report_json_path" in artifact_keys
    assert "catastrophic_veto_candidate_attribution_json_path" in artifact_keys
    assert "catastrophic_veto_trade_attribution_csv_path" in artifact_keys
    assert "catastrophic_veto_strategy_comparison_json_path" in artifact_keys
    assert "catastrophic_veto_policy_json_path" in artifact_keys
    assert "catastrophic_veto_filtered_strategy_report_json_path" in artifact_keys
    assert "catastrophic_veto_removed_trades_csv_path" in artifact_keys
    assert "catastrophic_veto_removed_symbols_csv_path" in artifact_keys
    assert "catastrophic_veto_full_replay_report_json_path" in artifact_keys
    assert "catastrophic_veto_full_replay_trade_ledger_csv_path" in artifact_keys
    assert "catastrophic_veto_full_replay_equity_csv_path" in artifact_keys
    assert "catastrophic_veto_filtered_candidates_csv_path" in artifact_keys
    assert "catastrophic_veto_blocked_candidates_csv_path" in artifact_keys
    assert "catastrophic_veto_replay_seam_report_json_path" in artifact_keys
    assert "catastrophic_veto_bounceback_report_json_path" in artifact_keys
    assert "catastrophic_veto_extreme_only_policy_proposal_json_path" in artifact_keys
    assert "catastrophic_veto_policy_variant_comparison_json_path" in artifact_keys
    assert "catastrophic_veto_policy_frontier_report_json_path" in artifact_keys
    assert "catastrophic_veto_loser_bounceback_casebook_json_path" in artifact_keys
    assert "catastrophic_veto_taxonomy_improvement_plan_json_path" in artifact_keys
    assert "news_event_taxonomy_report_json_path" in artifact_keys
    assert "news_duplicate_grouping_report_json_path" in artifact_keys
    assert "news_point_in_time_text_safety_report_json_path" in artifact_keys
    assert "news_text_keyword_baseline_report_json_path" in artifact_keys
    assert "walk_forward_validation_report_json_path" in artifact_keys
    assert "placebo_permutation_report_json_path" in artifact_keys
    assert "exposure_matched_controls_json_path" in artifact_keys


def test_artifact_validation_report_marks_missing_workflow_artifact(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._artifact_validation_report(paths)

    assert report["artifact_presence_status"] == "FAILED"
    assert report["status_scope"] == "ARTIFACT_PRESENCE_ONLY"
    assert "news_validation_workflow_map_json_path" in report["missing_artifacts"]
    assert report["validation_passed"] is False


def test_workflow_map_marks_future_validation_and_text_models_blocking(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._news_validation_workflow_map(paths)
    nodes = {node["node_id"]: node for node in report["nodes"]}

    assert report["research_only"] is True
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert report["validation_passed"] is False
    assert "price_model_candidates" in nodes
    assert "future_walk_forward" in nodes
    assert "future_text_models" in nodes
    assert nodes["future_walk_forward"]["status"] == "NOT_IMPLEMENTED"
    assert nodes["future_walk_forward"]["blocks_final_validation"] is True
    assert nodes["text_model_readiness"]["status"] == "NOT_READY"
    assert nodes["text_model_readiness"]["blocks_final_validation"] is True
    assert "event_taxonomy_research" in nodes
    assert "duplicate_grouping_heuristic" in nodes
    assert "point_in_time_text_safety" in nodes
    assert "keyword_text_baseline" in nodes
    assert "catastrophic_veto_bounceback" in nodes
    assert "extreme_only_policy_proposal" in nodes
    assert nodes["validation_stage_placeholders"]["contains_blocking_stages"] is True
    assert nodes["validation_stage_placeholders"]["blocking_stage_count"] > 0
    assert "Validation stage placeholder container includes incomplete gates that block final validation." in nodes["validation_stage_placeholders"]["warnings"]
    assert any(edge["from"] == "frozen_config" and edge["to"] == "pseudo_holdout_report" for edge in report["edges"])


def test_validation_dependency_graph_blocks_final_validation(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._validation_dependency_graph(paths)
    gates = {gate["gate_name"]: gate for gate in report["gates"]}

    assert report["all_required_gates_complete"] is False
    assert report["all_required_gates_passed"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert report["validation_passed"] is False
    assert report["is_final_validation"] is False
    assert gates["pseudo_holdout_gate"]["passed"] is False
    assert gates["pseudo_holdout_gate"]["blocks_final_validation"] is True
    assert gates["walk_forward"]["implemented"] is True
    assert gates["walk_forward"]["blocks_final_validation"] is True
    assert gates["walk_forward"]["passed"] is False
    assert gates["text_model_readiness"]["status"] == "NOT_READY"
    assert gates["text_model_readiness"]["passed"] is False
    assert "event_taxonomy_research" in gates
    assert "duplicate_grouping_heuristic" in gates
    assert "point_in_time_text_safety" in gates
    assert "keyword_text_baseline" in gates
    assert "catastrophic_veto_bounceback" in gates
    assert "extreme_only_policy_proposal" in gates
    assert "pseudo_holdout_gate" in report["blocked_by"]


def test_validation_readiness_dashboard_blocks_unsafe_next_steps(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._validation_readiness_dashboard(paths)

    assert report["overall_status"] == "DEVELOPMENT_ONLY"
    assert report["research_only"] is True
    assert report["production_signal"] is False
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False
    assert report["validation_label"] == "PSEUDO_HOLDOUT"
    assert report["holdout_type"] == "PSEUDO_HOLDOUT"
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert report["validation_passed"] is False
    assert report["finbert_readiness"] == "NOT_READY"
    assert report["text_model_ready"] is False
    assert report["transformer_ready"] is False
    assert report["catastrophic_veto_bounceback_status"] == "UNAVAILABLE_INPUT"
    assert report["extreme_only_policy_proposal_status"] == "UNAVAILABLE_INPUT"
    assert "walk-forward not implemented" in report["top_blockers"]
    assert "FinBERT" in report["unsafe_next_steps"]
    assert "implement walk-forward validation" in report["safe_next_steps"]


def test_artifact_lineage_report_declares_required_relationships(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._artifact_lineage_report(paths)
    pairs = {(row["source_artifact"], row["target_artifact"]) for row in report["lineage"]}

    assert report["status"] == "DECLARED"
    assert report["research_only"] is True
    assert report["validation_passed"] is False
    assert ("decile_join.json", "validation_readiness_dashboard.json") in pairs
    assert ("decile_reconciliation.json", "validation_readiness_dashboard.json") in pairs
    assert ("split.json", "grid_selection.json") in pairs
    assert ("frozen.json", "holdout.json") in pairs
    assert ("text.json", "dependency.json") in pairs
    assert ("news_evidence_readiness_report.json", "news_event_taxonomy_report.json") in pairs
    assert ("news_text_keyword_baseline_report.json", "validation_readiness_dashboard.json") in pairs
    assert ("catastrophic_veto_removed_trades.csv", "catastrophic_veto_bounceback_report.json") in pairs
    assert ("catastrophic_veto_bounceback_report.json", "catastrophic_veto_extreme_only_policy_proposal.json") in pairs


def test_artifact_manifest_includes_workflow_readiness_artifacts(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._research_artifact_manifest(paths)
    artifacts = report["artifacts"]

    assert "news_validation_workflow_map_json_path" in artifacts
    assert "validation_dependency_graph_json_path" in artifacts
    assert "validation_readiness_dashboard_json_path" in artifacts
    assert "news_event_taxonomy_report_json_path" in artifacts
    assert "news_duplicate_grouping_report_json_path" in artifacts
    assert "news_point_in_time_text_safety_report_json_path" in artifacts
    assert "news_text_keyword_baseline_report_json_path" in artifacts
    assert "catastrophic_veto_bounceback_report_json_path" in artifacts
    assert "catastrophic_veto_extreme_only_policy_proposal_json_path" in artifacts
    assert "catastrophic_veto_policy_variant_comparison_json_path" in artifacts
    assert "catastrophic_veto_policy_frontier_report_json_path" in artifacts
    assert "catastrophic_veto_loser_bounceback_casebook_json_path" in artifacts
    assert "catastrophic_veto_taxonomy_improvement_plan_json_path" in artifacts
    assert "artifact_lineage_report_json_path" in artifacts
    assert "news_validation_gap_analysis_json_path" in artifacts
    assert "news_transformer_readiness_json_path" in artifacts
    assert "news_transformer_training_plan_json_path" in artifacts
    assert "catastrophic_news_audit_json_path" in artifacts
    assert "catastrophic_news_candidates_csv_path" in artifacts
    assert "catastrophic_news_veto_report_json_path" in artifacts
    assert "catastrophic_veto_candidate_attribution_json_path" in artifacts
    assert "catastrophic_veto_trade_attribution_csv_path" in artifacts
    assert "catastrophic_veto_strategy_comparison_json_path" in artifacts
    assert "catastrophic_veto_policy_json_path" in artifacts
    assert "catastrophic_veto_filtered_strategy_report_json_path" in artifacts
    assert "catastrophic_veto_removed_trades_csv_path" in artifacts
    assert "catastrophic_veto_removed_symbols_csv_path" in artifacts
    assert "catastrophic_veto_full_replay_report_json_path" in artifacts
    assert "catastrophic_veto_full_replay_trade_ledger_csv_path" in artifacts
    assert "catastrophic_veto_full_replay_equity_csv_path" in artifacts
    assert "catastrophic_veto_filtered_candidates_csv_path" in artifacts
    assert "catastrophic_veto_blocked_candidates_csv_path" in artifacts
    assert "catastrophic_veto_replay_seam_report_json_path" in artifacts
    assert "walk_forward_validation_report_json_path" in artifacts
    assert "placebo_permutation_report_json_path" in artifacts
    assert "concentration_fragility_report_json_path" in artifacts
    assert report["research_only"] is True
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False


def test_gap_analysis_lists_critical_blockers_and_blocks_paper_live(tmp_path: Path) -> None:
    paths = _minimal_research_paths(tmp_path)

    report = research._news_validation_gap_analysis(paths)
    gap_ids = {gap["gap_id"] for gap in report["gaps"]}

    assert report["status"] == "OPEN_GAPS_BLOCK_FINAL_VALIDATION"
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert report["validation_passed"] is False
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False
    assert "walk_forward_not_implemented" in gap_ids
    assert "pseudo_holdout_not_genuine" in gap_ids
    assert "text_model_readiness_not_ready" in gap_ids
    assert "catastrophic_news_veto_not_validated" in gap_ids
    assert "catastrophic_veto_full_replay_not_computed" in gap_ids
    assert "catastrophic_veto_full_replay_not_available" in gap_ids
    assert "validation spine not complete" in report["finbert_blockers"]
    assert "final validation status is NOT_FINAL_VALIDATION" in report["paper_live_blockers"]


def test_catastrophic_news_taxonomy_contains_blocking_bankruptcy_category() -> None:
    taxonomy = catastrophic_news_taxonomy_report()
    categories = {category["category_id"]: category for category in taxonomy["categories"]}

    category = categories["bankruptcy_or_administration"]

    assert taxonomy["status"] == "PRESENT"
    assert category["severity"] == "CATASTROPHIC"
    assert category["blocks_contrarian_entry"] is True
    assert category["requires_manual_review"] is True
    assert "bankruptcy" in category["keywords"]


def test_catastrophic_news_classifier_blocks_bankruptcy_with_point_in_time_timestamp() -> None:
    result = classify_catastrophic_news_event(
        headline="Issuer files for bankruptcy protection",
        summary="Trading remains volatile after Chapter 11 filing.",
        publication_timestamp="2026-01-02T08:00:00Z",
        availability_timestamp="2026-01-02T08:05:00Z",
        symbol="XYZ",
        candidate_id="XYZ-2026-01-02",
    )

    assert result["matched"] is True
    assert "bankruptcy_or_administration" in result["matched_categories"]
    assert result["highest_severity"] == "CATASTROPHIC"
    assert result["blocks_contrarian_entry"] is True
    assert result["requires_manual_review"] is True
    assert result["point_in_time_safe"] is True
    assert result["classification_method"] == "DETERMINISTIC_TAXONOMY"


def test_catastrophic_news_classifier_missing_text_is_unknown_and_not_safe() -> None:
    result = classify_catastrophic_news_event(
        symbol="XYZ",
        candidate_id="XYZ-2026-01-02",
    )

    assert result["matched"] is False
    assert result["highest_severity"] == "UNKNOWN"
    assert result["classification_method"] == "UNAVAILABLE_INPUT"
    assert result["requires_manual_review"] is True
    assert result["point_in_time_safe"] is False
    assert any("UNAVAILABLE_INPUT" in warning for warning in result["warnings"])


def test_catastrophic_news_audit_and_veto_are_research_only_guardrails() -> None:
    rows = [
        {
            "candidate_id": "XYZ-2026-01-02",
            "symbol": "XYZ",
            "headline": "XYZ files for bankruptcy protection",
            "publication_timestamp": "2026-01-02T08:00:00Z",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        }
    ]

    audit, candidates, veto = research._catastrophic_news_artifacts(rows)

    assert audit["status"] == "PASSED_WITH_WARNINGS"
    assert audit["blocked_candidate_count"] == 1
    assert audit["manual_review_candidate_count"] == 1
    assert audit["categories"]
    assert audit["validation_label"] == "PSEUDO_HOLDOUT"
    assert audit["validation_passed"] is False
    assert audit["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert audit["paper_trading_enabled"] is False
    assert audit["live_trading_enabled"] is False
    assert "bankruptcy" in candidates[0]["matched_terms"]
    assert candidates[0]["research_only_veto_would_apply"] is True
    assert veto["veto_enabled_in_strategy"] is False
    assert veto["used_in_replay"] is False
    assert veto["training_enabled"] is False
    assert veto["inference_enabled"] is False
    assert veto["would_block_candidate_count"] == 1


def test_catastrophic_veto_policy_disables_paper_live_and_blocks_unknowns() -> None:
    policy = research._catastrophic_veto_policy()

    assert policy["enabled_for_research"] is True
    assert policy["policy_stage"] == "RESEARCH_ONLY"
    assert policy["enforcement_stage"] == "AUDIT_OR_RESEARCH_SIMULATION_ONLY"
    assert policy["catastrophic_veto_policy_mode"] == "STRICT_SAFETY"
    assert policy["allowed_policy_modes"] == [
        "STRICT_SAFETY",
        "CONFIRMED_ONLY_RESEARCH",
        "MANUAL_REVIEW_RESEARCH",
    ]
    assert policy["enabled_for_paper_trading"] is False
    assert policy["enabled_for_live_trading"] is False
    assert policy["paper_trading_allowed"] is False
    assert policy["live_trading_allowed"] is False
    assert policy["manual_review_required_before_any_execution"] is True
    assert policy["unknown_text_default"] == "DO_NOT_TREAT_AS_SAFE"
    assert policy["missing_availability_timestamp_default"] == "NOT_POINT_IN_TIME_SAFE"
    assert policy["default_action_for_catastrophic"] == "BLOCK_CONTRARIAN_ENTRY"
    assert policy["default_action_for_manual_review"] == "BLOCK_UNTIL_REVIEWED"
    assert policy["default_action_for_unknown"] == "DO_NOT_TREAT_AS_SAFE"
    assert policy["point_in_time_requirements"]["availability_timestamp_required"] is True
    assert policy["validation_passed"] is False
    assert policy["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_apply_catastrophic_veto_to_candidates_is_deterministic_and_non_mutating() -> None:
    rows = [
        {
            "candidate_id": "SAFE-1",
            "symbol": "SAFE",
            "headline": "ordinary earnings update",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
        {
            "candidate_id": "BAD-1",
            "symbol": "BAD",
            "headline": "BAD files for bankruptcy protection",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
        {
            "candidate_id": "UNKNOWN-1",
            "symbol": "UNK",
            "headline": "",
        },
        {
            "candidate_id": "NOASOF-1",
            "symbol": "NOASOF",
            "headline": "ordinary operations update",
        },
    ]
    original = [dict(row) for row in rows]

    result = research.apply_catastrophic_veto_to_candidates(rows)

    assert rows == original
    assert len(result["filtered_candidates"]) == 1
    assert result["filtered_candidates"][0]["candidate_id"] == "SAFE-1"
    assert len(result["blocked_candidates"]) == 3
    assert len(result["manual_review_candidates"]) == 1
    assert len(result["unknown_candidates"]) == 1
    assert len(result["confirmed_catastrophic_candidates"]) == 1
    assert len(result["unknown_text_candidates"]) == 1
    assert len(result["missing_availability_candidates"]) == 2
    assert result["filter_audit"]["strict_policy_blocked_candidate_count"] == 3
    assert result["filter_audit"]["confirmed_catastrophic_blocked_candidate_count"] == 1
    assert result["filter_audit"]["catastrophic_veto_policy_mode"] == "STRICT_SAFETY"
    assert result["filter_audit"]["base_candidates_mutated"] is False
    assert result["filter_audit"]["validation_passed"] is False


def test_catastrophic_veto_policy_modes_block_expected_candidate_sets() -> None:
    rows = [
        {
            "candidate_id": "SAFE-1",
            "symbol": "SAFE",
            "headline": "ordinary product update",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
        {
            "candidate_id": "BAD-1",
            "symbol": "BAD",
            "headline": "BAD files for bankruptcy protection",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
        {
            "candidate_id": "UNKNOWN-1",
            "symbol": "UNK",
            "headline": "",
        },
        {
            "candidate_id": "NOASOF-1",
            "symbol": "NOASOF",
            "headline": "ordinary operations update",
        },
    ]

    strict = research.apply_catastrophic_veto_to_candidates(rows)
    confirmed = research.apply_catastrophic_veto_to_candidates(rows, policy_mode="CONFIRMED_ONLY_RESEARCH")
    manual = research.apply_catastrophic_veto_to_candidates(rows, policy_mode="MANUAL_REVIEW_RESEARCH")

    assert {row["candidate_id"] for row in strict["blocked_candidates"]} == {"BAD-1", "UNKNOWN-1", "NOASOF-1"}
    assert {row["candidate_id"] for row in confirmed["blocked_candidates"]} == {"BAD-1"}
    assert {row["candidate_id"] for row in manual["blocked_candidates"]} == {"BAD-1"}
    assert confirmed["filter_audit"]["unknown_text_candidate_count"] == 1
    assert confirmed["filter_audit"]["missing_availability_candidate_count"] == 2
    assert confirmed["filter_audit"]["catastrophic_veto_policy_mode"] == "CONFIRMED_ONLY_RESEARCH"
    assert manual["filter_audit"]["catastrophic_veto_policy_mode"] == "MANUAL_REVIEW_RESEARCH"
    assert confirmed["filter_audit"]["validation_passed"] is False


def test_research_strategy_variant_input_seam_is_opt_in_and_copy_only() -> None:
    rows = [{"candidate_id": "A", "symbol": "AAA"}]
    spec = research.ResearchStrategyVariantSpec(
        base_variant_name="news_contrarian_rerank",
        new_variant_name="news_contrarian_rerank_catastrophic_veto",
    )

    output = research.build_research_strategy_variant_inputs(rows, spec)

    assert output["base_variant_name"] == "news_contrarian_rerank"
    assert output["new_variant_name"] == "news_contrarian_rerank_catastrophic_veto"
    assert output["filter_enabled"] is False
    assert output["default_behavior_unchanged"] is True
    assert output["paper_trading_enabled"] is False
    assert output["live_trading_enabled"] is False
    assert output["candidate_rows"] == rows
    assert output["candidate_rows"] is not rows
    output["candidate_rows"][0]["symbol"] = "CHANGED"
    assert rows[0]["symbol"] == "AAA"


def test_catastrophic_evidence_quality_and_policy_modes_are_research_only() -> None:
    rows = [
        {"candidate_id": "SAFE", "symbol": "AAA", "headline": "ordinary update", "availability_timestamp": "2026-01-02T08:00:00Z"},
        {"candidate_id": "BAD", "symbol": "BBB", "headline": "files for bankruptcy", "availability_timestamp": "2026-01-02T08:00:00Z"},
        {"candidate_id": "UNKNOWN", "symbol": "CCC", "headline": ""},
        {"candidate_id": "NOASOF", "symbol": "DDD", "headline": "ordinary update"},
    ]
    replay = {
        "trade_ledger": [
            {"candidate_id": row["candidate_id"], "strategy_variant": "news_contrarian_rerank"}
            for row in rows
        ]
    }
    full_report = {
        "full_replay_computed": True,
        "replay_impact_status": "FULL_REPLAY_COMPUTED_ZERO_CANDIDATES",
        "veto_metrics_status": "UNAVAILABLE_EMPTY_CANDIDATE_SET",
    }

    report, by_field, by_symbol, comparison, counts = research._catastrophic_news_evidence_quality_artifacts(
        rows,
        replay,
        full_report,
    )
    modes = {row["policy_mode"]: row for row in counts}

    assert report["candidate_count"] == 4
    assert report["confirmed_catastrophic_candidate_count"] == 1
    assert report["missing_text_count"] == 1
    assert report["missing_availability_timestamp_count"] == 2
    assert report["strict_policy_blocked_candidate_count"] == 3
    assert modes["STRICT_SAFETY"]["total_blocked_candidate_count"] == 3
    assert modes["CONFIRMED_ONLY_RESEARCH"]["total_blocked_candidate_count"] == 1
    assert modes["MANUAL_REVIEW_RESEARCH"]["total_blocked_candidate_count"] == 1
    assert modes["STRICT_SAFETY"]["full_replay_computed"] is True
    assert modes["CONFIRMED_ONLY_RESEARCH"]["full_replay_status"] == "COUNT_ONLY_NOT_REPLAYED"
    assert all(row["paper_trading_allowed"] is False for row in counts)
    assert all(row["live_trading_allowed"] is False for row in counts)
    assert all(row["validation_passed"] is False for row in counts)
    assert comparison["active_full_replay_policy_mode"] == "STRICT_SAFETY"
    assert by_field
    assert len(by_symbol) == 4
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_research_strategy_variant_input_seam_can_declare_filtered_variant() -> None:
    spec = research.ResearchStrategyVariantSpec(
        base_variant_name="news_contrarian_rerank",
        new_variant_name="news_contrarian_rerank_catastrophic_veto",
        candidate_filter=research.ResearchCandidateFilterSpec(
            filter_name="catastrophic_veto",
            enabled=True,
        ),
    )

    output = research.build_research_strategy_variant_inputs([], spec)

    assert output["filter_name"] == "catastrophic_veto"
    assert output["filter_enabled"] is True
    assert output["default_behavior_unchanged"] is False
    assert output["research_only"] is True


def test_news_evidence_lineage_reports_mapping_gaps_without_mutating_inputs() -> None:
    raw_rows = [{
        "candidate_id": "A",
        "symbol": "AAA",
        "headline": "ordinary update",
        "published_at_utc": "2026-01-02T07:00:00Z",
        "availability_timestamp": "2026-01-02T08:00:00Z",
        "source": "official",
        "provider": "test",
        "event_type": "company_update",
        "duplicate_group_id": "group-1",
    }]
    final_rows = [{"candidate_id": "A", "symbol": "AAA"}]
    original_raw = [dict(row) for row in raw_rows]
    original_final = [dict(row) for row in final_rows]
    stages = {
        "raw_news": {"available": True, "path": "raw.csv", "rows": raw_rows},
        "provider_normalized_news": {"available": False, "path": "UNAVAILABLE_UPSTREAM", "rows": []},
        "news_contract": {"available": False, "path": "UNAVAILABLE_UPSTREAM", "rows": []},
        "news_features": {"available": True, "path": "features.csv", "rows": final_rows},
        "joined_candidate_rows": {"available": True, "path": "IN_MEMORY", "rows": final_rows},
        "catastrophic_veto_input_rows": {"available": True, "path": "IN_MEMORY", "rows": final_rows},
    }

    report, by_stage, examples, readiness = research._news_evidence_lineage_artifacts(stages)
    gaps = {gap["field_name"]: gap for gap in report["field_mapping_gaps"]}
    final_stage = next(row for row in by_stage if row["stage"] == "catastrophic_veto_input_rows")

    assert final_stage["missing_text_count"] == 1
    assert final_stage["missing_availability_timestamp_count"] == 1
    assert gaps["headline_text"]["status"] == "FULLY_MISSING_FROM_STAGE"
    assert gaps["headline_text"]["present_in_stage"] == "raw_news"
    assert gaps["headline_text"]["recommended_fix"]
    assert gaps["availability_timestamp"]["blocks_catastrophic_veto"] is True
    assert readiness["status"] == "INSUFFICIENT"
    assert readiness["strict_veto_ready"] is False
    assert readiness["text_model_ready"] is False
    assert readiness["finbert_readiness"] == "NOT_READY"
    assert readiness["transformer_readiness"] == "NOT_READY"
    assert readiness["paper_trading_enabled"] is False
    assert readiness["live_trading_enabled"] is False
    assert readiness["validation_passed"] is False
    assert readiness["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert examples
    assert raw_rows == original_raw
    assert final_rows == original_final


def test_news_evidence_lineage_clears_propagated_headline_provider_and_availability_gaps() -> None:
    propagated = [{
        "candidate_id": "A",
        "symbol": "AAA",
        "headline_text": "ordinary update",
        "provider": "official_feed",
        "availability_timestamp": "2026-01-02T08:00:00Z",
    }]
    stages = {
        stage: {"available": True, "path": stage, "rows": propagated}
        for stage in (
            "raw_news",
            "provider_normalized_news",
            "news_contract",
            "news_features",
            "joined_candidate_rows",
            "catastrophic_veto_input_rows",
        )
    }

    report, by_stage, _examples, readiness = research._news_evidence_lineage_artifacts(stages)
    gaps = {gap["field_name"]: gap for gap in report["field_mapping_gaps"]}
    final_stage = next(row for row in by_stage if row["stage"] == "catastrophic_veto_input_rows")

    assert final_stage["has_headline_count"] == 1
    assert final_stage["has_provider_count"] == 1
    assert final_stage["has_availability_timestamp_count"] == 1
    assert gaps["headline_text"]["status"] == "PRESENT"
    assert gaps["provider"]["status"] == "PRESENT"
    assert gaps["availability_timestamp"]["status"] == "PRESENT"
    assert gaps["event_category"]["status"] == "UNAVAILABLE_UPSTREAM"
    assert report["field_mapping_gaps"][0]["recommended_fix"]
    assert readiness["strict_veto_ready"] is True
    assert readiness["text_model_ready"] is False


def test_news_evidence_lineage_reports_partial_coverage_without_calling_it_missing() -> None:
    rows = [
        {
            "candidate_id": "A",
            "symbol": "AAA",
            "headline_text": "ordinary update",
            "provider": "official_feed",
            "availability_timestamp": "2026-01-02T08:00:00Z",
        },
        {"candidate_id": "B", "symbol": "BBB"},
    ]
    stages = {
        "raw_news": {"available": True, "path": "raw", "rows": rows},
        "provider_normalized_news": {"available": False, "path": "UNAVAILABLE_UPSTREAM", "rows": []},
        "news_contract": {"available": False, "path": "UNAVAILABLE_UPSTREAM", "rows": []},
        "news_features": {"available": True, "path": "features", "rows": rows},
        "joined_candidate_rows": {"available": True, "path": "joined", "rows": rows},
        "catastrophic_veto_input_rows": {"available": True, "path": "veto", "rows": rows},
    }

    report, _by_stage, _examples, readiness = research._news_evidence_lineage_artifacts(stages)
    gaps = {gap["field_name"]: gap for gap in report["field_mapping_gaps"]}

    for field in ("headline_text", "provider", "availability_timestamp"):
        assert gaps[field]["status"] == "PARTIAL_COVERAGE"
        assert gaps[field]["present_in_stage"] == "catastrophic_veto_input_rows"
        assert gaps[field]["present_count"] == 1
        assert gaps[field]["missing_count"] == 1
        assert gaps[field]["coverage_ratio"] == pytest.approx(0.5)
    assert gaps["event_category"]["status"] == "UNAVAILABLE_UPSTREAM"
    assert gaps["duplicate_group_id"]["status"] == "UNAVAILABLE_UPSTREAM"
    assert gaps["source"]["status"] == "UNAVAILABLE_UPSTREAM"
    assert readiness["strict_veto_ready"] is True
    assert readiness["confirmed_only_veto_ready"] is True


def test_research_event_taxonomy_duplicate_safety_and_keyword_artifacts_are_deterministic() -> None:
    rows = [
        {
            "candidate_id": "A",
            "symbol": "AAA",
            "decision_timestamp": "2026-01-02T09:00:00Z",
            "headline_text": "AAA cuts guidance after earnings miss",
            "availability_timestamp": "2026-01-02T08:00:00Z",
            "publication_timestamp": "2026-01-02T07:50:00Z",
            "provider": "official",
        },
        {
            "candidate_id": "B",
            "symbol": "AAA",
            "decision_timestamp": "2026-01-02T09:00:00Z",
            "headline_text": "AAA cuts guidance after earnings miss",
            "availability_timestamp": "2026-01-02T08:00:00Z",
            "provider": "official",
        },
        {
            "candidate_id": "C",
            "symbol": "CCC",
            "decision_timestamp": "2026-01-02T09:00:00Z",
            "headline_text": "CCC announces ordinary update",
            "publication_timestamp": "2026-01-02T07:50:00Z",
        },
        {
            "candidate_id": "D",
            "symbol": "DDD",
            "decision_timestamp": "2026-01-02T09:00:00Z",
            "headline_text": "DDD faces lawsuit and regulatory investigation",
            "availability_timestamp": "2026-01-02T10:00:00Z",
        },
    ]

    taxonomy_report, taxonomy_counts, taxonomy_examples = research._news_event_taxonomy_artifacts(rows)
    duplicate_report, duplicate_examples = research._news_duplicate_grouping_artifacts(rows)
    safety_report, safety_examples = research._news_point_in_time_text_safety_artifacts(rows)
    keyword_report, keyword_scores = research._news_text_keyword_baseline_artifacts(rows)

    counts = {row["event_category_research"]: row["candidate_count"] for row in taxonomy_counts}
    assert taxonomy_report["status"] == "RESEARCH_RULES_READY"
    assert counts["guidance_cut"] == 2
    assert counts["litigation_or_regulatory"] == 1
    assert counts["uncategorized"] == 1
    assert taxonomy_examples
    assert duplicate_report["status"] == "HEURISTIC_ONLY"
    assert duplicate_report["duplicate_group_count"] == 1
    assert duplicate_report["duplicate_candidate_count"] == 2
    assert duplicate_examples[0]["duplicate_group_id_heuristic"] == duplicate_examples[1]["duplicate_group_id_heuristic"]
    assert safety_report["safe_text_count"] == 2
    assert safety_report["publication_only_count"] == 1
    assert safety_report["availability_after_decision_count"] == 1
    assert any(row["reason"] == "AVAILABILITY_AFTER_DECISION" for row in safety_examples)
    assert keyword_report["status"] == "RESEARCH_ONLY"
    score_by_id = {row["candidate_id"]: row for row in keyword_scores}
    assert score_by_id["A"]["guidance_negative_score"] > 0
    assert score_by_id["D"]["litigation_score"] > 0
    assert keyword_report["finbert_enabled"] is False
    assert keyword_report["transformer_enabled"] is False


def test_catastrophic_bounceback_labels_and_severity_groups_are_deterministic() -> None:
    assert research._bounceback_label({"net_return": "0.12"}) == "BOUNCED_BACK_STRONGLY"
    assert research._bounceback_label({"net_return": "0.04"}) == "BOUNCED_BACK_WEAKLY"
    assert research._bounceback_label({"net_return": "-0.04"}) == "DID_NOT_BOUNCE"
    assert research._bounceback_label({"net_return": "-0.12"}) == "SEVERE_LOSS"
    assert research._bounceback_label({"pnl": "100"}) == "UNAVAILABLE_OUTCOME"

    assert research._severity_group_for_candidate(
        {
            "headline_text": "Company cuts guidance after earnings miss",
            "availability_timestamp": "2026-01-02T08:00:00Z",
        }
    ) == "REVERSIBLE_BAD_NEWS"
    for headline in (
        "Company files for bankruptcy protection",
        "Company defaults on debt",
        "Company receives delisting notice",
    ):
        assert research._severity_group_for_candidate(
            {"headline_text": headline, "availability_timestamp": "2026-01-02T08:00:00Z"}
        ) == "EXTREME_DISTRESS"
    assert research._severity_group_for_candidate(
        {
            "headline_text": "Company discloses fraud and accounting irregularities",
            "availability_timestamp": "2026-01-02T08:00:00Z",
        }
    ) == "EXTREME_DISTRESS_OR_FRAUD"
    assert research._severity_group_for_candidate({"headline_text": "", "availability_timestamp": ""}) == "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"


def test_catastrophic_bounceback_artifacts_separate_winners_losers_and_unavailable_outcomes() -> None:
    rows = [
        {
            "candidate_id": "WIN",
            "symbol": "WIN",
            "headline_text": "WIN cuts guidance after earnings miss",
            "availability_timestamp": "2026-01-02T08:00:00Z",
        },
        {
            "candidate_id": "LOSS",
            "symbol": "LOSS",
            "headline_text": "LOSS files for bankruptcy protection",
            "availability_timestamp": "2026-01-02T08:00:00Z",
        },
        {
            "candidate_id": "UNK",
            "symbol": "UNK",
            "headline_text": "",
        },
    ]
    removed = [
        {"trade_id": "T-WIN", "candidate_id": "WIN", "symbol": "WIN", "net_return": "0.15"},
        {"trade_id": "T-LOSS", "candidate_id": "LOSS", "symbol": "LOSS", "net_return": "-0.20"},
        {"trade_id": "T-UNK", "candidate_id": "UNK", "symbol": "UNK"},
    ]
    replay = {
        "risk_metrics": {
            "news_contrarian_rerank": {"total_return_decimal": 0.50, "maximum_drawdown": -0.30, "Sharpe_ratio": 1.0},
            "news_contrarian_rerank_catastrophic_veto": {"total_return_decimal": 0.30, "maximum_drawdown": -0.20, "Sharpe_ratio": 0.9},
        }
    }
    policy_counts = [
        {"policy_mode": "CONFIRMED_ONLY_RESEARCH", "estimated_removed_trade_count": 1},
        {"policy_mode": "MANUAL_REVIEW_RESEARCH", "estimated_removed_trade_count": 1},
    ]

    report, by_category, examples, proposal = research._catastrophic_veto_bounceback_artifacts(
        rows,
        replay,
        removed,
        rows,
        {"candidate_count_before_veto": 3, "candidate_count_after_veto": 1},
        policy_counts,
    )

    by_group = {(row["event_category_research"], row["severity_group"]): row for row in by_category}
    assert report["status"] == "AVAILABLE"
    assert report["top_removed_winners"][0]["trade_id"] == "T-WIN"
    assert report["top_removed_losers"][0]["trade_id"] == "T-LOSS"
    assert by_group[("guidance_cut", "REVERSIBLE_BAD_NEWS")]["strong_bounceback_count"] == 1
    assert by_group[("catastrophic_or_distress", "EXTREME_DISTRESS")]["severe_loss_count"] == 1
    assert by_group[("uncategorized", "UNKNOWN_OR_INSUFFICIENT_EVIDENCE")]["unavailable_outcome_count"] == 1
    assert examples[0]["bounceback_label"] == "BOUNCED_BACK_STRONGLY"
    assert report["veto_breadth_diagnostic"]["strict_veto_breadth_status"] == "TOO_BROAD_FOR_RETURN"
    assert proposal["policy_name"] == "EXTREME_DISTRESS_ONLY_RESEARCH"
    assert proposal["policy_stage"] == "PROPOSED_NOT_REPLAYED"
    assert proposal["paper_trading_allowed"] is False
    assert proposal["live_trading_allowed"] is False
    assert proposal["validation_passed"] is False
    assert proposal["final_validation_status"] == "NOT_FINAL_VALIDATION"


def _policy_row(candidate_id: str, headline: str, *, symbol: str = "AAA") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "headline": headline,
        "availability_timestamp": "2026-01-01T08:00:00+00:00",
        "decision_timestamp": "2026-01-01T09:00:00+00:00",
        "price_score": 0.9,
    }


def test_extreme_distress_only_blocks_existential_distress_not_reversible_bad_news() -> None:
    rows = [
        _policy_row("bankruptcy", "Company files for bankruptcy protection"),
        _policy_row("default", "Issuer enters debt default"),
        _policy_row("delisting", "Exchange issues delisting notice"),
        _policy_row("suspension", "Trading suspension announced"),
        _policy_row("earnings", "Company reports earnings miss"),
        _policy_row("guidance", "Company cuts guidance"),
        _policy_row("downgrade", "Analyst downgrade follows weak quarter"),
    ]

    result = research.apply_catastrophic_policy_variant_to_candidates(rows, "EXTREME_DISTRESS_ONLY")
    blocked_ids = {row["candidate_id"] for row in result["blocked_candidates"]}

    assert {"bankruptcy", "default", "delisting", "suspension"} <= blocked_ids
    assert {"earnings", "guidance", "downgrade"}.isdisjoint(blocked_ids)
    assert result["filter_audit"]["paper_trading_enabled"] is False
    assert result["filter_audit"]["live_trading_enabled"] is False
    assert result["filter_audit"]["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_policy_variant_fraud_and_dilution_semantics_are_separate() -> None:
    rows = [
        _policy_row("fraud", "Fraud investigation and criminal probe announced"),
        _policy_row("auditor", "Auditor resignation after qualified audit"),
        _policy_row("distressed", "Emergency capital raise at deep discount"),
        _policy_row("ordinary", "Ordinary capital raise funds expansion"),
    ]

    fraud = research.apply_catastrophic_policy_variant_to_candidates(rows, "EXTREME_DISTRESS_OR_FRAUD")
    dilution = research.apply_catastrophic_policy_variant_to_candidates(rows, "DISTRESS_OR_DILUTION")

    assert {"fraud", "auditor"} <= {row["candidate_id"] for row in fraud["blocked_candidates"]}
    dilution_blocked = {row["candidate_id"] for row in dilution["blocked_candidates"]}
    assert "distressed" in dilution_blocked
    assert "ordinary" not in dilution_blocked


def test_unknown_evidence_is_reported_separately_not_confirmed_catastrophic() -> None:
    rows = [{"candidate_id": "unknown", "symbol": "UNK", "price_score": 0.7}]

    result = research.apply_catastrophic_policy_variant_to_candidates(rows, "EXTREME_DISTRESS_ONLY")

    assert result["blocked_candidates"] == []
    assert result["unknown_candidates"][0]["severity_group"] == "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"
    assert result["filter_audit"]["unknown_evidence_policy"] == "REPORT_SEPARATELY_NOT_APPROVED_FOR_PAPER_LIVE"


def _policy_variant_replay() -> dict[str, object]:
    base_metrics = {
        "ending_equity": 6.0,
        "total_return_decimal": 5.0,
        "maximum_drawdown": -0.36,
        "Sharpe_ratio": 1.1,
        "Calmar_ratio": 2.0,
        "cvar": -0.08,
        "trade_count": 4,
    }
    extreme_metrics = {**base_metrics, "ending_equity": 5.8, "total_return_decimal": 4.8, "maximum_drawdown": -0.32, "Sharpe_ratio": 1.08, "trade_count": 3}
    fraud_metrics = {**base_metrics, "ending_equity": 5.7, "total_return_decimal": 4.7, "maximum_drawdown": -0.31, "Sharpe_ratio": 1.05, "trade_count": 2}
    dilution_metrics = {**base_metrics, "ending_equity": 5.5, "total_return_decimal": 4.5, "maximum_drawdown": -0.30, "Sharpe_ratio": 1.0, "trade_count": 1}
    severe_metrics = {**base_metrics, "ending_equity": 5.65, "total_return_decimal": 4.65, "maximum_drawdown": -0.29, "Sharpe_ratio": 1.03, "trade_count": 1}
    base_trades = [
        {"trade_id": "t-bankruptcy", "candidate_id": "bankruptcy", "symbol": "AAA", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank", "net_return": -0.20},
        {"trade_id": "t-fraud", "candidate_id": "fraud", "symbol": "BBB", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank", "net_return": -0.15},
        {"trade_id": "t-distressed", "candidate_id": "distressed", "symbol": "CCC", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank", "net_return": 0.18},
        {"trade_id": "t-earnings", "candidate_id": "earnings", "symbol": "DDD", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank", "net_return": 0.04},
    ]
    return {
        "risk_metrics": {
            "news_contrarian_rerank": base_metrics,
            "news_contrarian_rerank_extreme_distress_only_veto": extreme_metrics,
            "news_contrarian_rerank_extreme_distress_or_fraud_veto": fraud_metrics,
            "news_contrarian_rerank_distress_or_dilution_veto": dilution_metrics,
            "news_contrarian_rerank_severe_loss_avoidance_veto": severe_metrics,
        },
        "daily_equity": {
            "news_contrarian_rerank_extreme_distress_only_veto": [{"date": "2026-01-03", "total_equity": 5.8}],
            "news_contrarian_rerank_extreme_distress_or_fraud_veto": [{"date": "2026-01-03", "total_equity": 5.7}],
            "news_contrarian_rerank_distress_or_dilution_veto": [{"date": "2026-01-03", "total_equity": 5.5}],
            "news_contrarian_rerank_severe_loss_avoidance_veto": [{"date": "2026-01-03", "total_equity": 5.65}],
        },
        "extra_research_variant_metadata": {
            "news_contrarian_rerank_extreme_distress_only_veto": {"research_only": True},
            "news_contrarian_rerank_extreme_distress_or_fraud_veto": {"research_only": True},
            "news_contrarian_rerank_distress_or_dilution_veto": {"research_only": True},
            "news_contrarian_rerank_severe_loss_avoidance_veto": {"research_only": True},
        },
        "trade_ledger": base_trades
        + [
            {**base_trades[1], "strategy_variant": "news_contrarian_rerank_extreme_distress_only_veto"},
            {**base_trades[2], "strategy_variant": "news_contrarian_rerank_extreme_distress_only_veto"},
            {**base_trades[3], "strategy_variant": "news_contrarian_rerank_extreme_distress_only_veto"},
            {**base_trades[2], "strategy_variant": "news_contrarian_rerank_extreme_distress_or_fraud_veto"},
            {**base_trades[3], "strategy_variant": "news_contrarian_rerank_extreme_distress_or_fraud_veto"},
            {**base_trades[3], "strategy_variant": "news_contrarian_rerank_distress_or_dilution_veto"},
            {**base_trades[3], "strategy_variant": "news_contrarian_rerank_severe_loss_avoidance_veto"},
        ],
    }


def test_policy_variant_artifacts_handle_metrics_frontier_and_guardrails() -> None:
    rows = [
        _policy_row("bankruptcy", "Company files for bankruptcy", symbol="AAA"),
        _policy_row("fraud", "Fraud and accounting irregularity disclosed", symbol="BBB"),
        _policy_row("distressed", "Emergency capital raise is highly dilutive", symbol="CCC"),
        _policy_row("earnings", "Earnings miss disappoints investors", symbol="DDD"),
        {"candidate_id": "unknown", "symbol": "UNK"},
    ]
    replay = _policy_variant_replay()
    original_base_metrics = dict(replay["risk_metrics"]["news_contrarian_rerank"])  # type: ignore[index]

    artifacts = research._catastrophic_policy_variant_artifacts(
        rows,
        replay,  # type: ignore[arg-type]
        {"veto_breadth_diagnostic": {"strict_veto_breadth_status": "TOO_BROAD_FOR_RETURN"}},
    )
    comparison, counts, metrics_rows, removed, bounceback, frontier_report, frontier, examples = artifacts

    assert comparison["status"] == "RESEARCH_ONLY_POLICY_VARIANTS"
    assert comparison["strict_veto_breadth_status"] == "TOO_BROAD_FOR_RETURN"
    assert "news_contrarian_rerank_soft_risk_reduce_veto" in comparison["count_only_variants"]
    assert frontier_report["status"] == "RESEARCH_ONLY_DIAGNOSTIC"
    assert frontier_report["best_balanced_policy"] in {row["policy_name"] for row in frontier}
    assert all(row["paper_trading_enabled"] is False for row in metrics_rows)
    assert all(row["live_trading_enabled"] is False for row in metrics_rows)
    assert any(row["policy_name"] == "SOFT_RISK_REDUCE" and row["return"] == "UNAVAILABLE_INPUT" for row in metrics_rows)
    assert any(row["bounceback_label"] == "BOUNCED_BACK_STRONGLY" for row in removed)
    assert any(row["severe_loss_count"] != "UNAVAILABLE_INPUT" for row in bounceback)
    assert any(row["example_type"] in {"blocked_winner", "top_severe_loss_avoided"} for row in examples)
    assert any(row["policy_name"] == "SOFT_RISK_REDUCE" and row["full_replay_computed"] is False for row in counts)
    assert replay["risk_metrics"]["news_contrarian_rerank"] == original_base_metrics  # type: ignore[index]

    second = research._catastrophic_policy_variant_artifacts(
        rows,
        replay,  # type: ignore[arg-type]
        {"veto_breadth_diagnostic": {"strict_veto_breadth_status": "TOO_BROAD_FOR_RETURN"}},
    )
    assert second[5]["best_balanced_policy"] == frontier_report["best_balanced_policy"]


def test_strict_veto_breadth_can_be_marked_too_broad_for_return() -> None:
    report = research._strict_veto_breadth_diagnostic(
        {
            "risk_metrics": {
                "news_contrarian_rerank": {"total_return_decimal": 5.0, "maximum_drawdown": -0.36, "Sharpe_ratio": 1.1},
                "news_contrarian_rerank_catastrophic_veto": {"total_return_decimal": 3.8, "maximum_drawdown": -0.36, "Sharpe_ratio": 1.0},
            }
        },
        [{"trade_id": "t1", "net_return": 0.2}],
        [],
    )

    assert report["strict_veto_breadth_status"] == "TOO_BROAD_FOR_RETURN"


def test_loser_bounceback_casebook_selects_cases_and_detects_generic_filings() -> None:
    rows = [
        {
            **_policy_row("loss", "10-Q filed by AXTI", symbol="AXTI"),
            "provider": "sec",
            "summary_text": "",
            "price_model_score": 0.81,
        },
        {
            **_policy_row("win", "NT 10-Q late filing filed by WIN", symbol="WIN"),
            "provider": "sec",
            "summary_text": "late filing notice",
            "price_model_score": 0.77,
        },
    ]
    removed = [
        {"trade_id": "T-LOSS", "candidate_id": "loss", "symbol": "AXTI", "entry_date": "2026-01-02", "exit_date": "2026-01-08", "net_return": "-0.24"},
        {"trade_id": "T-WIN", "candidate_id": "win", "symbol": "WIN", "entry_date": "2026-01-02", "exit_date": "2026-01-08", "net_return": "0.18"},
    ]

    report, cases, feature_diff, keyword_diff, plan = research._catastrophic_veto_loser_bounceback_casebook_artifacts(rows, removed)

    assert report["status"] == "AVAILABLE"
    assert [case["case_type"] for case in cases] == ["top_severe_loser", "top_strong_bounceback_winner"]
    generic_case = next(case for case in cases if case["candidate_id"] == "loss")
    late_case = next(case for case in cases if case["candidate_id"] == "win")
    assert generic_case["headline_is_generic_filing"] is True
    assert generic_case["event_category_research"] == "uncategorized"
    assert late_case["headline_is_generic_filing"] is False
    assert "NT_10_Q" in late_case["filing_forms_detected"]
    assert report["generic_filing_diagnostic"]["needs_filing_content_not_just_headline"] is True
    assert any(row["feature_name"] == "headline_is_generic_filing" for row in feature_diff)
    assert any(row["feature_name"] == "distress_score" for row in keyword_diff)
    assert plan["status"] == "PROPOSED"
    assert plan["proposal_only"] is True
    assert plan["paper_trading_enabled"] is False
    assert plan["live_trading_enabled"] is False
    assert plan["finbert_readiness"] == "NOT_READY"
    assert plan["transformer_enabled"] is False
    assert plan["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_casebook_reports_unavailable_fields_honestly() -> None:
    report, cases, feature_diff, _keyword_diff, _plan = research._catastrophic_veto_loser_bounceback_casebook_artifacts(
        [{"candidate_id": "missing", "symbol": "MISS"}],
        [{"trade_id": "T-MISS", "candidate_id": "missing", "symbol": "MISS", "net_return": "-0.20"}],
    )

    assert report["status"] == "AVAILABLE"
    assert cases[0]["headline_text"] == "UNAVAILABLE_INPUT"
    assert cases[0]["provider"] == "UNAVAILABLE_INPUT"
    assert cases[0]["availability_timestamp"] == "UNAVAILABLE_INPUT"
    assert any(row["feature_name"] == "news_score_decile_if_available" and row["difference"] == "UNAVAILABLE_INPUT" for row in feature_diff)


def test_no_effect_policy_frontier_does_not_name_fake_winner() -> None:
    rows = [_policy_row("ordinary", "Earnings miss but no distress", symbol="ORD")]
    base_metrics = {
        "ending_equity": 6.0,
        "total_return_decimal": 5.0,
        "maximum_drawdown": -0.36,
        "Sharpe_ratio": 1.1,
        "Calmar_ratio": 2.0,
        "cvar": -0.08,
        "trade_count": 1,
    }
    replay = {
        "risk_metrics": {
            "news_contrarian_rerank": base_metrics,
            "news_contrarian_rerank_extreme_distress_only_veto": dict(base_metrics),
            "news_contrarian_rerank_extreme_distress_or_fraud_veto": dict(base_metrics),
            "news_contrarian_rerank_distress_or_dilution_veto": dict(base_metrics),
            "news_contrarian_rerank_severe_loss_avoidance_veto": dict(base_metrics),
        },
        "daily_equity": {
            "news_contrarian_rerank_extreme_distress_only_veto": [{"date": "2026-01-03", "total_equity": 6.0}],
            "news_contrarian_rerank_extreme_distress_or_fraud_veto": [{"date": "2026-01-03", "total_equity": 6.0}],
            "news_contrarian_rerank_distress_or_dilution_veto": [{"date": "2026-01-03", "total_equity": 6.0}],
            "news_contrarian_rerank_severe_loss_avoidance_veto": [{"date": "2026-01-03", "total_equity": 6.0}],
        },
        "extra_research_variant_metadata": {
            "news_contrarian_rerank_extreme_distress_only_veto": {"research_only": True},
            "news_contrarian_rerank_extreme_distress_or_fraud_veto": {"research_only": True},
            "news_contrarian_rerank_distress_or_dilution_veto": {"research_only": True},
            "news_contrarian_rerank_severe_loss_avoidance_veto": {"research_only": True},
        },
        "trade_ledger": [
            {"trade_id": "T-ORD", "candidate_id": "ordinary", "symbol": "ORD", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank", "net_return": "0.02"},
            {"trade_id": "T-ORD", "candidate_id": "ordinary", "symbol": "ORD", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank_extreme_distress_only_veto", "net_return": "0.02"},
            {"trade_id": "T-ORD", "candidate_id": "ordinary", "symbol": "ORD", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank_extreme_distress_or_fraud_veto", "net_return": "0.02"},
            {"trade_id": "T-ORD", "candidate_id": "ordinary", "symbol": "ORD", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank_distress_or_dilution_veto", "net_return": "0.02"},
            {"trade_id": "T-ORD", "candidate_id": "ordinary", "symbol": "ORD", "entry_date": "2026-01-02", "strategy_variant": "news_contrarian_rerank_severe_loss_avoidance_veto", "net_return": "0.02"},
        ],
    }

    _comparison, _counts, _metrics, _removed, _bounceback, frontier_report, _frontier, _examples = research._catastrophic_policy_variant_artifacts(
        rows,
        replay,
        {"veto_breadth_diagnostic": {"strict_veto_breadth_status": "TOO_BROAD_FOR_RETURN"}},
    )

    assert frontier_report["frontier_status"] == "NO_EFFECT_FRONTIER"
    assert frontier_report["best_balanced_policy"] == "UNAVAILABLE_NO_EFFECT"
    assert frontier_report["best_return_preserving_policy"] == "UNAVAILABLE_NO_EFFECT"
    assert frontier_report["best_drawdown_reduction_policy"] == "UNAVAILABLE_NO_EFFECT"
    assert frontier_report["recommended_next_step"] == "inspect loser-vs-bounceback cases and improve taxonomy/source evidence"
    assert frontier_report["validation_passed"] is False


def test_extra_research_replay_variant_is_opt_in_separate_and_non_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"candidate_id": "A", "symbol": "AAA"}]
    filtered_rows = [{"candidate_id": "B", "symbol": "BBB"}]
    original_rows = [dict(row) for row in rows]
    calls: list[tuple[str, list[dict[str, object]]]] = []

    def fake_replay(candidate_rows: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
        calls.append((str(kwargs["variant"]), [dict(row) for row in candidate_rows]))
        return {"ledger": [], "daily_equity": [{"date": "2026-01-01"}], "action_events": []}

    monkeypatch.setattr(research, "_run_open_trade_replay", fake_replay)
    ledgers: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, object]]] = {}
    attribution: list[dict[str, object]] = []

    metadata = research._append_extra_research_variant_results(
        base_rows=rows,
        extra_research_variants=None,
        base_variant_settings={"news_contrarian_rerank": {"contrarian_rerank": True}},
        bars_by_symbol={},
        price_score_column="score",
        replay_config={},
        ledgers=ledgers,
        curves=curves,
        attribution_inputs=attribution,
    )
    assert metadata == {}
    assert calls == []

    spec = research.ResearchStrategyVariantSpec(
        base_variant_name="news_contrarian_rerank",
        new_variant_name="news_contrarian_rerank_catastrophic_veto",
        filtered_candidate_rows=filtered_rows,
        candidate_filter=research.ResearchCandidateFilterSpec("catastrophic_veto", enabled=True),
        metadata={"policy": "catastrophic_veto_v1"},
    )
    metadata = research._append_extra_research_variant_results(
        base_rows=rows,
        extra_research_variants=[spec],
        base_variant_settings={"news_contrarian_rerank": {"contrarian_rerank": True}},
        bars_by_symbol={},
        price_score_column="score",
        replay_config={},
        ledgers=ledgers,
        curves=curves,
        attribution_inputs=attribution,
    )

    assert calls == [("news_contrarian_rerank_catastrophic_veto", filtered_rows)]
    assert "news_contrarian_rerank_catastrophic_veto" in curves
    assert metadata["news_contrarian_rerank_catastrophic_veto"]["research_only"] is True
    assert metadata["news_contrarian_rerank_catastrophic_veto"]["paper_trading_enabled"] is False
    assert metadata["news_contrarian_rerank_catastrophic_veto"]["live_trading_enabled"] is False
    assert rows == original_rows


def test_catastrophic_veto_replay_seam_report_is_added_not_executed() -> None:
    report = research._catastrophic_veto_replay_seam_report()

    assert report["status"] == "ADAPTER_ADDED_NOT_EXECUTED"
    assert report["safe_filtered_variant_seam_status"] == "REPLAY_ADAPTER_AVAILABLE_OPT_IN_ONLY"
    assert report["full_replay_computed"] is False
    assert report["default_behavior_unchanged"] is True
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False
    assert report["validation_passed"] is False
    assert report["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert "absent from replay" in report["remaining_blocker"]
    assert "candidate rows" in report["candidate_construction_entrypoint"]
    assert "ResearchStrategyVariantSpec" in report["strategy_variant_construction_entrypoint"]
    assert "_build_open_trade_replay" in report["replay_input_entrypoint"]
    assert "trade ledger" in report["trade_ledger_writer_entrypoint"]


def test_catastrophic_veto_full_replay_blocker_is_explicit() -> None:
    blocker = research._catastrophic_veto_full_replay_blocker()

    assert blocker["safe_replay_insertion_point"] == "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM"
    assert "absent from replay" in blocker["full_replay_blocker"]
    assert "metrics, equity, and variant metadata" in blocker["full_replay_limitation"]
    assert "candidate rows" in blocker["candidate_input_source"]
    assert "_build_open_trade_replay" in blocker["replay_engine_entrypoint"]


def test_catastrophic_veto_attribution_counts_blocked_candidates_and_trades() -> None:
    rows = [
        {
            "candidate_id": "XYZ-2026-01-02",
            "symbol": "XYZ",
            "headline": "XYZ files for bankruptcy protection",
            "publication_timestamp": "2026-01-02T08:00:00Z",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
        {
            "candidate_id": "ABC-2026-01-03",
            "symbol": "ABC",
            "headline": "",
        },
    ]
    replay = {
        "trade_ledger": [
            {
                "candidate_id": "XYZ-2026-01-02",
                "symbol": "XYZ",
                "strategy_variant": "news_contrarian_rerank",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "net_return": "0.05",
                "pnl": "100.0",
            },
            {
                "candidate_id": "ABC-2026-01-03",
                "symbol": "ABC",
                "strategy_variant": "news_contrarian_rerank",
                "entry_date": "2026-01-03",
                "exit_date": "2026-01-06",
                "net_return": "-0.01",
                "pnl": "-20.0",
            },
        ],
        "portfolio_comparison": {
            "price_only": {"total_return_decimal": 0.1},
            "news_contrarian_rerank": {"total_return_decimal": 0.2},
        },
    }

    (
        attribution,
        trade_rows,
        comparison,
        _policy,
        filtered_report,
        removed_trades,
        removed_symbols,
        full_replay_report,
        full_replay_trade_ledger,
        full_replay_equity,
        filtered_candidates,
        blocked_candidates,
    ) = research._catastrophic_veto_strategy_artifacts(rows, replay)

    assert attribution["blocked_candidate_count"] == 1
    assert attribution["manual_review_candidate_count"] == 1
    assert attribution["unknown_candidate_count"] == 1
    assert attribution["executed_trade_count"] == 2
    assert attribution["blocked_executed_trade_count"] == 1
    assert attribution["manual_review_executed_trade_count"] == 1
    assert attribution["unknown_executed_trade_count"] == 1
    assert attribution["point_in_time_unsafe_count"] == 1
    assert trade_rows[0]["research_only_veto_would_apply"] is True
    assert trade_rows[1]["unknown_or_unavailable"] is True
    assert comparison["base_strategy"] == "news_contrarian_rerank"
    assert comparison["veto_strategy"] == "news_contrarian_rerank_catastrophic_veto"
    assert comparison["strategy_names"] == [
        "price_only",
        "news_contrarian_rerank",
        "news_contrarian_rerank_catastrophic_veto",
    ]
    assert comparison["replay_impact_status"] == "FULL_REPLAY_NOT_AVAILABLE"
    assert comparison["approximate_replay_impact_status"] == "APPROXIMATE_LEDGER_SIMULATION"
    assert comparison["safe_replay_insertion_point"] == "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM"
    assert "absent from replay" in comparison["full_replay_blocker"]
    assert "candidate rows" in comparison["candidate_input_source"]
    assert "_build_open_trade_replay" in comparison["replay_engine_entrypoint"]
    assert comparison["used_in_current_replay"] is False
    assert comparison["full_replay_computed"] is False
    assert comparison["approximate_simulation_used"] is True
    assert comparison["veto_enabled_for_paper_trading"] is False
    assert comparison["veto_enabled_for_live_trading"] is False
    assert comparison["base_metrics"]["news_contrarian_rerank"]["total_return_decimal"] == 0.2
    assert comparison["veto_metrics"]["approximate_removed_trade_count"] == 2
    assert comparison["validation_passed"] is False
    assert comparison["final_validation_status"] == "NOT_FINAL_VALIDATION"
    assert filtered_report["schema_name"] == "catastrophic_veto_filtered_strategy_report"
    assert filtered_report["replay_impact_status"] == "APPROXIMATE_LEDGER_SIMULATION"
    assert filtered_report["approximate_simulation_superseded_by_full_replay"] is False
    assert filtered_report["used_in_current_replay"] is False
    assert filtered_report["paper_trading_enabled"] is False
    assert filtered_report["live_trading_enabled"] is False
    assert filtered_report["delta_metrics"]["removed_pnl_contribution"] == pytest.approx(80.0)
    assert filtered_report["delta_metrics"]["removed_return_contribution"] == pytest.approx(0.04)
    assert len(removed_trades) == 2
    assert removed_trades[0]["removal_reason"] == "BLOCK_CONTRARIAN_ENTRY"
    assert removed_trades[0]["pnl"] == "100.0"
    assert removed_symbols
    assert full_replay_report["schema_name"] == "catastrophic_veto_full_replay_report"
    assert full_replay_report["replay_impact_status"] == "FULL_REPLAY_NOT_AVAILABLE"
    assert full_replay_report["safe_replay_insertion_point"] == "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM"
    assert "absent from replay" in full_replay_report["full_replay_blocker"]
    assert "metrics, equity, and variant metadata" in full_replay_report["full_replay_limitation"]
    assert "candidate rows" in full_replay_report["candidate_input_source"]
    assert "_build_open_trade_replay" in full_replay_report["replay_engine_entrypoint"]
    assert full_replay_report["full_replay_computed"] is False
    assert full_replay_report["approximate_simulation_used"] is True
    assert full_replay_report["used_in_current_replay"] is False
    assert full_replay_report["paper_trading_enabled"] is False
    assert full_replay_report["live_trading_enabled"] is False
    assert full_replay_trade_ledger == []
    assert full_replay_equity == []
    assert len(filtered_candidates) == 0
    assert len(blocked_candidates) == 2


def test_catastrophic_veto_full_replay_is_computed_only_from_separate_variant_output() -> None:
    rows = [
        {
            "candidate_id": "SAFE-1",
            "symbol": "SAFE",
            "headline": "ordinary earnings update",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
        {
            "candidate_id": "BAD-1",
            "symbol": "BAD",
            "headline": "BAD files for bankruptcy protection",
            "availability_timestamp": "2026-01-02T08:05:00Z",
        },
    ]
    price_metrics = {"total_return_decimal": 0.10, "ending_equity": 1.10}
    base_metrics = {"total_return_decimal": 0.20, "ending_equity": 1.20}
    veto_metrics = {"total_return_decimal": 0.15, "ending_equity": 1.15}
    replay = {
        "trade_ledger": [
            {"candidate_id": "SAFE-1", "symbol": "SAFE", "entry_date": "2026-01-03", "strategy_variant": "news_contrarian_rerank"},
            {"candidate_id": "BAD-1", "symbol": "BAD", "entry_date": "2026-01-03", "strategy_variant": "news_contrarian_rerank"},
            {"candidate_id": "SAFE-1", "symbol": "SAFE", "entry_date": "2026-01-03", "strategy_variant": "news_contrarian_rerank_catastrophic_veto"},
        ],
        "risk_metrics": {
            "price_only": price_metrics,
            "news_contrarian_rerank": base_metrics,
            "news_contrarian_rerank_catastrophic_veto": veto_metrics,
        },
        "daily_equity": {
            "news_contrarian_rerank_catastrophic_veto": [{"date": "2026-01-03", "total_equity": 1.15}],
        },
        "extra_research_variant_metadata": {
            "news_contrarian_rerank_catastrophic_veto": {"research_only": True},
        },
    }

    artifacts = research._catastrophic_veto_strategy_artifacts(rows, replay)
    comparison = artifacts[2]
    filtered_report = artifacts[4]
    full_report = artifacts[7]
    veto_ledger = artifacts[8]
    veto_equity = artifacts[9]

    assert full_report["replay_impact_status"] == "FULL_REPLAY_COMPUTED"
    assert full_report["full_replay_computed"] is True
    assert full_report["approximate_simulation_used"] is False
    assert full_report["used_in_current_replay"] is False
    assert full_report["base_contrarian_metrics"] == base_metrics
    assert full_report["veto_contrarian_metrics"] == veto_metrics
    assert full_report["removed_trade_count"] == 1
    assert full_report["replacement_trade_count"] == 0
    assert comparison["base_metrics"]["price_only"] == price_metrics
    assert comparison["base_metrics"]["news_contrarian_rerank"] == base_metrics
    assert comparison["veto_metrics"] == veto_metrics
    assert filtered_report["approximate_simulation_superseded_by_full_replay"] is True
    assert len(veto_ledger) == 1
    assert veto_equity == [{"date": "2026-01-03", "total_equity": 1.15}]
    assert full_report["paper_trading_enabled"] is False
    assert full_report["live_trading_enabled"] is False
    assert full_report["validation_passed"] is False
    assert full_report["final_validation_status"] == "NOT_FINAL_VALIDATION"


def test_catastrophic_veto_zero_candidate_replay_is_explicitly_qualified() -> None:
    rows = [{"candidate_id": "UNKNOWN-1", "symbol": "UNK", "headline": ""}]
    replay = {
        "trade_ledger": [
            {"candidate_id": "UNKNOWN-1", "symbol": "UNK", "entry_date": "2026-01-03", "strategy_variant": "news_contrarian_rerank"},
        ],
        "risk_metrics": {
            "price_only": {"total_return_decimal": 0.10},
            "news_contrarian_rerank": {"total_return_decimal": 0.20},
            "news_contrarian_rerank_catastrophic_veto": {},
        },
        "daily_equity": {"news_contrarian_rerank_catastrophic_veto": []},
        "extra_research_variant_metadata": {
            "news_contrarian_rerank_catastrophic_veto": {"research_only": True},
        },
    }

    artifacts = research._catastrophic_veto_strategy_artifacts(rows, replay)
    comparison = artifacts[2]
    full_report = artifacts[7]

    assert full_report["replay_impact_status"] == "FULL_REPLAY_COMPUTED_ZERO_CANDIDATES"
    assert full_report["veto_metrics_status"] == "UNAVAILABLE_EMPTY_CANDIDATE_SET"
    assert full_report["empty_output_reason"] == "STRICT_POLICY_BLOCKED_ALL_CANDIDATES"
    assert full_report["confirmed_catastrophic_candidate_count"] == 0
    assert full_report["unknown_text_candidate_count"] == 1
    assert full_report["missing_availability_candidate_count"] == 1
    assert full_report["strict_policy_blocked_candidate_count"] == 1
    assert full_report["delta_metrics"]["delta_return"] == "UNAVAILABLE_INPUT"
    assert comparison["full_replay_blocked_candidate_count"] == 1
    assert comparison["full_replay_removed_trade_count"] == 1
    assert comparison["approximate_blocked_candidate_count"] == 0
    assert comparison["catastrophic_veto_policy_mode"] == "STRICT_SAFETY"


def test_empty_catastrophic_veto_csv_has_headers(tmp_path: Path) -> None:
    path = tmp_path / "empty_veto_ledger.csv"

    research._write_csv(
        path,
        [],
        empty_fields=("trade_id", "candidate_id", "strategy_variant"),
    )

    assert path.read_text(encoding="utf-8").strip() == "trade_id,candidate_id,strategy_variant"


def test_catastrophic_veto_summary_lines_show_research_only_status(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "catastrophic_veto_candidate_attribution.json",
        {
            "blocked_candidate_count": 2,
            "manual_review_candidate_count": 3,
            "blocked_executed_trade_count": 1,
        },
    )
    _write_json(
        tmp_path / "catastrophic_veto_strategy_comparison.json",
        {"replay_impact_status": "APPROXIMATE_LEDGER_SIMULATION"},
    )
    _write_json(
        tmp_path / "catastrophic_veto_filtered_strategy_report.json",
        {
            "replay_impact_status": "APPROXIMATE_LEDGER_SIMULATION",
            "delta_metrics": {"removed_return_contribution": -0.04},
        },
    )
    _write_json(
        tmp_path / "catastrophic_veto_full_replay_report.json",
        {"replay_impact_status": "FULL_REPLAY_NOT_AVAILABLE"},
    )
    _write_json(
        tmp_path / "catastrophic_news_evidence_quality_report.json",
        {"status": "INSUFFICIENT_FOR_STRICT_VETO", "usable_for_strict_veto_count": 0},
    )

    text = "\n".join(research._catastrophic_veto_summary_lines(tmp_path))

    assert "catastrophic news veto: RESEARCH_ONLY / NOT_CURRENT_STRATEGY" in text
    assert "catastrophic blocked candidates: 2" in text
    assert "catastrophic veto scenario: FULL_REPLAY_NOT_AVAILABLE" in text
    assert "catastrophic veto approximate simulation: APPROXIMATE_LEDGER_SIMULATION" in text
    assert "catastrophic veto blocked trades: 1" in text
    assert "catastrophic veto full replay: FULL_REPLAY_NOT_AVAILABLE" in text
    assert "catastrophic veto trades removed/replaced: UNAVAILABLE_INPUT/UNAVAILABLE_INPUT" in text
    assert "catastrophic veto paper/live allowed: False / False" in text
    assert "catastrophic evidence quality: INSUFFICIENT_FOR_STRICT_VETO" in text
    assert "catastrophic usable strict-veto candidates: 0" in text
    assert "catastrophic policy modes: STRICT_SAFETY / CONFIRMED_ONLY_RESEARCH / MANUAL_REVIEW_RESEARCH" in text
    assert "catastrophic veto delta return: UNAVAILABLE_INPUT" in text
    assert "manual review candidates: 3" in text


def test_validation_stage_placeholders_include_catastrophic_veto_attribution_only() -> None:
    placeholders = research._validation_stage_placeholders()
    veto = placeholders["catastrophic_veto_strategy_comparison"]

    assert veto["status"] == "APPROXIMATE_LEDGER_SIMULATION"
    assert veto["replay_impact_status"] == "APPROXIMATE_LEDGER_SIMULATION"
    assert veto["full_replay_status"] == "FULL_REPLAY_NOT_AVAILABLE"
    assert veto["safe_replay_insertion_point"] == "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM"
    assert veto["safe_filtered_variant_seam_status"] == "REPLAY_ADAPTER_AVAILABLE_OPT_IN_ONLY"
    assert "replay output does not contain the catastrophic-veto variant" in veto["full_replay_blocker"]
    assert veto["metric_output_allowed"] is False
    assert veto["blocks_final_validation"] is True
    assert veto["veto_strategy"] == "news_contrarian_rerank_catastrophic_veto"


def test_news_risk_summary_shows_catastrophic_veto_status_concisely() -> None:
    summary = {
        "output_dir": "out",
        "strategy_comparison": [],
        "cost_robustness": [],
        "diagnostics": {
            "paper_orders_enabled": False,
            "live_orders_enabled": False,
            "validation_label": "PSEUDO_HOLDOUT",
        },
        "catastrophic_veto": {
            "replay_impact_status": "FULL_REPLAY_NOT_AVAILABLE",
            "approximate_replay_impact_status": "APPROXIMATE_LEDGER_SIMULATION",
            "blocked_candidate_count": 2,
            "blocked_trade_count": 1,
            "manual_review_candidate_count": 3,
            "delta_return": -0.04,
        },
        "news_evidence": {
            "status": "INSUFFICIENT",
            "candidate_count": 131892,
            "has_any_text_count": 0,
            "has_availability_timestamp_count": 0,
        },
        "winners": {},
        "warnings": [],
    }

    text = format_news_risk_overlay_summary(summary, [], mode="summary")

    assert "catastrophic news veto: RESEARCH_ONLY / NOT_CURRENT_STRATEGY" in text
    assert "catastrophic veto scenario: FULL_REPLAY_NOT_AVAILABLE" in text
    assert "catastrophic veto approximate simulation: APPROXIMATE_LEDGER_SIMULATION" in text
    assert "catastrophic blocked candidates: 2" in text
    assert "catastrophic veto blocked trades: 1" in text
    assert "catastrophic veto delta return: -0.04" in text
    assert "manual review candidates: 3" in text
    assert "FULL_REPLAY_COMPUTED" not in text
    assert "PSEUDO_HOLDOUT" in text
    assert "paper/live trading enabled: False / False" in text
    assert "news evidence readiness: INSUFFICIENT" in text
    assert "news evidence text coverage: 0 / 131892" in text
    assert "news evidence availability timestamps: 0 / 131892" in text
    assert len(text.splitlines()) <= 40


def test_news_transformer_contracts_and_readiness_are_disabled_scaffold() -> None:
    example = NewsSequenceExample(
        symbol="AAPL",
        decision_timestamp="2024-01-01T00:00:00+00:00",
        availability_timestamp=None,
        publication_timestamp=None,
        candidate_id="candidate-1",
        strategy_variant="news_contrarian_rerank",
    )

    failures = validate_news_sequence_schema([example.__dict__])
    report = build_news_transformer_readiness_report([example.__dict__])

    assert "headline/article text missing" in failures
    assert "availability timestamps missing" in failures
    assert report["status"] == "NOT_READY"
    assert report["transformer_readiness"] == "NOT_READY"
    assert report["bert_readiness"] == "NOT_READY"
    assert report["finbert_readiness"] == "NOT_READY"
    assert report["enabled"] is False
    assert report["training_enabled"] is False
    assert report["inference_enabled"] is False
    assert report["used_in_strategy"] is False
    assert report["used_in_replay"] is False
    assert report["paper_trading_enabled"] is False
    assert report["live_trading_enabled"] is False


def test_news_transformer_training_plan_is_plan_only_and_disabled() -> None:
    plan = build_news_transformer_training_plan()

    assert plan["status"] == "NOT_READY"
    assert plan["plan_only"] is True
    assert plan["training_enabled"] is False
    assert plan["inference_enabled"] is False
    assert plan["optional_dependencies_imported"] is False
    assert plan["model_downloads_enabled"] is False
    assert plan["validation_passed"] is False


def test_news_transformer_split_validation_rejects_random_split() -> None:
    assert validate_no_random_split([{"split_name": "random_train"}]) is False
    assert validate_no_random_split([{"split_name": "development"}, {"split_name": "final_holdout"}]) is True


def _minimal_research_paths(tmp_path: Path) -> research.NewsRiskResearchPaths:
    kwargs = {
        "output_dir": tmp_path,
        "dataset_csv_path": tmp_path / "dataset.csv",
        "coverage_json_path": tmp_path / "coverage.json",
        "leakage_json_path": tmp_path / "leakage.json",
        "metrics_json_path": tmp_path / "metrics.json",
        "portfolio_json_path": tmp_path / "portfolio.json",
        "accounting_json_path": tmp_path / "accounting.json",
        "accounting_audit_json_path": tmp_path / "accounting_audit.json",
        "equity_curve_csv_path": tmp_path / "equity.csv",
        "drawdown_curve_csv_path": tmp_path / "drawdown.csv",
        "trade_ledger_csv_path": tmp_path / "trade_ledger.csv",
        "daily_equity_price_only_csv_path": tmp_path / "daily_price.csv",
        "daily_equity_news_cash_csv_path": tmp_path / "daily_cash.csv",
        "daily_equity_news_replacement_csv_path": tmp_path / "daily_replacement.csv",
        "daily_equity_news_reduced_size_csv_path": tmp_path / "daily_reduced.csv",
        "open_trade_portfolio_json_path": tmp_path / "open_portfolio.json",
        "replay_risk_metrics_json_path": tmp_path / "risk.json",
        "action_attribution_json_path": tmp_path / "action.json",
        "score_direction_audit_json_path": tmp_path / "score_audit.json",
        "news_score_deciles_csv_path": tmp_path / "deciles.csv",
        "corrected_news_score_deciles_csv_path": tmp_path / "corrected_deciles.csv",
        "decile_join_audit_json_path": tmp_path / "decile_join.json",
        "decile_trade_reconciliation_json_path": tmp_path / "decile_reconciliation.json",
        "news_score_direction_report_json_path": tmp_path / "score_report.json",
        "news_score_direction_summary_md_path": tmp_path / "score.md",
        "replay_action_attribution_json_path": tmp_path / "replay_action.json",
        "event_category_analysis_json_path": tmp_path / "events.json",
        "contrarian_strategy_comparison_json_path": tmp_path / "contrarian.json",
        "contrarian_trade_ledger_csv_path": tmp_path / "contrarian_trades.csv",
        "price_stabilisation_comparison_json_path": tmp_path / "stabilisation.json",
        "resilience_filter_analysis_json_path": tmp_path / "resilience.json",
        "extreme_event_archive_csv_path": tmp_path / "extreme.csv",
        "extreme_event_memory_report_json_path": tmp_path / "extreme.json",
        "cost_scenario_comparison_json_path": tmp_path / "cost.json",
        "chronological_split_manifest_json_path": tmp_path / "split.json",
        "experiment_registry_jsonl_path": tmp_path / "registry.jsonl",
        "contrarian_grid_results_csv_path": tmp_path / "grid.csv",
        "contrarian_grid_selection_json_path": tmp_path / "grid_selection.json",
        "contrarian_fold_results_csv_path": tmp_path / "folds.csv",
        "contrarian_parameter_stability_json_path": tmp_path / "stability.json",
        "contrarian_frozen_config_json_path": tmp_path / "frozen.json",
        "contrarian_holdout_report_json_path": tmp_path / "holdout.json",
        "contrarian_holdout_trade_ledger_csv_path": tmp_path / "holdout_trades.csv",
        "contrarian_holdout_equity_csv_path": tmp_path / "holdout_equity.csv",
        "contrarian_holdout_comparison_md_path": tmp_path / "holdout.md",
        "contrarian_walk_forward_folds_csv_path": tmp_path / "wf.csv",
        "contrarian_walk_forward_summary_json_path": tmp_path / "wf.json",
        "contrarian_chronological_validation_plan_json_path": tmp_path / "contrarian_chronological_validation_plan.json",
        "contrarian_chronological_periods_csv_path": tmp_path / "contrarian_chronological_periods.csv",
        "contrarian_walk_forward_validation_report_json_path": tmp_path / "contrarian_walk_forward_validation_report.json",
        "contrarian_placebo_permutation_report_json_path": tmp_path / "contrarian_placebo_permutation_report.json",
        "contrarian_placebo_permutation_results_csv_path": tmp_path / "contrarian_placebo_permutation_results.csv",
        "contrarian_matched_control_report_json_path": tmp_path / "contrarian_matched_control_report.json",
        "contrarian_matched_control_results_csv_path": tmp_path / "contrarian_matched_control_results.csv",
        "contrarian_profit_concentration_report_json_path": tmp_path / "contrarian_profit_concentration_report.json",
        "contrarian_trade_fragility_by_symbol_csv_path": tmp_path / "contrarian_trade_fragility_by_symbol.csv",
        "contrarian_trade_fragility_by_year_csv_path": tmp_path / "contrarian_trade_fragility_by_year.csv",
        "contrarian_top_trade_removal_csv_path": tmp_path / "contrarian_top_trade_removal.csv",
        "contrarian_year_regime_report_json_path": tmp_path / "contrarian_year_regime_report.json",
        "contrarian_year_regime_results_csv_path": tmp_path / "contrarian_year_regime_results.csv",
        "contrarian_year_regime_examples_csv_path": tmp_path / "contrarian_year_regime_examples.csv",
        "contrarian_symbol_year_ablation_report_json_path": tmp_path / "contrarian_symbol_year_ablation_report.json",
        "contrarian_without_top_symbols_csv_path": tmp_path / "contrarian_without_top_symbols.csv",
        "contrarian_without_top_years_csv_path": tmp_path / "contrarian_without_top_years.csv",
        "contrarian_cost_slippage_robustness_report_json_path": tmp_path / "contrarian_cost_slippage_robustness_report.json",
        "contrarian_cost_slippage_robustness_csv_path": tmp_path / "contrarian_cost_slippage_robustness.csv",
        "contrarian_data_validity_audit_json_path": tmp_path / "contrarian_data_validity_audit.json",
        "intraday_5min_expansion_plan_json_path": tmp_path / "intraday_5min_expansion_plan.json",
        "contrarian_placebo_results_csv_path": tmp_path / "placebo.csv",
        "contrarian_placebo_summary_json_path": tmp_path / "placebo.json",
        "contrarian_matched_controls_json_path": tmp_path / "matched.json",
        "contrarian_contribution_by_year_csv_path": tmp_path / "year.csv",
        "contrarian_contribution_by_symbol_csv_path": tmp_path / "symbol.csv",
        "contrarian_concentration_report_json_path": tmp_path / "concentration.json",
        "universe_survivorship_audit_json_path": tmp_path / "survivorship.json",
        "universe_membership_by_date_csv_path": tmp_path / "membership.csv",
        "corporate_action_audit_json_path": tmp_path / "corporate.json",
        "missing_news_bias_report_json_path": tmp_path / "missing.json",
        "covered_vs_uncovered_candidates_csv_path": tmp_path / "covered.csv",
        "text_model_readiness_json_path": tmp_path / "text.json",
        "validation_stage_placeholders_json_path": tmp_path / "placeholders.json",
        "parallel_execution_report_json_path": tmp_path / "parallel.json",
        "replay_assumptions_json_path": tmp_path / "assumptions.json",
        "replay_data_audit_json_path": tmp_path / "data_audit.json",
        "artifact_manifest_json_path": tmp_path / "artifact_manifest.json",
        "artifact_validation_report_json_path": tmp_path / "artifact_validation.json",
        "news_validation_workflow_map_json_path": tmp_path / "workflow.json",
        "validation_dependency_graph_json_path": tmp_path / "dependency.json",
        "validation_readiness_dashboard_json_path": tmp_path / "validation_readiness_dashboard.json",
        "artifact_lineage_report_json_path": tmp_path / "artifact_lineage_report.json",
        "news_validation_gap_analysis_json_path": tmp_path / "news_validation_gap_analysis.json",
        "news_transformer_readiness_json_path": tmp_path / "news_transformer_readiness.json",
        "news_transformer_training_plan_json_path": tmp_path / "news_transformer_training_plan.json",
        "catastrophic_news_audit_json_path": tmp_path / "catastrophic_news_audit.json",
        "catastrophic_news_candidates_csv_path": tmp_path / "catastrophic_news_candidates.csv",
        "catastrophic_news_veto_report_json_path": tmp_path / "catastrophic_news_veto_report.json",
        "catastrophic_veto_candidate_attribution_json_path": tmp_path / "catastrophic_veto_candidate_attribution.json",
        "catastrophic_veto_trade_attribution_csv_path": tmp_path / "catastrophic_veto_trade_attribution.csv",
        "catastrophic_veto_strategy_comparison_json_path": tmp_path / "catastrophic_veto_strategy_comparison.json",
        "catastrophic_veto_policy_json_path": tmp_path / "catastrophic_veto_policy.json",
        "catastrophic_veto_filtered_strategy_report_json_path": tmp_path / "catastrophic_veto_filtered_strategy_report.json",
        "catastrophic_veto_removed_trades_csv_path": tmp_path / "catastrophic_veto_removed_trades.csv",
        "catastrophic_veto_removed_symbols_csv_path": tmp_path / "catastrophic_veto_removed_symbols.csv",
        "catastrophic_veto_full_replay_report_json_path": tmp_path / "catastrophic_veto_full_replay_report.json",
        "catastrophic_veto_full_replay_trade_ledger_csv_path": tmp_path / "catastrophic_veto_full_replay_trade_ledger.csv",
        "catastrophic_veto_full_replay_equity_csv_path": tmp_path / "catastrophic_veto_full_replay_equity.csv",
        "catastrophic_veto_filtered_candidates_csv_path": tmp_path / "catastrophic_veto_filtered_candidates.csv",
        "catastrophic_veto_blocked_candidates_csv_path": tmp_path / "catastrophic_veto_blocked_candidates.csv",
        "catastrophic_veto_replay_seam_report_json_path": tmp_path / "catastrophic_veto_replay_seam_report.json",
        "catastrophic_veto_bounceback_report_json_path": tmp_path / "catastrophic_veto_bounceback_report.json",
        "catastrophic_veto_bounceback_by_category_csv_path": tmp_path / "catastrophic_veto_bounceback_by_category.csv",
        "catastrophic_veto_bounceback_examples_csv_path": tmp_path / "catastrophic_veto_bounceback_examples.csv",
        "catastrophic_veto_extreme_only_policy_proposal_json_path": tmp_path / "catastrophic_veto_extreme_only_policy_proposal.json",
        "catastrophic_veto_policy_variant_comparison_json_path": tmp_path / "catastrophic_veto_policy_variant_comparison.json",
        "catastrophic_veto_policy_variant_counts_csv_path": tmp_path / "catastrophic_veto_policy_variant_counts.csv",
        "catastrophic_veto_policy_variant_metrics_csv_path": tmp_path / "catastrophic_veto_policy_variant_metrics.csv",
        "catastrophic_veto_policy_variant_removed_trades_csv_path": tmp_path / "catastrophic_veto_policy_variant_removed_trades.csv",
        "catastrophic_veto_policy_variant_bounceback_csv_path": tmp_path / "catastrophic_veto_policy_variant_bounceback.csv",
        "catastrophic_veto_policy_frontier_report_json_path": tmp_path / "catastrophic_veto_policy_frontier_report.json",
        "catastrophic_veto_policy_frontier_csv_path": tmp_path / "catastrophic_veto_policy_frontier.csv",
        "catastrophic_veto_policy_variant_examples_csv_path": tmp_path / "catastrophic_veto_policy_variant_examples.csv",
        "catastrophic_veto_loser_bounceback_casebook_json_path": tmp_path / "catastrophic_veto_loser_bounceback_casebook.json",
        "catastrophic_veto_loser_bounceback_cases_csv_path": tmp_path / "catastrophic_veto_loser_bounceback_cases.csv",
        "catastrophic_veto_loser_bounceback_feature_diff_csv_path": tmp_path / "catastrophic_veto_loser_bounceback_feature_diff.csv",
        "catastrophic_veto_loser_bounceback_keyword_diff_csv_path": tmp_path / "catastrophic_veto_loser_bounceback_keyword_diff.csv",
        "catastrophic_veto_taxonomy_improvement_plan_json_path": tmp_path / "catastrophic_veto_taxonomy_improvement_plan.json",
        "catastrophic_veto_parked_status_json_path": tmp_path / "catastrophic_veto_parked_status.json",
        "catastrophic_news_evidence_quality_report_json_path": tmp_path / "catastrophic_news_evidence_quality_report.json",
        "catastrophic_news_evidence_quality_by_field_csv_path": tmp_path / "catastrophic_news_evidence_quality_by_field.csv",
        "catastrophic_news_evidence_quality_by_symbol_csv_path": tmp_path / "catastrophic_news_evidence_quality_by_symbol.csv",
        "catastrophic_veto_policy_mode_comparison_json_path": tmp_path / "catastrophic_veto_policy_mode_comparison.json",
        "catastrophic_veto_policy_mode_counts_csv_path": tmp_path / "catastrophic_veto_policy_mode_counts.csv",
        "news_evidence_lineage_report_json_path": tmp_path / "news_evidence_lineage_report.json",
        "news_evidence_lineage_by_stage_csv_path": tmp_path / "news_evidence_lineage_by_stage.csv",
        "news_evidence_missing_field_examples_csv_path": tmp_path / "news_evidence_missing_field_examples.csv",
        "news_evidence_readiness_report_json_path": tmp_path / "news_evidence_readiness_report.json",
        "news_event_taxonomy_report_json_path": tmp_path / "news_event_taxonomy_report.json",
        "news_event_taxonomy_counts_csv_path": tmp_path / "news_event_taxonomy_counts.csv",
        "news_event_taxonomy_examples_csv_path": tmp_path / "news_event_taxonomy_examples.csv",
        "news_duplicate_grouping_report_json_path": tmp_path / "news_duplicate_grouping_report.json",
        "news_duplicate_grouping_examples_csv_path": tmp_path / "news_duplicate_grouping_examples.csv",
        "news_point_in_time_text_safety_report_json_path": tmp_path / "news_point_in_time_text_safety_report.json",
        "news_point_in_time_text_safety_examples_csv_path": tmp_path / "news_point_in_time_text_safety_examples.csv",
        "news_text_keyword_baseline_report_json_path": tmp_path / "news_text_keyword_baseline_report.json",
        "news_text_keyword_baseline_scores_csv_path": tmp_path / "news_text_keyword_baseline_scores.csv",
        "walk_forward_validation_report_json_path": tmp_path / "walk_forward_validation_report.json",
        "walk_forward_fold_results_csv_path": tmp_path / "walk_forward_fold_results.csv",
        "placebo_permutation_report_json_path": tmp_path / "placebo_permutation_report.json",
        "placebo_permutation_results_csv_path": tmp_path / "placebo_permutation_results.csv",
        "exposure_matched_controls_json_path": tmp_path / "exposure_matched_controls.json",
        "trade_count_matched_controls_json_path": tmp_path / "trade_count_matched_controls.json",
        "concentration_fragility_report_json_path": tmp_path / "concentration_fragility_report.json",
        "shadow_csv_path": tmp_path / "shadow.csv",
        "manifest_json_path": tmp_path / "manifest.json",
        "markdown_path": tmp_path / "README.md",
    }
    return research.NewsRiskResearchPaths(**kwargs)


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
