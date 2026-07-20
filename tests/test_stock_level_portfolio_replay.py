import csv
import inspect
import json

import pytest

from core.research.ml.data.datasets import build_dataset, dataset_leakage_audit
from core.research.ml.exposure_input import (
    SELECTOR_DERIVED_SOURCE_TYPE,
    validate_exposure_input_resolution,
)
from core.research.ml.features.labels import ShouldReduceExposureLabelBuilder
from core.research.ml.stock_level import stock_level_portfolio_replay
from core.research.ml.stock_level.stock_level_portfolio_replay import (
    assert_stock_selector_rebalance_dataset_reuse_compatible,
    build_stock_selector_rebalance_dataset_from_artifacts,
    build_stock_level_portfolio_replay,
    write_stock_selector_rebalance_dataset,
    write_stock_level_portfolio_replay,
)


def _rows():
    rows = []
    for date_index, rebalance_date in enumerate(("2024-01-01", "2024-01-11", "2024-01-21")):
        for index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD")):
            score = 4 - index + date_index * 0.01
            rows.append({"rebalance_date": rebalance_date, "symbol": symbol, "fold_id": date_index + 1, "actual_forward_return_10d": (3 - index) / 100, "actual_benchmark_return_10d": 0.01, "ml_signal": score, "predicted_momentum_120d": score})
    return rows


def _build(**overrides):
    arguments = {"benchmark": {"walk_forward": {"out_of_sample_only": True}}, "signal_columns": ("ml_signal", "predicted_momentum_120d"), "top_n": 2, "max_position_weight": 0.5}
    arguments.update(overrides)
    return build_stock_level_portfolio_replay(_rows(), **arguments)


def test_replay_is_oos_only_and_selection_is_deterministic():
    rows = _rows() + [{"rebalance_date": "2024-02-01", "symbol": "ZZZ", "fold_id": "", "actual_forward_return_10d": 9, "actual_benchmark_return_10d": 0.01, "ml_signal": 99}]
    summary, _, holdings, payload = build_stock_level_portfolio_replay(rows, benchmark={"walk_forward": {"out_of_sample_only": True}}, signal_columns=("ml_signal",), top_n=2, max_position_weight=0.5)
    selected = [row["symbol"] for row in holdings if row["policy"] == "long_only_top_n_equal_weight" and row["rebalance_date"] == "2024-01-01"]
    assert selected == ["AAA", "BBB"]
    assert "ZZZ" not in {row["symbol"] for row in holdings}
    assert payload["training_performed"] is False
    assert summary


def test_equal_weights_turnover_costs_and_caps():
    summary, curves, holdings, _ = _build()
    selected = [row for row in holdings if row["strategy_id"] == "ml_signal|long_only_top_n_equal_weight"]
    for rebalance_date in {row["rebalance_date"] for row in selected}:
        assert sum(row["weight"] for row in selected if row["rebalance_date"] == rebalance_date) == 1.0
    row = next(row for row in summary if row["strategy_id"] == "ml_signal|long_only_top_n_equal_weight")
    assert row["max_position_weight"] == 0.5
    assert row["transaction_cost_drag"] > 0
    assert row["net_return"] < row["gross_return"]
    first = next(row for row in curves if row["strategy_id"] == row["strategy_id"] and row["signal_column"] == "ml_signal" and row["policy"] == "long_only_top_n_equal_weight")
    assert first["turnover"] == 1.0
    assert first["benchmark_return"] == 0.01


def test_replay_rejects_missing_benchmark_return():
    rows = _rows()
    for row in rows:
        row["actual_benchmark_return_10d"] = ""

    with pytest.raises(ValueError, match="Missing benchmark return"):
        build_stock_level_portfolio_replay(
            rows,
            benchmark={"walk_forward": {"out_of_sample_only": True}},
            signal_columns=("ml_signal",),
            top_n=2,
            max_position_weight=0.5,
        )


def test_replay_rejects_conflicting_benchmark_returns_for_date():
    rows = _rows()
    rows[0]["actual_benchmark_return_10d"] = 0.02

    with pytest.raises(ValueError, match="Benchmark return must be identical"):
        build_stock_level_portfolio_replay(
            rows,
            benchmark={"walk_forward": {"out_of_sample_only": True}},
            signal_columns=("ml_signal",),
            top_n=2,
            max_position_weight=0.5,
        )


def test_replay_rejects_conflicting_benchmark_target_timestamps_for_date():
    rows = _rows()
    for row in rows:
        row["benchmark_label_end_timestamp"] = "2024-01-11"
        row["benchmark_label_available_timestamp"] = "2024-01-12"
    rows[0]["benchmark_label_end_timestamp"] = "2024-01-12"

    with pytest.raises(ValueError, match="Benchmark target timestamps must be identical"):
        build_stock_level_portfolio_replay(
            rows,
            benchmark={"walk_forward": {"out_of_sample_only": True}},
            signal_columns=("ml_signal",),
            top_n=2,
            max_position_weight=0.5,
        )


def test_long_short_has_expected_exposure():
    _, _, holdings, _ = _build(allow_short=True)
    rows = [row for row in holdings if row["strategy_id"] == "ml_signal|long_short_top_bottom_decile_equal_weight" and row["rebalance_date"] == "2024-01-01"]
    assert sum(row["weight"] for row in rows) == 0.0
    assert sum(abs(row["weight"]) for row in rows) == 1.0


def test_writer_creates_all_artifacts(tmp_path):
    predictions = tmp_path / "predictions.csv"
    fields = list(_rows()[0])
    predictions.write_text(",".join(fields) + "\n" + "\n".join(",".join(str(row[field]) for field in fields) for row in _rows()), encoding="utf-8")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"walk_forward": {"out_of_sample_only": True}}), encoding="utf-8")
    paths = write_stock_level_portfolio_replay({"ml": {"output_dir": str(tmp_path), "stock_level_model_oos_predictions_path": str(predictions), "stock_level_model_ranking_benchmark_path": str(benchmark), "stock_portfolio_replay_signal_columns": ["ml_signal", "predicted_momentum_120d"], "stock_portfolio_replay_top_n": 2, "stock_portfolio_replay_max_position_weight": 0.5}})
    assert all(path.exists() for path in (paths.csv_path, paths.json_path, paths.markdown_path, paths.equity_curves_path, paths.holdings_path))
    payload = json.loads(paths.json_path.read_text())
    assert payload["promotion_thresholds_changed"] is False
    latest = json.loads((tmp_path / "latest_completed.json").read_text())
    run_dir = tmp_path / "runs" / latest["run_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["kind"] == "stock_selector_replay"
    assert manifest["run_status"] == "complete"
    assert (run_dir / paths.equity_curves_path.name).exists()


def test_stock_selector_rebalance_dataset_preserves_replay_artifacts(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)

    rows, metadata = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
    )

    assert [row["rebalance_date"] for row in rows] == ["2024-01-01", "2024-01-11"]
    assert [row["selected_symbols"] for row in rows] == ["AAA,BBB", "AAA,CCC"]
    assert json.loads(rows[0]["selected_weights"]) == {"AAA": 0.6, "BBB": 0.4}
    assert rows[0]["portfolio_return_next_period"] == 0.02
    assert rows[0]["benchmark_return_next_period"] == 0.005
    assert rows[0]["champion_excess_return"] == pytest.approx(0.015)
    assert rows[0]["transaction_cost_drag"] == 0.001
    assert rows[1]["turnover"] == 0.8
    assert rows[0]["feature_id"].startswith("stock_selector_")
    assert rows[0]["feature_id"] != rows[1]["feature_id"]
    assert rows[0]["feature_date"] < rows[0]["label_start_date"] <= rows[0]["label_end_date"]
    assert metadata["selected_signal"] == "ml_signal"
    assert metadata["selected_policy"] == "long_only_top_n_equal_weight"
    assert metadata["source_dataset_hash"] == "source-hash-1"
    assert metadata["source_type"] == SELECTOR_DERIVED_SOURCE_TYPE
    contract = metadata["input_source_contract"]
    assert contract["selector_dataset_identity"] == "source-hash-1"
    assert contract["strict_oos_fold_identity"]["fold_ids"] == ["1", "2"]
    assert contract["holdings_and_weights_checksum"]
    assert contract["return_lineage"]["field"] == "net_return"
    assert contract["benchmark_lineage"]["field"] == "benchmark_return"
    assert contract["exposure_label_contract"] == "should_reduce_exposure"


def test_stock_selector_rebalance_uses_real_benchmark_for_relative_labels(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    equity_rows = list(csv.DictReader(paths["equity"].open("r", encoding="utf-8", newline="")))
    for row in equity_rows:
        if row["signal_column"] == "ml_signal" and row["rebalance_date"] == "2024-01-01":
            row["net_return"] = "0.02"
            row["gross_return"] = "0.021"
            row["benchmark_return"] = "0.03"
            row["equity"] = "1.02"
        if row["signal_column"] == "ml_signal" and row["rebalance_date"] == "2024-01-11":
            row["net_return"] = "-0.01"
            row["gross_return"] = "-0.009"
            row["benchmark_return"] = "-0.03"
            row["equity"] = "1.0098"
    _write_csv(paths["equity"], equity_rows)

    rows, _ = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
    )

    assert rows[0]["portfolio_return_next_period"] == 0.02
    assert rows[0]["benchmark_return_next_period"] == 0.03
    assert rows[0]["champion_excess_return"] == pytest.approx(-0.01)
    assert rows[0]["underperforms_spy"] == 1
    assert rows[1]["portfolio_return_next_period"] == -0.01
    assert rows[1]["benchmark_return_next_period"] == -0.03
    assert rows[1]["champion_excess_return"] == pytest.approx(0.02)
    assert rows[1]["underperforms_spy"] == 0


def test_stock_selector_rebalance_dataset_satisfies_label_and_leakage_contract(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    rows, _ = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
    )

    labels = ShouldReduceExposureLabelBuilder().build(rows)
    dataset = build_dataset(rows, labels.rows, "should_reduce_exposure")

    assert labels.label_name == "should_reduce_exposure"
    assert labels.rows[0]["feature_id"] == rows[0]["feature_id"]
    assert dataset.sample_count == 2
    assert "future_max_drawdown" not in dataset.features[0]
    assert "portfolio_return_next_period" not in dataset.features[0]
    assert "portfolio_gross_return_next_period" not in dataset.features[0]
    assert "turnover" in dataset.features[0]
    assert dataset_leakage_audit(dataset)["leakage_check_passed"] is True


def test_stock_selector_rebalance_dataset_rejects_unknown_signal_and_policy(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)

    with pytest.raises(ValueError, match="Unknown stock selector signal"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="missing_signal",
            selected_policy="long_only_top_n_equal_weight",
        )

    with pytest.raises(ValueError, match="Unknown stock selector portfolio policy"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="missing_policy",
        )


def test_stock_selector_rebalance_dataset_rejects_missing_artifact_and_duplicate_holdings(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing stock selector rebalance source artifacts"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=tmp_path / "missing.csv",
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )

    _append_csv_row(paths["holdings"], {
        "rebalance_date": "2024-01-01",
        "strategy_id": "ml_signal|long_only_top_n_equal_weight",
        "signal_column": "ml_signal",
        "policy": "long_only_top_n_equal_weight",
        "symbol": "AAA",
        "weight": "0.1",
        "side": "long",
    })
    with pytest.raises(ValueError, match="Duplicate holding"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )


def test_stock_selector_rebalance_dataset_rejects_missing_benchmark_return(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    rows = list(csv.DictReader(paths["equity"].open("r", encoding="utf-8", newline="")))
    for row in rows:
        row["benchmark_return"] = ""
    _write_csv(paths["equity"], rows)

    with pytest.raises(ValueError, match="Missing benchmark_return"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )


def test_stock_selector_rebalance_rejects_final_fit_and_missing_fold_lineage(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    prediction_rows = list(csv.DictReader(paths["predictions"].open("r", encoding="utf-8", newline="")))
    prediction_rows[0]["prediction_scope"] = "final_fit"
    _write_csv(paths["predictions"], prediction_rows)

    with pytest.raises(ValueError, match="Final-fit selector predictions"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )

    paths = _write_selector_rebalance_source_artifacts(tmp_path / "missing-fold")
    prediction_rows = list(csv.DictReader(paths["predictions"].open("r", encoding="utf-8", newline="")))
    prediction_rows[0]["fold_id"] = ""
    _write_csv(paths["predictions"], prediction_rows)

    with pytest.raises(ValueError, match="require fold_id"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )


def test_selector_derived_input_resolution_rejects_legacy_and_validates_contract(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    output = tmp_path / "stock_selector_rebalance_dataset.csv"
    metadata = tmp_path / "stock_selector_rebalance_dataset.json"
    write_stock_selector_rebalance_dataset({
        "ml": {
            "output_dir": str(tmp_path),
            "stock_selector_rebalance_source_dir": str(tmp_path),
            "stock_selector_rebalance_predictions_path": str(paths["predictions"]),
            "stock_selector_rebalance_selected_signal": "ml_signal",
            "stock_selector_rebalance_selected_policy": "long_only_top_n_equal_weight",
            "stock_selector_rebalance_dataset_path": str(output),
            "stock_selector_rebalance_metadata_path": str(metadata),
        }
    })

    identity = validate_exposure_input_resolution({
        "ml": {
            "label_type": "should_reduce_exposure",
            "feature_set": "selector_derived_rebalance_v1",
            "exposure_production_campaign": True,
            "exposure_input_source_type": SELECTOR_DERIVED_SOURCE_TYPE,
            "stock_selector_rebalance_dataset_path": str(output),
        }
    })

    assert identity["source_type"] == SELECTOR_DERIVED_SOURCE_TYPE
    assert identity["input_source_contract"]["dataset_identity"]

    with pytest.raises(RuntimeError, match="cannot resolve legacy"):
        validate_exposure_input_resolution({
            "ml": {
                "feature_set": "selector_derived_rebalance_v1",
                "exposure_production_campaign": True,
                "exposure_input_source_type": SELECTOR_DERIVED_SOURCE_TYPE,
                "stock_selector_rebalance_dataset_path": str(tmp_path / "expanded_rebalance_dataset.csv"),
            }
        })


def test_legacy_exposure_input_must_be_explicitly_labelled(tmp_path):
    with pytest.raises(RuntimeError, match="explicitly labelled"):
        validate_exposure_input_resolution({
            "ml": {
                "label_type": "should_reduce_exposure",
                "feature_set": "expanded_rebalance_v1",
                "expanded_rebalance_dataset_path": str(tmp_path / "expanded_rebalance_dataset.csv"),
            }
        })

    identity = validate_exposure_input_resolution({
        "ml": {
            "label_type": "should_reduce_exposure",
            "feature_set": "expanded_rebalance_v1",
            "legacy_research_exposure_input": True,
            "expanded_rebalance_dataset_path": str(tmp_path / "expanded_rebalance_dataset.csv"),
        }
    })
    assert identity["source_type"] == "legacy_expanded_rebalance_v1"


def test_selector_derived_dataset_identity_changes_with_policy(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    first, first_meta = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
    )
    second, second_meta = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
        cost_bps=99,
    )

    assert first == second
    assert (
        first_meta["input_source_contract"]["dataset_identity"]
        != second_meta["input_source_contract"]["dataset_identity"]
    )


def test_stock_selector_rebalance_feature_ids_are_row_order_stable(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    first, _ = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
    )
    equity_rows = list(csv.DictReader(paths["equity"].open("r", encoding="utf-8", newline="")))
    _write_csv(paths["equity"], list(reversed(equity_rows)))
    second, _ = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=paths["predictions"],
        summary_path=paths["summary"],
        equity_curves_path=paths["equity"],
        holdings_path=paths["holdings"],
        selected_signal="ml_signal",
        selected_policy="long_only_top_n_equal_weight",
    )

    assert [row["feature_id"] for row in first] == [row["feature_id"] for row in second]


def test_stock_selector_rebalance_rejects_unmatched_holdings_and_duplicate_outcomes(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    _append_csv_row(paths["holdings"], {
        "rebalance_date": "2024-01-31",
        "strategy_id": "ml_signal|long_only_top_n_equal_weight",
        "signal_column": "ml_signal",
        "policy": "long_only_top_n_equal_weight",
        "symbol": "ZZZ",
        "weight": "0.1",
        "side": "long",
    })
    with pytest.raises(ValueError, match="Holdings have no matching selected equity outcomes"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )

    paths = _write_selector_rebalance_source_artifacts(tmp_path / "dupes")
    _append_csv_row(paths["equity"], {
        "rebalance_date": "2024-01-01",
        "strategy_id": "ml_signal|long_only_top_n_equal_weight",
        "signal_column": "ml_signal",
        "policy": "long_only_top_n_equal_weight",
        "gross_return": "0.02",
        "transaction_cost_drag": "0.001",
        "net_return": "0.019",
        "turnover": "0.5",
        "equity": "1.019",
        "benchmark_return": "0.004",
    })
    with pytest.raises(ValueError, match="Duplicate equity curve outcome"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )


def test_stock_selector_rebalance_writer_outputs_dataset_and_metadata(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    output = tmp_path / "stock_selector_rebalance_dataset.csv"
    metadata = tmp_path / "stock_selector_rebalance_dataset.json"

    result = write_stock_selector_rebalance_dataset({
        "ml": {
            "output_dir": str(tmp_path),
            "stock_selector_rebalance_source_dir": str(tmp_path),
            "stock_selector_rebalance_predictions_path": str(paths["predictions"]),
            "stock_selector_rebalance_selected_signal": "ml_signal",
            "stock_selector_rebalance_selected_policy": "long_only_top_n_equal_weight",
            "stock_selector_rebalance_dataset_path": str(output),
            "stock_selector_rebalance_metadata_path": str(metadata),
        }
    })

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(metadata.read_text(encoding="utf-8"))

    assert result.row_count == 2
    assert rows[0]["dataset_hash"] == payload["dataset_hash"]
    assert rows[0]["dataset_hash"] == rows[1]["dataset_hash"]
    assert payload["training_performed"] is False
    assert output.name == "stock_selector_rebalance_dataset.csv"


def test_stock_selector_rebalance_writer_reuses_compatible_immutable_dataset(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    output = tmp_path / "stock_selector_rebalance_dataset.csv"
    metadata = tmp_path / "stock_selector_rebalance_dataset.json"
    config = {
        "ml": {
            "output_dir": str(tmp_path),
            "stock_selector_rebalance_source_dir": str(tmp_path),
            "stock_selector_rebalance_predictions_path": str(paths["predictions"]),
            "stock_selector_rebalance_selected_signal": "ml_signal",
            "stock_selector_rebalance_selected_policy": "long_only_top_n_equal_weight",
            "stock_selector_rebalance_dataset_path": str(output),
            "stock_selector_rebalance_metadata_path": str(metadata),
        }
    }

    write_stock_selector_rebalance_dataset(config)
    first_payload = json.loads(metadata.read_text(encoding="utf-8"))
    write_stock_selector_rebalance_dataset(config)
    second_payload = json.loads(metadata.read_text(encoding="utf-8"))

    assert first_payload == second_payload
    assert_stock_selector_rebalance_dataset_reuse_compatible(
        tmp_path,
        first_payload["dataset_hash"],
    )


def test_stock_selector_rebalance_writer_rejects_incompatible_existing_output(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    output = tmp_path / "stock_selector_rebalance_dataset.csv"
    metadata = tmp_path / "stock_selector_rebalance_dataset.json"
    write_stock_selector_rebalance_dataset({
        "ml": {
            "output_dir": str(tmp_path),
            "stock_selector_rebalance_source_dir": str(tmp_path),
            "stock_selector_rebalance_predictions_path": str(paths["predictions"]),
            "stock_selector_rebalance_selected_signal": "ml_signal",
            "stock_selector_rebalance_selected_policy": "long_only_top_n_equal_weight",
            "stock_selector_rebalance_dataset_path": str(output),
            "stock_selector_rebalance_metadata_path": str(metadata),
        }
    })
    replacement_paths = _write_selector_rebalance_source_artifacts(tmp_path / "replacement")
    replacement_rows = list(csv.DictReader(replacement_paths["predictions"].open("r", encoding="utf-8", newline="")))
    for row in replacement_rows:
        row["dataset_hash"] = "source-hash-2"
    _write_csv(replacement_paths["predictions"], replacement_rows)

    with pytest.raises(RuntimeError, match="reuse identity mismatch"):
        write_stock_selector_rebalance_dataset({
            "ml": {
                "output_dir": str(tmp_path),
                "stock_selector_rebalance_source_dir": str(replacement_paths["summary"].parent),
                "stock_selector_rebalance_predictions_path": str(replacement_paths["predictions"]),
                "stock_selector_rebalance_selected_signal": "ml_signal",
                "stock_selector_rebalance_selected_policy": "long_only_top_n_equal_weight",
                "stock_selector_rebalance_dataset_path": str(output),
                "stock_selector_rebalance_metadata_path": str(metadata),
            }
        })


def test_stock_selector_rebalance_requires_selector_dataset_identity(tmp_path):
    paths = _write_selector_rebalance_source_artifacts(tmp_path)
    prediction_rows = list(csv.DictReader(paths["predictions"].open("r", encoding="utf-8", newline="")))
    for row in prediction_rows:
        row["dataset_hash"] = ""
    _write_csv(paths["predictions"], prediction_rows)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary.pop("dataset_hash")
    paths["summary"].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="requires selector dataset identity"):
        build_stock_selector_rebalance_dataset_from_artifacts(
            predictions_path=paths["predictions"],
            summary_path=paths["summary"],
            equity_curves_path=paths["equity"],
            holdings_path=paths["holdings"],
            selected_signal="ml_signal",
            selected_policy="long_only_top_n_equal_weight",
        )


def test_replay_has_no_operational_imports():
    source = inspect.getsource(stock_level_portfolio_replay)
    assert all(token not in source for token in ("core.interfaces.broker", "core.paper", "core.entities.order", "paper_trading"))


def _write_selector_rebalance_source_artifacts(tmp_path):
    predictions = tmp_path / "stock_level_model_oos_predictions.csv"
    _write_csv(predictions, [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "fold_id": "1", "ml_signal": "0.9", "dataset_hash": "source-hash-1"},
        {"rebalance_date": "2024-01-01", "symbol": "BBB", "fold_id": "1", "ml_signal": "0.8", "dataset_hash": "source-hash-1"},
        {"rebalance_date": "2024-01-11", "symbol": "AAA", "fold_id": "2", "ml_signal": "0.7", "dataset_hash": "source-hash-1"},
        {"rebalance_date": "2024-01-11", "symbol": "CCC", "fold_id": "2", "ml_signal": "0.6", "dataset_hash": "source-hash-1"},
    ])
    summary = tmp_path / "stock_level_portfolio_replay_summary.json"
    summary.write_text(json.dumps({
        "walk_forward": {"out_of_sample_only": True},
        "signal_columns": ["ml_signal"],
        "policies": ["long_only_top_n_equal_weight"],
        "dataset_hash": "summary-hash",
    }), encoding="utf-8")
    equity = tmp_path / "stock_level_portfolio_replay_equity_curves.csv"
    _write_csv(equity, [
        {"rebalance_date": "2024-01-01", "strategy_id": "ml_signal|long_only_top_n_equal_weight", "signal_column": "ml_signal", "policy": "long_only_top_n_equal_weight", "gross_return": "0.021", "transaction_cost_drag": "0.001", "net_return": "0.02", "turnover": "1.0", "equity": "1.02", "benchmark_return": "0.005"},
        {"rebalance_date": "2024-01-11", "strategy_id": "ml_signal|long_only_top_n_equal_weight", "signal_column": "ml_signal", "policy": "long_only_top_n_equal_weight", "gross_return": "-0.009", "transaction_cost_drag": "0.001", "net_return": "-0.01", "turnover": "0.8", "equity": "1.0098", "benchmark_return": "0.02"},
        {"rebalance_date": "2024-01-01", "strategy_id": "other_signal|long_only_top_n_equal_weight", "signal_column": "other_signal", "policy": "long_only_top_n_equal_weight", "gross_return": "0.5", "transaction_cost_drag": "0", "net_return": "0.5", "turnover": "1", "equity": "1.5", "benchmark_return": "0.01"},
    ])
    holdings = tmp_path / "stock_level_portfolio_replay_holdings.csv"
    _write_csv(holdings, [
        {"rebalance_date": "2024-01-01", "strategy_id": "ml_signal|long_only_top_n_equal_weight", "signal_column": "ml_signal", "policy": "long_only_top_n_equal_weight", "symbol": "BBB", "weight": "0.4", "side": "long"},
        {"rebalance_date": "2024-01-01", "strategy_id": "ml_signal|long_only_top_n_equal_weight", "signal_column": "ml_signal", "policy": "long_only_top_n_equal_weight", "symbol": "AAA", "weight": "0.6", "side": "long"},
        {"rebalance_date": "2024-01-11", "strategy_id": "ml_signal|long_only_top_n_equal_weight", "signal_column": "ml_signal", "policy": "long_only_top_n_equal_weight", "symbol": "CCC", "weight": "0.25", "side": "long"},
        {"rebalance_date": "2024-01-11", "strategy_id": "ml_signal|long_only_top_n_equal_weight", "signal_column": "ml_signal", "policy": "long_only_top_n_equal_weight", "symbol": "AAA", "weight": "0.75", "side": "long"},
    ])
    return {"predictions": predictions, "summary": summary, "equity": equity, "holdings": holdings}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv_row(path, row):
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writerow(row)
