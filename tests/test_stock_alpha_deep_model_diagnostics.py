import inspect
import json
import math

import pytest

from config.config_loader import load_config
from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.stock_alpha_deep_model_diagnostics import REQUIRED_FIELDS, diagnose_sequence_fold, write_stock_alpha_deep_model_diagnostics
from core.research.ml.stock_level.stock_alpha_model_sets import FULL_SEQUENCE_MODELS, MODEL_SETS, TARGET_MODEL_SETS, TABULAR_MODELS
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_sequence_regressors import SequenceRegressorConfig, TorchSequenceReturnRegressor
from core.research.ml.stock_level_benchmark_types import AUXILIARY_TARGET_COLUMNS, MODEL_NAMES, SEQUENCE_MODEL_NAMES, TABULAR_MODEL_NAMES


def _rows(count=4):
    return [{"rebalance_date": f"2024-01-{i + 1:02d}", "symbol": "AAA", "feature": float(i), "actual_forward_return_10d": 0.1} for i in range(count)]


def _diagnose(all_rows, train, test, predictor, **kwargs):
    return diagnose_sequence_fold(model_name=kwargs.get("model_name", "dlinear"), model_set="diagnostic_single_model", run_size="dev", target_column="actual_forward_return_10d", fold_id=0, all_rows=all_rows, train_rows=train, test_rows=test, feature_columns=("feature",), sequence_length=kwargs.get("sequence_length", 2), predictor=predictor, unavailable_reason=kwargs.get("unavailable_reason"))


def test_no_windows_created_and_required_fields():
    rows = _rows(1); result = _diagnose(rows, rows, rows, lambda *_: [])
    assert result["skip_reason"] == "no_sequence_windows_created"
    assert set(REQUIRED_FIELDS) <= set(result)


def test_predictions_all_nan():
    rows = _rows(); result = _diagnose(rows, rows[:3], rows[3:], lambda _a, test, _b: [math.nan] * len(test))
    assert result["skip_reason"] == "predictions_all_nan_after_model"
    assert result["finite_prediction_count"] == 0


def test_merge_back_mismatch():
    rows = _rows(); result = _diagnose(rows, rows[:3], rows[3:], lambda _a, _b, _c: ([1.0], [("2099-01-01", "ZZZ")]))
    assert result["skip_reason"] == "prediction_merge_failed"


def test_successful_finite_predictions():
    rows = _rows(); result = _diagnose(rows, rows[:3], rows[3:], lambda _a, test, _b: [1.0] * len(test))
    assert result["skip_reason"] is None
    assert result["finite_prediction_count"] == 1
    assert result["merge_matched_rows"] == 1


def test_prediction_count_mismatch_is_reported():
    rows = _rows()
    result = _diagnose(rows, rows[:3], rows[3:], lambda *_: [])
    assert result["skip_reason"] == "prediction_count_mismatch"
    assert "0 predictions for 1 sequence test windows" in result["error_message"]


def test_diagnostics_report_non_finite_sequence_inputs():
    rows = _rows()
    rows[1]["feature"] = math.nan
    result = _diagnose(
        rows,
        rows[:3],
        rows[3:],
        lambda _train, test, _rows: (
            [1.0 for _ in test],
            None,
            {
                "x_train_non_finite_count_after_preprocessing": 1,
                "x_test_non_finite_count_after_preprocessing": 0,
            },
        ),
    )
    assert result["feature_nan_count_before_imputation"] > 0
    assert result["skip_reason"] == "non_finite_sequence_inputs"


def test_required_fields_cover_all_diagnostic_row_keys():
    rows = _rows()
    result = _diagnose(
        rows,
        rows[:3],
        rows[3:],
        lambda _train, test, _rows: (
            [1.0 for _ in test],
            None,
            {
                "x_train_finite_ratio_after_preprocessing": 1.0,
                "x_train_non_finite_count": 0,
                "x_train_non_finite_count_after_preprocessing": 0,
                "x_test_non_finite_count": 0,
                "x_test_finite_ratio_after_preprocessing": 1.0,
                "feature_imputation_strategy": "train_fold_median_else_zero",
                "x_test_non_finite_count_after_preprocessing": 0,
                "y_train_non_finite_count": 0,
            },
        ),
    )
    assert set(result) <= set(REQUIRED_FIELDS)


def test_dlinear_sequence_regressor_imputes_nan_features_and_predicts_finite():
    pytest.importorskip("torch")
    model = TorchSequenceReturnRegressor(
        SequenceRegressorConfig(
            architecture="dlinear",
            sequence_length=2,
            epochs=1,
            batch_size=2,
            random_seed=7,
        )
    )
    train_sequences = [
        [[1.0, math.nan], [2.0, 4.0]],
        [[2.0, 8.0], [3.0, math.nan]],
        [[3.0, 12.0], [4.0, 16.0]],
    ]
    model.fit(train_sequences, [0.1, 0.2, 0.3])
    values = model.predict([[[1000.0, math.nan], [2000.0, 24.0]]])

    assert len(values) == 1
    assert math.isfinite(values[0])
    assert model.diagnostics["feature_nan_count_before_imputation"] == 2
    assert model.diagnostics["feature_nan_count_after_imputation"] == 0
    assert model.diagnostics["test_feature_nan_count_before_imputation"] == 1
    assert model.diagnostics["test_feature_nan_count_after_imputation"] == 0
    assert model.diagnostics["postprocessed_finite_prediction_count"] == 1
    assert model.diagnostics["train_loss_finite"] is True
    assert model.diagnostics["model_parameters_finite"] is True


def test_sequence_imputation_uses_train_only_statistics():
    pytest.importorskip("torch")
    model = TorchSequenceReturnRegressor(
        SequenceRegressorConfig(
            architecture="dlinear",
            sequence_length=2,
            epochs=1,
            batch_size=2,
            random_seed=7,
        )
    )
    train_sequences = [
        [[1.0], [2.0]],
        [[3.0], [math.nan]],
        [[5.0], [7.0]],
    ]
    model.fit(train_sequences, [0.1, 0.2, 0.3])
    model.predict([[[9999.0], [math.nan]]])

    assert float(model.feature_impute_values.reshape(-1)[0]) == 3.0


def test_unavailable_news_model():
    rows = _rows(); result = _diagnose(rows, rows[:3], rows[3:], None, model_name="news_analysis_transformer", unavailable_reason="no point-in-time news")
    assert result["skip_reason"] == "model_unavailable_by_design"


def test_dev_diagnostic_dlinear_config_is_dev_only_and_isolated():
    config = load_config("config/config.stock_alpha_dev_diagnostic_dlinear.yaml", overlay_project_config=True)
    settings = StockLevelResearchConfig.from_mapping(config)
    assert settings.run_size == "dev"
    assert settings.include_sequence_models is True
    assert settings.include_engineered_features is True
    assert config["ml"]["stock_deep_diagnostic_model"] == "dlinear"
    assert settings.dev_max_dates == 180
    assert settings.dev_max_symbols == 200
    assert stock_alpha_output_dir(config).as_posix().endswith("stock_alpha_deep_diagnostic/dev")
    assert settings.artifact_path.as_posix().endswith("stock_alpha_diagnostic/dev/stock_level_prediction_artifacts_enriched.csv")
    assert config["ml"]["stock_alpha_stages"]["portfolio_replay"] is False
    assert config["ml"]["stock_alpha_stages"]["portfolio_policy_sweep"] is False


def test_dev_diagnostic_model_configs_are_isolated():
    dlinear = load_config("config/config.stock_alpha_dev_diagnostic_dlinear.yaml", overlay_project_config=True)
    market = load_config("config/config.stock_alpha_dev_diagnostic_market_context_encoder.yaml", overlay_project_config=True)
    momentum = load_config("config/config.stock_alpha_dev_diagnostic_momentum_transformer.yaml", overlay_project_config=True)
    all_deep = load_config("config/config.stock_alpha_dev_diagnostic_all_deep.yaml", overlay_project_config=True)

    assert dlinear["ml"]["stock_deep_diagnostic_model"] == "dlinear"
    assert market["ml"]["stock_deep_diagnostic_model"] == "market_context_encoder"
    assert momentum["ml"]["stock_deep_diagnostic_model"] == "momentum_transformer"
    assert stock_alpha_output_dir(dlinear) == stock_alpha_output_dir(market) == stock_alpha_output_dir(momentum) == stock_alpha_output_dir(all_deep)
    assert "news_analysis_transformer" not in all_deep["ml"]["stock_deep_diagnostic_models"]
    assert {"dlinear", "patchtst", "transformer", "itransformer", "momentum_transformer", "multitask_transformer", "market_context_encoder", "temporal_fusion_transformer"} == set(all_deep["ml"]["stock_deep_diagnostic_models"])
    assert market["ml"]["stock_alpha_stages"]["portfolio_replay"] is False
    assert momentum["ml"]["stock_alpha_stages"]["portfolio_policy_sweep"] is False


def test_complete_stock_alpha_model_universe_is_covered():
    all_deep = load_config("config/config.stock_alpha_dev_diagnostic_all_deep.yaml", overlay_project_config=True)
    assert len(MODEL_NAMES) == 13
    assert tuple(TABULAR_MODEL_NAMES) == tuple(TABULAR_MODELS)
    assert tuple(SEQUENCE_MODEL_NAMES) == tuple(FULL_SEQUENCE_MODELS)
    assert tuple(MODEL_SETS["fast"]) == tuple(TABULAR_MODEL_NAMES)
    assert tuple(MODEL_SETS["standard"]) == (*TABULAR_MODEL_NAMES, "dlinear", "market_context_encoder")
    assert tuple(MODEL_SETS["full"]) == tuple(MODEL_NAMES)
    assert tuple(TARGET_MODEL_SETS["ultrafast"]) == ("ridge", "elastic_net")
    assert set(all_deep["ml"]["stock_deep_diagnostic_models"]) == set(SEQUENCE_MODEL_NAMES) - {"news_analysis_transformer"}


def test_write_deep_model_diagnostics_reports_fold_merge_details(tmp_path, monkeypatch):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    _write_artifact(artifact)

    class FakeSequenceModel:
        def fit(self, sequences, targets, auxiliary_targets=None):
            assert len(sequences) == len(targets)

        def predict(self, sequences):
            return [0.25 for _ in sequences]

    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_available_feature_columns", lambda _rows, include_engineered: ("feature",))
    monkeypatch.setattr(diagnostics, "_factories_for_model_set", lambda *args, **kwargs: ({}, {"dlinear": FakeSequenceModel}))

    paths = write_stock_alpha_deep_model_diagnostics(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "deep"),
                "stock_alpha_run_size": "dev",
                "stock_level_prediction_artifacts_path": str(artifact),
                "stock_ranker_include_engineered_features": True,
                "stock_deep_diagnostic_model": "dlinear",
                "stock_ranker_min_train_dates": 3,
                "stock_ranker_test_window_dates": 2,
                "stock_ranker_embargo_dates": 1,
                "stock_ranker_sequence_length": 2,
                "stock_alpha_dev_max_dates": 10,
                "stock_alpha_dev_max_symbols": 5,
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    row = payload["diagnostics"][0]
    assert payload["source_artifact_path"] == str(artifact)
    assert paths.json_path.parent.name == "dlinear"
    assert paths.json_path.parent.parent.name == "deep_diagnostics"
    assert row["sequence_train_windows"] > 0
    assert row["sequence_test_windows"] > 0
    assert row["finite_prediction_count"] == row["prediction_count"]
    assert row["merge_matched_rows"] == row["prediction_count"]
    assert row["output_prediction_column"] == "stock_level_predicted_forward_return_10d_dlinear"
    assert set(REQUIRED_FIELDS) <= set(row)


def test_deep_model_diagnostics_are_model_isolated_and_indexed(tmp_path, monkeypatch):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    _write_artifact(artifact)

    class FakeSequenceModel:
        def fit(self, sequences, targets, auxiliary_targets=None):
            pass

        def predict(self, sequences):
            return [0.25 for _ in sequences]

    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_available_feature_columns", lambda _rows, include_engineered: ("feature",))
    monkeypatch.setattr(
        diagnostics,
        "_factories_for_model_set",
        lambda *args, **kwargs: (
            {},
            {
                "dlinear": FakeSequenceModel,
                "market_context_encoder": FakeSequenceModel,
            },
        ),
    )
    base_config = {
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "deep"),
            "stock_alpha_run_size": "dev",
            "stock_level_prediction_artifacts_path": str(artifact),
            "stock_ranker_include_engineered_features": True,
            "stock_ranker_min_train_dates": 3,
            "stock_ranker_test_window_dates": 2,
            "stock_ranker_embargo_dates": 1,
            "stock_ranker_sequence_length": 2,
            "stock_alpha_dev_max_dates": 10,
            "stock_alpha_dev_max_symbols": 5,
        }
    }

    dlinear_paths = write_stock_alpha_deep_model_diagnostics(
        {**base_config, "ml": {**base_config["ml"], "stock_deep_diagnostic_model": "dlinear"}}
    )
    market_paths = write_stock_alpha_deep_model_diagnostics(
        {**base_config, "ml": {**base_config["ml"], "stock_deep_diagnostic_model": "market_context_encoder"}}
    )

    assert dlinear_paths.json_path != market_paths.json_path
    assert dlinear_paths.json_path.parent.name == "dlinear"
    assert market_paths.json_path.parent.name == "market_context_encoder"
    assert dlinear_paths.json_path.exists()
    assert market_paths.json_path.exists()
    index = json.loads((tmp_path / "deep" / "dev" / "stock_alpha_deep_model_diagnostics_index.json").read_text(encoding="utf-8"))
    assert {row["model_name"] for row in index["models"]} == {"dlinear", "market_context_encoder"}
    assert all(row["status"] == "passed" for row in index["models"])


def test_deep_model_diagnostics_multi_model_config_writes_each_model(tmp_path, monkeypatch):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    _write_artifact(artifact)

    class FakeSequenceModel:
        def fit(self, sequences, targets, auxiliary_targets=None):
            pass

        def predict(self, sequences):
            return [0.25 for _ in sequences]

    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_available_feature_columns", lambda _rows, include_engineered: ("feature",))
    monkeypatch.setattr(
        diagnostics,
        "_factories_for_model_set",
        lambda *args, **kwargs: ({}, {"dlinear": FakeSequenceModel, "patchtst": FakeSequenceModel}),
    )

    write_stock_alpha_deep_model_diagnostics(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "deep"),
                "stock_alpha_run_size": "dev",
                "stock_level_prediction_artifacts_path": str(artifact),
                "stock_ranker_include_engineered_features": True,
                "stock_deep_diagnostic_models": ["dlinear", "patchtst"],
                "stock_ranker_min_train_dates": 3,
                "stock_ranker_test_window_dates": 2,
                "stock_ranker_embargo_dates": 1,
                "stock_ranker_sequence_length": 2,
                "stock_alpha_dev_max_dates": 10,
                "stock_alpha_dev_max_symbols": 5,
            }
        }
    )

    root = tmp_path / "deep" / "dev" / "deep_diagnostics"
    assert (root / "dlinear" / "stock_alpha_deep_model_diagnostics.json").exists()
    assert (root / "patchtst" / "stock_alpha_deep_model_diagnostics.json").exists()
    index = json.loads((tmp_path / "deep" / "dev" / "stock_alpha_deep_model_diagnostics_index.json").read_text(encoding="utf-8"))
    assert {row["model_name"] for row in index["models"]} == {"dlinear", "patchtst"}


def test_deep_model_diagnostics_index_marks_pass_fail_and_unavailable(tmp_path):
    from core.research.ml.stock_level.stock_alpha_deep_model_diagnostics import _write_index

    root = tmp_path / "deep" / "dev" / "deep_diagnostics"
    _write_model_payload(root / "dlinear", "dlinear", [{"prediction_count": 2, "finite_prediction_count": 2, "sequence_train_windows": 1, "sequence_test_windows": 1, "merge_matched_rows": 2, "train_loss_finite": True, "model_parameters_finite": True, "skip_reason": None}])
    _write_model_payload(root / "patchtst", "patchtst", [{"prediction_count": 2, "finite_prediction_count": 0, "sequence_train_windows": 1, "sequence_test_windows": 1, "merge_matched_rows": 2, "train_loss_finite": False, "model_parameters_finite": True, "skip_reason": "loss_nan"}])
    _write_model_payload(root / "news_analysis_transformer", "news_analysis_transformer", [{"prediction_count": 0, "finite_prediction_count": 0, "sequence_train_windows": 0, "sequence_test_windows": 0, "merge_matched_rows": 0, "skip_reason": "model_unavailable_by_design"}])
    _write_model_payload(root / "multitask_transformer", "multitask_transformer", [{"prediction_count": 0, "finite_prediction_count": 0, "sequence_train_windows": 1, "sequence_test_windows": 1, "merge_matched_rows": 0, "skip_reason": "missing_auxiliary_targets"}])

    _write_index(tmp_path / "deep" / "dev")

    index = json.loads((tmp_path / "deep" / "dev" / "stock_alpha_deep_model_diagnostics_index.json").read_text(encoding="utf-8"))
    statuses = {row["model_name"]: row for row in index["models"]}
    assert statuses["dlinear"]["status"] == "passed"
    assert statuses["dlinear"]["finite_prediction_ratio"] == 1.0
    assert statuses["dlinear"]["sequence_windows_created"] is True
    assert statuses["dlinear"]["merge_back_matched"] is True
    assert statuses["patchtst"]["status"] == "failed"
    assert statuses["patchtst"]["skip_reasons"] == ["loss_nan"]
    assert statuses["news_analysis_transformer"]["status"] == "unavailable"
    assert statuses["multitask_transformer"]["status"] == "skipped_missing_required_inputs"
    assert index["model_coverage"]["total_model_count"] == 13
    assert index["model_coverage"]["tested_model_count"] == 4
    assert index["model_coverage"]["passed_model_count"] == 1
    assert index["model_coverage"]["failed_model_count"] == 1
    assert index["model_coverage"]["unavailable_model_count"] == 1
    assert index["model_coverage"]["skipped_missing_required_inputs_count"] == 1


def test_multitask_diagnostic_reports_missing_auxiliary_targets():
    rows = _rows()
    result = diagnose_sequence_fold(
        model_name="multitask_transformer",
        model_set="diagnostic_single_model",
        run_size="dev",
        target_column="actual_forward_return_10d",
        fold_id=0,
        all_rows=rows,
        train_rows=rows[:3],
        test_rows=rows[3:],
        feature_columns=("feature",),
        sequence_length=2,
        predictor=None,
        unavailable_reason="missing_auxiliary_targets: required=['actual_forward_return_5d'] available=['actual_forward_return_10d']",
        auxiliary_target_columns=AUXILIARY_TARGET_COLUMNS,
    )

    assert result["skip_reason"] == "missing_auxiliary_targets"
    assert result["auxiliary_target_columns"] == list(AUXILIARY_TARGET_COLUMNS)
    assert result["auxiliary_target_count"] == len(AUXILIARY_TARGET_COLUMNS)
    assert "required" in result["error_message"]


def test_multitask_diagnostic_passes_aligned_auxiliary_targets(tmp_path, monkeypatch):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    _write_artifact(artifact, auxiliary=True)

    class FakeMultitaskModel:
        def __init__(self):
            self.diagnostics = {"train_loss_finite": True, "model_parameters_finite": True}

        def fit(self, sequences, targets, auxiliary_targets=None):
            assert auxiliary_targets is not None
            assert len(auxiliary_targets) == len(sequences)
            assert len(auxiliary_targets[0]) == len(AUXILIARY_TARGET_COLUMNS)

        def predict(self, sequences):
            return [0.25 for _ in sequences]

    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_available_feature_columns", lambda _rows, include_engineered: ("feature",))
    monkeypatch.setattr(diagnostics, "_factories_for_model_set", lambda *args, **kwargs: ({}, {"multitask_transformer": FakeMultitaskModel}))
    paths = write_stock_alpha_deep_model_diagnostics(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "deep"),
                "stock_alpha_run_size": "dev",
                "stock_level_prediction_artifacts_path": str(artifact),
                "stock_ranker_include_engineered_features": True,
                "stock_deep_diagnostic_model": "multitask_transformer",
                "stock_ranker_min_train_dates": 3,
                "stock_ranker_test_window_dates": 2,
                "stock_ranker_embargo_dates": 1,
                "stock_ranker_sequence_length": 2,
                "stock_alpha_dev_max_dates": 10,
                "stock_alpha_dev_max_symbols": 5,
            }
        }
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    row = payload["diagnostics"][0]
    assert row["skip_reason"] is None
    assert row["auxiliary_alignment_status"] == "aligned"
    assert row["auxiliary_y_train_shape"][1] == len(AUXILIARY_TARGET_COLUMNS)
    assert row["finite_prediction_count"] == row["prediction_count"]


def test_news_transformer_diagnostic_unavailable_without_valid_features(tmp_path, monkeypatch):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    _write_artifact(artifact)

    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_available_feature_columns", lambda _rows, include_engineered: ("feature",))
    monkeypatch.setattr(diagnostics, "_factories_for_model_set", lambda *args, **kwargs: ({}, {"news_analysis_transformer": object}))
    paths = write_stock_alpha_deep_model_diagnostics(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "deep"),
                "stock_alpha_run_size": "dev",
                "stock_level_prediction_artifacts_path": str(artifact),
                "stock_ranker_include_engineered_features": True,
                "stock_deep_diagnostic_model": "news_analysis_transformer",
                "stock_ranker_min_train_dates": 3,
                "stock_ranker_test_window_dates": 2,
                "stock_ranker_embargo_dates": 1,
                "stock_ranker_sequence_length": 2,
                "stock_alpha_news_features_path": str(tmp_path / "missing_news_features.csv"),
                "stock_alpha_news_enable_transformer": False,
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
                "promotion_thresholds_changed": False,
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    row = payload["diagnostics"][0]
    assert row["skip_reason"] == "model_unavailable_by_design"
    assert row["error_message"] == "stock_alpha_news_enable_transformer_false"


def test_generic_news_columns_do_not_unlock_news_transformer(tmp_path, monkeypatch):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    _write_artifact(artifact, generic_news=True)
    generic_features = tmp_path / "generic_news_features.csv"
    generic_features.write_text("rebalance_date,symbol,news_random_score\n2024-01-01,AAA,1\n", encoding="utf-8")

    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_available_feature_columns", lambda _rows, include_engineered: ("feature", "news_random_score"))
    monkeypatch.setattr(diagnostics, "_factories_for_model_set", lambda *args, **kwargs: ({}, {"news_analysis_transformer": object}))
    paths = write_stock_alpha_deep_model_diagnostics(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "deep"),
                "stock_alpha_run_size": "dev",
                "stock_level_prediction_artifacts_path": str(artifact),
                "stock_ranker_include_engineered_features": True,
                "stock_deep_diagnostic_model": "news_analysis_transformer",
                "stock_ranker_min_train_dates": 3,
                "stock_ranker_test_window_dates": 2,
                "stock_ranker_embargo_dates": 1,
                "stock_ranker_sequence_length": 2,
                "stock_alpha_news_features_path": str(generic_features),
                "stock_alpha_news_enable_transformer": True,
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
                "promotion_thresholds_changed": False,
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    row = payload["diagnostics"][0]
    assert row["skip_reason"] == "model_unavailable_by_design"
    assert row["error_message"] == "missing_required_news_feature_columns"


def test_diagnostics_module_has_no_operational_imports():
    from core.research.ml.stock_level import stock_alpha_deep_model_diagnostics
    source = inspect.getsource(stock_alpha_deep_model_diagnostics)
    assert all(term not in source for term in ("paper_trading", "live_trading", "infrastructure.broker", "core.entities.order"))


def _write_artifact(path, *, auxiliary=False, generic_news=False):
    extra_header = ",actual_forward_return_5d,actual_future_volatility,actual_future_drawdown" if auxiliary else ""
    extra_row = ",0.05,0.10,-0.20" if auxiliary else ""
    if generic_news:
        extra_header += ",news_random_score,sentiment_blob"
        extra_row += ",1.0,0.5"
    path.write_text(
        "\n".join(
            [
                f"rebalance_date,symbol,feature,actual_forward_return_10d,predicted_momentum_120d,predicted_risk_adjusted_momentum{extra_header}",
                *[
                    f"2024-01-{date_index + 1:02d},AAA,{date_index / 10},0.1,0.2,0.3{extra_row}"
                    for date_index in range(8)
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_model_payload(path, model_name, diagnostics):
    path.mkdir(parents=True)
    (path / "stock_alpha_deep_model_diagnostics.json").write_text(
        json.dumps({"model_name": model_name, "fold_count": len(diagnostics), "diagnostics": diagnostics}),
        encoding="utf-8",
    )
