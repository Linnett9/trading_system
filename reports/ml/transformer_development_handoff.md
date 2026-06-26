# Transformer / Meta-Ensemble Development Handoff

Research-only context for future transformer, ranking, calibration, and meta-ensemble work. Do not edit broker, paper trading, live trading, or execution code for this track.

## Current Research Flow

1. `main.py --mode ml-research --config configs/research/<experiment>.yaml`
2. `application/services/ml_commands.py::run_ml_research`
3. `core/research/ml/experiment_runner.py::MLExperimentRunner.run`
4. Feature rows -> labels -> `MLDataset` -> chronological holdout / walk-forward -> model registry -> metrics, calibration, prediction artifacts, overlays, HTML report.
5. `main.py --mode ml-meta-ensemble --config configs/research/regime_transformer_meta_ensemble_v1.yaml`
6. `core/research/ml/meta_ensemble.py::run_meta_ensemble` merges source `prediction_artifacts.csv` files, trains the meta model, writes metrics and leaderboard.

## File Map

| Path | Purpose | Key classes/functions | Important config keys |
|---|---|---|---|
| `core/research/ml/transformer_model.py` | Baseline PyTorch sequence transformer classifier. | `TransformerSequenceMLModel`, `TransformerTrainingSummary`, `_make_tiny_transformer_classifier` | `ml.model_type=transformer`, `sequence_length`, `transformer_d_model`, `transformer_heads`, `transformer_layers`, `transformer_feedforward`, `transformer_dropout`, `transformer_epochs`, `transformer_batch_size`, `transformer_learning_rate`, `transformer_weight_decay`, `transformer_device` |
| `core/research/ml/patchtst_model.py` | PatchTST-style sequence classifier. | `PatchTSTSequenceMLModel`, `_build_patchtst_module` | `ml.model_type=patchtst`, `patchtst_sequence_length`, `patchtst_patch_length`, `patchtst_patch_stride`, `patchtst_d_model`, `patchtst_heads`, `patchtst_layers`, `patchtst_feedforward`, `patchtst_dropout`, `patchtst_epochs`, `patchtst_batch_size`, `patchtst_learning_rate`, `patchtst_weight_decay`, `patchtst_device`, `patchtst_pos_weight` |
| `core/research/ml/dlinear_model.py` | DLinear sequence classifier baseline. | `DLinearSequenceMLModel` | `ml.model_type=dlinear`, `dlinear_sequence_length`, `dlinear_epochs`, `dlinear_batch_size`, `dlinear_learning_rate`, `dlinear_weight_decay`, `dlinear_device`, `dlinear_pos_weight` |
| `core/research/ml/models.py` | ML model interface and registry/factory. | `IMLModel`, `LogisticRegressionMLModel`, `TreeClassifierMLModel`, `build_ml_model` | `ml.model_type`, `ml.shadow_model_type`, `ml.random_seed`, class-weight settings, model-specific hyperparameters |
| `core/research/ml/sequence_dataset.py` | Converts chronological tabular rows into rolling sequence datasets. | `SequenceMLDataset`, `build_sequence_dataset` | `sequence_length`, model-specific sequence lengths |
| `core/research/ml/config.py` | Typed experiment config facade. | `MLExperimentConfig.from_config` | `model_type`, `feature_set`, `label_type`, `prediction_horizon`, `label_horizon_days`, `decision_threshold`, `test_fraction`, `walk_forward_folds`, `random_seed`, `output_dir` |
| `core/research/ml/experiment_runner.py` | Main research orchestration. Builds data, trains model, writes artifacts, comparisons, calibration, overlays, HTML report. | `MLExperimentRunner`, `MLExperimentResult`, `run`, `build_expanded_rebalance_dataset` | `ml.feature_set`, `label_type`, `prediction_target`, `benchmark_symbols`, `comparison_models`, `overlay_comparison_models`, `shadow_thresholds`, `shadow_reduced_exposures`, `decision_threshold`, `calibration_bin_count`, `expanded_rebalance_dataset`, all model hyperparameters |
| `core/research/ml/features.py` | Market/regime feature generation. | `HistoricalFeatureBuilder`, `add_champion_state_features`, `write_feature_rows` | `benchmark_symbols`, historical provider settings, feature set assumptions |
| `core/research/ml/labels.py` | Research label builders. | `RiskRegimeLabelBuilder`, `DrawdownRiskLabelBuilder`, `ChampionSuccessLabelBuilder`, `ShouldReduceExposureLabelBuilder` | `label_type`, `prediction_target`, `label_horizon_days`, `prediction_horizon`, `drawdown_risk_threshold` |
| `core/research/ml/datasets.py` | Joins feature and label rows into model-ready datasets. | `MLDataset`, `build_dataset`, `write_dataset` | Label name from label builder; excludes metadata fields such as dates/universe symbols |
| `core/research/ml/validation.py` | Chronological holdout and rolling walk-forward split logic. | `ChronologicalSplit`, `WalkForwardFold`, `chronological_holdout`, `rolling_walk_forward` | `test_fraction`, `walk_forward_folds`, optional train/test date keys |
| `core/research/ml/evaluation.py` | Classification metrics. | `classification_metrics` | Decision threshold already applied before metrics |
| `core/research/ml/calibration.py` | Raw, Platt, isotonic, and temperature calibration comparison. | `build_probability_calibration`, `compare_calibration_methods` | `calibration_bin_count` |
| `core/research/ml/overlay.py` | Shadow overlay simulation and label-direction rules. | `simulate_shadow_overlay`, `should_reduce_exposure`, `overlay_decision_rule` | `label_type`, `decision_threshold`, `shadow_thresholds`, `shadow_reduced_exposures`, `shadow_transaction_cost_bps` |
| `core/research/ml/rule_overlay.py` | Rule-based exposure studies and drawdown diagnostics. | `run_rule_exposure_study`, `run_volatility_managed_walk_forward`, `run_drawdown_risk_diagnostics` | `shadow_transaction_cost_bps`, drawdown/volatility study thresholds |
| `core/research/ml/meta_ensemble.py` | Builds meta dataset from source prediction artifacts, trains logistic meta ensemble, evaluates overlay, writes leaderboard. | `MetaEnsembleResult`, `run_meta_ensemble`, `build_meta_dataset_rows`, `_fit_logistic`, `_overlay_summary` | `model_type=meta_ensemble`, `ensemble_name`, `label_type`, `source_prediction_dirs`, `expanded_rebalance_dataset_path`, `meta_dataset_path`, `meta_model_type`, `decision_threshold`, `decision_thresholds`, `reduced_exposures` |
| `core/research/ml/leaderboard.py` | Combines champion, source model, and meta model metrics into JSON/Markdown leaderboard. | `write_leaderboard`, `_source_row`, `_walk_forward_balanced_accuracy` | Reads source `metrics.json`, `probability_calibration.json`, `calibrated_probability_calibration.json`, `holdout_shadow_overlay.json`; meta metrics/calibration/overlay passed directly |
| `core/research/ml/html_report.py` | Static HTML research report from artifacts. | `write_research_html_report` | Reads output-dir artifacts: metrics, calibration, walk-forward, model comparison, overlays, confusion matrix |
| `core/research/ml/diagnostics.py` | Probability summaries and ranking diagnostics. | `rolling_base_rate_probabilities`, `probability_summary`, `build_ranking_diagnostics` | `rolling_base_rate_lookback_samples`, `ranking_quantile_count` |
| `core/research/ml/rebalance_dataset.py` | Champion and expanded rebalance datasets with new forward/vol/drawdown labels. | `build_champion_rebalance_rows`, `build_expanded_rebalance_rows`, `write_expanded_rebalance_audit`, `should_reduce_exposure_label` | `expanded_rebalance_dataset.rebalance_frequencies`, `top_n_values`, `weightings`, `universe_paths`, `reduce_drawdown_threshold`, `reduce_excess_return_threshold`, `reduce_volatility_adjusted_threshold`, `backtest.years`, `ml.research_years` |
| `application/services/ml_commands.py` | CLI service layer for research ML modes. | `run_ml_research`, `run_ml_data_inventory`, `run_ml_build_universes`, `run_ml_expanded_rebalance_dataset`, `run_ml_meta_ensemble` | `ml.output_dir`, inventory/universe paths, data thresholds, meta ensemble config |
| `application/services/runtime_overrides.py` | Runtime config overlays for CLI modes/universes/research-year behavior. | `apply_runtime_overrides`, `apply_fast_mode` | `--universe`, `--fast`, `--years`, `ml.research_years`, `backtest.years`, Stooq provider dirs |
| `application/services/stooq_bulk_commands.py` | Research-only Stooq bulk import command. | `run_stooq_bulk_import`, `_select_raw_candidates` | `stooq_bulk_extracted_dir`, `stooq_bulk_zip_path`, `stooq_parquet_dir`, `resume_stooq_bulk_import`, CLI `--top`, `--all-raw`, `--asset-class`, `--min-rows`, `--exclude-warrants-units-rights` |
| `infrastructure/data/stooq_bulk_importer.py` | Validates raw Stooq ASCII files and writes Parquet; supports flat and recursive raw layouts. | `StooqBulkImporter`, `StooqBulkImportManifest`, `StooqRawSymbolCandidate`, `raw_symbol_candidates`, `select_raw_symbols`, `import_symbols_with_manifest` | Import dirs, `minimum_history_years`, `history_coverage_tolerance_days`, raw asset/min-row selection |
| `core/research/ml/data_inventory.py` | Scans Stooq Parquet inventory and liquidity/coverage. | `SymbolInventory`, `build_data_inventory`, `inspect_symbol_file`, `write_inventory_reports` | `stooq_parquet_dir`, `inventory_output_dir`, `min_history_years`, `max_latest_gap_days`, `min_average_dollar_volume_252d` |
| `core/research/ml/universe_builder.py` | Builds `current_32`, `us_liquid_100`, `us_liquid_250`, `us_liquid_500`. | `UniverseBuildResult`, `build_universe_files` | `inventory_path`, `universe_output_dir`, `stooq_parquet_dir`, inventory thresholds |
| `infrastructure/data/stooq_parquet_data_feed.py` | Historical data feed over processed Stooq Parquet. | `StooqParquetDataFeed` | `backtest.provider=stooq_parquet`, `backtest.data_dir`, `ml.stooq_parquet_dir` |
| `config/config_loader.py` | Defaults and config validation. | `load_config`, `merge_defaults`, `validate_config` | Allowed `ml.model_type`, label types, sequence lengths, transformer/PatchTST validation, calibration bins |

## Key Research Configs

| Path | Purpose | Notes |
|---|---|---|
| `configs/research/dlinear_should_reduce_exposure.yaml` | DLinear over expanded rebalance dataset. | Current should-reduce-exposure DLinear source for meta ensemble. |
| `configs/research/patchtst_should_reduce_exposure.yaml` | PatchTST over expanded rebalance dataset. | Current PatchTST source for meta ensemble. |
| `configs/research/transformer_should_reduce_exposure.yaml` | Transformer over expanded rebalance dataset. | Current transformer source for meta ensemble. |
| `configs/research/regime_transformer_meta_ensemble_v1.yaml` | Meta ensemble over DLinear/PatchTST/Transformer prediction artifacts. | Source dirs, thresholds, reduced exposures live here. |
| `configs/research/*champion_success.yaml` | Older champion-success experiments. | Uses opposite overlay direction: reduce exposure when probability is below threshold. |
| `configs/research/*drawdown_risk.yaml` | Drawdown-risk experiments. | Useful for trend/regime scorer labels and diagnostics. |

## Tests To Keep Close

| Path | Coverage |
|---|---|
| `tests/test_transformer_research.py` | Transformer model construction, sequence alignment, save/load. |
| `tests/test_ml_research.py` | End-to-end research runner artifacts, labels, overlays, calibration, reports. |
| `tests/test_ml_leaderboard.py` | Leaderboard rows and overlay/reporting fields. |
| `tests/test_meta_ensemble.py` | Meta dataset, overlay math, label direction, holdout/test-only counts. |
| `tests/test_overlay_decision_rules.py` | Label-direction rules for `should_reduce_exposure` and `champion_success`. |
| `tests/test_ml_universe_builder.py` | Inventory-driven universe generation. |
| `tests/test_stooq_bulk_importer.py` | Recursive Stooq import, manifest/resume, raw candidate selection. |
| `tests/test_ml_data_inventory.py` | Data inventory coverage/liquidity behavior. |

## Exact Next Edit Targets

### iTransformer Cross-Asset Ranker

Edit/add:
- Add `core/research/ml/itransformer_model.py`.
- Extend `core/research/ml/models.py::build_ml_model` with `model_type == "itransformer_ranker"`.
- Extend `config/config_loader.py::validate_config` allowed `ml.model_type` and validate `itransformer_*` keys.
- Add config: `configs/research/itransformer_cross_asset_ranker.yaml`.
- Add tests: `tests/test_itransformer_research.py`.

Likely config keys:
- `itransformer_sequence_length`, `itransformer_d_model`, `itransformer_heads`, `itransformer_layers`, `itransformer_feedforward`, `itransformer_dropout`, `itransformer_epochs`, `itransformer_batch_size`, `itransformer_learning_rate`, `itransformer_weight_decay`, `itransformer_device`.

Design note:
- Current dataset interface is row-wise classification. A cross-asset ranker likely needs grouped samples by rebalance date/universe. Add a small grouped dataset builder rather than forcing rank labels into `MLDataset` too early.

### Momentum Transformer Trend / Regime Scorer

Edit/add:
- Add `core/research/ml/momentum_transformer_model.py` or extend `transformer_model.py` only if it remains a simple sequence classifier.
- Add label/target support in `core/research/ml/labels.py` if using a new trend/regime label.
- Add trend/regime features in `core/research/ml/features.py` or use existing `price_regime_v1` features first.
- Register model in `core/research/ml/models.py`.
- Validate config keys in `config/config_loader.py`.
- Add config: `configs/research/momentum_transformer_regime_scorer.yaml`.
- Add tests near `tests/test_transformer_research.py`.

Useful existing labels:
- `risk_regime`, `drawdown_risk`, `should_reduce_exposure`, `champion_success`.

### Upgraded Meta Ensemble With LightGBM / Gradient Boosting

Edit/add:
- `core/research/ml/meta_ensemble.py`: replace `_fit_logistic` / `_predict_logistic` with a meta-model factory such as `_fit_meta_model` and `_predict_meta_model`.
- `core/research/ml/models.py`: optional shared gradient boosting wrapper if reusing the registry; otherwise keep meta models private to `meta_ensemble.py`.
- `config/config_loader.py`: allow/validate `ml.meta_model_type` values such as `logistic_regression`, `gradient_boosting`, `lightgbm`.
- `configs/research/regime_transformer_meta_ensemble_v1.yaml`: set `meta_model_type`.
- `tests/test_meta_ensemble.py`: prove model selection and finite overlay math.

Dependency note:
- If LightGBM is not already in requirements, prefer sklearn `HistGradientBoostingClassifier` or existing `GradientBoostingClassifier` first, then add LightGBM as an optional dependency with graceful error messaging.

### Walk-Forward Meta Ensemble Evaluation

Edit/add:
- `core/research/ml/meta_ensemble.py`: add chronological / rolling folds over `meta_dataset_rows`; write `walk_forward_metrics.json` and walk-forward overlay summaries.
- Reuse `core/research/ml/validation.py` concepts, or add a meta-specific splitter if CSV rows need date grouping.
- `core/research/ml/leaderboard.py`: populate meta `walk_forward_balanced_accuracy`.
- `tests/test_meta_ensemble.py`: prove folds train only on prior dates and evaluate only holdout/test dates.

Important guard:
- Group by `rebalance_date` so rows from the same rebalance date do not leak across train/test folds.

### Threshold Sweep / Promotion Gates

Edit/add:
- `core/research/ml/experiment_runner.py`: source-model threshold sweeps already exist; tighten output fields/gates if needed.
- `core/research/ml/meta_ensemble.py`: add threshold sweep over `decision_thresholds` and `reduced_exposures` for meta ensemble.
- `core/research/ml/leaderboard.py`: add gate fields: finite overlay, minimum sample count, max drawdown non-worsening, calibration ceiling, walk-forward balanced accuracy floor.
- Add `core/research/ml/promotion_gates.py` if gate logic grows beyond simple reporting.
- Tests: `tests/test_meta_ensemble.py`, `tests/test_ml_leaderboard.py`, `tests/test_ml_research.py`.

Suggested gate fields:
- `promotion_candidate`, `gate_min_overlay_sample_count`, `gate_finite_overlay`, `gate_max_drawdown_not_worse`, `gate_brier_score_max`, `gate_ece_max`, `gate_walk_forward_balanced_accuracy_min`.

## Safe Commands

Focused tests:

```bash
python -m pytest tests/test_transformer_research.py tests/test_meta_ensemble.py tests/test_ml_leaderboard.py tests/test_overlay_decision_rules.py
```

Research runner smoke command:

```bash
python main.py --mode ml-research --config configs/research/transformer_should_reduce_exposure.yaml
```

Meta ensemble command:

```bash
python main.py --mode ml-meta-ensemble --config configs/research/regime_transformer_meta_ensemble_v1.yaml
```

Avoid long training until the model registry/config/tests are in place.
