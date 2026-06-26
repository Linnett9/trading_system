# Multi-Task Transformer Implementation Plan

## Scope

This plan covers a research-only implementation path for multi-task transformer heads using the existing design in `reports/ml/multitask_transformer_heads_design.md`. It should not modify broker, paper trading, live trading, execution, dataset generation, meta ensemble, leaderboard, or artifact runtime code until the planned implementation phase begins.

The goal is to keep the current single-task `should_reduce_exposure` flow stable while adding optional regression heads for richer research signals.

Status: phase 1 model, registry, config validation, isolated research config, and focused tests have been implemented for `multitask_transformer`. Auxiliary prediction artifact columns and meta ensemble v3 ingestion remain future phases.

## First Model To Upgrade

Start with the existing Transformer family, preferably as a new `multitask_transformer` model type rather than changing every model at once.

Recommended first target:

- `TransformerSequenceMLModel` behavior, implemented either in a new `core/research/ml/multitask_transformer_model.py` or as a small subclass/wrapper around the existing transformer implementation.

Why not all models at once:

- The artifact contract should be proven once before spreading it across PatchTST, iTransformer, and Momentum Transformer.
- Multi-head loss masking, target scaling, and artifact output need careful leakage checks.
- A new model type keeps the existing `transformer`, `patchtst`, `dlinear`, `itransformer`, and `momentum_transformer` baselines directly comparable.

PatchTST is the best phase 2 candidate because it is already a strong sequence model, but it should wait until the first multi-task artifact schema is stable.

## Exact Files To Modify Later

Core model and registry:

- `core/research/ml/multitask_transformer_model.py`
  - New research-only model implementation.
  - Key classes/functions: `MultiTaskTransformerSequenceMLModel`, multi-head forward pass, classification head, regression heads, loss calculation, `fit`, `predict_proba`, `predict`, `predict_multitask`, `save`, `load`.
- `core/research/ml/models.py`
  - Register model type `multitask_transformer`.
  - Keep existing model registrations unchanged.
- `core/research/ml/transformer_model.py`
  - Optional only if shared encoder utilities are factored out.
  - Avoid broad rewrites.

Experiment and artifact flow:

- `core/research/ml/experiment_runner.py`
  - Future phase only: detect optional multi-task outputs and write additional artifact columns.
  - Preserve existing `predict_proba` and legacy artifact behavior.
- `core/research/ml/config.py`
  - Add typed config fields only if this module owns ML model config schema.
- `config/config_loader.py`
  - Add defaults and validation for `multitask_transformer` config keys.

Configs:

- `configs/research/multitask_transformer_should_reduce_exposure.yaml`
  - First single multi-task config.
- Optional later:
  - `configs/research/patchtst_multitask_should_reduce_exposure.yaml`
  - `configs/research/itransformer_multitask_should_reduce_exposure.yaml`

Reports/docs:

- `reports/ml/multitask_transformer_heads_design.md`
  - Source design reference.
- `reports/ml/meta_ensemble_v3_design.md`
  - Future consumer design.

Tests:

- `tests/test_multitask_transformer.py`
  - New focused model tests.
- `tests/test_sequence_model_registry.py`
  - Registry recognizes `multitask_transformer`.
- `tests/test_transformer_research.py`
  - Forward pass, save/load, finite outputs.
- `tests/test_config_loader.py`
  - Config defaults and validation.
- `tests/test_prediction_artifacts.py`
  - Artifact schema and provenance for optional multi-task columns.
- `tests/test_ml_research.py`
  - End-to-end runner compatibility.
- Future phase 3 only: `tests/test_meta_ensemble.py`
  - Meta ensemble v3 consumes predicted auxiliary columns, never actual labels.

## Required Config Keys

Minimum config for phase 1:

```yaml
ml:
  model_type: multitask_transformer
  label_type: should_reduce_exposure

  multitask_enabled: true
  multitask_primary_target: should_reduce_exposure
  multitask_regression_targets:
    - forward_return_5d
    - forward_return_10d
    - future_volatility
    - future_drawdown
    - max_adverse_excursion
    - max_favourable_excursion

  multitask_loss:
    classification_weight: 1.0
    regression_loss: huber
    huber_delta: 1.0
    missing_target_policy: mask_head_loss
    target_standardization: train_fold_only
    regression_weights:
      forward_return_5d: 0.25
      forward_return_10d: 0.25
      future_volatility: 0.20
      future_drawdown: 0.20
      max_adverse_excursion: 0.15
      max_favourable_excursion: 0.15

  multitask_outputs:
    write_auxiliary_predictions: true
    write_head_metrics: true
```

Existing transformer keys should remain available:

- `sequence_length`
- `epochs`
- `batch_size`
- `learning_rate`
- `hidden_size` or `d_model`
- `num_layers`
- `num_heads`
- `dropout`
- `decision_threshold`
- `calibration_method`
- `output_dir`

## Artifact Schema Changes

Existing prediction artifact columns must remain unchanged:

- `date`
- `rebalance_date`
- `feature_id`
- `variant_id`
- `model_type`
- `label_type`
- `split`
- `fold`
- `actual_label`
- `raw_probability`
- `calibrated_probability`
- `prediction`
- `decision_threshold`
- `research_label`
- `dataset_hash`
- `source_dataset_row_count`
- `train_sample_count`
- `test_sample_count`
- `generated_at`

Optional multi-task columns to add later:

- `predicted_forward_return_5d`
- `actual_forward_return_5d`
- `predicted_forward_return_10d`
- `actual_forward_return_10d`
- `predicted_future_volatility`
- `actual_future_volatility`
- `predicted_future_drawdown`
- `actual_future_drawdown`
- `predicted_max_adverse_excursion`
- `actual_max_adverse_excursion`
- `predicted_max_favourable_excursion`
- `actual_max_favourable_excursion`

Optional JSON metadata additions:

- `multitask_enabled`
- `multitask_primary_target`
- `multitask_regression_targets`
- `multitask_loss_weights`
- `target_standardization`
- `missing_target_counts_by_head`
- `head_metrics`
- `head_output_columns`

The artifact writer should continue overwriting artifacts, preserving provenance, and keeping dataset hash checks compatible with the meta ensemble.

## Backward Compatibility

Single-task behavior stays the default.

Rules:

- Existing `model_type: transformer` remains single-task unless explicitly configured otherwise.
- Prefer a new `model_type: multitask_transformer` for phase 1.
- Existing `predict_proba` must return only the `should_reduce_exposure` probability expected by `MLExperimentRunner`.
- Existing `predict` must return the binary decision using `decision_threshold`.
- Extra outputs are optional and accessed through a separate method such as `predict_multitask` or `predict_auxiliary`.
- If auxiliary columns are absent, current leaderboard and meta ensemble behavior should be unchanged.
- Existing configs and reports should not be overwritten by the new multi-task config.

## Leakage Prevention

The regression targets are future labels and must never become input features.

Validation rules:

- Block feature columns with prefixes or exact names:
  - `forward_`
  - `future_`
  - `max_adverse_`
  - `max_favourable_`
  - `actual_`
  - `research_label`
- Build feature tensors before attaching target tensors.
- Fit feature scalers on training folds only.
- Fit regression target scalers on training folds only.
- Apply target inverse transforms only after prediction.
- Mask missing regression targets per head instead of dropping rows globally unless explicitly configured.
- Preserve chronological splits and avoid random leakage across rebalance dates.
- Ensure any calendar features are known at prediction time.
- Keep `dataset_hash` tied to the source dataset used for all heads.

## Validation Plan

Classification head:

- Balanced accuracy.
- Precision, recall, F1.
- Brier score.
- Expected Calibration Error.
- Confusion matrix.
- Overlay sanity checks remain outside phase 1 unless existing runner already computes them.

Regression heads:

- MAE.
- RMSE.
- Median absolute error.
- Spearman rank correlation.
- Directional accuracy for forward returns.
- Finite output checks.
- Per-head missing target counts.

Joint model checks:

- Total loss is finite.
- Per-head losses are finite.
- Loss weights are applied correctly.
- Missing labels only mask their own regression head.
- Classification output shape is `(n_samples,)` or `(n_samples, 2)` according to existing convention.
- Regression output shape is `(n_samples, n_regression_heads)`.
- Save/load preserves all heads and config.

## Meta Ensemble V3 Feed

Phase 3 should consume predicted auxiliary columns only. It must not consume actual future labels from artifacts.

Candidate meta features:

- `predicted_forward_return_5d` as expected short-horizon return.
- `predicted_forward_return_10d` as expected medium-horizon return.
- `predicted_future_volatility` as expected volatility.
- `predicted_future_drawdown` as expected drawdown.
- `predicted_max_adverse_excursion` as expected adverse path risk.
- `predicted_max_favourable_excursion` as expected favourable path potential.
- Existing `raw_probability` and `calibrated_probability`.

V3 rules:

- Require matching `dataset_hash` across all source artifacts.
- Require source artifact timestamps and row counts.
- Reject mixed stale artifacts.
- Namespace source columns by model where needed, for example `multitask_transformer_predicted_future_drawdown`.
- Keep `actual_*` columns for evaluation only, never as features.

## Tests To Add

Phase 1 tests:

- Registry recognizes `multitask_transformer`.
- Config loader accepts required multi-task keys.
- Invalid regression target names fail clearly.
- Forward pass returns classification probability and all regression heads.
- Outputs are finite on synthetic data.
- Missing regression labels are masked per head.
- Loss weights change the joint loss as expected.
- Save/load round trip preserves predictions.
- End-to-end runner writes standard prediction artifacts with provenance.
- Optional auxiliary artifact columns appear only when enabled.
- Existing single-task transformer config still produces the legacy artifact schema.
- Leakage guard rejects future label columns in feature tensors.

Phase 2 tests:

- PatchTST multi-task forward pass.
- iTransformer multi-task forward pass if added.
- Momentum Transformer auxiliary outputs remain compatible with the shared schema.

Phase 3 tests:

- Meta ensemble v3 loads auxiliary predicted columns.
- Meta ensemble v3 rejects `actual_*` columns as features.
- Meta ensemble v3 rejects mixed dataset hashes.
- Promotion gates include auxiliary-feature model rows without changing live execution code.

## Migration Path

Phase 1: single model, single config

- Add `multitask_transformer` as a new research-only model type.
- Add `configs/research/multitask_transformer_should_reduce_exposure.yaml`.
- Add optional auxiliary artifact columns.
- Keep existing single-task Transformer, PatchTST, DLinear, iTransformer, Momentum Transformer, leaderboard, and meta ensemble behavior unchanged.

Phase 2: add to other transformer models

- Port the proven multi-head pattern to PatchTST first.
- Consider iTransformer next if cross-asset ranking benefits from auxiliary return/risk heads.
- Consider Momentum Transformer only if trend/regime outputs need shared training with future return/risk labels.

Phase 3: meta ensemble v3 consumes extra outputs

- Extend meta ensemble v3 to ingest predicted auxiliary columns.
- Preserve dataset hash and provenance checks.
- Add promotion gates using walk-forward balanced accuracy, Brier/ECE, overlay return delta, drawdown improvement, turnover, and reduced exposure days.
- Continue reporting `selected_classifier`, `selected_calibrated`, and `selected_overlay` separately.

## Research-Only Boundary

This work must stay under research ML modules, configs, reports, and tests. It must not modify broker adapters, paper trading services, live trading services, execution models, order placement, or portfolio execution behavior.
