# Multi-Task Transformer Heads Design

Research-only design note. Do not implement runtime code in this step.

## Purpose

Multi-task transformer heads should let one sequence encoder learn shared market state while producing both the current classification target and forward-looking research diagnostics. The main production-adjacent target remains research-only `should_reduce_exposure`; auxiliary regression heads should improve representation quality and give richer model diagnostics.

No output from this design should directly change broker, paper trading, live trading, execution, or portfolio behavior.

## Heads

### Classification Head

Target:

- `should_reduce_exposure`

Output:

- `probability_should_reduce_exposure`

Training objective:

- Binary cross-entropy with logits.
- Optional positive-class weighting, matching the current `*_pos_weight: auto` sequence-model convention.

Decision rule:

- For `should_reduce_exposure`, reduce exposure when probability is greater than or equal to the decision threshold.

### Regression Heads

Targets:

- `forward_return_5d`
- `forward_return_10d`
- `future_volatility`
- `future_drawdown`
- `max_adverse_excursion`
- `max_favourable_excursion`

Outputs:

- `predicted_forward_return_5d`
- `predicted_forward_return_10d`
- `predicted_future_volatility`
- `predicted_future_drawdown`
- `predicted_max_adverse_excursion`
- `predicted_max_favourable_excursion`

Training objective:

- Huber loss by default for robustness to fat tails.
- Mean squared error may be reported as a metric, but should not be the first choice for training.
- Targets should be standardized on the training fold only, then inverse-transformed for reports.

## Architecture

Proposed shape:

- Shared transformer encoder over rolling market/rebalance features.
- One classification head for `should_reduce_exposure`.
- One small regression head per forward label.
- Optional shared regression trunk before the individual regression heads if head count grows.

The design can apply to a generic Transformer, PatchTST-style encoder, iTransformer encoder, or Momentum Transformer encoder. The first implementation should keep the head layer separate from the encoder so head outputs can be tested independently.

## Loss Weighting

Combined loss:

```text
total_loss =
  classification_weight * BCE(should_reduce_exposure)
  + forward_return_5d_weight * Huber(forward_return_5d)
  + forward_return_10d_weight * Huber(forward_return_10d)
  + future_volatility_weight * Huber(future_volatility)
  + future_drawdown_weight * Huber(future_drawdown)
  + max_adverse_excursion_weight * Huber(max_adverse_excursion)
  + max_favourable_excursion_weight * Huber(max_favourable_excursion)
```

Initial conservative weights:

```yaml
ml:
  multitask_classification_weight: 1.0
  multitask_forward_return_5d_weight: 0.20
  multitask_forward_return_10d_weight: 0.20
  multitask_future_volatility_weight: 0.15
  multitask_future_drawdown_weight: 0.20
  multitask_max_adverse_excursion_weight: 0.15
  multitask_max_favourable_excursion_weight: 0.10
```

Rules:

- Classification remains the primary objective.
- Auxiliary losses should not dominate total loss.
- Report per-head losses separately.
- Consider uncertainty weighting only after fixed weights are stable.
- If a target is missing for a row, mask that head’s loss for that row rather than dropping the full sample.

## Artifact Outputs

Existing provenance fields must remain on every prediction artifact:

- `dataset_hash`
- `source_dataset_row_count`
- `train_sample_count`
- `test_sample_count`
- `generated_at`

Candidate `prediction_artifacts.csv` columns:

- `probability_should_reduce_exposure`
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
- `classification_loss`
- `forward_return_5d_loss`
- `forward_return_10d_loss`
- `future_volatility_loss`
- `future_drawdown_loss`
- `max_adverse_excursion_loss`
- `max_favourable_excursion_loss`

Candidate JSON metadata fields:

- `multitask_enabled`
- `multitask_heads`
- `multitask_loss_weights`
- `target_standardization`
- `head_metrics`
- `missing_target_counts_by_head`
- `research_only`
- `trading_impact: none`

## Metrics

Classification metrics:

- Balanced accuracy.
- Precision/recall/F1.
- ROC AUC if available.
- Brier score.
- Expected Calibration Error.
- Confusion matrix.

Regression metrics per head:

- MAE.
- RMSE.
- Median absolute error.
- Spearman rank correlation.
- Directional accuracy for forward returns.
- Sign accuracy for drawdown/MAE style targets where meaningful.

Overlay/report metrics:

- Overlay return delta.
- Max drawdown improvement.
- Reduced-exposure days.
- Turnover.
- Finite sanity checks.

## Leakage Prevention

Strict rules:

- Forward labels must never be used as input features.
- Any column beginning with `forward_`, `future_`, `max_adverse_`, or `max_favourable_` is target-only unless explicitly whitelisted as an actual target field.
- Feature values must be computed only from data available at or before `feature_date`.
- Target standardization parameters must be fit on train folds only.
- Calibration must be fit on train/out-of-fold predictions only, not final holdout labels.
- Same `rebalance_date` groups must not be split across train/test folds.
- Purging must remove training rows whose label windows overlap the test start.
- Prediction artifacts from different `dataset_hash` values must not be mixed.

Validation checks:

- Denylist future-looking input columns before model training.
- Emit `input_feature_count` and `target_head_count`.
- Emit missing target counts per head.
- Fail if any target column is all null in train or test.
- Fail if any predicted output is non-finite.
- Report train/test date ranges for every run.

## Promotion Criteria

Multi-task heads should not be promoted merely because one auxiliary regression metric improves.

Minimum promotion evidence:

- Classification balanced accuracy is at least as good as the single-task baseline in walk-forward evaluation.
- Brier score and ECE are not materially worse than the best calibrated single-task model.
- Overlay return delta is finite and improves or does not materially degrade versus baseline.
- Max drawdown impact improves or does not materially degrade.
- Reduced-exposure days and turnover remain within configured bounds.
- Regression heads show useful signal out of sample, especially rank correlation and directional accuracy.
- Results hold across walk-forward folds, not only final holdout.
- All artifact provenance fields are present and dataset hashes match across source models.

Selection reporting should separate:

- Best classifier.
- Best calibrated classifier.
- Best overlay model.
- Best auxiliary regression signal.

## Tests To Add

Unit tests:

- Multi-task head forward pass returns one classification logit and all regression outputs.
- Output tensors have expected shape for batch size and head count.
- Loss function masks missing targets per head.
- Loss weighting changes total loss without changing individual head losses.
- Target standardization is fit only on train data.
- Future-looking columns are rejected as model inputs.

Model/registry tests:

- `build_ml_model()` can create the multi-task transformer model type when implemented.
- Invalid head config fails clearly.
- Save/load preserves head config, target scalers, and output shape.

Runner/artifact tests:

- End-to-end research run writes classification and regression artifact columns.
- `prediction_artifacts.csv/json` include dataset provenance.
- Non-finite predictions fail before report generation.
- Missing target counts are reported.

Validation/meta tests:

- Same-date rebalance rows do not leak across folds.
- Meta-ensemble refuses mixed dataset hashes.
- Meta-ensemble v3 can ingest multi-task diagnostic columns as features without using actual future labels.
- Promotion gates report classification, calibration, overlay, and auxiliary-head criteria separately.

## Research-Only Boundary

Allowed:

- Offline model training.
- Research reports.
- Shadow overlays.
- Meta-ensemble research features.
- Promotion-gate analysis.

Not allowed:

- Broker changes.
- Paper trading behavior changes.
- Live trading behavior changes.
- Execution model changes.
- Direct portfolio sizing changes outside research reports.
