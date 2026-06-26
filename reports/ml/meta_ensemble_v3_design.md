# Meta Ensemble V3 Design

Research-only design note. Do not implement runtime code in this step.

Meta ensemble v3 should extend the current meta-ensemble research flow so it can consume richer source-model diagnostics from iTransformer, Momentum Transformer, and future multi-task heads while preserving strict provenance checks and promotion gates.

## Research-Only Boundary

Allowed:

- Offline research evaluation.
- Meta-feature construction from source prediction artifacts.
- Walk-forward model comparison.
- Shadow overlay reporting.
- Promotion-gate evidence.

Not allowed:

- Broker changes.
- Paper trading changes.
- Live trading changes.
- Execution changes.
- Direct portfolio sizing changes.
- Runtime changes to `experiment_runner`, `meta_ensemble`, `leaderboard`, or artifact writing in this design step.

## Source Columns Consumed By V3

Meta ensemble v3 should continue to consume each source model's primary probability columns, then add optional diagnostic columns when present.

Required existing source columns:

- `feature_id`
- `rebalance_date`
- `model_type`
- `label_type`
- `split`
- `actual_label`
- `raw_probability`
- `calibrated_probability`
- `decision_threshold`

Optional v3 source diagnostic columns:

- `rank_score`: iTransformer cross-asset rank score.
- `trend_score`: Momentum Transformer trend-strength score.
- `regime_score`: Momentum Transformer regime-compatibility score.
- `size_multiplier`: research-only exposure multiplier suggestion.
- `expected_return`: expected forward return.
- `expected_volatility`: expected future volatility.
- `expected_drawdown`: expected future drawdown.
- `predicted_max_adverse_excursion`
- `predicted_max_favourable_excursion`
- `predicted_forward_return_5d`
- `predicted_forward_return_10d`
- `predicted_future_volatility`
- `predicted_future_drawdown`

Consumption rules:

- Missing optional columns should not fail the run; they should be recorded in the meta dataset audit.
- Present optional columns must be finite or explicitly blank/null.
- Actual future labels such as `actual_forward_return_5d` may be used for reporting/evaluation only, not as meta learner features.
- Each source model's optional fields should be namespaced in the meta dataset, for example `itransformer_rank_score` and `momentum_transformer_trend_score`.

## Required V3 Source Artifact Schema

Minimum required `prediction_artifacts.csv` columns:

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
- `dataset_hash`
- `source_dataset_row_count`
- `train_sample_count`
- `test_sample_count`
- `generated_at`
- `research_label`

Optional source diagnostic columns:

- `rank_score`
- `trend_score`
- `regime_score`
- `size_multiplier`
- `expected_return`
- `expected_volatility`
- `expected_drawdown`
- `predicted_max_adverse_excursion`
- `predicted_max_favourable_excursion`
- `predicted_forward_return_5d`
- `predicted_forward_return_10d`
- `predicted_future_volatility`
- `predicted_future_drawdown`

Required `prediction_artifacts.json` fields:

- `model_type`
- `label_type`
- `feature_set`
- `dataset_hash`
- `data_hash`
- `source_dataset_row_count`
- `train_sample_count`
- `test_sample_count`
- `generated_at`
- `validation_method`
- `row_count`
- `research_only`
- `trading_impact`

Recommended v3 JSON fields:

- `source_optional_columns`
- `missing_optional_column_count`
- `non_finite_prediction_count`
- `calibration_method`
- `artifact_schema_version`

## Dataset Hash And Provenance Rules

Hard rules:

- Every source artifact must include `dataset_hash`.
- Every source artifact CSV row used by v3 must include the same `dataset_hash` as its JSON metadata.
- All source models in one v3 meta run must share the same `dataset_hash`.
- All source artifacts must report non-empty `source_dataset_row_count`, `train_sample_count`, `test_sample_count`, and `generated_at`.
- If any source artifact is missing provenance, v3 should fail with a clear rerun instruction.
- If artifacts from different dataset hashes are mixed, v3 should fail before building a meta dataset.

Audit fields:

- `source_dataset_hash`
- `source_dataset_row_counts_by_model`
- `source_artifact_generated_at_by_model`
- `source_optional_columns_by_model`
- `missing_optional_columns_by_model`
- `meta_feature_count`
- `meta_row_count`
- `same_rebalance_date_leakage_check`

## Meta Feature Construction

Base meta features:

- Per-source raw probabilities.
- Per-source calibrated probabilities.
- Probability disagreement features.
- Probability rank/order features.
- Variant context features already present in the expanded rebalance dataset.

V3 diagnostic meta features:

- `itransformer_rank_score`
- `momentum_transformer_trend_score`
- `momentum_transformer_regime_score`
- `momentum_transformer_size_multiplier`
- `expected_return_by_model`
- `expected_volatility_by_model`
- `expected_drawdown_by_model`
- `mae_prediction_by_model`
- `mfe_prediction_by_model`
- Crosses such as trend score times expected drawdown, regime score times volatility, and probability disagreement times rank score.

Safety rule:

- Use predicted diagnostics only. Do not use actual future outcomes as meta learner features.

## Meta Learner Candidates

V3 should compare:

- Logistic regression.
- Ridge logistic regression.
- Random forest.
- Gradient boosting.
- LightGBM, if installed and importable.

Candidate handling:

- Optional LightGBM should degrade gracefully when unavailable.
- Broken LightGBM imports should be reported clearly.
- Every candidate should produce comparable holdout, walk-forward, calibration, and overlay summaries.
- Selection should not assume the highest balanced accuracy is the best trading overlay.

## Promotion Gates

Promotion gates should use walk-forward evidence first, then final holdout as confirmation.

Required gates:

- `walk_forward_balanced_accuracy`
- `brier_score`
- `expected_calibration_error`
- `overlay_return_delta`
- `max_drawdown_improvement`
- `turnover`
- `reduced_exposure_days`
- Finite sanity checks for all overlay and return fields.
- Minimum overlay sample count.
- Maximum stale/missing optional feature rate, if optional v3 columns are expected.

Suggested gate interpretation:

- Balanced accuracy must clear a configured minimum or beat the source-model baseline.
- Brier/ECE must not materially degrade versus the best calibrated source model.
- Overlay return delta must be finite and preferably positive.
- Drawdown should improve or not materially worsen.
- Turnover and reduced-exposure days must remain within configured bounds.
- A candidate with excellent accuracy but poor calibration or damaging overlay behavior should not be selected as the overlay model.

## Selection Roles

V3 should report three separate selections.

### `selected_classifier`

Purpose:

- Best predictive classifier by walk-forward classification quality.

Ranking inputs:

- Walk-forward balanced accuracy.
- Holdout balanced accuracy as secondary confirmation.
- Minimum sample count and finite probability checks.

Selection reason example:

```text
selected for highest walk-forward balanced accuracy among finite candidates
```

### `selected_calibrated`

Purpose:

- Best probability estimator for downstream risk-aware decisions.

Ranking inputs:

- Brier score.
- Expected Calibration Error.
- Calibration stability across folds.
- Walk-forward balanced accuracy as a tie-breaker.

Selection reason example:

```text
selected for lowest Brier score with acceptable ECE and finite holdout probabilities
```

### `selected_overlay`

Purpose:

- Best research overlay utility after promotion-gate scoring.

Ranking inputs:

- Promotion gate score.
- Overlay return delta.
- Max drawdown improvement.
- Turnover.
- Reduced-exposure days.
- Brier/ECE sanity.
- Walk-forward balanced accuracy floor.

Selection reason example:

```text
selected for best promotion-gate score after calibration, drawdown, turnover, and finite-overlay checks
```

Important rule:

- `selected_classifier`, `selected_calibrated`, and `selected_overlay` may be different learners. The leaderboard should show this explicitly.

## Expected Reports

V3 should eventually write:

- `meta_dataset.csv`
- `meta_dataset_audit.json`
- `meta_model_comparison.json`
- `walk_forward_metrics.json`
- `probability_calibration.json`
- `threshold_sweep.json`
- `promotion_gates.json`
- `overlay_model_comparison.json`
- `leaderboard.json`
- `leaderboard.md`

Recommended new fields:

- `source_optional_columns_by_model`
- `selected_classifier`
- `selected_calibrated`
- `selected_overlay`
- `selection_reason`
- `promotion_gate_rank`
- `promotion_gate_score`
- `overlay_start_date`
- `overlay_end_date`
- `overlay_sample_count`
- `overlay_baseline_return`
- `overlay_adjusted_return`
- `overlay_return_delta`

## Tests Needed

Artifact/provenance tests:

- V3 refuses source artifacts missing `dataset_hash`.
- V3 refuses mixed dataset hashes.
- V3 refuses CSV/JSON hash mismatch.
- V3 reports source row counts and generation timestamps.

Schema tests:

- V3 consumes optional diagnostic columns when present.
- V3 records missing optional columns without failing.
- V3 rejects non-finite optional diagnostic values used as features.
- Actual future outcome columns are excluded from meta features.

Meta dataset tests:

- Source optional columns are namespaced by model.
- Probability disagreement features are computed correctly.
- Same `rebalance_date` rows are not split across train/test folds.
- Reduced-exposure days are computed only on holdout/test dates.

Learner comparison tests:

- Logistic, ridge logistic, random forest, and gradient boosting are compared.
- LightGBM is included when importable and marked unavailable when not.
- Broken LightGBM dependency produces a clear warning, not a silent bad result.

Selection tests:

- Best balanced-accuracy learner becomes `selected_classifier`.
- Best Brier/ECE learner becomes `selected_calibrated`.
- Best promotion-gate learner becomes `selected_overlay`.
- Selection reasons are populated.
- Highest balanced accuracy is not automatically selected as overlay.

Promotion-gate tests:

- Overlay return delta must be finite.
- Drawdown improvement is included in ranking.
- Turnover and reduced-exposure days affect promotion-gate ranking.
- Minimum overlay sample count is enforced.
- Candidate with non-finite overlay fields is rejected or ranked last.

Regression tests:

- Doubling the expanded dataset forces regenerated source artifacts with new dataset hashes.
- V3 refuses stale source artifacts from older dataset hashes.
- Leaderboard rows include all selected roles without overwriting the source model rows.

## Implementation Boundary For Future Work

Future implementation should be staged:

1. Add schema parsing for optional source columns.
2. Add meta dataset audit fields.
3. Add learner comparison with optional LightGBM.
4. Add selection roles.
5. Add promotion gate ranking.
6. Update leaderboard/report output.
7. Only then rerun full source models and meta ensemble.

Keep every step research-only.
