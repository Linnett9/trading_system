# Temporal Fusion Transformer Design

Research-only design note. Do not implement runtime code in this step.

## Purpose

Temporal Fusion Transformer, or TFT, is a candidate research model for richer interpretable sequence forecasting in the ML research stack. It should be evaluated as a research-only source model for `should_reduce_exposure` and multi-horizon auxiliary targets.

Proposed model type:

```yaml
ml:
  model_type: temporal_fusion_transformer
```

Proposed config path:

```text
configs/research/tft_should_reduce_exposure.yaml
```

## Why TFT Might Be Useful

TFT is attractive because it is designed for structured time-series forecasting with mixed input types and interpretable components.

Potential advantages:

- Interpretability: TFT can expose variable-selection weights, attention weights, and gating diagnostics.
- Feature selection: variable selection networks can learn which historical, static, and known-future inputs matter at each forecast point.
- Gating: gated residual networks can suppress noisy feature pathways instead of forcing every feature through the model equally.
- Multi-horizon outputs: TFT is naturally suited to predicting multiple future horizons from one encoder.
- Known future inputs: calendar and rebalance-schedule features can be supplied as future-known covariates without leaking market outcomes.

Useful known-future examples:

- Day of week.
- Month.
- Quarter.
- Month-end flag.
- Week-of-month.
- Scheduled rebalance frequency.
- Days until next rebalance.
- Days since last rebalance.
- Holiday/session calendar flags, if known in advance.

## How TFT Differs From Current Sequence Models

| Model | Current Role | TFT Difference |
|---|---|---|
| PatchTST | Patch-based temporal encoder over historical feature windows. | TFT adds explicit variable selection, gating, static context, and known-future covariates. |
| iTransformer | Inverted feature-token attention over rolling historical features. | TFT separates observed historical, static, and known-future inputs and provides more interpretable selection/gating artifacts. |
| Momentum Transformer | Trend/regime-sensitive classifier with optional trend/regime/size components. | TFT is broader: a multi-horizon structured forecaster with interpretable feature selection, not specifically trend/regime shaped. |
| Existing Transformer | Generic sequence classifier over historical feature rows. | TFT should be more structured, with feature-type separation, gating, variable selection, and multi-output forecasting heads. |

## Proposed Config

Initial research config:

```yaml
backtest:
  provider: stooq_parquet
  data_dir: data/processed/stooq_parquet

ml:
  enabled: true
  mode: research
  model_type: temporal_fusion_transformer
  shadow_model_type: temporal_fusion_transformer
  label_type: should_reduce_exposure
  prediction_target: should_reduce_exposure
  feature_set: expanded_rebalance_v1
  research_label: EXPANDED_REBALANCE_CURRENT_32_RESEARCH_ONLY
  minimum_history_years: 4
  output_dir: reports/ml/tft_should_reduce_exposure

  benchmark_symbols:
    - SPY
    - QQQ

  sequence_length: 64
  tft_encoder_length: 64
  tft_prediction_horizons: [5, 10]
  tft_hidden_size: 64
  tft_attention_heads: 4
  tft_lstm_layers: 1
  tft_dropout: 0.15
  tft_epochs: 30
  tft_batch_size: 64
  tft_learning_rate: 0.001
  tft_weight_decay: 0.0005
  tft_pos_weight: auto

  tft_known_future_features:
    - day_of_week
    - month
    - is_month_end
    - rebalance_frequency
    - days_until_next_rebalance

  comparison_models:
    - temporal_fusion_transformer
  overlay_comparison_models:
    - temporal_fusion_transformer
```

## Inputs

TFT should separate input groups explicitly.

### Observed Historical Features

Observed historical features are only known after market data is observed.

Examples:

- Returns over multiple horizons.
- Realized volatility.
- ATR.
- Drawdown.
- Breadth.
- Relative volume.
- Liquidity and dollar volume.
- Market/sector context.
- Existing expanded rebalance features.

### Static Features

Static features are optional and should be treated as metadata known before the forecast.

Examples:

- Universe name.
- Universe size bucket.
- Asset class bucket.
- Sector bucket, if symbol-level modeling is introduced.
- Rebalance variant metadata such as top-n bucket and weighting method.

### Known Future / Calendar Features

Known future features must be truly known at prediction time.

Examples:

- Day of week.
- Month.
- Quarter.
- Month-end or quarter-end flag.
- Scheduled rebalance date.
- Rebalance frequency.
- Days until next scheduled rebalance.
- Days since last scheduled rebalance.
- Market holiday/session calendar flags if available before prediction time.

Known-future inputs must not include future market prices, future realized volatility, future returns, revised macro data, or any post-event data.

## Outputs

Primary classification output:

- `probability_should_reduce_exposure`

Auxiliary multi-horizon outputs:

- `predicted_forward_return_5d`
- `predicted_forward_return_10d`
- `predicted_future_volatility`
- `predicted_future_drawdown`

Optional later outputs:

- quantile forecasts for returns/drawdown.
- forecast uncertainty.
- risk multiplier derived from probability and auxiliary heads.

Initial training objective:

- Binary cross-entropy for `should_reduce_exposure`.
- Huber loss for regression heads.
- Weighted multi-task loss with classification as the primary objective.

## Interpretability Artifacts

TFT should justify its inclusion through interpretability artifacts, not just raw metrics.

Candidate artifacts:

- `feature_selection_weights.csv`
- `attention_weights.csv`
- `gating_diagnostics.csv`
- `static_variable_selection.json`
- `historical_variable_selection.json`
- `known_future_variable_selection.json`
- `interpretability_summary.json`

Important fields:

- feature name.
- input group: observed, static, known future.
- split/fold.
- rebalance date.
- average selection weight.
- attention horizon.
- gating activation summary.

Interpretability artifacts should be research reports only and must not control trading decisions.

## Leakage Risks

Main risks:

- Known-future inputs are not actually known at prediction time.
- Calendar features accidentally include realized future rebalance outcomes.
- Future-normalized indicators use a denominator or percentile fit on the full dataset.
- Volatility, drawdown, or return labels leak into inputs.
- Post-event news, sentiment, analyst revisions, or restated data are included.
- Target scaling is fit on all rows instead of train folds only.
- Same rebalance date appears in both train and test folds.

Validation rules:

- Denylist future-looking input columns such as `forward_*`, `future_*`, `max_adverse_*`, `max_favourable_*`, and `actual_*`.
- Known-future feature builder must be deterministic from calendar/schedule data.
- Feature normalization must be fit on training folds only.
- Walk-forward folds must respect chronology and group same rebalance dates.
- Prediction artifacts must include dataset hash and provenance fields.
- Meta-ensemble consumers must reject mixed dataset hashes.

## Meta-Ensemble V3 Integration

TFT can feed meta-ensemble v3 as both a source classifier and a diagnostic source.

Potential source fields:

- `temporal_fusion_transformer_raw_probability`
- `temporal_fusion_transformer_calibrated_probability`
- `tft_predicted_forward_return_5d`
- `tft_predicted_forward_return_10d`
- `tft_predicted_future_volatility`
- `tft_predicted_future_drawdown`
- `tft_feature_selection_entropy`
- `tft_attention_concentration`
- `tft_gating_saturation`

Useful meta features:

- TFT probability disagreement versus DLinear/PatchTST/Transformer/iTransformer/Momentum Transformer.
- TFT expected return minus expected drawdown.
- TFT future volatility crossed with Momentum Transformer trend score.
- TFT feature-selection entropy as a confidence proxy.
- Attention concentration crossed with regime score.

Rules:

- Meta-ensemble v3 should consume predicted outputs and interpretability summaries only.
- Actual future labels must never be used as meta features.
- TFT artifacts must share the same dataset hash as all other source artifacts.

## Tests Required

Model tests:

- Registry can construct `temporal_fusion_transformer`.
- Forward pass returns one probability per input row.
- Multi-horizon regression heads return expected shapes.
- Known-future feature inputs are accepted separately from observed historical inputs.
- Save/load preserves output shape and config.

Config tests:

- Config validation accepts `configs/research/tft_should_reduce_exposure.yaml`.
- Invalid attention-head dimensions fail clearly.
- Invalid known-future feature declarations fail clearly.
- Invalid prediction horizons fail clearly.

Leakage tests:

- Future-looking columns are rejected from observed inputs.
- Known-future features are generated only from calendar/schedule data.
- Normalization/scaling is fit on train folds only.
- Same rebalance date does not leak across folds.

Artifact tests:

- Prediction artifacts include standard provenance.
- Optional TFT output columns are present only after artifact schema support is intentionally added.
- Interpretability artifacts are written with expected columns.
- Non-finite probabilities, regression outputs, attention weights, or selection weights fail validation.

Meta-ensemble tests:

- V3 can ingest TFT probability and optional diagnostic columns.
- V3 ignores actual future labels as features.
- Mixed dataset hashes are rejected.
- TFT interpretability-derived confidence features are namespaced.

Promotion tests:

- TFT must improve or not materially degrade walk-forward balanced accuracy.
- Brier/ECE must be acceptable.
- Overlay return delta must be finite.
- Drawdown impact must improve or not materially worsen.
- Turnover and reduced-exposure days must stay within configured bounds.

## Research-Only Boundary

TFT may be used for:

- Offline research runs.
- Shadow overlays.
- Interpretability reports.
- Meta-ensemble v3 features.
- Promotion-gate analysis.

TFT must not:

- Place orders.
- Modify broker adapters.
- Modify paper trading behavior.
- Modify live trading behavior.
- Modify execution behavior.
- Directly change operational portfolio sizing.
