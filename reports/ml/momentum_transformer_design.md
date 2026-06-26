# Momentum Transformer Design

Research-only design note. Do not modify broker, paper trading, live trading, execution, runtime experiment orchestration, meta ensemble, leaderboard, or artifact code as part of this design step.

## Purpose

Momentum Transformer is a proposed research model for estimating whether current leadership is durable enough to keep risk exposure high. It should complement the current should-reduce-exposure classifiers by producing interpretable trend/regime signals, not direct trade instructions.

Primary purposes:

- Estimate trend strength across the selected universe and benchmarks.
- Estimate regime compatibility between current market conditions and momentum/dual-momentum exposure.
- Produce a position-size multiplier for research overlays and future meta-ensemble features.

It should remain research-only. Its outputs may be used in reports, shadow overlays, and meta-ensemble experiments, but must not directly control broker, paper, live, or execution behavior.

## Model Type

Proposed model type:

```yaml
ml:
  model_type: momentum_transformer
```

Proposed config path:

```text
configs/research/momentum_transformer_should_reduce_exposure.yaml
```

## How It Differs

| Model | Current Role | Momentum Transformer Difference |
|---|---|---|
| DLinear | Linear rolling-window sequence baseline. | Momentum Transformer should learn nonlinear regime/trend interactions and output trend/regime/size diagnostics, not only a binary probability. |
| PatchTST | Temporal patch transformer over rolling feature windows. | PatchTST focuses on temporal patches. Momentum Transformer should explicitly encode momentum horizons, volatility/drawdown state, and regime compatibility. |
| Transformer | Generic sequence classifier over historical feature rows. | Momentum Transformer should be domain-shaped around trend persistence, risk regime, and exposure scaling rather than generic sequence classification. |
| iTransformer | Cross-feature/cross-asset token attention over inverted feature tokens. | Momentum Transformer should be more decision-contextual: it should score whether the prevailing trend supports the current exposure/ranking regime and how much size should be scaled. |

## Proposed Config

Initial research config:

```yaml
backtest:
  provider: stooq_parquet
  data_dir: data/processed/stooq_parquet

ml:
  enabled: true
  mode: research
  model_type: momentum_transformer
  shadow_model_type: momentum_transformer
  label_type: should_reduce_exposure
  prediction_target: should_reduce_exposure
  feature_set: expanded_rebalance_v1
  research_label: EXPANDED_REBALANCE_CURRENT_32_RESEARCH_ONLY
  minimum_history_years: 4
  output_dir: reports/ml/momentum_transformer_should_reduce_exposure

  benchmark_symbols:
    - SPY
    - QQQ

  sequence_length: 64
  momentum_transformer_sequence_length: 64
  momentum_transformer_d_model: 64
  momentum_transformer_heads: 4
  momentum_transformer_layers: 2
  momentum_transformer_feedforward: 128
  momentum_transformer_dropout: 0.15
  momentum_transformer_epochs: 30
  momentum_transformer_batch_size: 64
  momentum_transformer_learning_rate: 0.001
  momentum_transformer_weight_decay: 0.0005
  momentum_transformer_pos_weight: auto

  momentum_transformer_trend_horizons: [21, 63, 126, 252]
  momentum_transformer_volatility_horizons: [21, 63]
  momentum_transformer_drawdown_horizons: [63, 126]
  momentum_transformer_size_multiplier_floor: 0.25
  momentum_transformer_size_multiplier_ceiling: 1.25

  comparison_models:
    - momentum_transformer
  overlay_comparison_models:
    - momentum_transformer
```

## Input Features

Use only features available at or before each `feature_date`.

Feature groups:

- Returns: 1d/5d/10d/21d/63d/126d/252d returns, excess returns versus SPY/QQQ, selected basket returns.
- Volatility: realized volatility over 21d/63d/126d, volatility ratios, volatility percentile/rank.
- ATR: ATR level, ATR percent of price, ATR trend/change, ATR percentile if available.
- Drawdown: current drawdown from recent highs, max drawdown over 63d/126d, drawdown recovery state.
- Relative volume: volume ratio versus moving average, dollar-volume rank, liquidity deterioration flags.
- Market and sector context: SPY/QQQ trend, breadth above SMA, sector concentration, sector trend agreement, benchmark participation.
- Regime labels/context: current rule-based regime label, risk-on/risk-off flags, drawdown guard state, chop filter state, breadth-scaled exposure state.

Avoid forward labels such as `forward_return_5d`, `future_volatility`, `future_drawdown`, `max_adverse_excursion`, and `max_favourable_excursion` as model inputs. Those are target/evaluation labels only.

## Outputs

Required model/report outputs:

- `probability_should_reduce_exposure`: probability that exposure should be reduced for the next rebalance horizon.
- `trend_score`: continuous score, preferably `[-1, 1]`, where positive means trend persistence supports exposure.
- `regime_score`: continuous score, preferably `[0, 1]`, estimating compatibility between current market regime and momentum exposure.
- `size_multiplier`: bounded exposure multiplier, for example `[0.25, 1.25]`, for research overlays only.

Suggested derivation:

- `probability_should_reduce_exposure`: classifier head trained on `should_reduce_exposure`.
- `trend_score`: auxiliary regression/ranking head trained on future excess return or trend persistence proxy.
- `regime_score`: auxiliary head trained on realized regime compatibility, drawdown avoidance, or rule-based regime outcomes.
- `size_multiplier`: deterministic bounded mapping from probability/trend/regime outputs, not a direct trading command.

## Artifact Columns To Add Later

When runtime implementation happens, extend prediction artifacts only after the model is implemented and tests are ready.

Candidate `prediction_artifacts.csv` columns:

- `momentum_transformer_probability_should_reduce_exposure`
- `momentum_transformer_trend_score`
- `momentum_transformer_regime_score`
- `momentum_transformer_size_multiplier`
- `momentum_transformer_trend_bucket`
- `momentum_transformer_regime_bucket`
- `momentum_transformer_size_bucket`
- `momentum_transformer_aux_loss`
- `momentum_transformer_trend_horizons`
- `momentum_transformer_feature_group_version`

Existing provenance fields must remain:

- `dataset_hash`
- `source_dataset_row_count`
- `train_sample_count`
- `test_sample_count`
- `generated_at`

## Meta-Ensemble V3 Integration

Momentum Transformer should feed meta-ensemble v3 as a source model plus diagnostic feature provider.

Suggested source-model fields:

- Raw/calibrated `probability_should_reduce_exposure`.
- `trend_score`.
- `regime_score`.
- `size_multiplier`.
- Trend/regime buckets.

Suggested meta features:

- Disagreement between Momentum Transformer and DLinear/PatchTST/Transformer/iTransformer probabilities.
- Trend score minus recent champion excess return.
- Regime score crossed with volatility/drawdown state.
- Size multiplier crossed with reduced-exposure days/turnover penalty.
- Calibration quality of the probability head.

Selection policy:

- Do not assume the best classifier is the best overlay model.
- Report separate best classifier, best calibrated model, and best overlay utility model.
- Promotion gates should include walk-forward balanced accuracy, Brier/ECE, overlay return delta, drawdown impact, turnover, reduced-exposure days, and finite sanity checks.

## Tests To Add

Future implementation tests:

- Registry: `build_ml_model("momentum_transformer")` returns the correct research model class.
- Forward pass: model returns one probability, trend score, regime score, and size multiplier per input row.
- Output bounds: probability in `[0, 1]`, regime score in `[0, 1]`, trend score in configured range, size multiplier within configured floor/ceiling.
- Config validation: invalid sequence length, head count, `d_model % heads`, and invalid multiplier bounds fail clearly.
- Save/load: saved model reloads and predicts the same output shape.
- Artifact generation: end-to-end `MLExperimentRunner.run()` writes prediction artifacts with provenance and Momentum Transformer diagnostic columns after artifact support is intentionally added.
- Meta v3 ingestion: meta dataset builder reads Momentum Transformer probability and auxiliary diagnostic fields without mixing dataset hashes.
- Leakage tests: features from future labels are excluded from model inputs.
- Walk-forward tests: folds train only on prior dates and keep same-date rebalance groups together.

## Leakage Risks

Main risks:

- Accidentally using future labels as inputs, especially `future_*`, `forward_*`, MAE/MFE, or next-period return fields.
- Same rebalance date appearing in both train and test folds.
- Sector or universe statistics computed using symbols/data unavailable at `feature_date`.
- Size multiplier being tuned on final holdout overlay results instead of walk-forward-only evaluation.
- Calibrating probability heads on test rows.
- Using source prediction artifacts from mismatched dataset hashes in meta-ensemble v3.

Validation rules:

- Input feature allowlist or denylist must exclude forward-looking columns.
- Chronological split must purge overlapping label windows.
- Walk-forward evaluation must group by `rebalance_date`.
- All generated artifacts must include dataset hash and sample-count provenance.
- Meta-ensemble v3 must reject mixed dataset hashes.
- Report train/test date ranges and overlay sample counts for every Momentum Transformer run.

## Research-Only Boundary

Momentum Transformer must not:

- Place orders.
- Change broker adapters.
- Change paper trading behavior.
- Change live trading behavior.
- Change execution models.
- Directly modify portfolio sizing in production or paper flows.

Allowed uses:

- Research reports.
- Shadow overlays.
- Offline leaderboard comparisons.
- Meta-ensemble v3 research features.
- Promotion-gate evidence.
