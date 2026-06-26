# Market Context Encoder Design

Research-only design note. Do not implement runtime code in this step.

## Purpose

The Market Context Encoder is a proposed research model that summarizes broad market state for use in regime classification, risk multiplier estimation, shadow overlays, and future meta-ensemble features.

Primary goals:

- Classify market regime from broad market, sector, volatility, drawdown, breadth, and liquidity context.
- Estimate a research-only risk multiplier that describes how compatible current conditions are with full momentum exposure.
- Provide compact context embeddings/features for meta-ensemble v3.

The encoder must not directly change broker, paper trading, live trading, execution, or portfolio sizing behavior.

## Proposed Model Type

Suggested future model type:

```yaml
ml:
  model_type: market_context_encoder
```

Suggested future config:

```text
configs/research/market_context_encoder_should_reduce_exposure.yaml
```

## Inputs

Use only values available at or before `feature_date`.

Market context features:

- SPY/QQQ returns over 5d, 21d, 63d, 126d, 252d.
- SPY/QQQ distance from moving averages.
- Realized volatility over 21d, 63d, 126d.
- Volatility ratios and volatility percentile/rank.
- Current drawdown from recent highs.
- Max drawdown over 63d and 126d.
- Breadth above SMA 50/100/200.
- Breadth change since last rebalance.
- Relative volume and dollar-volume deterioration.
- Sector trend agreement and sector concentration.
- Benchmark participation and leadership-filter context.
- Existing rule-based regime labels such as risk-on/risk-off, chop, drawdown guard, breadth-scaled exposure.

Do not use forward-looking labels as inputs:

- `forward_return_5d`
- `forward_return_10d`
- `future_volatility`
- `future_drawdown`
- `max_adverse_excursion`
- `max_favourable_excursion`

## Outputs

Primary outputs:

- `market_regime_probability_risk_off`
- `market_regime_class`
- `risk_multiplier`

Recommended auxiliary outputs:

- `volatility_regime_score`
- `drawdown_regime_score`
- `breadth_regime_score`
- `liquidity_regime_score`
- `context_embedding_*` fields for meta-ensemble research use.

Risk multiplier:

- Bounded continuous value, for example `[0.25, 1.25]`.
- Research-only estimate of exposure compatibility.
- Must not directly size positions in paper/live systems.

## Architecture

Initial design:

- Shared market context encoder over chronological feature rows.
- Classification head for regime class or risk-off probability.
- Regression head for `risk_multiplier`.
- Optional embedding projection head for meta-ensemble features.

Candidate encoder types:

- Compact MLP over current context features for a first baseline.
- Temporal Transformer over rolling market context.
- iTransformer-style feature-token encoder where each market/sector context series is a token.

Start simple:

- Build an MLP or small temporal encoder first.
- Compare against logistic regression and gradient boosting.
- Only add larger transformer capacity if walk-forward evidence justifies it.

## Training Targets

Possible classification targets:

- `should_reduce_exposure`
- `drawdown_risk`
- rule-based risk-off regime
- high future volatility regime
- negative future return regime

Possible risk multiplier target:

- Derived from forward drawdown/volatility/return outcomes.
- Example mapping:
  - High future drawdown or high volatility: lower multiplier.
  - Positive forward return and low drawdown: higher multiplier.
  - Flat/uncertain regime: neutral multiplier near `1.0`.

The target should be defined in research labels only and reported clearly.

## Artifact Outputs

Future `prediction_artifacts.csv` candidate columns:

- `market_regime_probability_risk_off`
- `market_regime_class`
- `risk_multiplier`
- `volatility_regime_score`
- `drawdown_regime_score`
- `breadth_regime_score`
- `liquidity_regime_score`
- `context_embedding_001`
- `context_embedding_002`
- `context_embedding_003`

Required provenance fields must remain:

- `dataset_hash`
- `source_dataset_row_count`
- `train_sample_count`
- `test_sample_count`
- `generated_at`

Future metadata fields:

- `market_context_encoder_version`
- `context_feature_groups`
- `risk_multiplier_bounds`
- `regime_target_definition`
- `research_only`
- `trading_impact: none`

## Meta-Ensemble Use

Meta-ensemble v3 can consume:

- `market_regime_probability_risk_off`
- `risk_multiplier`
- regime scores
- compact context embeddings

Suggested meta features:

- Source model probability crossed with risk-off probability.
- Momentum trend score crossed with market regime score.
- iTransformer rank score crossed with breadth regime score.
- Expected drawdown crossed with risk multiplier.
- Probability disagreement conditioned on market regime class.

Safety rule:

- Use predicted context outputs only. Do not use realized future regime outcomes as meta features.

## Validation And Leakage Rules

Rules:

- All context features must be computed using data available at or before `feature_date`.
- Target labels must start after `feature_date`.
- Same `rebalance_date` rows should not be split across train/test folds.
- Target standardization for risk multiplier regression must be fit on train folds only.
- Calibration for regime probabilities must not use final holdout labels.
- Artifacts must include dataset hash and provenance fields.
- Meta-ensemble consumers must reject mixed dataset hashes.

Leakage checks:

- Denylist input columns beginning with `forward_`, `future_`, `max_adverse_`, `max_favourable_`, or `actual_`.
- Audit feature date, label start date, and label end date ranges.
- Report missing or non-finite context features.
- Report regime target distribution by train/test split.

## Promotion Criteria

Do not promote solely on classification accuracy.

Minimum research evidence:

- Walk-forward balanced accuracy improves or does not materially degrade versus simple baselines.
- Brier score and ECE are acceptable for regime probability.
- Risk multiplier improves overlay drawdown or volatility-adjusted return in research-only shadow tests.
- Overlay return delta is finite.
- Max drawdown impact improves or does not materially worsen.
- Turnover and reduced-exposure days stay within configured bounds.
- Results are stable across folds and not only final holdout.

## Tests To Add Later

Model tests:

- Encoder forward pass returns regime probability, class, and risk multiplier.
- Risk multiplier is bounded.
- Save/load preserves output shape.
- Invalid config fails clearly.

Feature/leakage tests:

- Future-looking columns are rejected as inputs.
- Feature dates precede label windows.
- Same rebalance date is not split across folds.

Artifact tests:

- Prediction artifacts include context output columns and provenance.
- Non-finite outputs fail before reports are written.
- Metadata includes context feature groups and multiplier bounds.

Meta-ensemble tests:

- v3 can ingest context outputs as optional source features.
- Actual future regime labels are excluded from meta features.
- Mixed dataset hashes are rejected.

Promotion tests:

- Selection considers balanced accuracy, calibration, overlay return delta, drawdown impact, turnover, and reduced-exposure days.
- Candidate with high accuracy but damaging overlay behavior is not selected as the overlay model.

## Research-Only Boundary

The Market Context Encoder may support:

- Research reports.
- Shadow overlays.
- Meta-ensemble v3 features.
- Promotion-gate analysis.

It must not:

- Place orders.
- Modify broker adapters.
- Modify paper trading services.
- Modify live trading services.
- Modify execution models.
- Directly change portfolio sizing in operational flows.
