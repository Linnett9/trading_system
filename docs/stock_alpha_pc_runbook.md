# Stock-Alpha PC / Full-Dataset Runbook

This runbook is for future PC/full-dataset stock-alpha experiments. It is
research-only guidance and does not imply that any full run has been executed.

Expected guardrails for every step:

- `research_only: true`
- `trading_impact: none`
- `production_validated: false`
- `promotion_thresholds_changed: false`

Use the project Python and cap nested numeric-library threads:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Do not touch broker, paper, live, order execution, or production portfolio code
for this workflow.

## Sequence

| Step | Purpose | Config | Safe to run now | Overnight suitable |
|---|---|---|---|---|
| 1 | Full stock-alpha enriched benchmark generation | `config/config.yaml` as template; create/review a dedicated full config before running | No | Yes, after config review |
| 2 | Full ensemble generation | `config/config.stock_alpha_ensemble_full_enriched.yaml` | No, waits for step 1 output | Yes |
| 3 | Preflight full-enriched portfolio coarse | `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml` | Yes, inspection only; expected to fail until step 2 output exists | No |
| 4 | Run full-enriched portfolio coarse | `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml` | No, waits for step 2 output | Usually no; coarse is intended to be manageable |
| 5 | Inspect realized exposure, drawdown, underinvestment | coarse sweep outputs | Yes after step 4 | No |
| 6 | Run full-enriched refine | `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_refine.yaml` | No, only after step 5 is sane | Possibly |
| 7 | Run full-grid overnight only if coarse/refine are sane | `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_full_grid.yaml` | No | Yes |
| 8 | Generate news features after real news contract exists | `config/config.stock_alpha_news_features_full_template.yaml` | No, waits for `data/news/stock_alpha_news_contract.csv` | Possibly |
| 9 | Run news transformer readiness preflight | `config/config.stock_alpha_news_readiness_preflight_tiny_fixture.yaml` as command template | Yes, inspection only; expected not safe while disabled | No |
| 10 | Keep news transformer disabled until PIT features pass gates | same news config family | No model enablement by default | No |

## 1. Full Stock-Alpha Enriched Benchmark

Current repository state has benchmark/dev stock-alpha configs and downstream
full ensemble/sweep configs, but no dedicated full overnight stock-alpha config.
Before running on a PC, create or review a full config derived from
`config/config.yaml` that explicitly sets:

```yaml
ml:
  stock_alpha_run_size: full
  research_only: true
  trading_impact: none
  production_validated: false
  promotion_thresholds_changed: false
```

Command shape:

```bash
"$PY" main.py \
  --mode ml-overnight-stock-alpha \
  --config <reviewed-full-stock-alpha-config.yaml>
```

Expected input files:

- project stock-alpha data inputs already used by the benchmark workflow
- reviewed full stock-alpha config

Expected output files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/full/enriched/stock_level_model_oos_predictions.csv`
- full stock-alpha summary and manifest files under `.../stock_alpha/full/`

Approximate run size: full dataset, heavy.

Safe to run now: no. Create/review a dedicated full config first.

## 2. Full Ensemble Generation

Config:

- `config/config.stock_alpha_ensemble_full_enriched.yaml`

Command:

```bash
"$PY" main.py \
  --mode ml-stock-alpha-ensemble \
  --config config/config.stock_alpha_ensemble_full_enriched.yaml
```

Expected input files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/full/enriched/stock_level_model_oos_predictions.csv`

Expected output files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_ensemble_full/full/ensemble/average_rank/stock_alpha_ensemble_average_rank_predictions.csv`
- ensemble evaluation JSON, leaderboard CSV, and Markdown report under `.../stock_alpha_ensemble_full/full/ensemble/`

Approximate run size: full OOS prediction rows, moderate to heavy.

Safe to run now: no, unless the full enriched OOS predictions file exists.

Suitable for overnight: yes.

## 3. Preflight Full-Enriched Portfolio Coarse

Config:

- `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml`

Command:

```bash
"$PY" main.py \
  --mode ml-stock-alpha-experiment-preflight \
  --config config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml
```

Expected input files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_ensemble_full/full/ensemble/average_rank/stock_alpha_ensemble_average_rank_predictions.csv`

Expected output files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse/full/preflight/stock_alpha_experiment_preflight.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse/full/preflight/stock_alpha_experiment_preflight.md`

Approximate run size: inspection only; no portfolio sweep.

Safe to run now: yes. It should report `safe_to_run: false` until the full
ensemble predictions file exists.

Suitable for overnight: no.

## 4. Run Full-Enriched Portfolio Coarse

Config:

- `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml`

Command:

```bash
"$PY" main.py \
  --mode ml-stock-alpha-ensemble-portfolio-sweep \
  --config config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml
```

Expected input files:

- full ensemble predictions from step 2

Expected output files:

- `.../stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse/full/portfolio_sweep/ensemble/policy_sweep_raw.csv`
- `.../policy_sweep_ranked.csv`
- `.../stock_alpha_ensemble_portfolio_policy_sweep.json`
- `.../stock_alpha_ensemble_portfolio_policy_sweep.md`
- top-policy detail files only

Approximate run size: coarse grid, about 72 policies.

Safe to run now: no, wait for successful preflight.

Suitable for overnight: usually no; this is the first live PC sanity pass.

## 5. Inspect Coarse Results

Inspect:

- realized exposure bucket, not only target exposure bucket
- `underinvested_policy`
- `exposure_utilization_ratio`
- max drawdown
- cost-adjusted Sharpe
- cost-adjusted return
- top-N violation diagnostics
- whether holdings/trades are top-policy-scoped only

Command:

```bash
"$PY" -m json.tool \
  reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse/full/portfolio_sweep/ensemble/stock_alpha_ensemble_portfolio_policy_sweep.json \
  | less
```

Safe to run now: only after step 4.

## 6. Run Full-Enriched Refine

Config:

- `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_refine.yaml`

Command:

```bash
"$PY" main.py \
  --mode ml-stock-alpha-ensemble-portfolio-sweep \
  --config config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_refine.yaml
```

Expected input files:

- full ensemble predictions from step 2
- coarse report reviewed and sane

Expected output files:

- `.../stock_alpha_portfolio_sweep_ensemble_full_enriched_refine/full/portfolio_sweep/ensemble/policy_sweep_raw.csv`
- `.../policy_sweep_ranked.csv`
- `.../stock_alpha_ensemble_portfolio_policy_sweep.json`
- `.../stock_alpha_ensemble_portfolio_policy_sweep.md`

Approximate run size: refine grid, about 256 policies.

Safe to run now: no, only after coarse is sane.

Suitable for overnight: possibly, depending on PC runtime.

## 7. Full Grid Overnight

Config:

- `config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_full_grid.yaml`

Command:

```bash
caffeinate -dimsu env \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$PY" main.py \
  --mode ml-stock-alpha-ensemble-portfolio-sweep \
  --config config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_full_grid.yaml
```

Expected input files:

- full ensemble predictions from step 2
- coarse and refine reports reviewed and sane

Expected output files:

- full-grid policy sweep raw/ranked CSV, JSON, Markdown, and top-policy detail files under `.../stock_alpha_portfolio_sweep_ensemble_full_enriched_full_grid/full/portfolio_sweep/ensemble/`

Approximate run size: full grid, about 1080 policies.

Safe to run now: no.

Suitable for overnight: yes, only after coarse/refine are sane.

## 8. News Feature Generation

Config:

- `config/config.stock_alpha_news_features_full_template.yaml`

Command:

```bash
"$PY" main.py \
  --mode ml-stock-alpha-news-features \
  --config config/config.stock_alpha_news_features_full_template.yaml
```

Expected input files:

- `data/news/stock_alpha_news_contract.csv`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_ensemble_full/full/ensemble/average_rank/stock_alpha_ensemble_average_rank_predictions.csv`

Expected output files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_features_full/full/stock_alpha_news_features.csv`
- `.../stock_alpha_news_features_full/full/news_features/stock_alpha_news_features_audit.json`
- `.../stock_alpha_news_features_full/full/news_features/stock_alpha_news_features_audit.md`

Approximate run size: proportional to full stock/date rows and real article
count.

Safe to run now: no, only after real point-in-time
`data/news/stock_alpha_news_contract.csv` exists.

Suitable for overnight: possibly.

## 9. News Transformer Gate

`news_analysis_transformer` remains disabled unless:

- `stock_alpha_news_enable_transformer: true` is explicitly set in a reviewed config
- the real point-in-time news contract validates
- generated news features contain all required aggregate columns
- symbol/date coverage thresholds pass
- no synthetic zero-news fake coverage is present
- no future article leakage is detected

Default safe state:

```yaml
stock_alpha_news_enable_transformer: false
```

Do not enable it in full/all-deep/portfolio configs until the news feature
validation reports are clean.

## Local Tiny News Fixture Smoke

This smoke path uses committed tiny fixtures under
`tests/fixtures/stock_alpha_news/`. It proves the local point-in-time news
feature and readiness gates work end to end. It is not evidence of model
performance and must not be used for promotion decisions.

Generate tiny fixture news features:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-features \
  --config config/config.stock_alpha_news_features_tiny_fixture.yaml
```

Run the research-only readiness preflight before attempting any
`news_analysis_transformer` diagnostics or training:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-readiness-preflight \
  --config config/config.stock_alpha_news_readiness_preflight_tiny_fixture.yaml
```

Expected output files:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_readiness_preflight_tiny_fixture/dev/stock_alpha_news_readiness_preflight.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_readiness_preflight_tiny_fixture/dev/stock_alpha_news_readiness_preflight.md`

The preflight does not train a model. If `stock_alpha_news_enable_transformer`
is false, `safe_to_train_news_transformer` must be false.

Run the disabled readiness diagnostic. This should report
`stock_alpha_news_enable_transformer_false`:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-deep-diagnostics \
  --config config/config.stock_alpha_dev_diagnostic_news_transformer_tiny_fixture_disabled.yaml \
  --profile development
```

Run the enabled-readiness diagnostic only for fixture smoke validation. This
keeps the fixture config isolated and research-only:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-deep-diagnostics \
  --config config/config.stock_alpha_dev_diagnostic_news_transformer_tiny_fixture_enabled.yaml \
  --profile development
```

Expected fixture inputs:

- `tests/fixtures/stock_alpha_news/news_contract_tiny.csv`
- `tests/fixtures/stock_alpha_news/stock_rows_tiny.csv`

Expected fixture outputs:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_features_tiny_fixture/dev/stock_alpha_news_features.csv`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_features_tiny_fixture/dev/news_features/stock_alpha_news_features_audit.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_deep_diagnostic_tiny_news/dev/deep_diagnostics/news_analysis_transformer/stock_alpha_deep_model_diagnostics.json`

## Tiny Raw News Ingest Smoke

This smoke path starts from a committed raw-provider-style export and proves the
safe plumbing only: raw export to canonical PIT contract, generated features,
and disabled readiness preflight. It does not train a model and is not evidence
of model performance.

Ingest the tiny raw provider/export fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-contract-ingest \
  --config config/config.stock_alpha_news_contract_ingest_tiny_fixture.yaml
```

Generate features from the canonical contract produced by ingest:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-features \
  --config config/config.stock_alpha_news_features_tiny_ingest_fixture.yaml
```

Run readiness preflight. It should remain not safe while
`stock_alpha_news_enable_transformer: false`:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-readiness-preflight \
  --config config/config.stock_alpha_news_readiness_preflight_tiny_ingest_fixture.yaml
```

Expected tiny-ingest outputs:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_contract_ingest_tiny_fixture/dev/stock_alpha_news_contract.csv`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_contract_ingest_tiny_fixture/dev/news_contract_ingest/stock_alpha_news_contract_ingest_audit.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_features_tiny_ingest_fixture/dev/stock_alpha_news_features.csv`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_readiness_preflight_tiny_ingest_fixture/dev/stock_alpha_news_readiness_preflight.json`

## Provider Column Mapping

Real provider exports should be adapted through
`stock_alpha_news_provider_column_map` instead of manually editing CSV files.
The ingest step maps provider-specific headers into the canonical point-in-time
contract, then applies the same validation, dedupe, timestamp, and event-type
rules.

Example:

```yaml
ml:
  stock_alpha_news_provider_column_map:
    article_id: id
    symbol: ticker
    published_at_utc: published_at
    source: provider
    headline: title
    body_or_summary: summary
    sentiment_score: sentiment
    relevance_score: relevance
    novelty_score: novelty
    event_type: category
    language: lang
    ingested_at: collected_at
```

The safe real-data sequence is unchanged: ingest raw provider/export news,
generate canonical news features, run readiness preflight, and only then run the
enabled diagnostic template if the preflight is safe.

## Real PIT News Development Templates

These templates are placeholders for a future real point-in-time news archive.
They do not assume `data/news/stock_alpha_news_contract.csv` exists, and they
do not enable the transformer by default.

1. Ingest a raw provider/export news CSV into the canonical PIT contract:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-contract-ingest \
  --config config/config.stock_alpha_news_contract_ingest_real_template.yaml
```

Expected outputs:

- `data/news/stock_alpha_news_contract.csv`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_contract_ingest_real/dev/news_contract_ingest/stock_alpha_news_contract_ingest_audit.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_contract_ingest_real/dev/news_contract_ingest/stock_alpha_news_contract_ingest_audit.md`

If the raw source file is missing or required columns are absent, the ingest
must fail clearly and must not write a fake contract.

2. Generate canonical news features after the PIT contract and stock rows exist:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-features \
  --config config/config.stock_alpha_news_features_real_template.yaml
```

3. Run the readiness preflight with the transformer still disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-news-readiness-preflight \
  --config config/config.stock_alpha_news_readiness_preflight_real_template.yaml
```

4. Inspect the ingest audit, feature audit, and preflight outputs:

- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_contract_ingest_real/dev/news_contract_ingest/stock_alpha_news_contract_ingest_audit.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_features_real/dev/news_features/stock_alpha_news_features_audit.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_readiness_preflight_real/dev/stock_alpha_news_readiness_preflight.json`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_news_readiness_preflight_real/dev/stock_alpha_news_readiness_preflight.md`

The preflight should remain not-safe while
`stock_alpha_news_enable_transformer: false`. Review missing columns, coverage,
PIT audit metadata, and any blocking issues before changing templates.

Only after preflight passes, use the enabled diagnostic template:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-deep-diagnostics \
  --config config/config.stock_alpha_dev_diagnostic_news_transformer_real_enabled_template.yaml \
  --profile development
```

The default diagnostic template for real data is disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py \
  --mode ml-stock-alpha-deep-diagnostics \
  --config config/config.stock_alpha_dev_diagnostic_news_transformer_real_disabled_template.yaml \
  --profile development
```

Both diagnostic templates are dev-sized and limited to
`news_analysis_transformer`; neither should be promoted or used in benchmark/full
configs without separate review.
