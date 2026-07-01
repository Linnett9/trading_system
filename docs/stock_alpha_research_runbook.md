# Stock-Alpha Research Runbook

This runbook was generated from `PROJECT_TREE.txt`, the CLI/service command
surface, and stock-alpha config/stage module names. It is a practical guide for
running the stock-alpha research workflow safely. It does not claim that any
benchmark was run.

For deeper context, see:
[stock-alpha pipeline deep dive](stock_alpha_pipeline_deep_dive.md),
[data lineage](data_lineage.md),
[model catalog](model_catalog.md),
[metrics glossary](metrics_glossary.md), and
[research gates and trade conditions](research_gates_and_trade_conditions.md).

## Purpose

Stock-alpha research evaluates stock-level prediction artifacts, engineered
alpha features, ranking models, target definitions, portfolio replay, policy
sweeps, experiment reports, optional feature attribution, and an overnight
summary.

The workflow is research-only. Outputs should carry:

- `research_only: true`
- `trading_impact: none`
- `production_validated: false`
- `promotion_thresholds_changed: false`

These fields mean the result is evidence for review, not permission to trade.

## Python

Use the pyenv interpreter:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
```

Avoid system `python3` for project runs because dependencies may be missing.

## Run Sizes

| Run size | Purpose | Typical use |
|---|---|---|
| `dev` | Small, bounded smoke workflow | First pass after code/config changes. |
| `benchmark` | Standard benchmark profile | Main overnight research run. |
| `full` | Largest stock-alpha run size | Use only when dev and benchmark are healthy. |

The dev smoke command forces `stock_alpha_run_size=dev`, caps dates/symbols and
policy configs, sets model workers to 1, and disables attribution.

## Canonical Output Directories

Stock-alpha outputs should land under the stock-alpha report root:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/dev/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/full/
```

Do not rely on legacy outputs such as:

```text
reports/ml/benchmark/ml/
```

unless legacy output paths are explicitly enabled.

## Preflight

Set the interpreter and cap nested numeric-library threads:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Confirm the benchmark profile still requests 4 stock-alpha outer workers:

```bash
grep -E "stock_alpha_feature_n_jobs|stock_ranker_model_n_jobs|stock_target_comparison_n_jobs|stock_portfolio_policy_sweep_n_jobs|sklearn_n_jobs" configs/research/profiles/benchmark.yaml
```

Expected benchmark settings:

```text
stock_alpha_feature_n_jobs: 4
stock_ranker_model_n_jobs: 4
stock_target_comparison_n_jobs: 4
stock_portfolio_policy_sweep_n_jobs: 4
sklearn_n_jobs: 1
```

## Commands

### Dev Smoke

Run this before benchmark/full:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py --mode ml-stock-alpha-dev-smoke --profile benchmark
```

Expected report:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/dev/stock_alpha_dev_smoke_report.json
```

### Parallelism Audit

Run this before benchmark/full:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$PY" main.py --mode ml-stock-alpha-parallelism-audit --profile benchmark
```

Expected reports:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/stock_alpha_parallelism_audit.json
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/stock_alpha_parallelism_audit.md
```

### Benchmark Overnight Run

Use `caffeinate` for the benchmark overnight command:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
caffeinate -dimsu env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$PY" main.py --mode ml-overnight-stock-alpha --profile benchmark
```

Expected summary:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/overnight_stock_alpha_summary.json
```

### Monitor Logs

If the command writes a timestamped log in `logs/`, monitor it with:

```bash
tail -f logs/stock_alpha_benchmark_*.log
```

If no matching log exists, monitor the terminal output. The overnight runner
prints stage starts and ends.

### Inspect Reports

List canonical benchmark outputs:

```bash
find reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark -maxdepth 3 -type f | sort
```

Inspect key JSON files:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
"$PY" -m json.tool reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/overnight_stock_alpha_summary.json | less
"$PY" -m json.tool reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/stock_alpha_parallelism_audit.json | less
```

Verify the legacy output directory is not receiving new files during a run:

```bash
MARKER="$(mktemp)"
touch "$MARKER"
caffeinate -dimsu env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$PY" main.py --mode ml-overnight-stock-alpha --profile benchmark
find reports/ml/benchmark/ml -type f -newer "$MARKER" -print
```

The final `find` should print nothing for a canonical-only benchmark run.

## Stages

1. Stock-level artifact

   Builds `stock_level_prediction_artifacts.csv`,
   `stock_level_prediction_artifacts.json`, and
   `stock_level_prediction_artifacts.md`. These must be written under the
   canonical run-size output directory.

2. Alpha features

   Reads the canonical base artifact and writes
   `stock_level_prediction_artifacts_enriched.csv` plus
   `stock_level_alpha_feature_audit.{csv,json,md}`.

3. Model ranking benchmark

   Ranks models and writes `stock_level_model_ranking_benchmark.{csv,json,md}`
   and `stock_level_model_oos_predictions.csv`. Overnight runs produce baseline
   and enriched benchmark subdirectories.

4. Target comparison

   Compares configured target columns and writes
   `stock_level_target_comparison.{csv,json,md}`. Missing or insufficient
   targets should appear as skipped targets, not silent success.

5. Portfolio replay

   Replays portfolio policies over OOS predictions and writes summary, equity
   curves, and holdings outputs. This is research-only and should not place
   orders.

6. Portfolio policy sweep

   Sweeps policy settings and writes sweep summary, equity curves, and top
   holdings. Dev caps the number of policy configs; benchmark/full allow more.

7. Experiment report

   Validates expected outputs, output roots, run size, guardrails, and freshness.
   Writes `stock_alpha_experiment_report.{json,md}` and appends a registry row.

8. Optional attribution

   Writes feature attribution CSV/JSON/Markdown when enabled. It is disabled in
   dev smoke and disabled by default in config for overnight unless explicitly
   turned on.

9. Overnight summary

   Writes `overnight_stock_alpha_summary.{json,md}` with comparisons, winners,
   portfolio summaries, stage timings, parallelism, guardrails, output metadata,
   and artifact paths.

## Parallelism

The benchmark profile requests four effective outer workers for:

- `stock_alpha_feature_n_jobs: 4`
- `stock_ranker_model_n_jobs: 4`
- `stock_target_comparison_n_jobs: 4`
- `stock_portfolio_policy_sweep_n_jobs: 4`

It keeps nested workers capped:

- `sklearn_n_jobs: 1`
- `torch_num_threads: 1`
- `numpy_num_threads: 1`
- BLAS thread env vars set to `1`

Why: the outer stages are the intended parallelism layer. If each worker also
starts BLAS/sklearn/torch worker pools, CPU oversubscription can make the run
slower and less stable.

How to tell workers are active:

- Inspect `stock_alpha_parallelism_audit.json`.
- Check each stage's `requested_workers` and `effective_workers`.
- Inspect `overnight_stock_alpha_summary.json` `parallelism`.
- During a run, system monitors should show multiple Python workers during
  parallel-capable stages, but overnight stages themselves remain sequential.

## Key Fields to Inspect

In `overnight_stock_alpha_summary.json`:

- `output_root`
- `output_dir`
- `run_size`
- `legacy_output_paths_allowed`
- `artifacts`
- `artifact_status`
- `stage_timings`
- `parallelism`
- `comparisons`
- `winners`
- `portfolio_replay`
- `portfolio_policy_sweep`
- `research_only`
- `trading_impact`
- `production_validated`
- `promotion_thresholds_changed`

In experiment reports:

- `validation_passed`
- `validation.errors`
- `validation.warnings`
- `validation.unexpected_output_paths`
- `validation.legacy_output_paths_detected`
- `validation.legacy_output_paths_allowed`

In dev smoke reports:

- `status`
- `effective_caps`
- `target_column_availability`
- `skipped_targets`
- `policy_sweep_baseline_coverage`
- `experiment_validation`

## Common Failure Modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Import/dependency error with system Python | Wrong interpreter | Use `PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python`. |
| Alpha features cannot find base artifact | Stock artifact wrote outside canonical root | `artifact_status.path`, `artifacts.stock_artifact`, `output_dir`. |
| Legacy files are reused | Legacy paths enabled or stale path config | `stock_alpha_allow_legacy_output_paths`, experiment validation legacy fields. |
| Target comparison skips targets | Missing target columns or insufficient non-null data | `skipped_targets`, `target_column_availability`. |
| Policy sweep cannot compare baseline | Baseline signal missing from OOS predictions | `policy_sweep_baseline_coverage`. |
| Files appear in `reports/ml/benchmark/ml/` | Noncanonical output root | Run the marker/find command above and inspect stage output paths. |
| Oversubscription warning | Nested workers or total effective workers too high | Parallelism audit `oversubscription_warnings`. |
| `promotion_thresholds_changed` is true | Promotion gate changed or report corrupted | Stop and review the producing module/config before using results. |
| `production_validated` is true unexpectedly | Research output is claiming production status | Treat as a guardrail failure. |

## Safe Operating Order

1. Run focused tests for code changes.
2. Run dev smoke.
3. Run parallelism audit.
4. Run benchmark overnight with `caffeinate`.
5. Inspect summary, experiment report, target skips, policy sweep baseline
   coverage, and guardrails.
6. Consider full only after dev and benchmark are clean.
