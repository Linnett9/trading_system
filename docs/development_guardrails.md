# Development Guardrails

This guide was generated from `PROJECT_TREE.txt`, config/profile names, and the
current research command surface. It is meant to keep research work reproducible
and separated from paper/live/order execution.

For detailed stock-alpha acceptance boundaries, see
[research gates and trade conditions](research_gates_and_trade_conditions.md).

## Python and Dependencies

Use pyenv Python 3.11.6 for project commands:

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
```

Avoid system `python3` because dependencies may be missing.

Do not install packages or create virtual environments during implementation
unless explicitly requested. Use the existing project interpreter and existing
environment.

## Long Runs

- Do not run benchmarks unless explicitly requested.
- Do not run long commands for documentation-only or small implementation
  tasks.
- Do not edit Python source while long benchmark jobs are running.
- Use focused tests during implementation.
- Run dev smoke before benchmark/full stock-alpha runs.
- Run the parallelism audit before benchmark/full stock-alpha runs.

## Research vs Execution

Keep broker, paper, live, and order execution code isolated.

Research-only modules must not place trades. In particular, stock-alpha research
under `core/research/ml/stock_level/` should remain separate from:

- `application/services/paper_*`
- `core/paper/`
- `core/entities/order.py`
- `core/execution/`
- `infrastructure/broker/`
- `infrastructure/brokers/`

If a research module needs execution simulation, prefer explicit abstractions
and tests. Do not hide broker or order-placement logic inside metrics,
reporting, portfolio replay, or model code.

## Promotion and Production Guardrails

Do not change promotion thresholds unless explicitly requested.

Research additions should keep:

```text
promotion_thresholds_changed: false
production_validated: false
research_only: true
trading_impact: none
```

`production_validated: false` means the output is not approved for live or paper
trading. Treat it as research evidence only.

If a report shows `promotion_thresholds_changed: true`, stop and inspect the
producing module or config before using the output.

## Generated Outputs

Generated reports, logs, and cache files generally should not be committed.

Common generated locations:

- `cache/`
- `reports/`
- `logs/`
- `data/processed/`

Do not delete or move generated outputs during a task unless explicitly asked.
For path bugs, validate new output roots without silently copying old files.

## Stock-Alpha Output Roots

Canonical stock-alpha directories:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/dev/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/full/
```

Do not rely on legacy output paths unless explicitly enabled:

```text
reports/ml/benchmark/ml/
```

Resume/report validation should check canonical files when
`stock_alpha_allow_legacy_output_paths: false`.

## Parallelism Guardrails

Benchmark stock-alpha parallelism should keep outer stage workers at 4 and
nested worker/thread caps at 1:

```text
stock_alpha_feature_n_jobs: 4
stock_ranker_model_n_jobs: 4
stock_target_comparison_n_jobs: 4
stock_portfolio_policy_sweep_n_jobs: 4
sklearn_n_jobs: 1
```

For benchmark/full shell runs, cap BLAS-style threads:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Do not loosen promotion gates or parallelism caps as part of unrelated fixes.

## Implementation Hygiene

- Inspect existing files before coding.
- Make minimal targeted changes.
- Add or update focused tests for Python code changes.
- Do not touch broker/paper/live/order execution code for research-only fixes.
- Avoid broad refactors.
- Preserve existing behaviour unless a task explicitly changes it.
- Do not edit config defaults unless the task asks for it.
- Do not change tests for docs-only work.
- For docs-only work, edit only documentation files.

## Safe Stock-Alpha Sequence

```bash
PY=/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" main.py --mode ml-stock-alpha-dev-smoke --profile benchmark
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$PY" main.py --mode ml-stock-alpha-parallelism-audit --profile benchmark
```

Only after those are healthy, retry benchmark/full as requested.
