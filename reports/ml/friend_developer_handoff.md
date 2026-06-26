# Friend Developer Handoff

## Current Goal

Build a research-first ML layer for an algorithmic trading system that can evaluate whether a dual-momentum portfolio should reduce exposure. The near-term goal is better data, better labels, better evaluation, and stronger research-only model comparison before anything is considered for paper/live trading.

Important safety rule: do not modify broker, paper trading, live trading, or execution code. Keep this ML work research-only.

## Architecture

- `core/`: domain and research logic. ML research lives under `core/research/ml/`.
- `application/`: command orchestration and runtime services.
- `infrastructure/`: data adapters and external file/provider integrations.
- `config/` and `configs/`: config defaults, validation, and experiment YAMLs.
- `data/`: raw/processed market data and generated universes.
- `reports/ml/`: research artifacts, leaderboards, diagnostics, reports.
- `tests/`: pytest coverage for research pipelines and data utilities.

## Implemented So Far

- Recursive Stooq bulk importer for official raw directory layout.
- Stooq Parquet data feed and data inventory.
- Real `us_liquid_100`, `us_liquid_250`, and `us_liquid_500` universe generation.
- Expanded rebalance dataset generation with duplicate-universe detection.
- Additional research labels including forward returns, future volatility, future drawdown, MAE, and MFE.
- DLinear, PatchTST, Transformer sequence models.
- Meta ensemble v2 over DLinear/PatchTST/Transformer prediction artifacts.
- Calibration comparison: raw, Platt, isotonic, temperature scaling.
- HTML reports, leaderboards, threshold sweeps, overlay summaries, and promotion-gate reporting.

## Dataset Status

- Processed Stooq Parquet symbols: 514
- Usable symbols after coverage/liquidity filters: 379
- `us_liquid_100`: real
- `us_liquid_250`: real
- `us_liquid_500`: currently 379 symbols because only 379 pass filters
- Expanded rebalance dataset: 9,552 rows

## ML Status

- DLinear implemented.
- PatchTST implemented.
- Transformer implemented.
- Meta ensemble v2 implemented.
- Meta ensemble v2 includes:
  - walk-forward evaluation
  - calibration comparison
  - threshold sweep
  - meta learner comparison
  - promotion gates
  - finite overlay sanity checks
  - drawdown, turnover, and reduced-exposure-day reporting

Meta learners currently supported:
- logistic regression
- ridge logistic
- random forest
- gradient boosting
- LightGBM if installed and usable

## Current Blocker

LightGBM is installed, but macOS is missing the `libomp` runtime dependency.

Fix one of two ways:

```bash
brew install libomp
```

or uninstall LightGBM if you do not want to support it:

```bash
python -m pip uninstall lightgbm
```

The meta ensemble handles unavailable LightGBM as optional, but a broken install can still block imports depending on the environment.

## Recommended Next Steps

1. Fix the LightGBM/libomp issue.
2. Rerun the meta ensemble after the dependency fix.
3. Rerun the full model leaderboard on the 9,552-row expanded rebalance dataset.
4. Implement iTransformer as a cross-asset ranker.
5. Implement Momentum Transformer as a trend/regime scorer.
6. Keep all ML work research-only until promotion gates and paper-trading review are intentionally designed.

## Key Commands

Import raw Stooq data:

```bash
python main.py --mode import-stooq-bulk --top 500 --asset-class stocks --min-rows 2000 --exclude-warrants-units-rights
```

Build inventory:

```bash
python main.py --mode ml-data-inventory
```

Build universes:

```bash
python main.py --mode ml-build-universes
```

Build expanded rebalance dataset:

```bash
python main.py --mode ml-expanded-rebalance-dataset --config configs/research/dlinear_should_reduce_exposure.yaml
```

Run source models:

```bash
python main.py --mode ml-research --config configs/research/dlinear_should_reduce_exposure.yaml
python main.py --mode ml-research --config configs/research/patchtst_should_reduce_exposure.yaml
python main.py --mode ml-research --config configs/research/transformer_should_reduce_exposure.yaml
```

Run meta ensemble:

```bash
python main.py --mode ml-meta-ensemble --config configs/research/regime_transformer_meta_ensemble_v1.yaml
```

Focused tests:

```bash
python -m pytest tests/test_meta_ensemble.py tests/test_ml_leaderboard.py tests/test_overlay_decision_rules.py tests/test_ml_research.py
```

## Key Files And Directories

- `reports/ml/transformer_development_handoff.md`: deeper technical handoff.
- `reports/ml/regime_transformer_meta_ensemble_v1/leaderboard.md`: current leaderboard.
- `reports/ml/regime_transformer_meta_ensemble_v1/`: meta ensemble v2 artifacts.
- `cache/ml/expanded_rebalance_dataset.csv`: expanded research dataset.
- `data/processed/stooq_parquet/`: processed Stooq Parquet files.
- `data/reference/universes/`: generated universes.
- `configs/research/dlinear_should_reduce_exposure.yaml`
- `configs/research/patchtst_should_reduce_exposure.yaml`
- `configs/research/transformer_should_reduce_exposure.yaml`
- `configs/research/regime_transformer_meta_ensemble_v1.yaml`
- `core/research/ml/experiment_runner.py`
- `core/research/ml/meta_ensemble.py`
- `core/research/ml/leaderboard.py`
- `core/research/ml/rebalance_dataset.py`
- `core/research/ml/models.py`
- `core/research/ml/dlinear_model.py`
- `core/research/ml/patchtst_model.py`
- `core/research/ml/transformer_model.py`
- `core/research/ml/calibration.py`
- `core/research/ml/universe_builder.py`
- `infrastructure/data/stooq_bulk_importer.py`
- `application/services/ml_commands.py`
- `application/services/stooq_bulk_commands.py`
- `config/config_loader.py`

## Watchouts

- Do not promote models based on one metric.
- Keep duplicate-universe detection active.
- Preserve new labels.
- Check label direction:
  - `should_reduce_exposure`: reduce when probability is greater than or equal to threshold.
  - `champion_success`: reduce when probability is below threshold.
- Treat overlay returns as decimal returns, not percentages.
- Promotion requires finite overlay math, reasonable calibration, walk-forward performance, and drawdown/turnover review.
