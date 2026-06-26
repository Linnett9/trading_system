# Multi-Task Transformer Smoke Runbook

Research-only. No broker, paper trading, live trading, or execution code is involved.

## 1. Run the Current-32 Smoke Experiment

```bash
~/.pyenv/versions/3.11.6/bin/python main.py --mode ml-research --config configs/research/multitask_transformer_32_symbol_smoke.yaml
```

Outputs:

- `reports/ml/multitask_transformer_32_symbol_smoke/prediction_artifacts.csv`
- `reports/ml/multitask_transformer_32_symbol_smoke/prediction_artifacts.json`
- `cache/ml/multitask_transformer_32_symbol_smoke/expanded_rebalance_dataset.csv`

The smoke config uses:

- current-32 universe only
- one variant: monthly, top 3, equal weighting
- CPU training
- short sequence length
- dedicated cache path: `cache/ml/multitask_transformer_32_symbol_smoke`

## 2. Artifact Columns for Meta-Ensemble v3

Safe source columns:

- `raw_probability`
- `calibrated_probability`
- `predicted_forward_return_5d`
- `predicted_forward_return_10d`
- `predicted_future_volatility`
- `predicted_future_drawdown`

Evaluation-only columns:

- `actual_label`
- `actual_forward_return_5d`
- `actual_forward_return_10d`
- `actual_future_volatility`
- `actual_future_drawdown`

Meta-ensemble ingestion must use only source probabilities and `predicted_*` columns. It must ignore `actual_*`, `future_*`, `forward_return_*`, and label-window columns unless they are model-prefixed predictions.

## 3. Meta-Ensemble v3 Smoke Wiring

The current meta ensemble requires at least two source prediction directories with the same `dataset_hash`. To run a v3 smoke comparison, generate another source model against the same current-32 smoke cache/dataset, then configure:

```yaml
ml:
  model_type: meta_ensemble
  ensemble_name: meta_ensemble_v3_current32_smoke
  label_type: should_reduce_exposure
  output_dir: reports/ml/meta_ensemble_v3_current32_smoke
  expanded_rebalance_dataset_path: cache/ml/multitask_transformer_32_symbol_smoke/expanded_rebalance_dataset.csv
  meta_dataset_path: cache/ml/multitask_transformer_32_symbol_smoke/meta_ensemble_v3_dataset.csv
  source_prediction_dirs:
    - reports/ml/multitask_transformer_32_symbol_smoke
    - reports/ml/<second_current32_smoke_model>
```

Then run:

```bash
~/.pyenv/versions/3.11.6/bin/python main.py --mode ml-meta-ensemble --config <meta_ensemble_v3_smoke_config.yaml>
```

The ingestion path now prefixes auxiliary prediction features by source model, for example:

- `multitask_transformer_predicted_forward_return_5d`
- `multitask_transformer_predicted_future_volatility`

It records ignored leakage candidates in `meta_dataset_audit.json`.
