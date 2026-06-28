# Expanded Rebalance Dataset Row Count Diagnostic

Generated: 2026-06-25

Scope: diagnostics only. No generation logic, model logic, broker, paper trading, live trading, or execution code was changed for this report.

## Summary

The drop from approximately 9,564 rows has two distinct causes:

1. Intentional duplicate-universe detection removes 4,782 duplicate rows in the 10-year ML research path.
2. The standalone `ml-expanded-rebalance-dataset` command uses `backtest.years=5`, while `ml-research` applies `ml.research_years=10`. That shorter history window reduces the deduplicated dataset from 4,782 rows to 2,070 rows.

The new research labels did not remove rows. All current generated rows have non-empty values for:

- `forward_return_5d`
- `forward_return_10d`
- `future_volatility`
- `future_drawdown`
- `future_max_drawdown`
- `max_adverse_excursion`
- `max_favourable_excursion`
- `should_reduce_exposure`

## Current Artifact State

Current `cache/ml/expanded_rebalance_dataset.csv` after the later DLinear research run:

| Metric | Count |
|---|---:|
| Dataset rows | 4,782 |
| Audit rows | 4,782 |
| Variant definitions | 36 |
| Non-skipped variants | 18 |
| Skipped variants | 18 |
| Skipped reason | `duplicate_universe_symbol_set` |

## Rows By Universe

10-year `ml-research` path, after duplicate detection:

| Universe | Rows |
|---|---:|
| `current_32` | 4,782 |
| `us_liquid_100` | 0 |

5-year standalone `ml-expanded-rebalance-dataset` path, after duplicate detection:

| Universe | Rows |
|---|---:|
| `current_32` | 2,070 |
| `us_liquid_100` | 0 |

The universe sets are currently identical after rebuilding from the available local Parquet inventory. `current_32` and `us_liquid_100` contain the same 32 symbols, just in different order. Because duplicate detection sorts the available symbol set, order does not matter.

## Rows By Rebalance Frequency

10-year `ml-research` path:

| Frequency | Rows |
|---|---:|
| `monthly` | 636 |
| `biweekly` | 1,386 |
| `weekly` | 2,760 |

5-year standalone path:

| Frequency | Rows |
|---|---:|
| `monthly` | 276 |
| `biweekly` | 600 |
| `weekly` | 1,194 |

## Rows By Top N

10-year `ml-research` path:

| Top N | Rows |
|---:|---:|
| 3 | 1,594 |
| 5 | 1,594 |
| 7 | 1,594 |

5-year standalone path:

| Top N | Rows |
|---:|---:|
| 3 | 690 |
| 5 | 690 |
| 7 | 690 |

## Rows By Weighting

10-year `ml-research` path:

| Weighting | Rows |
|---|---:|
| `equal` | 2,391 |
| `inverse_volatility` | 2,391 |

5-year standalone path:

| Weighting | Rows |
|---|---:|
| `equal` | 1,035 |
| `inverse_volatility` | 1,035 |

## Removed By Duplicate-Universe Detection

10-year `ml-research` path:

| Metric | Count |
|---|---:|
| Duplicate variants skipped | 18 |
| Estimated duplicate rows removed | 4,782 |
| Hypothetical rows without duplicate detection | 9,564 |
| Actual rows with duplicate detection | 4,782 |

5-year standalone path:

| Metric | Count |
|---|---:|
| Duplicate variants skipped | 18 |
| Estimated duplicate rows removed | 2,070 |
| Hypothetical rows without duplicate detection | 4,140 |
| Actual rows with duplicate detection | 2,070 |

Responsible code path:

- [core/research/ml/rebalance_dataset.py](/Users/brandonlinnett/Desktop/trading_system/core/research/ml/rebalance_dataset.py:198): `symbol_set_key = tuple(sorted(set(available_symbols)))`
- [core/research/ml/rebalance_dataset.py](/Users/brandonlinnett/Desktop/trading_system/core/research/ml/rebalance_dataset.py:199): checks `seen_universe_symbol_sets`
- [core/research/ml/rebalance_dataset.py](/Users/brandonlinnett/Desktop/trading_system/core/research/ml/rebalance_dataset.py:200): skips duplicate universes
- [core/research/ml/rebalance_dataset.py](/Users/brandonlinnett/Desktop/trading_system/core/research/ml/rebalance_dataset.py:212): records `reason: duplicate_universe_symbol_set`

This explains the 9,564 to 4,782 reduction exactly and appears intentional.

## Removed Because Of Insufficient History

There are two separate meanings here:

Symbol/universe availability:

| Metric | Count |
|---|---:|
| Local Parquet symbols scanned | 32 |
| Inventory symbols included | 32 |
| Missing symbols in `current_32` during generation | 0 |
| Variants skipped for `insufficient_available_symbols` | 0 |

Command history window:

| Comparison | Rows Removed |
|---|---:|
| 10-year deduplicated path to 5-year deduplicated path | 2,712 |
| 10-year no-dedup hypothetical to 5-year no-dedup hypothetical | 5,424 |

Responsible code path for the 5-year standalone behavior:

- [application/services/runtime_overrides.py](/Users/brandonlinnett/Desktop/trading_system/application/services/runtime_overrides.py:114): starts the conditional history override
- [application/services/runtime_overrides.py](/Users/brandonlinnett/Desktop/trading_system/application/services/runtime_overrides.py:115): applies `ml.research_years` only to `ml-research`, `ml-smoke-test`, and `champion-robustness`
- [application/cli.py](/Users/brandonlinnett/Desktop/trading_system/application/cli.py:355): dispatches `ml-expanded-rebalance-dataset`

Because `ml-expanded-rebalance-dataset` is not in the `ml.research_years` override set, it retains the base `backtest.years=5`. That is why the standalone command printed 2,070 rows, while the later DLinear `ml-research` run rebuilt 4,782 rows.

This looks like an unintended inconsistency in command orchestration, not a label or model issue.

## Removed Because New Research Labels Could Not Be Computed

| Label Field | Missing Values In Current Dataset |
|---|---:|
| `forward_return_5d` | 0 |
| `forward_return_10d` | 0 |
| `future_volatility` | 0 |
| `future_drawdown` | 0 |
| `future_max_drawdown` | 0 |
| `max_adverse_excursion` | 0 |
| `max_favourable_excursion` | 0 |
| `should_reduce_exposure` | 0 |

Rows removed by new-label computation: 0.

The new labels are computed inside already-included rows and do not add a filtering condition.

## Removed For Other Filtering Reasons

These counts are from replaying the non-duplicate variants and checking each selection before it becomes a dataset row.

10-year `ml-research` path:

| Reason | Count |
|---|---:|
| Included | 4,782 |
| Missing feature after lookback warmup | 126 |
| Insufficient future horizon at end of sample | 84 |
| Missing equity or benchmark date | 0 |
| New label computation failure | 0 |

5-year standalone path:

| Reason | Count |
|---|---:|
| Included | 2,070 |
| Missing feature after lookback warmup | 126 |
| Insufficient future horizon at end of sample | 84 |
| Missing equity or benchmark date | 0 |
| New label computation failure | 0 |

The warmup/horizon exclusions are pre-existing mechanics:

- feature rows require the historical feature lookback;
- label rows require enough future bars for the configured horizon.

They do not explain the recent 9,564 to 2,070 drop.

## Conclusion

The 9,564 to 4,782 reduction is explained exactly by duplicate-universe detection after `current_32` and `us_liquid_100` became the same available 32-symbol set.

The 4,782 to 2,070 reduction is caused by the standalone expanded-dataset command using 5 years of backtest history instead of the 10-year ML research window. The exact orchestration path is `apply_runtime_overrides`, which does not include `ml-expanded-rebalance-dataset` in the modes that receive `ml.research_years`.

No evidence points to the new research labels removing rows.
