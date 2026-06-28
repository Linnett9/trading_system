# Dataset Growth Diagnostic

Generated: 2026-06-25

Scope: research-only diagnostic. No code, broker, paper trading, live trading, or execution functionality was modified.

## Basis Of Comparison

The previous 4,782-row expanded rebalance dataset is represented by the `current_32` slice of the current CSV. This matches the earlier row-count diagnostic where the deduplicated 10-year dataset contained only `current_32`.

The current 9,552-row dataset is:

- `cache/ml/expanded_rebalance_dataset.csv`
- 9,552 data rows
- active universes in this dataset: `current_32`, `us_liquid_100`

`us_liquid_250` and `us_liquid_500` exist as generated universe files, but they are not currently included in the expanded rebalance dataset config that produced this CSV.

## Summary

| Metric | Previous 4,782 Rows | Current 9,552 Rows | New Rows |
|---|---:|---:|---:|
| Rows | 4,782 | 9,552 | 4,770 |
| Unique rebalance dates | 518 | 518 | 517 |
| Unique universes | 1 | 2 | 1 |
| Unique selected symbol combinations | 1,129 | 2,084 | 956 |
| Average symbols selected per row | 3.8446 | 4.0412 | 4.2384 |
| Duplicate model feature-vector rows | 903 | 1,739 | 836 |
| Unique model feature vectors | 3,879 | 7,813 | 3,934 |
| Duplicate label-tuple rows | 4,117 | 8,650 | 4,106 |
| Unique label tuples | 665 | 902 | 664 |

Definitions:

- Duplicate model feature-vector rows use the same numeric feature extraction as `core/research/ml/datasets.py`, excluding dates, identifiers, selected symbols, outcomes, and labels.
- Duplicate label-tuple rows use `(feature_date, label_start_date, label_end_date, should_reduce_exposure)`.
- New rows are the non-`current_32` rows, currently `us_liquid_100`.

## Label Distribution

| Dataset | Label 0 | Label 1 | Positive Rate |
|---|---:|---:|---:|
| Previous 4,782 rows | 1,053 | 3,729 | 77.98% |
| Current 9,552 rows | 2,785 | 6,767 | 70.72% |
| New 4,770 rows | 1,732 | 3,038 | 63.69% |

The added rows reduce the positive-label rate, which is useful: the dataset is less dominated by `should_reduce_exposure=1` than the previous `current_32`-only slice.

## Regime Distribution

| Regime | Previous Rows | Current Rows | New Rows |
|---|---:|---:|---:|
| `risk-on` | 2,449 | 5,588 | 3,139 |
| `cash` | 891 | 1,377 | 486 |
| `partial-risk` | 538 | 1,080 | 542 |
| `fast-reentry` | 690 | 902 | 212 |
| `chop-filter` | 214 | 605 | 391 |

## Rows By Universe

| Universe | Previous Rows | Current Rows | New Rows |
|---|---:|---:|---:|
| `current_32` | 4,782 | 4,782 | 0 |
| `us_liquid_100` | 0 | 4,770 | 4,770 |
| `us_liquid_250` | 0 | 0 | 0 |
| `us_liquid_500` | 0 | 0 | 0 |

Universe file status:

| Universe File | Symbols | Available Count | Included In Current Expanded Dataset |
|---|---:|---:|---|
| `current_32.yaml` | 514 | 514 | yes |
| `us_liquid_100.yaml` | 100 | 379 | yes |
| `us_liquid_250.yaml` | 250 | 379 | no |
| `us_liquid_500.yaml` | 379 | 379 | no |

## Rows By Frequency, Top N, And Weighting

Rows by universe and rebalance frequency:

| Universe | Monthly | Biweekly | Weekly |
|---|---:|---:|---:|
| `current_32` | 636 | 1,386 | 2,760 |
| `us_liquid_100` | 636 | 1,380 | 2,754 |

Rows by universe and `top_n`:

| Universe | Top 3 | Top 5 | Top 7 |
|---|---:|---:|---:|
| `current_32` | 1,594 | 1,594 | 1,594 |
| `us_liquid_100` | 1,590 | 1,590 | 1,590 |

Rows by universe and weighting:

| Universe | Equal | Inverse Volatility |
|---|---:|---:|
| `current_32` | 2,391 | 2,391 |
| `us_liquid_100` | 2,385 | 2,385 |

## Selected Symbol Combination Uniqueness

| Dataset | Rows | Unique Selected Symbol Combinations | Duplicate Selected-Combo Rows |
|---|---:|---:|---:|
| Previous 4,782 rows | 4,782 | 1,129 | 3,653 |
| Current 9,552 rows | 9,552 | 2,084 | 7,468 |
| New 4,770 rows | 4,770 | 956 | 3,814 |

Of the 4,770 new rows:

| Comparison Against Previous Dataset | Count | Percent Of New Rows |
|---|---:|---:|
| New rows with model feature vectors not seen before | 4,770 | 100.00% |
| New rows with full row signatures not seen before | 4,770 | 100.00% |
| New rows with selected symbol combinations not seen before | 4,284 | 89.81% |

Interpretation: every added row is unique at the model-feature level compared with the previous dataset. Some selected symbol combinations repeat, but the rows still differ by date, universe context, rank/weight/correlation features, labels, or outcomes.

## Duplicate Feature Vectors

| Dataset | Duplicate Feature-Vector Rows | Unique Feature Vectors |
|---|---:|---:|
| Previous 4,782 rows | 903 | 3,879 |
| Current 9,552 rows | 1,739 | 7,813 |
| New 4,770 rows | 836 | 3,934 |

These are duplicates after applying the actual ML numeric feature extraction. The current dataset nearly doubles unique model feature vectors from 3,879 to 7,813.

## Duplicate Labels

| Dataset | Duplicate Label-Tuple Rows | Unique Label Tuples |
|---|---:|---:|
| Previous 4,782 rows | 4,117 | 665 |
| Current 9,552 rows | 8,650 | 902 |
| New 4,770 rows | 4,106 | 664 |

This is expected because many portfolio variants share the same feature date and forward label window. It is not evidence of duplicated feature vectors.

## Conclusion

The growth from 4,782 to 9,552 rows is primarily the addition of `us_liquid_100` variants alongside the previous `current_32` variants.

The added 4,770 rows are genuinely unique compared with the previous dataset by model feature vector and full row signature. They also add 955 net new selected-symbol combinations at the current-dataset level, while lowering the positive label rate from 77.98% to 70.72%.

The current expanded dataset does not yet include `us_liquid_250` or `us_liquid_500`, even though those universe files are real and available.
