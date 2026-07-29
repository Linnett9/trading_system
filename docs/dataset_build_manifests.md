# Dataset Build Manifests

Ticket 57 adds a generic, versioned dataset-build manifest and read-only stale
parent guard. The first integration is intentionally bounded to the expanded
rebalance dataset producer and its opt-in prebuilt-dataset consumer.

## Starting-State Audit

| Dataset family | Current producers | Current consumers | Pre-existing read risk |
|---|---|---|---|
| Canonical prices | `infrastructure/data/canonical_daily_v2_builder.py`, `scripts/build_canonical_daily_v2.py`, Stooq/market parquet importers | feature builders, selector dataset builder, stock prediction artifacts, research feeds | parquet feeds can be selected by config without a dataset-build manifest proving current price/corporate-action authority |
| Feature banks | `core/research/ml/features/features.py`, `core/research/ml/stock_level/stock_level_alpha_features_builder.py`, news/fundamental feature builders | `MLDatasetPipeline`, stock selectors, stock-alpha diagnostics, portfolio replay inputs | cached/generated CSV/Parquet features are accepted by several downstream commands using path/config checks rather than parent manifests |
| Expanded rebalance datasets | `MLRebalancePipeline.build_expanded_rebalance_features()` and `core/research/ml/data/rebalance_dataset.py` | `MLExperimentRunner` exposure training, `ml-research-batch`, return-mechanics audits, stock prediction artifact source discovery | `read_existing_expanded_rebalance_dataset` loads an existing CSV by path; now optionally guarded |
| Selector datasets | `core/research/ml/stock_level/selector_dataset.py`, `scripts/build_canonical_v2_selector_dataset.py` | `bounded_selector_runner`, full daily selector campaign, component publication paths | `rows.parquet` and `baseline_scores.parquet` are read from a frozen root; existing selector lineage is specific but not yet the generic build-manifest guard |
| Forecast artifacts | stock-level prediction artifact writers, bounded selector date partitions, ordinary/wave4 component publishers | meta ensemble, target comparison, portfolio replay, policy sweep, promotion reports | artifact-lineage verification exists, but forecast artifact inputs are not yet normalized under this dataset-build manifest contract |
| Labels and targets | `core/research/ml/features/labels.py`, `prediction_artifacts/targets.py`, expanded rebalance outcome labels | dataset pipeline, bounded selector runner, target comparison, ranking diagnostics | target code/registry identity exists in some artifacts, but generic label-code/config stale checks are not universal |
| Frozen experiment inputs | selector operational input packages, immutable exposure runs, campaign jobs | component publication, wave4 adapters, resume checks | package checks exist, but not yet represented as dataset-build parent manifests |
| News/fundamental feature datasets | stock-alpha news contract/feature-store modules, `stock_fundamentals.py` | news readiness, news transformer diagnostics, alpha enrichment | generated CSV/Parquet feature stores are inspected by stage-specific audits; generic stale-parent guard is not yet integrated |
| Model training inputs | `write_dataset()`, frozen selector `rows.parquet`, expanded rebalance CSV, prediction artifact datasets | `MLExperimentRunner`, bounded selector runner, stock-alpha model suites | several model commands can read generated CSV/Parquet from config paths without rechecking current parents |

## Manifest Contract

The generic manifest schema is `dataset_build_manifest_v1` and is implemented in
`core/research/ml/dataset_build_manifest.py`. It records dataset identity,
producer identity, source paths/manifests/content hashes, authority versions,
feature and label code versions, configuration hash, optional seed, row/key
counts, duplicate keys, symbol/entity count, decision and knowledge-cutoff
ranges, partition info, output hashes, parent artifact IDs, source-control
identity, dirty-tree state, rebuildability status, and a deterministic
`manifest_hash`.

## Guard Contract

The guard returns `dataset_lineage_check_v1` with:

- `status`: `CURRENT`, `STALE`, `UNVERIFIED`, `MISSING_PARENT`,
  `CONFLICTING_PARENT`, or `LEGACY_NO_MANIFEST`
- deterministic `reasons`
- `changed_parents`
- `missing_parents`
- manifest version and lineage nodes
- `permitted_use`: `PROMOTION_GRADE`, `RESEARCH_ONLY`, `DIAGNOSTIC_ONLY`, or
  `BLOCKED`
- `dataset_rebuilt=false`, `dataset_modified=false`, `source_modified=false`

Legacy datasets without manifests are diagnostic only. Dirty-tree or unknown
authority/code/config fields are unverified and cannot satisfy promotion-grade
requests. Stale, missing, or conflicting parents are blocked.

## First Integration

Producer:

- `MLRebalancePipeline.build_expanded_rebalance_features()` writes
  `expanded_rebalance_dataset.csv.manifest.json` next to the generated CSV.

Consumer:

- The `read_existing_expanded_rebalance_dataset` branch can opt into the guard
  with `ml.dataset_lineage_guard_enabled: true` or
  `ml.guard_existing_expanded_rebalance_dataset: true`.

CLI:

```text
python main.py --mode ml-dataset-lineage-check --dataset-path <path> --intended-use promotion-grade
```

Optional expectations include producer module/command, schema, config hash,
authority versions, and feature/label code versions. The command is feedless
and never rebuilds or mutates datasets.

## Frozen Selector Extension

Ticket 57A extends the same generic manifest contract to frozen selector dataset
roots produced by `core/research/ml/stock_level/selector_dataset.py`. The
producer writes `rows.parquet.manifest.json` beside `rows.parquet`, while the
existing selector-specific `manifest.json` remains the frozen selector authority.

The bounded selector runner can opt into the combined guard with
`ml.frozen_selector_dataset_lineage_guard_enabled: true` or
`ml.stock_selector_bounded.frozen_dataset_lineage_guard_enabled: true`. An
explicit promotion-grade CLI run can also enable it with
`--require-promotion-grade`. The guard composes:

- the generic dataset-build stale-parent check;
- the existing frozen selector lineage verifier;
- artifact-lineage verification when a prediction artifact manifest is supplied.

The combined report records each layer's status, permitted use, blocking
reasons, changed and missing parents, authority versions, and promotion
eligibility. It is read-only and never rebuilds or mutates datasets.

## Next Integration Batch

1. Stock-level prediction artifacts and portfolio replay: bridge artifact links
   to dataset-build manifests.
2. Stock feature banks and alpha feature enrichments: emit manifests for
   generated feature CSV/Parquet stores.
3. News and fundamentals feature datasets: add source manifest hashes for
   provider/contract inputs and authority records.
