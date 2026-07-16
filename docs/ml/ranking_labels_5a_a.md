# Ticket 5A-A — Ranking Labels and Frozen Grouped Dataset

## Scope

This foundation completes Wave 5.1 without training a ranking model. It creates
authoritative synthetic contracts for continuous percentile relevance,
margin-filtered pairwise preferences, grouped row-wise ranking datasets and
framework-neutral exports.

The existing quintile and decile algorithms remain owned by
`core.research.ml.ranking`. This module validates their outputs but does not
reimplement them. LightGBM and XGBoost are not imported or installed.

## Continuous percentile relevance

`continuous_percentile_relevance_v1` groups rows by decision date. Within a group,
returns are sorted ascending:

- the lowest return maps toward zero;
- the highest maps toward one;
- higher relevance is better;
- an economic tie receives the average zero-based rank of the tied positions,
  divided by `group_size - 1`.

Canonical asset ID and row ID order records and provide tie evidence. They never
assign different labels to equal realised returns. Missing, non-finite or immature
targets fail closed. Single-row and undersized groups fail under the explicit
minimum-group policy; no row or group is dropped.

## Pairwise return-margin labels

`pairwise_return_margin_v1` generates unordered candidate pairs only within a
decision date. The asset with the larger realised return is the winner. A pair is
retained when:

\[
r_{\rm winner}-r_{\rm loser}\geq m,
\]

where \(m\) is the configured nonnegative minimum margin.

Exact ties and sub-margin differences are separately counted and excluded.
Self-pairs, reversed duplicates and cross-date pairs are impossible by
construction.

When candidates exceed the per-date budget, each pair receives a SHA-256 priority
derived from contract ID, pair ID and margin. The lowest priorities are retained,
then records are published in canonical pair-ID order. No random sampling or
seed is used. Pair checksums, per-date group checksums and the complete pair
population checksum are recorded.

Pairwise evidence remains in `pairwise_ranking_dataset_v1`; it is not forced into
an ordinary single-row relevance table.

## Frozen grouped-ranking dataset

`grouped_ranking_dataset_v1` supports:

- quintile integers 0–4;
- decile integers 0–9;
- continuous percentile values in `[0,1]`.

Rows are ordered by decision date, canonical asset ID and row ID. Every decision
date forms one contiguous, nonempty query group. Undersized groups fail; they are
not silently removed.

Each decision date has exactly one split role. Training dates must strictly
precede validation dates, and row IDs and asset/date identities cannot overlap.
Feature values must be finite and available by the decision timestamp. Targets
must mature by the allowed cutoff.

Immutable evidence includes:

- feature-schema checksum;
- target-contract checksum;
- ranking-label-contract checksum;
- ordered row and label checksums;
- decision-date checksum;
- group-size-vector checksum;
- split checksum;
- dataset checksum.

## Framework-neutral exports

The LightGBM-style export provides the ordered feature matrix, labels and group
sizes. Group sizes describe consecutive rows belonging to each query:

`[5, 7, 6]` means the first five rows form query zero, the next seven query one,
and the final six query two.

The XGBoost-style export provides the same rows and labels plus a qid vector:

`[0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2]`.

Verification decodes qids and requires them to reproduce the exact LightGBM
group-size vector. The generic mapping additionally records absolute and
group-relative positions.

## Existing quintile and decile boundary

Compatibility adapters validate existing relevance outputs for:

- exact existing contract ID;
- integer type and range;
- complete row population;
- minimum group size;
- date grouping;
- maturity and target identity when supplied;
- label and group checksums.

When the existing output lacks target or maturity metadata, it is returned as
`LEGACY_COMPATIBLE` with `REQUIRED_METADATA_NOT_BOUND`; missing metadata is not
invented.

Standard LambdaMART-style graded relevance commonly requires nonnegative integer
labels. Quintile and decile outputs satisfy that representation after full
contract validation. Continuous percentile labels must not be passed blindly
into integer-relevance objectives: a future model contract must explicitly
support them or perform a separately registered conversion.

## Limitations and deferred work

These contracts do not determine whether a ranking model is useful. Pair budgets
can alter the distribution of training comparisons even when selection is
deterministic. Percentile relevance discards return magnitude. Quintile and
decile relevance can be unstable in small cross-sections.

Future ranking-model work requires installed dependencies, bounded search
accounting, strict-OOS grouped datasets, dependency-aware inference and protected
final audit.

