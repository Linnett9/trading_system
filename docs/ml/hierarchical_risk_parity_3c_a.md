# Ticket 3C-A — Hierarchical Risk Parity Portfolio Foundation

Status: `IMPLEMENTED_SYNTHETIC_ONLY_INTEGRATION_DEFERRED`.

Implementation owner: `core/research/ml/hierarchical_risk_parity.py`.

## Contracts and variants

- Input: `hierarchical_risk_parity_input_v1`
- Result: `hierarchical_risk_parity_result_v1`
- Comparison: `hierarchical_risk_parity_comparison_v1`
- Standard candidate-universe HRP: `standard_candidate_universe_hrp_v1`
- Top 20: `hrp_top_20_v1`
- Top 40: `hrp_top_40_v1`
- Sector-first extension: `sector_first_hrp_v1`

The input records exact asset, observation and complete return-history checksums.
Creation timestamps do not affect logical result identity.

## Correlation distance

The implemented recognized HRP distance is:

`d_ij = sqrt((1-rho_ij)/2)`

Covariance is converted to correlation using covariance diagonal volatility.
Correlations are symmetrized, bounded to `[-1,1]`, and have exact unit diagonal.
The resulting distance is symmetric with zero diagonal.

Constant or near-constant series are blocked because their correlation is not
meaningful. Near-perfect or duplicated non-constant series produce deterministic
near-zero/zero distances and remain visible to clustering.

## Clustering and quasi-diagonalisation

- Library: SciPy
- Linkage: single linkage
- Input: canonical condensed upper-triangle distance vector
- Quasi-diagonal order: `scipy.cluster.hierarchy.leaves_list`
- Tie foundation: canonically ordered asset population and deterministic condensed
  distance ordering

The linkage matrix, leaf order, ordered assets and cluster-tree checksum are
reported. Every asset must appear exactly once.

No future portfolio outcome is used to select the linkage method.

## Cluster variance and recursive bisection

For a cluster covariance submatrix `Sigma_C`:

1. form inverse-variance weights  
   `v_i proportional to 1 / Sigma_C,ii`;
2. normalize `v`;
3. compute cluster variance  
   `sigma_C^2 = v' Sigma_C v`.

After quasi-diagonal ordering, each current cluster is split into deterministic
left and right halves. Sibling allocation is:

`left_fraction = right_variance / (left_variance + right_variance)`

`right_fraction = left_variance / (left_variance + right_variance)`

The process recurses until individual assets remain. Every split, cluster
membership, variance and allocation fraction is recorded.

## Constraint post-processing

Raw HRP weights are reported separately from final constrained weights.

Post-processing supports:

- exact risky exposure;
- stock floors and caps;
- sector caps;
- liquidity eligibility;
- optional cash residual.

It uses deterministic proportional water filling guided by raw HRP weights.
Cap-limited weight is redistributed while preserving raw HRP priority as far as
feasible. Any changed solution emits `CONSTRAINED_POST_PROCESSING_APPLIED`.

The constrained result is never described as unconstrained HRP. Infeasible stock or
sector capacity fails closed.

## Top-k variants

Top-20 and top-40 selection use supplied selector scores only:

1. descending score;
2. canonical asset ID as the final tie-breaker.

Selection never uses realized future returns. Insufficient eligible candidates
returns `INSUFFICIENT_DATA`. Excluded assets receive zero new weight.

## Sector-first extension

Sector-first HRP is explicitly an extension rather than standard HRP.

1. Within each sector, form an inverse-volatility weighted sector return series.
2. Run HRP across sector return series.
3. Run ordinary HRP among assets within each sector.
4. Multiply sector and within-sector weights.
5. Apply stock/sector/liquidity post-processing separately.

Aggregation contract: `inverse_volatility_sector_return_v1`.

Single-asset sectors receive within-sector weight one. Missing or ambiguous sector
identities are rejected.

## Results and verification

Results contain raw/final weights, selected/excluded assets, covariance,
correlation/distance checksums, linkage, leaf order, tree identity, split ledger,
sector evidence, turnover, variance, annualized volatility, concentration, caps,
liquidity exclusions and lineage identities.

The independent verifier recomputes the entire deterministic HRP result and checks:

- population and full history identity;
- covariance/correlation/distance;
- linkage and leaf order;
- tree checksum;
- recursive splits;
- raw and final weights;
- logical result checksum.

Changing returns, linkage, leaves, split allocations or weights fails verification.

## Ex-ante comparison

The generic comparison accepts compatible equal-weight, inverse-volatility,
linear-shrinkage minimum-variance, HRP, sector-first HRP, aim-portfolio or unchanged
holdings results when populations and histories match.

It reports variance, volatility, concentration, effective holdings, maximum weight,
sector concentration, turnover and cluster concentration. It does not calculate
historical return, Sharpe ratio or promotion.

## Limitations and point-in-time requirements

HRP does not estimate expected returns. It is a risk-allocation method whose value
depends on covariance quality, clustering stability, turnover, constraints and
costs. Historical improvement is not guaranteed.

Future use requires strict-OOS candidate selection, return history available before
the decision cutoff, point-in-time eligibility/sectors/liquidity, reconciled prior
holdings, registered policy/covariance identities and no future covariance data.

Current static historical-universe limitations remain applicable. ADV capacity is
`UNVERIFIED`.

## Statuses

- `VALID`
- `INFEASIBLE`
- `INSUFFICIENT_DATA`
- `INVALID_INPUT`
- `UNSUPPORTED_CONFIGURATION`
- `CLUSTERING_FAILURE`
- `NUMERICAL_FAILURE`

## Verification

```powershell
pytest -q tests/test_hierarchical_risk_parity.py
```

Result: `18 passed in 1.48s`.

No real data, fitting, prediction, replay, evaluation, portfolio mutation or order
generation occurred.

