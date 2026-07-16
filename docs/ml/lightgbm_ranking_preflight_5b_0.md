# Ticket 5B-0 — LightGBM Ranking Dependency Preflight

## Dependency decision

`requirements.txt` is the repository's only dependency manifest. There is no
lockfile or separate research optional-dependency group. Existing core numerical
packages use exact pins, so LightGBM is declared as `lightgbm==4.6.0`. This means
the dependency is available to the standard repository environment, although
the new owner remains research-only.

LightGBM 4.6.0 was installed from the prebuilt `py3-none-win_amd64` wheel with
`--no-deps`. NumPy 2.0.2, SciPy 1.17.1 and scikit-learn 1.6.1 were already
present; no indirect dependency changed. The successful fits and predictions
prove the packaged Windows/OpenMP runtime resolves in this environment.

## Wrapper boundary

`lightgbm_ranking_dependency_preflight_v1` accepts only finite rectangular
feature matrices, unique deterministically ordered row IDs, nonnegative integer
relevance labels and positive group sizes whose sum exactly equals the row
count. Quintile, decile and generic nonnegative integer relevance are supported.
Continuous-percentile, fractional and negative relevance are rejected before
LightGBM is called.

This is a capability wrapper, not a selector or model-research owner. It uses
only an eight-row synthetic fixture and writes a temporary model solely to the
caller-provided temporary directory.

## Confirmed capabilities

Both `objective="rank_xendcg"` and `objective="lambdarank"` accept integer
relevance and grouped query sizes through `LGBMRanker`. Rank-XENDCG repeated fits
produce identical predictions at `rtol=0`, `atol=1e-12`. Model save/reload also
reproduces predictions at that tolerance.

The bounded CPU policy is:

- `random_state=1729`;
- `deterministic=True`;
- `force_col_wise=True`;
- `n_jobs` restricted to one or two;
- fixed bagging, feature-fraction and data-random seeds;
- no row or feature subsampling;
- `max_bin=31`;
- `verbosity=-1`.

Invalid group sums and empty groups fail before fitting. The wrapper likewise
rejects negative labels and continuous percentile labels for these
integer-relevance objectives.

## Scope boundary

No real selector or market data is loaded. No historical evaluation, replay,
parameter search, component publication or complete ranking selector is
implemented here. XGBoost, CatBoost and other ranking packages remain absent.
