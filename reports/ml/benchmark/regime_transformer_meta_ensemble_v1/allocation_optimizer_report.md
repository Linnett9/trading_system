# Allocation Optimizer

Research-only optimizer. Parameters selected on out-of-fold data and evaluated once on frozen holdout; not production-valid.

Sampler requested: bayesian
Sampler used: bayesian
Optuna available: True
Fallback reason: none
Objective mode: robustness_adjusted_canonical_score
Candidates: 256

## Selected Diagnostic Policy

Candidate: bayesian_trial_0192
Selection objective: -0.12307764486925432
Selected by robustness objective: True
Selected params: {'candidate_id': 'bayesian_trial_0192', 'trial_number': 192, 'mapping_method': 'quantile', 'return_weight': 0.683924477889202, 'drawdown_weight': 0.8688133282205927, 'volatility_weight': 0.14818090554900962, 'min_exposure': 0.20043188653014893, 'max_exposure': 0.7266098353016318, 'neutral_exposure': 0.7266098353016318, 'max_exposure_change': 0.11501624652831528, 'transaction_cost_bps': 5.0}
Holdout return: 23719.262712546642
Holdout max drawdown: 0.7692464761548898

Research only. Trading impact: none. Production validated: false.
