# Allocation Optimizer

Research-only optimizer. Parameters selected on out-of-fold data and evaluated once on frozen holdout; not production-valid.

Sampler requested: bayesian
Sampler used: random
Optuna available: False
Fallback reason: Optuna is not installed; falling back to deterministic random search
Candidates: 256

## Selected Diagnostic Policy

Candidate: random_candidate_0147
Selection objective: -0.9607651302408233
Selected params: {'candidate_id': 'random_candidate_0147', 'trial_number': 146, 'mapping_method': 'quantile', 'return_weight': 0.637234805257327, 'drawdown_weight': 0.2555195040214593, 'volatility_weight': 0.16406117796892056, 'min_exposure': 0.003074834359953149, 'max_exposure': 0.790425808605531, 'neutral_exposure': 0.40359459506408557, 'max_exposure_change': 0.10695691591238075, 'transaction_cost_bps': 5.0}
Holdout return: 0.4466351605736918
Holdout max drawdown: 0.022630323288094067

Research only. Trading impact: none. Production validated: false.
