# Adjusted Replay Alignment Audit

Research only. Trading impact: none. Production validated: false.

Aligned correctly: False
Explanation verdict: not_aligned_missing_adjusted_prices
Missing adjusted price rows: 770
Invalid adjusted periods: 240
Valid adjusted periods: 190
Valid adjusted independent periods: 20
Date misalignment rows: 842
Symbol mismatch rows: 770
Large return-delta rows: 0
Large candidate net-return delta rows: 533
Adjustment-ratio jump rows: 2

|candidate|rows|coverage|valid periods|invalid periods|missing adjusted|date mismatch|symbol mismatch|large delta|max abs delta|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|exact_champion_replay|881|0.772781|95|120|385|421|385|0|0.023725|
|selected_bayesian_optimizer_diagnostic_policy|881|0.772781|95|120|385|421|385|0|0.023725|

## Biggest Return Deltas

|candidate|rebalance|symbol|raw return|adjusted return|delta|ratio start|ratio end|missing adjusted|
|---|---|---|---:|---:|---:|---:|---:|---|
|exact_champion_replay|2026-02-02|AZN|0.067573|0.091298|0.023725|0.978259|1.000000|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-02-02|AZN|0.067573|0.091298|0.023725|0.978259|1.000000|False|
|exact_champion_replay|2025-08-01|AU|0.469852|0.491616|0.021764|0.947655|0.961686|False|
|selected_bayesian_optimizer_diagnostic_policy|2025-08-01|AU|0.469852|0.491616|0.021764|0.947655|0.961686|False|
|exact_champion_replay|2025-08-18|AU|0.462756|0.484414|0.021658|0.947655|0.961686|False|
|selected_bayesian_optimizer_diagnostic_policy|2025-08-18|AU|0.462756|0.484414|0.021658|0.947655|0.961686|False|
|exact_champion_replay|2025-07-14|AU|0.366479|0.386712|0.020233|0.947655|0.961687|False|
|selected_bayesian_optimizer_diagnostic_policy|2025-07-14|AU|0.366479|0.386712|0.020233|0.947655|0.961687|False|
|exact_champion_replay|2025-08-04|AU|0.362799|0.382977|0.020178|0.947655|0.961686|False|
|selected_bayesian_optimizer_diagnostic_policy|2025-08-04|AU|0.362799|0.382977|0.020178|0.947655|0.961686|False|

## Biggest Candidate Net-Return Deltas

|candidate|rebalance|symbol|raw net|adjusted net|delta|missing adjusted|
|---|---|---|---:|---:|---:|---|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-30|AXTI|0.137954|0.688315|0.550360|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-30|AZN|0.137954|0.688315|0.550360|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-30|CIEN|0.137954|0.688315|0.550360|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-30|MU|0.137954|0.688315|0.550360|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-30|VIAV|0.137954|0.688315|0.550360|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-09|AXTI|0.099562|0.555752|0.456190|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-09|AZN|0.099562|0.555752|0.456190|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-09|CIEN|0.099562|0.555752|0.456190|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-09|MU|0.099562|0.555752|0.456190|False|
|selected_bayesian_optimizer_diagnostic_policy|2026-03-09|VIAV|0.099562|0.555752|0.456190|False|

Research only. Trading impact: none. Production validated: false.
