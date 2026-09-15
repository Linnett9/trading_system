# R53B R53A Source Selection Errors

R53A is preserved as an audit artifact. R53B rejects sources that do not bind to terminal tournament identity.

## random_forest

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/resolved_performance_summary_v3.json
- R53A status: COMPLETED_DIFFERENT_WINDOW
- R53B authority: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY
- R53B class: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY
- Correction: Only current partial-window progress was found; no eligible terminal authority selected.

## extra_trees

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/resolved_performance_summary_v3.json
- R53A status: COMPLETED_DIFFERENT_WINDOW
- R53B authority: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY
- R53B class: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY
- Correction: Only current partial-window progress was found; no eligible terminal authority selected.

## gradient_boosting

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/19_tabular_scale_results.csv
- R53A status: COMPLETED_DIFFERENT_WINDOW
- R53B authority: TERMINAL_AUTHORITY_NOT_LOCAL
- R53B class: TERMINAL_AUTHORITY_NOT_LOCAL
- Correction: No eligible terminal tournament authority was found locally.

## mlp

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/resolved_performance_summary_v3.json
- R53A status: COMPLETED_DIFFERENT_WINDOW
- R53B authority: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY
- R53B class: LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY
- Correction: Only current partial-window progress was found; no eligible terminal authority selected.

## lightgbm_rank_xendcg

- R53A source: mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/resolved_performance_summary_v3.json
- R53A status: ACCEPTED_IMPORTED_FINAL
- R53B authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/R47A_xendcg_import_authority.json
- R53B class: CURRENT_DS24_IMPORTED_MAC_TERMINAL
- Correction: Selected eligible terminal authority by tournament lineage/source class.

## lightgbm_lambdarank

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/23_ranking_scale_results.csv
- R53A status: ACCEPTED_IMPORTED_FINAL
- R53B authority: LAMBDARANK_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER
- R53B class: LAMBDARANK_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER
- Correction: R49/R51 prove LambdaRank transfer files are missing locally; historical LambdaRank rows are rejected.

## DLinear

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/21_sequence_scale_results.csv
- R53A status: COMPLETED_DIFFERENT_WINDOW
- R53B authority: DLINEAR_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER
- R53B class: DLINEAR_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER
- Correction: No local Mac DLinear terminal estate or trusted terminal summary was found; historical Dell rows are rejected.

## Temporal Fusion Transformer

- R53A source: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/21_sequence_scale_results.csv
- R53A status: COMPLETED_DIFFERENT_WINDOW
- R53B authority: NOT_COMPLETED
- R53B class: NOT_COMPLETED
- Correction: Open R53B lifecycle family; historical/provisional metrics are excluded from the completed scorecard.
