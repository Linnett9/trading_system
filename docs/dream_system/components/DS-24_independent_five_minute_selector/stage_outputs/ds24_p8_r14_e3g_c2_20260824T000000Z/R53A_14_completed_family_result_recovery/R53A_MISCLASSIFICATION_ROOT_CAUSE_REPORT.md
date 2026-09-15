# R53A Misclassification Root Cause Report

R53A preserves R53 as the first discovery pass and records the corrected historical result authority in a separate output root.

## ridge_policy_v1_control

- R53 reported: COMPLETE_LEGACY_RESULT_LIMITED / LEGACY_LIMITED.
- R53A recovered: COMPLETE / QUARANTINED_AFTER_COMPLETION.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/19_tabular_scale_results.csv.
- Authority SHA256: 5c1378aba59a29ffd71efb53581720ce871950c28bfe8f60d45f531d5f86dd0a.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## pca_ridge_policy_v1_control

- R53 reported: COMPLETE_LEGACY_RESULT_LIMITED / LEGACY_LIMITED.
- R53A recovered: COMPLETE / QUARANTINED_AFTER_COMPLETION.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/pca_ridge_policy_v1_control/r31_performance_summary.json.
- Authority SHA256: 4a87fa2d2d4b55311f80474cebd12c0f5eb61600a13f867051b5e87e5b9fdab6.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## spline_additive_ridge

- R53 reported: COMPLETE_LEGACY_RESULT_LIMITED / LEGACY_LIMITED.
- R53A recovered: COMPLETE / QUARANTINED_AFTER_COMPLETION.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/spline_additive_ridge/r31_performance_summary.json.
- Authority SHA256: 04aa89388495eb00ee8c6c1b3abd55fc90bf3a3b37b269fbf4a3bd2184907e7b.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## elastic_net

- R53 reported: COMPLETE_FINAL_RESULT_AVAILABLE / ACCEPTED_FINAL.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/elastic_net/metrics_only_v3_r40_elastic_net/resolved_performance_summary_v3.json.
- Authority SHA256: b31417b9b6165143cf6c5084f10d7b5a4e62fcf2922af47b95603cf060b7128d.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## rff_ridge

- R53 reported: COMPLETE_FINAL_RESULT_AVAILABLE / ACCEPTED_FINAL.
- R53A recovered: COMPLETE / COMPLETED_ZERO_COST_ONLY.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/rff_ridge/metrics_only_v3_r37_rff_retry/resolved_performance_summary_v3.json.
- Authority SHA256: 87c7004430c8ebee24072f1b0dff3419c33c89eea154daf2db5ce7f33e31045b.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## huber

- R53 reported: COMPLETE_FINAL_RESULT_AVAILABLE / ACCEPTED_FINAL.
- R53A recovered: COMPLETE / COMPLETED_ZERO_COST_ONLY.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/huber/metrics_only_v3_r37_huber_replay/resolved_performance_summary_v3.json.
- Authority SHA256: 6e2c3d8374453dede892b72a5d1b0d714db008b5961ab3a4f38ef79572f84604.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## mlp

- R53 reported: RUNNING_PROVISIONAL_RESULT / PROVISIONAL_RUNNING.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/mlp/metrics_only_v3_r37_mlp_direct_gen7/resolved_performance_summary_v3.json.
- Authority SHA256: dde5d615d4d4d22fa45b8c306af0010a34cbfa93ec86231b10b6a404fca47b97.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## random_forest

- R53 reported: RUNNING_PROVISIONAL_RESULT / PROVISIONAL_RUNNING.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/random_forest/metrics_only_v3_r40_random_forest/resolved_performance_summary_v3.json.
- Authority SHA256: c9aaf84930286e69855245a0067b64822048340db49a51b42002812b7b655758.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## extra_trees

- R53 reported: RUNNING_PROVISIONAL_RESULT / PROVISIONAL_RUNNING.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r14_policy_workers/extra_trees/metrics_only_v3_r40_extra_trees/resolved_performance_summary_v3.json.
- Authority SHA256: 0fddc1354f6822ab3110b94552ee88adb5b61905890580994669b735da4a7ca6.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## gradient_boosting

- R53 reported: RUNNING_PROVISIONAL_RESULT / PROVISIONAL_RUNNING.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/19_tabular_scale_results.csv.
- Authority SHA256: 5c1378aba59a29ffd71efb53581720ce871950c28bfe8f60d45f531d5f86dd0a.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## lightgbm_rank_xendcg

- R53 reported: COMPLETE_IMPORTED_RESULT_AVAILABLE / ACCEPTED_IMPORTED_FINAL.
- R53A recovered: COMPLETE / ACCEPTED_IMPORTED_FINAL.
- Winning authority: mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg/metrics_only_v3/resolved_performance_summary_v3.json.
- Authority SHA256: 95d86023e1c50fe8b2599668d7d8f3313eff8ce178e16e2c425601296c8c1a7c.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## lightgbm_lambdarank

- R53 reported: EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE / MISSING.
- R53A recovered: COMPLETE / ACCEPTED_IMPORTED_FINAL.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/23_ranking_scale_results.csv.
- Authority SHA256: 18f673fa61da2d578ee3e40c281353b9b8e8961aeaaa2401955c6138ffa11f9d.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## DLinear

- R53 reported: RUNNING_PROVISIONAL_RESULT / PROVISIONAL_RUNNING.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/21_sequence_scale_results.csv.
- Authority SHA256: 85943af6cf18e03c471ce1e9b8c2a9ef905c22b315ddb6ad7330721e89a51ba5.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.

## Temporal Fusion Transformer

- R53 reported: CONFIGURATION_AUTHORITY_REQUIRED / BLOCKED.
- R53A recovered: COMPLETE / COMPLETED_DIFFERENT_WINDOW.
- Winning authority: docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c1_20260823T000000Z/21_sequence_scale_results.csv.
- Authority SHA256: 85943af6cf18e03c471ce1e9b8c2a9ef905c22b315ddb6ad7330721e89a51ba5.
- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.
- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.
