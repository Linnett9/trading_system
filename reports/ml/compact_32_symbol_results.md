# reports/ml/dlinear_32_symbol_universe

## metadata.json
```json
{
    "timestamp": "2026-06-25T12:56:31.414379",
    "config_hash": "4e33e21645934285d0a02737ea0b2550977cf68267f3d86db4dc1e5e80f6475c",
    "data_hash": "ae840498b199443fd58abc8c5a1444b8dbe27e119c359c1336215acfc6aba0f7",
    "git_commit": "ede8a7b187f9e88722048de9e5a7591b21f3cba4",
    "model_type": "dlinear",
    "feature_set": "price_regime_v1",
    "label_type": "champion_success",
    "random_seed": 42,
    "experiment_config": {
        "model_type": "dlinear",
        "feature_set": "price_regime_v1",
        "label_type": "champion_success",
        "train_start": null,
        "train_end": null,
        "test_start": null,
        "test_end": null,
        "prediction_horizon": 42,
        "label_horizon_days": 42,
        "drawdown_risk_threshold": 0.08,
        "decision_threshold": 0.5,
        "class_weight_balanced": true,
        "test_fraction": 0.2,
        "walk_forward_folds": 3,
        "random_seed": 42,
        "output_dir": "reports/ml/dlinear_32_symbol_universe"
    },
    "validation": {
        "method": "purged_chronological_holdout",
        "train_sample_count": 1730,
        "test_sample_count": 442,
        "test_start_date": "2024-07-16",
        "purged_train_samples": 42
    },
    "research_only": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## metrics.json
```json
{
    "mode": "research",
    "model_type": "dlinear",
    "feature_set": "price_regime_v1",
    "label_type": "champion_success",
    "decision_threshold": 0.5,
    "class_weight": "balanced",
    "train_sample_count": 1730,
    "test_sample_count": 442,
    "feature_count": 33,
    "test_start_date": "2024-07-16",
    "purged_train_samples": 42,
    "metrics": {
        "accuracy": 0.5610859728506787,
        "precision": 0.6502242152466368,
        "recall": 0.5555555555555556,
        "f1": 0.5991735537190082,
        "balanced_accuracy": 0.5623081645181093,
        "samples": 442
    },
    "baselines": {
        "noop": {
            "accuracy": 0.4095022624434389,
            "precision": null,
            "recall": 0.0,
            "f1": null,
            "balanced_accuracy": 0.5,
            "samples": 442
        },
        "majority_class": {
            "predicted_class": 1,
            "metrics": {
                "accuracy": 0.5904977375565611,
                "precision": 0.5904977375565611,
                "recall": 1.0,
                "f1": 0.7425320056899004,
                "balanced_accuracy": 0.5,
                "samples": 442
            }
        }
    },
    "note": "Research-only out-of-sample evaluation; ML does not affect trading decisions.",
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## walk_forward_metrics.json
```json
{
    "model_type": "dlinear",
    "fold_count": 3,
    "folds": [
        {
            "fold": 1,
            "train_sample_count": 842,
            "test_sample_count": 442,
            "test_start_date": "2020-12-31",
            "purged_train_samples": 42,
            "metrics": {
                "accuracy": 0.49321266968325794,
                "precision": 0.4780058651026393,
                "recall": 0.7799043062200957,
                "f1": 0.5927272727272728,
                "balanced_accuracy": 0.5079779041830093,
                "samples": 442
            },
            "baselines": {
                "noop": {
                    "accuracy": 0.5271493212669683,
                    "precision": null,
                    "recall": 0.0,
                    "f1": null,
                    "balanced_accuracy": 0.5,
                    "samples": 442
                },
                "majority_class": {
                    "predicted_class": 1,
                    "metrics": {
                        "accuracy": 0.47285067873303166,
                        "precision": 0.47285067873303166,
                        "recall": 1.0,
                        "f1": 0.6420890937019968,
                        "balanced_accuracy": 0.5,
                        "samples": 442
                    }
                }
            }
        },
        {
            "fold": 2,
            "train_sample_count": 1284,
            "test_sample_count": 442,
            "test_start_date": "2022-10-04",
            "purged_train_samples": 42,
            "metrics": {
                "accuracy": 0.3167420814479638,
                "precision": 0.35135135135135137,
                "recall": 0.3305084745762712,
                "f1": 0.3406113537117904,
                "balanced_accuracy": 0.31573967418133947,
                "samples": 442
            },
            "baselines": {
                "noop": {
                    "accuracy": 0.4660633484162896,
                    "precision": null,
                    "recall": 0.0,
                    "f1": null,
                    "balanced_accuracy": 0.5,
                    "samples": 442
                },
                "majority_class": {
                    "predicted_class": 1,
                    "metrics": {
                        "accuracy": 0.5339366515837104,
                        "precision": 0.5339366515837104,
                        "recall": 1.0,
                        "f1": 0.696165191740413,
                        "balanced_accuracy": 0.5,
                        "samples": 442
                    }
                }
            }
        },
        {
            "fold": 3,
            "train_sample_count": 1726,
            "test_sample_count": 442,
            "test_start_date": "2024-07-10",
            "purged_train_samples": 42,
            "metrics": {
                "accuracy": 0.5746606334841629,
                "precision": 0.6636363636363637,
                "recall": 0.5615384615384615,
                "f1": 0.6083333333333333,
                "balanced_accuracy": 0.5774725274725274,
                "samples": 442
            },
            "baselines": {
                "noop": {
                    "accuracy": 0.4117647058823529,
                    "precision": null,
                    "recall": 0.0,
                    "f1": null,
                    "balanced_accuracy": 0.5,
                    "samples": 442
                },
                "majority_class": {
                    "predicted_class": 1,
                    "metrics": {
                        "accuracy": 0.5882352941176471,
                        "precision": 0.5882352941176471,
                        "recall": 1.0,
                        "f1": 0.7407407407407407,
                        "balanced_accuracy": 0.5,
                        "samples": 442
                    }
                }
            }
        }
    ],
    "research_only": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## holdout_shadow_overlay.json
```json
{
    "mode": "final_holdout_shadow_research_only",
    "model_type": "dlinear",
    "success_threshold": 0.2,
    "reduced_exposure": 0.7,
    "rebalance_only": true,
    "overlay_probability": "champion_success_probability",
    "transaction_cost_bps": 5.0,
    "test_start_date": "2024-07-16",
    "result": {
        "base_total_return": 0.3368612147099357,
        "overlay_total_return": 0.292974888704409,
        "base_max_drawdown": -0.16282774936141575,
        "overlay_max_drawdown": -0.16282774936141575,
        "reduced_exposure_days": 188,
        "evaluated_days": 441,
        "overlay_turnover": 1.8000000000000003,
        "estimated_cost": 0.0009
    },
    "trading_impact": "none",
    "candidate_frozen_before_holdout": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## calibrated_probability_calibration.json
```json
{
    "evaluation": "chronological_holdout_train_quantile_calibration",
    "model_type": "dlinear",
    "label_type": "champion_success",
    "calibrator": "train_fold_quantile_bin_observed_rate",
    "raw_calibration": {
        "sample_count": 442,
        "positive_rate": 0.5904977375565611,
        "brier_score": 0.33344750212259505,
        "base_rate_brier_score": 0.24181015949714202,
        "brier_skill_score": -0.37896398900698847,
        "expected_calibration_error": 0.26862628527709936,
        "maximum_calibration_error": 0.5426004081964493,
        "bins": [
            {
                "lower_bound": 0.0,
                "upper_bound": 0.1,
                "sample_count": 185,
                "mean_predicted_probability": 0.0063496633456515865,
                "observed_positive_rate": 0.4972972972972973,
                "calibration_error": 0.49094763395164576
            },
            {
                "lower_bound": 0.1,
                "upper_bound": 0.2,
                "sample_count": 9,
                "mean_predicted_probability": 0.14499085396528244,
                "observed_positive_rate": 0.6666666666666666,
                "calibration_error": 0.5216758127013842
            },
            {
                "lower_bound": 0.2,
                "upper_bound": 0.3,
                "sample_count": 5,
                "mean_predicted_probability": 0.2573995918035507,
                "observed_positive_rate": 0.8,
                "calibration_error": 0.5426004081964493
            },
            {
                "lower_bound": 0.3,
                "upper_bound": 0.4,
                "sample_count": 10,
                "mean_predicted_probability": 0.36197397112846375,
                "observed_positive_rate": 0.6,
                "calibration_error": 0.23802602887153623
            },
            {
                "lower_bound": 0.4,
                "upper_bound": 0.5,
                "sample_count": 10,
                "mean_predicted_probability": 0.45973993837833405,
                "observed_positive_rate": 0.8,
                "calibration_error": 0.340260061621666
            },
            {
                "lower_bound": 0.5,
                "upper_bound": 0.6,
                "sample_count": 135,
                "mean_predicted_probability": 0.5376332623625272,
                "observed_positive_rate": 0.4962962962962963,
                "calibration_error": -0.041336966066230885
            },
            {
                "lower_bound": 0.6,
                "upper_bound": 0.7,
                "sample_count": 5,
                "mean_predicted_probability": 0.6709562182426453,
                "observed_positive_rate": 0.8,
                "calibration_error": 0.12904378175735476
            },
            {
                "lower_bound": 0.7,
                "upper_bound": 0.8,
                "sample_count": 5,
                "mean_predicted_probability": 0.7567616462707519,
                "observed_positive_rate": 0.8,
                "calibration_error": 0.043238353729248113
            },
            {
                "lower_bound": 0.8,
                "upper_bound": 0.9,
                "sample_count": 7,
                "mean_predicted_probability": 0.8292226025036403,
                "observed_positive_rate": 1.0,
                "calibration_error": 0.17077739749635967
            },
            {
                "lower_bound": 0.9,
                "upper_bound": 1.0,
                "sample_count": 71,
                "mean_predicted_probability": 0.9870312348218031,
                "observed_positive_rate": 0.8873239436619719,
                "calibration_error": -0.09970729115983124
            }
        ]
    },
    "calibrated_calibration": {
        "sample_count": 442,
        "positive_rate": 0.5904977375565611,
        "brier_score": 0.3325253628156774,
        "base_rate_brier_score": 0.24181015949714202,
        "brier_skill_score": -0.37515050445846776,
        "expected_calibration_error": 0.2816807469986655,
        "maximum_calibration_error": 0.5113636363636364,
        "bins": [
            {
                "lower_bound": 0.0,
                "upper_bound": 0.1,
                "sample_count": 176,
                "mean_predicted_probability": 0.0,
                "observed_positive_rate": 0.5113636363636364,
                "calibration_error": 0.5113636363636364
            },
            {
                "lower_bound": 0.3,
                "upper_bound": 0.4,
                "sample_count": 171,
                "mean_predicted_probability": 0.3699421965317935,
                "observed_positive_rate": 0.5087719298245614,
                "calibration_error": 0.13882973329276793
            },
            {
                "lower_bound": 0.9,
                "upper_bound": 1.0,
                "sample_count": 95,
                "mean_predicted_probability": 0.997505324003651,
                "observed_positive_rate": 0.8842105263157894,
                "calibration_error": -0.11329479768786155
            }
        ]
    },
    "raw_brier_score": 0.33344750212259505,
    "calibrated_brier_score": 0.3325253628156774,
    "brier_delta_calibrated_minus_raw": -0.0009221393069176465,
    "research_only": true,
    "trading_impact": "none",
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## dataset_audit.json
```json
{
    "sample_count": 2214,
    "feature_count": 33,
    "date_coverage": [
        "2017-06-28",
        "2026-04-20"
    ],
    "class_balance": {
        "positive": 1194,
        "negative": 1020,
        "positive_rate": 0.5392953929539296
    },
    "dropped_rows_insufficient_label_horizon": 42,
    "leakage_check_passed": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## rebalance_dataset_audit.json
```json
{
    "row_count": 106,
    "good_period_rate": 0.6886792452830188,
    "bad_period_rate": 0.1792452830188679,
    "underperforms_spy_rate": 0.4811320754716981,
    "drawdown_event_rate": 0.19811320754716982,
    "history_years": 10,
    "recommended_generalization_years": 10,
    "minimum_history_years": 9,
    "sector_reference_path": "data/reference/sector_by_symbol.json",
    "research_only": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

---
# reports/ml/patchtst_32_symbol_universe

## metadata.json
```json
{
    "timestamp": "2026-06-25T13:09:46.837425",
    "config_hash": "4e9a97f8090b19118177725fb96097f711a14da623398d99011106110e2e0511",
    "data_hash": "026bcc69f6c73883609828dfa7833ae95ba43125cdd591604f93ee344201b432",
    "git_commit": "ede8a7b187f9e88722048de9e5a7591b21f3cba4",
    "model_type": "patchtst",
    "feature_set": "price_regime_v1",
    "label_type": "champion_success",
    "random_seed": 42,
    "experiment_config": {
        "model_type": "patchtst",
        "feature_set": "price_regime_v1",
        "label_type": "champion_success",
        "train_start": null,
        "train_end": null,
        "test_start": null,
        "test_end": null,
        "prediction_horizon": 42,
        "label_horizon_days": 42,
        "drawdown_risk_threshold": 0.08,
        "decision_threshold": 0.5,
        "class_weight_balanced": true,
        "test_fraction": 0.2,
        "walk_forward_folds": 3,
        "random_seed": 42,
        "output_dir": "reports/ml/patchtst_32_symbol_universe"
    },
    "validation": {
        "method": "purged_chronological_holdout",
        "train_sample_count": 1730,
        "test_sample_count": 442,
        "test_start_date": "2024-07-16",
        "purged_train_samples": 42
    },
    "research_only": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## metrics.json
```json
{
    "mode": "research",
    "model_type": "patchtst",
    "feature_set": "price_regime_v1",
    "label_type": "champion_success",
    "decision_threshold": 0.5,
    "class_weight": "balanced",
    "train_sample_count": 1730,
    "test_sample_count": 442,
    "feature_count": 33,
    "test_start_date": "2024-07-16",
    "purged_train_samples": 42,
    "metrics": {
        "accuracy": 0.4343891402714932,
        "precision": 0.531578947368421,
        "recall": 0.38549618320610685,
        "f1": 0.4469026548672566,
        "balanced_accuracy": 0.44552586938083116,
        "samples": 442
    },
    "baselines": {
        "noop": {
            "accuracy": 0.4072398190045249,
            "precision": null,
            "recall": 0.0,
            "f1": null,
            "balanced_accuracy": 0.5,
            "samples": 442
        },
        "majority_class": {
            "predicted_class": 1,
            "metrics": {
                "accuracy": 0.5927601809954751,
                "precision": 0.5927601809954751,
                "recall": 1.0,
                "f1": 0.7443181818181818,
                "balanced_accuracy": 0.5,
                "samples": 442
            }
        }
    },
    "note": "Research-only out-of-sample evaluation; ML does not affect trading decisions.",
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## walk_forward_metrics.json
```json
{
    "model_type": "patchtst",
    "fold_count": 3,
    "folds": [
        {
            "fold": 1,
            "train_sample_count": 842,
            "test_sample_count": 442,
            "test_start_date": "2020-12-31",
            "purged_train_samples": 42,
            "metrics": {
                "accuracy": 0.45475113122171945,
                "precision": 0.43795620437956206,
                "recall": 0.5797101449275363,
                "f1": 0.498960498960499,
                "balanced_accuracy": 0.46219549799568305,
                "samples": 442
            },
            "baselines": {
                "noop": {
                    "accuracy": 0.5316742081447964,
                    "precision": null,
                    "recall": 0.0,
                    "f1": null,
                    "balanced_accuracy": 0.5,
                    "samples": 442
                },
                "majority_class": {
                    "predicted_class": 1,
                    "metrics": {
                        "accuracy": 0.4683257918552036,
                        "precision": 0.4683257918552036,
                        "recall": 1.0,
                        "f1": 0.6379044684129429,
                        "balanced_accuracy": 0.5,
                        "samples": 442
                    }
                }
            }
        },
        {
            "fold": 2,
            "train_sample_count": 1284,
            "test_sample_count": 442,
            "test_start_date": "2022-10-04",
            "purged_train_samples": 42,
            "metrics": {
                "accuracy": 0.5497737556561086,
                "precision": 0.5483028720626631,
                "recall": 0.8898305084745762,
                "f1": 0.678513731825525,
                "balanced_accuracy": 0.5250123416159289,
                "samples": 442
            },
            "baselines": {
                "noop": {
                    "accuracy": 0.4660633484162896,
                    "precision": null,
                    "recall": 0.0,
                    "f1": null,
                    "balanced_accuracy": 0.5,
                    "samples": 442
                },
                "majority_class": {
                    "predicted_class": 1,
                    "metrics": {
                        "accuracy": 0.5339366515837104,
                        "precision": 0.5339366515837104,
                        "recall": 1.0,
                        "f1": 0.696165191740413,
                        "balanced_accuracy": 0.5,
                        "samples": 442
                    }
                }
            }
        },
        {
            "fold": 3,
            "train_sample_count": 1726,
            "test_sample_count": 442,
            "test_start_date": "2024-07-10",
            "purged_train_samples": 42,
            "metrics": {
                "accuracy": 0.5113122171945701,
                "precision": 0.6134020618556701,
                "recall": 0.4576923076923077,
                "f1": 0.5242290748898678,
                "balanced_accuracy": 0.5228021978021978,
                "samples": 442
            },
            "baselines": {
                "noop": {
                    "accuracy": 0.4117647058823529,
                    "precision": null,
                    "recall": 0.0,
                    "f1": null,
                    "balanced_accuracy": 0.5,
                    "samples": 442
                },
                "majority_class": {
                    "predicted_class": 1,
                    "metrics": {
                        "accuracy": 0.5882352941176471,
                        "precision": 0.5882352941176471,
                        "recall": 1.0,
                        "f1": 0.7407407407407407,
                        "balanced_accuracy": 0.5,
                        "samples": 442
                    }
                }
            }
        }
    ],
    "research_only": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## holdout_shadow_overlay.json
```json
{
    "mode": "final_holdout_shadow_research_only",
    "model_type": "patchtst",
    "success_threshold": 0.2,
    "reduced_exposure": 0.7,
    "rebalance_only": true,
    "overlay_probability": "champion_success_probability",
    "transaction_cost_bps": 5.0,
    "test_start_date": "2024-07-16",
    "result": {
        "base_total_return": 0.3277028094910013,
        "overlay_total_return": 0.24374735830589622,
        "base_max_drawdown": -0.15587793644033,
        "overlay_max_drawdown": -0.15587793644033,
        "reduced_exposure_days": 261,
        "evaluated_days": 441,
        "overlay_turnover": 0.9000000000000001,
        "estimated_cost": 0.00045000000000000004
    },
    "trading_impact": "none",
    "candidate_frozen_before_holdout": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## calibrated_probability_calibration.json
```json
{
    "evaluation": "chronological_holdout_train_quantile_calibration",
    "model_type": "patchtst",
    "label_type": "champion_success",
    "calibrator": "train_fold_quantile_bin_observed_rate",
    "raw_calibration": {
        "sample_count": 442,
        "positive_rate": 0.5927601809954751,
        "brier_score": 0.4508207499610182,
        "base_rate_brier_score": 0.2413955488216886,
        "brier_skill_score": -0.8675603264500353,
        "expected_calibration_error": 0.39584901970575276,
        "maximum_calibration_error": 0.6346179870844453,
        "bins": [
            {
                "lower_bound": 0.0,
                "upper_bound": 0.1,
                "sample_count": 211,
                "mean_predicted_probability": 0.01467111244162112,
                "observed_positive_rate": 0.6492890995260664,
                "calibration_error": 0.6346179870844453
            },
            {
                "lower_bound": 0.1,
                "upper_bound": 0.2,
                "sample_count": 16,
                "mean_predicted_probability": 0.15041284635663033,
                "observed_positive_rate": 0.5625,
                "calibration_error": 0.4120871536433697
            },
            {
                "lower_bound": 0.2,
                "upper_bound": 0.3,
                "sample_count": 14,
                "mean_predicted_probability": 0.23826648188488825,
                "observed_positive_rate": 0.8571428571428571,
                "calibration_error": 0.6188763752579689
            },
            {
                "lower_bound": 0.3,
                "upper_bound": 0.4,
                "sample_count": 5,
                "mean_predicted_probability": 0.3394982874393463,
                "observed_positive_rate": 0.4,
                "calibration_error": 0.06050171256065373
            },
            {
                "lower_bound": 0.4,
                "upper_bound": 0.5,
                "sample_count": 6,
                "mean_predicted_probability": 0.4716931879520416,
                "observed_positive_rate": 0.16666666666666666,
                "calibration_error": -0.305026521285375
            },
            {
                "lower_bound": 0.5,
                "upper_bound": 0.6,
                "sample_count": 130,
                "mean_predicted_probability": 0.5353597439093399,
                "observed_positive_rate": 0.47692307692307695,
                "calibration_error": -0.058436666986262986
            },
            {
                "lower_bound": 0.6,
                "upper_bound": 0.7,
                "sample_count": 5,
                "mean_predicted_probability": 0.6319692611694336,
                "observed_positive_rate": 0.4,
                "calibration_error": -0.23196926116943362
            },
            {
                "lower_bound": 0.7,
                "upper_bound": 0.8,
                "sample_count": 5,
                "mean_predicted_probability": 0.76231849193573,
                "observed_positive_rate": 0.2,
                "calibration_error": -0.56231849193573
            },
            {
                "lower_bound": 0.8,
                "upper_bound": 0.9,
                "sample_count": 5,
                "mean_predicted_probability": 0.8544096112251282,
                "observed_positive_rate": 0.4,
                "calibration_error": -0.45440961122512813
            },
            {
                "lower_bound": 0.9,
                "upper_bound": 1.0,
                "sample_count": 45,
                "mean_predicted_probability": 0.9740063667297363,
                "observed_positive_rate": 0.7555555555555555,
                "calibration_error": -0.21845081117418075
            }
        ]
    },
    "calibrated_calibration": {
        "sample_count": 442,
        "positive_rate": 0.5927601809954751,
        "brier_score": 0.4460276198163708,
        "base_rate_brier_score": 0.2413955488216886,
        "brier_skill_score": -0.8477044087744865,
        "expected_calibration_error": 0.36093165589935383,
        "maximum_calibration_error": 0.6552836859531961,
        "bins": [
            {
                "lower_bound": 0.0,
                "upper_bound": 0.1,
                "sample_count": 206,
                "mean_predicted_probability": 0.004910488804085537,
                "observed_positive_rate": 0.6601941747572816,
                "calibration_error": 0.6552836859531961
            },
            {
                "lower_bound": 0.4,
                "upper_bound": 0.5,
                "sample_count": 174,
                "mean_predicted_probability": 0.4624277456647405,
                "observed_positive_rate": 0.4942528735632184,
                "calibration_error": 0.03182512789847791
            },
            {
                "lower_bound": 0.8,
                "upper_bound": 0.9,
                "sample_count": 23,
                "mean_predicted_probability": 0.8728323699421964,
                "observed_positive_rate": 0.391304347826087,
                "calibration_error": -0.4815280221161094
            },
            {
                "lower_bound": 0.9,
                "upper_bound": 1.0,
                "sample_count": 39,
                "mean_predicted_probability": 0.9982214317474435,
                "observed_positive_rate": 0.7948717948717948,
                "calibration_error": -0.20334963687564866
            }
        ]
    },
    "raw_brier_score": 0.4508207499610182,
    "calibrated_brier_score": 0.4460276198163708,
    "brier_delta_calibrated_minus_raw": -0.004793130144647384,
    "research_only": true,
    "trading_impact": "none",
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## dataset_audit.json
```json
{
    "sample_count": 2214,
    "feature_count": 33,
    "date_coverage": [
        "2017-06-28",
        "2026-04-20"
    ],
    "class_balance": {
        "positive": 1193,
        "negative": 1021,
        "positive_rate": 0.5388437217705511
    },
    "dropped_rows_insufficient_label_horizon": 42,
    "leakage_check_passed": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

## rebalance_dataset_audit.json
```json
{
    "row_count": 106,
    "good_period_rate": 0.6792452830188679,
    "bad_period_rate": 0.18867924528301888,
    "underperforms_spy_rate": 0.4716981132075472,
    "drawdown_event_rate": 0.19811320754716982,
    "history_years": 10,
    "recommended_generalization_years": 10,
    "minimum_history_years": 9,
    "sector_reference_path": "data/reference/sector_by_symbol.json",
    "research_only": true,
    "research_label": "VALIDATED_9Y_RESEARCH_CURRENT_SURVIVOR_UNIVERSE",
    "production_validated": false
}
```

---
