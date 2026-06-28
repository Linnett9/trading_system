# Stock-Level Feature Attribution and Ablation

Research only. Trading impact: none. Production validated: false.

- Models completed: ridge, elastic_net, random_forest, gradient_boosting
- Eligible rows: 81485
- Permutation repeats per fold: 3
- Promotion thresholds changed: false

## ridge

### Attribution

| Feature | Coefficient | Normalized magnitude | Tree importance | Permutation IC drop |
|---|---:|---:|---:|---:|
| predicted_momentum_60d | -0.006894 | 0.329406 |  | 0.019561 |
| predicted_drawdown_60d | -0.003985 | 0.190419 |  | 0.026381 |
| predicted_momentum_120d | 0.003749 | 0.179130 |  | 0.020443 |
| predicted_volatility_20d | 0.002490 | 0.118986 |  | 0.016053 |
| predicted_momentum_20d | 0.001881 | 0.089860 |  | 0.002248 |
| predicted_risk_adjusted_momentum | 0.001422 | 0.067942 |  | -0.000551 |
| predicted_liquidity_score | 0.000501 | 0.024256 |  | 0.001091 |

### Leave-One-Feature-Out Ablation

| Removed feature | Spearman IC | IC delta | Spread | Spread delta | Hit rate | Risk-adjusted spread | Spread Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| predicted_momentum_120d | 0.034847 | -0.007325 | 0.012487 | -0.002854 | 0.545113 | 0.083618 | 1.831196 |
| predicted_volatility_20d | 0.041147 | -0.001025 | 0.014044 | -0.001297 | 0.550016 | 0.182654 | 2.138658 |
| predicted_liquidity_score | 0.041535 | -0.000637 | 0.015307 | -0.000034 | 0.551324 | 0.159315 | 2.222649 |
| predicted_momentum_20d | 0.042420 | 0.000248 | 0.015259 | -0.000081 | 0.553122 | 0.177237 | 2.206189 |
| predicted_risk_adjusted_momentum | 0.042724 | 0.000552 | 0.016757 | 0.001417 | 0.556064 | 0.213303 | 2.346710 |
| predicted_drawdown_60d | 0.043852 | 0.001680 | 0.016147 | 0.000806 | 0.559823 | 0.221341 | 2.238956 |
| predicted_momentum_60d | 0.046088 | 0.003916 | 0.017684 | 0.002343 | 0.551487 | 0.203965 | 2.294566 |

## elastic_net

### Attribution

| Feature | Coefficient | Normalized magnitude | Tree importance | Permutation IC drop |
|---|---:|---:|---:|---:|
| predicted_momentum_60d | -0.006732 | 0.327951 |  | 0.018633 |
| predicted_drawdown_60d | -0.003986 | 0.194164 |  | 0.023625 |
| predicted_momentum_120d | 0.003695 | 0.180003 |  | 0.017139 |
| predicted_volatility_20d | 0.002464 | 0.120015 |  | 0.015222 |
| predicted_momentum_20d | 0.001845 | 0.089870 |  | 0.003725 |
| predicted_risk_adjusted_momentum | 0.001315 | 0.064040 |  | -0.000883 |
| predicted_liquidity_score | 0.000491 | 0.023957 |  | 0.001098 |

### Leave-One-Feature-Out Ablation

| Removed feature | Spearman IC | IC delta | Spread | Spread delta | Hit rate | Risk-adjusted spread | Spread Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| predicted_momentum_120d | 0.034787 | -0.007379 | 0.012745 | -0.002712 | 0.544949 | 0.080802 | 1.866064 |
| predicted_volatility_20d | 0.041111 | -0.001055 | 0.014214 | -0.001243 | 0.550016 | 0.190568 | 2.155052 |
| predicted_liquidity_score | 0.041565 | -0.000602 | 0.015432 | -0.000025 | 0.551487 | 0.162382 | 2.235658 |
| predicted_momentum_20d | 0.042316 | 0.000150 | 0.015340 | -0.000116 | 0.553122 | 0.183428 | 2.208623 |
| predicted_risk_adjusted_momentum | 0.042680 | 0.000514 | 0.016788 | 0.001332 | 0.556718 | 0.209980 | 2.345430 |
| predicted_drawdown_60d | 0.043756 | 0.001590 | 0.016470 | 0.001013 | 0.560314 | 0.238133 | 2.285524 |
| predicted_momentum_60d | 0.046091 | 0.003925 | 0.017626 | 0.002169 | 0.551324 | 0.202340 | 2.291079 |

## random_forest

### Attribution

| Feature | Coefficient | Normalized magnitude | Tree importance | Permutation IC drop |
|---|---:|---:|---:|---:|
| predicted_drawdown_60d |  | 0.261098 | 0.261098 | 0.034143 |
| predicted_liquidity_score |  | 0.152168 | 0.152168 | 0.005328 |
| predicted_volatility_20d |  | 0.147982 | 0.147982 | 0.009224 |
| predicted_momentum_60d |  | 0.122437 | 0.122437 | 0.007138 |
| predicted_momentum_120d |  | 0.119774 | 0.119774 | 0.011292 |
| predicted_momentum_20d |  | 0.107924 | 0.107924 | 0.005667 |
| predicted_risk_adjusted_momentum |  | 0.088616 | 0.088616 | 0.002019 |

### Leave-One-Feature-Out Ablation

| Removed feature | Spearman IC | IC delta | Spread | Spread delta | Hit rate | Risk-adjusted spread | Spread Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| predicted_momentum_120d | 0.031840 | -0.009822 | 0.007363 | -0.004868 | 0.546747 | -0.021308 | 1.198262 |
| predicted_volatility_20d | 0.034995 | -0.006667 | 0.010549 | -0.001682 | 0.544786 | 0.119912 | 1.863519 |
| predicted_drawdown_60d | 0.036389 | -0.005273 | 0.007821 | -0.004410 | 0.541026 | 0.040133 | 1.400670 |
| predicted_momentum_20d | 0.040356 | -0.001307 | 0.012547 | 0.000316 | 0.546093 | 0.121323 | 2.189528 |
| predicted_risk_adjusted_momentum | 0.040861 | -0.000802 | 0.012111 | -0.000120 | 0.548545 | 0.112858 | 2.141978 |
| predicted_liquidity_score | 0.042159 | 0.000496 | 0.012477 | 0.000246 | 0.546584 | 0.156563 | 2.141390 |
| predicted_momentum_60d | 0.043129 | 0.001467 | 0.013178 | 0.000947 | 0.546911 | 0.134847 | 2.264911 |

## gradient_boosting

### Attribution

| Feature | Coefficient | Normalized magnitude | Tree importance | Permutation IC drop |
|---|---:|---:|---:|---:|
| predicted_drawdown_60d |  | 0.335115 | 0.335115 | 0.020278 |
| predicted_volatility_20d |  | 0.196951 | 0.196951 | 0.015507 |
| predicted_risk_adjusted_momentum |  | 0.162006 | 0.162006 | 0.000090 |
| predicted_momentum_60d |  | 0.114434 | 0.114434 | -0.000330 |
| predicted_momentum_120d |  | 0.068119 | 0.068119 | 0.007740 |
| predicted_liquidity_score |  | 0.067205 | 0.067205 | 0.001227 |
| predicted_momentum_20d |  | 0.056169 | 0.056169 | -0.000025 |

### Leave-One-Feature-Out Ablation

| Removed feature | Spearman IC | IC delta | Spread | Spread delta | Hit rate | Risk-adjusted spread | Spread Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| predicted_volatility_20d | 0.029067 | -0.003680 | 0.006937 | -0.001356 | 0.541680 | 0.048000 | 1.243674 |
| predicted_drawdown_60d | 0.029364 | -0.003382 | 0.003722 | -0.004571 | 0.535633 | -0.005403 | 0.615553 |
| predicted_momentum_120d | 0.030605 | -0.002142 | 0.007599 | -0.000694 | 0.544949 | 0.025380 | 1.181841 |
| predicted_liquidity_score | 0.031187 | -0.001560 | 0.007903 | -0.000390 | 0.538084 | 0.013090 | 1.242421 |
| predicted_risk_adjusted_momentum | 0.032854 | 0.000107 | 0.008481 | 0.000188 | 0.543478 | 0.045929 | 1.348195 |
| predicted_momentum_20d | 0.034254 | 0.001508 | 0.008986 | 0.000693 | 0.540209 | 0.039513 | 1.445977 |
| predicted_momentum_60d | 0.034364 | 0.001618 | 0.009538 | 0.001246 | 0.540700 | 0.077904 | 1.527822 |
