# Project State

## Goal
Build a Python trading system for backtesting first, then paper trading.

## Current Strategy Focus
Dual momentum portfolio strategy.

## Current Best Candidate
scaled_fast_reentry_chop

Latest risk-regime experiment:
- return: 112.86%
- SPY benchmark: 63.75%
- excess vs SPY: +49.11%
- Sharpe: 1.19
- max drawdown: 15.17%
- turnover: 591.43%

## Current Problem
The default command:

python main.py --mode dual-momentum

currently gives a bad result:
- return: 3.05%
- Sharpe: 0.15
- max drawdown: 28.28%
- cash: 60.38%

This suggests the default dual-momentum mode may be using the wrong configuration, likely involving quality/cooldown/over-filtering.

## Current Priority
Fix default dual-momentum so it runs the intended candidate baseline:
- scaled_fast_reentry_chop
- mix: 75%
- off: 25%
- fallback: 0%
- chop: true
- quality: false
- cooldown: false
- decay: false

## Do Not Add Yet
- Random Forest
- XGBoost
- LSTM
- live trading
- new strategies