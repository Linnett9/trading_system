# Trading System

A Python trading research system for testing strategies with realistic-ish
execution, risk management, portfolio accounting, parameter optimization, and
walk-forward validation.

This project is currently a research/backtesting tool, not a live trading
system.

## What It Does

- Loads historical market data from Alpaca.
- Runs strategies over candle data.
- Calculates EMA, ATR, RSI, and volatility.
- Applies risk sizing and ATR exits.
- Simulates fills with spread and slippage.
- Tracks open/closed trades and portfolio equity.
- Saves backtest reports as JSON.
- Runs parameter optimization.
- Runs walk-forward validation.
- Compares strategy returns to buy-and-hold benchmarks.
- Analyzes trade quality, duration, and time in market.
- Analyzes capital utilisation, cash drag, and average exposure.
- Applies configurable research quality gates.
- Exports strategy comparison summaries to CSV.
- Supports configurable position sizing modes.
- Compares active strategies against a first-class buy-and-hold baseline.
- Classifies market and volatility regime for strategy filters.

## Indicators

Indicators are calculated in:

```text
core/services/indicator_service.py
```

Current indicators include:

- EMA
- SMA
- RSI
- ATR
- ADX
- volatility
- volatility percentile
- Bollinger bands
- Bollinger bandwidth
- volume moving average
- relative volume
- Donchian high/low channels

These are passed into `StrategyContext` so strategies can use them without
calculating indicators directly. ADX, relative volume, volatility percentile,
and Bollinger bandwidth are mainly used as filters so the bot can avoid weak
trends, low-confirmation breakouts, and poor mean-reversion regimes.

## Main Commands

Run all tests:

```bash
pytest -q tests
```

Run a normal multi-symbol backtest:

```bash
python main.py --mode backtest
```

Run parameter optimization:

```bash
python main.py --mode optimize
```

Run walk-forward validation:

```bash
python main.py --mode walk-forward
```

Compare multiple strategies:

```bash
python main.py --mode compare-strategies
```

Print every comparison row instead of only the top results:

```bash
python main.py --mode compare-strategies --all-results
```

Show fold-level walk-forward details:

```bash
python main.py --mode walk-forward --details
```

## Configuration

Edit:

```text
config/config.yaml
```

Important sections:

```yaml
backtest:
  symbols:
    - AAPL
    - MSFT
    - TSLA
    - SPY
    - QQQ
  timeframe: "1Day"
  years: 5
  starting_equity: 500
  warmup_bars: 200

strategy:
  name: "ema_crossover"
  ema_fast_period: 50
  ema_slow_period: 200

risk:
  manager: "atr"
  max_risk_per_trade: 0.01
  max_exposure: 0.20
  atr_stop_multiplier: 2.0
  atr_take_profit_multiplier: 3.0
  trailing_atr_multiplier: 3.0

position_sizing:
  mode: "fixed_fractional"
  target_exposure: 0.20
  max_exposure: 0.20

research:
  optimization_metric: "composite"
  report_top_n: 10
  parallel_workers: 1
  parallel_mode: "thread"
  two_stage_enabled: true
  two_stage_top_n: 5
  stage_one_max_combinations: 80
  early_stop_max_drawdown: 0.30
  early_stop_equity_floor_pct: 0.70
  optimizer_min_closed_trades: 1
  min_closed_trades: 20
  min_profit_factor: 1.1
  max_drawdown: 0.20
  min_time_in_market: 0.02
  max_time_in_market: 0.95
  require_positive_excess: true
  require_sharpe_edge: true

cache:
  enabled: true
  data_dir: "cache/data"
  results_dir: "cache/results"
```

## Reports

Backtest reports are saved to:

```text
reports/backtests/
```

Walk-forward reports are saved to:

```text
reports/walk_forward/
```

Strategy comparison summaries are saved to:

```text
reports/summary/latest_experiment.csv
```

Each walk-forward report contains:

- symbol
- timeframe
- fold windows
- best training parameters
- train return and Sharpe
- test return and Sharpe
- closed trades
- benchmark return
- benchmark Sharpe and max drawdown
- excess return versus benchmark
- trade analysis, including win rate, expectancy, duration, and time in market
- capital utilisation, including exposure, cash percentage, and position value
- signal diagnostics, including BUY/SELL/HOLD counts and blocked signals

## How To Read Walk-Forward Output

Example:

```text
Symbol | Folds | Test Ret | Benchmark | Excess | Sharpe | Bench Sh | Trades | Report
AAPL   |     3 |   -0.03% |    18.43% | -18.46% |  -1.10 |    1.24 |      2 | ...
```

Meaning:

- `Test Ret`: average out-of-sample strategy return across folds.
- `Benchmark`: average buy-and-hold return over the same test windows.
- `Excess`: strategy return minus benchmark return.
- `Sharpe`: average test-window Sharpe.
- `Bench Sh`: benchmark Sharpe over the same test windows.
- `Trades`: total closed trades across all test folds.

The equal-weight average line treats every symbol equally:

```text
Equal-weight average | test=... | benchmark=... | excess=... | sharpe=... | trades=...
```

This is not a true portfolio backtest with shared capital. It is a quick
research summary that asks:

```text
If each symbol mattered equally, did the strategy beat buy-and-hold?
```

If `excess` is strongly negative, the strategy is probably not adding value
yet, even if some symbols show positive Sharpe.

## Current Code Flow

Normal backtest:

```text
main.py
  -> load config
  -> fetch candles from Alpaca
  -> core.research.backtest_runner.run_backtest()
  -> BacktestEngine.run()
  -> MarketDataService stores candles
  -> IndicatorService calculates indicators
  -> Strategy generates signal
  -> Risk manager validates and sizes
  -> ExecutionEngine creates fill
  -> TradeManager opens/closes trades
  -> PortfolioEngine updates equity
  -> BacktestResult returned and saved
```

Walk-forward:

```text
main.py --mode walk-forward
  -> fetch candles per symbol
  -> split candles into train/test folds
  -> optimize parameters on train window
  -> run best params on test window
  -> compare test result to buy-and-hold
  -> save walk-forward JSON report
```

## What The Current Results Suggest

If the output looks like:

```text
test=0.00%
benchmark=20.00%
excess=-20.00%
```

then the strategy is mostly sitting out or producing tiny returns while the
asset itself moved strongly. That tells you the current strategy/risk/exits
are not capturing enough of the trend.

Low trade counts are also a warning. A fold with `trades=0` or `trades=1` is
not enough evidence to trust the Sharpe value.

Trade analysis fields help explain why a strategy underperforms:

- `win_rate`
- `average_win`
- `average_loss`
- `expectancy`
- `profit_factor`
- `average_trade_duration_days`
- `time_in_market_percent`

For example, a strategy with positive trades but very low
`time_in_market_percent` may be too inactive rather than structurally broken.

Capital utilisation fields help identify under-investment:

- `average_position_value`
- `average_exposure_percent`
- `max_exposure_percent`
- `average_cash_percent`
- `average_leverage`

For a `$500` account with `max_exposure: 0.20`, a fully sized position is only
about `$100`. If reports show high cash and low average exposure, returns may
be limited by capital deployment rather than entry quality.

Position sizing modes live in:

```text
core/risk/position_sizer.py
```

Available modes:

- `fixed_fractional`: target a fixed percentage of equity per position.
- `fixed_dollar`: target a fixed dollar value per position.
- `atr`: size by ATR stop distance and exposure cap.
- `volatility`: reduce exposure as volatility rises.

The default research setup uses `fixed_fractional` at `20%` target exposure so
a `$500` account puts roughly `$100` into a fully sized position. You can test
larger or smaller sizing by changing `position_sizing.target_exposure` or the
`target_exposure` values in the research grids.

Signal diagnostics explain strategy activity:

- `buy_signals`
- `sell_signals`
- `hold_signals`
- `duplicate_buy_skips`
- `flat_sell_skips`
- `risk_blocked_signals`
- `stop_loss_exits`
- `take_profit_exits`

Use these to separate three different problems:

```text
No BUY signals        -> entry rules are too strict.
Many risk blocks      -> sizing/risk rules are too restrictive.
Many early exits      -> stops/take-profits may be too tight.
```

Trend strategies can also use ATR trailing stops. A `null` take-profit in the
research grid means no fixed take-profit, allowing the trailing stop or
strategy exit rule to manage the position.

The strategy comparison table includes compact signal columns:

```text
B/S/H  -> BUY / SELL / HOLD signals
Blocks -> duplicate BUY / flat SELL / risk-blocked signals
Exits  -> stop-loss / take-profit exits
```

## Strategy Comparison

Use strategy comparison when the current strategy does not trade enough or
does not beat the benchmark:

```bash
python main.py --mode compare-strategies
```

This runs the configured strategy families across symbols and ranks them by:

- qualified score
- composite score
- walk-forward excess return
- walk-forward Sharpe
- closed trades
- max drawdown
- profit factor
- capital exposure
- benchmark-adjusted risk metrics

Configured strategy families currently include:

- EMA crossover
- EMA + RSI filter
- EMA + RSI pullback
- RSI mean reversion
- RSI sideways mean reversion
- Donchian breakout
- Donchian + volatility filter
- Bollinger mean reversion
- Trend pullback
- Buy-and-hold baseline

`QScore` is the ranking score. It applies hard penalties for:

```text
negative excess return
too few trades
0 passed folds
low profit factor
```

Buy-and-hold uses a separate benchmark scoring path, so it is not punished for
having a small trade count.

Trend pullback uses:

```text
Trend: 50 EMA > 200 EMA
Entry: price pulls back near the 20 SMA
Exit: extension above the 20 SMA or bearish trend flip
Optional filter: bull market and non-high-volatility regime
```

Buy-and-hold is included as a normal strategy row so the framework can compare
timing systems against an always-invested baseline in the same output.

The comparison command also writes:

```text
reports/summary/latest_experiment.csv
```

That CSV is the easiest place to sort by excess return, exposure, profit
factor, drawdown, or trades.

By default, the terminal prints only the top configured rows:

```yaml
research:
  report_top_n: 10
```

The full result set is still saved to CSV. Use `--all-results` when you want to
print every row.

## Speed Controls

The research loop has several controls to keep iteration manageable:

- Candle cache: `cache/data/`
- Backtest result cache: `cache/results/`
- Top-N terminal output: `research.report_top_n`
- Two-stage optimisation: `research.two_stage_enabled`
- Stage-one search cap: `research.stage_one_max_combinations`
- Optional worker count: `research.parallel_workers`
- Optional parallel mode: `research.parallel_mode`, either `thread` or `process`

Process mode can be faster for larger searches, but `thread` is the safer
default while developing because it avoids process startup overhead and
pickle-related surprises.

Fast development setup:

```yaml
backtest:
  symbols:
    - AAPL
    - SPY
  years: 2

research:
  report_top_n: 10
  stage_one_max_combinations: 40
  min_closed_trades: 10
```

Full research setup:

```yaml
backtest:
  symbols:
    - AAPL
    - MSFT
    - TSLA
    - SPY
    - QQQ
  years: 5

research:
  stage_one_max_combinations: 80
  min_closed_trades: 20
```

## Suggested Improvements

Improve diagnostics first:

1. Add parameter stability reports across folds.
2. Add benchmark equity curves, not just benchmark summary metrics.
3. Add exit-reason breakdowns by strategy and symbol.
4. Add true shared-capital multi-asset portfolio testing.

Next implementation plan:

1. Run strategy comparison with the new position sizing.
2. Inspect `B/S/H`, `Blocks`, `Exits`, `Exposure`, `Cash`, and `Passed`.
3. Compare active strategies directly against the buy-and-hold rows.
4. If trades remain low, loosen entry rules or add more active variants.
5. If a strategy finally shows positive excess, build a stability report.

Then improve strategy logic:

1. Add a trend filter: only long if close is above a long EMA.
2. Add a volatility filter: skip entries during extreme volatility.
3. Add RSI confirmation for trend entries.
4. Test alternative strategies: breakout, mean reversion, RSI, ensemble.

Then improve portfolio realism:

1. Shared capital across symbols.
2. Max portfolio exposure.
3. Correlation control.
4. Sector/symbol allocation limits.
5. Cash drag and commission modelling.

## Testing

Run the full local suite:

```bash
pytest -q tests
```

Some Alpaca integration tests may be skipped if the Alpaca SDK is unavailable.
That is expected for local unit testing.
