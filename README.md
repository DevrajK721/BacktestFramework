# BacktestFramework

A learning-focused Python framework for transparent single- and multi-asset
portfolio backtests. It prioritises explicit timing, cost accounting, and a
complete audit trail over execution-simulation complexity.

## What it supports

- Separate, validated OHLCV CSV files for every asset.
- Yahoo Finance downloads for 15m, 1h, 3h, 6h, 1d, 1wk, 1mo, and 1y bars.
  Multi-asset downloads write one CSV per ticker.
- Per-asset strategies that emit signed scores from -1 (short) to 1 (long),
  with 0 meaning inactive.
- Equal-weight, fixed-weight, inverse-volatility, and custom portfolio
  construction.
- Gross- and net-exposure conventions, long/short portfolios, an explicit
  gross-exposure limit, and cash when allocations are inactive.
- Daily, weekly, monthly, quarterly, or every-N-bar rebalancing.
- One uniform transaction-cost rate or a separate rate for each asset.
- Close-to-close portfolio accounting with next-bar execution.
- An equal-weight, long-only buy-and-hold benchmark.
- A PDF report containing configuration, performance metrics, equity curves,
  drawdown, weights, and turnover. It is generated from temporary LaTeX and
  chart files; only the final PDF is retained.

## Setup

Python 3.11 or newer and a local LaTeX installation providing pdflatex are
required for PDF reports.

~~~
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -v
~~~

There is intentionally no command-line interface. Research runs should be
ordinary, version-controlled Python scripts so their data choices and
assumptions are visible.

## Complete portfolio script

Use [examples/portfolio_backtest.py](/Users/devrajkatkoria/Documents/BacktestFramework/examples/portfolio_backtest.py)
as the starting point for a real run. It shows the full flow:

1. Download selected tickers once with download_yfinance_ohlcv_many.
2. Supply a mapping from ticker to its CSV path.
3. Load the universe, retaining only timestamps shared by every asset.
4. Generate a signal for each asset.
5. Construct and rebalance target weights.
6. Run both the portfolio and its equal-weight benchmark.
7. Create reports/name.pdf.

The essential API looks like:

~~~
from backtester.engine import run_equal_weight_buy_and_hold, run_portfolio_backtest
from backtester.portfolio import InverseVolatilityPortfolio, PortfolioConfig, build_target_weights
from backtester.portfolio_data import close_price_panel, generate_signals, load_ohlcv_universe

universe = load_ohlcv_universe({"SPY": "data/raw/SPY.csv", "TLT": "data/raw/TLT.csv"})
prices = close_price_panel(universe)
signals = generate_signals(universe, strategy_factory=make_strategy_for_ticker)

config = PortfolioConfig(rebalance_frequency="monthly", volatility_lookback=60)
weights = build_target_weights(signals, prices, InverseVolatilityPortfolio(), config)
result = run_portfolio_backtest(prices, weights, asset_cost_rates={"SPY": 0.0002, "TLT": 0.0002})
benchmark = run_equal_weight_buy_and_hold(prices)
~~~

Set periods_per_year explicitly when reporting. Typical daily, weekly,
monthly, and yearly values are 252, 52, 12, and 1. Intraday annualisation
depends on the asset's trading session and must be selected deliberately.

## Portfolio construction

EqualWeightPortfolio assigns equal magnitude to each active score.
FixedWeightPortfolio({"SPY": 0.6, "TLT": 0.4}) applies those fixed magnitudes
when the asset is active; inactive allocations become cash and are not
re-normalised. InverseVolatilityPortfolio uses trailing return volatility and
defaults to the configurable 60-bar lookback.

### Exposure conventions

In gross mode, absolute asset weights sum to target_gross_exposure. For
example, 50% long and 50% short has 100% gross and 0% net exposure.

In net mode, the asset weights sum to target_net_exposure and, when both long
and short signals exist, use gross_exposure_limit. For example, a 100% net
target and a 140% gross limit gives 120% long and -20% short exposure. An
impossible positive net target with only short signals (or vice versa) raises
a clear error instead of creating unwanted positions.

### Custom constructors

Write a class that inherits from PortfolioConstructor and implements
construct_weights(signals, prices, config). It receives the full aligned
history and returns a DataFrame with the same index and ticker columns as
signals. Keep it free of execution timing, returns, and costs: the engine owns
those calculations.

~~~
import pandas as pd
from backtester.portfolio import PortfolioConstructor

class MyConstructor(PortfolioConstructor):
    def construct_weights(self, signals: pd.DataFrame, prices: pd.DataFrame, config):
        return signals.div(signals.abs().sum(axis=1), axis=0).fillna(0.0)
~~~

Pass MyConstructor() to build_target_weights. Built-in constructors apply the
selected exposure convention automatically; custom constructors are
responsible for their own target-allocation logic, while the framework always
enforces the gross-exposure limit.

## Timing and costs

A strategy and constructor may use information through bar t to form target
weights at t. The engine executes those weights for the return ending at t + 1:

~~~
executed_weights[t] = target_weights[t - 1]
gross_return[t] = sum(executed_weights[t] * asset_return[t])
~~~

Turnover is the sum of absolute per-asset changes in executed weights.
Transaction costs are then calculated per asset and summed. This means a
long-to-short reversal correctly costs two units of that asset's turnover.

## Data contract and limitations

Every CSV must contain date, open, high, low, close, volume. The loader
rejects missing data, duplicate dates, non-positive prices, invalid OHLC
relationships, and negative volume. A multi-asset run uses only the
intersection of timestamps and rejects any remaining missing close.

This is still a research backtester, not a brokerage simulator: it uses
close-to-close returns, assumes fills at the next bar's return period, and
does not model bid-ask spreads, market impact, borrow fees, margin, taxes, or
corporate-action edge cases beyond adjusted Yahoo prices.
