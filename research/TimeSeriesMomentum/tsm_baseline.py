# Simple baseline strategy 
import pandas as pd 
import numpy as np
import json 
from pathlib import Path 

from backtester.portfolio import (FixedWeightPortfolio,
                                PortfolioConfig, build_target_weights)
from backtester.portfolio_data import close_price_panel, load_ohlcv_universe

from backtester.engine import run_portfolio_backtest

TRAIN_RANGE = ("2008-07-01", "2015-06-30")
VALIDATION_RANGE = ("2015-07-01", "2019-06-30")
TEST_RANGE = ("2019-07-01", "2026-07-01") # DO NOT TOUCH

tickers = json.load(open("research/TimeSeriesMomentum/data/metadata.json"))["Universe Tickers"]

raw_data_directory = Path("research/TimeSeriesMomentum/data/raw")
ticker_map = {
    ticker: raw_data_directory / f"{ticker}.csv"
    for ticker in tickers
}
universe = load_ohlcv_universe(ticker_map)
prices = close_price_panel(universe)

# Momentum signal
momentum_df = np.sign(prices.pct_change(252)).fillna(0.0)

# Rebalance monthly (5 bps transaction cost)
portfolio_constructor = FixedWeightPortfolio({ticker: 0.10 for ticker in tickers})
portfolio_config = PortfolioConfig(rebalance_frequency="monthly")
target_weights = build_target_weights(
    signals=momentum_df,
    prices=prices,
    constructor=portfolio_constructor,
    config=portfolio_config,
)
cost_rate = 5 / 10000  # 5 bps transaction cost
initial_capital = 100_000

train_prices = prices.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
train_target_weights = target_weights.loc[train_prices.index]
result = run_portfolio_backtest(
    prices=train_prices,
    target_weights=train_target_weights,
    cost_rate=cost_rate,
    initial_capital=initial_capital,
)

results_directory = Path("research/TimeSeriesMomentum/results/baseline")
results_directory.mkdir(parents=True, exist_ok=True)
result.performance.to_csv(results_directory / "train_performance.csv")
result.target_weights.to_csv(results_directory / "train_target_weights.csv")
result.executed_weights.to_csv(results_directory / "train_executed_weights.csv")
result.asset_turnover.to_csv(results_directory / "train_asset_turnover.csv")

print(result.performance.tail())

