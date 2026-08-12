# Validation to tune parameteres 
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

# Build the momentum signal (different lookbacks)
LOOKBACKS = [7, 21, 63, 126, 189, 252]
REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly", "quarterly", "yearly"]

for lb in LOOKBACKS:
    momentum_df = np.sign(prices.pct_change(lb)).fillna(0.0)
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

    validation_prices = prices.loc[VALIDATION_RANGE[0]:VALIDATION_RANGE[1]]
    validation_target_weights = target_weights.loc[validation_prices.index]
    result = run_portfolio_backtest(
        prices=validation_prices,
        target_weights=validation_target_weights,
        cost_rate=cost_rate,
        initial_capital=initial_capital,        
    )

    results_dir = Path("research/TimeSeriesMomentum/results/validation")
    results_dir.mkdir(parents=True, exist_ok=True)
    result.performance.to_csv(results_dir / f"validation_performance_lb{lb}.csv", index=True)
    result.target_weights.to_csv(results_dir / f"validation_target_weights_lb{lb}.csv", index=True)
    result.executed_weights.to_csv(results_dir / f"validation_executed_weights_lb{lb}.csv", index=True) 
    result.asset_turnover.to_csv(results_dir / f"validation_asset_turnover_lb{lb}.csv", index=True)

    print(result.performance.tail())



