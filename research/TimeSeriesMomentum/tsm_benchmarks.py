# Passive benchmark strategies
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

cost_rate = 5 / 10000  # 5 bps transaction cost
initial_capital = 100_000

train_prices = prices.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]

# Strategy 1: SPY buy-and-hold ($100,000 initial capital)
portfolio_constructor = FixedWeightPortfolio({"SPY": 1.0})
portfolio_config = PortfolioConfig(rebalance_frequency="monthly") # Doesn't matter, it won't rebalance
spy_prices = prices[["SPY"]]

target_weights = build_target_weights(
    signals=pd.DataFrame(1.0, index=spy_prices.index, columns=spy_prices.columns),
    prices=spy_prices,
    constructor=portfolio_constructor,
    config=portfolio_config,
)

train_spy_prices = spy_prices.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
train_target_weights = target_weights.loc[train_spy_prices.index]

result = run_portfolio_backtest(
    prices=train_spy_prices,
    target_weights=train_target_weights,
    cost_rate=cost_rate,
    initial_capital=initial_capital,
)

results_directory = Path("research/TimeSeriesMomentum/results/benchmarks")
results_directory.mkdir(parents=True, exist_ok=True)
result.performance.to_csv(results_directory / "s1_train_performance.csv")
result.target_weights.to_csv(results_directory / "s1_train_target_weights.csv")
result.executed_weights.to_csv(results_directory / "s1_train_executed_weights.csv")
result.asset_turnover.to_csv(results_directory / "s1_train_asset_turnover.csv")

print(result.performance.tail())

# Strategy 3-13: Buy-and-hold each individual asset (5 bps transaction cost)
individual_results = {}

for ticker in tickers:
    asset_prices = prices[[ticker]]

    portfolio_constructor = FixedWeightPortfolio({ticker: 1.0})
    portfolio_config = PortfolioConfig(rebalance_frequency="monthly")

    target_weights = build_target_weights(
        signals=pd.DataFrame(1.0, index=asset_prices.index, columns=asset_prices.columns),
        prices=asset_prices,
        constructor=portfolio_constructor,
        config=portfolio_config,
    )

    train_asset_prices = asset_prices.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
    train_target_weights = target_weights.loc[train_asset_prices.index]

    result = run_portfolio_backtest(
        prices=train_asset_prices,
        target_weights=train_target_weights,
        cost_rate=cost_rate,
        initial_capital=initial_capital,
    )

    individual_results[ticker] = result

    result.performance.to_csv(
        results_directory / f"{ticker}_train_performance.csv"
    )

# Strategy 2: Equal-weight long only portfolio (5 bps transaction cost)
individual_equity_curves = pd.DataFrame(
    {
        ticker: result.performance["equity_curve"]
        for ticker, result in individual_results.items()
    }
)

equal_weight_equity = individual_equity_curves.mean(axis=1)
equal_weight_net_return = equal_weight_equity.pct_change().fillna(0.0)

equal_weight_performance = pd.DataFrame(
    {
        "net_return": equal_weight_net_return,
        "equity_curve": equal_weight_equity,
    }
)

equal_weight_performance.to_csv(
    results_directory / "equal_weight_long_only_train_performance.csv"
)
