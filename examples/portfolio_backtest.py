"""A complete script-based multi-asset portfolio backtest."""

from backtester.engine import run_equal_weight_buy_and_hold, run_portfolio_backtest
from backtester.portfolio import (
    InverseVolatilityPortfolio,
    PortfolioConfig,
    build_target_weights,
)
from backtester.portfolio_data import (
    close_price_panel,
    download_yfinance_ohlcv_many,
    generate_signals,
    load_ohlcv_universe,
)
from backtester.reporting import create_portfolio_report
from backtester.strategies.moving_average import MovingAverageCrossover


# Run this once when the source CSVs do not already exist.
# download_yfinance_ohlcv_many(
#     tickers=["SPY", "TLT", "GLD"],
#     start="2018-01-01",
#     end="2025-01-01",
#     interval="1d",
#     output_directory="data/raw",
# )

csv_paths = {
    "SPY": "data/raw/SPY.csv",
    "TLT": "data/raw/TLT.csv",
    "GLD": "data/raw/GLD.csv",
}
universe = load_ohlcv_universe(csv_paths)
prices = close_price_panel(universe)

signals = generate_signals(
    universe,
    strategy_factory=lambda ticker: MovingAverageCrossover(
        fast_window=20,
        slow_window=100,
    ),
)

config = PortfolioConfig(
    exposure_mode="gross",
    target_gross_exposure=1.0,
    gross_exposure_limit=1.0,
    rebalance_frequency="monthly",
    volatility_lookback=60,
)
weights = build_target_weights(
    signals,
    prices,
    constructor=InverseVolatilityPortfolio(),
    config=config,
)

costs = {"SPY": 0.0002, "TLT": 0.0002, "GLD": 0.0003}
result = run_portfolio_backtest(
    prices,
    weights,
    asset_cost_rates=costs,
    initial_capital=10_000.0,
)
benchmark = run_equal_weight_buy_and_hold(
    prices,
    asset_cost_rates=costs,
    initial_capital=10_000.0,
)

report = create_portfolio_report(
    result,
    benchmark,
    config,
    output_path="reports/multi_asset_ma_report.pdf",
    periods_per_year=252,
    strategy_name="Monthly inverse-volatility moving-average portfolio",
    asset_cost_rates=costs,
)
print(f"Saved {report}")
