"""A complete built-in multi-asset portfolio backtest."""

from pathlib import Path

from backtester.engine import run_equal_weight_buy_and_hold, run_portfolio_backtest
from backtester.portfolio import (
    InverseVolatilityPortfolio,
    PortfolioConfig,
    build_target_weights,
)
from backtester.portfolio_data import (
    close_price_panel,
    ensure_yfinance_ohlcv_csvs,
    generate_signals,
    load_ohlcv_universe,
)
from backtester.reporting import create_portfolio_report
from backtester.strategies.moving_average import MovingAverageCrossover


TICKERS = ("SPY", "TLT", "GLD")
DATA_START = "2018-01-01"
DATA_END = "2025-01-01"
DATA_INTERVAL = "1d"
DATA_DIRECTORY = Path("data/raw")

FAST_WINDOW = 20
SLOW_WINDOW = 100
PORTFOLIO_CONFIG = PortfolioConfig(
    exposure_mode="gross",
    target_gross_exposure=1.0,
    target_net_exposure=0.0,
    gross_exposure_limit=1.0,
    rebalance_frequency="monthly",
    volatility_lookback=60,
)
ASSET_COST_RATES = {"SPY": 0.0002, "TLT": 0.0002, "GLD": 0.0003}
INITIAL_CAPITAL = 10_000.0
PERIODS_PER_YEAR = 252
REPORT_PATH = Path("reports/multi_asset_ma_report.pdf")


csv_paths = ensure_yfinance_ohlcv_csvs(
    tickers=TICKERS,
    start=DATA_START,
    end=DATA_END,
    interval=DATA_INTERVAL,
    output_directory=DATA_DIRECTORY,
)
universe = load_ohlcv_universe(csv_paths)
prices = close_price_panel(universe)
signals = generate_signals(
    universe,
    strategy_factory=lambda ticker: MovingAverageCrossover(
        fast_window=FAST_WINDOW,
        slow_window=SLOW_WINDOW,
    ),
)
weights = build_target_weights(
    signals,
    prices,
    constructor=InverseVolatilityPortfolio(),
    config=PORTFOLIO_CONFIG,
)

result = run_portfolio_backtest(
    prices,
    weights,
    asset_cost_rates=ASSET_COST_RATES,
    initial_capital=INITIAL_CAPITAL,
)
benchmark = run_equal_weight_buy_and_hold(
    prices,
    asset_cost_rates=ASSET_COST_RATES,
    initial_capital=INITIAL_CAPITAL,
)
report = create_portfolio_report(
    result,
    benchmark,
    PORTFOLIO_CONFIG,
    output_path=REPORT_PATH,
    periods_per_year=PERIODS_PER_YEAR,
    strategy_name="Monthly inverse-volatility moving-average portfolio",
    asset_cost_rates=ASSET_COST_RATES,
)
print(f"Saved {report}")
