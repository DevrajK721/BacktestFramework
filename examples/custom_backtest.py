"""A from-scratch template with custom assets, signal, and constructor.

Every setting is declared explicitly below: this example intentionally relies
on no strategy, portfolio, cost, reporting, or configuration defaults.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtester.engine import run_equal_weight_buy_and_hold, run_portfolio_backtest
from backtester.portfolio import (
    PortfolioConfig,
    PortfolioConstructor,
    build_target_weights,
)
from backtester.portfolio_data import (
    close_price_panel,
    ensure_yfinance_ohlcv_csvs,
    generate_signals,
    load_ohlcv_universe,
)
from backtester.reporting import create_portfolio_report
from backtester.strategy import Strategy


class ThresholdMomentumSignal(Strategy):
    """Go long/short only when a trailing close return clears a threshold."""

    def __init__(self, lookback_bars: int, return_threshold: float) -> None:
        if lookback_bars < 1:
            raise ValueError("lookback_bars must be at least 1")
        if return_threshold < 0.0:
            raise ValueError("return_threshold must be non-negative")
        self.lookback_bars = lookback_bars
        self.return_threshold = return_threshold

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        trailing_return = data["close"].pct_change(
            periods=self.lookback_bars,
            fill_method=None,
        ).fillna(0.0)
        signal = pd.Series(0.0, index=data.index, name="target_position")
        signal.loc[trailing_return > self.return_threshold] = 1.0
        signal.loc[trailing_return < -self.return_threshold] = -1.0
        return signal


class SignalStrengthPortfolio(PortfolioConstructor):
    """Allocate gross exposure in proportion to each active signal strength."""

    def construct_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: PortfolioConfig,
    ) -> pd.DataFrame:
        if config.exposure_mode != "gross":
            raise ValueError("SignalStrengthPortfolio is written for gross mode")
        active_strength = signals.abs().sum(axis=1).replace(0.0, np.nan)
        proportional_weights = signals.div(active_strength, axis=0).fillna(0.0)
        return proportional_weights * config.target_gross_exposure


TICKERS = ("SPY", "TLT", "GLD")
DATA_START = "2018-01-01"
DATA_END = "2025-01-01"
DATA_INTERVAL = "1d"
DATA_DIRECTORY = Path("data/raw")

MOMENTUM_LOOKBACK_BARS = 126
MOMENTUM_RETURN_THRESHOLD = 0.05
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
REPORT_PATH = Path("reports/custom_momentum_report.pdf")


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
    strategy_factory=lambda ticker: ThresholdMomentumSignal(
        lookback_bars=MOMENTUM_LOOKBACK_BARS,
        return_threshold=MOMENTUM_RETURN_THRESHOLD,
    ),
)
weights = build_target_weights(
    signals,
    prices,
    constructor=SignalStrengthPortfolio(),
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
    strategy_name="Monthly threshold-momentum strength portfolio",
    asset_cost_rates=ASSET_COST_RATES,
)
print(f"Saved {report}")
