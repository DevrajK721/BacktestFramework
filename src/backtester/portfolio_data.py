"""Data helpers for aligned, multi-asset portfolio research."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

from backtester.data import download_yfinance_ohlcv, load_ohlcv_csv
from backtester.strategy import Strategy


def load_ohlcv_universe(paths: Mapping[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Load separate ticker CSVs and retain only their shared timestamps."""
    if not paths:
        raise ValueError("paths must not be empty")
    if any(not ticker.strip() for ticker in paths):
        raise ValueError("ticker names must not be empty")
    universe = {ticker: load_ohlcv_csv(path) for ticker, path in paths.items()}
    shared_index = _shared_index(universe)
    if shared_index.empty:
        raise ValueError("assets have no shared timestamps")
    return {ticker: data.loc[shared_index].copy() for ticker, data in universe.items()}


def close_price_panel(universe: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return aligned close prices, rejecting absent or invalid observations."""
    if not universe:
        raise ValueError("universe must not be empty")
    for ticker, data in universe.items():
        if "close" not in data.columns:
            raise ValueError(f"{ticker} data must contain a close column")
    shared_index = _shared_index(universe)
    if shared_index.empty:
        raise ValueError("assets have no shared timestamps")
    prices = pd.DataFrame(
        {ticker: data.loc[shared_index, "close"] for ticker, data in universe.items()},
        index=shared_index,
    )
    if prices.isna().any().any() or (prices <= 0.0).any().any():
        raise ValueError("aligned close prices must be present and positive")
    return prices.astype(float)


def download_yfinance_ohlcv_many(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    output_directory: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download each ticker separately, optionally writing one CSV per asset."""
    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("tickers must not be empty")
    if len(set(ticker_list)) != len(ticker_list):
        raise ValueError("tickers must be unique")
    destination = Path(output_directory) if output_directory is not None else None
    result: dict[str, pd.DataFrame] = {}
    for ticker in ticker_list:
        path = destination / f"{ticker}.csv" if destination is not None else None
        result[ticker] = download_yfinance_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            output_path=path,
        )
    return result


def generate_signals(
    universe: Mapping[str, pd.DataFrame],
    strategy_factory: Callable[[str], Strategy],
) -> pd.DataFrame:
    """Generate one aligned signed signal column per asset.

    ``strategy_factory`` receives a ticker and should create its strategy.  A
    factory avoids unintentionally sharing mutable strategy state across assets.
    """
    if not universe:
        raise ValueError("universe must not be empty")
    shared_index = _shared_index(universe)
    signals: dict[str, pd.Series] = {}
    for ticker, data in universe.items():
        signal = strategy_factory(ticker).generate_positions(data.loc[shared_index])
        if not signal.index.equals(shared_index):
            raise ValueError(f"{ticker} strategy signal must match the aligned index")
        signals[ticker] = signal
    result = pd.DataFrame(signals, index=shared_index, dtype=float)
    if result.isna().any().any():
        raise ValueError("strategy signals must not contain missing values")
    if ((result < -1.0) | (result > 1.0)).any().any():
        raise ValueError("strategy signals must be within range [-1.0, 1.0]")
    return result


def _shared_index(universe: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    iterator = iter(universe.values())
    try:
        shared_index = next(iterator).index
    except StopIteration as error:
        raise ValueError("universe must not be empty") from error
    for data in iterator:
        shared_index = shared_index.intersection(data.index)
    return pd.DatetimeIndex(shared_index.sort_values(), name="date")
