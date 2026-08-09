"""Portfolio construction and scheduling for multi-asset backtests.

Strategies produce one signed score per asset.  A portfolio constructor turns
those scores into target asset weights; this module then applies the requested
rebalance schedule.  It deliberately contains no return or cost accounting -
that belongs to :mod:`backtester.engine`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd


ExposureMode = Literal["gross", "net"]
RebalanceFrequency = Literal["daily", "weekly", "monthly", "quarterly"] | int


@dataclass(frozen=True)
class PortfolioConfig:
    """Controls exposure, scheduling, and inverse-volatility lookback.

    In ``gross`` mode, the absolute asset weights sum to
    ``target_gross_exposure``.  In ``net`` mode, the asset weights sum to
    ``target_net_exposure`` and use ``gross_exposure_limit`` whenever both
    long and short signals are present.  Therefore, a net-long/short
    portfolio needs ``gross_exposure_limit`` to be greater than the absolute
    net target.
    """

    exposure_mode: ExposureMode = "gross"
    target_gross_exposure: float = 1.0
    target_net_exposure: float = 1.0
    gross_exposure_limit: float = 1.0
    rebalance_frequency: RebalanceFrequency = "monthly"
    volatility_lookback: int = 60

    def __post_init__(self) -> None:
        if self.exposure_mode not in {"gross", "net"}:
            raise ValueError("exposure_mode must be 'gross' or 'net'")
        if self.target_gross_exposure < 0.0:
            raise ValueError("target_gross_exposure must be non-negative")
        if self.gross_exposure_limit < 0.0:
            raise ValueError("gross_exposure_limit must be non-negative")
        if self.target_gross_exposure > self.gross_exposure_limit:
            raise ValueError(
                "target_gross_exposure cannot exceed gross_exposure_limit"
            )
        if abs(self.target_net_exposure) > self.gross_exposure_limit:
            raise ValueError(
                "absolute target_net_exposure cannot exceed gross_exposure_limit"
            )
        if self.volatility_lookback < 2:
            raise ValueError("volatility_lookback must be at least 2")
        _validate_rebalance_frequency(self.rebalance_frequency)


class PortfolioConstructor(ABC):
    """Base interface for built-in and user-defined portfolio constructors.

    A custom constructor receives every aligned price and signal observation,
    and must return a DataFrame of *raw desired weights* with exactly the same
    index and ticker columns as ``signals``.  ``build_target_weights`` applies
    the rebalance schedule and validates the resulting exposures.
    """

    @abstractmethod
    def construct_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: PortfolioConfig,
    ) -> pd.DataFrame:
        """Return desired asset weights before rebalance scheduling."""


class EqualWeightPortfolio(PortfolioConstructor):
    """Allocate equal magnitude to every non-zero signed signal."""

    def construct_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: PortfolioConfig,
    ) -> pd.DataFrame:
        _validate_signals_and_prices(signals, prices)
        signs = np.sign(signals)
        magnitudes = signs.abs()
        raw = signs.div(magnitudes.sum(axis=1).replace(0.0, np.nan), axis=0)
        return _apply_exposure_targets(raw.fillna(0.0), config)


class FixedWeightPortfolio(PortfolioConstructor):
    """Apply fixed asset magnitudes when their corresponding signal is active.

    Inactive assets become cash.  Fixed weights are never re-normalised, so an
    inactive allocation stays uninvested exactly as requested.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        self.weights = pd.Series(weights, dtype=float)
        if self.weights.index.has_duplicates:
            raise ValueError("fixed weight tickers must be unique")
        if self.weights.isna().any() or (self.weights < 0.0).any():
            raise ValueError("fixed weights must be finite and non-negative")

    def construct_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: PortfolioConfig,
    ) -> pd.DataFrame:
        _validate_signals_and_prices(signals, prices)
        if set(self.weights.index) != set(signals.columns):
            raise ValueError("fixed weights must contain exactly the signal tickers")
        magnitudes = self.weights.reindex(signals.columns)
        if magnitudes.sum() > config.gross_exposure_limit + 1e-12:
            raise ValueError("fixed weights exceed gross_exposure_limit")
        raw = np.sign(signals).mul(magnitudes, axis=1)
        if config.exposure_mode == "net":
            return _apply_exposure_targets(raw, config)
        return raw


class InverseVolatilityPortfolio(PortfolioConstructor):
    """Allocate active signed signals inversely to trailing return volatility."""

    def construct_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: PortfolioConfig,
    ) -> pd.DataFrame:
        _validate_signals_and_prices(signals, prices)
        returns = prices.pct_change(fill_method=None)
        volatility = returns.rolling(
            config.volatility_lookback,
            min_periods=config.volatility_lookback,
        ).std(ddof=1)
        inverse_volatility = (1.0 / volatility).replace([np.inf, -np.inf], np.nan)
        raw = np.sign(signals) * inverse_volatility
        raw = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        return _apply_exposure_targets(raw.fillna(0.0), config)


def build_target_weights(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    constructor: PortfolioConstructor,
    config: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """Construct and hold target weights until the next rebalance date."""
    config = config or PortfolioConfig()
    _validate_signals_and_prices(signals, prices)
    raw_weights = constructor.construct_weights(signals, prices, config)
    _validate_weight_frame(raw_weights, signals.index, signals.columns)

    rebalance = rebalance_mask(signals.index, config.rebalance_frequency)
    target_weights = raw_weights.where(rebalance, np.nan).ffill().fillna(0.0)
    _validate_exposure(target_weights, config)
    return target_weights.astype(float)


def rebalance_mask(index: pd.DatetimeIndex, frequency: RebalanceFrequency) -> pd.Series:
    """Return dates on which a new target portfolio may be formed."""
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("index must be a non-empty DatetimeIndex")
    _validate_rebalance_frequency(frequency)

    if isinstance(frequency, int):
        values = np.arange(len(index)) % frequency == 0
    elif frequency == "daily":
        values = np.ones(len(index), dtype=bool)
    else:
        periods = {
            "weekly": index.to_period("W"),
            "monthly": index.to_period("M"),
            "quarterly": index.to_period("Q"),
        }[frequency]
        values = np.r_[True, periods[1:] != periods[:-1]]
    return pd.Series(values, index=index, name="rebalance")


def _apply_exposure_targets(raw_weights: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Scale signed raw weights to the configured gross or net convention."""
    if config.exposure_mode == "gross":
        gross = raw_weights.abs().sum(axis=1).replace(0.0, np.nan)
        return raw_weights.mul(config.target_gross_exposure).div(gross, axis=0).fillna(0.0)

    result = pd.DataFrame(0.0, index=raw_weights.index, columns=raw_weights.columns)
    for timestamp, row in raw_weights.iterrows():
        long_assets = row[row > 0.0]
        short_assets = row[row < 0.0]
        if long_assets.empty and short_assets.empty:
            continue
        if long_assets.empty:
            if config.target_net_exposure > 0.0:
                raise ValueError(
                    "positive target_net_exposure cannot be met with only short signals"
                )
            gross = abs(config.target_net_exposure)
            result.loc[timestamp, short_assets.index] = (
                short_assets / short_assets.abs().sum() * gross
            )
            continue
        if short_assets.empty:
            if config.target_net_exposure < 0.0:
                raise ValueError(
                    "negative target_net_exposure cannot be met with only long signals"
                )
            gross = abs(config.target_net_exposure)
            result.loc[timestamp, long_assets.index] = (
                long_assets / long_assets.sum() * gross
            )
            continue

        gross = config.gross_exposure_limit
        long_total = (gross + config.target_net_exposure) / 2.0
        short_total = (gross - config.target_net_exposure) / 2.0
        result.loc[timestamp, long_assets.index] = (
            long_assets / long_assets.sum() * long_total
        )
        result.loc[timestamp, short_assets.index] = (
            short_assets / short_assets.abs().sum() * short_total
        )
    return result


def _validate_signals_and_prices(signals: pd.DataFrame, prices: pd.DataFrame) -> None:
    if signals.empty or prices.empty:
        raise ValueError("signals and prices must not be empty")
    if not signals.index.equals(prices.index) or not signals.columns.equals(prices.columns):
        raise ValueError("signals and prices must have matching index and columns")
    if signals.isna().any().any() or prices.isna().any().any():
        raise ValueError("signals and prices must not contain missing values")
    if not np.isfinite(signals.to_numpy(dtype=float)).all():
        raise ValueError("signals must be finite")
    if ((signals < -1.0) | (signals > 1.0)).any().any():
        raise ValueError("signals must be within range [-1.0, 1.0]")
    if (prices <= 0.0).any().any():
        raise ValueError("prices must be positive")


def _validate_weight_frame(
    weights: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
) -> None:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("construct_weights must return a DataFrame")
    if not weights.index.equals(index) or not weights.columns.equals(columns):
        raise ValueError("weights must have the same index and columns as signals")
    if weights.isna().any().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("weights must be finite and contain no missing values")


def _validate_exposure(weights: pd.DataFrame, config: PortfolioConfig) -> None:
    gross = weights.abs().sum(axis=1)
    if (gross > config.gross_exposure_limit + 1e-10).any():
        raise ValueError("weights exceed gross_exposure_limit")


def _validate_rebalance_frequency(frequency: RebalanceFrequency) -> None:
    if isinstance(frequency, bool):
        raise ValueError("rebalance_frequency cannot be a boolean")
    if isinstance(frequency, int):
        if frequency < 1:
            raise ValueError("integer rebalance_frequency must be at least 1")
        return
    if frequency not in {"daily", "weekly", "monthly", "quarterly"}:
        raise ValueError(
            "rebalance_frequency must be daily, weekly, monthly, quarterly, or a positive integer"
        )
