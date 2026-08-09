import pandas as pd
import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass

from backtester.costs import calculate_turnover, calculate_transaction_costs

def run_backtest(data: pd.DataFrame, target_positions: pd.Series, cost_rate: float = 0.0, initial_capital: float = 1.0) -> pd.DataFrame:
    """Run a close-to-close backtest for one asset with proportional costs.

    Parameters
    ----------
    data:
        Clean market data indexed by date and containing a ``close`` column.
    target_positions:
        Strategy target exposures aligned to ``data.index``. A target set using
        date *t* information is applied to the return ending on date *t + 1*.
    cost_rate:
        Proportional transaction cost per unit of turnover. For example,
        ``0.0005`` represents 5 basis points.
    initial_capital:
        Starting portfolio value used to scale the equity curve.

    Returns
    -------
    pd.DataFrame
        An audit table with ``close``, ``asset_return``, ``target_position``,
        ``executed_position``, ``gross_return``, ``turnover``,
        ``transaction_cost``, ``net_return``, and ``equity_curve`` columns.
    """
    if "close" not in data.columns:
        raise ValueError("data must contain a close column")

    if not target_positions.index.equals(data.index):
        raise ValueError("target_positions index must equal data index")

    if target_positions.isna().any():
        raise ValueError("target_positions contain missing values")

    if ((target_positions < -1.0) | (target_positions > 1.0)).any():
        raise ValueError("target_positions must be within range [-1.0, 1.0]")

    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive")

    # Compute asset return 
    asset_return = data["close"].pct_change(fill_method=None).fillna(0.0)

    executed_position = target_positions.shift(1).fillna(0.0) 
    turnover = calculate_turnover(executed_position)
    transaction_cost = calculate_transaction_costs(turnover, cost_rate)

    gross_return = asset_return * executed_position
    net_return = gross_return - transaction_cost

    equity_curve = (1.0 + net_return).cumprod() * initial_capital

    # Output dataframe 
    df_mapping = {
        "close": data["close"],
        "asset_return": asset_return,
        "target_position": target_positions,
        "executed_position": executed_position,
        "gross_return": gross_return,
        "turnover": turnover, 
        "transaction_cost": transaction_cost, 
        "net_return": net_return,
        "equity_curve": equity_curve,
    }
    
    return pd.DataFrame(df_mapping, index=data.index)


def run_buy_and_hold_backtest(
    data: pd.DataFrame,
    cost_rate: float = 0.0,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Run an all-long benchmark through the standard backtest engine."""
    target_positions = pd.Series(
        1.0,
        index=data.index,
        dtype=float,
        name="target_position",
    )
    return run_backtest(
        data,
        target_positions,
        cost_rate=cost_rate,
        initial_capital=initial_capital,
    )


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Full audit trail produced by :func:`run_portfolio_backtest`."""

    performance: pd.DataFrame
    target_weights: pd.DataFrame
    executed_weights: pd.DataFrame
    asset_returns: pd.DataFrame
    asset_turnover: pd.DataFrame
    asset_transaction_costs: pd.DataFrame


def run_portfolio_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_rate: float = 0.0,
    asset_cost_rates: Mapping[str, float] | pd.Series | None = None,
    initial_capital: float = 1.0,
) -> PortfolioBacktestResult:
    """Run a multi-asset, close-to-close portfolio backtest.

    Target weights formed with information at bar ``t`` are executed for the
    return ending at ``t + 1``.  Turnover and proportional costs are calculated
    per asset, then summed into a portfolio-level audit table.
    """
    _validate_portfolio_inputs(prices, target_weights, cost_rate, initial_capital)
    cost_rates = _resolve_asset_cost_rates(prices.columns, cost_rate, asset_cost_rates)

    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    executed_weights = target_weights.shift(1).fillna(0.0)
    asset_turnover = executed_weights.sub(
        executed_weights.shift(1, fill_value=0.0)
    ).abs()
    asset_transaction_costs = asset_turnover.mul(cost_rates, axis=1)

    gross_return = (executed_weights * asset_returns).sum(axis=1)
    turnover = asset_turnover.sum(axis=1)
    transaction_cost = asset_transaction_costs.sum(axis=1)
    net_return = gross_return - transaction_cost
    equity_curve = (1.0 + net_return).cumprod() * initial_capital
    performance = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
        },
        index=prices.index,
    )
    return PortfolioBacktestResult(
        performance=performance,
        target_weights=target_weights.copy(),
        executed_weights=executed_weights,
        asset_returns=asset_returns,
        asset_turnover=asset_turnover,
        asset_transaction_costs=asset_transaction_costs,
    )


def run_equal_weight_buy_and_hold(
    prices: pd.DataFrame,
    cost_rate: float = 0.0,
    asset_cost_rates: Mapping[str, float] | pd.Series | None = None,
    initial_capital: float = 1.0,
) -> PortfolioBacktestResult:
    """Run a long-only, equal-weight buy-and-hold benchmark."""
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("prices must contain at least one asset")
    target_weights = pd.DataFrame(
        1.0 / prices.shape[1], index=prices.index, columns=prices.columns
    )
    return run_portfolio_backtest(
        prices,
        target_weights,
        cost_rate=cost_rate,
        asset_cost_rates=asset_cost_rates,
        initial_capital=initial_capital,
    )


def _validate_portfolio_inputs(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_rate: float,
    initial_capital: float,
) -> None:
    if prices.empty or target_weights.empty:
        raise ValueError("prices and target_weights must not be empty")
    if not prices.index.equals(target_weights.index) or not prices.columns.equals(target_weights.columns):
        raise ValueError("prices and target_weights must have matching index and columns")
    if prices.isna().any().any() or target_weights.isna().any().any():
        raise ValueError("prices and target_weights must not contain missing values")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("prices must be finite")
    if not np.isfinite(target_weights.to_numpy(dtype=float)).all():
        raise ValueError("target_weights must be finite")
    if (prices <= 0.0).any().any():
        raise ValueError("prices must be positive")
    if cost_rate < 0.0:
        raise ValueError("cost_rate must be non-negative")
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive")


def _resolve_asset_cost_rates(
    columns: pd.Index,
    cost_rate: float,
    asset_cost_rates: Mapping[str, float] | pd.Series | None,
) -> pd.Series:
    if asset_cost_rates is None:
        return pd.Series(cost_rate, index=columns, dtype=float)
    rates = pd.Series(asset_cost_rates, dtype=float)
    if set(rates.index) != set(columns):
        raise ValueError("asset_cost_rates must contain exactly the price tickers")
    rates = rates.reindex(columns)
    if rates.isna().any() or not np.isfinite(rates.to_numpy()).all() or (rates < 0.0).any():
        raise ValueError("asset_cost_rates must be non-negative and finite")
    return rates
