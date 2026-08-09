import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes 

def plot_equity_curves(strategy_equity: pd.Series, benchmark_equity: pd.Series, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(strategy_equity.index, strategy_equity, label="Strategy")
    ax.plot(benchmark_equity.index, benchmark_equity, label="Buy and Hold")

    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.4)

    return ax 

def plot_drawdown(drawdown: pd.Series, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    drawdown_percent = drawdown * 100.0
    ax.plot(drawdown_percent.index, drawdown_percent, label="Drawdown")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.fill_between(
        drawdown_percent.index,
        drawdown_percent,
        0.0,
        alpha=0.2,
    )
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)

    return ax


def plot_portfolio_weights(weights: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot signed target portfolio weights through time."""
    if weights.empty:
        raise ValueError("weights must not be empty")
    if ax is None:
        _, ax = plt.subplots()
    weights.plot.area(ax=ax, stacked=True, linewidth=0.0, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Target Portfolio Weights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.25)
    ax.legend(title="Asset", loc="best")
    return ax


def plot_turnover(turnover: pd.Series, ax: Axes | None = None) -> Axes:
    """Plot portfolio turnover charged on each execution date."""
    if turnover.empty:
        raise ValueError("turnover must not be empty")
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(turnover.index, turnover, width=1.0, color="#4c78a8", alpha=0.8)
    ax.set_title("Portfolio Turnover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(axis="y", alpha=0.25)
    return ax
