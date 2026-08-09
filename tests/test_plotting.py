import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from backtester.plotting import plot_drawdown, plot_equity_curves


def date_series(values: list[float], name: str) -> pd.Series:
    index = pd.date_range(
        "2024-01-02",
        periods=len(values),
        freq="D",
        name="date",
    )
    return pd.Series(values, index=index, dtype=float, name=name)


def test_plot_equity_curves_draws_strategy_and_benchmark() -> None:
    strategy = date_series([1.0, 1.1, 1.05], "equity_curve")
    benchmark = date_series([1.0, 1.05, 1.1], "equity_curve")

    ax = plot_equity_curves(strategy, benchmark)

    assert isinstance(ax, Axes)
    assert ax.get_title() == "Equity Curve"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Equity"
    assert [line.get_label() for line in ax.get_lines()] == [
        "Strategy",
        "Buy and Hold",
    ]
    assert ax.get_legend() is not None

    plt.close(ax.figure)


def test_plot_drawdown_draws_series_and_zero_reference_line() -> None:
    drawdown = date_series([0.0, -0.1, -0.05], "drawdown")

    ax = plot_drawdown(drawdown)

    assert isinstance(ax, Axes)
    assert ax.get_title() == "Drawdown"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Drawdown (%)"
    assert ax.get_lines()[0].get_label() == "Drawdown"
    assert len(ax.get_lines()) == 2
    assert len(ax.collections) == 1

    plt.close(ax.figure)


def test_plot_functions_use_a_supplied_axis() -> None:
    strategy = date_series([1.0, 1.1], "equity_curve")
    benchmark = date_series([1.0, 1.05], "equity_curve")
    drawdown = date_series([0.0, -0.1], "drawdown")
    figure, axes = plt.subplots(nrows=2)

    assert plot_equity_curves(strategy, benchmark, ax=axes[0]) is axes[0]
    assert plot_drawdown(drawdown, ax=axes[1]) is axes[1]

    plt.close(figure)
