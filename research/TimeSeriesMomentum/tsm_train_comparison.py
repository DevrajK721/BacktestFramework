import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load baseline and benchmark results
baseline = pd.read_csv(
    "research/TimeSeriesMomentum/results/baseline/train_performance.csv",
    index_col=0,
    parse_dates=True,
)
spy_long = pd.read_csv(
    "research/TimeSeriesMomentum/results/benchmarks/s1_train_performance.csv",
    index_col=0,
    parse_dates=True,
)
equal_long = pd.read_csv(
    "research/TimeSeriesMomentum/results/benchmarks/equal_weight_long_only_train_performance.csv",
    index_col=0,
    parse_dates=True,
)

with open("research/TimeSeriesMomentum/data/metadata.json", encoding="utf-8") as file:
    tickers = json.load(file)["Universe Tickers"]

individual_results = {
    ticker: pd.read_csv(
        f"research/TimeSeriesMomentum/results/benchmarks/{ticker}_train_performance.csv",
        index_col=0,
        parse_dates=True,
    )
    for ticker in tickers
}

def summary_row(name, performance, total_turnover=None, cumulative_cost=None):
    net_returns = performance["net_return"]
    equity_curve = performance["equity_curve"]
    years = len(net_returns) / 252

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0
    annualised_volatility = net_returns.std(ddof=1) * np.sqrt(252)
    sharpe = (
        net_returns.mean() / net_returns.std(ddof=1) * np.sqrt(252)
        if net_returns.std(ddof=1) > 0.0
        else np.nan
    )
    drawdown = equity_curve.div(equity_curve.cummax()).sub(1.0)

    return {
        "Portfolio": name,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": annualised_volatility,
        "Sharpe Ratio (Rf = 0%)": sharpe,
        "Maximum Drawdown": drawdown.min(),
        "Total Turnover": (
            performance["turnover"].sum()
            if total_turnover is None
            else total_turnover
        ),
        "Cumulative Transaction Cost": (
            performance["transaction_cost"].sum()
            if cumulative_cost is None
            else cumulative_cost
        ),
    }

results_directory = Path(
    "research/TimeSeriesMomentum/results/train_comparison"
)
results_directory.mkdir(parents=True, exist_ok=True)

comparison_table = pd.DataFrame(
    [
        summary_row("TSM baseline", baseline),
        summary_row(
            "Equal-weight long-only",
            equal_long,
            total_turnover=1.0,
            cumulative_cost=5 / 10_000,
        ),
        summary_row("SPY buy-and-hold", spy_long),
    ]
)

comparison_table.to_csv(
    results_directory / "train_performance_comparison.csv",
    index=False,
)

individual_table = pd.DataFrame(
    [
        summary_row(ticker, performance)
        for ticker, performance in individual_results.items()
    ]
).drop(columns=["Total Turnover", "Cumulative Transaction Cost"])

individual_table.to_csv(
    results_directory / "individual_buy_and_hold_comparison.csv",
    index=False,
)

equity_curves = pd.DataFrame(
    {
        "TSM baseline": baseline["equity_curve"],
        "Equal-weight long-only": equal_long["equity_curve"],
        "SPY buy-and-hold": spy_long["equity_curve"],
    }
)

fig, ax = plt.subplots(figsize=(10, 6))
equity_curves.plot(ax=ax, linewidth=1.2)
ax.set_title("Development-Period Net Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio value")
ax.legend()
fig.tight_layout()

figures_directory = Path("research/TimeSeriesMomentum/report/figures")
figures_directory.mkdir(parents=True, exist_ok=True)
fig.savefig(figures_directory / "train_equity_curves.svg")
plt.close(fig)
