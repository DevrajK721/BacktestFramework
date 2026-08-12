import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


LOOKBACKS = [7, 21, 63, 126, 189, 252]
COST_BPS = [0, 1, 2, 3, 5, 10, 20]
VALIDATION_RESULTS_DIRECTORY = Path("research/TimeSeriesMomentum/results/validation")
COST_SENSITIVITY_DIRECTORY = VALIDATION_RESULTS_DIRECTORY / "cost_sensitivity"
FIGURES_DIRECTORY = Path("research/TimeSeriesMomentum/report/figures")


def summary_row(lookback: int, performance: pd.DataFrame) -> dict[str, float | int]:
    net_returns = performance["net_return"]
    equity_curve = performance["equity_curve"]
    years = len(net_returns) / 252
    drawdown = equity_curve.div(equity_curve.cummax()).sub(1.0)

    return {
        "Lookback": lookback,
        "Total Return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0,
        "CAGR": (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0,
        "Annualised Volatility": net_returns.std(ddof=1) * np.sqrt(252),
        "Sharpe Ratio (Rf = 0%)": net_returns.mean() / net_returns.std(ddof=1) * np.sqrt(252),
        "Maximum Drawdown": drawdown.min(),
        "Total Turnover": performance["turnover"].sum(),
        "Cumulative Transaction Cost": performance["transaction_cost"].sum(),
    }


performances = {
    lookback: pd.read_csv(
        VALIDATION_RESULTS_DIRECTORY / f"validation_performance_lb{lookback}.csv",
        index_col=0,
        parse_dates=True,
    )
    for lookback in LOOKBACKS
}

comparison_table = pd.DataFrame(
    [summary_row(lookback, performance) for lookback, performance in performances.items()]
)
comparison_table.to_csv(
    VALIDATION_RESULTS_DIRECTORY / "validation_lookback_comparison.csv",
    index=False,
)

equity_curves = pd.DataFrame(
    {
        f"{lookback}-day": performance["equity_curve"]
        for lookback, performance in performances.items()
    }
)

FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
equity_curves.plot(ax=ax, linewidth=1.2)
ax.set_title("Validation-Period Net Equity Curves by Momentum Lookback")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio value")
ax.legend(title="Lookback")
fig.tight_layout()
fig.savefig(FIGURES_DIRECTORY / "validation_lookback_equity_curves.svg")
plt.close(fig)

print(comparison_table.to_string(index=False))

cost_sensitivity = pd.read_csv(
    COST_SENSITIVITY_DIRECTORY / "cost_sensitivity_summary.csv"
)
sharpe_by_cost = cost_sensitivity.pivot(
    index="Lookback",
    columns="Cost (bps)",
    values="Sharpe Ratio (Rf = 0%)",
).reindex(LOOKBACKS)
sharpe_by_cost.to_csv(
    COST_SENSITIVITY_DIRECTORY / "cost_sensitivity_sharpe_table.csv"
)

fig, ax = plt.subplots(figsize=(10, 6))
for lookback, values in sharpe_by_cost.iterrows():
    ax.plot(
        values.index,
        values,
        marker="o",
        linewidth=1.5,
        label=f"{lookback}-day",
    )
ax.axhline(0.0, color="#555555", linewidth=0.9)
ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, label="0.5 reference")
ax.set_title("Validation Sharpe Ratio Sensitivity to Transaction Costs")
ax.set_xlabel("One-way transaction cost (basis points)")
ax.set_ylabel("Annualised Sharpe ratio (Rf = 0%)")
ax.set_xticks(COST_BPS)
ax.legend(title="Lookback", ncol=2)
fig.tight_layout()
fig.savefig(FIGURES_DIRECTORY / "validation_cost_sensitivity_sharpe.svg")
plt.close(fig)
