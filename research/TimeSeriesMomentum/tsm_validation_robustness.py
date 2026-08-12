"""Validation-period cost sensitivity and rebalancing robustness checks.

Cost sensitivity is evaluated across the full pre-specified lookback family.
Rebalancing robustness is intentionally opt-in and may be run only after a
lookback has been selected and documented from the preceding validation stage.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.engine import run_portfolio_backtest
from backtester.portfolio import FixedWeightPortfolio, PortfolioConfig, build_target_weights
from backtester.portfolio_data import close_price_panel, load_ohlcv_universe


VALIDATION_RANGE = ("2015-07-01", "2019-06-30")
LOOKBACKS = [7, 21, 63, 126, 189, 252]
COST_BPS = [0, 1, 2, 3, 5, 10, 20]
REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly"]
BASE_COST_BPS = 5
INITIAL_CAPITAL = 100_000

# Leave this disabled until the cost-sensitivity evidence supports a documented
# lookback decision.  Do not use the test period to make this decision.
RUN_REBALANCE_ROBUSTNESS = False
SELECTED_LOOKBACK: int | None = None

DATA_DIRECTORY = Path("research/TimeSeriesMomentum/data/raw")
RESULTS_DIRECTORY = Path("research/TimeSeriesMomentum/results/validation")
COST_RESULTS_DIRECTORY = RESULTS_DIRECTORY / "cost_sensitivity"
REBALANCE_RESULTS_DIRECTORY = RESULTS_DIRECTORY / "rebalance_robustness"


def performance_summary(
    performance: pd.DataFrame,
    lookback: int,
    cost_bps: int,
    rebalance_frequency: str,
) -> dict[str, float | int | str]:
    net_returns = performance["net_return"]
    equity_curve = performance["equity_curve"]
    years = len(net_returns) / 252
    drawdown = equity_curve.div(equity_curve.cummax()).sub(1.0)
    return_volatility = net_returns.std(ddof=1)

    return {
        "Lookback": lookback,
        "Cost (bps)": cost_bps,
        "Rebalance Frequency": rebalance_frequency,
        "Start": performance.index[0].date().isoformat(),
        "End": performance.index[-1].date().isoformat(),
        "Total Return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0,
        "CAGR": (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0,
        "Annualised Volatility": return_volatility * np.sqrt(252),
        "Sharpe Ratio (Rf = 0%)": net_returns.mean() / return_volatility * np.sqrt(252),
        "Maximum Drawdown": drawdown.min(),
        "Total Turnover": performance["turnover"].sum(),
        "Cumulative Transaction Cost": performance["transaction_cost"].sum(),
        "Trade Count": int((performance["turnover"] > 0.0).sum()),
    }


def save_result(result, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result.performance.to_csv(directory / f"{stem}_performance.csv")
    result.target_weights.to_csv(directory / f"{stem}_target_weights.csv")
    result.executed_weights.to_csv(directory / f"{stem}_executed_weights.csv")
    result.asset_turnover.to_csv(directory / f"{stem}_asset_turnover.csv")


def run_cost_sensitivity(
    prices: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    validation_prices = prices.loc[VALIDATION_RANGE[0] : VALIDATION_RANGE[1]]
    portfolio_constructor = FixedWeightPortfolio({ticker: 0.10 for ticker in tickers})
    portfolio_config = PortfolioConfig(rebalance_frequency="monthly")
    summaries = []

    for lookback in LOOKBACKS:
        signals = np.sign(prices.pct_change(lookback)).fillna(0.0)
        target_weights = build_target_weights(
            signals=signals,
            prices=prices,
            constructor=portfolio_constructor,
            config=portfolio_config,
        ).loc[validation_prices.index]

        for cost_bps in COST_BPS:
            result = run_portfolio_backtest(
                prices=validation_prices,
                target_weights=target_weights,
                cost_rate=cost_bps / 10_000,
                initial_capital=INITIAL_CAPITAL,
            )
            stem = f"validation_lb{lookback}_monthly_cost{cost_bps}bps"
            save_result(result, COST_RESULTS_DIRECTORY, stem)
            summaries.append(
                performance_summary(
                    result.performance,
                    lookback=lookback,
                    cost_bps=cost_bps,
                    rebalance_frequency="monthly",
                )
            )

    summary = pd.DataFrame(summaries)
    COST_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(COST_RESULTS_DIRECTORY / "cost_sensitivity_summary.csv", index=False)
    return summary


def run_rebalance_robustness(
    prices: pd.DataFrame,
    tickers: list[str],
    selected_lookback: int,
) -> pd.DataFrame:
    if selected_lookback not in LOOKBACKS:
        raise ValueError("selected_lookback must be one of the pre-specified lookbacks")

    validation_prices = prices.loc[VALIDATION_RANGE[0] : VALIDATION_RANGE[1]]
    signals = np.sign(prices.pct_change(selected_lookback)).fillna(0.0)
    portfolio_constructor = FixedWeightPortfolio({ticker: 0.10 for ticker in tickers})
    summaries = []

    for rebalance_frequency in REBALANCE_FREQUENCIES:
        target_weights = build_target_weights(
            signals=signals,
            prices=prices,
            constructor=portfolio_constructor,
            config=PortfolioConfig(rebalance_frequency=rebalance_frequency),
        ).loc[validation_prices.index]
        result = run_portfolio_backtest(
            prices=validation_prices,
            target_weights=target_weights,
            cost_rate=BASE_COST_BPS / 10_000,
            initial_capital=INITIAL_CAPITAL,
        )
        stem = f"validation_lb{selected_lookback}_{rebalance_frequency}_cost{BASE_COST_BPS}bps"
        save_result(result, REBALANCE_RESULTS_DIRECTORY, stem)
        summaries.append(
            performance_summary(
                result.performance,
                lookback=selected_lookback,
                cost_bps=BASE_COST_BPS,
                rebalance_frequency=rebalance_frequency,
            )
        )

    summary = pd.DataFrame(summaries)
    REBALANCE_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REBALANCE_RESULTS_DIRECTORY / "rebalance_robustness_summary.csv", index=False)
    return summary


def main() -> None:
    tickers = json.loads(
        (Path("research/TimeSeriesMomentum/data/metadata.json")).read_text()
    )["Universe Tickers"]
    ticker_map = {ticker: DATA_DIRECTORY / f"{ticker}.csv" for ticker in tickers}
    prices = close_price_panel(load_ohlcv_universe(ticker_map))

    cost_summary = run_cost_sensitivity(prices, tickers)
    print("\nCost sensitivity summary")
    print(cost_summary.to_string(index=False))

    if not RUN_REBALANCE_ROBUSTNESS:
        print("\nRebalancing robustness not run: select and document a lookback first.")
        return
    if SELECTED_LOOKBACK is None:
        raise ValueError("Set SELECTED_LOOKBACK before running rebalancing robustness")

    rebalance_summary = run_rebalance_robustness(
        prices,
        tickers,
        selected_lookback=SELECTED_LOOKBACK,
    )
    print("\nRebalancing robustness summary")
    print(rebalance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
