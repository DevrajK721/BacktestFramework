# Exploratory Data Analysis for Time Series Momentum
# Note to self: Do in jupyter notebook next time. 
import json
import os

import matplotlib

# This research script writes figures to disk and does not require a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from backtester.portfolio_data import close_price_panel, load_ohlcv_universe


# Create a coverage and quality table
COVERAGE_TABLE_PATH = "research/TimeSeriesMomentum/data/coverage_quality.csv"
PROCESSED_DATA_DIRECTORY = "research/TimeSeriesMomentum/data/processed"
FIGURES_DIRECTORY = "research/TimeSeriesMomentum/report/figures"


def main() -> None:
    os.makedirs(PROCESSED_DATA_DIRECTORY, exist_ok=True)
    os.makedirs(FIGURES_DIRECTORY, exist_ok=True)

    coverage_table = pd.DataFrame(
        columns=[
            "Ticker",
            "First Date",
            "Last Date",
            "No. of Observations",
            "No. of duplicate observations",
            "No. of missing observations",
            "Minimum Adjusted Close",
            "Maximum Adjusted Close",
            "Largest Absolute Daily Return",
        ]
    )
    with open("research/TimeSeriesMomentum/data/metadata.json", encoding="utf-8") as metadata_file:
        tickers = json.load(metadata_file)["Universe Tickers"]
    ticker_map = dict(
        zip(
            tickers,
            [f"research/TimeSeriesMomentum/data/raw/{ticker}.csv" for ticker in tickers],
            strict=True,
        )
    )
    universe = load_ohlcv_universe(ticker_map)
    prices = close_price_panel(universe)
    returns = prices.pct_change(fill_method=None)

    for ticker in tickers:
        first_date = prices[ticker].first_valid_index().date()
        last_date = prices[ticker].last_valid_index().date()
        num_observations = prices[ticker].count()
        num_duplicates = universe[ticker].index.duplicated().sum()
        num_missing = prices[ticker].isna().sum()
        min_adj_close = prices[ticker].min()
        max_adj_close = prices[ticker].max()
        largest_abs_daily_return = returns[ticker].abs().max()

        coverage_table.loc[len(coverage_table)] = [
            ticker,
            first_date,
            last_date,
            num_observations,
            num_duplicates,
            num_missing,
            min_adj_close,
            max_adj_close,
            largest_abs_daily_return,
        ]

    coverage_table.to_csv(COVERAGE_TABLE_PATH, index=False)
    print(f"Coverage and quality table saved to {COVERAGE_TABLE_PATH}.")

    # Compute raw asset returns and save one aligned processed return panel.
    returns.to_csv(f"{PROCESSED_DATA_DIRECTORY}/returns.csv", index=True)

    # Compute asset summaries and save to CSV
    asset_summaries = pd.DataFrame(
        columns=[
            "Ticker",
            "Mean Daily Return",
            "Annualised Volatility",
            "Minimum Daily Return",
            "Maximum Daily Return",
            "Skewness of Daily Returns",
            "Percentage of Positive Daily Returns",
        ]
    )
    for ticker in tickers:
        daily_returns = returns[ticker].dropna()
        mean_daily_return = daily_returns.mean()
        annualised_volatility = daily_returns.std() * (252 ** 0.5)
        min_daily_return = daily_returns.min()
        max_daily_return = daily_returns.max()
        skewness_daily_return = daily_returns.skew()
        percentage_positive_daily_returns = (daily_returns > 0).mean() * 100

        asset_summaries.loc[len(asset_summaries)] = [
            ticker,
            mean_daily_return,
            annualised_volatility,
            min_daily_return,
            max_daily_return,
            skewness_daily_return,
            percentage_positive_daily_returns,
        ]

    asset_summaries.to_csv("research/TimeSeriesMomentum/data/data_summary.csv", index=False)

    # Investigate extreme returns
    extremes = pd.DataFrame(columns=["Ticker", "Date", "Return", "Absolute Return"])
    for ticker in tickers:
        # Find the five largest absolute daily returns for each asset.
        largest_return_dates = returns[ticker].abs().nlargest(5).index
        largest_returns = returns.loc[largest_return_dates, ticker]
        for date, return_value in largest_returns.items():
            extremes.loc[len(extremes)] = [
                ticker,
                date.date(),
                return_value,
                abs(return_value),
            ]

    extremes.to_csv("research/TimeSeriesMomentum/data/return_extremes.csv", index=False)

    # Analyse relationships across assets (Pearson correlation matrix and heatmap).
    correlation_matrix = returns.corr()
    correlation_matrix.to_csv("research/TimeSeriesMomentum/data/correlation_matrix.csv", index=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(tickers)), labels=tickers, rotation=45, ha="right")
    ax.set_yticks(range(len(tickers)), labels=tickers)
    for row, ticker_row in enumerate(tickers):
        for column, ticker_column in enumerate(tickers):
            ax.text(
                column,
                row,
                f"{correlation_matrix.loc[ticker_row, ticker_column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Pearson Correlation Matrix of Daily Asset Returns")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIRECTORY}/correlation_matrix_heatmap.svg")
    plt.close(fig)

    # Analyse risk and drawdowns (annualised volatility and buy-and-hold drawdowns).
    volatility = pd.DataFrame(
        {
            "Ticker": tickers,
            "Annualised Volatility": [returns[ticker].std() * (252 ** 0.5) for ticker in tickers],
        }
    )
    normalised_prices = prices.div(prices.iloc[0])
    drawdowns = normalised_prices.div(normalised_prices.cummax()).sub(1.0)

    # Volatility plot
    fig, ax = plt.subplots(figsize=(10, 6))
    volatility.set_index("Ticker").plot(kind="bar", ax=ax, legend=False, color="#007C83")
    ax.set_title("Annualised Volatility of Daily Asset Returns")
    ax.set_ylabel("Annualised volatility")
    ax.set_xlabel("Ticker")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIRECTORY}/volatility_bar_chart.svg")
    plt.close(fig)

    # Drawdown plot
    fig, ax = plt.subplots(figsize=(10, 6))
    drawdowns.plot(ax=ax, linewidth=1.1)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Buy-and-Hold Drawdowns by Asset")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIRECTORY}/drawdown_line_chart.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
