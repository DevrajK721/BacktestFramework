# Data acquisition script for Time Series Momentum project

# Assets to be loaded
"""
1. SPY
2. IEF
3. TLT
4. EFA
5. EEM
6. LQD
7. GLD
8. DBC
9. UUP
10. VNQ
"""
# Date Range: 2007-07-01 to 2026-07-01 INCLUSIVE (Add extra 1 day to account for yfinance)

from pathlib import Path

from backtester.portfolio_data import (
    close_price_panel,
    ensure_yfinance_ohlcv_csvs,
    load_ohlcv_universe,
)


TICKERS = ("SPY", "IEF", "TLT", "EFA", "EEM", "LQD", "GLD", "DBC", "UUP", "VNQ")
START_DATE = "2007-06-30"  # yfinance treats the start date as inclusive, so this includes 2007-07-01.
# yfinance treats the end date as exclusive, so this includes 2026-07-01.
END_DATE = "2026-07-02"
INTERVAL = "1d"
DATA_DIRECTORY = Path("research/TimeSeriesMomentum/data/raw")


def main() -> None:
    """Download missing CSVs, then validate and align the full universe."""
    csv_paths = ensure_yfinance_ohlcv_csvs(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        output_directory=DATA_DIRECTORY,
    )
    universe = load_ohlcv_universe(csv_paths)
    prices = close_price_panel(universe)

    print(f"Validated {len(TICKERS)} assets on {len(prices)} common trading dates.")
    print(f"Aligned sample: {prices.index.min().date()} through {prices.index.max().date()}")


if __name__ == "__main__":
    main()
