from pathlib import Path

import pandas as pd 
import yfinance as yf

REQ_COLS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLS = ("open", "high", "low", "close")
SUPPORTED_YFINANCE_INTERVALS = ("15m", "1h", "3h", "6h", "1d", "1wk", "1mo", "1y")
_YFINANCE_DOWNLOAD_INTERVALS = {
    "15m": "15m",
    "1h": "1h",
    "3h": "1h",
    "6h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
    "1y": "1mo",
}
_RESAMPLE_RULES = {"3h": "3h", "6h": "6h", "1y": "YE"}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load bar-based OHLCV data from a CSV file and validate its schema."""
    return validate_ohlcv(pd.read_csv(path))


def download_yfinance_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download, validate, and optionally save adjusted single-ticker OHLCV data.

    ``end`` is exclusive, following yfinance's download convention. Prices are
    adjusted for splits and dividends by passing ``auto_adjust=True``
    explicitly. Three-hour, six-hour, and yearly bars are resampled from
    yfinance's native one-hour or one-month bars respectively.
    """
    if not ticker.strip():
        raise ValueError("ticker must not be empty")
    if interval not in SUPPORTED_YFINANCE_INTERVALS:
        raise ValueError(
            f"interval must be one of {SUPPORTED_YFINANCE_INTERVALS}"
        )

    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("start must be earlier than end")

    downloaded = yf.download(
        tickers=ticker,
        start=start_timestamp,
        end=end_timestamp,
        interval=_YFINANCE_DOWNLOAD_INTERVALS[interval],
        auto_adjust=True,
        progress=False,
        group_by="column",
        multi_level_index=False,
    )
    if downloaded.empty:
        raise ValueError("yfinance returned no data")

    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)

    downloaded.columns = [
        str(column).lower().replace(" ", "_") for column in downloaded.columns
    ]
    required_price_columns = [*PRICE_COLS, "volume"]
    missing_columns = set(required_price_columns) - set(downloaded.columns)
    if missing_columns:
        raise ValueError(
            f"yfinance result is missing required columns: {sorted(missing_columns)}"
        )
    downloaded = downloaded.loc[:, required_price_columns]

    if interval in _RESAMPLE_RULES:
        downloaded = downloaded.resample(_RESAMPLE_RULES[interval]).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()

    raw_data = downloaded.rename_axis("date").reset_index()
    clean_data = validate_ohlcv(raw_data)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_csv(destination, index_label="date")

    return clean_data

def validate_ohlcv(data : pd.DataFrame) -> pd.DataFrame:
    df = data.copy() # Do not mutate object

    # Check every name in REQ_COL exists
    missing_columns = set(REQ_COLS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    # Convert date to datetime object
    df["date"] = pd.to_datetime(df["date"], errors="raise") 

    # Reject duplicate dates 
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates are not allowed")

    # Check for missing values
    if df.loc[:, REQ_COLS].isna().any().any():
        raise ValueError("missing values are not allowed")

    # OHLC Checks 
    if (df.loc[:, PRICE_COLS] <= 0).any().any():
        raise ValueError("prices must be positive")

    if (df["volume"] < 0).any():
        raise ValueError("volume must be non-negative")

    if ((df["high"] < df["open"]) | (df["high"] < df["close"])).any():
        raise ValueError("high must not be below open or close")

    if ((df["low"] > df["open"]) | (df["low"] > df["close"])).any():
        raise ValueError("low must not be above open or close")

    df = df.loc[:, list(REQ_COLS)]

    df = df.set_index("date").sort_index()
    
    return df
