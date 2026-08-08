import pandas as pd 

REQ_COLS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLS = ("open", "high", "low", "close")

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


