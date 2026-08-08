# Test for market data validation behaviour 
import pandas as pd 
import pytest

from backtester.data import validate_ohlcv

def valid_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "open": [100.0, 102.0, 103.0], 
                "high": [103.0, 104.0, 105.0],
                "low": [99.0, 101.0, 102.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000, 1200, 900],
            }
        ) # Every test will start with good data (this data) then one thing will be changed to see which (if any) rules fail


def test_valid_ohlcv_returns_date_indexed_data() -> None:
    result = validate_ohlcv(valid_ohlcv())

    assert isinstance(result.index, pd.DatetimeIndex) # Checks that DataFrame index is a Pandas datetime index, not string or other type 
    assert result.index.name == "date" # Check name of date column is "date"
    assert result.index.is_monotonic_increasing # Checks date is strictly increasing 
    assert list(result.columns) == ["open", "high", "low", "close", "volume"] # Check correct OHLCV data loaded
    assert len(result) == 3 # Specific to example given 


def test_duplicate_dates_raise_value_error() -> None:
    data = valid_ohlcv()
    data.loc[1, "date"] = "2024-01-02"

    with pytest.raises(ValueError, match="Duplicate"):
        validate_ohlcv(data) # Should reject duplicated data 

def test_unsorted_dates_are_sorted() -> None:
    data = valid_ohlcv().iloc[[2, 0, 1]]

    result = validate_ohlcv(data)

    assert result.index.is_monotonic_increasing

def test_missing_required_column_raises_value_error() -> None:
    data = valid_ohlcv().drop(columns="volume")

    with pytest.raises(ValueError, match="missing required columns"):
        validate_ohlcv(data)

def test_missing_value_raises_value_error() -> None:
    data = valid_ohlcv()
    data.loc[0, "close"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_ohlcv(data)

def test_non_positive_price_raises_value_error() -> None:
    data = valid_ohlcv()
    data.loc[0, "close"] = 0.0

    with pytest.raises(ValueError, match="positive"):
        validate_ohlcv(data)

def test_negative_volume_raises_value_error() -> None:
    data = valid_ohlcv()
    data.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="volume"):
        validate_ohlcv(data)

def test_high_below_close_raises_value_error() -> None:
    data = valid_ohlcv()
    data.loc[0, "high"] = 101.0  # close is 102.0

    with pytest.raises(ValueError, match="high"):
        validate_ohlcv(data)

def test_low_above_open_raises_value_error() -> None:
    data = valid_ohlcv()
    data.loc[0, "low"] = 101.0  # open is 100.0

    with pytest.raises(ValueError, match="low"):
        validate_ohlcv(data)

def test_validation_does_not_mutate_input() -> None:
    data = valid_ohlcv()
    original = data.copy(deep=True)

    validate_ohlcv(data)

    pd.testing.assert_frame_equal(data, original)
