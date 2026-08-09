# Test for market data validation behaviour 
import numpy as np
import pandas as pd 
import pytest

from backtester.data import (
    download_yfinance_ohlcv,
    load_ohlcv_csv,
    validate_ohlcv,
)

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


def test_load_ohlcv_csv_validates_and_standardizes_data(tmp_path) -> None:
    path = tmp_path / "ohlcv.csv"
    data = valid_ohlcv().iloc[[2, 0, 1]]
    data.to_csv(path, index=False)

    result = load_ohlcv_csv(path)
    expected = validate_ohlcv(data)

    pd.testing.assert_frame_equal(result, expected)


def test_download_yfinance_ohlcv_validates_and_saves_data(
    monkeypatch,
    tmp_path,
) -> None:
    downloaded = valid_ohlcv().rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    downloaded = downloaded.set_index("Date")
    calls: dict[str, object] = {}

    def fake_download(**kwargs) -> pd.DataFrame:
        calls.update(kwargs)
        return downloaded

    monkeypatch.setattr("backtester.data.yf.download", fake_download)
    output_path = tmp_path / "spy.csv"

    result = download_yfinance_ohlcv(
        ticker="SPY",
        start="2024-01-02",
        end="2024-01-05",
        interval="1d",
        output_path=output_path,
    )

    assert calls["tickers"] == "SPY"
    assert calls["interval"] == "1d"
    assert calls["auto_adjust"] is True
    assert calls["multi_level_index"] is False
    assert output_path.exists()
    pd.testing.assert_frame_equal(result, validate_ohlcv(valid_ohlcv()))
    pd.testing.assert_frame_equal(load_ohlcv_csv(output_path), result)


def test_download_yfinance_ohlcv_rejects_unsupported_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        download_yfinance_ohlcv(
            ticker="SPY",
            start="2024-01-02",
            end="2024-01-05",
            interval="5m",
        )


def test_download_yfinance_ohlcv_resamples_three_hour_bars(monkeypatch) -> None:
    index = pd.date_range("2024-01-02 09:00", periods=4, freq="1h", name="Date")
    downloaded = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [101.0, 102.0, 103.0, 104.0],
            "Volume": [1, 2, 3, 4],
        },
        index=index,
    )
    calls: dict[str, object] = {}

    def fake_download(**kwargs) -> pd.DataFrame:
        calls.update(kwargs)
        return downloaded

    monkeypatch.setattr("backtester.data.yf.download", fake_download)

    result = download_yfinance_ohlcv(
        ticker="SPY",
        start="2024-01-02",
        end="2024-01-03",
        interval="3h",
    )

    assert calls["interval"] == "1h"
    np.testing.assert_allclose(result["open"], [100.0, 103.0])
    np.testing.assert_allclose(result["high"], [104.0, 105.0])
    np.testing.assert_allclose(result["low"], [99.0, 102.0])
    np.testing.assert_allclose(result["close"], [103.0, 104.0])
    np.testing.assert_allclose(result["volume"], [6.0, 4.0])


@pytest.mark.parametrize(
    ("requested_interval", "download_interval"),
    [("15m", "15m"), ("1h", "1h"), ("6h", "1h"), ("1y", "1mo")],
)
def test_download_yfinance_ohlcv_uses_expected_source_interval(
    monkeypatch,
    requested_interval,
    download_interval,
) -> None:
    downloaded = valid_ohlcv().rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).set_index("Date")
    downloaded.index = pd.DatetimeIndex(downloaded.index, name="Date")
    calls: dict[str, object] = {}

    def fake_download(**kwargs) -> pd.DataFrame:
        calls.update(kwargs)
        return downloaded

    monkeypatch.setattr("backtester.data.yf.download", fake_download)

    download_yfinance_ohlcv(
        ticker="SPY",
        start="2024-01-02",
        end="2024-01-05",
        interval=requested_interval,
    )

    assert calls["interval"] == download_interval
