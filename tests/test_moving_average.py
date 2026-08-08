import pandas as pd 
import pytest 

from backtester.strategies.moving_average import MovingAverageCrossover 

def close_data(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range(
            "2024-01-03",
            periods=len(closes),
            freq="D",
            name="date",
        )

    return pd.DataFrame({"close": closes}, index=index)

def test_rising_prices_generate_long_position_after_warmup() -> None:
    data = close_data([1.0, 2.0, 3.0, 4.0, 5.0])
    strategy = MovingAverageCrossover(fast_window=2, slow_window=3)

    result = strategy.generate_positions(data)

    expected = pd.Series(
            [0.0, 0.0, 1.0, 1.0, 1.0],
            index=data.index,
            name="target_position",
        )

    pd.testing.assert_series_equal(result, expected)

def test_equal_moving_averages_produce_flat_position() -> None:
    data = close_data([1.0, 1.0, 1.0, 1.0])
    strategy = MovingAverageCrossover(fast_window=2, slow_window=3)

    result = strategy.generate_positions(data)

    assert (result == 0.0).all()

def test_invalid_window_order_raises_value_error() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=3, slow_window=3)


