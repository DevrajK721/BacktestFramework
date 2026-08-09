import numpy as np 
import pandas as pd 
import pytest

from backtester.engine import run_backtest 

def close_data(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range(
                "2024-01-02", 
                periods=len(closes),
                freq="D",
                name="date",
            )
    return pd.DataFrame({"close": closes}, index=index)

def positions(values: list[float], index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
                values, 
                index=index, 
                dtype=float,
                name="target_position",
            )

def test_always_long_earns_asset_returns_after_first_date() -> None:
    data = close_data([100.0, 110.0, 99.0]) 
    targets = positions([1.0, 1.0, 1.0], data.index)

    result = run_backtest(data, targets)

    np.testing.assert_allclose(
            result["executed_position"],
            [0.0, 1.0, 1.0],
        )
    np.testing.assert_allclose(
            result["gross_return"],
            [0.0, 0.1, -0.1],
        )
    np.testing.assert_allclose(
            result["asset_return"],
            [0.0, 0.1, -0.1]
        )

def test_signal_does_not_earn_same_day_return() -> None:
    data = close_data([100.0, 110.0, 121.0])
    targets = positions([0.0, 1.0, 1.0], data.index)

    result = run_backtest(data, targets)

    np.testing.assert_allclose(
        result["executed_position"],
        [0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(
        result["gross_return"],
        [0.0, 0.0, 0.1],
    )

def test_always_flat_has_zero_gross_returns() -> None:
    data = close_data([100.0, 110.0, 99.0])
    targets = positions([0.0, 0.0, 0.0], data.index)

    result = run_backtest(data, targets)

    np.testing.assert_allclose(
        result["gross_return"],
        [0.0, 0.0, 0.0],
    )

def test_always_short_inverts_asset_returns() -> None:
    data = close_data([100.0, 110.0, 99.0])
    targets = positions([-1.0, -1.0, -1.0], data.index)

    result = run_backtest(data, targets)

    np.testing.assert_allclose(
        result["gross_return"],
        [0.0, -0.1, 0.1],
    )

def test_constant_prices_have_zero_gross_returns() -> None:
    data = close_data([100.0, 100.0, 100.0])
    targets = positions([0.0, 1.0, -1.0], data.index)

    result = run_backtest(data, targets)

    np.testing.assert_allclose(
        result["gross_return"],
        [0.0, 0.0, 0.0],
    )


def test_missing_close_column_raises_value_error() -> None:
    data = close_data([100.0, 110.0]).drop(columns="close")
    targets = positions([0.0, 1.0], data.index)

    with pytest.raises(ValueError, match="close"):
        run_backtest(data, targets)


def test_mismatched_position_index_raises_value_error() -> None:
    data = close_data([100.0, 110.0])
    other_index = pd.date_range(
        "2024-02-01",
        periods=2,
        freq="D",
        name="date",
    )
    targets = positions([0.0, 1.0], other_index)

    with pytest.raises(ValueError, match="index"):
        run_backtest(data, targets)


def test_nan_target_position_raises_value_error() -> None:
    data = close_data([100.0, 110.0])
    targets = positions([0.0, float("nan")], data.index)

    with pytest.raises(ValueError, match="missing"):
        run_backtest(data, targets)


def test_out_of_range_target_position_raises_value_error() -> None:
    data = close_data([100.0, 110.0])
    targets = positions([0.0, 1.1], data.index)

    with pytest.raises(ValueError, match="range"):
        run_backtest(data, targets)

def test_cost_is_charged_when_position_is_executed() -> None:
    data = close_data([100.0, 110.0, 121.0])
    targets = positions([0.0, 1.0, 1.0], data.index)

    result = run_backtest(data, targets, cost_rate=0.001)

    np.testing.assert_allclose(
        result["turnover"],
        [0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(
        result["transaction_cost"],
        [0.0, 0.0, 0.001],
    )
    np.testing.assert_allclose(
        result["net_return"],
        [0.0, 0.0, 0.099],
    )

def test_long_to_short_reversal_has_two_units_of_turnover() -> None:
    data = close_data([100.0, 110.0, 99.0])
    targets = positions([1.0, -1.0, -1.0], data.index)

    result = run_backtest(data, targets, cost_rate=0.001)

    np.testing.assert_allclose(
        result["executed_position"],
        [0.0, 1.0, -1.0],
    )
    np.testing.assert_allclose(
        result["turnover"],
        [0.0, 1.0, 2.0],
    )
    np.testing.assert_allclose(
        result["transaction_cost"],
        [0.0, 0.001, 0.002],
    )
    np.testing.assert_allclose(
        result["net_return"],
        [0.0, 0.099, 0.098],
    )

def test_equity_curve_compounds_net_returns() -> None:
    data = close_data([100.0, 110.0, 99.0])
    targets = positions([1.0, 1.0, 1.0], data.index)

    result = run_backtest(data, targets)

    np.testing.assert_allclose(
        result["equity_curve"],
        [1.0, 1.1, 0.99],
    )