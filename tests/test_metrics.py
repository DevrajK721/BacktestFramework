import pandas as pd
import pytest

from backtester.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_total_return,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_downside_deviation,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_hit_rate,
    calculate_average_positive_return,
    calculate_average_negative_return,
    calculate_trade_count,
    calculate_total_turnover,
)

def equity_curve(values: list[float]) -> pd.Series:
    index = pd.date_range(
        "2024-01-02",
        periods=len(values),
        freq="D",
        name="date",
    )
    return pd.Series(values, index=index, dtype=float, name="equity_curve")


def test_total_return() -> None:
    result = calculate_total_return(equity_curve([100.0, 110.0, 99.0]))

    assert result == pytest.approx(-0.01)


def test_cagr_uses_number_of_return_intervals() -> None:
    result = calculate_cagr(
        equity_curve([100.0, 110.0, 121.0]),
        periods_per_year=2,
    )

    assert result == pytest.approx(0.21)


def test_annualized_volatility_uses_sample_standard_deviation() -> None:
    returns = pd.Series([0.01, -0.01], dtype=float)

    result = calculate_annualized_volatility(
        returns,
        periods_per_year=2,
    )

    assert result == pytest.approx(0.02)


def test_cagr_requires_at_least_two_equity_observations() -> None:
    with pytest.raises(ValueError, match="two"):
        calculate_cagr(equity_curve([100.0]))


def test_volatility_requires_at_least_two_returns() -> None:
    with pytest.raises(ValueError, match="two"):
        calculate_annualized_volatility(pd.Series([0.01]))

def test_drawdown_correctly_calculated() -> None:
    equity_curve = pd.Series([100, 120, 90, 110])
    drawdown = calculate_drawdown(equity_curve)
    expected_drawdown = pd.Series([0.0, 0.0, -0.25, -0.08333333333333333], name="drawdown")

    pd.testing.assert_series_equal(drawdown, expected_drawdown)

    max_drawdown = calculate_max_drawdown(equity_curve)
    assert max_drawdown == pytest.approx(-0.25)

def test_rising_equity_has_no_drawdown() -> None:
    curve = equity_curve([100.0, 110.0, 120.0])

    result = calculate_drawdown(curve)

    pd.testing.assert_series_equal(
        result,
        pd.Series([0.0, 0.0, 0.0], index=curve.index, name="drawdown"),
    )
    assert calculate_max_drawdown(curve) == 0.0

def test_empty_equity_curve_raises_value_error_for_drawdown() -> None:
    empty_curve = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="empty"):
        calculate_drawdown(empty_curve)

    with pytest.raises(ValueError, match="empty"):
        calculate_max_drawdown(empty_curve)

def test_sharpe_ratio_calculation() -> None:
    returns = pd.Series([0.15, 0.25])
    risk_free_rate = 0.21
    periods_per_year = 2

    result = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

    assert result == pytest.approx(2.0)

def test_sharpe_ratio_raises_when_volatility_is_zero() -> None:
    returns = pd.Series([0.01, 0.01])

    with pytest.raises(ValueError, match="zero"):
        calculate_sharpe_ratio(returns)


def test_downside_deviation_uses_all_return_periods() -> None:
    returns = pd.Series([0.01, -0.01])

    result = calculate_downside_deviation(returns, periods_per_year=2)

    assert result == pytest.approx(0.01)


def test_downside_deviation_is_zero_without_negative_returns() -> None:
    returns = pd.Series([0.01, 0.02])

    result = calculate_downside_deviation(returns)

    assert result == 0.0


def test_downside_deviation_requires_at_least_two_returns() -> None:
    with pytest.raises(ValueError, match="two"):
        calculate_downside_deviation(pd.Series([0.01]))


def test_downside_deviation_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="empty"):
        calculate_downside_deviation(pd.Series(dtype=float))


def test_sortino_ratio_uses_compounded_risk_free_rate() -> None:
    returns = pd.Series([0.05, 0.25])

    result = calculate_sortino_ratio(
        returns,
        risk_free_rate=0.21,
        periods_per_year=2,
    )

    assert result == pytest.approx(2.0)


def test_sortino_ratio_raises_when_downside_deviation_is_zero() -> None:
    returns = pd.Series([0.15, 0.25])

    with pytest.raises(ValueError, match="zero"):
        calculate_sortino_ratio(
            returns,
            risk_free_rate=0.21,
            periods_per_year=2,
        )


def test_calmar_ratio() -> None:
    curve = equity_curve([100.0, 150.0, 120.0])

    result = calculate_calmar_ratio(curve, periods_per_year=2)

    assert result == pytest.approx(1.0)


def test_calmar_ratio_raises_without_drawdown() -> None:
    curve = equity_curve([100.0, 110.0, 120.0])

    with pytest.raises(ValueError, match="drawdown"):
        calculate_calmar_ratio(curve, periods_per_year=2)


def test_return_diagnostics_exclude_flat_periods_from_hit_rate() -> None:
    returns = pd.Series([0.0, 0.01, -0.02, 0.03])

    assert calculate_hit_rate(returns) == pytest.approx(2.0 / 3.0)
    assert calculate_average_positive_return(returns) == pytest.approx(0.02)
    assert calculate_average_negative_return(returns) == pytest.approx(-0.02)


def test_hit_rate_requires_at_least_one_non_zero_return() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        calculate_hit_rate(pd.Series([0.0, 0.0]))


def test_turnover_diagnostics_distinguish_events_from_units() -> None:
    turnover = pd.Series([0.0, 1.0, 0.0, 2.0])

    assert calculate_trade_count(turnover) == 2
    assert calculate_total_turnover(turnover) == pytest.approx(3.0)
