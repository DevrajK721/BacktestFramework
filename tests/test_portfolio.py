import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_equal_weight_buy_and_hold, run_portfolio_backtest
from backtester.portfolio import (
    EqualWeightPortfolio,
    FixedWeightPortfolio,
    InverseVolatilityPortfolio,
    PortfolioConfig,
    build_target_weights,
)


def price_panel(periods: int = 4) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=periods, freq="D", name="date")
    return pd.DataFrame(
        {
            "AAA": 100.0 * (1.1 ** np.arange(periods)),
            "BBB": 100.0 * (0.9 ** np.arange(periods)),
        },
        index=index,
    )


def test_equal_weight_long_short_portfolio_uses_next_bar_weights() -> None:
    prices = price_panel(3)
    signals = pd.DataFrame({"AAA": 1.0, "BBB": -1.0}, index=prices.index)
    weights = build_target_weights(
        signals,
        prices,
        EqualWeightPortfolio(),
        PortfolioConfig(rebalance_frequency="daily"),
    )

    result = run_portfolio_backtest(prices, weights)

    np.testing.assert_allclose(weights.to_numpy(), [[0.5, -0.5]] * 3)
    np.testing.assert_allclose(result.executed_weights.iloc[0], [0.0, 0.0])
    np.testing.assert_allclose(result.performance["gross_return"], [0.0, 0.1, 0.1])


def test_fixed_weights_leave_inactive_allocations_as_cash() -> None:
    prices = price_panel(3)
    signals = pd.DataFrame(
        {"AAA": [1.0, 1.0, 0.0], "BBB": [1.0, 0.0, 0.0]},
        index=prices.index,
    )
    weights = build_target_weights(
        signals,
        prices,
        FixedWeightPortfolio({"AAA": 0.6, "BBB": 0.4}),
        PortfolioConfig(rebalance_frequency="daily"),
    )

    np.testing.assert_allclose(weights.to_numpy(), [[0.6, 0.4], [0.6, 0.0], [0.0, 0.0]])


def test_net_mode_allocates_long_and_short_legs_to_target_and_gross_limit() -> None:
    prices = price_panel(3)
    signals = pd.DataFrame({"AAA": 1.0, "BBB": -1.0}, index=prices.index)
    weights = build_target_weights(
        signals,
        prices,
        EqualWeightPortfolio(),
        PortfolioConfig(
            exposure_mode="net",
            target_net_exposure=0.5,
            gross_exposure_limit=1.0,
            rebalance_frequency="daily",
        ),
    )

    np.testing.assert_allclose(weights.sum(axis=1), 0.5)
    np.testing.assert_allclose(weights.abs().sum(axis=1), 1.0)
    np.testing.assert_allclose(weights.iloc[0], [0.75, -0.25])


def test_net_mode_rejects_impossible_positive_target_with_only_shorts() -> None:
    prices = price_panel(3)
    signals = pd.DataFrame({"AAA": -1.0, "BBB": -1.0}, index=prices.index)

    with pytest.raises(ValueError, match="only short"):
        build_target_weights(
            signals,
            prices,
            EqualWeightPortfolio(),
            PortfolioConfig(exposure_mode="net", rebalance_frequency="daily"),
        )


def test_monthly_schedule_holds_weights_between_rebalances() -> None:
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    prices = pd.DataFrame({"AAA": [100.0, 101.0, 102.0, 103.0]}, index=index)
    signals = pd.DataFrame({"AAA": [1.0, -1.0, -1.0, 1.0]}, index=index)

    weights = build_target_weights(
        signals,
        prices,
        EqualWeightPortfolio(),
        PortfolioConfig(rebalance_frequency="monthly"),
    )

    np.testing.assert_allclose(weights["AAA"], [1.0, 1.0, -1.0, -1.0])


def test_asset_specific_costs_are_charged_per_asset_turnover() -> None:
    prices = price_panel(3)
    weights = pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=prices.index)

    result = run_portfolio_backtest(
        prices,
        weights,
        asset_cost_rates={"AAA": 0.001, "BBB": 0.002},
    )

    np.testing.assert_allclose(result.performance["turnover"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(result.performance["transaction_cost"], [0.0, 0.0015, 0.0])


def test_inverse_volatility_stays_cash_until_lookback_exists() -> None:
    prices = price_panel(5)
    signals = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=prices.index)
    weights = build_target_weights(
        signals,
        prices,
        InverseVolatilityPortfolio(),
        PortfolioConfig(volatility_lookback=3, rebalance_frequency="daily"),
    )

    np.testing.assert_allclose(weights.iloc[:3], 0.0)
    np.testing.assert_allclose(weights.iloc[3:].sum(axis=1), 1.0)


def test_equal_weight_buy_and_hold_is_long_only() -> None:
    prices = price_panel(3)
    result = run_equal_weight_buy_and_hold(prices)

    np.testing.assert_allclose(result.target_weights, 0.5)
