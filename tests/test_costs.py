import numpy as np
import pandas as pd
import pytest

from backtester.costs import (
    calculate_transaction_costs,
    calculate_turnover,
)


def executed_positions(values: list[float]) -> pd.Series:
    index = pd.date_range(
        "2024-01-02",
        periods=len(values),
        freq="D",
        name="date",
    )

    return pd.Series(
        values,
        index=index,
        dtype=float,
        name="executed_position",
    )


def test_turnover_includes_full_position_reversal() -> None:
    positions = executed_positions([0.0, 1.0, 1.0, -1.0])

    result = calculate_turnover(positions)

    assert result.name == "turnover"
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0, 2.0])


def test_initial_position_is_entered_from_flat() -> None:
    positions = executed_positions([1.0, 1.0])

    result = calculate_turnover(positions)

    np.testing.assert_allclose(result, [1.0, 0.0])


def test_transaction_costs_are_proportional_to_turnover() -> None:
    turnover = executed_positions([0.0, 1.0, 0.0, 2.0])
    turnover.name = "turnover"

    result = calculate_transaction_costs(turnover, cost_rate=0.0005)

    assert result.name == "transaction_cost"
    np.testing.assert_allclose(result, [0.0, 0.0005, 0.0, 0.001])


def test_unchanged_position_has_zero_turnover_and_cost() -> None:
    positions = executed_positions([0.0, 0.0, 0.0])

    turnover = calculate_turnover(positions)
    transaction_cost = calculate_transaction_costs(turnover, cost_rate=0.001)

    np.testing.assert_allclose(turnover, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(transaction_cost, [0.0, 0.0, 0.0])


def test_negative_cost_rate_raises_value_error() -> None:
    turnover = executed_positions([0.0, 1.0])
    turnover.name = "turnover"

    with pytest.raises(ValueError, match="non-negative"):
        calculate_transaction_costs(turnover, cost_rate=-0.0005)
