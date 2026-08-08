import pandas as pd 
import pytest 

from backtester.strategy import Strategy 

# Test-Only Implementation
class AlwaysLongStrategy(Strategy):
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index, name="target_position",)

def sample_data() -> pd.DataFrame:
    index = pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            name="date",
            )

    return pd.DataFrame(
            {"close": [100.0, 102.0, 101.0]},
            index=index,
            )

def test_strategy_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Strategy() 

def test_concrete_strategy_returns_aligned_target_positions() -> None:
    data = sample_data()
    result = AlwaysLongStrategy().generate_positions(data)

    assert isinstance(result, pd.Series)
    assert result.index.equals(data.index)
    assert result.name == "target_position"
    assert (result == 1.0).all()
