import pandas as pd 
from backtester.strategy import Strategy 

class MovingAverageCrossover(Strategy):
    def __init__(self, fast_window: int, slow_window: int) -> None:
        if fast_window <= 0 or slow_window <= 0 or fast_window >= slow_window:
            raise ValueError("Moving Average windows wrongly initialized")

        self.fast_window = fast_window 
        self.slow_window = slow_window 

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data["close"].rolling(
                window=self.fast_window,
                min_periods=self.fast_window,
                ).mean()

        slow_ma = data["close"].rolling(
                window=self.slow_window,
                min_periods=self.slow_window,
                ).mean()

        positions = (fast_ma > slow_ma).astype(float) # Crossover strategy 
        positions.name = "target_position"

        return positions 
