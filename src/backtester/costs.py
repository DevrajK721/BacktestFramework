import pandas as pd 

def calculate_turnover(executed_positions: pd.Series) -> pd.Series:
    turnover = (
            executed_positions.sub(executed_positions.shift(1, fill_value=0.0)).abs())
    turnover.name = "turnover"
    return turnover

def calculate_transaction_costs(turnover: pd.Series, cost_rate: float) -> pd.Series:
    if cost_rate < 0:
        raise ValueError("Cost rate must be non-negative")

    transaction_cost = cost_rate * turnover
    transaction_cost.name = "transaction_cost"
    return transaction_cost
    


