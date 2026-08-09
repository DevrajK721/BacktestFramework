import pandas as pd
from backtester.costs import calculate_turnover, calculate_transaction_costs

def run_backtest(data: pd.DataFrame, target_positions: pd.Series, cost_rate: float = 0.0, initial_capital: float = 1.0) -> pd.DataFrame:
    """Run a close-to-close backtest for one asset with proportional costs.

    Parameters
    ----------
    data:
        Clean market data indexed by date and containing a ``close`` column.
    target_positions:
        Strategy target exposures aligned to ``data.index``. A target set using
        date *t* information is applied to the return ending on date *t + 1*.
    cost_rate:
        Proportional transaction cost per unit of turnover. For example,
        ``0.0005`` represents 5 basis points.

    Returns
    -------
    pd.DataFrame
        An audit table with ``close``, ``asset_return``, ``target_position``,
        ``executed_position``, ``gross_return``, ``turnover``,
        ``transaction_cost``, and ``net_return`` columns.
    """
    if "close" not in data.columns:
        raise ValueError("data must contain a close column")

    if not target_positions.index.equals(data.index):
        raise ValueError("target_positions index must equal data index")

    if target_positions.isna().any():
        raise ValueError("target_positions contain missing values")

    if ((target_positions < -1.0) | (target_positions > 1.0)).any():
        raise ValueError("target_positions must be within range [-1.0, 1.0]")

    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive")

    # Compute asset return 
    asset_return = data["close"].pct_change(fill_method=None).fillna(0.0)

    executed_position = target_positions.shift(1).fillna(0.0) 
    turnover = calculate_turnover(executed_position)
    transaction_cost = calculate_transaction_costs(turnover, cost_rate)

    gross_return = asset_return * executed_position
    net_return = gross_return - transaction_cost

    equity_curve = (1.0 + net_return).cumprod() * initial_capital

    # Output dataframe 
    df_mapping = {
        "close": data["close"],
        "asset_return": asset_return,
        "target_position": target_positions,
        "executed_position": executed_position,
        "gross_return": gross_return,
        "turnover": turnover, 
        "transaction_cost": transaction_cost, 
        "net_return": net_return,
        "equity_curve": equity_curve,
    }
    
    return pd.DataFrame(df_mapping, index=data.index)
