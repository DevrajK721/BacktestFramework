import pandas as pd 
import numpy as np

def calculate_total_return(equity_curve: pd.Series) -> float:
    """
    Calculate the total return of an equity curve. 
    
    Parameters:
    equity_curve (pd.Series): A pandas Series representing the equity curve, where the index is the time and the values are the equity values.

    Returns:
    float: The total return of the equity curve, calculated as (final value - initial value) / initial value.
    """

    if equity_curve.empty:
        raise ValueError("The equity curve is empty.")
    
    initial_value = equity_curve.iloc[0] # iloc is used instead of loc to avoid potential issues with non-unique indices 
    final_value = equity_curve.iloc[-1]

    total_return = (final_value - initial_value) / initial_value 
    return total_return

def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) of an equity curve. 

    Parameters:
    equity_curve (pd.Series): A pandas Series representing the equity curve, where the index is the time and the values are the equity values.
    periods_per_year (int): The number of periods in a year (default is 252 for daily data).

    Returns:
    float: The CAGR of the equity curve, calculated as ((final value / initial value) ** (1 / years)) - 1.
    """

    if equity_curve.empty:
        raise ValueError("The equity curve is empty.")
    
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    
    # Calculate the number of years based on the length of the equity curve and periods per year
    if len(equity_curve) < 2:
        raise ValueError("The equity curve must contain at least two data points to calculate CAGR.")
    if periods_per_year <= 0:
        raise ValueError("The number of periods per year must be positive.")
    if initial_value <= 0 or final_value <= 0:
        raise ValueError("Equity values must be positive to calculate CAGR.")
    
    number_of_intervals = len(equity_curve) - 1
    years = number_of_intervals / periods_per_year

    if years <= 0:
        raise ValueError("The number of years must be positive.")

    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr

def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility of a series of returns.

    Parameters:
    returns (pd.Series): A pandas Series representing the returns, where the index is the time and the values are the returns.
    periods_per_year (int): The number of periods in a year (default is 252 for daily data).

    Returns:
    float: The annualized volatility of the returns, calculated as the standard deviation of returns multiplied by the square root of periods per year.
    """

    if returns.empty:
        raise ValueError("The returns series is empty.")
    if periods_per_year <= 0:
        raise ValueError("The number of periods per year must be positive.")
    if len(returns) < 2:
        raise ValueError("The returns series must contain at least two data points to calculate volatility.")
    
    volatility = returns.std(ddof=1) * (periods_per_year ** 0.5)
    return volatility

def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate the drawdown of an equity curve.

    Parameters:
    equity_curve (pd.Series): A pandas Series representing the equity curve, where the index is the time and the values are the equity values.

    Returns:
    pd.Series: A pandas Series representing the drawdown, where the index is the time and the values are the drawdown values.
    """

    if equity_curve.empty:
        raise ValueError("The equity curve is empty.")
    
    running_peak = equity_curve.cummax() # Maximum equity observed up to each point in time
    drawdown = equity_curve / (running_peak) - 1 # Drawdown is calculated as the percentage drop from the running peak
    drawdown.name = "drawdown"
    return drawdown

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown of an equity curve.

    Parameters:
    equity_curve (pd.Series): A pandas Series representing the equity curve, where the index is the time and the values are the equity values.

    Returns:
    float: The maximum drawdown of the equity curve, calculated as the minimum value of the drawdown series.
    """

    if equity_curve.empty:
        raise ValueError("The equity curve is empty.")
    
    drawdown = calculate_drawdown(equity_curve)
    max_drawdown = drawdown.min() # Maximum drawdown is the minimum value of the drawdown series
    return max_drawdown

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio of a series of returns. 
    
    Parameters:
    returns (pd.Series): A pandas Series representing the returns, where the index is the time and the values are the returns.
    risk_free_rate (float): The risk-free rate of return (default is 0.0).
    periods_per_year (int): The number of periods in a year (default is 252 for daily data).

    Returns:
    float: The Sharpe ratio of the returns, calculated as (mean returns - risk-free rate) / standard deviation of returns.  
    """

    if returns.empty:
        raise ValueError("The returns series is empty.")
    if periods_per_year <= 0:
        raise ValueError("The number of periods per year must be positive.")
    if len(returns) < 2:
        raise ValueError("The returns series must contain at least two data points to calculate Sharpe ratio.")

    periodic_risk_free_rate = ((1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0)
    excess_returns = returns - periodic_risk_free_rate
    mean_excess_return = excess_returns.mean()
    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        raise ValueError("Volatility is zero, cannot calculate Sharpe ratio.")

    sharpe_ratio = (mean_excess_return / volatility) * (periods_per_year ** 0.5)
    return sharpe_ratio 

def calculate_downside_deviation(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualised downside deviation relative to a zero target.

    Positive returns contribute zero downside; the mean squared downside is
    taken across every return period, not only the negative observations.
    """
    if returns.empty:
        raise ValueError("The returns series is empty.")
    if periods_per_year <= 0:
        raise ValueError("The number of periods per year must be positive.")
    if len(returns) < 2:
        raise ValueError(
            "The returns series must contain at least two data points "
            "to calculate downside deviation."
        )
    if returns.isna().any():
        raise ValueError("The returns series contains missing values.")

    downside_returns = returns.clip(upper=0.0)
    return float(np.sqrt((downside_returns**2).mean()) * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualised Sortino ratio using an annual risk-free rate.

    The annual risk-free rate is converted to a compounded per-period rate.
    Downside deviation is calculated from returns in excess of that rate.
    """
    if returns.empty:
        raise ValueError("The returns series is empty.")
    if periods_per_year <= 0:
        raise ValueError("The number of periods per year must be positive.")
    if len(returns) < 2:
        raise ValueError(
            "The returns series must contain at least two data points "
            "to calculate Sortino ratio."
        )
    if returns.isna().any():
        raise ValueError("The returns series contains missing values.")

    periodic_risk_free_rate = (
        (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    )
    excess_returns = returns - periodic_risk_free_rate
    downside_deviation = calculate_downside_deviation(
        excess_returns,
        periods_per_year,
    )

    if downside_deviation == 0.0:
        raise ValueError("Downside deviation is zero, cannot calculate Sortino ratio.")

    annualized_excess_return = excess_returns.mean() * periods_per_year
    return float(annualized_excess_return / downside_deviation)


def calculate_calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate CAGR divided by the absolute maximum drawdown."""
    cagr = calculate_cagr(equity_curve, periods_per_year)
    maximum_drawdown = calculate_max_drawdown(equity_curve)

    if maximum_drawdown == 0.0:
        raise ValueError("Maximum drawdown is zero, cannot calculate Calmar ratio.")

    return float(cagr / abs(maximum_drawdown))


def calculate_hit_rate(returns: pd.Series) -> float:
    """Return the fraction of non-zero return periods that are positive.

    Zero-return periods are excluded so a strategy's hit rate is not inflated
    or diluted merely because it was flat.
    """
    if returns.empty:
        raise ValueError("The returns series is empty.")
    if returns.isna().any():
        raise ValueError("The returns series contains missing values.")

    active_returns = returns[returns != 0.0]
    if active_returns.empty:
        raise ValueError("The returns series contains no non-zero returns.")

    return float((active_returns > 0.0).mean())


def calculate_average_positive_return(returns: pd.Series) -> float:
    """Return the mean of strictly positive return periods."""
    if returns.empty:
        raise ValueError("The returns series is empty.")
    if returns.isna().any():
        raise ValueError("The returns series contains missing values.")

    positive_returns = returns[returns > 0.0]
    if positive_returns.empty:
        raise ValueError("The returns series contains no positive returns.")

    return float(positive_returns.mean())


def calculate_average_negative_return(returns: pd.Series) -> float:
    """Return the mean of strictly negative return periods."""
    if returns.empty:
        raise ValueError("The returns series is empty.")
    if returns.isna().any():
        raise ValueError("The returns series contains missing values.")

    negative_returns = returns[returns < 0.0]
    if negative_returns.empty:
        raise ValueError("The returns series contains no negative returns.")

    return float(negative_returns.mean())


def calculate_trade_count(turnover: pd.Series) -> int:
    """Count rebalance events, defined as dates with non-zero turnover.

    A long-to-short reversal is one rebalance event, even though it has two
    units of turnover. Total turnover records that economic size separately.
    """
    if turnover.empty:
        raise ValueError("The turnover series is empty.")
    if turnover.isna().any():
        raise ValueError("The turnover series contains missing values.")
    if (turnover < 0.0).any():
        raise ValueError("Turnover must be non-negative.")

    return int((turnover > 0.0).sum())


def calculate_total_turnover(turnover: pd.Series) -> float:
    """Return the sum of turnover units across the backtest."""
    if turnover.empty:
        raise ValueError("The turnover series is empty.")
    if turnover.isna().any():
        raise ValueError("The turnover series contains missing values.")
    if (turnover < 0.0).any():
        raise ValueError("Turnover must be non-negative.")

    return float(turnover.sum())
