import numpy as np
import pandas as pd

from backtester.engine import run_equal_weight_buy_and_hold, run_portfolio_backtest
from backtester.portfolio import PortfolioConfig
from backtester.reporting import create_portfolio_report


def test_report_creates_pdf_without_preserving_tex_source(tmp_path) -> None:
    index = pd.date_range("2024-01-02", periods=70, freq="D", name="date")
    prices = pd.DataFrame(
        {
            "AAA": 100.0 * np.cumprod(1.0 + np.full(len(index), 0.001)),
            "BBB": 100.0 * np.cumprod(1.0 + np.full(len(index), -0.0004)),
        },
        index=index,
    )
    weights = pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=index)
    result = run_portfolio_backtest(prices, weights, cost_rate=0.0005)
    benchmark = run_equal_weight_buy_and_hold(prices, cost_rate=0.0005)
    output = tmp_path / "portfolio_report.pdf"

    report = create_portfolio_report(
        result,
        benchmark,
        PortfolioConfig(rebalance_frequency="monthly"),
        output,
        periods_per_year=252,
        strategy_name="Test portfolio",
    )

    assert report == output
    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.rglob("*.tex"))
