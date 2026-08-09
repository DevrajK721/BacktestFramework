"""PDF reporting for completed portfolio backtests.

The LaTeX source and chart files are created in a temporary directory. Only
the compiled PDF is retained at the requested destination.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib

# Reports are generated without a display server, including from ordinary
# research scripts on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.engine import PortfolioBacktestResult
from backtester.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown,
    calculate_hit_rate,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_total_turnover,
    calculate_trade_count,
)
from backtester.plotting import (
    plot_drawdown,
    plot_equity_curves,
    plot_portfolio_weights,
    plot_turnover,
)
from backtester.portfolio import PortfolioConfig


def create_portfolio_report(
    result: PortfolioBacktestResult,
    benchmark: PortfolioBacktestResult,
    config: PortfolioConfig,
    output_path: str | Path,
    periods_per_year: float,
    strategy_name: str = "Portfolio strategy",
    asset_cost_rates: Mapping[str, float] | None = None,
) -> Path:
    """Compile a polished PDF report and retain no editable LaTeX source."""
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    if not result.performance.index.equals(benchmark.performance.index):
        raise ValueError("result and benchmark must share the same date index")
    destination = Path(output_path)
    if destination.suffix.lower() != ".pdf":
        raise ValueError("output_path must end in .pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to generate PDF reports")

    with tempfile.TemporaryDirectory(prefix="backtester-report-") as temporary:
        workspace = Path(temporary)
        figure_paths = _save_figures(result, benchmark, workspace)
        tex_path = workspace / "report.tex"
        tex_path.write_text(
            _render_tex(
                result,
                benchmark,
                config,
                periods_per_year,
                strategy_name,
                asset_cost_rates,
                figure_paths,
            ),
            encoding="ascii",
        )
        completed = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(workspace),
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        compiled_pdf = workspace / "report.pdf"
        if completed.returncode != 0 or not compiled_pdf.exists():
            message = completed.stdout + completed.stderr
            raise RuntimeError(f"PDF compilation failed: {message[-3000:]}")
        shutil.copy2(compiled_pdf, destination)
    return destination


def _save_figures(
    result: PortfolioBacktestResult,
    benchmark: PortfolioBacktestResult,
    directory: Path,
) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    figure, ax = plt.subplots(figsize=(8.3, 4.2))
    plot_equity_curves(
        result.performance["equity_curve"], benchmark.performance["equity_curve"], ax=ax
    )
    figure.tight_layout()
    figures["equity"] = directory / "equity.png"
    figure.savefig(figures["equity"], dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8.3, 2.2))
    plot_drawdown(calculate_drawdown(result.performance["equity_curve"]), ax=ax)
    figure.tight_layout()
    figures["drawdown"] = directory / "drawdown.png"
    figure.savefig(figures["drawdown"], dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8.3, 2.6))
    plot_portfolio_weights(result.target_weights, ax=ax)
    figure.tight_layout()
    figures["weights"] = directory / "weights.png"
    figure.savefig(figures["weights"], dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8.3, 2.0))
    plot_turnover(result.performance["turnover"], ax=ax)
    figure.tight_layout()
    figures["turnover"] = directory / "turnover.png"
    figure.savefig(figures["turnover"], dpi=180)
    plt.close(figure)
    return figures


def _render_tex(
    result: PortfolioBacktestResult,
    benchmark: PortfolioBacktestResult,
    config: PortfolioConfig,
    periods_per_year: float,
    strategy_name: str,
    asset_cost_rates: Mapping[str, float] | None,
    figure_paths: Mapping[str, Path],
) -> str:
    performance = result.performance
    index = performance.index
    strategy_metrics = _metric_rows(performance, periods_per_year)
    benchmark_metrics = _metric_rows(benchmark.performance, periods_per_year)
    assets = ", ".join(map(_tex_escape, result.target_weights.columns))
    costs = _cost_text(asset_cost_rates, result.target_weights.columns)
    ending = chr(92) * 2
    metric_table = chr(10).join(
        f"{_tex_escape(name)} & {_format_metric(strategy)} & {_format_metric(benchmark)} {ending}"
        for name, strategy, benchmark in zip(
            strategy_metrics.index, strategy_metrics, benchmark_metrics, strict=True
        )
    )
    config_rows = chr(10).join(
        [
            f"Strategy & {_tex_escape(strategy_name)} {ending}",
            f"Assets & {assets} {ending}",
            f"Sample & {index.min().date()} to {index.max().date()} ({len(index)} bars) {ending}",
            f"Annualisation & {_format_number(periods_per_year)} periods/year {ending}",
            f"Exposure mode & {_tex_escape(config.exposure_mode)} {ending}",
            f"Gross target / limit & {_format_percent(config.target_gross_exposure)} / {_format_percent(config.gross_exposure_limit)} {ending}",
            f"Net target & {_format_percent(config.target_net_exposure)} {ending}",
            f"Rebalance frequency & {_tex_escape(str(config.rebalance_frequency))} {ending}",
            f"Inverse-volatility lookback & {config.volatility_lookback} bars {ending}",
            f"Transaction costs & {_tex_escape(costs)} {ending}",
        ]
    )
    tex = f"""¤documentclass[10pt]{{article}}
¤usepackage[margin=0.72in]{{geometry}}
¤usepackage{{booktabs}}
¤usepackage{{graphicx}}
¤usepackage{{xcolor}}
¤usepackage{{fancyhdr}}
¤usepackage{{array}}
¤usepackage{{titlesec}}
¤definecolor{{navy}}{{HTML}}{{16324F}}
¤definecolor{{slate}}{{HTML}}{{4F6272}}
¤titleformat{{¤section}}{{¤large¤bfseries¤color{{navy}}}}{{}}{{0em}}{{}}
¤pagestyle{{fancy}}
¤fancyhf{{}}
¤lhead{{¤color{{slate}}Backtest Framework}}
¤rhead{{¤color{{slate}}Portfolio Research Report}}
¤cfoot{{¤thepage}}
¤setlength{{¤parindent}}{{0pt}}
¤begin{{document}}

{{¤LARGE¤bfseries¤color{{navy}} Portfolio Backtest Report}}¤¤[0.35em]
{{¤large {_tex_escape(strategy_name)}}}¤¤[1.0em]

¤section*{{Research configuration}}
¤begin{{tabular}}{{@{{}}p{{0.31¤linewidth}}p{{0.65¤linewidth}}@{{}}}}
¤toprule
¤textbf{{Field}} & ¤textbf{{Value}} ¤¤
¤midrule
{config_rows}
¤bottomrule
¤end{{tabular}}

¤section*{{Performance summary}}
¤begin{{tabular}}{{@{{}}lrr@{{}}}}
¤toprule
¤textbf{{Metric}} & ¤textbf{{Strategy}} & ¤textbf{{Equal-weight buy-and-hold}} ¤¤
¤midrule
{metric_table}
¤bottomrule
¤end{{tabular}}

¤vspace{{0.8em}}
¤includegraphics[width=¤linewidth]{{{figure_paths["equity"].as_posix()}}}

¤newpage
¤section*{{Risk and allocation diagnostics}}
¤includegraphics[width=¤linewidth]{{{figure_paths["drawdown"].as_posix()}}}¤¤[0.25em]
¤includegraphics[width=¤linewidth]{{{figure_paths["weights"].as_posix()}}}¤¤[0.25em]
¤includegraphics[width=¤linewidth]{{{figure_paths["turnover"].as_posix()}}}

¤end{{document}}
"""
    return tex.replace("¤", chr(92))


def _metric_rows(performance: pd.DataFrame, periods_per_year: float) -> pd.Series:
    returns = performance["net_return"].iloc[1:]
    equity = performance["equity_curve"]
    calculations = {
        "Total return": lambda: calculate_total_return(equity),
        "CAGR": lambda: calculate_cagr(equity, periods_per_year),
        "Annualised volatility": lambda: calculate_annualized_volatility(returns, periods_per_year),
        "Maximum drawdown": lambda: calculate_max_drawdown(equity),
        "Sharpe ratio": lambda: calculate_sharpe_ratio(returns, periods_per_year=periods_per_year),
        "Sortino ratio": lambda: calculate_sortino_ratio(returns, periods_per_year=periods_per_year),
        "Calmar ratio": lambda: calculate_calmar_ratio(equity, periods_per_year),
        "Hit rate": lambda: calculate_hit_rate(returns),
        "Trade count": lambda: calculate_trade_count(performance["turnover"]),
        "Total turnover": lambda: calculate_total_turnover(performance["turnover"]),
    }
    values: dict[str, float] = {}
    for name, calculation in calculations.items():
        try:
            values[name] = float(calculation())
        except ValueError:
            values[name] = np.nan
    return pd.Series(values)


def _format_metric(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return _format_number(value)


def _format_number(value: float) -> str:
    return f"{value:,.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * value:,.1f}" + chr(92) + "%"


def _cost_text(asset_cost_rates: Mapping[str, float] | None, columns: pd.Index) -> str:
    if asset_cost_rates is None:
        return "Uniform rate supplied to engine"
    ordered = [f"{ticker}: {10000.0 * asset_cost_rates[ticker]:.1f} bps" for ticker in columns]
    return "; ".join(ordered)


def _tex_escape(value: object) -> str:
    text = str(value)
    slash = chr(92)
    replacements = {
        slash: slash + "textbackslash{}",
        "&": slash + "&",
        "%": slash + "%",
        "$": slash + "$",
        "#": slash + "#",
        "_": slash + "_",
        "{": slash + "{",
        "}": slash + "}",
        "~": slash + "textasciitilde{}",
        "^": slash + "textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)
