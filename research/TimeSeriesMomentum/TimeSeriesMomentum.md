# Multi-Asset Time-Series Momentum with Risk-Aware Portfolio Construction 

**Core Research Question**: Do medium-term price trends contain enough information across multiple liquid asset classes to construct a diversified systematic portfolio with superior risk-adjusted characteristics relative to reasonable passive benchmarks? 

**Secondary Research Question**: Does volatility-aware position sizing materially improve the robustness and risk-adjusted performance of the raw momentum signal? 

## Hypotheses 
### Null Hypothesis ($H_0$)
Past medium-term asset returns contain no economically useful information about future asset returns sufficient to construct a robust systematic portfolio after realistic implementation assumptions. 

### Alternative Hypothesis ($H_1$)
Assets displaying positive medium-term trends tend to continue outperforming assets displaying negative medium-term trends sufficiently to support a diversified systematic trend-following portfolio.

### Secondary Hypothesis 
Volatility-aware portfolio construction improves the risk-adjusted characteristics and robustness of the raw momentum strategy. 

## Asset Universe 
For this research project, 10 liquid instruments spanning several risk profiles are considered. 

The universe consists of the following assets
1. State Street SPDR S\&P 500 ETF Trust (SPY)
    - Broad US equity representing US large-cap equity beta. 
2. iShares 7-10 Year Treasury Bond ETF (IEF)
    - Represents high-quality government bond exposure with moderate duration.
3. iShares 20+ Year Treasury Bond ETF (TLT)
    - Represents high-quality government bond exposure with long duration. 
4. iShares MSCI EAFE ETF (EFA)
    - Provides exposure to developed international equity markets. 
5. iShares MSCI Emerging Markets ETF (EEM)
    - Represents higher volatility emerging-market equity risk. 
6. iShares iBoxx $ Inv Grade Corporate Bond ETF (LQD)
    - Adds credit-spread risk distinct from government bonds. 
7. SPDR Gold Shares (GLD)
    - Provides precious-metal exposure, often associated with inflation. 
8. Invesco DB Commodity Index Tracking (DBC)
    - Represents diversified commodity futures exposure. 
9. Invesco DB US Dollar Index Bullish Fund (UUP)
    - Adds a broad long USD currency exposure. 
10. Vanguard Real Estate Index Fund ETF (VNQ)
    - Represents listed US real estate. 

## Data Contract 
The data source is Yahoo Finance, accessed through the `yfinance` Python library. The frequency is daily. The requested download range is 2007-07-01 through 2026-07-01 inclusive; because Yahoo Finance end dates are exclusive, the download request will use `end="2026-07-02"`. The raw downloaded CSVs will be retained unchanged and the retrieval date recorded when the data are obtained.

The strategy uses a 252-trading-day warm-up. The implementation must determine signal availability from the realised common trading calendar rather than assuming a calendar date; a signal is flat until 252 prior observations are available.

### Temporal Split 
- Train: 2008-07-01 through 2015-06-30
- Validation: 2015-07-01 through 2019-06-30
- Test: 2019-07-01 through 2026-07-01

These periods are mutually exclusive. Observations before the train period are retained only to initialise rolling calculations and are not included in reported train performance.

### Data Adjustment Policy 
Using adjusted OHLCV prices with data only being utilized where there is strict common-date intersection across all instruments. No forward filling will be used, duplicates will be rejected, alongside missing prices, non-positive prices, and invalid observations. 

## Signal Construction 
The baseline lookback is set to 252 trading days with the candidate family for validation set to `L = [7, 21, 63, 126, 189, 252]`. The signal is set to:
- `+1`: Positive momentum
- `0`: Zero momentum or unavailable history
- `-1`: Negative momentum

Note that targets are formed using closing data **at** $t$ with the engine applying them to the return ending at $t+1$ to avoid look-ahead bias. Validation may select lookback window from this family.  

### Momentum Equation 
The momentum equation used for the signal is

$$M_{i,t,L} = \frac{P_{i,t}}{P_{i,t-L}} - 1.$$

The result from this calculation is assigned as follows:

$$
s_{i,t} =
\begin{cases}
+1 & \text{if } M_{i,t,L} > 0, \\
0 & \text{if } M_{i,t,L} = 0 \text{ or insufficient history is available}, \\
-1 & \text{if } M_{i,t,L} < 0.
\end{cases}
$$

## Baseline Portfolio 
The baseline portfolio holds equal absolute weighting: $w_{i,t} = s_{i,t} / 10$ with monthly rebalancing and 100% gross exposure. Once all assets have sufficient history, the portfolio's net exposure is signal-dependent rather than constrained to zero.

### Inverse-Volatility Portfolio
The inverse-volatility variant is evaluated only after the equal-weight baseline. For volatility window $W$,

$$\hat{\sigma}_{i,t} = \operatorname{std}(r_{i,t-W+1:t})\sqrt{252},$$

using only returns available through date $t$. The raw weight is $s_{i,t} / \hat{\sigma}_{i,t}$ and the signed weights are normalised so that absolute weights sum to 100% gross exposure. The candidate volatility windows are `W = [63, 126, 252]`. An asset with unavailable or zero estimated volatility receives zero weight. Individual absolute weights are capped at 35%; any residual allocation is redistributed among uncapped active assets while preserving the 100% gross-exposure limit. No portfolio-level volatility target or leverage is used in the initial study.

The individual-weight cap is not currently a capability of the generic engine and will require a small, tested portfolio-construction extension before this variant is implemented.

### Validation and Model-Selection Protocol
Model selection is staged to avoid a large joint parameter search:

1. Using the equal-weight, monthly baseline, compare only the predefined momentum lookbacks on the validation period. Select a lookback from a broad, stable region after considering validation Sharpe, maximum drawdown, turnover, and cost sensitivity; do not select an isolated Sharpe maximum.
2. Hold the selected lookback and monthly frequency fixed. Compare equal-weight construction with the inverse-volatility candidates on the validation period.
3. Treat daily, weekly, and monthly rebalancing as a robustness comparison for the selected construction; do not select a frequency solely because it has the highest validation Sharpe.

All validation choices and their rationale will be recorded before the test period is run.

## Benchmarks 
The benchmarks used to compare this strategy against are:
- Equal-weight long only allocation across the same ten instruments. 
- SPY buy-and-hold 
- Individual buy-and-hold asset results for attribution context.

## Transaction Costs 
Transaction costs are modelled as one-way proportional costs per unit of turnover. The same cost is applied to every asset in a given run. The baseline assumption is 5 bps, and the predefined sensitivity grid is `cost (bps) = [0, 1, 2, 3, 5, 10, 20]`.


## Evaluation Metrics 
The following evaluation metrics will be used to assess performance:
- CAGR 
- Annualised volatility
- Sharpe Ratio 
- Sortino Ratio 
- Maximum Drawdown 
- Calmar Ratio
- Turnover 
- Trade count 
- Portfolio worst month and year
- Portfolio best month and year
- Return skewness
- Rolling volatility
- Rolling Sharpe ratio
- Return contribution by asset
- Worst and best month/year by asset for attribution context

Unless otherwise stated, the Sharpe and Sortino ratios use a zero risk-free rate. Performance metrics are calculated from net portfolio returns after transaction costs.

## Software Test Plan
Use small deterministic synthetic examples to test positive, negative, zero, and insufficient-history momentum signals; confirm that future price changes cannot change earlier signals. Test equal-weight normalisation, inverse-volatility scaling, unavailable-volatility handling, the individual-weight cap, and gross-exposure limits. Test that weights change only on scheduled rebalances and that the engine applies targets only to the next bar's return.

## Out-of-Sample Protocol
The test period remains untouched until the train/validation process is complete. The selected equal-weight and inverse-volatility specifications, together with the predefined cost scenarios and rebalance-frequency robustness comparison, will each be run once over the full test period and reported side-by-side. No parameter, asset, portfolio-construction, or timing change will be made in response to test-period performance.
