# Portfolio Risk & Performance Report

*Generated 2026-07-16 00:53*

> **Note:** generated from *synthetic* demo data (seeded GBM). Run without `--offline` and with `yfinance` installed for real market data.

## Configuration

- **Assets:** AAPL, MSFT, NVDA, JPM, TLT, GLD
- **Benchmark:** SPY
- **Period:** 2020-01-01 to 2025-12-31
- **Rebalancing:** M

### Target Weights

| Asset | Weight |
|---|---|
| AAPL | 20.0% |
| MSFT | 20.0% |
| NVDA | 15.0% |
| JPM | 15.0% |
| TLT | 20.0% |
| GLD | 10.0% |

## Headline Metrics — Portfolio vs Benchmark

|           | Ann. Return   | Ann. Volatility   |   Sharpe |   Sortino | Max Drawdown   |   Calmar | VaR 95% (1d)   | CVaR 95% (1d)   | Beta   | Alpha (ann.)   | Info Ratio   |
|:----------|:--------------|:------------------|---------:|----------:|:---------------|---------:|:---------------|:----------------|:-------|:---------------|:-------------|
| Portfolio | 1.48%         | 16.19%            |    -0.07 |     -0.1  | -32.30%        |     0.05 | 1.66%          | 2.06%           | 0.72   | 0.95%          | 0.14         |
| Benchmark | -0.20%        | 15.69%            |    -0.18 |     -0.26 | -38.00%        |    -0.01 | 1.58%          | 2.04%           | —      | —              | —            |

## Per-Asset Metrics

|      | Ann. Return   | Ann. Volatility   |   Sharpe |   Sortino |   Calmar | Max Drawdown   | VaR 95% (1d)   | CVaR 95% (1d)   | Beta   | Alpha (ann.)   | Info Ratio   |
|:-----|:--------------|:------------------|---------:|----------:|---------:|:---------------|:---------------|:----------------|:-------|:---------------|:-------------|
| AAPL | 4.00%         | 30.07%            |     0.15 |      0.21 |     0.07 | -55.75%        | 3.09%          | 3.78%           | 1.10   | 7.69%          | 0.30         |
| GLD  | 10.11%        | 14.49%            |     0.47 |      0.69 |     0.44 | -23.18%        | 1.40%          | 1.77%           | 0.09   | 7.00%          | 0.47         |
| JPM  | -14.17%       | 20.24%            |    -0.85 |     -1.15 |    -0.2  | -69.65%        | 2.08%          | 2.70%           | 0.79   | -14.87%        | -0.87        |
| MSFT | 1.72%         | 29.45%            |     0.07 |      0.1  |     0.03 | -53.47%        | 2.98%          | 3.77%           | 1.08   | 5.23%          | 0.21         |
| NVDA | -0.35%        | 30.46%            |     0.01 |      0.02 |    -0.01 | -48.92%        | 3.18%          | 3.85%           | 1.18   | 3.78%          | 0.13         |
| SPY  | -0.20%        | 15.69%            |    -0.18 |     -0.26 |    -0.01 | -38.00%        | 1.58%          | 2.04%           | —      | —              | —            |
| TLT  | -0.14%        | 7.13%             |    -0.53 |     -0.74 |    -0.01 | -12.02%        | 0.74%          | 0.92%           | -0.09  | -4.07%         | -0.05        |

## Worst Drawdown

- **Depth:** -32.30%
- **Peak:** 2024-11-15
- **Trough:** 2025-09-03
- **Recovery:** not yet recovered

## Return Contribution by Asset

| Asset | Contribution |
|---|---|
| GLD | +8.18% |
| AAPL | +5.52% |
| MSFT | +2.24% |
| TLT | -0.18% |
| NVDA | -0.32% |
| JPM | -9.19% |

## Monte Carlo Optimization

Best Sharpe portfolio found across 5,000 random long-only allocations:
**Sharpe 0.42** at 9.3% return / 12.6% volatility
(AAPL: 8%, MSFT: 11%, NVDA: 3%, JPM: 0%, TLT: 5%, GLD: 73%).

## Charts

![01_cumulative_returns](figures/01_cumulative_returns.png)

![02_drawdown](figures/02_drawdown.png)

![03_rolling_volatility](figures/03_rolling_volatility.png)

![04_rolling_sharpe](figures/04_rolling_sharpe.png)

![05_correlation_heatmap](figures/05_correlation_heatmap.png)

![06_return_distribution](figures/06_return_distribution.png)

![07_efficient_frontier](figures/07_efficient_frontier.png)

![08_monthly_returns](figures/08_monthly_returns.png)

![09_weight_drift](figures/09_weight_drift.png)
