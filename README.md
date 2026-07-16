# Quant Portfolio Analytics

A professional-grade Python toolkit for portfolio risk and performance analysis. It ingests real market data, cleans and validates it, computes institutional-style performance and risk metrics, runs Monte Carlo portfolio optimization, and produces a full chart suite plus a self-contained Markdown report — via either a one-command CLI pipeline or an interactive Streamlit dashboard.

![Efficient Frontier](reports/figures/07_efficient_frontier.png)

## Features

**Analytics engine**
- Daily, cumulative, and rolling returns; annualized return (CAGR) and volatility
- Risk-adjusted metrics: Sharpe, Sortino, Calmar, Information Ratio
- Full drawdown analysis: depth, peak/trough/recovery dates, underwater chart
- Tail risk: historical VaR and CVaR (expected shortfall) at 95% confidence
- Benchmark-relative analytics: CAPM beta, Jensen's alpha, tracking error
- Correlation analysis across assets

**Portfolio construction**
- Custom weights with monthly / quarterly / yearly rebalancing or buy-and-hold
- Allocation drift visualization (why rebalancing matters)
- Per-asset return contribution attribution
- Monte Carlo efficient frontier (5,000 simulated portfolios) with max-Sharpe and min-volatility portfolio identification

**Engineering**
- Clean modular architecture: acquisition → cleaning → analytics → visualization → reporting
- Local CSV caching to avoid repeated API calls
- Deterministic synthetic-data fallback (`--offline`) so the project runs and demos anywhere, with no network and no API keys
- `CleaningReport` audit trail documenting every transformation applied to raw data
- Unit-tested metrics (pytest) validated against hand-computed values
- Zero duplicated logic between the CLI and the dashboard — both call the same `src/` modules

## Tech Stack

Python 3.10+ · pandas · NumPy · matplotlib · seaborn · yfinance · Streamlit · PyYAML · pytest

## Installation

```bash
git clone https://github.com/<you>/quant-portfolio-analytics.git
cd quant-portfolio-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Full pipeline (report + 9 charts):**
```bash
python run_analysis.py
```

**Custom portfolio:**
```bash
python run_analysis.py \
  --tickers AAPL MSFT NVDA TLT GLD \
  --benchmark SPY \
  --weights AAPL=0.3 MSFT=0.25 NVDA=0.15 TLT=0.2 GLD=0.1 \
  --rebalance Q --start 2019-01-01 --end 2025-12-31
```

**Offline demo (no internet / no yfinance needed):**
```bash
python run_analysis.py --offline
```

**Interactive dashboard:**
```bash
streamlit run app.py
```

**Tests:**
```bash
pytest
```

Defaults live in `config.yaml`; any CLI flag overrides them. Outputs land in `reports/analysis_report.md` and `reports/figures/`.

## Project Structure

```
quant-portfolio-analytics/
├── run_analysis.py        # CLI pipeline entry point (6-step ETL → report)
├── app.py                 # Streamlit dashboard (reuses src/ modules)
├── config.yaml            # Default tickers, weights, dates, rebalancing
├── requirements.txt
├── src/
│   ├── data_loader.py     # yfinance download + caching + synthetic fallback
│   ├── preprocessing.py   # Cleaning, alignment, validation + audit report
│   ├── metrics.py         # All performance & risk metrics
│   ├── portfolio.py       # Weights, rebalancing, Monte Carlo frontier
│   ├── visualization.py   # 9 publication-quality charts
│   └── report.py          # Markdown report generator
├── tests/
│   └── test_metrics.py    # Unit tests with hand-verified expected values
├── data/cache/            # Cached price downloads (gitignored)
└── reports/
    ├── analysis_report.md # Generated report
    └── figures/           # Generated charts
```

## Sample Insights

Running the default six-asset portfolio (tech + financials + bonds + gold) against SPY produces findings like:

- The 20% TLT allocation cut portfolio max drawdown meaningfully below the benchmark's, at the cost of upside during equity rallies — visible directly in the underwater chart.
- Buy-and-hold allocation drift shows the tech sleeve growing well beyond target weight over multi-year horizons, quantifying why periodic rebalancing matters for risk control.
- The Monte Carlo frontier locates a max-Sharpe allocation that typically shifts weight from the highest-volatility names toward the low-correlation sleeves (bonds, gold).

*(Exact numbers depend on the date range and live market data; the `--offline` mode is clearly labeled as synthetic.)*

## Design Decisions

- **Metrics take returns, not prices.** Every function in `metrics.py` operates on daily return series, making them composable — the same Sharpe function works for a single stock, a portfolio, or a rolling window.
- **Synthetic fallback instead of hard network dependency.** Demos, tests, and CI never break because an API rate-limits. The fallback is seeded (reproducible) and loudly labeled.
- **Monte Carlo frontier over closed-form optimization.** Random-portfolio sampling is transparent, easy to explain in interviews, and visualizes the entire feasible region rather than just the optimum. A `scipy.optimize` version is a natural extension.
- **Charts return Figures.** The dashboard and CLI share one visualization codebase.

## Future Improvements

- Closed-form mean–variance optimization (SLSQP) with weight constraints
- Factor exposure analysis (Fama–French three-factor regression via `statsmodels`)
- Transaction cost modeling in the rebalancing simulator
- HTML/PDF report export
- CI workflow (GitHub Actions) running the test suite on push

## Disclaimer

Educational project. Nothing here is investment advice.
