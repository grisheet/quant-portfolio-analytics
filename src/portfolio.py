"""
portfolio.py
------------
Portfolio construction, rebalancing simulation, and efficient-frontier
exploration via Monte Carlo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import TRADING_DAYS, annualized_return, annualized_volatility, sharpe_ratio


# ---------------------------------------------------------------------------
# Portfolio return construction
# ---------------------------------------------------------------------------

def normalize_weights(weights: dict[str, float], tickers: list[str]) -> pd.Series:
    """Validate weights against available tickers and normalize to sum to 1."""
    w = pd.Series({t.upper(): float(v) for t, v in weights.items()})
    missing = [t for t in w.index if t not in tickers]
    if missing:
        raise ValueError(f"Weights reference tickers not in the data: {missing}")
    if (w < 0).any():
        raise ValueError("Short positions are not supported in this model.")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return w / total


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series,
                      rebalance: str | None = "M") -> pd.Series:
    """
    Daily portfolio returns for a given weight vector.

    rebalance:
      - "M" / "Q" / "Y": reset to target weights at each period start.
      - None: buy-and-hold - weights drift with performance.
    """
    rets = asset_returns[weights.index].copy()

    if rebalance is None:
        wealth = (1.0 + rets).cumprod()
        port_value = wealth.mul(weights, axis=1).sum(axis=1)
        port_value = pd.concat(
            [pd.Series([1.0], index=[rets.index[0] - pd.Timedelta(days=1)]), port_value]
        )
        return port_value.pct_change().dropna()

    freq_map = {"M": "MS", "Q": "QS", "Y": "YS"}
    period = rets.index.to_period({"M": "M", "Q": "Q", "Y": "Y"}[rebalance])
    _ = freq_map  # kept for clarity of supported options

    out = []
    for _, block in rets.groupby(period):
        wealth = (1.0 + block).cumprod()
        value = wealth.mul(weights, axis=1).sum(axis=1)
        prev = pd.Series([1.0], index=[block.index[0] - pd.Timedelta(days=1)])
        out.append(pd.concat([prev, value]).pct_change().dropna())
    return pd.concat(out)


def weight_drift(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Evolution of buy-and-hold allocation weights over time."""
    wealth = (1.0 + asset_returns[weights.index]).cumprod().mul(weights, axis=1)
    return wealth.div(wealth.sum(axis=1), axis=0)


def return_contribution(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Approximate contribution of each asset to total portfolio return."""
    total_asset_return = (1.0 + asset_returns[weights.index]).prod() - 1.0
    contrib = total_asset_return * weights
    return contrib.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Monte Carlo efficient frontier
# ---------------------------------------------------------------------------

def monte_carlo_frontier(asset_returns: pd.DataFrame, n_portfolios: int = 5000,
                         risk_free_rate: float = 0.0, seed: int = 7) -> pd.DataFrame:
    """
    Sample random long-only portfolios and compute their annualized
    return / volatility / Sharpe.

    Used to visualize the feasible region and locate the max-Sharpe and
    min-volatility portfolios.
    """
    rng = np.random.default_rng(seed)
    mean_daily = asset_returns.mean().values
    cov_daily = asset_returns.cov().values
    n_assets = asset_returns.shape[1]

    w = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    ann_ret = w @ mean_daily * TRADING_DAYS
    ann_vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov_daily, w) * TRADING_DAYS)
    sharpe = (ann_ret - risk_free_rate) / ann_vol

    frontier = pd.DataFrame({"return": ann_ret, "volatility": ann_vol, "sharpe": sharpe})
    for i, col in enumerate(asset_returns.columns):
        frontier[f"w_{col}"] = w[:, i]
    return frontier


def optimal_portfolios(frontier: pd.DataFrame) -> dict[str, pd.Series]:
    """Extract the max-Sharpe and min-volatility portfolios from the sample."""
    return {
        "max_sharpe": frontier.loc[frontier["sharpe"].idxmax()],
        "min_volatility": frontier.loc[frontier["volatility"].idxmin()],
    }


def portfolio_summary(port_rets: pd.Series, bench_rets: pd.Series | None,
                      risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Side-by-side headline metrics for the portfolio and its benchmark."""
    from src import metrics as m

    def _row(r: pd.Series) -> dict:
        return {
            "Ann. Return": annualized_return(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free_rate),
            "Sortino": m.sortino_ratio(r, risk_free_rate),
            "Max Drawdown": m.max_drawdown(r),
            "Calmar": m.calmar_ratio(r),
            "VaR 95% (1d)": m.value_at_risk(r),
            "CVaR 95% (1d)": m.conditional_var(r),
        }

    rows = {"Portfolio": _row(port_rets)}
    if bench_rets is not None:
        rows["Benchmark"] = _row(bench_rets)
        rows["Portfolio"]["Beta"] = m.beta(port_rets, bench_rets)
        rows["Portfolio"]["Alpha (ann.)"] = m.alpha(port_rets, bench_rets, risk_free_rate)
        rows["Portfolio"]["Info Ratio"] = m.information_ratio(port_rets, bench_rets)
    return pd.DataFrame(rows).T
