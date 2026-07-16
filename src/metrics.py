"""
metrics.py
----------
Core financial performance and risk metrics.

All functions operate on *daily simple returns* (pd.Series or pd.DataFrame)
unless stated otherwise. Annualization assumes 252 trading days by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------

def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily returns from a price series/frame."""
    return prices.pct_change().dropna(how="all")


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Growth of $1 invested at the start (cumulative compounded return)."""
    return (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate (CAGR) implied by the return series."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    total_growth = (1.0 + r).prod()
    if total_growth <= 0:
        return -1.0
    return float(total_growth ** (periods / len(r)) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of daily returns."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods))


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 periods: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio. `risk_free_rate` is an annual rate (e.g. 0.04)."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = r - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  periods: int = TRADING_DAYS) -> float:
    """Annualized Sortino ratio: penalizes downside deviation only."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = r - rf_daily
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return np.nan
    return float(excess.mean() / downside_dev * np.sqrt(periods))


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized return divided by absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(annualized_return(returns, periods) / abs(mdd))


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point in time (0 at peaks, negative below peaks)."""
    wealth = (1.0 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline over the full period (negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else np.nan


def drawdown_details(returns: pd.Series) -> dict:
    """Max drawdown magnitude, peak date, trough date, and recovery date."""
    dd = drawdown_series(returns)
    if dd.empty:
        return {}
    trough = dd.idxmin()
    peak = dd.loc[:trough][dd.loc[:trough] == 0].index
    peak_date = peak[-1] if len(peak) else dd.index[0]
    post = dd.loc[trough:]
    recovered = post[post == 0].index
    recovery_date = recovered[0] if len(recovered) else None
    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak_date,
        "trough_date": trough,
        "recovery_date": recovery_date,
    }


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical 1-day VaR. Returns a positive loss number (e.g. 0.021 = 2.1%)."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return float(-np.percentile(r, 100 * (1 - confidence)))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall (CVaR): mean loss on days beyond the VaR threshold."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    cutoff = np.percentile(r, 100 * (1 - confidence))
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


# ---------------------------------------------------------------------------
# Benchmark-relative metrics
# ---------------------------------------------------------------------------

def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """CAPM beta versus a benchmark."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


def alpha(returns: pd.Series, benchmark_returns: pd.Series,
          risk_free_rate: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualized CAPM (Jensen's) alpha versus a benchmark."""
    b = beta(returns, benchmark_returns)
    if np.isnan(b):
        return np.nan
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    daily_alpha = (aligned.iloc[:, 0].mean() - rf_daily) - b * (aligned.iloc[:, 1].mean() - rf_daily)
    return float(daily_alpha * periods)


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series,
                   periods: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of active returns."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(periods))


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                      periods: int = TRADING_DAYS) -> float:
    """Annualized active return divided by tracking error."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std(ddof=1)
    if te == 0:
        return np.nan
    return float(active.mean() / te * np.sqrt(periods))


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_volatility(returns: pd.DataFrame | pd.Series, window: int = 63,
                       periods: int = TRADING_DAYS):
    """Rolling annualized volatility (default window: one quarter)."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_returns(returns: pd.DataFrame | pd.Series, window: int = 252):
    """Rolling compounded return over `window` days (default: one year)."""
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def rolling_sharpe(returns: pd.Series, window: int = 126,
                   risk_free_rate: float = 0.0,
                   periods: int = TRADING_DAYS) -> pd.Series:
    """Rolling annualized Sharpe ratio (default window: six months)."""
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = returns - rf_daily
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return mean / std * np.sqrt(periods)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(returns: pd.DataFrame, benchmark: str | None = None,
                  risk_free_rate: float = 0.0) -> pd.DataFrame:
    """One-row-per-asset summary of all headline metrics."""
    rows = {}
    bench = returns[benchmark] if benchmark and benchmark in returns else None
    for col in returns.columns:
        r = returns[col]
        row = {
            "Ann. Return": annualized_return(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free_rate),
            "Sortino": sortino_ratio(r, risk_free_rate),
            "Calmar": calmar_ratio(r),
            "Max Drawdown": max_drawdown(r),
            "VaR 95% (1d)": value_at_risk(r),
            "CVaR 95% (1d)": conditional_var(r),
        }
        if bench is not None and col != benchmark:
            row["Beta"] = beta(r, bench)
            row["Alpha (ann.)"] = alpha(r, bench, risk_free_rate)
            row["Info Ratio"] = information_ratio(r, bench)
        rows[col] = row
    return pd.DataFrame(rows).T
