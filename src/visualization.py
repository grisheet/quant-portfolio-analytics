"""
visualization.py
----------------
Publication-quality charts. Every function saves a PNG and returns the
matplotlib Figure so the Streamlit dashboard can reuse the same code.

Chart design principles:
- One message per chart.
- Portfolio always plotted in the same accent color; benchmark in gray.
- Percent axes formatted as percents; no unlabeled axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from src import metrics as m

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

ACCENT   = "#1f6feb"  # portfolio
GRAY     = "#8b949e"  # benchmark
NEGATIVE = "#d1242f"

sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    return fig


def _pct_axis(ax, axis: str = "y"):
    fmt = mtick.PercentFormatter(1.0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# 1. Cumulative performance
# ---------------------------------------------------------------------------

def plot_cumulative_returns(port_rets: pd.Series, bench_rets: pd.Series | None,
                            bench_name: str = "Benchmark") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    (m.cumulative_returns(port_rets) - 1).plot(ax=ax, color=ACCENT, lw=2, label="Portfolio")
    if bench_rets is not None:
        (m.cumulative_returns(bench_rets) - 1).plot(ax=ax, color=GRAY, lw=1.6,
                                                     label=bench_name, alpha=0.9)
    ax.set_title("Cumulative Return: Portfolio vs Benchmark")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("")
    _pct_axis(ax)
    ax.legend(frameon=False)
    return _save(fig, "01_cumulative_returns")


# ---------------------------------------------------------------------------
# 2. Drawdown
# ---------------------------------------------------------------------------

def plot_drawdown(port_rets: pd.Series) -> plt.Figure:
    dd = m.drawdown_series(port_rets)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values, 0, color=NEGATIVE, alpha=0.35)
    ax.plot(dd.index, dd.values, color=NEGATIVE, lw=1)
    trough = dd.idxmin()
    ax.annotate(f"Max DD: {dd.min():.1%}", xy=(trough, dd.min()),
                xytext=(10, 10), textcoords="offset points", fontsize=10)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    _pct_axis(ax)
    return _save(fig, "02_drawdown")


# ---------------------------------------------------------------------------
# 3. Rolling volatility
# ---------------------------------------------------------------------------

def plot_rolling_volatility(port_rets: pd.Series, bench_rets: pd.Series | None,
                            window: int = 63) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m.rolling_volatility(port_rets, window).plot(ax=ax, color=ACCENT, lw=1.8, label="Portfolio")
    if bench_rets is not None:
        m.rolling_volatility(bench_rets, window).plot(ax=ax, color=GRAY, lw=1.5,
                                                       label="Benchmark", alpha=0.9)
    ax.set_title(f"Rolling {window}-Day Annualized Volatility")
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("")
    _pct_axis(ax)
    ax.legend(frameon=False)
    return _save(fig, "03_rolling_volatility")


# ---------------------------------------------------------------------------
# 4. Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(port_rets: pd.Series, window: int = 126,
                        risk_free_rate: float = 0.0) -> plt.Figure:
    rs = m.rolling_sharpe(port_rets, window, risk_free_rate)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(rs.index, rs.values, color=ACCENT, lw=1.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color=GRAY, lw=0.8, ls="--")
    ax.text(rs.index[-1], 1, " Sharpe = 1", va="center", fontsize=9, color=GRAY)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    return _save(fig, "04_rolling_sharpe")


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(asset_returns: pd.DataFrame) -> plt.Figure:
    corr = asset_returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Daily Return Correlations")
    return _save(fig, "05_correlation_heatmap")


# ---------------------------------------------------------------------------
# 6. Return distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(port_rets: pd.Series, confidence: float = 0.95) -> plt.Figure:
    var = m.value_at_risk(port_rets, confidence)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(port_rets, bins=80, kde=True, color=ACCENT, alpha=0.5, ax=ax)
    ax.axvline(-var, color=NEGATIVE, ls="--", lw=1.5)
    ax.text(-var, ax.get_ylim()[1] * 0.9,
            f" VaR {confidence:.0%}: {var:.2%}", color=NEGATIVE, fontsize=10)
    ax.set_title("Distribution of Daily Portfolio Returns")
    ax.set_xlabel("Daily return")
    _pct_axis(ax, "x")
    return _save(fig, "06_return_distribution")


# ---------------------------------------------------------------------------
# 7. Efficient frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(frontier: pd.DataFrame, current: dict[str, float],
                            optima: dict[str, pd.Series]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(frontier["volatility"], frontier["return"],
                    c=frontier["sharpe"], cmap="viridis", s=8, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="Sharpe ratio", shrink=0.85)
    ms, mv = optima["max_sharpe"], optima["min_volatility"]
    ax.scatter(ms["volatility"], ms["return"], marker="*", s=340,
               color="#e3b341", edgecolor="black", zorder=5, label="Max Sharpe")
    ax.scatter(mv["volatility"], mv["return"], marker="D", s=110,
               color="#2da44e", edgecolor="black", zorder=5, label="Min Volatility")
    ax.scatter(current["volatility"], current["return"], marker="X", s=180,
               color=NEGATIVE, edgecolor="black", zorder=5, label="Current Portfolio")
    ax.set_title("Monte Carlo Efficient Frontier (5,000 portfolios)")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    _pct_axis(ax, "x")
    _pct_axis(ax, "y")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "07_efficient_frontier")


# ---------------------------------------------------------------------------
# 8. Monthly returns heatmap
# ---------------------------------------------------------------------------

def plot_monthly_returns(port_rets: pd.Series) -> plt.Figure:
    monthly = (1 + port_rets).resample("ME").prod() - 1
    table = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.strftime("%b"),
        "Return": monthly.values,
    })
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = table.pivot(index="Year", columns="Month", values="Return")
    pivot = pivot.reindex(columns=[mo for mo in month_order if mo in pivot.columns])
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(pivot) + 2))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
                cbar=False, linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Monthly Portfolio Returns")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _save(fig, "08_monthly_returns")


# ---------------------------------------------------------------------------
# 9. Allocation drift
# ---------------------------------------------------------------------------

def plot_weight_drift(drift: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(drift.index, drift.T.values, labels=drift.columns,
                 colors=sns.color_palette("crest", n_colors=drift.shape[1]),
                 alpha=0.9)
    ax.set_title("Buy-and-Hold Allocation Drift")
    ax.set_ylabel("Portfolio weight")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    _pct_axis(ax)
    ax.legend(loc="upper left", frameon=False,
              ncol=min(4, drift.shape[1]), fontsize=9)
    return _save(fig, "09_weight_drift")
