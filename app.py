"""
app.py
------
Interactive Streamlit dashboard reusing the exact same analytics modules
as the CLI pipeline — no duplicated logic.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src import metrics as m
from src import visualization as viz
from src.data_loader import load_prices
from src.portfolio import (monte_carlo_frontier, normalize_weights,
                           optimal_portfolios, portfolio_returns,
                           portfolio_summary, weight_drift)
from src.preprocessing import clean_prices, validate_prices

st.set_page_config(page_title="Portfolio Analytics", page_icon="📈", layout="wide")

UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "JNJ", "XOM",
            "PG", "TLT", "AGG", "GLD", "QQQ", "IWM"]


@st.cache_data(show_spinner="Loading market data...")
def get_data(tickers: tuple[str, ...], start: str, end: str, offline: bool) -> pd.DataFrame:
    prices = load_prices(list(tickers), start, end, offline=offline)
    prices, _ = clean_prices(prices)
    validate_prices(prices)
    return prices


# --- Sidebar controls -------------------------------------------------------
st.sidebar.title("Portfolio Setup")
tickers = st.sidebar.multiselect("Assets", UNIVERSE,
                                 default=["AAPL", "MSFT", "NVDA", "TLT", "GLD"])
benchmark = st.sidebar.selectbox("Benchmark", ["SPY", "QQQ", "VTI"], index=0)
start = st.sidebar.date_input("Start", pd.Timestamp("2020-01-01")).isoformat()
end = st.sidebar.date_input("End", pd.Timestamp("2025-12-31")).isoformat()
rebalance = st.sidebar.selectbox("Rebalancing", ["M", "Q", "Y", "none"], index=0)
rf = st.sidebar.slider("Risk-free rate", 0.0, 0.08, 0.04, 0.005, format="%.3f")
offline = st.sidebar.toggle("Offline demo mode (synthetic data)", value=False)

st.sidebar.subheader("Weights")
raw_weights = {t: st.sidebar.slider(t, 0.0, 1.0, round(1 / max(len(tickers), 1), 2), 0.05)
               for t in tickers}

if len(tickers) < 2:
    st.info("Select at least two assets to begin.")
    st.stop()

# --- Pipeline ----------------------------------------------------------------
all_tickers = tuple(sorted(set(tickers) | {benchmark}))
prices = get_data(all_tickers, start, end, offline)
asset_rets = m.daily_returns(prices)
bench_rets = asset_rets[benchmark]
weights = normalize_weights(raw_weights, list(asset_rets.columns))
port_rets = portfolio_returns(asset_rets, weights, None if rebalance == "none" else rebalance)

# --- Header metrics ----------------------------------------------------------
st.title("📈 Portfolio Risk & Performance Analytics")
if offline:
    st.warning("Offline mode: charts below use synthetic demo data, not real prices.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ann. Return", f"{m.annualized_return(port_rets):.1%}")
c2.metric("Ann. Volatility", f"{m.annualized_volatility(port_rets):.1%}")
c3.metric("Sharpe", f"{m.sharpe_ratio(port_rets, rf):.2f}")
c4.metric("Max Drawdown", f"{m.max_drawdown(port_rets):.1%}")
c5.metric("Beta vs " + benchmark, f"{m.beta(port_rets, bench_rets):.2f}")

# --- Tabs ---------------------------------------------------------------------
tab_perf, tab_risk, tab_corr, tab_opt = st.tabs(
    ["Performance", "Risk", "Diversification", "Optimization"]
)

with tab_perf:
    st.pyplot(viz.plot_cumulative_returns(port_rets, bench_rets, benchmark))
    st.pyplot(viz.plot_monthly_returns(port_rets))
    st.subheader("Portfolio vs Benchmark")
    st.dataframe(portfolio_summary(port_rets, bench_rets, rf).style.format("{:.3f}"))

with tab_risk:
    st.pyplot(viz.plot_drawdown(port_rets))
    st.pyplot(viz.plot_rolling_volatility(port_rets, bench_rets))
    st.pyplot(viz.plot_return_distribution(port_rets))

with tab_corr:
    st.pyplot(viz.plot_correlation_heatmap(asset_rets[weights.index]))
    st.pyplot(viz.plot_weight_drift(weight_drift(asset_rets, weights)))

with tab_opt:
    frontier = monte_carlo_frontier(asset_rets[weights.index], risk_free_rate=rf)
    optima = optimal_portfolios(frontier)
    current = {"volatility": m.annualized_volatility(port_rets),
               "return": m.annualized_return(port_rets)}
    st.pyplot(viz.plot_efficient_frontier(frontier, current, optima))
    ms = optima["max_sharpe"]
    st.subheader("Suggested Max-Sharpe Weights")
    ms_w = {c.replace("w_", ""): ms[c] for c in ms.index if c.startswith("w_")}
    st.dataframe(pd.Series(ms_w, name="Weight").to_frame().style.format("{:.1%}"))
