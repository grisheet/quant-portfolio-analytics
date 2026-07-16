"""Unit tests for src/metrics.py — run with `pytest`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import metrics as m
from src.data_loader import generate_synthetic_prices
from src.portfolio import normalize_weights, portfolio_returns
from src.preprocessing import clean_prices


@pytest.fixture
def simple_returns() -> pd.Series:
    dates = pd.bdate_range("2023-01-02", periods=5)
    return pd.Series([0.10, -0.05, 0.02, 0.00, 0.03], index=dates)


def test_cumulative_returns(simple_returns):
    cum = m.cumulative_returns(simple_returns)
    expected = 1.10 * 0.95 * 1.02 * 1.00 * 1.03
    assert np.isclose(cum.iloc[-1], expected)


def test_annualized_return_flat():
    dates = pd.bdate_range("2023-01-02", periods=252)
    daily = (1.10) ** (1 / 252) - 1  # exactly 10% over one trading year
    r = pd.Series(daily, index=dates)
    assert np.isclose(m.annualized_return(r), 0.10, atol=1e-10)


def test_max_drawdown_known_path():
    # 100 -> 120 -> 90 -> 110 : max drawdown = 90/120 - 1 = -25%
    prices = pd.Series([100, 120, 90, 110],
                       index=pd.bdate_range("2023-01-02", periods=4))
    r = m.daily_returns(prices)
    assert np.isclose(m.max_drawdown(r), -0.25)


def test_sharpe_zero_for_zero_excess():
    dates = pd.bdate_range("2023-01-02", periods=252)
    rf = 0.04
    rf_daily = (1 + rf) ** (1 / 252) - 1
    r = pd.Series(rf_daily, index=dates) + pd.Series(
        np.random.default_rng(0).normal(0, 1e-9, 252), index=dates)
    assert abs(m.sharpe_ratio(r, rf)) < 0.5  # ~zero excess return


def test_var_cvar_ordering(simple_returns):
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 1000),
                  index=pd.bdate_range("2020-01-01", periods=1000))
    var = m.value_at_risk(r)
    cvar = m.conditional_var(r)
    assert cvar >= var > 0  # expected shortfall is always at least VaR


def test_beta_of_benchmark_is_one():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 500),
                  index=pd.bdate_range("2020-01-01", periods=500))
    assert np.isclose(m.beta(r, r), 1.0)


def test_portfolio_weights_normalize():
    w = normalize_weights({"aapl": 2, "msft": 2}, ["AAPL", "MSFT"])
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["AAPL"], 0.5)


def test_portfolio_returns_shapes():
    prices = generate_synthetic_prices(["AAPL", "TLT"], "2022-01-01", "2023-12-31")
    prices, _ = clean_prices(prices)
    rets = m.daily_returns(prices)
    w = normalize_weights({"AAPL": 0.6, "TLT": 0.4}, list(rets.columns))
    for reb in ["M", "Q", None]:
        pr = portfolio_returns(rets, w, reb)
        assert len(pr) > 0 and pr.isna().sum() == 0


def test_cleaning_removes_bad_prices():
    dates = pd.bdate_range("2023-01-02", periods=10)
    df = pd.DataFrame(
        {"A": np.linspace(100, 110, 10), "B": np.linspace(50, 55, 10)},
        index=dates,
    )
    df.iloc[3, 0] = 0      # bad print
    df.iloc[5, 1] = np.nan  # gap
    cleaned, report = clean_prices(df)
    assert cleaned.isna().sum().sum() == 0
    assert report.zero_or_negative_prices_fixed == 1
