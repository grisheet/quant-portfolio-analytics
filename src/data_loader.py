"""
data_loader.py
--------------
Data acquisition layer.

Priority order:
1. Local cache (data/cache/*.csv) - avoids re-downloading and rate limits.
2. yfinance download (adjusted close prices).
3. Synthetic data generator (clearly labeled) - guarantees the project runs
   offline for demos, tests, and CI.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_prices(tickers: list[str], start: str, end: str,
                use_cache: bool = True, offline: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of adjusted close prices (index = dates, columns = tickers).

    Set offline=True to skip the network entirely and use cached or synthetic data.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / _cache_key(tickers, start, end)

    if use_cache and cache_file.exists():
        logger.info("Loading prices from cache: %s", cache_file.name)
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if not offline:
        try:
            prices = _download_yfinance(tickers, start, end)
            prices.to_csv(cache_file)
            return prices
        except Exception as exc:  # network failure, missing package, bad ticker
            logger.warning("Download failed (%s). Falling back to synthetic data.", exc)

    logger.warning("Using SYNTHETIC data - results are for demonstration only.")
    prices = generate_synthetic_prices(tickers, start, end)
    return prices


# ---------------------------------------------------------------------------
# yfinance download
# ---------------------------------------------------------------------------

def _download_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the project runs without it
    logger.info("Downloading %d tickers from Yahoo Finance...", len(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # single-ticker case
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("No data returned - check tickers and date range.")
    return prices


# ---------------------------------------------------------------------------
# Synthetic data (geometric Brownian motion with realistic correlations)
# ---------------------------------------------------------------------------

# Rough annual drift / volatility presets keyed by asset "personality".
_PRESETS = {
    "index": (0.09, 0.16),
    "tech":  (0.14, 0.30),
    "value": (0.08, 0.20),
    "bond":  (0.03, 0.07),
    "gold":  (0.05, 0.15),
}

_TICKER_STYLE = {
    "SPY": "index", "QQQ": "tech",   "VTI": "index", "IWM": "value",
    "AAPL": "tech", "MSFT": "tech",  "NVDA": "tech",  "GOOGL": "tech",
    "AMZN": "tech", "JPM": "value",  "JNJ": "value",  "XOM": "value",
    "PG": "value",  "TLT": "bond",   "AGG": "bond",   "GLD": "gold",
}


def generate_synthetic_prices(tickers: list[str], start: str, end: str,
                               seed: int = 42) -> pd.DataFrame:
    """
    Correlated GBM price paths.
    Deterministic (seeded) so demo outputs are reproducible.
    Clearly synthetic - never present these as real market results.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days, n_assets = len(dates), len(tickers)

    mu = np.empty(n_assets)
    sigma = np.empty(n_assets)
    for i, t in enumerate(tickers):
        style = _TICKER_STYLE.get(t.upper(), "value")
        mu[i], sigma[i] = _PRESETS[style]

    # Correlation: equities correlate ~0.6 with each other, bonds slightly
    # negative with equities, gold near zero.
    corr = np.full((n_assets, n_assets), 0.6)
    np.fill_diagonal(corr, 1.0)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            si = _TICKER_STYLE.get(ti.upper(), "value")
            sj = _TICKER_STYLE.get(tj.upper(), "value")
            if i != j and "bond" in (si, sj) and si != sj:
                corr[i, j] = -0.2
            elif i != j and "gold" in (si, sj) and si != sj:
                corr[i, j] = 0.1

    chol = np.linalg.cholesky(_nearest_psd(corr))
    dt = 1.0 / 252.0
    z = rng.standard_normal((n_days, n_assets)) @ chol.T
    daily = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_prices = np.cumsum(daily, axis=0)
    prices = 100.0 * np.exp(log_prices)
    return pd.DataFrame(prices, index=dates, columns=[t.upper() for t in tickers])


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so Cholesky always succeeds."""
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 1e-8, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    return f"{ '-'.join(sorted(t.upper() for t in tickers))}_{start}_{end}.csv"
