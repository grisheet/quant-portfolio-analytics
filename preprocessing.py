"""
preprocessing.py
----------------
Data cleaning, alignment, and validation.

Real market data is messy: tickers list on different dates, exchanges have
different holidays, and providers occasionally return gaps or zero prices.
This module turns raw downloads into an analysis-ready price panel and
reports exactly what it changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Audit trail of every transformation applied to the raw data."""
    initial_rows: int = 0
    final_rows: int = 0
    dropped_all_nan_rows: int = 0
    forward_filled_cells: int = 0
    dropped_leading_nan_rows: int = 0
    zero_or_negative_prices_fixed: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Rows: {self.initial_rows} -> {self.final_rows}",
            f"Dropped all-NaN rows: {self.dropped_all_nan_rows}",
            f"Dropped leading rows before all assets trade: {self.dropped_leading_nan_rows}",
            f"Forward-filled cells (holiday/gap alignment): {self.forward_filled_cells}",
            f"Zero/negative prices repaired: {self.zero_or_negative_prices_fixed}",
        ]
        return "\n".join(lines + self.notes)


def clean_prices(prices: pd.DataFrame, max_ffill: int = 5) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a raw price panel:

    1. Sort the index and drop duplicate dates.
    2. Replace zero/negative prices (bad data) with NaN.
    3. Drop rows where every asset is NaN (non-trading days).
    4. Trim leading rows until every asset has started trading.
    5. Forward-fill short gaps (holidays, missing prints) up to `max_ffill` days.
    6. Drop any remaining NaN rows.

    Returns the cleaned frame plus a CleaningReport for transparency.
    """
    report = CleaningReport(initial_rows=len(prices))
    df = prices.copy()

    df = df[~df.index.duplicated(keep="first")].sort_index()

    bad = (df <= 0).sum().sum()
    if bad:
        report.zero_or_negative_prices_fixed = int(bad)
        df = df.where(df > 0)

    before = len(df)
    df = df.dropna(how="all")
    report.dropped_all_nan_rows = before - len(df)

    first_valid = df.apply(lambda col: col.first_valid_index()).max()
    before = len(df)
    df = df.loc[first_valid:]
    report.dropped_leading_nan_rows = before - len(df)
    if report.dropped_leading_nan_rows:
        report.notes.append(
            f"Note: analysis starts {first_valid.date()} — earliest date all assets trade."
        )

    nan_before = int(df.isna().sum().sum())
    df = df.ffill(limit=max_ffill)
    report.forward_filled_cells = nan_before - int(df.isna().sum().sum())

    df = df.dropna()
    report.final_rows = len(df)

    logger.info("Cleaning complete:\n%s", report.summary())
    return df, report


def validate_prices(prices: pd.DataFrame) -> None:
    """Fail fast if the cleaned panel is unusable for analysis."""
    if prices.empty:
        raise ValueError("Price panel is empty after cleaning.")
    if len(prices) < 60:
        raise ValueError(
            f"Only {len(prices)} rows of data — need at least 60 trading days "
            "for meaningful rolling statistics."
        )
    if prices.isna().any().any():
        raise ValueError("NaNs remain after cleaning — inspect the raw data.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price index is not sorted by date.")
