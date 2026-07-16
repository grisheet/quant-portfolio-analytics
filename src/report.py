"""
report.py
---------
Generates a self-contained Markdown report (reports/analysis_report.md)
embedding the metric tables and figures produced by the pipeline.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

_PCT_COLS = {"Ann. Return", "Ann. Volatility", "Max Drawdown",
             "VaR 95% (1d)", "CVaR 95% (1d)", "Alpha (ann.)"}


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in _PCT_COLS:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        else:
            out[col] = out[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    return out


def generate_report(*, tickers: list[str], benchmark: str, start: str, end: str,
                    weights: pd.Series, rebalance: str | None,
                    portfolio_summary: pd.DataFrame, asset_summary: pd.DataFrame,
                    drawdown_info: dict, contribution: pd.Series,
                    optima: dict, data_source_note: str,
                    figures: list[str]) -> Path:
    """Write the Markdown report and return its path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    weight_lines = "\n".join(f"| {t} | {w:.1%} |" for t, w in weights.items())
    contrib_lines = "\n".join(f"| {t} | {c:+.2%} |" for t, c in contribution.items())
    fig_lines = "\n\n".join(f"![{Path(f).stem}](figures/{Path(f).name})" for f in figures)

    recovery = drawdown_info.get("recovery_date")
    recovery_str = recovery.date().isoformat() if recovery is not None else "not yet recovered"

    ms = optima["max_sharpe"]
    ms_weights = ", ".join(
        f"{c.replace('w_', '')}: {ms[c]:.0%}" for c in ms.index if c.startswith("w_")
    )

    md = f"""# Portfolio Risk & Performance Report

*Generated {ts}*

{data_source_note}

## Configuration

- **Assets:** {", ".join(tickers)}
- **Benchmark:** {benchmark}
- **Period:** {start} to {end}
- **Rebalancing:** {rebalance or "buy-and-hold"}

### Target Weights

| Asset | Weight |
|---|---|
{weight_lines}

## Headline Metrics - Portfolio vs Benchmark

{_fmt(portfolio_summary).to_markdown()}

## Per-Asset Metrics

{_fmt(asset_summary).to_markdown()}

## Worst Drawdown

- **Depth:** {drawdown_info.get("max_drawdown", float("nan")):.2%}
- **Peak:** {drawdown_info.get("peak_date").date()}
- **Trough:** {drawdown_info.get("trough_date").date()}
- **Recovery:** {recovery_str}

## Return Contribution by Asset

| Asset | Contribution |
|---|---|
{contrib_lines}

## Monte Carlo Optimization

Best Sharpe portfolio found across 5,000 random long-only allocations:
**Sharpe {ms["sharpe"]:.2f}** at {ms["return"]:.1%} return / {ms["volatility"]:.1%} volatility
({ms_weights}).

## Charts

{fig_lines}
"""
    path = REPORT_DIR / "analysis_report.md"
    path.write_text(md)
    return path
