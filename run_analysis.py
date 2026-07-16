"""
run_analysis.py
---------------
End-to-end pipeline entry point.

Examples:
    python run_analysis.py
    python run_analysis.py --tickers AAPL MSFT NVDA TLT GLD --benchmark SPY
    python run_analysis.py --weights AAPL=0.3 MSFT=0.3 NVDA=0.2 TLT=0.2 --rebalance Q
    python run_analysis.py --offline          # synthetic data, no network needed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from src import metrics as m
from src import visualization as viz
from src.data_loader import load_prices
from src.portfolio import (monte_carlo_frontier, normalize_weights,
                           optimal_portfolios, portfolio_returns,
                           portfolio_summary, return_contribution, weight_drift)
from src.preprocessing import clean_prices, validate_prices
from src.report import generate_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("pipeline")

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    p = argparse.ArgumentParser(description="Portfolio risk & performance analytics")
    p.add_argument("--tickers", nargs="+", default=cfg.get("tickers"))
    p.add_argument("--benchmark", default=cfg.get("benchmark", "SPY"))
    p.add_argument("--start", default=cfg.get("start", "2020-01-01"))
    p.add_argument("--end", default=cfg.get("end", "2025-12-31"))
    p.add_argument("--weights", nargs="+", default=None,
                   help="e.g. AAPL=0.4 MSFT=0.6 (default: equal weight)")
    p.add_argument("--rebalance", choices=["M", "Q", "Y", "none"],
                   default=cfg.get("rebalance", "M"))
    p.add_argument("--risk-free-rate", type=float, default=cfg.get("risk_free_rate", 0.04))
    p.add_argument("--offline", action="store_true",
                   help="Skip network; use cached or synthetic data")
    args = p.parse_args()

    if args.tickers is None:
        args.tickers = ["AAPL", "MSFT", "NVDA", "JPM", "TLT", "GLD"]
    args.tickers = [t.upper() for t in args.tickers]

    if args.weights:
        pairs = dict(w.split("=") for w in args.weights)
        args.weights = {k.upper(): float(v) for k, v in pairs.items()}
    else:
        cfg_weights = cfg.get("weights")
        args.weights = ({k.upper(): float(v) for k, v in cfg_weights.items()}
                        if cfg_weights else
                        {t: 1.0 / len(args.tickers) for t in args.tickers})
    return args


def main() -> None:
    args = parse_args()
    rebalance = None if args.rebalance == "none" else args.rebalance
    all_tickers = sorted(set(args.tickers) | {args.benchmark})

    # 1. Acquire ------------------------------------------------------------
    logger.info("Step 1/6: loading prices for %s", all_tickers)
    prices = load_prices(all_tickers, args.start, args.end, offline=args.offline)
    is_synthetic = args.offline or "yfinance" not in sys.modules

    # 2. Clean --------------------------------------------------------------
    logger.info("Step 2/6: cleaning and validating data")
    prices, _cleaning = clean_prices(prices)
    validate_prices(prices)

    # 3. Transform ----------------------------------------------------------
    logger.info("Step 3/6: computing returns")
    asset_rets = m.daily_returns(prices)
    bench_rets = asset_rets[args.benchmark]
    weights = normalize_weights(args.weights, list(asset_rets.columns))
    port_rets = portfolio_returns(asset_rets, weights, rebalance)

    # 4. Analyze ------------------------------------------------------------
    logger.info("Step 4/6: computing metrics")
    port_summary = portfolio_summary(port_rets, bench_rets, args.risk_free_rate)
    asset_summary = m.summary_table(asset_rets, args.benchmark, args.risk_free_rate)
    dd_info = m.drawdown_details(port_rets)
    contribution = return_contribution(asset_rets, weights)
    frontier = monte_carlo_frontier(asset_rets[weights.index],
                                    risk_free_rate=args.risk_free_rate)
    optima = optimal_portfolios(frontier)

    print("\n=== Portfolio vs Benchmark ===")
    print(port_summary.round(3).to_string())

    # 5. Visualize ----------------------------------------------------------
    logger.info("Step 5/6: rendering charts")
    viz.plot_cumulative_returns(port_rets, bench_rets, args.benchmark)
    viz.plot_drawdown(port_rets)
    viz.plot_rolling_volatility(port_rets, bench_rets)
    viz.plot_rolling_sharpe(port_rets, risk_free_rate=args.risk_free_rate)
    viz.plot_correlation_heatmap(asset_rets[weights.index])
    viz.plot_return_distribution(port_rets)
    current_point = {
        "volatility": m.annualized_volatility(port_rets),
        "return": m.annualized_return(port_rets),
    }
    viz.plot_efficient_frontier(frontier, current_point, optima)
    viz.plot_monthly_returns(port_rets)
    viz.plot_weight_drift(weight_drift(asset_rets, weights))

    # 6. Report -------------------------------------------------------------
    logger.info("Step 6/6: writing report")
    note = ("> **Note:** generated from *synthetic* demo data (seeded GBM). "
            "Run without `--offline` and with `yfinance` installed for real market data."
            if is_synthetic else "")
    figures = sorted(str(f) for f in viz.FIG_DIR.glob("*.png"))
    report_path = generate_report(
        tickers=args.tickers, benchmark=args.benchmark, start=args.start,
        end=args.end, weights=weights, rebalance=rebalance,
        portfolio_summary=port_summary, asset_summary=asset_summary,
        drawdown_info=dd_info, contribution=contribution, optima=optima,
        data_source_note=note, figures=figures,
    )
    logger.info("Done. Report: %s | Figures: %s", report_path, viz.FIG_DIR)


if __name__ == "__main__":
    main()
