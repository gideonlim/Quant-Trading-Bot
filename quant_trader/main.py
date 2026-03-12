#!/usr/bin/env python3
"""
Quant Trading Bot — Main Entry Point
=====================================

Usage:
    python -m quant_trader.main backtest              Run backtest on historical data
    python -m quant_trader.main trade --dry-run       Generate signals without executing
    python -m quant_trader.main trade                  Execute one trading cycle (paper)
    python -m quant_trader.main status                 Show portfolio status
    python -m quant_trader.main signals                Show current signals without trading

Requires environment variables:
    ALPACA_API_KEY       Your Alpaca API key
    ALPACA_SECRET_KEY    Your Alpaca secret key
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_trader import config
from quant_trader.alpaca_client import AlpacaClient
from quant_trader.indicators import compute_all_features
from quant_trader.strategy import Strategy
from quant_trader.backtest import BacktestEngine
from quant_trader.executor import TradeExecutor
from quant_trader.universe import get_dynamic_universe


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def check_api_keys() -> bool:
    """Verify Alpaca API keys are set."""
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        print("\n  ERROR: Alpaca API keys not found!")
        print("  Set them as environment variables:")
        print("    export ALPACA_API_KEY='your-key-here'")
        print("    export ALPACA_SECRET_KEY='your-secret-here'")
        print("\n  Get free keys at: https://app.alpaca.markets/signup\n")
        return False
    return True


# ── Backtest Command ──────────────────────────────────────────────────────────

def cmd_backtest(args):
    """Run historical backtest."""
    print("\n  Fetching historical data for backtest...")
    print(f"  Universe: {len(config.FALLBACK_UNIVERSE)} stocks")
    print(f"  Period: {args.start} to {args.end}\n")

    if not check_api_keys():
        print("  Running backtest with synthetic data for demonstration...\n")
        result = run_synthetic_backtest(args)
    else:
        client = AlpacaClient()
        start_fetch = (
            pd.Timestamp(args.start) - timedelta(days=config.DATA_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")

        # Fetch data
        symbols = config.FALLBACK_UNIVERSE[:args.universe_size]
        print(f"  Fetching {len(symbols)} symbols from Alpaca...")
        data = client.get_multi_bars(
            symbols=symbols,
            timeframe="1Day",
            start=start_fetch,
            end=args.end,
            limit=10000,
        )
        print(f"  Received data for {len(data)} symbols\n")

        if len(data) < 10:
            print("  WARNING: Not enough data. Check API keys and date range.\n")
            return

        # Run backtest
        strategy = Strategy()
        engine = BacktestEngine(strategy=strategy, initial_capital=config.TOTAL_CAPITAL)
        result = engine.run(data, start_date=args.start, end_date=args.end)

    # Print results
    print(result.summary())

    # Save equity curve plot
    plot_path = str(Path(args.output_dir) / "backtest_equity.png")
    result.plot_equity(save_path=plot_path)
    print(f"\n  Equity curve saved to: {plot_path}")

    # Save detailed trades
    if result.trades:
        trades_data = [
            {
                "symbol": t.symbol, "entry_date": str(t.entry_date.date()),
                "exit_date": str(t.exit_date.date()), "entry_price": t.entry_price,
                "exit_price": t.exit_price, "shares": t.shares, "pnl": t.pnl,
                "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason,
                "hold_days": t.hold_days,
            }
            for t in result.trades
        ]
        trades_path = str(Path(args.output_dir) / "backtest_trades.json")
        with open(trades_path, "w") as f:
            json.dump(trades_data, f, indent=2)
        print(f"  Trade log saved to: {trades_path}")

    # Save metrics
    metrics_path = str(Path(args.output_dir) / "backtest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}\n")


def run_synthetic_backtest(args):
    """
    Run a backtest with synthetic data when API keys aren't available.
    Generates realistic random-walk price data for demonstration.
    """
    np.random.seed(42)
    symbols = config.FALLBACK_UNIVERSE[:args.universe_size]

    start_fetch = pd.Timestamp(args.start) - timedelta(days=config.DATA_LOOKBACK_DAYS)
    dates = pd.bdate_range(start_fetch, args.end)

    data = {}
    for symbol in symbols:
        n = len(dates)
        # Random walk with slight upward drift (mimics equity market)
        drift = np.random.uniform(0.0001, 0.0005)
        vol = np.random.uniform(0.01, 0.03)
        returns = np.random.normal(drift, vol, n)

        # Add some momentum clustering
        momentum_regime = np.random.choice([1, -1], n, p=[0.55, 0.45])
        returns = returns * (1 + 0.3 * np.sign(np.convolve(momentum_regime, np.ones(10)/10, mode='same')))

        price = 100 * np.exp(np.cumsum(returns))
        high = price * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = price * (1 - np.abs(np.random.normal(0, 0.005, n)))
        volume = np.random.lognormal(15, 0.5, n).astype(int)

        df = pd.DataFrame({
            "open": price * (1 + np.random.normal(0, 0.002, n)),
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
        }, index=dates[:n])

        data[symbol] = df

    strategy = Strategy()
    engine = BacktestEngine(strategy=strategy, initial_capital=config.TOTAL_CAPITAL)
    return engine.run(data, start_date=args.start, end_date=args.end)


# ── Trade Command ─────────────────────────────────────────────────────────────

def cmd_trade(args):
    """Execute one trading cycle."""
    if not check_api_keys():
        return

    executor = TradeExecutor(
        log_dir=str(Path(args.output_dir) / "trade_logs"),
    )

    # Dynamic universe — executor will build it via get_dynamic_universe
    summary = executor.run_cycle(symbols=None, dry_run=args.dry_run)

    print(json.dumps(summary, indent=2, default=str))


# ── Status Command ────────────────────────────────────────────────────────────

def cmd_status(args):
    """Show current portfolio status."""
    if not check_api_keys():
        return

    executor = TradeExecutor()
    print(executor.get_portfolio_summary())


# ── Signals Command ───────────────────────────────────────────────────────────

def cmd_signals(args):
    """Show current strategy signals without trading."""
    if not check_api_keys():
        return

    client = AlpacaClient()

    # Build dynamic universe (or fall back to static list)
    print(f"\n  Building stock universe...")
    symbols = get_dynamic_universe(client, top_n=args.universe_size)
    print(f"  Universe: {len(symbols)} stocks selected by liquidity")
    print(f"  Fetching historical data...")
    start_date = (datetime.now() - timedelta(days=config.DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    data = client.get_multi_bars(symbols=symbols, timeframe="1Day", start=start_date)

    # Compute features
    enriched = {}
    total = len(data)
    for idx, (sym, df) in enumerate(data.items(), 1):
        pct = idx / total * 100
        bar_len = 30
        filled = int(bar_len * idx / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  Computing features: {bar} {idx}/{total} ({pct:.0f}%)")
        sys.stdout.flush()
        if len(df) >= config.TREND_SMA_PERIOD:
            enriched[sym] = compute_all_features(df)

    print(f"\n  Processed {len(enriched)} symbols\n")

    # Generate signals
    strategy = Strategy()
    acct = client.get_account()
    capital = float(acct.get("portfolio_value", config.TOTAL_CAPITAL))
    target = strategy.generate_signals(enriched, capital=capital)

    if not target.signals:
        print("  No signals generated (all stocks filtered out)\n")
        return

    # Display signals
    print("=" * 80)
    print("  CURRENT SIGNALS")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Symbol':<8} {'Score':>8} {'Mom1M':>8} {'Mom6M':>8} "
          f"{'RSI':>6} {'Entry':>10} {'SL':>10} {'TP':>10} {'Alloc':>7}")
    print("  " + "-" * 75)

    for i, sig in enumerate(target.signals, 1):
        print(
            f"  {i:<5} {sig.symbol:<8} {sig.score:>8.3f} "
            f"{sig.momentum_fast * 100:>7.1f}% {sig.momentum_slow * 100:>7.1f}% "
            f"{sig.rsi:>6.1f} ${sig.entry_price:>9.2f} "
            f"${sig.stop_loss:>9.2f} ${sig.take_profit:>9.2f} "
            f"{sig.position_size_pct * 100:>6.1f}%"
        )

    print("=" * 80)
    print(f"  Total signals: {target.num_positions}")
    print(f"  Portfolio value: ${capital:,.2f}\n")

    # Save signals to file
    signals_path = Path(args.output_dir) / "current_signals.json"
    signals_data = [
        {
            "rank": i, "symbol": s.symbol, "score": round(s.score, 4),
            "momentum_fast": round(s.momentum_fast, 4),
            "momentum_slow": round(s.momentum_slow, 4),
            "rsi": round(s.rsi, 1),
            "entry": s.entry_price, "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "allocation_pct": round(s.position_size_pct * 100, 2),
        }
        for i, s in enumerate(target.signals, 1)
    ]
    with open(signals_path, "w") as f:
        json.dump(signals_data, f, indent=2)
    print(f"  Signals saved to: {signals_path}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quant Trading Bot — Multi-Factor Momentum Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quant_trader.main backtest
  python -m quant_trader.main backtest --start 2024-01-01 --end 2025-12-31
  python -m quant_trader.main trade --dry-run
  python -m quant_trader.main trade
  python -m quant_trader.main signals
  python -m quant_trader.main status
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "-o", "--output-dir", default=".",
        help="Output directory for results (default: current directory)"
    )
    parser.add_argument(
        "--universe-size", type=int, default=50,
        help="Number of stocks in universe (default: 50)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument("--start", default=config.BACKTEST_START, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", default=config.BACKTEST_END, help="End date (YYYY-MM-DD)")

    # Trade
    trade_parser = subparsers.add_parser("trade", help="Execute one trading cycle")
    trade_parser.add_argument("--dry-run", action="store_true", help="Don't execute, just show")

    # Status
    subparsers.add_parser("status", help="Show portfolio status")

    # Signals
    subparsers.add_parser("signals", help="Show current signals")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "trade":
        cmd_trade(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "signals":
        cmd_signals(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
