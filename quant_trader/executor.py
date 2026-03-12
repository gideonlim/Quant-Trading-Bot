"""
Live execution engine for paper trading via Alpaca.
Bridges strategy signals to actual order management.
"""

import csv
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from . import config
from .alpaca_client import AlpacaClient
from .indicators import compute_all_features
from .strategy import Strategy, Signal, PortfolioTarget
from .universe import get_dynamic_universe

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of an executed order."""
    timestamp: str
    symbol: str
    side: str
    qty: int
    price: float
    order_type: str
    reason: str
    order_id: str = ""


class TradeExecutor:
    """
    Manages the full lifecycle of live/paper trading:
    1. Fetch latest market data
    2. Run strategy to generate signals
    3. Reconcile signals vs. current positions
    4. Execute orders (entries, exits, rebalances)
    5. Log everything
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        strategy: Strategy = None,
        log_dir: str = None,
    ):
        self.client = AlpacaClient(api_key=api_key, secret_key=secret_key, paper=True)
        self.strategy = strategy or Strategy()
        self.log_dir = Path(log_dir or "trade_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.execution_log: List[ExecutionRecord] = []

        # CSV trade ledger — persists in project root across all runs for the FY
        self.trade_ledger_path = Path(__file__).resolve().parent.parent / "trade_ledger.csv"

    # ── Data Fetching ─────────────────────────────────────────────────────────

    def fetch_universe_data(
        self, symbols: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for the trading universe.
        Returns dict of symbol -> DataFrame with features computed.
        """
        if not symbols:
            symbols = get_dynamic_universe(self.client)
            logger.info(f"Dynamic universe: {len(symbols)} stocks")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.now() - timedelta(days=config.DATA_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Fetching data for {len(symbols)} symbols ({start_date} to {end_date})")

        raw_data = self.client.get_multi_bars(
            symbols=symbols,
            timeframe="1Day",
            start=start_date,
            end=end_date,
            limit=1000,
        )

        import sys
        enriched = {}
        total = len(raw_data)
        for idx, (symbol, df) in enumerate(raw_data.items(), 1):
            pct = idx / total * 100
            bar_len = 30
            filled = int(bar_len * idx / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r  Computing features: {bar} {idx}/{total} ({pct:.0f}%)")
            sys.stdout.flush()

            if len(df) < config.TREND_SMA_PERIOD:
                logger.debug(f"Skipping {symbol}: only {len(df)} bars (need {config.TREND_SMA_PERIOD})")
                continue
            try:
                enriched[symbol] = compute_all_features(df)
            except Exception as e:
                logger.warning(f"Feature computation failed for {symbol}: {e}")

        sys.stdout.write("\n")
        sys.stdout.flush()
        logger.info(f"Data ready for {len(enriched)} / {len(symbols)} symbols")
        return enriched

    # ── Position Reconciliation ───────────────────────────────────────────────

    def get_current_state(self) -> Tuple[Dict[str, dict], float, float]:
        """
        Get current account state.
        Returns: (positions_dict, portfolio_value, cash)
        """
        positions = {}
        for pos in self.client.get_positions():
            positions[pos["symbol"]] = {
                "qty": int(pos["qty"]),
                "avg_entry": float(pos["avg_entry_price"]),
                "market_value": float(pos["market_value"]),
                "unrealized_pnl": float(pos["unrealized_pl"]),
                "current_price": float(pos["current_price"]),
            }

        acct = self.client.get_account()
        portfolio_value = float(acct["portfolio_value"])
        cash = float(acct["cash"])

        return positions, portfolio_value, cash

    def reconcile(
        self, target: PortfolioTarget, current_positions: Dict[str, dict],
        portfolio_value: float
    ) -> Tuple[List[dict], List[dict]]:
        """
        Compare target portfolio to current positions.
        Returns: (orders_to_close, orders_to_open)
        """
        target_symbols = {s.symbol for s in target.signals}
        current_symbols = set(current_positions.keys())

        # Positions to close: currently held but not in target
        to_close = []
        for sym in current_symbols - target_symbols:
            to_close.append({
                "symbol": sym,
                "side": "sell",
                "qty": current_positions[sym]["qty"],
                "reason": "not_in_target",
            })

        # Positions to open: in target but not currently held
        to_open = []
        signal_map = {s.symbol: s for s in target.signals}
        for sym in target_symbols - current_symbols:
            signal = signal_map[sym]
            dollar_allocation = portfolio_value * signal.position_size_pct
            price = signal.entry_price
            qty = int(dollar_allocation / price)
            if qty > 0:
                to_open.append({
                    "symbol": sym,
                    "side": "buy",
                    "qty": qty,
                    "price": price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "score": signal.score,
                    "reason": "new_signal",
                })

        return to_close, to_open

    # ── Order Execution ───────────────────────────────────────────────────────

    def execute_trades(
        self, to_close: List[dict], to_open: List[dict], use_brackets: bool = True
    ) -> List[ExecutionRecord]:
        """
        Execute the reconciled orders.
        Closes first, then opens (to free up capital).
        """
        records = []

        # ── Close positions ───────────────────────────────────────────
        for order in to_close:
            try:
                result = self.client.close_position(order["symbol"])
                rec = ExecutionRecord(
                    timestamp=datetime.now().isoformat(),
                    symbol=order["symbol"],
                    side="sell",
                    qty=order["qty"],
                    price=0,  # Market order, filled async
                    order_type="market_close",
                    reason=order["reason"],
                    order_id=result.get("id", ""),
                )
                records.append(rec)
                logger.info(f"CLOSE: {order['symbol']} x{order['qty']} ({order['reason']})")
            except Exception as e:
                logger.error(f"Failed to close {order['symbol']}: {e}")

        # ── Open new positions ────────────────────────────────────────
        for order in to_open:
            try:
                if use_brackets and order.get("stop_loss") and order.get("take_profit"):
                    result = self.client.submit_bracket_order(
                        symbol=order["symbol"],
                        qty=order["qty"],
                        side="buy",
                        take_profit_price=order["take_profit"],
                        stop_loss_price=order["stop_loss"],
                    )
                else:
                    result = self.client.submit_order(
                        symbol=order["symbol"],
                        qty=order["qty"],
                        side="buy",
                        order_type="market",
                    )

                rec = ExecutionRecord(
                    timestamp=datetime.now().isoformat(),
                    symbol=order["symbol"],
                    side="buy",
                    qty=order["qty"],
                    price=order.get("price", 0),
                    order_type="bracket" if use_brackets else "market",
                    reason=order["reason"],
                    order_id=result.get("id", ""),
                )
                records.append(rec)
                logger.info(
                    f"OPEN: {order['symbol']} x{order['qty']} "
                    f"(score={order.get('score', 0):.3f}, "
                    f"SL={order.get('stop_loss', 0):.2f}, "
                    f"TP={order.get('take_profit', 0):.2f})"
                )
            except Exception as e:
                logger.error(f"Failed to open {order['symbol']}: {e}")

        self.execution_log.extend(records)

        # Append to persistent CSV trade ledger
        if records:
            self._append_to_trade_ledger(records, to_close, to_open)

        return records

    # ── CSV Trade Ledger ──────────────────────────────────────────────────────

    LEDGER_COLUMNS = [
        "date", "time", "symbol", "side", "qty", "price",
        "order_type", "stop_loss", "take_profit", "score",
        "reason", "order_id",
    ]

    def _append_to_trade_ledger(
        self, records: List[ExecutionRecord],
        to_close: List[dict], to_open: List[dict],
    ):
        """
        Append executed trades to trade_ledger.csv.
        Creates the file with headers if it doesn't exist yet.
        """
        file_exists = self.trade_ledger_path.exists()

        # Build lookup for extra fields from order dicts
        open_map = {o["symbol"]: o for o in to_open}
        close_map = {o["symbol"]: o for o in to_close}

        try:
            with open(self.trade_ledger_path, "a", newline="") as f:
                writer = csv.writer(f)

                # Write header row if new file
                if not file_exists:
                    writer.writerow(self.LEDGER_COLUMNS)

                for rec in records:
                    ts = datetime.fromisoformat(rec.timestamp)
                    order_info = open_map.get(rec.symbol, close_map.get(rec.symbol, {}))

                    writer.writerow([
                        ts.strftime("%Y-%m-%d"),
                        ts.strftime("%H:%M:%S"),
                        rec.symbol,
                        rec.side,
                        rec.qty,
                        round(rec.price, 2),
                        rec.order_type,
                        order_info.get("stop_loss", ""),
                        order_info.get("take_profit", ""),
                        round(order_info.get("score", 0), 4) if order_info.get("score") else "",
                        rec.reason,
                        rec.order_id,
                    ])

            logger.info(
                f"Trade ledger updated: {len(records)} rows appended to {self.trade_ledger_path}"
            )
        except Exception as e:
            logger.error(f"Failed to write trade ledger: {e}")

    # ── Full Run Cycle ────────────────────────────────────────────────────────

    def run_cycle(self, symbols: List[str] = None, dry_run: bool = False) -> dict:
        """
        Run one complete trading cycle:
        1. Fetch data
        2. Generate signals
        3. Reconcile with current positions
        4. Execute trades (or log if dry_run)

        Returns a summary dict.
        """
        logger.info("=" * 60)
        logger.info("  STARTING TRADING CYCLE")
        logger.info("=" * 60)

        # Check market status
        try:
            clock = self.client.get_clock()
            is_open = clock.get("is_open", False)
            logger.info(f"Market open: {is_open}")
            if not is_open and not dry_run:
                logger.warning("Market is closed. Running in analysis-only mode.")
                dry_run = True
        except Exception as e:
            logger.warning(f"Could not check market status: {e}")

        # 1. Fetch data
        data = self.fetch_universe_data(symbols)
        if not data:
            logger.error("No data available. Aborting cycle.")
            return {"status": "error", "reason": "no_data"}

        # 2. Generate signals
        current_positions, portfolio_value, cash = self.get_current_state()
        logger.info(
            f"Account: Value=${portfolio_value:,.2f} | Cash=${cash:,.2f} | "
            f"Positions={len(current_positions)}"
        )

        target = self.strategy.generate_signals(data, capital=portfolio_value)
        logger.info(f"Strategy generated {target.num_positions} signals")

        # 3. Reconcile
        to_close, to_open = self.reconcile(target, current_positions, portfolio_value)
        logger.info(f"Reconciliation: {len(to_close)} to close, {len(to_open)} to open")

        # 4. Execute or log
        if dry_run:
            logger.info("DRY RUN — no orders submitted")
            summary = {
                "status": "dry_run",
                "portfolio_value": portfolio_value,
                "cash": cash,
                "current_positions": len(current_positions),
                "signals_generated": target.num_positions,
                "to_close": [o["symbol"] for o in to_close],
                "to_open": [(o["symbol"], o["qty"], o.get("score", 0)) for o in to_open],
            }
        else:
            records = self.execute_trades(to_close, to_open)
            summary = {
                "status": "executed",
                "portfolio_value": portfolio_value,
                "cash": cash,
                "orders_closed": len(to_close),
                "orders_opened": len(to_open),
                "execution_records": len(records),
            }

        # Save log
        self._save_cycle_log(summary, target)

        logger.info(f"Cycle complete: {summary['status']}")
        return summary

    def _save_cycle_log(self, summary: dict, target: PortfolioTarget):
        """Save cycle results to a JSON log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"cycle_{timestamp}.json"

        log_data = {
            "timestamp": timestamp,
            "summary": summary,
            "signals": [
                {
                    "symbol": s.symbol,
                    "score": s.score,
                    "direction": s.direction,
                    "momentum_fast": s.momentum_fast,
                    "momentum_slow": s.momentum_slow,
                    "rsi": s.rsi,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "position_pct": s.position_size_pct,
                }
                for s in target.signals
            ],
        }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        logger.info(f"Cycle log saved: {log_file}")

    # ── Monitoring ────────────────────────────────────────────────────────────

    def get_portfolio_summary(self) -> str:
        """Get a human-readable portfolio summary."""
        positions, portfolio_value, cash = self.get_current_state()

        lines = [
            "=" * 60,
            "  PORTFOLIO SUMMARY",
            "=" * 60,
            f"  Portfolio Value:  ${portfolio_value:>12,.2f}",
            f"  Cash:             ${cash:>12,.2f}",
            f"  Invested:         ${portfolio_value - cash:>12,.2f}",
            f"  Positions:        {len(positions):>12d}",
            "",
        ]

        if positions:
            lines.append("  ── Positions ──")
            lines.append(f"  {'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8}")
            lines.append("  " + "-" * 56)

            total_pnl = 0
            for sym, pos in sorted(positions.items()):
                pnl = pos["unrealized_pnl"]
                pnl_pct = (pos["current_price"] / pos["avg_entry"] - 1) * 100
                total_pnl += pnl
                lines.append(
                    f"  {sym:<8} {pos['qty']:>6d} "
                    f"${pos['avg_entry']:>9.2f} ${pos['current_price']:>9.2f} "
                    f"${pnl:>9.2f} {pnl_pct:>7.2f}%"
                )
            lines.append("  " + "-" * 56)
            lines.append(f"  {'Total P&L':>36} ${total_pnl:>9.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)
