"""
Event-driven backtesting engine.
Simulates the strategy on historical data with realistic constraints.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from . import config
from .indicators import compute_all_features
from .strategy import Strategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed trade."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    side: str               # "long"
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_days: int


@dataclass
class Position:
    """An active position in the backtest."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    side: str
    peak_price: float       # Highest price since entry (for trailing stop)
    atr_at_entry: float
    target_weight: float


class BacktestEngine:
    """
    Vectorized + event-driven hybrid backtester.

    Workflow:
    1. Pre-compute all features for the entire history.
    2. Walk forward day by day.
    3. Every REBALANCE_DAYS, re-run the strategy to get new targets.
    4. Between rebalances, check exit conditions daily.
    """

    def __init__(self, strategy: Strategy = None, initial_capital: float = None):
        self.strategy = strategy or Strategy()
        self.initial_capital = initial_capital or config.TOTAL_CAPITAL
        self.slippage_bps = config.SLIPPAGE_BPS
        self.commission_per_share = config.COMMISSION_PER_SHARE

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None,
    ) -> "BacktestResult":
        """
        Run the backtest.

        Args:
            data: Dict of symbol -> OHLCV DataFrame (raw, features will be computed)
            start_date: Start of backtest period (YYYY-MM-DD)
            end_date: End of backtest period (YYYY-MM-DD)

        Returns:
            BacktestResult with equity curve, trades, and performance metrics.
        """
        start = pd.Timestamp(start_date or config.BACKTEST_START)
        end = pd.Timestamp(end_date or config.BACKTEST_END)

        logger.info(f"Starting backtest: {start.date()} to {end.date()}")
        logger.info(f"Universe: {len(data)} stocks | Capital: ${self.initial_capital:,.0f}")

        # ── Step 1: Compute features for all stocks ───────────────────────
        enriched_data = {}
        for symbol, df in data.items():
            if len(df) < config.TREND_SMA_PERIOD + 50:
                continue
            try:
                enriched_data[symbol] = compute_all_features(df.copy())
            except Exception as e:
                logger.warning(f"Failed to compute features for {symbol}: {e}")

        logger.info(f"Features computed for {len(enriched_data)} stocks")

        # ── Step 2: Get trading dates ─────────────────────────────────────
        # Use a reference stock to get the trading calendar
        ref_symbol = next(iter(enriched_data))
        all_dates = enriched_data[ref_symbol].index
        trading_dates = all_dates[(all_dates >= start) & (all_dates <= end)]

        if trading_dates.empty:
            raise ValueError(f"No trading dates in range {start} to {end}")

        # ── Step 3: Walk-forward simulation ───────────────────────────────
        cash = self.initial_capital
        positions: Dict[str, Position] = {}
        trades: List[Trade] = []
        equity_curve = []
        days_since_rebalance = config.REBALANCE_DAYS  # Force rebalance on first day

        for date in trading_dates:
            # Get current prices
            current_prices = {}
            for symbol, df in enriched_data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date]

            # ── Update peak prices for trailing stops ─────────────────
            for sym, pos in positions.items():
                if sym in current_prices:
                    price = current_prices[sym]["close"]
                    pos.peak_price = max(pos.peak_price, price)

            # ── Check exit conditions ─────────────────────────────────
            exits_to_process = []
            for sym, pos in positions.items():
                if sym not in current_prices:
                    continue
                cur_price = current_prices[sym]["close"]
                days_held = len(trading_dates[
                    (trading_dates >= pos.entry_date) & (trading_dates <= date)
                ])

                should_exit, reason = self.strategy.should_exit(
                    sym, pos.entry_price, cur_price,
                    pos.peak_price, pos.atr_at_entry, days_held
                )
                if should_exit:
                    exits_to_process.append((sym, cur_price, reason, days_held))

            # Process exits
            for sym, exit_price, reason, days_held in exits_to_process:
                pos = positions[sym]
                # Apply slippage
                adj_exit = exit_price * (1 - self.slippage_bps / 10000)
                pnl = (adj_exit - pos.entry_price) * pos.shares
                pnl -= self.commission_per_share * pos.shares * 2  # Entry + exit

                trade = Trade(
                    symbol=sym,
                    entry_date=pos.entry_date,
                    exit_date=date,
                    entry_price=pos.entry_price,
                    exit_price=round(adj_exit, 2),
                    shares=pos.shares,
                    side="long",
                    pnl=round(pnl, 2),
                    pnl_pct=round((adj_exit / pos.entry_price - 1) * 100, 2),
                    exit_reason=reason,
                    hold_days=days_held,
                )
                trades.append(trade)
                cash += pos.shares * adj_exit
                del positions[sym]

            # ── Rebalance check ───────────────────────────────────────
            days_since_rebalance += 1
            if days_since_rebalance >= config.REBALANCE_DAYS:
                days_since_rebalance = 0

                # Get portfolio value
                portfolio_value = cash
                for sym, pos in positions.items():
                    if sym in current_prices:
                        portfolio_value += pos.shares * current_prices[sym]["close"]

                # Check max drawdown circuit breaker
                if portfolio_value < self.initial_capital * (1 - config.MAX_DRAWDOWN_PCT):
                    logger.warning(
                        f"Drawdown breaker triggered at {date.date()}: "
                        f"${portfolio_value:,.0f} < ${self.initial_capital * (1 - config.MAX_DRAWDOWN_PCT):,.0f}"
                    )
                    # Close all positions
                    for sym in list(positions.keys()):
                        if sym in current_prices:
                            pos = positions[sym]
                            exit_p = current_prices[sym]["close"] * (1 - self.slippage_bps / 10000)
                            pnl = (exit_p - pos.entry_price) * pos.shares
                            trades.append(Trade(
                                symbol=sym, entry_date=pos.entry_date,
                                exit_date=date, entry_price=pos.entry_price,
                                exit_price=round(exit_p, 2), shares=pos.shares,
                                side="long", pnl=round(pnl, 2),
                                pnl_pct=round((exit_p / pos.entry_price - 1) * 100, 2),
                                exit_reason="drawdown_breaker", hold_days=0,
                            ))
                            cash += pos.shares * exit_p
                            del positions[sym]
                    continue

                # Generate new signals
                target = self.strategy.generate_signals(
                    enriched_data, as_of_date=date, capital=portfolio_value
                )

                # Determine which positions to close (not in new targets)
                target_symbols = set(target.symbols)
                for sym in list(positions.keys()):
                    if sym not in target_symbols and sym in current_prices:
                        pos = positions[sym]
                        exit_p = current_prices[sym]["close"] * (1 - self.slippage_bps / 10000)
                        pnl = (exit_p - pos.entry_price) * pos.shares
                        days_held = len(trading_dates[
                            (trading_dates >= pos.entry_date) & (trading_dates <= date)
                        ])
                        trades.append(Trade(
                            symbol=sym, entry_date=pos.entry_date,
                            exit_date=date, entry_price=pos.entry_price,
                            exit_price=round(exit_p, 2), shares=pos.shares,
                            side="long", pnl=round(pnl, 2),
                            pnl_pct=round((exit_p / pos.entry_price - 1) * 100, 2),
                            exit_reason="rebalance_exit", hold_days=days_held,
                        ))
                        cash += pos.shares * exit_p
                        del positions[sym]

                # Open new positions
                for signal in target.signals:
                    sym = signal.symbol
                    if sym in positions:
                        continue  # Already holding
                    if sym not in current_prices:
                        continue

                    entry_price = current_prices[sym]["close"] * (1 + self.slippage_bps / 10000)
                    allocation = portfolio_value * signal.position_size_pct
                    shares = int(allocation / entry_price)

                    if shares <= 0:
                        continue
                    cost = shares * entry_price + self.commission_per_share * shares
                    if cost > cash:
                        shares = int((cash - 100) / entry_price)  # Leave $100 buffer
                        if shares <= 0:
                            continue
                        cost = shares * entry_price

                    positions[sym] = Position(
                        symbol=sym,
                        entry_date=date,
                        entry_price=round(entry_price, 2),
                        shares=shares,
                        side="long",
                        peak_price=entry_price,
                        atr_at_entry=signal.atr,
                        target_weight=signal.position_size_pct,
                    )
                    cash -= cost

            # ── Record equity ─────────────────────────────────────────
            portfolio_value = cash
            for sym, pos in positions.items():
                if sym in current_prices:
                    portfolio_value += pos.shares * current_prices[sym]["close"]

            equity_curve.append({
                "date": date,
                "equity": round(portfolio_value, 2),
                "cash": round(cash, 2),
                "num_positions": len(positions),
            })

        # Close remaining positions at end
        for sym, pos in list(positions.items()):
            if sym in current_prices:
                exit_p = current_prices[sym]["close"]
                pnl = (exit_p - pos.entry_price) * pos.shares
                trades.append(Trade(
                    symbol=sym, entry_date=pos.entry_date,
                    exit_date=trading_dates[-1], entry_price=pos.entry_price,
                    exit_price=round(exit_p, 2), shares=pos.shares,
                    side="long", pnl=round(pnl, 2),
                    pnl_pct=round((exit_p / pos.entry_price - 1) * 100, 2),
                    exit_reason="backtest_end", hold_days=0,
                ))

        equity_df = pd.DataFrame(equity_curve).set_index("date")
        return BacktestResult(
            equity_curve=equity_df,
            trades=trades,
            initial_capital=self.initial_capital,
        )


class BacktestResult:
    """Container for backtest results with performance analytics."""

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Trade],
        initial_capital: float,
    ):
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self._metrics = None

    @property
    def metrics(self) -> dict:
        """Compute comprehensive performance metrics."""
        if self._metrics is not None:
            return self._metrics

        eq = self.equity_curve["equity"]
        returns = eq.pct_change().dropna()

        # Basic returns
        total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        days = (eq.index[-1] - eq.index[0]).days
        years = days / 365.25
        cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 0.01)) - 1) * 100

        # Risk
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
        sharpe = (returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0

        # Drawdown
        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        max_dd = drawdown.min() * 100
        max_dd_date = drawdown.idxmin()

        # Sortino
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = (returns.mean() * 252 / downside_vol) if downside_vol > 0 else 0

        # Calmar
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

        # Trade stats
        if self.trades:
            trade_pnls = [t.pnl for t in self.trades]
            winners = [t for t in self.trades if t.pnl > 0]
            losers = [t for t in self.trades if t.pnl <= 0]
            win_rate = len(winners) / len(self.trades) * 100

            avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
            avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
            profit_factor = (
                sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers))
                if losers and sum(t.pnl for t in losers) != 0 else float("inf")
            )
            avg_hold = np.mean([t.hold_days for t in self.trades])
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0
            trade_pnls = []

        self._metrics = {
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "annual_volatility_pct": round(annual_vol, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
            "total_trades": len(self.trades),
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_hold_days": round(avg_hold, 1),
            "final_equity": round(eq.iloc[-1], 2),
            "total_pnl": round(sum(trade_pnls), 2),
        }
        return self._metrics

    def summary(self) -> str:
        """Human-readable performance summary."""
        m = self.metrics
        lines = [
            "=" * 60,
            "  BACKTEST PERFORMANCE SUMMARY",
            "=" * 60,
            f"  Period:             {self.equity_curve.index[0].date()} to {self.equity_curve.index[-1].date()}",
            f"  Initial Capital:    ${self.initial_capital:>12,.2f}",
            f"  Final Equity:       ${m['final_equity']:>12,.2f}",
            f"  Total P&L:          ${m['total_pnl']:>12,.2f}",
            "",
            "  ── Returns ──",
            f"  Total Return:       {m['total_return_pct']:>10.2f}%",
            f"  CAGR:               {m['cagr_pct']:>10.2f}%",
            f"  Annual Volatility:  {m['annual_volatility_pct']:>10.2f}%",
            "",
            "  ── Risk-Adjusted ──",
            f"  Sharpe Ratio:       {m['sharpe_ratio']:>10.3f}",
            f"  Sortino Ratio:      {m['sortino_ratio']:>10.3f}",
            f"  Calmar Ratio:       {m['calmar_ratio']:>10.3f}",
            f"  Max Drawdown:       {m['max_drawdown_pct']:>10.2f}%",
            f"  Max DD Date:        {m['max_drawdown_date']:>10s}",
            "",
            "  ── Trade Stats ──",
            f"  Total Trades:       {m['total_trades']:>10d}",
            f"  Win Rate:           {m['win_rate_pct']:>10.1f}%",
            f"  Avg Win:            {m['avg_win_pct']:>10.2f}%",
            f"  Avg Loss:           {m['avg_loss_pct']:>10.2f}%",
            f"  Profit Factor:      {m['profit_factor']:>10.2f}",
            f"  Avg Hold Days:      {m['avg_hold_days']:>10.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot_equity(self, save_path: str = None):
        """Plot equity curve with drawdown."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                         sharex=True, gridspec_kw={"hspace": 0.1})
        fig.suptitle("Backtest Performance", fontsize=14, fontweight="bold")

        eq = self.equity_curve["equity"]
        dates = eq.index

        # Equity curve
        ax1.plot(dates, eq, color="#2196F3", linewidth=1.5, label="Portfolio")
        ax1.axhline(self.initial_capital, color="gray", linestyle="--", alpha=0.5,
                     label=f"Initial (${self.initial_capital:,.0f})")
        ax1.fill_between(dates, self.initial_capital, eq,
                         where=eq >= self.initial_capital, alpha=0.1, color="green")
        ax1.fill_between(dates, self.initial_capital, eq,
                         where=eq < self.initial_capital, alpha=0.1, color="red")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        ax2.fill_between(dates, 0, dd, color="red", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Equity plot saved to {save_path}")
        plt.close()
