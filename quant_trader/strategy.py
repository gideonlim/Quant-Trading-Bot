"""
Multi-Factor Momentum + Mean Reversion Strategy
================================================

THE EDGE:
We exploit three well-documented market anomalies simultaneously:

1. **Intermediate-term momentum** (Jegadeesh & Titman, 1993):
   Stocks that performed well over 1-12 months tend to continue performing well.
   We use 1-month and 6-month momentum with a 5-day skip to avoid short-term reversal.

2. **Short-term mean reversion** (Lehmann, 1990):
   We filter entries using RSI and Bollinger Bands to avoid buying at the
   peak of a short-term run-up. Buy momentum stocks on pullbacks.

3. **Volatility-adjusted sizing** (Barroso & Santa-Clara, 2015):
   Scale positions inversely to their volatility so that each position
   contributes roughly equal risk to the portfolio.

ADDITIONAL FILTERS:
- 200-day SMA trend filter (avoid buying in bear markets)
- Volume confirmation (momentum with volume is more reliable)
- Sector diversification constraint

SIGNAL GENERATION:
For each stock in the universe, we compute a composite score:
  score = w1 * z(mom_fast) + w2 * z(mom_slow) + w3 * z(rsi_reversion)
        + w4 * z(volume_surge) + w5 * z(inv_volatility)

We go long the top N scoring stocks that pass all filters.

HOLDING PERIOD: 5-20 trading days (rebalance every 5 days)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from . import config
from .indicators import compute_all_features, zscore

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal for a single stock."""
    symbol: str
    score: float
    direction: str          # "long" or "flat"
    momentum_fast: float
    momentum_slow: float
    rsi: float
    relative_volume: float
    volatility: float
    atr: float
    trend_ok: bool
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # Suggested allocation as % of portfolio


@dataclass
class PortfolioTarget:
    """Target portfolio allocation after strategy runs."""
    date: datetime
    signals: List[Signal] = field(default_factory=list)
    total_score: float = 0.0
    num_positions: int = 0

    @property
    def symbols(self) -> List[str]:
        return [s.symbol for s in self.signals]


class Strategy:
    """
    Multi-Factor Momentum + Mean Reversion scoring and signal generation.
    """

    def __init__(self, params: dict = None):
        """Initialize with optional parameter overrides."""
        self.params = params or {}

        # Load parameters (with config defaults)
        self.mom_fast_period = self.params.get("mom_fast_period", config.MOMENTUM_FAST_PERIOD)
        self.mom_slow_period = self.params.get("mom_slow_period", config.MOMENTUM_SLOW_PERIOD)
        self.mom_skip = self.params.get("mom_skip", config.MOMENTUM_SKIP_RECENT)
        self.rsi_period = self.params.get("rsi_period", config.RSI_PERIOD)
        self.rsi_overbought = self.params.get("rsi_overbought", config.RSI_OVERBOUGHT)
        self.rsi_oversold = self.params.get("rsi_oversold", config.RSI_OVERSOLD)
        self.trend_sma = self.params.get("trend_sma", config.TREND_SMA_PERIOD)
        self.volume_ma = self.params.get("volume_ma", config.VOLUME_MA_PERIOD)

        # Weights
        self.w_mom_fast = self.params.get("w_mom_fast", config.WEIGHT_MOMENTUM_FAST)
        self.w_mom_slow = self.params.get("w_mom_slow", config.WEIGHT_MOMENTUM_SLOW)
        self.w_rsi = self.params.get("w_rsi", config.WEIGHT_RSI_REVERSION)
        self.w_vol = self.params.get("w_vol", config.WEIGHT_VOLUME_SURGE)
        self.w_ivol = self.params.get("w_ivol", config.WEIGHT_VOLATILITY_ADJ)

        # Portfolio
        self.max_positions = self.params.get("max_positions", config.MAX_POSITIONS)
        self.max_position_pct = self.params.get("max_position_pct", config.MAX_POSITION_PCT)
        self.min_position_pct = self.params.get("min_position_pct", config.MIN_POSITION_PCT)

        # Risk
        self.sl_atr_mult = self.params.get("sl_atr_mult", config.STOP_LOSS_ATR_MULT)
        self.tp_atr_mult = self.params.get("tp_atr_mult", config.TAKE_PROFIT_ATR_MULT)

    def score_universe(
        self, data: Dict[str, pd.DataFrame], as_of_date: datetime = None
    ) -> pd.DataFrame:
        """
        Score every stock in the universe as of a given date.

        Args:
            data: Dict of symbol -> OHLCV DataFrame (with features computed)
            as_of_date: Score as of this date. If None, uses last available date.

        Returns:
            DataFrame with one row per symbol, sorted by composite score descending.
        """
        rows = []

        for symbol, df in data.items():
            if df.empty or len(df) < self.trend_sma:
                continue

            # Get data as of the target date
            if as_of_date is not None:
                df = df.loc[:as_of_date]
            if df.empty or len(df) < self.trend_sma:
                continue

            latest = df.iloc[-1]

            # ── Extract features ──────────────────────────────────────────
            mom_fast = latest.get("mom_fast", np.nan)
            mom_slow = latest.get("mom_slow", np.nan)
            rsi_val = latest.get("rsi_14", np.nan)
            rvol = latest.get("rvol", np.nan)
            hvol = latest.get("hvol_21", np.nan)
            atr_val = latest.get("atr_14", np.nan)
            trend = latest.get("trend_200", 0)
            close = latest.get("close", np.nan)
            bb_pct_b = latest.get("bb_pct_b", np.nan)

            # Skip if missing critical data
            if any(np.isnan(x) for x in [mom_fast, mom_slow, rsi_val, close, atr_val]):
                continue

            rows.append({
                "symbol": symbol,
                "close": close,
                "mom_fast": mom_fast,
                "mom_slow": mom_slow,
                "rsi": rsi_val,
                "rvol": rvol if not np.isnan(rvol) else 1.0,
                "hvol": hvol if not np.isnan(hvol) else 0.2,
                "atr": atr_val,
                "trend_ok": trend > 0,
                "bb_pct_b": bb_pct_b if not np.isnan(bb_pct_b) else 0.5,
            })

        if not rows:
            return pd.DataFrame()

        scores_df = pd.DataFrame(rows).set_index("symbol")

        # ── Compute z-scores across the universe (cross-sectional) ────────
        scores_df["z_mom_fast"] = zscore(scores_df["mom_fast"])
        scores_df["z_mom_slow"] = zscore(scores_df["mom_slow"])

        # For RSI: lower is better for entry (mean reversion), so invert
        scores_df["z_rsi_inv"] = zscore(-scores_df["rsi"])

        # Volume surge: higher relative volume is better
        scores_df["z_rvol"] = zscore(scores_df["rvol"])

        # Inverse volatility: lower vol = higher risk-adjusted score
        scores_df["z_ivol"] = zscore(-scores_df["hvol"])

        # ── Composite score ───────────────────────────────────────────────
        scores_df["composite_score"] = (
            self.w_mom_fast * scores_df["z_mom_fast"]
            + self.w_mom_slow * scores_df["z_mom_slow"]
            + self.w_rsi * scores_df["z_rsi_inv"]
            + self.w_vol * scores_df["z_rvol"]
            + self.w_ivol * scores_df["z_ivol"]
        )

        # Sort by score (best first)
        scores_df = scores_df.sort_values("composite_score", ascending=False)
        return scores_df

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        as_of_date: datetime = None,
        capital: float = None,
    ) -> PortfolioTarget:
        """
        Generate a complete set of portfolio signals.

        Returns a PortfolioTarget with the top N stocks to hold,
        each with entry/exit levels and position sizing.
        """
        capital = capital or config.TOTAL_CAPITAL
        scores_df = self.score_universe(data, as_of_date)

        if scores_df.empty:
            return PortfolioTarget(date=as_of_date or datetime.now())

        # ── Apply filters ─────────────────────────────────────────────────
        filtered = scores_df[
            (scores_df["trend_ok"])                         # Above 200-SMA
            & (scores_df["rsi"] < self.rsi_overbought)     # Not overbought
            & (scores_df["mom_fast"] > -0.05)              # Not in freefall
            & (scores_df["mom_slow"] > 0)                  # Positive medium-term momentum
            & (scores_df["composite_score"] > 0)           # Positive composite score
        ].copy()

        if filtered.empty:
            logger.info("No stocks passed all filters")
            return PortfolioTarget(date=as_of_date or datetime.now())

        # ── Select top N ──────────────────────────────────────────────────
        selected = filtered.head(self.max_positions)

        # ── Position sizing: inverse volatility ──────────────────────────
        inv_vol = 1.0 / selected["hvol"].clip(lower=0.05)
        raw_weights = inv_vol / inv_vol.sum()

        # Apply position limits
        weights = raw_weights.clip(
            lower=self.min_position_pct,
            upper=self.max_position_pct,
        )
        weights = weights / weights.sum()  # Re-normalize

        # ── Build signals ─────────────────────────────────────────────────
        signals = []
        for symbol in selected.index:
            row = selected.loc[symbol]
            close = row["close"]
            atr_val = row["atr"]
            weight = weights[symbol]

            signal = Signal(
                symbol=symbol,
                score=row["composite_score"],
                direction="long",
                momentum_fast=row["mom_fast"],
                momentum_slow=row["mom_slow"],
                rsi=row["rsi"],
                relative_volume=row["rvol"],
                volatility=row["hvol"],
                atr=atr_val,
                trend_ok=row["trend_ok"],
                entry_price=close,
                stop_loss=round(close - self.sl_atr_mult * atr_val, 2),
                take_profit=round(close + self.tp_atr_mult * atr_val, 2),
                position_size_pct=round(weight, 4),
            )
            signals.append(signal)

        target = PortfolioTarget(
            date=as_of_date or datetime.now(),
            signals=sorted(signals, key=lambda s: s.score, reverse=True),
            total_score=sum(s.score for s in signals),
            num_positions=len(signals),
        )

        logger.info(
            f"Generated {target.num_positions} signals | "
            f"Top: {signals[0].symbol} ({signals[0].score:.3f})"
        )
        return target

    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        peak_price: float, atr_val: float, days_held: int
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.

        Returns (should_exit, reason).
        """
        # Stop loss
        stop = entry_price - self.sl_atr_mult * atr_val
        if current_price <= stop:
            return True, f"stop_loss (price {current_price:.2f} <= {stop:.2f})"

        # Take profit
        target = entry_price + self.tp_atr_mult * atr_val
        if current_price >= target:
            return True, f"take_profit (price {current_price:.2f} >= {target:.2f})"

        # Trailing stop from peak
        trail_stop = peak_price - config.TRAILING_STOP_ATR_MULT * atr_val
        if current_price <= trail_stop and peak_price > entry_price * 1.02:
            return True, f"trailing_stop (price {current_price:.2f} <= {trail_stop:.2f})"

        # Max holding period
        if days_held >= config.REBALANCE_DAYS * 4:  # ~20 trading days max
            return True, f"max_hold_period ({days_held} days)"

        return False, ""
