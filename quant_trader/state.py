"""
Persistent bot state manager.

Tracks rebalance scheduling, cycle counts, and run history so the bot
can make intelligent decisions across stateless GitHub Actions runs.

State is stored in bot_state.json at the project root and committed
back to the repo after each run.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parent.parent / "bot_state.json"


class BotState:
    """
    Reads and writes bot_state.json.

    Fields:
        last_rebalance_date  - Last date a full rebalance was executed
        last_run_date        - Last date the bot ran at all
        cycle_count          - Total number of completed trade cycles
        total_orders         - Total orders submitted lifetime
        consecutive_errors   - Number of consecutive failed runs (resets on success)
        created_at           - When the state file was first created
    """

    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        """Load state from disk, or create defaults if missing."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                logger.debug(f"State loaded: {data}")
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read state file, starting fresh: {e}")

        # First-ever run defaults
        return {
            "last_rebalance_date": None,
            "last_run_date": None,
            "cycle_count": 0,
            "total_orders": 0,
            "consecutive_errors": 0,
            "peak_portfolio_value": None,
            "created_at": date.today().isoformat(),
        }

    def save(self):
        """Persist state to disk."""
        try:
            STATE_FILE.write_text(json.dumps(self._data, indent=2))
            logger.debug(f"State saved: {self._data}")
        except OSError as e:
            logger.error(f"Failed to save state: {e}")

    # ── Rebalance scheduling ──────────────────────────────────────────────────

    def should_rebalance(self, rebalance_every_n_days: int = 5) -> bool:
        """
        Return True if it's time for a full rebalance.

        Logic:
        - If we've never rebalanced, always rebalance.
        - Otherwise, rebalance when at least N trading days have passed
          since the last rebalance.
        """
        last = self._data.get("last_rebalance_date")
        if not last:
            logger.info("No prior rebalance found — will rebalance now.")
            return True

        last_date = date.fromisoformat(last)
        today = date.today()
        calendar_days = (today - last_date).days

        # Approximate: 5 trading days ≈ 7 calendar days
        # Use calendar days as a simple proxy
        approx_trading_days = calendar_days * 5 / 7
        should = approx_trading_days >= rebalance_every_n_days

        logger.info(
            f"Last rebalance: {last} ({calendar_days} calendar days ago, "
            f"~{approx_trading_days:.1f} trading days) | "
            f"Rebalance due: {should}"
        )
        return should

    def mark_rebalanced(self):
        """Record that a rebalance just happened."""
        self._data["last_rebalance_date"] = date.today().isoformat()
        logger.info(f"Rebalance date updated to {self._data['last_rebalance_date']}")

    # ── Run tracking ──────────────────────────────────────────────────────────

    def mark_run_started(self):
        """Record that a run started today."""
        self._data["last_run_date"] = date.today().isoformat()

    def mark_run_success(self, orders_placed: int = 0):
        """Record a successful run."""
        self._data["cycle_count"] = self._data.get("cycle_count", 0) + 1
        self._data["total_orders"] = self._data.get("total_orders", 0) + orders_placed
        self._data["consecutive_errors"] = 0
        self.save()
        logger.info(
            f"Cycle #{self._data['cycle_count']} complete | "
            f"Total orders: {self._data['total_orders']}"
        )

    def mark_run_error(self):
        """Record a failed run."""
        self._data["consecutive_errors"] = self._data.get("consecutive_errors", 0) + 1
        self.save()
        logger.warning(
            f"Consecutive errors: {self._data['consecutive_errors']}"
        )

    # ── Portfolio peak tracking (for circuit breaker) ────────────────────────

    def update_peak_value(self, current_value: float):
        """Update high-water mark if current value exceeds it."""
        peak = self._data.get("peak_portfolio_value")
        if peak is None or current_value > peak:
            self._data["peak_portfolio_value"] = current_value
            logger.info(f"New portfolio peak: ${current_value:,.2f}")

    def get_drawdown_pct(self, current_value: float) -> float:
        """Return current drawdown as a fraction (0.0 to 1.0) from peak."""
        peak = self._data.get("peak_portfolio_value")
        if not peak or peak <= 0:
            return 0.0
        return max(0.0, (peak - current_value) / peak)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def cycle_count(self) -> int:
        return self._data.get("cycle_count", 0)

    @property
    def total_orders(self) -> int:
        return self._data.get("total_orders", 0)

    @property
    def consecutive_errors(self) -> int:
        return self._data.get("consecutive_errors", 0)

    @property
    def peak_portfolio_value(self) -> Optional[float]:
        return self._data.get("peak_portfolio_value")

    @property
    def last_rebalance_date(self) -> Optional[str]:
        return self._data.get("last_rebalance_date")

    @property
    def last_run_date(self) -> Optional[str]:
        return self._data.get("last_run_date")

    def summary(self) -> str:
        """One-line status string for logging."""
        peak = self.peak_portfolio_value
        peak_str = f"${peak:,.0f}" if peak else "n/a"
        return (
            f"Cycle #{self.cycle_count} | "
            f"Last rebalance: {self.last_rebalance_date or 'never'} | "
            f"Total orders: {self.total_orders} | "
            f"Peak: {peak_str} | "
            f"Consecutive errors: {self.consecutive_errors}"
        )
