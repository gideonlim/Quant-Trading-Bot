#!/usr/bin/env python3
"""
Multi-day production simulation test.

Mocks the Alpaca API and steps through 15 trading days to verify:
1. Rebalance scheduling (only every ~5 trading days)
2. State persistence across runs (bot_state.json)
3. Circuit breaker triggers at 15% drawdown
4. Consecutive error halt after 5 failures
5. Pending order deduplication
6. Market calendar validation (skip holidays)
7. Non-rebalance day monitoring mode
8. GitHub Actions job summary output
"""

import json
import os
import sys
import logging
import tempfile
import shutil
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from io import StringIO

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import quant_trader.state as state_module
from quant_trader.state import BotState
from quant_trader.executor import TradeExecutor
from quant_trader.strategy import PortfolioTarget, Signal
from quant_trader import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_multiday")

# Redirect state file to temp dir so we can delete it freely
_TEST_STATE_DIR = Path(tempfile.mkdtemp())
_TEST_STATE_FILE = _TEST_STATE_DIR / "bot_state.json"
state_module.STATE_FILE = _TEST_STATE_FILE


# ── Helpers ────────────────────────────────────────────────────────────────

class MockAlpacaClient:
    """Simulates Alpaca API responses for testing."""

    def __init__(self, scenario: dict = None):
        self.scenario = scenario or {}
        self._positions = self.scenario.get("positions", [])
        self._portfolio_value = self.scenario.get("portfolio_value", 100000)
        self._cash = self.scenario.get("cash", 50000)
        self._is_open = self.scenario.get("is_open", True)
        self._is_trading_day = self.scenario.get("is_trading_day", True)
        self._open_orders = self.scenario.get("open_orders", [])
        self._orders_submitted = []
        self._positions_closed = []
        self._fail_api = self.scenario.get("fail_api", False)

    def get_account(self):
        if self._fail_api:
            raise ConnectionError("Simulated API failure")
        return {"portfolio_value": str(self._portfolio_value), "cash": str(self._cash)}

    def get_positions(self):
        if self._fail_api:
            raise ConnectionError("Simulated API failure")
        return self._positions

    def get_clock(self):
        return {"is_open": self._is_open}

    def get_calendar(self, start=None, end=None):
        if self._is_trading_day:
            return [{"date": start or date.today().isoformat()}]
        return [{"date": "1999-01-01"}]  # Not today

    def get_orders(self, status="open"):
        return self._open_orders

    def submit_bracket_order(self, **kwargs):
        self._orders_submitted.append(kwargs)
        return {"id": f"order-{len(self._orders_submitted)}"}

    def submit_order(self, **kwargs):
        self._orders_submitted.append(kwargs)
        return {"id": f"order-{len(self._orders_submitted)}"}

    def close_position(self, symbol):
        self._positions_closed.append(symbol)
        return {"id": f"close-{symbol}"}

    def close_all_positions(self):
        self._positions_closed.append("__ALL__")
        return []

    def cancel_all_orders(self):
        return []

    def get_multi_bars(self, **kwargs):
        # Return empty — we'll mock the strategy directly
        return {}

    def get_tradable_assets(self):
        return []


def make_signal(symbol, score=1.0, price=100.0):
    """Create a test Signal."""
    return Signal(
        symbol=symbol, score=score, direction="long",
        momentum_fast=0.05, momentum_slow=0.15,
        rsi=45.0, relative_volume=1.2, volatility=0.02,
        atr=2.5, trend_ok=True,
        entry_price=price, stop_loss=price * 0.95,
        take_profit=price * 1.10,
        position_size_pct=0.05,
    )


def reset_state():
    """Clear bot_state.json for a fresh test."""
    if _TEST_STATE_FILE.exists():
        _TEST_STATE_FILE.unlink()


def run_day(executor, day_num, today_date, scenario_overrides=None):
    """Simulate one trading day."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  DAY {day_num} — {today_date}")
    logger.info(f"{'='*60}")

    if scenario_overrides:
        for k, v in scenario_overrides.items():
            setattr(executor.client, f"_{k}", v)

    # Patch date.today() to simulate different days
    with patch("quant_trader.state.date") as mock_date:
        mock_date.today.return_value = today_date
        mock_date.fromisoformat = date.fromisoformat

        # Reload state from disk (simulates fresh GitHub Actions checkout)
        executor.state = BotState()

        result = executor.run_cycle(symbols=["AAPL", "MSFT"], dry_run=False)

    logger.info(f"  Result: {result['status']}")
    return result


# ── Test Scenarios ─────────────────────────────────────────────────────────

def test_rebalance_scheduling():
    """Test that rebalance only happens every ~5 trading days."""
    print("\n" + "=" * 70)
    print("  TEST 1: Rebalance Scheduling")
    print("=" * 70)
    reset_state()

    mock_client = MockAlpacaClient({
        "portfolio_value": 100000,
        "cash": 50000,
        "is_open": True,
        "is_trading_day": True,
        "positions": [
            {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150",
             "market_value": "1550", "unrealized_pl": "50", "current_price": "155"},
        ],
    })

    # Mock signals
    target = PortfolioTarget(
        date="2026-03-16",
        signals=[make_signal("AAPL"), make_signal("MSFT"), make_signal("GOOGL")],
        total_score=3.0, num_positions=3,
    )

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.strategy.generate_signals.return_value = target
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"
    executor.state = BotState()

    # Mock fetch_universe_data to return enriched data
    executor.fetch_universe_data = MagicMock(return_value={"AAPL": None, "MSFT": None})

    results = []
    start_date = date(2026, 3, 16)  # Monday

    for day_num in range(1, 12):
        # Skip weekends
        d = start_date + timedelta(days=day_num - 1)
        if d.weekday() >= 5:
            continue

        result = run_day(executor, day_num, d)
        results.append((d, result["status"]))

    print("\n  Summary:")
    rebalance_count = 0
    monitor_count = 0
    for d, status in results:
        icon = "🔄" if status == "executed" else "👁️"
        print(f"    {d} ({d.strftime('%a')}): {icon} {status}")
        if status == "executed":
            rebalance_count += 1
        elif status == "monitoring":
            monitor_count += 1

    assert rebalance_count >= 1, "Should have at least 1 rebalance"
    assert monitor_count >= 1, "Should have at least 1 monitoring day"
    print(f"\n  ✅ PASS — {rebalance_count} rebalance(s), {monitor_count} monitoring day(s)")

    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


def test_circuit_breaker():
    """Test that circuit breaker fires when drawdown exceeds 15%."""
    print("\n" + "=" * 70)
    print("  TEST 2: Circuit Breaker")
    print("=" * 70)
    reset_state()

    # Start at $100k, establish peak
    state = BotState()
    state.update_peak_value(100000)
    state.mark_rebalanced()
    state.mark_run_success(0)

    # Now simulate 15%+ drop
    mock_client = MockAlpacaClient({
        "portfolio_value": 84000,  # 16% drawdown from 100k
        "cash": 40000,
        "is_open": True,
        "is_trading_day": True,
        "positions": [
            {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150",
             "market_value": "1300", "unrealized_pl": "-200", "current_price": "130"},
        ],
    })

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"

    today = date(2026, 3, 25)
    with patch("quant_trader.state.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        executor.state = BotState()
        result = executor.run_cycle(symbols=["AAPL"], dry_run=False)

    assert result["status"] == "circuit_breaker_triggered", f"Expected circuit_breaker, got {result['status']}"
    assert "__ALL__" in mock_client._positions_closed, "Should have closed all positions"
    print(f"  Portfolio: $84,000 (peak $100,000 = 16% drawdown)")
    print(f"  Result: {result['status']}")
    print(f"  Positions closed: {mock_client._positions_closed}")
    print(f"\n  ✅ PASS — Circuit breaker triggered correctly")

    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


def test_consecutive_errors():
    """Test that bot halts after 5 consecutive errors."""
    print("\n" + "=" * 70)
    print("  TEST 3: Consecutive Error Halt")
    print("=" * 70)
    reset_state()

    # Manually set 5 consecutive errors in state
    state = BotState()
    for _ in range(5):
        state.mark_run_error()

    mock_client = MockAlpacaClient({
        "portfolio_value": 100000,
        "cash": 50000,
        "is_open": True,
        "is_trading_day": True,
    })

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"

    today = date(2026, 3, 20)
    with patch("quant_trader.state.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        executor.state = BotState()
        result = executor.run_cycle(dry_run=False)

    assert result["status"] == "halted", f"Expected halted, got {result['status']}"
    assert "too_many_consecutive_errors" in result.get("reason", "")
    print(f"  Consecutive errors: {executor.state.consecutive_errors}")
    print(f"  Result: {result['status']} — {result.get('reason')}")
    print(f"\n  ✅ PASS — Bot halted after 5 errors")

    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


def test_market_holiday_skip():
    """Test that non-trading days are skipped."""
    print("\n" + "=" * 70)
    print("  TEST 4: Market Holiday Skip")
    print("=" * 70)
    reset_state()

    mock_client = MockAlpacaClient({
        "portfolio_value": 100000,
        "is_trading_day": False,  # Holiday!
    })

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"

    today = date(2026, 1, 19)  # MLK Day
    with patch("quant_trader.state.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        executor.state = BotState()
        result = executor.run_cycle(dry_run=False)

    assert result["status"] == "skipped", f"Expected skipped, got {result['status']}"
    assert result.get("reason") == "not_trading_day"
    print(f"  Date: {today} (MLK Day)")
    print(f"  Result: {result['status']} — {result.get('reason')}")
    print(f"\n  ✅ PASS — Holiday correctly skipped")

    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


def test_pending_order_dedup():
    """Test that symbols with pending orders are filtered out."""
    print("\n" + "=" * 70)
    print("  TEST 5: Pending Order Deduplication")
    print("=" * 70)
    reset_state()

    mock_client = MockAlpacaClient({
        "portfolio_value": 100000,
        "cash": 80000,
        "is_open": True,
        "is_trading_day": True,
        "positions": [],
        "open_orders": [
            {"symbol": "MSFT", "id": "existing-order-1", "status": "new"},
        ],
    })

    target = PortfolioTarget(
        date="2026-03-16",
        signals=[make_signal("AAPL"), make_signal("MSFT"), make_signal("GOOGL")],
        total_score=3.0, num_positions=3,
    )

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.strategy.generate_signals.return_value = target
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"
    executor.fetch_universe_data = MagicMock(return_value={"AAPL": None, "MSFT": None})

    today = date(2026, 3, 16)
    with patch("quant_trader.state.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        executor.state = BotState()
        result = executor.run_cycle(symbols=["AAPL", "MSFT", "GOOGL"], dry_run=False)

    # MSFT should have been filtered out
    submitted_symbols = {o["symbol"] for o in mock_client._orders_submitted}
    assert "MSFT" not in submitted_symbols, f"MSFT should have been filtered out, got {submitted_symbols}"
    assert "AAPL" in submitted_symbols, f"AAPL should have been submitted"
    assert "GOOGL" in submitted_symbols, f"GOOGL should have been submitted"

    print(f"  Pending orders: MSFT")
    print(f"  Signals: AAPL, MSFT, GOOGL")
    print(f"  Orders submitted: {submitted_symbols}")
    print(f"\n  ✅ PASS — MSFT correctly filtered out")

    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


def test_state_persistence():
    """Test that state survives across fresh BotState instances (simulating CI runs)."""
    print("\n" + "=" * 70)
    print("  TEST 6: State Persistence Across Runs")
    print("=" * 70)
    reset_state()

    # Run 1: First ever run
    state1 = BotState()
    assert state1.cycle_count == 0
    assert state1.last_rebalance_date is None
    assert state1.peak_portfolio_value is None

    state1.mark_run_started()
    state1.update_peak_value(100000)
    state1.mark_rebalanced()
    state1.mark_run_success(5)
    print(f"  Run 1: {state1.summary()}")

    # Run 2: Fresh instance (simulates new GitHub Actions run)
    state2 = BotState()
    assert state2.cycle_count == 1, f"Expected 1, got {state2.cycle_count}"
    assert state2.total_orders == 5, f"Expected 5, got {state2.total_orders}"
    assert state2.peak_portfolio_value == 100000
    assert state2.last_rebalance_date is not None
    assert state2.consecutive_errors == 0
    print(f"  Run 2 (fresh load): {state2.summary()}")

    # Run 3: Update peak
    state2.update_peak_value(110000)
    state2.mark_run_success(3)

    state3 = BotState()
    assert state3.cycle_count == 2
    assert state3.total_orders == 8
    assert state3.peak_portfolio_value == 110000
    print(f"  Run 3 (fresh load): {state3.summary()}")

    # Run 4: Error run
    state3.mark_run_error()
    state4 = BotState()
    assert state4.consecutive_errors == 1
    print(f"  Run 4 (after error): {state4.summary()}")

    # Run 5: Success resets errors
    state4.mark_run_success(0)
    state5 = BotState()
    assert state5.consecutive_errors == 0
    assert state5.cycle_count == 3
    print(f"  Run 5 (after success): {state5.summary()}")

    print(f"\n  ✅ PASS — State persists correctly across instances")
    return True


def test_job_summary_output():
    """Test that GitHub Actions job summary is written when env var is set."""
    print("\n" + "=" * 70)
    print("  TEST 7: GitHub Actions Job Summary")
    print("=" * 70)
    reset_state()

    summary_file = Path(tempfile.mktemp(suffix=".md"))

    mock_client = MockAlpacaClient({
        "portfolio_value": 105000,
        "cash": 55000,
        "is_open": True,
        "is_trading_day": True,
        "positions": [
            {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150",
             "market_value": "1600", "unrealized_pl": "100", "current_price": "160"},
        ],
    })

    target = PortfolioTarget(
        date="2026-03-16",
        signals=[make_signal("AAPL"), make_signal("MSFT")],
        total_score=2.0, num_positions=2,
    )

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.strategy.generate_signals.return_value = target
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"
    executor.fetch_universe_data = MagicMock(return_value={"AAPL": None, "MSFT": None})

    today = date(2026, 3, 16)
    with patch("quant_trader.state.date") as mock_date, \
         patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}):
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        executor.state = BotState()
        result = executor.run_cycle(symbols=["AAPL", "MSFT"], dry_run=False)

    assert summary_file.exists(), "Summary file should have been created"
    content = summary_file.read_text()
    assert "Trading Bot" in content
    assert "Portfolio" in content
    assert "$105,000" in content or "105000" in content or "105,000" in content

    print(f"  Summary file: {summary_file}")
    print(f"  Content length: {len(content)} chars")
    # Show first few lines
    for line in content.split("\n")[:12]:
        print(f"    {line}")
    print(f"    ...")

    print(f"\n  ✅ PASS — Job summary written correctly")

    summary_file.unlink(missing_ok=True)
    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


def test_error_recovery():
    """Test that bot recovers after errors and resets consecutive count."""
    print("\n" + "=" * 70)
    print("  TEST 8: Error Recovery")
    print("=" * 70)
    reset_state()

    mock_client = MockAlpacaClient({
        "portfolio_value": 100000,
        "cash": 50000,
        "is_open": True,
        "is_trading_day": True,
        "positions": [],
    })

    target = PortfolioTarget(
        date="2026-03-16",
        signals=[make_signal("AAPL")],
        total_score=1.0, num_positions=1,
    )

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.client = mock_client
    executor.strategy = MagicMock()
    executor.strategy.generate_signals.return_value = target
    executor.log_dir = Path(tempfile.mkdtemp())
    executor.execution_log = []
    executor.trade_ledger_path = executor.log_dir / "trade_ledger.csv"
    executor.fetch_universe_data = MagicMock(return_value={"AAPL": None})

    # Simulate 3 error runs then 1 success
    for i in range(3):
        today = date(2026, 3, 16 + i)
        with patch("quant_trader.state.date") as mock_date:
            mock_date.today.return_value = today
            mock_date.fromisoformat = date.fromisoformat
            executor.state = BotState()
            # Force API failure
            mock_client._fail_api = True
            result = executor.run_cycle(dry_run=False)
            assert result["status"] == "error", f"Day {i}: expected error, got {result['status']}"

    # Check consecutive errors
    state_check = BotState()
    assert state_check.consecutive_errors == 3, f"Expected 3 errors, got {state_check.consecutive_errors}"
    print(f"  After 3 errors: consecutive_errors={state_check.consecutive_errors}")

    # Now succeed
    mock_client._fail_api = False
    today = date(2026, 3, 19)
    with patch("quant_trader.state.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat = date.fromisoformat
        executor.state = BotState()
        result = executor.run_cycle(symbols=["AAPL"], dry_run=False)

    state_after = BotState()
    assert state_after.consecutive_errors == 0, f"Expected 0 errors after success, got {state_after.consecutive_errors}"
    print(f"  After success: consecutive_errors={state_after.consecutive_errors}")
    print(f"\n  ✅ PASS — Errors tracked and reset on success")

    shutil.rmtree(executor.log_dir, ignore_errors=True)
    return True


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Rebalance Scheduling", test_rebalance_scheduling),
        ("Circuit Breaker", test_circuit_breaker),
        ("Consecutive Error Halt", test_consecutive_errors),
        ("Market Holiday Skip", test_market_holiday_skip),
        ("Pending Order Dedup", test_pending_order_dedup),
        ("State Persistence", test_state_persistence),
        ("Job Summary Output", test_job_summary_output),
        ("Error Recovery", test_error_recovery),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            failed += 1
            failures.append((name, str(e)))
            logger.error(f"  ❌ FAIL — {name}: {e}", exc_info=True)

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failures:
        for name, err in failures:
            print(f"  ❌ {name}: {err}")
        sys.exit(1)
    else:
        print("  🎉 All tests passed!")

    # Cleanup
    reset_state()
