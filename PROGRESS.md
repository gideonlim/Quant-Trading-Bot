# Quant Trading Bot — Progress & Documentation

## Project Overview

A quantitative trading system for US equities that exploits multi-factor momentum combined with mean-reversion entry timing. Built for Alpaca paper trading with $100,000 capital. Moderate frequency — holding periods of days to weeks with rebalancing every 5 trading days.

**Status: v2.0 — Production-hardened for autonomous GitHub Actions operation.**

---

## The Edge: Why This Strategy Should Work

The strategy combines three well-documented market anomalies that have persisted across decades of academic research:

### 1. Cross-Sectional Momentum (Primary Signal)
Jegadeesh & Titman (1993) showed that stocks with strong 3-12 month returns tend to continue outperforming for the next 1-3 months. We use 1-month and 6-month momentum with a 5-day skip period to avoid the well-documented short-term reversal effect (Lehmann, 1990).

**Why it persists:** Behavioral underreaction to news, gradual information diffusion, and institutional herding create predictable price patterns that take weeks to fully play out.

### 2. Mean-Reversion Entry Timing (Entry Filter)
Rather than blindly buying momentum stocks, we use RSI and Bollinger Bands to time entries on pullbacks within the momentum trend. This avoids buying at the top of short-term spikes, significantly improving average entry prices and reducing drawdowns.

### 3. Volatility-Adjusted Position Sizing (Risk Management)
Barroso & Santa-Clara (2015) demonstrated that scaling momentum positions inversely to their recent volatility dramatically improves Sharpe ratios and reduces crash risk. Each position contributes roughly equal risk to the portfolio.

### Additional Filters
- **200-day SMA trend filter:** Avoid buying stocks in structural downtrends
- **Volume confirmation:** Momentum accompanied by above-average volume is more reliable
- **Sector diversification:** No more than 30% exposure to any single sector

---

## Architecture

```
quant_trader/
├── __init__.py          # Package init
├── __main__.py          # Entry point for `python -m quant_trader`
├── config.py            # All tunable parameters
├── alpaca_client.py     # REST client for Alpaca APIs
├── indicators.py        # Technical indicators & feature engineering
├── strategy.py          # Multi-factor scoring & signal generation
├── backtest.py          # Historical backtesting engine
├── executor.py          # Live paper trading execution
└── main.py              # CLI interface
```

### Module Responsibilities

**config.py** — Central parameter store. Every tunable number lives here: momentum lookback windows, RSI thresholds, position sizing limits, risk parameters, and more. Change the strategy's behavior without touching any other file.

**alpaca_client.py** — Lightweight REST client built on `requests` (no third-party Alpaca SDK needed). Handles authentication, rate limiting, retries, multi-bar data fetching, order submission (including bracket orders with stop-loss/take-profit), and position management.

**indicators.py** — Computes all technical features from raw OHLCV data using only pandas/numpy:
- Momentum returns (with skip period)
- RSI (Wilder's smoothing)
- Bollinger Bands (%B and bandwidth)
- SMA/EMA, MACD
- ATR, historical volatility, volatility ratio
- Relative volume, On-Balance Volume
- Cross-sectional z-scores for ranking

**strategy.py** — The brain. For each stock in the universe:
1. Compute z-scores of all factors across the universe (cross-sectional)
2. Calculate weighted composite score
3. Apply filters (trend, RSI overbought, minimum momentum)
4. Select top N stocks by score
5. Size positions using inverse-volatility weighting
6. Set ATR-based stop-loss and take-profit levels

**backtest.py** — Walk-forward backtester that simulates the strategy day-by-day:
- Rebalances every 5 trading days
- Checks exit conditions (stop-loss, take-profit, trailing stop) daily
- Accounts for slippage and commissions
- Includes a max-drawdown circuit breaker (15%)
- Generates full performance analytics: Sharpe, Sortino, Calmar, win rate, profit factor, etc.
- Produces equity curve and drawdown plots

**executor.py** — Bridges strategy signals to Alpaca order execution:
- Fetches live data and computes features
- Generates signals from the strategy
- Reconciles target portfolio vs. current positions
- Submits bracket orders (market entry + SL + TP)
- Logs every cycle to JSON for audit trail
- Supports dry-run mode for analysis without execution

**main.py** — CLI with four commands:
- `backtest` — Run historical backtest with synthetic or real data
- `trade` — Execute one trading cycle (with `--dry-run` option)
- `signals` — View current strategy signals
- `status` — View portfolio status

---

## How to Run

### Prerequisites
```bash
# Set Alpaca API keys (get free ones at https://app.alpaca.markets/signup)
export ALPACA_API_KEY='your-paper-trading-key'
export ALPACA_SECRET_KEY='your-paper-trading-secret'
```

### Commands
```bash
# Navigate to the project parent directory
cd "Quant Trading Bot"

# Run backtest (works without API keys using synthetic data)
python -m quant_trader.main backtest

# Run backtest with custom dates
python -m quant_trader.main backtest --start 2024-01-01 --end 2025-12-31

# View current signals (requires API keys)
python -m quant_trader.main signals

# Dry run — see what trades would be made without executing
python -m quant_trader.main trade --dry-run

# Execute one trading cycle on paper account
python -m quant_trader.main trade

# Check portfolio status
python -m quant_trader.main status
```

### Output Files
- `backtest_equity.png` — Equity curve with drawdown chart
- `backtest_trades.json` — Every trade with entry/exit details
- `backtest_metrics.json` — Performance metrics (Sharpe, CAGR, etc.)
- `current_signals.json` — Latest strategy signals
- `trade_logs/cycle_*.json` — Execution audit trail

---

## Strategy Parameters (config.py)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Momentum fast | 21 days | ~1 month short-term momentum |
| Momentum slow | 126 days | ~6 month medium-term momentum |
| Skip recent | 5 days | Avoid short-term reversal effect |
| RSI period | 14 | Standard RSI |
| RSI overbought | 75 | Don't buy overextended stocks |
| Trend SMA | 200 days | Classic bull/bear filter |
| Max positions | 15 | Diversification with concentration |
| Max position % | 10% | No single stock dominates |
| Stop loss | 2x ATR | Volatility-adjusted stop |
| Take profit | 4x ATR | 2:1 reward-to-risk |
| Trailing stop | 1.5x ATR | Lock in profits |
| Rebalance | 5 days | Weekly-ish rebalancing |
| Max drawdown | 15% | Circuit breaker |

---

## Risk Management

The system has multiple layers of risk controls:

1. **Position-level:** ATR-based stop-loss (2x ATR), take-profit (4x ATR), and trailing stop (1.5x ATR)
2. **Portfolio-level:** Max 15 positions, max 10% per position, inverse-volatility sizing
3. **Sector-level:** Max 30% in any one sector
4. **Drawdown breaker:** Liquidate all positions if portfolio drops 15% from peak
5. **Entry filters:** RSI < 75, positive 6-month momentum, above 200-SMA

---

## Dependencies

The system uses only Python standard library + these packages:
- `pandas` — Data manipulation
- `numpy` — Numerical computation
- `matplotlib` — Charting
- `requests` — HTTP client for Alpaca API

No external TA libraries, no Alpaca SDK. Everything is self-contained.

---

## v2.0 Production Hardening (2026-03-16)

The bot is now fully autonomous and can run continuously on GitHub Actions without manual intervention.

### New: Persistent State Management (`state.py` + `bot_state.json`)
Tracks rebalance dates, cycle counts, peak portfolio value, and consecutive errors across stateless CI runs. The state file is committed back to the repo after each run.

### New: Rebalance Scheduling
The executor now only performs a full rebalance every ~5 trading days (configurable via `REBALANCE_DAYS`). On non-rebalance days it runs in monitoring-only mode — bracket orders on Alpaca handle stop-loss and take-profit exits automatically.

### New: Pending Orders Deduplication
Before placing new buy orders, the executor fetches all open/pending orders from Alpaca and filters out symbols that already have unfilled orders, preventing duplicate positions.

### New: Portfolio Drawdown Circuit Breaker
Tracks a high-water mark (peak portfolio value) persistently. If portfolio drops 15%+ from peak, the bot cancels all orders and liquidates all positions. A 5% warning threshold logs early alerts.

### New: Consecutive Error Guard
If 5 or more runs fail consecutively, the bot halts and requires manual intervention (reset `consecutive_errors` in `bot_state.json` to resume).

### New: Market Calendar Validation
Before running, the bot checks the Alpaca calendar API to verify today is an actual trading day. Skips entirely on market holidays (Presidents' Day, MLK Day, etc.), saving API calls.

### New: GitHub Actions Job Summary
Each run writes a rich Markdown summary to `$GITHUB_STEP_SUMMARY` including portfolio table, current positions, top signals, and drawdown metrics — visible directly in the Actions UI.

### Improved: Workflow Reliability
- Pip dependency caching for faster installs
- Market holiday pre-check step (skips entire job on holidays)
- Push retry with rebase (handles concurrent commits)
- Commits `bot_state.json` and `universe_cache.json` back to repo
- 20-minute timeout (up from 15)

---

## Next Steps / Future Improvements

- [ ] Add sector-aware diversification using GICS classification
- [ ] Implement walk-forward optimization for parameter tuning
- [ ] Add short-selling capability for bear market alpha
- [ ] Integrate earnings calendar to avoid holding through announcements
- [ ] Add real-time monitoring dashboard (web UI)
- [ ] Implement intraday entry timing for better execution
- [ ] Add Monte Carlo simulation for confidence intervals
- [ ] Add Slack/Discord notifications on circuit breaker or errors

---

*Last updated: 2026-03-16*
