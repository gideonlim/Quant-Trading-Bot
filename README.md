# Quant Trading Bot

A quantitative trading system for US equities using a multi-factor momentum + mean reversion strategy. Built for Alpaca paper trading with Python.

---

## Setup

### 1. Install Python dependencies

The bot only needs standard packages. If you don't have them:

```bash
pip install pandas numpy matplotlib requests
```

### 2. Add your Alpaca API keys

Open `keys.env` in the project folder and replace the placeholders:

```
ALPACA_API_KEY=your-actual-api-key
ALPACA_SECRET_KEY=your-actual-secret-key
```

Get free paper trading keys at: https://app.alpaca.markets/signup

### 3. Navigate to the project folder

```bash
cd "Quant Trading Bot"
```

---

## Commands

All commands are run from the project folder using:

```
python -m quant_trader.main <command> [options]
```

### `signals` — View current buy/sell signals

Shows what the strategy wants to buy right now, ranked by score, with entry prices, stop-losses, take-profits, and allocation percentages. Does not place any trades.

```bash
python -m quant_trader.main signals
```

Saves output to: `current_signals.json`

---

### `trade --dry-run` — Preview trades without executing

Compares the strategy's target portfolio to your current Alpaca positions and shows what it *would* buy and sell — but does not submit any orders.

```bash
python -m quant_trader.main trade --dry-run
```

---

### `trade` — Execute one trading cycle

Runs the full cycle: fetches data, generates signals, closes positions that have fallen out of the rankings, and opens new positions with bracket orders (automatic stop-loss and take-profit). Each run is logged.

```bash
python -m quant_trader.main trade
```

Saves output to: `trade_logs/cycle_<timestamp>.json`

---

### `status` — Check your portfolio

Shows current positions, entry prices, unrealized P&L, and account value.

```bash
python -m quant_trader.main status
```

---

### `backtest` — Run a historical backtest

Simulates the strategy on past data and produces performance metrics, an equity curve chart, and a full trade log.

```bash
# Default: 2023-01-01 to 2025-12-31
python -m quant_trader.main backtest

# Custom date range
python -m quant_trader.main backtest --start 2024-01-01 --end 2025-12-31
```

Saves output to:
- `backtest_equity.png` — equity curve and drawdown chart
- `backtest_metrics.json` — Sharpe, CAGR, max drawdown, win rate, etc.
- `backtest_trades.json` — every trade with entry/exit details

---

## Global Options

These go **before** the command name:

```bash
python -m quant_trader.main [options] <command>
```

| Option | Description |
|--------|-------------|
| `-v` or `--verbose` | Show detailed debug logging |
| `-o <dir>` or `--output-dir <dir>` | Directory for output files (default: current folder) |
| `--universe-size <N>` | Number of stocks to evaluate (default: 50) |

Example:

```bash
python -m quant_trader.main -v --universe-size 100 signals
```

---

## Typical Workflow

1. **Run `signals`** to see what the strategy likes today
2. **Run `trade --dry-run`** to preview exactly what orders would be placed
3. **Run `trade`** to execute on your Alpaca paper account
4. **Run `status`** anytime to check positions and P&L
5. **Repeat every ~5 trading days** — the strategy rebalances on each `trade` run, closing stale positions and opening new ones

---

## How It Works

The strategy scores every stock in the universe using five factors:

- **6-month momentum** (35%) — stocks trending up over months tend to continue
- **1-month momentum** (25%) — shorter-term confirmation of the trend
- **RSI mean reversion** (15%) — prefer buying on pullbacks, not at peaks
- **Low volatility** (15%) — calmer stocks get higher risk-adjusted scores
- **Volume surge** (10%) — momentum with volume is more reliable

Stocks must also be above their 200-day moving average (trend filter) and have RSI below 75 (not overbought). The top 15 scoring stocks are selected, sized by inverse volatility, and each gets an ATR-based stop-loss, take-profit, and trailing stop.

---

## Project Structure

```
Quant Trading Bot/
├── keys.env                 # Your Alpaca API keys
├── README.md                # This file
├── PROGRESS.md              # Detailed strategy documentation
├── quant_trader/
│   ├── config.py            # All tunable parameters
│   ├── alpaca_client.py     # Alpaca REST API client
│   ├── universe.py          # Dynamic stock universe selection
│   ├── indicators.py        # Technical indicators (RSI, Bollinger, ATR, etc.)
│   ├── strategy.py          # Multi-factor scoring and signal generation
│   ├── backtest.py          # Historical backtesting engine
│   ├── executor.py          # Live paper trading execution
│   └── main.py              # CLI entry point
```

---

## Customization

**Change parameters** — edit `config.py` to adjust momentum windows, RSI thresholds, number of positions, stop-loss levels, scoring weights, and more.

**Add indicators** — add functions to `indicators.py`, then reference them in `strategy.py`'s scoring logic.

**Different strategy** — write a new class that returns a `PortfolioTarget` (see `strategy.py` for the interface). The backtester and executor work with any strategy that produces signals in that format.

---

## Notes

- This bot uses **paper trading only** — no real money is at risk
- The stock universe is built **dynamically** each day by scanning Alpaca for the most liquid US equities, cached for 24 hours
- Market data comes from Alpaca's **free IEX feed**
- All orders are **bracket orders** with built-in stop-loss and take-profit
- A **15% max drawdown circuit breaker** will liquidate all positions if triggered
