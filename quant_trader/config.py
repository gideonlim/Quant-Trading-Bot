"""
Configuration for the Quant Trading Bot.
All tunable parameters live here.
"""

import os
from pathlib import Path


def load_env_file():
    """
    Load variables from a .env file into os.environ.
    Searches for .env in the project root (parent of quant_trader/).
    Does NOT override variables that are already set in the environment.
    """
    # Look for .env in the project root (one level up from this file)
    env_paths = [
        Path(__file__).resolve().parent.parent / "keys.env",   # project root
        Path.cwd() / "keys.env",                               # current working dir
        Path(__file__).resolve().parent.parent / ".env",       # fallback: .env
        Path.cwd() / ".env",                                   # fallback: .env in cwd
    ]
    for env_path in env_paths:
        if env_path.is_file():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")  # Remove surrounding quotes
                    if key and key not in os.environ:    # Don't override existing env vars
                        os.environ[key] = value
            return  # Stop after first .env found


# Load .env on import so keys are available everywhere
load_env_file()

# ── Alpaca API ────────────────────────────────────────────────────────────────
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"       # Paper trading
ALPACA_DATA_URL = "https://data.alpaca.markets"            # Market data
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# ── Universe ──────────────────────────────────────────────────────────────────
# We trade liquid, large/mid-cap US equities from the S&P 500.
# The full list is fetched dynamically, but here's a curated fallback.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "ACN", "TMO", "ABT", "DHR", "LIN", "CRM", "NKE", "NEE", "TXN",
    "PM", "UPS", "RTX", "LOW", "HON", "AMGN", "UNP", "INTC", "IBM",
    "BA", "CAT", "GE", "SBUX", "AMAT", "DE", "ISRG", "ADP", "BKNG",
    "MDLZ", "ADI", "GILD", "MMC", "REGN", "VRTX", "SYK", "ZTS",
    "CI", "BDX", "CB", "MO", "DUK", "SO", "CL", "CME", "ICE", "PLD",
    "EQIX", "APD", "SHW", "NSC", "EMR", "ETN", "WM", "GD", "FIS",
    "AON", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "PANW",
    "ORCL", "ADBE", "NOW", "INTU", "PYPL", "ABNB", "UBER", "DASH",
    "SQ", "COIN", "SNOW", "DDOG", "NET", "CRWD",
]

# ── Strategy Parameters ───────────────────────────────────────────────────────
# Multi-Factor Momentum + Mean Reversion

# Momentum look-back windows (trading days)
MOMENTUM_FAST_PERIOD = 21       # ~1 month  (short-term momentum)
MOMENTUM_SLOW_PERIOD = 126      # ~6 months (medium-term momentum)
MOMENTUM_SKIP_RECENT = 5        # Skip last 5 days (short-term reversal filter)

# Mean Reversion filters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 75             # Don't buy if RSI > this (overextended)
RSI_OVERSOLD = 30               # Attractive entry for mean reversion
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# Volume confirmation
VOLUME_MA_PERIOD = 20           # Volume must be above its 20-day MA

# Trend filter (avoid trading against the macro trend)
TREND_SMA_PERIOD = 200          # 200-day SMA as bull/bear filter

# ── Scoring Weights ───────────────────────────────────────────────────────────
# The composite score is a weighted sum of z-scored factors.
WEIGHT_MOMENTUM_FAST = 0.25
WEIGHT_MOMENTUM_SLOW = 0.35
WEIGHT_RSI_REVERSION = 0.15     # Lower RSI = higher score (mean reversion entry)
WEIGHT_VOLUME_SURGE = 0.10
WEIGHT_VOLATILITY_ADJ = 0.15    # Lower vol = higher risk-adj score

# ── Portfolio Construction ────────────────────────────────────────────────────
TOTAL_CAPITAL = 100_000         # USD
MAX_POSITIONS = 15              # Maximum concurrent positions
POSITION_SIZE_METHOD = "inverse_volatility"  # or "equal_weight"
MAX_POSITION_PCT = 0.10         # No single position > 10% of portfolio
MIN_POSITION_PCT = 0.02         # Minimum 2% to be worth trading

# ── Risk Management ──────────────────────────────────────────────────────────
STOP_LOSS_ATR_MULT = 2.0        # Stop loss = 2x ATR below entry
TAKE_PROFIT_ATR_MULT = 4.0      # Take profit = 4x ATR above entry
TRAILING_STOP_ATR_MULT = 1.5    # Trailing stop = 1.5x ATR from peak
MAX_PORTFOLIO_RISK_PCT = 0.02   # Risk no more than 2% of capital per trade
MAX_DRAWDOWN_PCT = 0.15         # Halt trading if drawdown exceeds 15%
MAX_SECTOR_EXPOSURE_PCT = 0.30  # No more than 30% in one sector
REBALANCE_DAYS = 5              # Re-evaluate positions every 5 trading days

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"
COMMISSION_PER_SHARE = 0.0      # Alpaca is commission-free
SLIPPAGE_BPS = 5                # 5 basis points slippage assumption

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_LOOKBACK_DAYS = 400        # Need ~400 days for 200-SMA + momentum calc
CACHE_DIR = "data_cache"
