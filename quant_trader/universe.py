"""
Dynamic stock universe selection.
Instead of a hardcoded list, we pull tradable assets from Alpaca
and filter for liquid, large-cap US equities.
"""

import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from . import config
from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# Cache file so we don't hit the API every single run
CACHE_FILE = Path(__file__).resolve().parent.parent / "universe_cache.json"
CACHE_MAX_AGE_HOURS = 24  # Refresh once per day


def get_dynamic_universe(
    client: AlpacaClient,
    min_price: float = 10.0,
    max_price: float = 10000.0,
    top_n: int = 150,
    force_refresh: bool = False,
) -> List[str]:
    """
    Build a trading universe dynamically from Alpaca's asset list.

    Steps:
    1. Pull all active, tradable US equities from Alpaca.
    2. Filter out penny stocks, OTC, non-shortable (proxy for illiquid), etc.
    3. Fetch recent volume data and filter for liquidity.
    4. Return top N most liquid symbols.

    Results are cached to disk for 24 hours to avoid redundant API calls.

    Args:
        client: AlpacaClient instance
        min_price: Minimum stock price (filters penny stocks)
        max_price: Maximum stock price
        top_n: Number of stocks to include in universe
        force_refresh: Ignore cache and rebuild

    Returns:
        List of ticker symbols
    """
    # ── Check cache first ─────────────────────────────────────────────────
    if not force_refresh and CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
            cached_time = datetime.fromisoformat(cache["timestamp"])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600

            if age_hours < CACHE_MAX_AGE_HOURS:
                symbols = cache["symbols"]
                logger.info(
                    f"Universe loaded from cache ({len(symbols)} symbols, "
                    f"{age_hours:.1f}h old)"
                )
                return symbols
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Cache invalid, rebuilding: {e}")

    # ── Pull all tradable assets ──────────────────────────────────────────
    logger.info("Building dynamic universe from Alpaca assets...")
    try:
        all_assets = client.get_tradable_assets()
    except Exception as e:
        logger.warning(f"Failed to fetch assets from Alpaca: {e}")
        logger.info("Falling back to static universe")
        return config.FALLBACK_UNIVERSE

    # ── Filter for quality ────────────────────────────────────────────────
    # Goal: ~800-1000 candidates (S&P 500 sized + a buffer of quality mid-caps)
    candidates = []
    for asset in all_assets:
        # Must be on NYSE or NASDAQ only (skip ARCA/BATS — mostly ETFs)
        exchange = asset.get("exchange", "")
        if exchange not in ("NYSE", "NASDAQ"):
            continue

        # Must be easily tradable
        if not asset.get("tradable", False):
            continue
        if not asset.get("shortable", False):
            continue

        # Must be flagged as easy-to-borrow (another liquidity proxy)
        if not asset.get("easy_to_borrow", False):
            continue

        symbol = asset.get("symbol", "")

        # Only plain letter tickers (no warrants, preferred shares, units)
        if not symbol.isalpha():
            continue
        # Skip very long tickers (usually small-cap/micro-cap)
        if len(symbol) > 5:
            continue

        candidates.append(symbol)

    # Cap at 300 candidates — enough to cover S&P 500 quality stocks
    # while keeping the volume scan fast enough for GitHub Actions free tier
    MAX_CANDIDATES = 500
    if len(candidates) > MAX_CANDIDATES:
        candidates = candidates[:MAX_CANDIDATES]

    logger.info(f"Initial filter: {len(candidates)} candidates from {len(all_assets)} assets")

    if len(candidates) < 50:
        logger.warning(f"Only {len(candidates)} candidates found, supplementing with fallback")
        candidate_set = set(candidates)
        for sym in config.FALLBACK_UNIVERSE:
            if sym not in candidate_set:
                candidates.append(sym)

    # ── Rank by recent volume (liquidity proxy) ──────────────────────────
    # Fetch 20 days of data for volume ranking
    # Do this in batches to be efficient
    logger.info(f"Fetching volume data for {len(candidates)} candidates...")

    volume_scores = {}
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Process in batches of 20 (Alpaca multi-bar limit)
    total_candidates = len(candidates)
    for i in range(0, total_candidates, 20):
        batch = candidates[i:i + 20]
        done = min(i + 20, total_candidates)
        pct = done / total_candidates * 100
        bar_len = 30
        filled = int(bar_len * done / total_candidates)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  Scanning liquidity: {bar} {done}/{total_candidates} ({pct:.0f}%)")
        sys.stdout.flush()
        try:
            bars_dict = client.get_multi_bars(
                symbols=batch,
                timeframe="1Day",
                start=start_date,
                limit=20,
                show_progress=False,
            )
            for sym, df in bars_dict.items():
                if df.empty:
                    continue
                # Average daily dollar volume = avg(close * volume)
                avg_dollar_vol = (df["close"] * df["volume"]).mean()
                last_price = df["close"].iloc[-1]

                # Apply price filter
                if last_price < min_price or last_price > max_price:
                    continue

                # Require minimum dollar volume ($1M/day avg)
                if avg_dollar_vol < 1_000_000:
                    continue

                volume_scores[sym] = avg_dollar_vol

        except Exception as e:
            logger.debug(f"Failed to fetch volume for batch starting at {i}: {e}")
            continue

        # Pause between batches to respect free-tier rate limits
        time.sleep(0.5)

    sys.stdout.write("\n")
    sys.stdout.flush()

    if not volume_scores:
        logger.warning("Volume fetch failed entirely, using fallback universe")
        return config.FALLBACK_UNIVERSE

    # ── Sort by dollar volume and take top N ──────────────────────────────
    ranked = sorted(volume_scores.items(), key=lambda x: x[1], reverse=True)
    universe = [sym for sym, _ in ranked[:top_n]]

    logger.info(
        f"Dynamic universe built: {len(universe)} stocks | "
        f"Top 5 by liquidity: {', '.join(universe[:5])}"
    )

    # ── Cache results ─────────────────────────────────────────────────────
    try:
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "count": len(universe),
            "symbols": universe,
            "volume_scores": {s: round(v, 0) for s, v in ranked[:top_n]},
        }
        CACHE_FILE.write_text(json.dumps(cache_data, indent=2))
        logger.info(f"Universe cached to {CACHE_FILE}")
    except Exception as e:
        logger.debug(f"Failed to cache universe: {e}")

    return universe
