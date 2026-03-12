"""
Lightweight Alpaca REST client built on `requests`.
Handles authentication, market data, and order management for paper trading.
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

from . import config

logger = logging.getLogger(__name__)


class AlpacaClient:
    """REST client for Alpaca Trading & Market Data APIs."""

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = config.ALPACA_BASE_URL if paper else "https://api.alpaca.markets"
        self.data_url = config.ALPACA_DATA_URL
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        })
        self._rate_limit_remaining = 200
        self._rate_limit_reset = 0

    # ── Helper ────────────────────────────────────────────────────────────────

    def _request(self, method: str, url: str, **kwargs) -> dict | list:
        """Make an API request with rate-limit awareness and retries."""
        for attempt in range(3):
            try:
                resp = self.session.request(method, url, **kwargs)

                # Track rate limits
                self._rate_limit_remaining = int(
                    resp.headers.get("x-ratelimit-remaining", 200)
                )
                if self._rate_limit_remaining < 5:
                    # x-ratelimit-reset is a Unix timestamp, not seconds-to-wait
                    reset_ts = int(resp.headers.get("x-ratelimit-reset", 0))
                    sleep_time = max(1, reset_ts - int(time.time())) if reset_ts > 0 else 5
                    sleep_time = min(sleep_time, 60)  # Never sleep more than 60s
                    logger.warning(f"Rate limit low ({self._rate_limit_remaining}), sleeping {sleep_time}s")
                    time.sleep(sleep_time)

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {wait}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json() if resp.text else {}

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP {resp.status_code}: {resp.text}")
                if resp.status_code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """Get account information."""
        return self._request("GET", f"{self.base_url}/v2/account")

    def get_buying_power(self) -> float:
        """Get current buying power."""
        acct = self.get_account()
        return float(acct.get("buying_power", 0))

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        acct = self.get_account()
        return float(acct.get("portfolio_value", 0))

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> List[dict]:
        """Get all open positions."""
        return self._request("GET", f"{self.base_url}/v2/positions")

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol."""
        try:
            return self._request("GET", f"{self.base_url}/v2/positions/{symbol}")
        except requests.exceptions.HTTPError:
            return None

    def close_position(self, symbol: str) -> dict:
        """Close a position entirely."""
        return self._request("DELETE", f"{self.base_url}/v2/positions/{symbol}")

    def close_all_positions(self) -> list:
        """Liquidate all positions."""
        return self._request("DELETE", f"{self.base_url}/v2/positions")

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,          # "buy" or "sell"
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float = None,
        stop_price: float = None,
        trail_percent: float = None,
    ) -> dict:
        """Submit a new order."""
        if qty <= 0:
            raise ValueError(f"Order qty must be positive, got {qty}")

        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(round(limit_price, 2))
        if stop_price is not None:
            payload["stop_price"] = str(round(stop_price, 2))
        if trail_percent is not None:
            payload["trail_percent"] = str(round(trail_percent, 2))

        logger.info(f"Submitting order: {side} {qty} {symbol} ({order_type})")
        return self._request("POST", f"{self.base_url}/v2/orders", json=payload)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "gtc",
    ) -> dict:
        """Submit an OTO bracket order with take-profit and stop-loss."""
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "take_profit": {"limit_price": str(round(take_profit_price, 2))},
            "stop_loss": {"stop_price": str(round(stop_loss_price, 2))},
        }
        logger.info(
            f"Bracket order: {side} {qty} {symbol} | "
            f"TP={take_profit_price:.2f} SL={stop_loss_price:.2f}"
        )
        return self._request("POST", f"{self.base_url}/v2/orders", json=payload)

    def get_orders(self, status: str = "open") -> list:
        """Get orders filtered by status."""
        return self._request(
            "GET", f"{self.base_url}/v2/orders", params={"status": status}
        )

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a specific order."""
        return self._request("DELETE", f"{self.base_url}/v2/orders/{order_id}")

    def cancel_all_orders(self) -> list:
        """Cancel all open orders."""
        return self._request("DELETE", f"{self.base_url}/v2/orders")

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str = None,
        end: str = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical bars for a symbol.
        Returns a DataFrame with columns: open, high, low, close, volume, vwap
        """
        params = {"timeframe": timeframe, "limit": limit, "adjustment": "split", "feed": "iex"}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        all_bars = []
        next_page_token = None

        while True:
            if next_page_token:
                params["page_token"] = next_page_token

            data = self._request(
                "GET",
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                params=params,
            )
            bars = data.get("bars", [])
            if not bars:
                break
            all_bars.extend(bars)
            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.rename(columns={
            "t": "timestamp", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume", "vw": "vwap",
        })
        df = df.set_index("timestamp").sort_index()

        # Drop any extra columns
        keep_cols = ["open", "high", "low", "close", "volume", "vwap"]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df

    def get_multi_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: str = None,
        end: str = None,
        limit: int = 1000,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols. Returns dict of symbol -> DataFrame."""
        results = {}
        total = len(symbols)

        # Alpaca supports multi-bar endpoint, batch in groups of 20
        for i in range(0, total, 20):
            batch = symbols[i : i + 20]

            # Progress indicator
            if show_progress and total > 20:
                done = min(i + 20, total)
                pct = done / total * 100
                bar_len = 30
                filled = int(bar_len * done / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r  Fetching: {bar} {done}/{total} stocks ({pct:.0f}%)")
                sys.stdout.flush()

            params = {
                "symbols": ",".join(batch),
                "timeframe": timeframe,
                "limit": limit,
                "adjustment": "split",
                "feed": "iex",
            }
            if start:
                params["start"] = start
            if end:
                params["end"] = end

            all_data = {}
            next_page_token = None

            while True:
                if next_page_token:
                    params["page_token"] = next_page_token

                data = self._request(
                    "GET",
                    f"{self.data_url}/v2/stocks/bars",
                    params=params,
                )
                bars_dict = data.get("bars", {})
                for sym, bars in bars_dict.items():
                    if sym not in all_data:
                        all_data[sym] = []
                    all_data[sym].extend(bars)

                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break

            for sym, bars in all_data.items():
                if bars:
                    df = pd.DataFrame(bars)
                    df["t"] = pd.to_datetime(df["t"])
                    df = df.rename(columns={
                        "t": "timestamp", "o": "open", "h": "high",
                        "l": "low", "c": "close", "v": "volume", "vw": "vwap",
                    })
                    df = df.set_index("timestamp").sort_index()
                    keep_cols = ["open", "high", "low", "close", "volume", "vwap"]
                    df = df[[c for c in keep_cols if c in df.columns]]
                    results[sym] = df

            # Small delay to be courteous to API
            if i + 20 < len(symbols):
                time.sleep(0.3)

        # Clear progress line
        if show_progress and total > 20:
            sys.stdout.write(f"\r  Fetching: {'█' * 30} {total}/{total} stocks (100%)  \n")
            sys.stdout.flush()

        return results

    def get_latest_trade(self, symbol: str) -> dict:
        """Get the latest trade for a symbol."""
        data = self._request(
            "GET", f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
        )
        return data.get("trade", {})

    def get_latest_quotes(self, symbols: List[str]) -> dict:
        """Get latest quotes for multiple symbols."""
        params = {"symbols": ",".join(symbols)}
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/quotes/latest", params=params
        )

    # ── Market Status ─────────────────────────────────────────────────────────

    def get_clock(self) -> dict:
        """Get market clock (open/close times, is_open)."""
        return self._request("GET", f"{self.base_url}/v2/clock")

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.get_clock()
        return clock.get("is_open", False)

    def get_calendar(self, start: str = None, end: str = None) -> list:
        """Get market calendar."""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._request(
            "GET", f"{self.base_url}/v2/calendar", params=params
        )

    # ── Assets ────────────────────────────────────────────────────────────────

    def get_tradable_assets(self) -> List[dict]:
        """Get list of tradable US equity assets."""
        assets = self._request(
            "GET",
            f"{self.base_url}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        )
        return [a for a in assets if a.get("tradable", False)]
