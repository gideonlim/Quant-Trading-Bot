"""
Technical indicators and feature engineering.
All computed with pandas/numpy — no external TA library needed.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Momentum ──────────────────────────────────────────────────────────────────

def momentum_return(close: pd.Series, period: int, skip_recent: int = 0) -> pd.Series:
    """
    Compute momentum as the return over `period` days,
    optionally skipping the most recent `skip_recent` days
    to avoid short-term reversal noise.

    This is the classic Jegadeesh & Titman (1993) momentum signal.
    """
    if skip_recent > 0:
        lagged = close.shift(skip_recent)
        return lagged / close.shift(period) - 1.0
    return close / close.shift(period) - 1.0


def rate_of_change(close: pd.Series, period: int) -> pd.Series:
    """Rate of change (percent change over N periods)."""
    return close.pct_change(period)


# ── Mean Reversion ────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).
    Returns values 0-100.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with columns: middle, upper, lower, pct_b, bandwidth
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (close - lower) / (upper - lower)  # %B: 0 = at lower, 1 = at upper
    bandwidth = (upper - lower) / middle

    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pct_b": pct_b,
        "bb_bandwidth": bandwidth,
    }, index=close.index)


# ── Trend ─────────────────────────────────────────────────────────────────────

def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def trend_filter(close: pd.Series, sma_period: int = 200) -> pd.Series:
    """
    Binary trend filter: 1 if price is above SMA, 0 otherwise.
    Used to avoid buying into a downtrend.
    """
    return (close > sma(close, sma_period)).astype(float)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD indicator."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    }, index=close.index)


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range — measures volatility.
    Used for position sizing and stop placement.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def historical_volatility(close: pd.Series, period: int = 21) -> pd.Series:
    """Annualized historical volatility (standard deviation of log returns)."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252)


def volatility_ratio(close: pd.Series, short_period: int = 5, long_period: int = 21) -> pd.Series:
    """
    Ratio of short-term to long-term volatility.
    Values > 1 indicate increasing volatility (potential breakout).
    """
    short_vol = historical_volatility(close, short_period)
    long_vol = historical_volatility(close, long_period)
    return short_vol / long_vol.replace(0, np.nan)


# ── Volume ────────────────────────────────────────────────────────────────────

def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return volume.rolling(period).mean()


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume relative to its moving average. > 1 means above-average volume."""
    return volume / volume_sma(volume, period)


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume flow."""
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


# ── Composite Feature Builder ─────────────────────────────────────────────────

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with OHLCV columns, compute all features.
    Returns the original DataFrame augmented with feature columns.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # Momentum
    df["mom_fast"] = momentum_return(close, 21, skip_recent=5)
    df["mom_slow"] = momentum_return(close, 126, skip_recent=5)
    df["mom_12m"] = momentum_return(close, 252, skip_recent=21)
    df["roc_5"] = rate_of_change(close, 5)
    df["roc_10"] = rate_of_change(close, 10)

    # Mean reversion
    df["rsi_14"] = rsi(close, 14)
    bb = bollinger_bands(close, 20, 2.0)
    df = pd.concat([df, bb], axis=1)

    # Trend
    df["sma_50"] = sma(close, 50)
    df["sma_200"] = sma(close, 200)
    df["trend_200"] = trend_filter(close, 200)
    df["above_sma50"] = (close > df["sma_50"]).astype(float)
    macd_df = macd(close)
    df = pd.concat([df, macd_df], axis=1)

    # Volatility
    df["atr_14"] = atr(high, low, close, 14)
    df["hvol_21"] = historical_volatility(close, 21)
    df["hvol_63"] = historical_volatility(close, 63)
    df["vol_ratio"] = volatility_ratio(close, 5, 21)

    # Volume
    df["rvol"] = relative_volume(vol, 20)
    df["obv"] = on_balance_volume(close, vol)

    return df


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score (for ranking across stocks)."""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std
