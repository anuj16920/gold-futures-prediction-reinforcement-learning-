"""Base 1-minute bar features — strictly causal (no lookahead).

All computations use only bar t and earlier:
- Returns/MA/Vol/ATR  : rolling on past bars
- RSI                 : Wilder EMA on past gains/losses
- MACD                : EWM on past prices
- Bollinger           : rolling mean/std on past bars
- Volume ratio        : volume / rolling mean volume
- Time features       : sin/cos encoding of hour-of-day (deterministic, not future)
"""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Returns / Moving Averages / Volatility / ATR
# ------------------------------------------------------------------ #

def compute_returns(close: pd.Series, lookbacks: list) -> pd.DataFrame:
    return pd.DataFrame(
        {f"ret_{lb}": np.log(close / close.shift(lb)) for lb in lookbacks},
        index=close.index,
    )


def compute_ma_features(close: pd.Series, windows: list) -> pd.DataFrame:
    out = {}
    for w in windows:
        ma = close.rolling(w, min_periods=1).mean()
        out[f"ma_{w}_ratio"] = (close / ma) - 1.0
    return pd.DataFrame(out, index=close.index)


def compute_volatility(close: pd.Series, window: int) -> pd.Series:
    return (
        close.pct_change().rolling(window, min_periods=2).std().rename(f"vol_{window}")
    )


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().rename(f"atr_{window}")


# ------------------------------------------------------------------ #
# RSI (Wilder smoothing — causal EWM)
# ------------------------------------------------------------------ #

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index. Uses Wilder EWM (alpha=1/window)."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rs  = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return ((rsi / 100.0) - 0.5).rename(f"rsi_{window}")  # normalise to [-0.5, 0.5]


# ------------------------------------------------------------------ #
# MACD
# ------------------------------------------------------------------ #

def compute_macd(close: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram — all causal EWM."""
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = (ema_fast - ema_slow) / close  # price-normalised
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=close.index,
    )


# ------------------------------------------------------------------ #
# Bollinger Bands
# ------------------------------------------------------------------ #

def compute_bollinger(close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Bollinger %B (position within bands) and bandwidth. Both causal."""
    ma  = close.rolling(window, min_periods=2).mean()
    std = close.rolling(window, min_periods=2).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb_pct   = (close - lower) / (upper - lower + 1e-9)  # 0=lower band, 1=upper
    bb_width = (upper - lower) / (ma + 1e-9)             # relative bandwidth
    # Centre bb_pct at 0
    return pd.DataFrame(
        {"bb_pct": bb_pct - 0.5, "bb_width": bb_width},
        index=close.index,
    )


# ------------------------------------------------------------------ #
# Volume ratio
# ------------------------------------------------------------------ #

def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume vs rolling mean — measures unusual activity."""
    avg = volume.rolling(window, min_periods=1).mean()
    return (volume / avg.replace(0, 1)).rename(f"vol_ratio_{window}")


# ------------------------------------------------------------------ #
# Time-of-day features (deterministic, never future)
# ------------------------------------------------------------------ #

def compute_time_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Sine/cosine encoding of hour-of-day (UTC). Fully deterministic."""
    if timestamps.tz is None:
        ts = timestamps.tz_localize("UTC")
    else:
        ts = timestamps.tz_convert("UTC")
    hour_frac = ts.hour + ts.minute / 60.0
    period    = 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour_frac / period),
            "hour_cos": np.cos(2 * np.pi * hour_frac / period),
        },
        index=timestamps,
    )
