"""Base 1-minute bar features: returns, moving averages, volatility, ATR.

All computations are strictly causal (use only bar t and bars before it).
- Log returns at lookback k:  log(close[t] / close[t-k])   — shift(k) ≥ 1 ✓
- MA ratio:   close[t] / rolling_mean(close, window)[t] - 1 — rolling uses past ✓
- Realized vol: rolling std of pct_change                   — rolling uses past ✓
- ATR:         rolling mean of True Range                   — rolling uses past ✓
"""
import numpy as np
import pandas as pd


def compute_returns(close: pd.Series, lookbacks: list) -> pd.DataFrame:
    out = {}
    for lb in lookbacks:
        out[f"ret_{lb}"] = np.log(close / close.shift(lb))
    return pd.DataFrame(out, index=close.index)


def compute_ma_features(close: pd.Series, windows: list) -> pd.DataFrame:
    out = {}
    for w in windows:
        ma = close.rolling(w, min_periods=1).mean()
        out[f"ma_{w}_ratio"] = (close / ma) - 1.0
    return pd.DataFrame(out, index=close.index)


def compute_volatility(close: pd.Series, window: int) -> pd.Series:
    return (
        close.pct_change()
        .rolling(window, min_periods=2)
        .std()
        .rename(f"vol_{window}")
    )


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().rename(f"atr_{window}")
