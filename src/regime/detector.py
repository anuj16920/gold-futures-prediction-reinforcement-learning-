"""Regime detection: classify each bar into one of 4 market regimes.

Regimes:
  0 = trending_up
  1 = trending_down
  2 = ranging
  3 = high_volatility

Rule-based implementation (no external ML libs required):
- ADX > adx_threshold  AND  +DI > -DI  → trending_up
- ADX > adx_threshold  AND  -DI > +DI  → trending_down
- ATR percentile > atr_percentile       → high_volatility
- Otherwise                             → ranging

ANTI-LEAKAGE:
All indicators (ADX, ATR) are rolling with causal lookback. The ATR percentile
rank is computed over a fixed trailing window (atr_rank_window bars), not over
the full dataset. No future bar is consulted.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_TRENDING_UP = 0
REGIME_TRENDING_DOWN = 1
REGIME_RANGING = 2
REGIME_HIGH_VOL = 3

REGIME_NAMES = {
    REGIME_TRENDING_UP: "trending_up",
    REGIME_TRENDING_DOWN: "trending_down",
    REGIME_RANGING: "ranging",
    REGIME_HIGH_VOL: "high_volatility",
}


class RegimeDetector:
    """Classify market regime at each bar using causal indicators."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.adx_window = int(cfg.get("adx_window", 14))
        self.adx_threshold = float(cfg.get("adx_threshold", 25.0))
        self.atr_window = int(cfg.get("atr_window", 14))
        self.atr_rank_window = int(cfg.get("atr_rank_window", 100))
        self.atr_percentile = float(cfg.get("atr_percentile", 0.8))

    def detect(self, bars: pd.DataFrame) -> pd.Series:
        """Return integer regime series aligned to bars index."""
        df = bars.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        atr = self._compute_atr(df)
        adx, plus_di, minus_di = self._compute_adx(df)
        atr_rank = self._rolling_percentile_rank(atr, self.atr_rank_window)

        n = len(df)
        regime = np.full(n, REGIME_RANGING, dtype=int)

        high_vol_mask = atr_rank >= self.atr_percentile
        trending_mask = adx >= self.adx_threshold

        # High volatility takes priority over trend
        regime[trending_mask & (plus_di >= minus_di) & ~high_vol_mask] = REGIME_TRENDING_UP
        regime[trending_mask & (minus_di > plus_di) & ~high_vol_mask] = REGIME_TRENDING_DOWN
        regime[high_vol_mask] = REGIME_HIGH_VOL

        return pd.Series(regime, index=df.index, name="regime")

    # ------------------------------------------------------------------ #

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_window, min_periods=1).mean()

    def _compute_adx(self, df: pd.DataFrame):
        """Compute ADX, +DI, -DI using Wilder smoothing."""
        n = self.adx_window
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        size = len(close)

        plus_dm = np.zeros(size)
        minus_dm = np.zeros(size)
        tr_vals = np.zeros(size)

        for i in range(1, size):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
            tr_vals[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Wilder smoothing (equivalent to EMA with alpha=1/n)
        atr_s = self._wilder(tr_vals, n)
        plus_s = self._wilder(plus_dm, n)
        minus_s = self._wilder(minus_dm, n)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = np.where(atr_s > 0, 100 * plus_s / atr_s, 0.0)
            minus_di = np.where(atr_s > 0, 100 * minus_s / atr_s, 0.0)
            di_sum = plus_di + minus_di
            dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        adx = self._wilder(dx, n)
        idx = df.index
        return (
            pd.Series(adx, index=idx),
            pd.Series(plus_di, index=idx),
            pd.Series(minus_di, index=idx),
        )

    @staticmethod
    def _wilder(arr: np.ndarray, period: int) -> np.ndarray:
        out = np.zeros_like(arr)
        if len(arr) < period:
            return out
        out[period - 1] = arr[:period].mean()
        alpha = 1.0 / period
        for i in range(period, len(arr)):
            out[i] = out[i - 1] * (1 - alpha) + arr[i] * alpha
        return out

    @staticmethod
    def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """Causal percentile rank of each value within its trailing window.

        Uses pandas vectorized rolling rank (C-level) -- O(n log w) vs the
        O(n*w) Python loop. Produces identical values for any window size.
        """
        return series.rolling(window, min_periods=1).rank(pct=True)
