"""Support/Resistance pivot feature engineering.

ANTI-LEAKAGE — pivot confirmation lag:
A swing high at bar i requires high[i] > high[i±k] for k in 1..pivot_window.
The right-hand check (k > 0) requires data from bars i+1 … i+pivot_window,
which are FUTURE at bar i.

Fix: the pivot is only CONFIRMED at bar i + pivot_window (the last bar
we needed to observe). Before that bar, swing_high/low remains False.

Example with pivot_window=3, spike at bar 10:
  bars 10, 11, 12 → not yet confirmed (need to see bars 11, 12, 13)
  bar  13         → swing_high[13] = True  ✓
  bars 14+        → carries as SR level via forward-fill
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SRFeatureEngineer:
    """Compute Support/Resistance features with correct pivot confirmation lag."""

    def __init__(self, config: dict):
        self.pivot_window = int(config.get("pivot_window", 5))
        self.atr_window = int(config.get("atr_window", 14))
        self.test_threshold_ticks = int(config.get("test_threshold_ticks", 5))
        self.tick_size = float(config.get("tick_size", 0.10))

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        ts_col = "timestamp" if "timestamp" in df.columns else df.index.name or "timestamp"

        swing_high, swing_low = self._detect_pivots(df)
        close = df["close"].values
        close_s = df["close"]

        last_sh = self._last_confirmed_level(swing_high, df["high"])
        last_sl = self._last_confirmed_level(swing_low, df["low"])

        threshold = self.test_threshold_ticks * self.tick_size
        result = pd.DataFrame(index=df.index)
        result["dist_to_resistance"] = (last_sh - close_s) / close_s
        result["dist_to_support"] = (close_s - last_sl) / close_s
        result["near_resistance"] = ((last_sh - close_s).abs() < threshold).astype(float)
        result["near_support"] = ((close_s - last_sl).abs() < threshold).astype(float)
        return result

    def _detect_pivots(self, bars: pd.DataFrame):
        """Return (swing_high, swing_low) boolean Series with confirmation lag.

        swing_high[i + pivot_window] = True  iff  bar i was a local maximum
        """
        n = len(bars)
        highs = bars["high"].values
        lows = bars["low"].values
        pw = self.pivot_window

        sh = np.zeros(n, dtype=bool)
        sl = np.zeros(n, dtype=bool)

        for i in range(pw, n - pw):
            # Left side: bars i-pw .. i-1 must all be lower than bars[i]
            left_high = highs[i - pw: i]
            right_high = highs[i + 1: i + pw + 1]
            if highs[i] > left_high.max() and highs[i] > right_high.max():
                sh[i + pw] = True  # confirmed at i+pw

            left_low = lows[i - pw: i]
            right_low = lows[i + 1: i + pw + 1]
            if lows[i] < left_low.min() and lows[i] < right_low.min():
                sl[i + pw] = True

        return (
            pd.Series(sh, index=bars.index, dtype=bool),
            pd.Series(sl, index=bars.index, dtype=bool),
        )

    def _last_confirmed_level(self, pivot_mask: pd.Series, price_series: pd.Series) -> pd.Series:
        """Forward-fill the price level of the most recent confirmed pivot."""
        levels = price_series.where(pivot_mask)
        return levels.ffill().fillna(price_series)
