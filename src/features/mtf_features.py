"""Multi-timeframe (MTF) feature engineering.

ANTI-LEAKAGE DESIGN:
At 1-min bar t, the N-min bar that contains t may still be open.
Example: at 09:03, the 5-min bar [09:00, 09:05) is not yet closed.
Using its close would require knowing prices at 09:03–09:04 (future).

Fix — three steps:
1. resample(N, label='left', closed='left') → bar at T uses ticks in [T, T+N)
   (the bar's CLOSE is the last 1-min close in the period, i.e., at T+N-1min)
2. .shift(1) on the resampled series → at time T we only see the bar from
   [T-N, T), which fully closed at T-1min. No bar currently in progress.
3. .reindex(1min_index, method='ffill') → every 1-min bar carries forward
   the most recent closed N-min value.

Verification: at 1-min bar 09:03, ffill gives us the shifted bar at 09:00,
which was the close of [08:55, 09:00) — the LAST fully-closed 5-min bar. ✓
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MTFFeatureEngineer:
    """Compute multi-timeframe return and volume features without lookahead."""

    def __init__(self, config: dict):
        self.timeframes = config.get("mtf_timeframes", ["5min", "15min", "30min", "1H", "4H"])

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = self._to_ts_index(bars)
        result = pd.DataFrame(index=df.index)

        for tf in self.timeframes:
            try:
                feats = self._compute_tf(df, tf)
                result = result.join(feats, how="left")
            except Exception as exc:
                logger.warning("MTF compute failed for tf=%s: %s", tf, exc)

        # Restore timestamp column
        result = result.reset_index().rename(columns={"index": "timestamp"})
        if result.columns[0] != "timestamp":
            result = result.rename(columns={result.columns[0]: "timestamp"})
        return result

    # ------------------------------------------------------------------ #

    def _compute_tf(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        close = df["close"]
        has_vol = "volume" in df.columns

        # Step 1: resample — bar at T uses data in [T, T+N)
        resampled_close = close.resample(tf, label="left", closed="left").last()
        if has_vol:
            resampled_vol = df["volume"].resample(tf, label="left", closed="left").sum()

        # Compute return of each completed N-min bar
        tf_ret = np.log(resampled_close / resampled_close.shift(1))

        # Step 2: shift(1) — only the PREVIOUS completed N-min bar is visible
        tf_ret_lagged = tf_ret.shift(1)
        if has_vol:
            vol_lagged = resampled_vol.shift(1)

        # Step 3: reindex back to 1-min with forward-fill
        out = pd.DataFrame(index=df.index)
        out[f"mtf_{tf}_ret_1"] = tf_ret_lagged.reindex(df.index, method="ffill")
        if has_vol:
            out[f"mtf_{tf}_vol"] = vol_lagged.reindex(df.index, method="ffill")

        return out

    def _to_ts_index(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("bars must have a 'timestamp' column or DatetimeIndex")
        return df.sort_index()
