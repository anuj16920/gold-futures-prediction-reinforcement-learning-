"""Tick-to-OHLCV bar aggregation with maintenance window filtering.

No lookahead: each bar [t, t+freq) is computed from ticks with ts_event in
[t, t+freq). Bar timestamp is t (left edge of interval).
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# CME daily maintenance: 17:00–18:00 ET = 22:00–23:00 UTC
_MAINT_START_UTC = 22
_MAINT_END_UTC = 23


class BarAggregator:
    """Convert raw trade ticks to fixed-duration OHLCV bars."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.timeframe = cfg.get("timeframe", "1min")
        self.exclude_maintenance = cfg.get("exclude_maintenance", True)

    def aggregate(self, ticks: pd.DataFrame) -> pd.DataFrame:
        df = ticks.copy().sort_values("ts_event")

        if self.exclude_maintenance:
            df = self._drop_maintenance(df)

        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trade_count"])

        df = df.set_index("ts_event")

        freq = self.timeframe
        ohlcv = df["price"].resample(freq, label="left", closed="left").ohlc()
        ohlcv["volume"] = df["size"].resample(freq, label="left", closed="left").sum()
        ohlcv["trade_count"] = df["size"].resample(freq, label="left", closed="left").count()

        ohlcv = ohlcv.dropna(subset=["open"]).reset_index()
        ohlcv = ohlcv.rename(columns={"ts_event": "timestamp"})
        return ohlcv

    def _drop_maintenance(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["ts_event"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        hour = ts.dt.hour
        mask = ~((hour >= _MAINT_START_UTC) & (hour < _MAINT_END_UTC))
        return df[mask]
