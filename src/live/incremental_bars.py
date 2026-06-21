"""Incremental tick-by-tick bar builder for live trading.

Mirrors BarAggregator exactly — same OHLCV logic, same maintenance filter.
flush_bar() returns a completed bar dict only when the current 1-min window
closes; returns None otherwise.
"""
import logging
from datetime import timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MAINT_START_UTC = 22
_MAINT_END_UTC = 23


class IncrementalBarBuilder:
    """Build OHLCV bars one tick at a time."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.timeframe = cfg.get("timeframe", "1min")
        self.exclude_maintenance = cfg.get("exclude_maintenance", True)
        self._freq = pd.tseries.frequencies.to_offset(self.timeframe)
        self._current_bar_start: Optional[pd.Timestamp] = None
        self._open = self._high = self._low = self._close = None
        self._volume = 0
        self._trade_count = 0
        self._completed_bars: list = []

    def add_tick(self, tick: dict) -> Optional[dict]:
        """Process one tick. Returns completed bar dict if a bar just closed."""
        ts = tick.get("ts_event") or tick.get("timestamp")
        if ts is None:
            return None
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        if self.exclude_maintenance and self._in_maintenance(ts):
            return None

        bar_start = ts.floor(self.timeframe)

        if self._current_bar_start is None:
            self._start_bar(bar_start, tick)
            return None

        if bar_start > self._current_bar_start:
            completed = self._finalize_bar()
            self._start_bar(bar_start, tick)
            return completed

        # Same bar — update
        price = float(tick["price"])
        size = int(tick.get("size", 0))
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._volume += size
        self._trade_count += 1
        return None

    def flush_bar(self) -> Optional[dict]:
        """Force-flush the current incomplete bar (call at session end)."""
        if self._current_bar_start is None:
            return None
        bar = self._finalize_bar()
        self._current_bar_start = None
        return bar

    # ------------------------------------------------------------------ #

    def _start_bar(self, bar_start: pd.Timestamp, tick: dict):
        price = float(tick["price"])
        self._current_bar_start = bar_start
        self._open = self._high = self._low = self._close = price
        self._volume = int(tick.get("size", 0))
        self._trade_count = 1

    def _finalize_bar(self) -> dict:
        return {
            "timestamp": self._current_bar_start,
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._volume,
            "trade_count": self._trade_count,
        }

    def _in_maintenance(self, ts: pd.Timestamp) -> bool:
        hour = ts.tz_convert("UTC").hour
        return _MAINT_START_UTC <= hour < _MAINT_END_UTC
