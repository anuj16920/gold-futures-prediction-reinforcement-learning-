"""Incremental feature computer for live trading.

Maintains a rolling window of completed bars and recomputes features
on each new bar using the SAME logic as the batch FeaturePipeline.
Loads pre-fitted normalization stats from disk so live normalization
exactly matches the training normalization (no leakage from live data).

Anti-leakage: the window at time t contains bars [..., t-2, t-1, t].
Features at t use only indices ≤ t. The MTF shift(1) and SR pivot
confirmation lag are preserved because we compute via FeaturePipeline.
"""
import logging
from collections import deque
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class IncrementalFeatureComputer:
    """Maintain rolling bar buffer and compute features incrementally."""

    def __init__(self, config: dict, norm_stats_path: str,
                 window_size: int = 200):
        from src.features.pipeline import FeaturePipeline
        self.pipeline = FeaturePipeline(config)
        self.pipeline.load_norm_stats(norm_stats_path)
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)

    def update(self, bar: dict) -> Optional[pd.Series]:
        """Add one completed bar and return its normalized feature vector."""
        self._buffer.append(bar)
        if len(self._buffer) < 2:
            return None

        bars_df = pd.DataFrame(list(self._buffer))
        # Ensure timestamp column exists
        if "timestamp" not in bars_df.columns and bars_df.columns[0] != "timestamp":
            bars_df = bars_df.rename(columns={bars_df.columns[0]: "timestamp"})

        try:
            features_df = self.pipeline.transform(bars_df)
        except Exception as exc:
            logger.warning("Feature computation failed: %s", exc)
            return None

        # Return the last row — features for the most recent bar
        return features_df.iloc[-1]

    @property
    def feature_names(self) -> list:
        return self.pipeline.feature_names

    @property
    def n_features(self) -> int:
        return self.pipeline.n_features
