"""Feature pipeline: compose all feature engineers and normalize.

ANTI-LEAKAGE for normalization:
  fit(train_bars)        → compute mean/std on TRAINING data only
  transform(any_bars)    → apply pre-fitted stats (no peeking at future)
  fit_transform(train, target) → fit on train, transform target

DO NOT call fit_transform(all_data, all_data). Always split first.

After fitting, call save_norm_stats() so the live IncrementalFeatureComputer
can load the exact same normalization parameters.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.base_features import (
    compute_atr,
    compute_ma_features,
    compute_returns,
    compute_volatility,
)
from src.features.mtf_features import MTFFeatureEngineer
from src.features.sr_features import SRFeatureEngineer

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Compose all feature engineers with leak-free normalization."""

    def __init__(self, config: dict):
        self.cfg = config
        self.return_lookbacks = config.get("return_lookbacks", [1, 5, 20])
        self.ma_windows = config.get("ma_windows", [20])
        self.vol_window = int(config.get("vol_window", 20))
        self.atr_window = int(config.get("atr_window", 14))
        self.clip_std = float(config.get("clip_std", 5.0))
        self.mtf = MTFFeatureEngineer(config)
        self.sr = SRFeatureEngineer(config)
        self.feature_mean_: Optional[pd.Series] = None
        self.feature_std_: Optional[pd.Series] = None
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, train_bars: pd.DataFrame) -> "FeaturePipeline":
        """Compute normalization stats from TRAINING data only."""
        raw = self._compute_raw(train_bars)
        self.feature_mean_ = raw.mean()
        self.feature_std_ = raw.std().replace(0.0, 1.0)
        self.fitted_ = True
        logger.info("Fitted on %d bars -> %d features", len(train_bars), len(self.feature_mean_))
        return self

    def transform(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Normalize using pre-fitted stats. No leakage: stats were set by fit()."""
        if not self.fitted_:
            raise RuntimeError("Call fit(train_bars) before transform().")
        raw = self._compute_raw(bars)
        raw = raw.reindex(columns=self.feature_mean_.index, fill_value=0.0)
        normed = (raw - self.feature_mean_) / self.feature_std_
        return normed.clip(-self.clip_std, self.clip_std)

    def fit_transform(self, train_bars: pd.DataFrame, transform_bars: pd.DataFrame) -> pd.DataFrame:
        """Fit on train_bars, then transform transform_bars."""
        return self.fit(train_bars).transform(transform_bars)

    def save_norm_stats(self, path: str) -> None:
        if not self.fitted_:
            raise RuntimeError("Pipeline not fitted.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "mean": self.feature_mean_.to_dict(),
            "std": self.feature_std_.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved norm stats -> %s", path)

    def load_norm_stats(self, path: str) -> "FeaturePipeline":
        with open(path) as f:
            stats = json.load(f)
        self.feature_mean_ = pd.Series(stats["mean"])
        self.feature_std_ = pd.Series(stats["std"])
        self.fitted_ = True
        return self

    @property
    def feature_names(self) -> list:
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        return list(self.feature_mean_.index)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _compute_raw(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute all raw (un-normalized) features from OHLCV bars."""
        df = bars.copy()

        # Ensure DatetimeIndex
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df = df.sort_index()

        close = df["close"]
        high = df["high"]
        low = df["low"]

        parts = [
            compute_returns(close, self.return_lookbacks),
            compute_ma_features(close, self.ma_windows),
            compute_volatility(close, self.vol_window).to_frame(),
            compute_atr(high, low, close, self.atr_window).to_frame(),
        ]

        # MTF features (anti-leakage: shift(1) inside MTFFeatureEngineer)
        mtf_input = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        mtf_raw = self.mtf.compute(mtf_input)
        ts_col = mtf_raw.columns[0]
        mtf_raw = mtf_raw.set_index(ts_col).reindex(df.index)
        parts.append(mtf_raw)

        # SR features (anti-leakage: pivot_window lag inside SRFeatureEngineer)
        sr_input = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        sr_raw = self.sr.compute(sr_input)
        sr_raw = sr_raw.set_index(sr_raw.columns[0]) if "timestamp" not in sr_raw.index.names else sr_raw
        # Handle case where sr_raw doesn't have timestamp index
        if len(sr_raw) == len(df):
            sr_raw.index = df.index
        else:
            sr_raw = sr_raw.reindex(df.index)
        parts.append(sr_raw)

        combined = pd.concat(parts, axis=1)

        # Fill NaNs caused by warm-up periods: ffill then 0 (safe — only fills
        # historical NaN from insufficient lookback, not future values)
        combined = combined.ffill().fillna(0.0)
        return combined
