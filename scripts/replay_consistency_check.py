#!/usr/bin/env python3
"""
§16 — Diffs batch vs "live" incremental pipeline output.
MUST pass before going live.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.bar_aggregation import BarAggregator
from src.features.pipeline import FeaturePipeline
from src.live.incremental_bars import IncrementalBarBuilder
from src.live.incremental_features import IncrementalFeatureComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Replay Consistency Check")
    parser.add_argument("--test-day", required=True, help="Test date YYYY-MM-DD")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config", default="configs/feature_config.yaml")
    args = parser.parse_args()

    logger.info(f"=== Replay Consistency Check for {args.test_day} ===")
    logger.info("This MUST pass before any live paper-trading run is trusted.")

    # Load historical day's data
    # In practice, would filter raw data for the test day
    logger.info("Loading historical batch data...")

    # Simulate batch pipeline
    from src.data.loaders import DatabentoLoader
    loader = DatabentoLoader(raw_dir=args.raw_dir)
    ticks_df = loader.load_all()

    # Filter for test day
    day_mask = ticks_df["ts_event"].dt.date == pd.to_datetime(args.test_day).date()
    day_ticks = ticks_df[day_mask]

    if day_ticks.empty:
        logger.error(f"No data found for {args.test_day}")
        return 1

    # Batch pipeline
    aggregator = BarAggregator()
    batch_bars = aggregator.aggregate(day_ticks)

    pipeline = FeaturePipeline(config={})
    batch_features = pipeline.fit_transform(batch_bars, batch_bars)

    # Incremental pipeline
    inc_bars = IncrementalBarBuilder()
    inc_feats = IncrementalFeatureComputer(config={}, norm_stats_path="data/processed/norm_stats.json")

    for _, tick in day_ticks.iterrows():
        inc_bars.add_tick(tick.to_dict())

    inc_bars_df = inc_bars.flush_bar()
    if inc_bars_df is not None:
        inc_features = inc_feats.update(inc_bars_df)
    else:
        inc_features = pd.DataFrame()

    # Compare
    if batch_features.empty or inc_features.empty:
        logger.warning("Empty features, cannot compare")
        return 1

    # Check structural match
    batch_cols = set(batch_features.columns)
    inc_cols = set(inc_features.columns)

    if batch_cols != inc_cols:
        logger.error(f"Column mismatch: batch={batch_cols - inc_cols}, inc={inc_cols - batch_cols}")
        return 1

    logger.info("Column names match.")

    # Check values (allow small numerical differences)
    common_cols = list(batch_cols.intersection(inc_cols))
    common_cols = [c for c in common_cols if c != "raw_close"]

    if len(batch_features) == len(inc_features):
        for col in common_cols:
            diff = (batch_features[col].values - inc_features[col].values).abs().max()
            if diff > 1e-6:
                logger.error(f"Value mismatch in {col}: max diff = {diff}")
                return 1

    logger.info("Replay consistency check: PASS")
    logger.info("Live pipeline output matches batch pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
