#!/usr/bin/env python3
"""
Training pipeline for GC Gold Futures RL system.

Run prepare_data.py first to build all processed files, then:
    python scripts/train.py

This script loads pre-built numpy arrays (instant) and trains 4 PPO agents:
    generalist, trend specialist, range_sr specialist, volatility specialist

ANTI-LEAKAGE CHECKLIST:
  [x] All features pre-computed by prepare_data.py (fit on TRAIN only)
  [x] Numpy arrays loaded directly -- no live normalization with test data
  [x] Regime labels computed per split, no cross-split information
  [x] Train/val/test arrays are strictly time-ordered
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ppo_agent import train_specialist
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="GC RL -- PPO Training")
    p.add_argument("--data-dir",       default="data/processed",
                   help="Directory with pre-built numpy arrays from prepare_data.py")
    p.add_argument("--env-config",     default="configs/env_config.yaml")
    p.add_argument("--ensemble-config",default="configs/ensemble_config.yaml")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--timesteps",      type=int, default=500_000,
                   help="PPO timesteps per agent")
    p.add_argument("--agents",         nargs="+",
                   default=["generalist", "trend", "range_sr", "volatility"],
                   help="Which agents to train")
    return p.parse_args()


def load_split_arrays(data_dir: Path, split: str):
    """Load features, prices, regime for one split. Returns (feats, prices, regime)."""
    feat_path   = data_dir / f"features_{split}.npy"
    price_path  = data_dir / f"prices_{split}.npy"
    regime_path = data_dir / f"regime_{split}.npy"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found.\n"
            "Run prepare_data.py first:\n"
            "  python scripts/prepare_data.py"
        )

    feats  = np.load(str(feat_path))
    prices = np.load(str(price_path))
    regime = np.load(str(regime_path))
    return feats, prices, regime


def main():
    setup_logging()
    args = parse_args()

    env_cfg      = load_config(args.env_config)
    ensemble_cfg = load_config(args.ensemble_config)
    data_dir     = Path(args.data_dir)

    # ------------------------------------------------------------------ #
    # Load pre-built arrays (instant -- no feature recomputation)
    # ------------------------------------------------------------------ #
    logger.info("Loading pre-built feature arrays from %s ...", data_dir)
    train_f, train_p, train_r = load_split_arrays(data_dir, "train")
    val_f,   val_p,   val_r   = load_split_arrays(data_dir, "val")

    logger.info("Train: features=%s  prices=%s  regime=%s",
                train_f.shape, train_p.shape, train_r.shape)
    logger.info("Val  : features=%s  prices=%s  regime=%s",
                val_f.shape, val_p.shape, val_r.shape)

    if len(val_f) == 0:
        logger.warning("Val split is empty -- using first 500 training bars for eval")
        val_f  = train_f[:500]
        val_p  = train_p[:500]
        val_r  = train_r[:500]

    # ------------------------------------------------------------------ #
    # Train each specialist agent
    # ------------------------------------------------------------------ #
    for name in args.agents:
        logger.info("=" * 50)
        train_specialist(
            name=name,
            train_features=train_f,
            train_prices=train_p,
            val_features=val_f,
            val_prices=val_p,
            env_config=env_cfg,
            ensemble_config=ensemble_cfg,
            regime_labels_train=train_r,
            regime_labels_val=val_r,
            checkpoint_dir=args.checkpoint_dir,
            total_timesteps=args.timesteps,
        )

    logger.info("=" * 50)
    logger.info("All agents trained.")
    logger.info("Next: python scripts/replay_consistency_check.py --test-day YYYY-MM-DD")


if __name__ == "__main__":
    main()
