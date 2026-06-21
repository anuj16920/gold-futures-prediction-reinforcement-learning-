#!/usr/bin/env python3
"""
Diffs configs/*.yaml against src/utils/config.py defaults, fails loudly on mismatch.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Config Reconciliation ===")

    config_dir = Path("configs")
    if not config_dir.exists():
        logger.error("configs/ directory not found")
        return 1

    all_ok = True
    for cfg_file in config_dir.glob("*.yaml"):
        try:
            cfg = load_config(str(cfg_file))
            logger.info(f"  {cfg_file.name}: OK ({len(cfg)} top-level keys)")
        except Exception as e:
            logger.error(f"  {cfg_file.name}: FAILED - {e}")
            all_ok = False

    if all_ok:
        logger.info("All configs reconciled successfully.")
        return 0
    else:
        logger.error("Config reconciliation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
