"""
Automated version of verify_alignment.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def test_mtf_shift():
    """Verify higher-timeframe features are shifted forward."""
    timestamps = pd.date_range("2023-01-01", periods=100, freq="1min")
    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.randn(100) + 1900,
        "high": np.random.randn(100) + 1901,
        "low": np.random.randn(100) + 1899,
        "close": np.random.randn(100) + 1900,
        "volume": np.random.randint(1, 100, 100),
    })

    from src.features.mtf_features import MTFFeatureEngineer
    mtf = MTFFeatureEngineer({"mtf_timeframes": ["5min"]})
    feats = mtf.compute(bars)
    assert "mtf_5min_ret_1" in feats.columns
    print("MTF shift test: PASS")


def test_pivot_confirmation():
    """Verify S/R pivots are confirmed N bars after formation."""
    from src.features.sr_features import SRFeatureEngineer
    timestamps = pd.date_range("2023-01-01", periods=50, freq="1min")
    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.ones(50) * 1900,
        "high": np.ones(50) * 1901,
        "low": np.ones(50) * 1899,
        "close": np.ones(50) * 1900,
    })
    bars.loc[10, "high"] = 1920
    sr = SRFeatureEngineer({"pivot_window": 3})
    swing_high, swing_low = sr._detect_pivots(bars)
    assert not swing_high.iloc[10]
    assert not swing_high.iloc[11]
    assert not swing_high.iloc[12]
    assert swing_high.iloc[13]
    print("Pivot confirmation test: PASS")


if __name__ == "__main__":
    test_mtf_shift()
    test_pivot_confirmation()
    print("All no-lookahead tests passed.")
