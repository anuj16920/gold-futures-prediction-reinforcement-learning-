"""
Contract-roll stitching correctness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.roll_logic import VolumeBasedRoller, PanamaBackAdjuster


def test_volume_based_roll():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    ticks = []
    for i, date in enumerate(dates):
        symbol = "GCZ3" if i < 5 else "GCG4"
        vol = 1000 if symbol == "GCZ3" and i < 5 else 2000
        ticks.append({
            "ts_event": pd.Timestamp(date),
            "symbol": symbol,
            "price": 1900.0 + i,
            "size": vol,
            "side": "A",
        })
    df = pd.DataFrame(ticks)
    roller = VolumeBasedRoller()
    continuous = roller.build_continuous(df)
    assert len(continuous) > 0
    print("Volume-based roll test: PASS")


def test_panama_adjustment():
    ticks = []
    for i in range(20):
        symbol = "GCZ3" if i < 10 else "GCG4"
        price = 1900.0 + i if i < 10 else 1920.0 + (i - 10)
        ticks.append({
            "ts_event": pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=i),
            "symbol": symbol,
            "price": price,
            "size": 100,
            "side": "A",
            "front_month": symbol,
        })
    df = pd.DataFrame(ticks)
    adjuster = PanamaBackAdjuster()
    adjusted = adjuster.apply(df)
    assert "price_adj" in adjusted.columns
    print("Panama adjustment test: PASS")


if __name__ == "__main__":
    test_volume_based_roll()
    test_panama_adjustment()
    print("All roll logic tests passed.")
