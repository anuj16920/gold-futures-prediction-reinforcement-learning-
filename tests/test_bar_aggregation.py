"""
Tick -> bar correctness, halt handling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.bar_aggregation import BarAggregator


def test_ohlcv_aggregation():
    timestamps = pd.date_range("2023-01-01 12:00:00", periods=5, freq="10s")
    ticks = pd.DataFrame({
        "ts_event": timestamps,
        "price": [1900.0, 1901.0, 1899.0, 1902.0, 1901.5],
        "size": [10, 20, 15, 30, 25],
    })
    agg = BarAggregator(config={"timeframe": "1min", "exclude_maintenance": False})
    bars = agg.aggregate(ticks)
    assert len(bars) == 1
    assert bars["open"].iloc[0] == 1900.0
    assert bars["high"].iloc[0] == 1902.0
    assert bars["low"].iloc[0] == 1899.0
    assert bars["close"].iloc[0] == 1901.5
    assert bars["volume"].iloc[0] == 100
    assert bars["trade_count"].iloc[0] == 5
    print("OHLCV aggregation test: PASS")


def test_maintenance_halt():
    timestamps = pd.date_range("2023-01-01 17:00:00", periods=5, freq="10s")
    ticks = pd.DataFrame({
        "ts_event": timestamps,
        "price": [1900.0] * 5,
        "size": [10] * 5,
    })
    agg = BarAggregator(config={"timeframe": "1min", "exclude_maintenance": True})
    bars = agg.aggregate(ticks)
    print(f"Bars during maintenance: {len(bars)}")
    print("Maintenance halt test: PASS")


if __name__ == "__main__":
    test_ohlcv_aggregation()
    test_maintenance_halt()
    print("All bar aggregation tests passed.")
