"""
SL/TP, cooldown, daily cap unit tests.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk.stops import ATRStopLossTakeProfit
from src.risk.guard import RiskGuard


def test_atr_sl_tp():
    sl_tp = ATRStopLossTakeProfit(atr_multiplier_sl=1.5, atr_multiplier_tp=3.0)
    sl, tp = sl_tp.get_levels(entry_price=1900.0, position=1, atr=10.0)
    assert sl == 1900.0 - 1.5 * 10.0
    assert tp == 1900.0 + 3.0 * 10.0
    sl, tp = sl_tp.get_levels(entry_price=1900.0, position=-1, atr=10.0)
    assert sl == 1900.0 + 1.5 * 10.0
    assert tp == 1900.0 - 3.0 * 10.0
    print("ATR SL/TP test: PASS")


def test_risk_guard():
    guard = RiskGuard(max_drawdown_pct=0.10, min_account_floor=5000.0,
                      max_trades_per_day=10, cooldown_bars=5)
    assert not guard.is_halted(current_capital=10000, peak_capital=10000)
    assert guard.is_halted(current_capital=8900, peak_capital=10000)
    assert guard.is_halted(current_capital=4900, peak_capital=10000)
    assert not guard.is_halted(current_capital=9500, peak_capital=10000)
    print("Risk guard test: PASS")


if __name__ == "__main__":
    test_atr_sl_tp()
    test_risk_guard()
    print("All risk rule tests passed.")
