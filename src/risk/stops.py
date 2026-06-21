"""ATR-based stop-loss and take-profit level computation.

No future data is used: entry_price and atr are both known at entry time.
"""


class ATRStopLossTakeProfit:
    """Compute SL/TP levels from entry price, position direction, and ATR."""

    def __init__(self, atr_multiplier_sl: float = 1.5, atr_multiplier_tp: float = 3.0):
        self.sl_mult = atr_multiplier_sl
        self.tp_mult = atr_multiplier_tp

    def get_levels(self, entry_price: float, position: int, atr: float):
        """Return (stop_loss, take_profit) for a given entry.

        position: +1 for long, -1 for short
        """
        if position == 1:
            sl = entry_price - self.sl_mult * atr
            tp = entry_price + self.tp_mult * atr
        elif position == -1:
            sl = entry_price + self.sl_mult * atr
            tp = entry_price - self.tp_mult * atr
        else:
            sl = tp = entry_price
        return sl, tp
