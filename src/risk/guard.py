"""Risk guard: halt trading when limits are breached.

All checks are retrospective — they use current and historical account
data only, never future prices.
"""
import logging

logger = logging.getLogger(__name__)


class RiskGuard:
    """Enforce drawdown, floor, daily trade, and cooldown limits."""

    def __init__(
        self,
        max_drawdown_pct: float = 0.10,
        min_account_floor: float = 5000.0,
        max_trades_per_day: int = 10,
        cooldown_bars: int = 5,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.min_account_floor = min_account_floor
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_bars = cooldown_bars

        self._trades_today: int = 0
        self._bars_since_last_trade: int = 999
        self._current_date = None

    def is_halted(self, current_capital: float, peak_capital: float) -> bool:
        """Return True if any hard risk limit is breached."""
        drawdown = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0.0
        if drawdown >= self.max_drawdown_pct:
            return True
        if current_capital <= self.min_account_floor:
            return True
        return False

    def can_trade(self, current_capital: float, peak_capital: float,
                  bar_date=None) -> bool:
        """Return True if a new trade is allowed (all soft + hard limits pass)."""
        if self.is_halted(current_capital, peak_capital):
            return False
        if bar_date is not None and bar_date != self._current_date:
            self._trades_today = 0
            self._current_date = bar_date
        if self._trades_today >= self.max_trades_per_day:
            return False
        if self._bars_since_last_trade < self.cooldown_bars:
            return False
        return True

    def record_trade(self):
        self._trades_today += 1
        self._bars_since_last_trade = 0

    def tick(self):
        """Call once per bar (even if no trade executed)."""
        self._bars_since_last_trade += 1

    def reset_day(self):
        self._trades_today = 0
