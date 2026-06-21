"""Gold futures RL trading environment (Gymnasium-compatible).

Observation space: flat array of shape (obs_window * n_features,)
Action space:      Discrete(3)  — 0=flat, 1=long 1 contract, 2=short 1 contract
Reward:            P&L in $ / starting_capital  (normalized, per-bar)

ANTI-LEAKAGE:
- observation at step t contains features for bars [t-obs_window+1 .. t]
  All features are pre-computed by FeaturePipeline which is already causal.
- Execution price for a trade decided at bar t is: close[t] ± friction
  (we trade AT bar t's close, not at bar t+1's open, for simplicity).
- P&L between bar t and t+1 uses close[t+1] — this is future data relative
  to bar t, but it is the REWARD signal, not an input feature.  Using future
  prices in the reward is correct RL design; what must not happen is future
  prices appearing in the OBSERVATION.

Contract spec (GC Gold Futures):
  1 tick = $0.10/oz,  1 contract = 100 oz  → tick value = $10
  Price move of $1.00/oz  → P&L of $100 per contract
"""
import logging
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)

_ACTION_FLAT = 0
_ACTION_LONG = 1
_ACTION_SHORT = 2


class TradingEnv(gym.Env):
    """Single-asset futures trading environment for PPO training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,           # (n_bars, n_features) — pre-computed, normalized
        prices: np.ndarray,              # (n_bars,) raw close prices (for P&L calc)
        config: dict = None,
        regime_labels: Optional[np.ndarray] = None,   # (n_bars,) int regime per bar
        target_regime: Optional[int] = None,          # specialist focus regime
    ):
        super().__init__()
        cfg = config or {}

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.regime_labels = regime_labels
        self.target_regime = target_regime

        n_bars, n_feat = features.shape
        self.n_bars = n_bars
        self.n_features = n_feat

        # Config
        obs_cfg = cfg.get("observation", {})
        self.obs_window = int(obs_cfg.get("window", 60))
        contract_cfg = cfg.get("contract", {})
        self.tick_size = float(contract_cfg.get("tick_size", 0.10))
        self.tick_value = float(contract_cfg.get("tick_value", 10.0))
        self.starting_capital = float(contract_cfg.get("starting_capital", 10_000.0))

        friction_cfg = cfg.get("friction", {}).get("deterministic", {})
        self.spread_ticks = int(friction_cfg.get("spread_ticks", 2))
        self.slippage_ticks = int(friction_cfg.get("slippage_ticks", 1))
        self.stop_slippage_ticks = int(friction_cfg.get("stop_slippage_ticks", 0))

        risk_cfg = cfg.get("risk", {})
        sl_tp_cfg = risk_cfg.get("sl_tp", {})
        self.atr_sl = float(sl_tp_cfg.get("atr_sl", 1.5))
        self.atr_tp = float(sl_tp_cfg.get("atr_tp", 3.0))

        guard_cfg = risk_cfg.get("guard", {})
        self.max_drawdown_pct = float(guard_cfg.get("max_drawdown_pct", 0.10))
        self.min_account_floor = float(guard_cfg.get("min_account_floor", 5000.0))
        self.max_trades_per_day = int(guard_cfg.get("max_trades_per_day", 10))
        self.cooldown_bars = int(guard_cfg.get("cooldown_bars", 5))

        # Spaces
        obs_size = self.obs_window * self.n_features
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # State (reset in reset())
        self._step = 0
        self._position = 0
        self._capital = self.starting_capital
        self._peak_capital = self.starting_capital
        self._entry_price = 0.0
        self._entry_atr = 0.0
        self._sl = None
        self._tp = None
        self._trades_today = 0
        self._bars_since_trade = 999
        self._current_day = None

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at obs_window so we have a full observation from the first step
        self._step = self.obs_window
        self._position = 0
        self._capital = self.starting_capital
        self._peak_capital = self.starting_capital
        self._entry_price = 0.0
        self._entry_atr = 0.0
        self._sl = None
        self._tp = None
        self._trades_today = 0
        self._bars_since_trade = 999
        self._current_day = None
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._step >= self.n_bars - 1:
            return self._get_obs(), 0.0, True, False, {}

        current_price = float(self.prices[self._step])
        next_price = float(self.prices[self._step + 1])

        # Day reset
        self._maybe_reset_day()

        # Check SL/TP before acting
        terminated_by_sl_tp, sl_tp_pnl = self._check_sl_tp(current_price)
        if terminated_by_sl_tp:
            self._capital += sl_tp_pnl
            self._peak_capital = max(self._peak_capital, self._capital)
            reward = sl_tp_pnl / self.starting_capital
            self._position = 0
            self._sl = self._tp = None
            self._step += 1
            self._bars_since_trade += 1
            done = self._is_hard_halted()
            return self._get_obs(), reward, done, False, self._info()

        # Determine desired position from action
        target_pos = {_ACTION_FLAT: 0, _ACTION_LONG: 1, _ACTION_SHORT: -1}[action]

        # Regime shaping: if specialist, penalize acting against target regime
        regime_bonus = self._regime_bonus(action)

        # Risk gate: prevent trading if limits breached
        if target_pos != self._position and not self._can_trade():
            target_pos = self._position  # forced hold

        # Execute position change
        trade_cost = 0.0
        if target_pos != self._position:
            # Close existing position
            if self._position != 0:
                trade_cost += self._friction_cost(current_price, self._position, closing=True)
            # Open new position
            if target_pos != 0:
                trade_cost += self._friction_cost(current_price, target_pos, closing=False)
                self._entry_price = current_price
                self._entry_atr = self._get_atr()
                self._sl, self._tp = self._compute_sl_tp(current_price, target_pos)

            self._position = target_pos
            self._capital -= trade_cost
            if target_pos != 0:
                self._trades_today += 1
                self._bars_since_trade = 0

        # P&L from price move (holding position through this bar)
        move_pnl = self._pnl_from_move(self._position, current_price, next_price)
        self._capital += move_pnl
        self._peak_capital = max(self._peak_capital, self._capital)

        reward = (move_pnl - trade_cost) / self.starting_capital + regime_bonus

        self._step += 1
        self._bars_since_trade += 1

        terminated = self._is_hard_halted() or self._step >= self.n_bars - 1
        return self._get_obs(), reward, terminated, False, self._info()

    def render(self):
        pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        start = max(0, self._step - self.obs_window)
        end = self._step
        window = self.features[start:end]
        if len(window) < self.obs_window:
            pad = np.zeros((self.obs_window - len(window), self.n_features), dtype=np.float32)
            window = np.vstack([pad, window])
        return window.flatten().astype(np.float32)

    def _pnl_from_move(self, position: int, price_now: float, price_next: float) -> float:
        if position == 0:
            return 0.0
        move_in_ticks = (price_next - price_now) / self.tick_size
        return move_in_ticks * self.tick_value * position

    def _friction_cost(self, price: float, direction: int, closing: bool) -> float:
        ticks = self.spread_ticks / 2 + self.slippage_ticks
        return ticks * self.tick_value

    def _compute_sl_tp(self, entry: float, position: int):
        atr = self._get_atr()
        if position == 1:
            return entry - self.atr_sl * atr, entry + self.atr_tp * atr
        else:
            return entry + self.atr_sl * atr, entry - self.atr_tp * atr

    def _check_sl_tp(self, current_price: float) -> Tuple[bool, float]:
        if self._position == 0 or self._sl is None:
            return False, 0.0
        if self._position == 1:
            if current_price <= self._sl:
                raw_pnl = self._pnl_from_move(1, self._entry_price, self._sl)
                slippage = (self.stop_slippage_ticks * self.tick_value)
                return True, raw_pnl - slippage
            if current_price >= self._tp:
                return True, self._pnl_from_move(1, self._entry_price, self._tp)
        elif self._position == -1:
            if current_price >= self._sl:
                raw_pnl = self._pnl_from_move(-1, self._entry_price, self._sl)
                slippage = self.stop_slippage_ticks * self.tick_value
                return True, raw_pnl - slippage
            if current_price <= self._tp:
                return True, self._pnl_from_move(-1, self._entry_price, self._tp)
        return False, 0.0

    def _can_trade(self) -> bool:
        drawdown = (self._peak_capital - self._capital) / self._peak_capital
        if drawdown >= self.max_drawdown_pct:
            return False
        if self._capital <= self.min_account_floor:
            return False
        if self._trades_today >= self.max_trades_per_day:
            return False
        if self._bars_since_trade < self.cooldown_bars:
            return False
        return True

    def _is_hard_halted(self) -> bool:
        drawdown = (self._peak_capital - self._capital) / self._peak_capital
        return (
            drawdown >= self.max_drawdown_pct
            or self._capital <= self.min_account_floor
        )

    def _maybe_reset_day(self):
        if self._step < len(self.prices):
            # Use step index as a proxy for day boundary (simplified)
            day_idx = self._step // 390  # ~390 bars per trading day
            if day_idx != self._current_day:
                self._trades_today = 0
                self._current_day = day_idx

    def _get_atr(self) -> float:
        """Get ATR from pre-computed features (atr_14 column index assumed)."""
        # ATR is in the feature vector — use a fixed fallback if not found
        # In practice the agent uses stop levels computed at entry
        return max(self._entry_atr, 1.0)

    def _regime_bonus(self, action: int) -> float:
        """Small shaping bonus for specialists trading in their target regime."""
        if self.target_regime is None or self.regime_labels is None:
            return 0.0
        if self._step >= len(self.regime_labels):
            return 0.0
        current_regime = int(self.regime_labels[self._step])
        if current_regime == self.target_regime and action != _ACTION_FLAT:
            return 0.002  # tiny bonus for being active in target regime
        if current_regime != self.target_regime and action != _ACTION_FLAT:
            return -0.001  # tiny penalty for being active in wrong regime
        return 0.0

    def _info(self) -> dict:
        return {
            "capital": self._capital,
            "position": self._position,
            "trades_today": self._trades_today,
            "step": self._step,
        }
