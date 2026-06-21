"""Ensemble voting: combine specialist agent actions.

Two modes:
  1. Baseline voting  — fixed regime-based weights from ensemble_config.yaml
  2. Meta-policy      — learned weights (MLP trained on specialist performance)

Voting weights at each bar are determined by:
  w_i = specialist_i.regime_weight[current_regime]
  final_action = argmax(sum_i(w_i * action_probs_i))

Performance tracking: if a specialist's rolling win-rate drops below
win_rate_threshold, its weight is down-weighted by downweight_factor.
"""
import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

REGIME_NAMES = {0: "trending", 1: "trending", 2: "ranging", 3: "vol"}


class SpecialistEnsemble:
    """Weighted voting ensemble of PPO specialist agents."""

    def __init__(self, config: dict):
        cfg = config
        self.specialists_cfg = cfg.get("specialists", {})
        self.voting = cfg.get("voting", {})
        self.perf_cfg = cfg.get("performance_tracking", {})

        self.win_rate_window = int(self.perf_cfg.get("window", 100))
        self.win_rate_threshold = float(self.perf_cfg.get("win_rate_threshold", 0.4))
        self.downweight_factor = float(self.perf_cfg.get("downweight_factor", 0.5))

        self._models: Dict[str, object] = {}
        self._perf_windows: Dict[str, deque] = {}

    def add_specialist(self, name: str, model):
        self._models[name] = model
        self._perf_windows[name] = deque(maxlen=self.win_rate_window)

    def predict(self, obs: np.ndarray, current_regime: int) -> int:
        """Return combined action using regime-weighted voting."""
        if not self._models:
            return 0  # flat

        regime_key = REGIME_NAMES.get(current_regime, "ranging")
        action_votes = np.zeros(3, dtype=float)

        for name, model in self._models.items():
            # Get action probabilities from PPO policy
            action_probs = self._get_action_probs(model, obs)
            # Regime weight for this specialist
            weight = self._get_regime_weight(name, regime_key)
            # Performance down-weight
            weight *= self._perf_multiplier(name)
            action_votes += weight * action_probs

        return int(np.argmax(action_votes))

    def record_trade_result(self, specialist_name: str, profitable: bool):
        if specialist_name in self._perf_windows:
            self._perf_windows[specialist_name].append(1 if profitable else 0)

    def get_weights(self, current_regime: int) -> Dict[str, float]:
        regime_key = REGIME_NAMES.get(current_regime, "ranging")
        weights = {}
        for name in self._models:
            w = self._get_regime_weight(name, regime_key) * self._perf_multiplier(name)
            weights[name] = w
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------ #

    def _get_action_probs(self, model, obs: np.ndarray) -> np.ndarray:
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy()[0]
        return probs

    def _get_regime_weight(self, name: str, regime_key: str) -> float:
        spec = self.specialists_cfg.get(name, {})
        rw = spec.get("regime_weight", {})
        return float(rw.get(regime_key, 1.0))

    def _perf_multiplier(self, name: str) -> float:
        buf = self._perf_windows.get(name)
        if buf and len(buf) >= 10:
            win_rate = np.mean(list(buf))
            if win_rate < self.win_rate_threshold:
                return self.downweight_factor
        return 1.0


class MetaPolicyEnsemble(SpecialistEnsemble):
    """Learned meta-policy that replaces fixed regime weights with a neural net.

    The meta-policy takes [obs, regime_one_hot] as input and outputs
    per-specialist weights (softmax). Trained via supervised regression on
    historical specialist returns.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        meta_cfg = config.get("meta_policy", {})
        self.hidden_dim = int(meta_cfg.get("hidden_dim", 64))
        self._meta_model: Optional[object] = None

    def build_meta_policy(self, obs_dim: int, n_specialists: int):
        """Build a simple MLP meta-policy."""
        import torch.nn as nn
        input_dim = obs_dim + 4  # obs + 4 regime one-hot
        self._meta_model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_specialists),
            nn.Softmax(dim=-1),
        )
        return self._meta_model

    def predict(self, obs: np.ndarray, current_regime: int) -> int:
        if self._meta_model is None:
            return super().predict(obs, current_regime)

        import torch
        regime_oh = np.zeros(4, dtype=np.float32)
        regime_oh[min(current_regime, 3)] = 1.0
        meta_input = np.concatenate([obs, regime_oh])

        with torch.no_grad():
            weights = self._meta_model(
                torch.FloatTensor(meta_input).unsqueeze(0)
            ).numpy()[0]

        names = list(self._models.keys())
        action_votes = np.zeros(3, dtype=float)
        for i, name in enumerate(names):
            probs = self._get_action_probs(self._models[name], obs)
            action_votes += weights[i] * probs

        return int(np.argmax(action_votes))
