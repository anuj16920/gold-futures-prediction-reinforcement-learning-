"""PPO specialist agent wrapper using Stable-Baselines3.

Each specialist is a standard PPO agent trained on TradingEnv with
regime-shaped rewards. Three specialists are trained:
  - trend:    target_regime = 0 (trending_up) or 1 (trending_down)
  - range_sr: target_regime = 2 (ranging)
  - vol:      target_regime = 3 (high_volatility)

A single generalist (no target_regime) is also trained as baseline.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.trading_env import TradingEnv

logger = logging.getLogger(__name__)

_SPECIALIST_CONFIGS = {
    "generalist": {"target_regime": None},
    "trend":      {"target_regime": 0},   # trending_up
    "range_sr":   {"target_regime": 2},   # ranging
    "volatility": {"target_regime": 3},   # high_volatility
}


def make_env(features, prices, env_config, regime_labels=None, target_regime=None):
    def _init():
        env = TradingEnv(
            features=features,
            prices=prices,
            config=env_config,
            regime_labels=regime_labels,
            target_regime=target_regime,
        )
        return Monitor(env)
    return _init


def train_specialist(
    name: str,
    train_features: np.ndarray,
    train_prices: np.ndarray,
    val_features: np.ndarray,
    val_prices: np.ndarray,
    env_config: dict,
    ensemble_config: dict,
    regime_labels_train: Optional[np.ndarray] = None,
    regime_labels_val: Optional[np.ndarray] = None,
    checkpoint_dir: str = "checkpoints",
    total_timesteps: int = 500_000,
) -> PPO:
    """Train one PPO specialist and save checkpoint."""
    spec_cfg = _SPECIALIST_CONFIGS.get(name, {"target_regime": None})
    target_regime = spec_cfg["target_regime"]

    logger.info("Training specialist: %s  (target_regime=%s)", name, target_regime)

    train_env = DummyVecEnv([make_env(
        train_features, train_prices, env_config,
        regime_labels_train, target_regime,
    )])
    eval_env = DummyVecEnv([make_env(
        val_features, val_prices, env_config,
        regime_labels_val, target_regime,
    )])

    ckpt_dir = Path(checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(ckpt_dir),
            name_prefix=f"ppo_{name}",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval_logs"),
            eval_freq=25_000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        ),
    ]

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"training_logs/{name}",
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = ckpt_dir / f"ppo_{name}_final"
    model.save(str(final_path))
    logger.info("Saved %s -> %s", name, final_path)
    return model


def load_specialist(name: str, checkpoint_dir: str = "checkpoints") -> Optional[PPO]:
    """Load a trained specialist from its best checkpoint."""
    best_path = Path(checkpoint_dir) / name / "best" / "best_model.zip"
    final_path = Path(checkpoint_dir) / name / f"ppo_{name}_final.zip"

    for path in [best_path, final_path]:
        if path.exists():
            logger.info("Loading %s from %s", name, path)
            return PPO.load(str(path))

    logger.warning("No checkpoint found for %s", name)
    return None
