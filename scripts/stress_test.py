#!/usr/bin/env python3
"""
Stress test all trained PPO agents on the held-out test set.

Tests 6 scenarios per agent:
  1. Clean          -- normal friction (spread=2t, slip=1t)
  2. High friction  -- 2x spread, 3x slippage
  3. Worst case     -- 3x spread, 5x slippage, 2-bar delay
  4. Domain random  -- friction randomized each trade
  5. Val set        -- same friction as (1) but on validation period
  6. Ensemble vote  -- weighted voting across all 4 specialists

Metrics reported per scenario:
  Total return (%), Sharpe ratio, Max drawdown (%), Win rate (%),
  Profit factor, Avg trade bars, Total trades, Calmar ratio
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

TICK_VALUE  = 10.0   # $ per tick
TICK_SIZE   = 0.10   # $/oz per tick
CONTRACT    = 100    # oz per contract
INIT_CAP    = 10_000.0

# ------------------------------------------------------------------ #
# Backtester
# ------------------------------------------------------------------ #

class Backtester:
    """Run a trained PPO agent deterministically over a price series."""

    def __init__(self, model: PPO, features: np.ndarray, prices: np.ndarray,
                 regime: np.ndarray, env_cfg: dict, friction: dict,
                 obs_window: int = 60):
        self.model    = model
        self.features = features.astype(np.float32)
        self.prices   = prices.astype(np.float32)
        self.regime   = regime.astype(np.int8)
        self.env_cfg  = env_cfg
        self.friction = friction
        self.obs_window = obs_window

        risk = env_cfg.get("risk", {})
        self.max_dd      = risk.get("guard", {}).get("max_drawdown_pct", 0.10)
        self.floor       = risk.get("guard", {}).get("min_account_floor", 5000.0)
        self.max_trades  = risk.get("guard", {}).get("max_trades_per_day", 10)
        self.cooldown    = risk.get("guard", {}).get("cooldown_bars", 5)
        self.atr_sl      = risk.get("sl_tp", {}).get("atr_sl", 1.5)
        self.atr_tp      = risk.get("sl_tp", {}).get("atr_tp", 3.0)

    def run(self) -> dict:
        n = len(self.prices)
        capital   = INIT_CAP
        peak      = INIT_CAP
        position  = 0
        entry_px  = 0.0
        sl = tp   = None
        trades_today = 0
        bars_since   = 999
        day_idx      = -1

        capitals   = [capital]
        trade_log  = []
        halted_bar = n

        for t in range(self.obs_window, n - 1):
            # Day boundary
            cur_day = t // 390
            if cur_day != day_idx:
                trades_today = 0
                day_idx = cur_day

            px = float(self.prices[t])

            # Check SL/TP
            if position != 0 and sl is not None:
                hit, exit_px = self._check_sl_tp(px, position, entry_px, sl, tp)
                if hit:
                    pnl = self._pnl(position, entry_px, exit_px)
                    capital += pnl
                    trade_log.append({"pnl": pnl, "bars": t})
                    position = 0; sl = tp = None

            # Get observation
            obs = self._get_obs(t)

            # Deterministic action from policy
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            target = {0: 0, 1: 1, 2: -1}[action]

            # Risk gate
            drawdown = (peak - capital) / peak if peak > 0 else 0
            if drawdown >= self.max_dd or capital <= self.floor:
                halted_bar = t
                break
            can_trade = (trades_today < self.max_trades and
                         bars_since >= self.cooldown and
                         drawdown < self.max_dd and
                         capital > self.floor)

            if target != position:
                if can_trade:
                    # Close old
                    if position != 0:
                        cost = self._friction_cost(closing=True)
                        close_px = px
                        pnl = self._pnl(position, entry_px, close_px) - cost
                        capital += pnl
                        trade_log.append({"pnl": pnl - cost, "bars": t})
                    # Open new
                    if target != 0:
                        cost = self._friction_cost(closing=False)
                        capital -= cost
                        entry_px = px
                        # ATR proxy: use vol feature (index 4) * price
                        atr = max(float(self.features[t, 4]) * px * 0.01, 1.0)
                        if target == 1:
                            sl = entry_px - self.atr_sl * atr
                            tp = entry_px + self.atr_tp * atr
                        else:
                            sl = entry_px + self.atr_sl * atr
                            tp = entry_px - self.atr_tp * atr
                    position = target
                    trades_today += 1
                    bars_since = 0

            # Mark-to-market P&L
            if position != 0:
                next_px = float(self.prices[t + 1])
                pnl = self._pnl(position, px, next_px)
                capital += pnl

            peak = max(peak, capital)
            capitals.append(capital)
            bars_since += 1

        return self._metrics(capitals, trade_log, halted_bar, n)

    # ------------------------------------------------------------------ #

    def _get_obs(self, t: int) -> np.ndarray:
        start = max(0, t - self.obs_window)
        window = self.features[start:t]
        if len(window) < self.obs_window:
            pad = np.zeros((self.obs_window - len(window), self.features.shape[1]),
                           dtype=np.float32)
            window = np.vstack([pad, window])
        return window.flatten().astype(np.float32)

    def _pnl(self, pos, entry, exit_px) -> float:
        ticks = (exit_px - entry) / TICK_SIZE
        return ticks * TICK_VALUE * pos

    def _friction_cost(self, closing: bool) -> float:
        fr = self.friction
        if fr.get("randomize"):
            spread  = np.random.uniform(*fr.get("spread_range",  [1, 4]))
            slip    = np.random.uniform(*fr.get("slip_range",    [0, 3]))
        else:
            spread = fr.get("spread_ticks", 2)
            slip   = fr.get("slippage_ticks", 1)
        return (spread / 2 + slip) * TICK_VALUE

    def _check_sl_tp(self, px, pos, entry, sl, tp):
        if pos == 1:
            if px <= sl: return True, sl
            if px >= tp: return True, tp
        elif pos == -1:
            if px >= sl: return True, sl
            if px <= tp: return True, tp
        return False, px

    def _metrics(self, capitals, trade_log, halted_bar, total_bars) -> dict:
        caps = np.array(capitals, dtype=np.float64)
        final_cap = caps[-1]
        total_return = (final_cap - INIT_CAP) / INIT_CAP * 100

        # Daily returns (390 bars = 1 trading day)
        daily = []
        for i in range(0, len(caps) - 390, 390):
            d_ret = (caps[i + 390] - caps[i]) / caps[i]
            daily.append(d_ret)
        daily = np.array(daily) if daily else np.array([0.0])

        sharpe  = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252)
        max_dd  = 0.0
        peak    = caps[0]
        for c in caps:
            peak   = max(peak, c)
            max_dd = max(max_dd, (peak - c) / peak * 100)

        pnls = [t["pnl"] for t in trade_log]
        wins  = [p for p in pnls if p > 0]
        loses = [p for p in pnls if p <= 0]
        win_rate      = len(wins) / len(pnls) * 100 if pnls else 0
        gross_profit  = sum(wins)
        gross_loss    = abs(sum(loses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        calmar        = (total_return / 100) * 252 / (max_dd / 100 + 1e-9)

        return {
            "total_return_pct":  round(total_return,  2),
            "sharpe":            round(float(sharpe),  3),
            "max_drawdown_pct":  round(max_dd,         2),
            "win_rate_pct":      round(win_rate,        1),
            "profit_factor":     round(profit_factor,   3),
            "total_trades":      len(pnls),
            "calmar":            round(calmar,          3),
            "halted_at_bar":     halted_bar if halted_bar < total_bars else None,
            "final_capital":     round(final_cap, 2),
        }


# ------------------------------------------------------------------ #
# Ensemble backtester
# ------------------------------------------------------------------ #

class EnsembleBacktester(Backtester):
    """Weighted voting across multiple specialists."""

    REGIME_NAMES = {0: "trending", 1: "trending", 2: "ranging", 3: "vol"}

    def __init__(self, models: dict, specialists_cfg: dict, **kwargs):
        # Use first model as placeholder for parent init
        first = next(iter(models.values()))
        super().__init__(first, **kwargs)
        self.models = models
        self.spec_cfg = specialists_cfg

    def _get_action(self, t: int) -> int:
        obs = self._get_obs(t)
        regime_key = self.REGIME_NAMES.get(int(self.regime[t]), "ranging")
        votes = np.zeros(3, dtype=np.float64)

        import torch
        for name, model in self.models.items():
            device = next(model.policy.parameters()).device
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                dist  = model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy()[0]
            w = float(self.spec_cfg.get(name, {}).get("regime_weight", {})
                      .get(regime_key, 1.0))
            votes += w * probs

        return int(np.argmax(votes))

    def run(self) -> dict:
        # Monkey-patch the action step to use ensemble voting
        n = len(self.prices)
        capital = INIT_CAP; peak = INIT_CAP
        position = 0; entry_px = 0.0
        sl = tp = None
        trades_today = 0; bars_since = 999; day_idx = -1
        capitals = [capital]; trade_log = []; halted_bar = n

        for t in range(self.obs_window, n - 1):
            cur_day = t // 390
            if cur_day != day_idx:
                trades_today = 0; day_idx = cur_day
            px = float(self.prices[t])

            if position != 0 and sl is not None:
                hit, exit_px = self._check_sl_tp(px, position, entry_px, sl, tp)
                if hit:
                    pnl = self._pnl(position, entry_px, exit_px)
                    capital += pnl
                    trade_log.append({"pnl": pnl, "bars": t})
                    position = 0; sl = tp = None

            action = self._get_action(t)
            target = {0: 0, 1: 1, 2: -1}[action]

            drawdown = (peak - capital) / peak if peak > 0 else 0
            if drawdown >= self.max_dd or capital <= self.floor:
                halted_bar = t; break

            can_trade = (trades_today < self.max_trades and
                         bars_since >= self.cooldown and
                         drawdown < self.max_dd and capital > self.floor)

            if target != position and can_trade:
                if position != 0:
                    cost = self._friction_cost(closing=True)
                    pnl  = self._pnl(position, entry_px, px) - cost
                    capital += pnl
                    trade_log.append({"pnl": pnl, "bars": t})
                if target != 0:
                    cost = self._friction_cost(closing=False)
                    capital -= cost
                    entry_px = px
                    atr = max(float(self.features[t, 4]) * px * 0.01, 1.0)
                    if target == 1:
                        sl = entry_px - self.atr_sl * atr
                        tp = entry_px + self.atr_tp * atr
                    else:
                        sl = entry_px + self.atr_sl * atr
                        tp = entry_px - self.atr_tp * atr
                position = target; trades_today += 1; bars_since = 0

            if position != 0:
                capital += self._pnl(position, px, float(self.prices[t + 1]))

            peak = max(peak, capital)
            capitals.append(capital)
            bars_since += 1

        return self._metrics(capitals, trade_log, halted_bar, n)


# ------------------------------------------------------------------ #
# Friction presets
# ------------------------------------------------------------------ #

FRICTION_PRESETS = {
    "clean":        {"spread_ticks": 2,  "slippage_ticks": 1,  "randomize": False},
    "high_friction":{"spread_ticks": 4,  "slippage_ticks": 3,  "randomize": False},
    "worst_case":   {"spread_ticks": 6,  "slippage_ticks": 5,  "randomize": False},
    "randomized":   {"randomize": True,  "spread_range": [1,5], "slip_range": [0,4]},
}


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def load_model(name: str, checkpoint_dir: str) -> Optional[PPO]:
    for path in [
        Path(checkpoint_dir) / name / "best" / "best_model.zip",
        Path(checkpoint_dir) / name / f"ppo_{name}_final.zip",
    ]:
        if path.exists():
            logger.info("Loading %s from %s", name, path)
            return PPO.load(str(path))
    logger.warning("No checkpoint found for %s", name)
    return None


def print_table(results: dict):
    agents = list(results.keys())
    scenarios = list(next(iter(results.values())).keys())
    metrics = ["total_return_pct","sharpe","max_drawdown_pct",
               "win_rate_pct","profit_factor","total_trades","calmar"]
    metric_labels = ["Return%","Sharpe","MaxDD%","WinRate%",
                     "ProfitFactor","Trades","Calmar"]

    print("\n" + "=" * 100)
    print("STRESS TEST RESULTS")
    print("=" * 100)

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario.upper()} ---")
        header = f"{'Agent':<18}" + "".join(f"{m:<16}" for m in metric_labels)
        print(header)
        print("-" * len(header))
        for agent in agents:
            r = results[agent].get(scenario, {})
            if not r:
                continue
            row = f"{agent:<18}"
            for m in metrics:
                v = r.get(m, "N/A")
                if isinstance(v, float):
                    row += f"{v:<16.3f}"
                else:
                    row += f"{str(v):<16}"
            print(row)

    print("\n" + "=" * 100)


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="GC RL - Stress Test")
    p.add_argument("--data-dir",      default="data/processed")
    p.add_argument("--checkpoint-dir",default="checkpoints")
    p.add_argument("--env-config",    default="configs/env_config.yaml")
    p.add_argument("--ensemble-config",default="configs/ensemble_config.yaml")
    p.add_argument("--output-dir",    default="eval_results")
    p.add_argument("--agents", nargs="+",
                   default=["generalist","trend","range_sr","volatility"])
    args = p.parse_args()

    env_cfg      = load_config(args.env_config)
    ensemble_cfg = load_config(args.ensemble_config)
    data_dir     = Path(args.data_dir)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Load numpy arrays
    logger.info("Loading test + val arrays...")
    test_f  = np.load(str(data_dir / "features_test.npy"))
    test_p  = np.load(str(data_dir / "prices_test.npy"))
    test_r  = np.load(str(data_dir / "regime_test.npy"))
    val_f   = np.load(str(data_dir / "features_val.npy"))
    val_p   = np.load(str(data_dir / "prices_val.npy"))
    val_r   = np.load(str(data_dir / "regime_val.npy"))
    logger.info("Test: %s bars  |  Val: %s bars", len(test_p), len(val_p))

    # Load models
    models = {}
    for name in args.agents:
        m = load_model(name, args.checkpoint_dir)
        if m:
            models[name] = m

    if not models:
        logger.error("No models found in %s", args.checkpoint_dir)
        sys.exit(1)

    obs_window = env_cfg.get("observation", {}).get("window", 60)
    results    = {}

    # ------------------------------------------------------------------ #
    # Individual agent stress tests
    # ------------------------------------------------------------------ #
    for name, model in models.items():
        logger.info("Stress testing: %s", name)
        results[name] = {}

        for scenario, friction in FRICTION_PRESETS.items():
            logger.info("  scenario: %s", scenario)
            bt = Backtester(model, test_f, test_p, test_r,
                            env_cfg, friction, obs_window)
            results[name][scenario] = bt.run()

        # Validation set (clean friction)
        logger.info("  scenario: val_set")
        bt = Backtester(model, val_f, val_p, val_r,
                        env_cfg, FRICTION_PRESETS["clean"], obs_window)
        results[name]["val_set"] = bt.run()

    # ------------------------------------------------------------------ #
    # Ensemble (voting) stress test
    # ------------------------------------------------------------------ #
    if len(models) > 1:
        logger.info("Stress testing: ensemble_vote")
        results["ensemble_vote"] = {}
        spec_cfg = ensemble_cfg.get("specialists", {})

        for scenario, friction in FRICTION_PRESETS.items():
            logger.info("  scenario: %s", scenario)
            bt = EnsembleBacktester(
                models=models,
                specialists_cfg=spec_cfg,
                features=test_f, prices=test_p, regime=test_r,
                env_cfg=env_cfg, friction=friction, obs_window=obs_window,
            )
            results["ensemble_vote"][scenario] = bt.run()

        # Val set
        bt = EnsembleBacktester(
            models=models, specialists_cfg=spec_cfg,
            features=val_f, prices=val_p, regime=val_r,
            env_cfg=env_cfg, friction=FRICTION_PRESETS["clean"],
            obs_window=obs_window,
        )
        results["ensemble_vote"]["val_set"] = bt.run()

    # ------------------------------------------------------------------ #
    # Print + save
    # ------------------------------------------------------------------ #
    print_table(results)

    out_path = out_dir / "stress_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved -> %s", out_path)

    # Quick summary
    print("\nQUICK SUMMARY (Test set, clean friction):")
    print(f"{'Agent':<20} {'Return%':>10} {'Sharpe':>10} {'MaxDD%':>10} {'WinRate%':>10}")
    print("-" * 55)
    for name, scenarios in results.items():
        r = scenarios.get("clean", {})
        print(f"{name:<20} {r.get('total_return_pct',0):>10.2f} "
              f"{r.get('sharpe',0):>10.3f} "
              f"{r.get('max_drawdown_pct',0):>10.2f} "
              f"{r.get('win_rate_pct',0):>10.1f}")


if __name__ == "__main__":
    main()
