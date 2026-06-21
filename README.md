# Gold Futures RL Trading System.

Reinforcement learning system for CME Gold (GC) futures using ensemble specialist PPO agents and regime-adaptive trading.

## Architecture

```
Raw Ticks (Databento GLBX.MDP3)
  -> VolumeBasedRoller  (expiry-sorted contract rolling)
  -> PanamaBackAdjuster (price continuity across rolls)
  -> BarAggregator      (1-min OHLCV, maintenance filtered)
  -> FeaturePipeline    (19 features, fit on train only)
  -> RegimeDetector     (ADX + ATR percentile, 4 regimes)
  -> 3 Specialist PPO Agents + 1 Generalist
  -> MetaPolicyEnsemble (learned voting weights)
  -> RiskGuard          (drawdown / daily trade limits)
```

## Agents

| Agent | Target Regime | Description |
|---|---|---|
| Generalist | All | Baseline single-policy PPO |
| Trend | Trending (ADX > 25) | Momentum following |
| Range/SR | Ranging | Support/resistance mean reversion |
| Volatility | High volatility | Event-driven, tight stops |

## Dataset

- Source: Databento GLBX.MDP3 (CME Globex Gold Futures)
- Period: 2023-01-18 to 2026-06-19 (3.5 years)
- Ticks: 93.6M trades, compressed .zst per day
- Bars: 1,176,650 one-minute OHLCV bars after processing

## Data Splits (strictly time-ordered, no shuffling)

| Split | Period | Bars |
|---|---|---|
| Train | 2023-01-18 to 2024-12-31 | 671,439 |
| Val | 2025-01-01 to 2025-09-30 | 255,202 |
| Test | 2025-10-01 to 2026-06-19 | 246,356 |

## Anti-Leakage Guarantees

- `FeaturePipeline.fit()` called on **training bars only**
- MTF features use `.shift(1)` on resampled series (no open-bar lookahead)
- SR pivots confirmed at `bar[i + pivot_window]` (not at bar `i`)
- Regime labels computed independently per split
- Normalization stats saved to `norm_stats.json` for live use

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Extract + process all data (run once, ~25 min)
python scripts/prepare_data.py

# Step 2: Train all 4 PPO agents (~2.5 hrs on GPU)
python scripts/train.py

# Step 3: Validate live/batch pipeline match before going live
python scripts/replay_consistency_check.py --test-day 2025-10-01
```

## Project Structure

```
src/
  data/         loaders, bar_aggregation, roll_logic
  features/     base_features, mtf_features, sr_features, pipeline
  live/         incremental_bars, incremental_features
  risk/         stops (ATR SL/TP), guard (drawdown limits)
  regime/       detector (ADX + ATR percentile, 4 regimes)
  envs/         trading_env (Gymnasium, Discrete(3) actions)
  agents/       ppo_agent (Stable-Baselines3 PPO)
  ensemble/     voting (SpecialistEnsemble + MetaPolicyEnsemble)
  utils/        config, gpu, logging

scripts/
  prepare_data.py           Full data pipeline -> numpy arrays
  train.py                  Train all specialists
  replay_consistency_check.py  Validate live == batch
  reconcile_configs.py      Config validation

configs/
  data_config.yaml          Roll logic, timeframe, session hours
  env_config.yaml           Obs window, contract specs, splits
  feature_config.yaml       Lookbacks, MTF timeframes, SR params
  regime_config.yaml        ADX threshold, ATR percentile
  ensemble_config.yaml      Specialist regime weights, meta-policy
  live_config.yaml          Databento live feed config

dashboard/
  index.html                Real-time monitoring UI
```

## Risk Controls

- Stop-loss: 1.5x ATR from entry
- Take-profit: 3.0x ATR from entry
- Max drawdown: 10% from peak equity
- Account floor: $5,000 minimum
- Max 10 trades per day
- 5-bar cooldown between entries
