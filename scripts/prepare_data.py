#!/usr/bin/env python3
"""
Full data preparation pipeline for GC Gold Futures RL system.

Produces self-contained numpy arrays for each split so that training
never touches the raw zip again — just np.load() and go.

Output layout
-------------
data/raw/
    all_ticks.parquet       All clean outright-futures ticks (93M rows)
data/processed/
    continuous.parquet      Rolled + Panama-adjusted continuous series
    bars_1min.parquet       1-min OHLCV bars (1.17M rows)
    metadata.json           Data stats, date ranges, split sizes
    norm_stats.json         Feature mean/std (fit on TRAIN only -- no leakage)
    features_train.npy      (N_train, n_feat) float32
    features_val.npy        (N_val,   n_feat) float32
    features_test.npy       (N_test,  n_feat) float32
    prices_train.npy        (N_train,) float32  close prices
    prices_val.npy
    prices_test.npy
    regime_train.npy        (N_train,) int8  regime labels 0-3
    regime_val.npy
    regime_test.npy

Data handling techniques used
------------------------------
- Incremental pyarrow parquet writer  : never loads all 93M ticks at once
- Columnar parquet reads              : only pull needed columns for rolling
- float32 / int32 dtypes              : halve memory vs float64
- Snappy compression                  : fast random-access parquet reads
- Sorted + indexed parquet            : fast range slices by timestamp
- numpy memmap-ready float32 arrays   : training loads in <1 second
"""
import argparse
import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bar_aggregation import BarAggregator
from src.data.roll_logic import VolumeBasedRoller, PanamaBackAdjuster, _contract_expiry_key
from src.features.pipeline import FeaturePipeline
from src.regime.detector import RegimeDetector
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# PyArrow schema — tight dtypes to minimise memory and parquet footprint  #
# ---------------------------------------------------------------------- #
RAW_SCHEMA = pa.schema([
    ("ts_event", pa.timestamp("ns", tz="UTC")),
    ("price",    pa.float32()),
    ("size",     pa.int32()),
    ("symbol",   pa.dictionary(pa.int16(), pa.string())),  # low-cardinality
    ("side",     pa.dictionary(pa.int8(),  pa.string())),
])

CONT_SCHEMA = pa.schema([
    ("ts_event",    pa.timestamp("ns", tz="UTC")),
    ("price",       pa.float32()),
    ("price_adj",   pa.float32()),
    ("size",        pa.int32()),
    ("symbol",      pa.dictionary(pa.int16(), pa.string())),
    ("front_month", pa.dictionary(pa.int16(), pa.string())),
])

BAR_SCHEMA = pa.schema([
    ("timestamp",   pa.timestamp("ms", tz="UTC")),
    ("open",        pa.float32()),
    ("high",        pa.float32()),
    ("low",         pa.float32()),
    ("close",       pa.float32()),
    ("volume",      pa.int32()),
    ("trade_count", pa.int32()),
])


# ---------------------------------------------------------------------- #
# Step 1 — Extract & clean raw ticks (incremental writer)                 #
# ---------------------------------------------------------------------- #

def extract_raw_ticks(zip_path: Path, out_path: Path) -> int:
    """Read every .trades.csv.zst from zip, filter spreads, write parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), RAW_SCHEMA, compression="snappy")
    total = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".trades.csv.zst"))
        logger.info("Extracting %d daily files from %s", len(names), zip_path.name)

        for i, name in enumerate(names, 1):
            raw = zf.read(name)
            df = _parse_zst(raw, name)
            if df is None or df.empty:
                continue

            table = _df_to_raw_table(df)
            writer.write_table(table)
            total += len(df)

            if i % 100 == 0:
                logger.info("  %d/%d files  |  %d ticks so far", i, len(names), total)

    writer.close()
    logger.info("Saved %d ticks -> %s", total, out_path)
    return total


def _parse_zst(raw_bytes: bytes, name: str = "") -> pd.DataFrame:
    try:
        dctx = zstd.ZstdDecompressor()
        df = pd.read_csv(
            io.TextIOWrapper(dctx.stream_reader(io.BytesIO(raw_bytes)), encoding="utf-8"),
            usecols=lambda c: c in {"ts_event", "price", "size", "symbol", "side"},
            low_memory=False,
        )
    except Exception as exc:
        logger.warning("Skip %s: %s", name, exc)
        return None

    if "ts_event" not in df.columns:
        return None

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df["price"]    = pd.to_numeric(df.get("price"), errors="coerce").astype("float32")
    df["size"]     = pd.to_numeric(df.get("size"),  errors="coerce").fillna(0).astype("int32")
    df["symbol"]   = df.get("symbol", "UNKNOWN").astype(str)
    df["side"]     = df.get("side",   "N").astype(str)

    df = df.dropna(subset=["ts_event", "price"])
    # Drop calendar spreads (symbol contains hyphen, e.g. GCG3-GCJ3)
    df = df[~df["symbol"].str.contains("-", na=False)]
    return df


def _df_to_raw_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(
        df[["ts_event", "price", "size", "symbol", "side"]],
        schema=RAW_SCHEMA,
        preserve_index=False,
    )


# ---------------------------------------------------------------------- #
# Step 2 — Roll + back-adjust                                             #
# ---------------------------------------------------------------------- #

def build_continuous(raw_path: Path, out_path: Path, cfg: dict) -> pd.DataFrame:
    """Load raw ticks (only needed columns), apply roll + Panama, save."""
    logger.info("Loading raw ticks for rolling (columns: ts_event, price, size, symbol)...")
    ticks = pq.read_table(
        str(raw_path),
        columns=["ts_event", "price", "size", "symbol"],
    ).to_pandas()

    # Restore proper types after pyarrow read
    ticks["ts_event"] = pd.to_datetime(ticks["ts_event"], utc=True)
    ticks["price"]    = ticks["price"].astype("float32")
    ticks["size"]     = ticks["size"].astype("int32")
    ticks["symbol"]   = ticks["symbol"].astype(str)

    roller = VolumeBasedRoller(config=cfg)
    continuous = roller.build_continuous(ticks)

    adjuster = PanamaBackAdjuster()
    continuous = adjuster.apply(continuous)

    continuous["price"]       = continuous["price"].astype("float32")
    continuous["price_adj"]   = continuous["price_adj"].astype("float32")
    continuous["size"]        = continuous["size"].astype("int32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        continuous[["ts_event", "price", "price_adj", "size", "symbol", "front_month"]],
        schema=CONT_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, str(out_path), compression="snappy")
    logger.info("Saved continuous series (%d ticks) -> %s", len(continuous), out_path)
    return continuous


# ---------------------------------------------------------------------- #
# Step 3 — Bar aggregation                                                #
# ---------------------------------------------------------------------- #

def build_bars(continuous: pd.DataFrame, out_path: Path, cfg: dict) -> pd.DataFrame:
    bar_cfg = {
        "timeframe": cfg.get("timeframe", "1min"),
        "exclude_maintenance": cfg.get("exclude_maintenance", True),
    }
    # Use price_adj for training so Panama-smooth prices feed the model
    cont_for_bars = continuous.rename(columns={"price_adj": "price_orig", "price": "price_raw"})
    cont_for_bars["price"] = cont_for_bars["price_orig"]  # bars use adj price

    aggregator = BarAggregator(config=bar_cfg)
    bars = aggregator.aggregate(cont_for_bars)

    bars["open"]        = bars["open"].astype("float32")
    bars["high"]        = bars["high"].astype("float32")
    bars["low"]         = bars["low"].astype("float32")
    bars["close"]       = bars["close"].astype("float32")
    bars["volume"]      = bars["volume"].astype("int32")
    bars["trade_count"] = bars["trade_count"].astype("int32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(bars, schema=BAR_SCHEMA, preserve_index=False)
    pq.write_table(table, str(out_path), compression="snappy")
    logger.info("Saved %d 1-min bars -> %s", len(bars), out_path)
    return bars


# ---------------------------------------------------------------------- #
# Step 4 — Time-ordered split (no shuffling)                              #
# ---------------------------------------------------------------------- #

def split_bars(bars: pd.DataFrame, splits_cfg: dict):
    ts = pd.to_datetime(bars["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")

    def _parse(s):
        return pd.Timestamp(s, tz="UTC") if s else None

    def _slice(start, end):
        mask = pd.Series(True, index=bars.index)
        if start: mask &= ts >= _parse(start)
        if end:   mask &= ts <= _parse(end)
        return bars[mask].copy().reset_index(drop=True)

    train = _slice(splits_cfg.get("train_start"), splits_cfg.get("train_end"))
    val   = _slice(splits_cfg.get("val_start"),   splits_cfg.get("val_end"))
    test  = _slice(splits_cfg.get("test_start"),  splits_cfg.get("test_end"))

    logger.info("Split: train=%d  val=%d  test=%d bars", len(train), len(val), len(test))
    if len(train) == 0:
        raise RuntimeError("Training split is empty. Check splits in env_config.yaml.")
    return train, val, test


# ---------------------------------------------------------------------- #
# Step 5 — Features: fit on TRAIN only, transform all splits             #
# ---------------------------------------------------------------------- #

def build_features(train, val, test, feat_cfg: dict, out_dir: Path):
    pipeline = FeaturePipeline(feat_cfg)
    logger.info("Fitting feature pipeline on %d training bars...", len(train))
    pipeline.fit(train)
    pipeline.save_norm_stats(str(out_dir / "norm_stats.json"))

    splits = {"train": train, "val": val, "test": test}
    feature_arrays = {}
    price_arrays   = {}

    for name, bars in splits.items():
        if len(bars) == 0:
            feature_arrays[name] = np.empty((0, pipeline.n_features), dtype=np.float32)
            price_arrays[name]   = np.empty((0,), dtype=np.float32)
            continue

        feats = pipeline.transform(bars).values.astype(np.float32)
        prices = bars["close"].values.astype(np.float32)

        np.save(str(out_dir / f"features_{name}.npy"), feats)
        np.save(str(out_dir / f"prices_{name}.npy"), prices)

        feature_arrays[name] = feats
        price_arrays[name]   = prices
        logger.info("  %s features: %s", name, feats.shape)

    return feature_arrays, price_arrays, pipeline


# ---------------------------------------------------------------------- #
# Step 6 — Regime labels (computed per split, causal indicators only)    #
# ---------------------------------------------------------------------- #

def build_regimes(train, val, test, regime_cfg: dict, out_dir: Path):
    detector = RegimeDetector(regime_cfg)
    splits   = {"train": train, "val": val, "test": test}
    regime_arrays = {}

    for name, bars in splits.items():
        if len(bars) == 0:
            regime_arrays[name] = np.empty((0,), dtype=np.int8)
            np.save(str(out_dir / f"regime_{name}.npy"), regime_arrays[name])
            continue
        labels = detector.detect(bars).values.astype(np.int8)
        np.save(str(out_dir / f"regime_{name}.npy"), labels)
        regime_arrays[name] = labels
        counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        logger.info("  %s regimes: %s", name, counts)

    return regime_arrays


# ---------------------------------------------------------------------- #
# Step 7 — Metadata                                                       #
# ---------------------------------------------------------------------- #

def save_metadata(bars, train, val, test, pipeline, out_dir: Path):
    def _range(df):
        if len(df) == 0:
            return {"start": None, "end": None, "bars": 0}
        ts = pd.to_datetime(df["timestamp"])
        return {
            "start": str(ts.min()),
            "end":   str(ts.max()),
            "bars":  len(df),
        }

    meta = {
        "total_bars":   len(bars),
        "n_features":   pipeline.n_features,
        "feature_names": pipeline.feature_names,
        "splits": {
            "train": _range(train),
            "val":   _range(val),
            "test":  _range(test),
        },
    }
    path = out_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata -> %s", path)
    return meta


# ---------------------------------------------------------------------- #
# CLI                                                                     #
# ---------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="GC RL - Full Data Preparation Pipeline")
    p.add_argument("--zip",            default="GLBX-20260620-TN6LS3GNWC.zip",
                   help="Path to Databento zip archive")
    p.add_argument("--raw-dir",        default="data/raw")
    p.add_argument("--output-dir",     default="data/processed")
    p.add_argument("--data-config",    default="configs/data_config.yaml")
    p.add_argument("--env-config",     default="configs/env_config.yaml")
    p.add_argument("--feature-config", default="configs/feature_config.yaml")
    p.add_argument("--regime-config",  default="configs/regime_config.yaml")
    p.add_argument("--skip-extract",   action="store_true",
                   help="Skip zip extraction if data/raw/all_ticks.parquet exists")
    p.add_argument("--skip-bars",      action="store_true",
                   help="Skip rolling+bars if data/processed/bars_1min.parquet exists")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    data_cfg    = load_config(args.data_config)
    env_cfg     = load_config(args.env_config)
    feat_cfg    = load_config(args.feature_config)
    regime_cfg  = load_config(args.regime_config)

    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.output_dir)
    zip_path = Path(args.zip)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_ticks_path  = raw_dir / "all_ticks.parquet"
    continuous_path = out_dir / "continuous.parquet"
    bars_path       = out_dir / "bars_1min.parquet"

    # ------------------------------------------------------------------ #
    # 1. Extract raw ticks -> all_ticks.parquet                           #
    # ------------------------------------------------------------------ #
    if args.skip_extract and raw_ticks_path.exists():
        logger.info("Skipping extraction -- %s exists", raw_ticks_path)
    else:
        if not zip_path.exists():
            logger.error("Zip not found: %s", zip_path)
            sys.exit(1)
        logger.info("=== Step 1: Extract raw ticks ===")
        extract_raw_ticks(zip_path, raw_ticks_path)

    # ------------------------------------------------------------------ #
    # 2-3. Roll + adjust + bar aggregate                                  #
    # ------------------------------------------------------------------ #
    if args.skip_bars and bars_path.exists():
        logger.info("Skipping roll+bars -- loading %s", bars_path)
        bars = pq.read_table(str(bars_path)).to_pandas()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    else:
        logger.info("=== Step 2: Roll + Panama adjust ===")
        continuous = build_continuous(raw_ticks_path, continuous_path, data_cfg)

        logger.info("=== Step 3: Aggregate to 1-min bars ===")
        bars = build_bars(continuous, bars_path, data_cfg)

    # ------------------------------------------------------------------ #
    # 4. Splits                                                           #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Time-ordered splits ===")
    train, val, test = split_bars(bars, env_cfg.get("splits", {}))

    # ------------------------------------------------------------------ #
    # 5. Features (fit on TRAIN only -- no leakage)                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5: Feature engineering ===")
    feature_arrays, price_arrays, pipeline = build_features(
        train, val, test, feat_cfg, out_dir
    )

    # ------------------------------------------------------------------ #
    # 6. Regime labels                                                    #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 6: Regime detection ===")
    regime_arrays = build_regimes(train, val, test, regime_cfg, out_dir)

    # ------------------------------------------------------------------ #
    # 7. Metadata                                                         #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 7: Save metadata ===")
    meta = save_metadata(bars, train, val, test, pipeline, out_dir)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Data preparation COMPLETE")
    logger.info("  Total bars    : %d", meta["total_bars"])
    logger.info("  Features      : %d", meta["n_features"])
    logger.info("  Train split   : %d bars  (%s to %s)",
                meta["splits"]["train"]["bars"],
                meta["splits"]["train"]["start"],
                meta["splits"]["train"]["end"])
    logger.info("  Val split     : %d bars  (%s to %s)",
                meta["splits"]["val"]["bars"],
                meta["splits"]["val"]["start"],
                meta["splits"]["val"]["end"])
    logger.info("  Test split    : %d bars  (%s to %s)",
                meta["splits"]["test"]["bars"],
                meta["splits"]["test"]["start"],
                meta["splits"]["test"]["end"])
    logger.info("")
    logger.info("Ready to train:")
    logger.info("  python scripts/train.py --skip-data-prep")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
