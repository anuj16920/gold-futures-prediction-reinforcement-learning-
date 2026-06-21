"""Load raw Databento zstd-compressed trade CSV files.

Supports:
- A directory of *.trades.csv.zst files
- A zip archive (GLBX-*.zip) containing *.trades.csv.zst entries
"""
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import zstandard as zstd

logger = logging.getLogger(__name__)

_KEEP_COLS = {"ts_event", "price", "size", "symbol", "side"}


class DatabentoLoader:
    """Load zstd-compressed Databento trade CSVs from disk.

    Priority order:
    1. raw_dir/*.trades.csv.zst  (already extracted)
    2. parent of raw_dir / GLBX-*.zip  (original download)
    """

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load_all(self) -> pd.DataFrame:
        zst_files = sorted(self.raw_dir.glob("*.trades.csv.zst"))
        if zst_files:
            frames = [self._read_zst_file(f) for f in zst_files]
            frames = [f for f in frames if f is not None]
        else:
            # Search for zip: raw_dir, raw_dir.parent (data/), project root
            search_dirs = [
                self.raw_dir,
                self.raw_dir.parent,
                self.raw_dir.parent.parent,
            ]
            zip_files = []
            for d in search_dirs:
                zip_files = sorted(d.glob("GLBX-*.zip"))
                if not zip_files:
                    zip_files = sorted(d.glob("*.zip"))
                if zip_files:
                    break
            if not zip_files:
                logger.warning("No trade data found in %s", self.raw_dir)
                return pd.DataFrame()
            frames = self._read_from_zip(zip_files[0])

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("ts_event").reset_index(drop=True)
        logger.info("Loaded %d ticks", len(df))
        return df

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_from_zip(self, zip_path: Path) -> list:
        frames = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".trades.csv.zst"))
            logger.info("Reading %d files from %s", len(names), zip_path.name)
            for name in names:
                raw_bytes = zf.read(name)
                df = self._decompress_and_parse(raw_bytes, name)
                if df is not None:
                    frames.append(df)
        return frames

    def _read_zst_file(self, path: Path) -> pd.DataFrame:
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
        return self._decompress_and_parse(raw_bytes, path.name)

    def _decompress_and_parse(self, raw_bytes: bytes, name: str = "") -> pd.DataFrame:
        try:
            dctx = zstd.ZstdDecompressor()
            # Use streaming reader to avoid max_output_size limits
            reader = dctx.stream_reader(io.BytesIO(raw_bytes))
            df = pd.read_csv(io.TextIOWrapper(reader, encoding="utf-8"), low_memory=False)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", name, exc)
            return None

        if "ts_event" not in df.columns:
            logger.warning("ts_event missing in %s, skipping", name)
            return None

        keep = [c for c in df.columns if c in _KEEP_COLS]
        df = df[keep].copy()

        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df["price"] = pd.to_numeric(df.get("price", pd.Series(dtype=float)), errors="coerce")
        df["size"] = pd.to_numeric(df.get("size", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)

        if "symbol" not in df.columns:
            df["symbol"] = "UNKNOWN"

        df = df.dropna(subset=["ts_event", "price"])

        # Drop calendar spreads (symbol contains '-', e.g. GCG3-GCJ3).
        # Keep only outright front-month contracts (e.g. GCG3, GCZ4).
        if "symbol" in df.columns:
            df = df[~df["symbol"].str.contains("-", na=False)]

        return df
