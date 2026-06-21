"""Volume-based contract rolling and Panama back-adjustment.

No-lookahead guarantees:
- Roll decisions use only the current day's volume (not next day's).
- Panama adjustment computes the price gap AT roll time using contemporaneous
  data, then shifts all prior history uniformly. Historical price levels
  change (expected for continuous back-adjusted series) but no future bar
  data is consumed to make the decision.
"""
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CME month code -> month number (for expiry-based sorting)
_CME_MONTH = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12,
}


def _contract_expiry_key(symbol: str):
    """Return (year, month) sort key for a GC futures symbol like GCG3, GCZ4.

    Single-digit year: 0->2030, 1->2031, ..., 9->2029 (2020s decade base).
    Unknown symbols sort last.
    """
    m = re.match(r'^GC([A-Z])(\d)$', symbol)
    if not m:
        return (9999, 99)
    month = _CME_MONTH.get(m.group(1), 99)
    yr_digit = int(m.group(2))
    # 3->2023, 4->2024, ..., 9->2029, 0->2030, 1->2031, 2->2032
    year = 2020 + yr_digit if yr_digit >= 3 else 2030 + yr_digit
    return (year, month)


class VolumeBasedRoller:
    """Build a continuous front-month series via volume-based roll logic.

    Roll decision: at end of day D, if next_contract.daily_volume >=
    current_contract.daily_volume * volume_threshold_ratio, roll to next.
    Decision uses only data available by end of day D.
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("roll", config or {})
        self.volume_threshold_ratio = float(cfg.get("volume_threshold_ratio", 1.0))
        self.min_roll_days = int(cfg.get("min_roll_days", 1))

    def build_continuous(self, ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks.empty:
            return ticks.copy()

        df = ticks.copy().sort_values("ts_event")
        df["_date"] = df["ts_event"].dt.normalize()

        daily_vol = (
            df.groupby(["_date", "symbol"])["size"]
            .sum()
            .unstack(fill_value=0)
        )

        # Sort by actual expiry date (year, month) — NOT lexicographically.
        # GCG4 (Feb 2024) must come after GCZ3 (Dec 2023), but lex order gives
        # GCZ3 < GCZ4 < GCZ5 which is wrong.
        symbols_sorted = sorted(df["symbol"].unique(), key=_contract_expiry_key)
        current_front = symbols_sorted[0]
        days_since_roll = 0
        front_by_date: dict = {}

        for date in sorted(daily_vol.index):
            row = daily_vol.loc[date]
            days_since_roll += 1

            if days_since_roll >= self.min_roll_days:
                try:
                    idx = symbols_sorted.index(current_front)
                    candidates = symbols_sorted[idx + 1:]
                except ValueError:
                    candidates = []

                for nxt in candidates:
                    cur_vol = float(row.get(current_front, 0))
                    nxt_vol = float(row.get(nxt, 0))
                    if nxt_vol >= cur_vol * self.volume_threshold_ratio:
                        logger.info("Roll %s -> %s on %s", current_front, nxt, date.date())
                        current_front = nxt
                        days_since_roll = 0
                        break

            front_by_date[date] = current_front

        df["front_month"] = df["_date"].map(front_by_date)
        continuous = df[df["symbol"] == df["front_month"]].copy()
        continuous = continuous.drop(columns=["_date"]).reset_index(drop=True)
        logger.info("Continuous series: %d ticks", len(continuous))
        return continuous


class PanamaBackAdjuster:
    """Apply Panama canal back-adjustment across contract rolls.

    For each roll event at index i:
        gap = price[i] - price[i-1]
    All prices before index i are shifted +gap so the series is continuous.

    Multiple rolls accumulate correctly: bars before roll k get the sum of
    all gaps at rolls k, k+1, ... (roll k gap shifts bars before k, roll
    k+1 gap shifts bars before k+1, which includes bars before k again).
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df = df.copy()
            df["price_adj"] = df.get("price", pd.Series(dtype=float))
            return df

        df = df.copy().sort_values("ts_event").reset_index(drop=True)
        prices = df["price"].values.astype(float)
        n = len(prices)

        if "front_month" not in df.columns:
            df["price_adj"] = prices
            return df

        symbols = df["front_month"].values

        # Identify roll indices and price gaps
        roll_events = []  # (roll_idx, gap)
        for i in range(1, n):
            if symbols[i] != symbols[i - 1]:
                gap = prices[i] - prices[i - 1]
                roll_events.append((i, gap))
                logger.debug("Panama gap at bar %d (%s->%s): %.4f",
                             i, symbols[i - 1], symbols[i], gap)

        # Apply cumulative adjustment to all pre-roll history
        adj = np.zeros(n)
        for roll_idx, gap in roll_events:
            adj[:roll_idx] += gap  # shift all bars before this roll upward by gap

        df["price_adj"] = prices + adj
        return df
