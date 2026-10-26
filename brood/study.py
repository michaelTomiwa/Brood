"""Measure what price did after each event, grouped by how big the surprise was."""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from .events import BUCKET_ORDER

# A price older than this is treated as missing (market closed, data gap).
MAX_STALENESS = timedelta(minutes=5)


def price_at(prices: pd.Series, when: pd.Timestamp) -> float:
    """Last known price at or before `when`, or NaN if it is stale."""
    idx = prices.index.searchsorted(when, side="right") - 1
    if idx < 0 or when - prices.index[idx] > MAX_STALENESS:
        return float("nan")
    return float(prices.iloc[idx])


def reactions(events: pd.DataFrame, prices: pd.Series, horizons: list[int],
              entry_delay: int = 0) -> pd.DataFrame:
    """One row per event with the % move over each horizon (in minutes).

    entry_delay=0 measures from the last price before the release.
    entry_delay=N measures from N minutes after it, which is what you could
    actually trade if your prop firm bans trading right around the news.
    """
    out = events.copy()
    for t_idx, t in events["datetime_utc"].items():
        if entry_delay:
            entry_time = t + timedelta(minutes=entry_delay)
        else:
            entry_time = t - timedelta(seconds=1)
        entry = price_at(prices, entry_time)
        out.loc[t_idx, "entry_price"] = entry
        for h in horizons:
            exit_price = price_at(prices, entry_time + timedelta(minutes=h))
            out.loc[t_idx, f"move_{h}m_pct"] = (exit_price / entry - 1) * 100
    return out


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence range for a win rate; wide when n is small."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def odds_table(rows: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Per event, surprise bucket and horizon: how often price rose, and by how much."""
    records = []
    for (event, bucket), group in rows.dropna(subset=["bucket"]).groupby(["event", "bucket"]):
        for h in horizons:
            moves = group[f"move_{h}m_pct"].dropna()
            n = len(moves)
            if n == 0:
                continue
            ups = int((moves > 0).sum())
            lo, hi = wilson_interval(ups, n)
            records.append({
                "event": event,
                "surprise": bucket,
                "horizon_min": h,
                "events": n,
                "price_up_pct": round(100 * ups / n, 1),
                "range_95_low": round(100 * lo, 1),
                "range_95_high": round(100 * hi, 1),
                "avg_move_pct": round(moves.mean(), 3),
                "avg_abs_move_pct": round(moves.abs().mean(), 3),
            })
    table = pd.DataFrame(records)
    if table.empty:
        return table
    table["surprise"] = pd.Categorical(table["surprise"], BUCKET_ORDER, ordered=True)
    return table.sort_values(["event", "surprise", "horizon_min"]).reset_index(drop=True)


def headline(table: pd.DataFrame, horizon: int) -> list[str]:
    """Plain-English summary lines for the biggest surprises at one horizon."""
    lines = []
    for _, r in table[(table["horizon_min"] == horizon)
                      & table["surprise"].isin(["big beat", "big miss"])].iterrows():
        direction = "UP" if r["price_up_pct"] >= 50 else "DOWN"
        share = r["price_up_pct"] if direction == "UP" else 100 - r["price_up_pct"]
        lines.append(
            f"{r['event']} {r['surprise']}: price went {direction} after {horizon}m "
            f"in {share:.0f}% of {r['events']} cases "
            f"(avg move {r['avg_move_pct']:+.3f}%)"
            + ("  <- too few events to trust" if r["events"] < 10 else "")
        )
    return lines
