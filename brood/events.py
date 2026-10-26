"""Load economic-calendar events and compute how much each one surprised the market."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["datetime_utc", "event", "actual"]

# Surprise buckets, in standard deviations of that event's historical surprises.
BUCKETS = [
    ("big miss", float("-inf"), -1.0),
    ("small miss", -1.0, -0.25),
    ("in line", -0.25, 0.25),
    ("small beat", 0.25, 1.0),
    ("big beat", 1.0, float("inf")),
]
BUCKET_ORDER = [name for name, _, _ in BUCKETS]


def load_events(path: str) -> pd.DataFrame:
    """Read an events CSV.

    Columns: datetime_utc, event, actual, forecast (optional), previous (optional).
    Numbers may carry %, K, M or B suffixes, as calendar sites print them.
    """
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    for col in ("actual", "forecast", "previous"):
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = df[col].map(parse_number)
    return df.sort_values("datetime_utc").reset_index(drop=True)


def parse_number(value) -> float:
    """Turn '0.3%', '256K', '-1.2M' or '' into a float (NaN when empty)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    text = str(value).strip().replace(",", "")
    if not text:
        return float("nan")
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1].upper(), 1.0)
    if scale != 1.0 or text.endswith("%"):
        text = text[:-1]
    return float(text) * scale


def add_surprise(df: pd.DataFrame, basis: str = "forecast") -> pd.DataFrame:
    """Add surprise, surprise_z and bucket columns.

    basis="forecast" compares actual with the market consensus (the right way).
    basis="previous" compares with the last release, a rough fallback when no
    consensus history is available.
    """
    if basis not in ("forecast", "previous"):
        raise ValueError("basis must be 'forecast' or 'previous'")
    out = df.copy()
    out["surprise"] = out["actual"] - out[basis]
    # Scale each event by its own surprise history so CPI (0.1%) and NFP (50K)
    # land on the same footing.
    std = out.groupby("event")["surprise"].transform(lambda s: s.std(ddof=0))
    out["surprise_z"] = out["surprise"] / std.where(std > 0)
    out.loc[out["surprise"] == 0, "surprise_z"] = 0.0
    out["bucket"] = out["surprise_z"].map(bucket_for)
    return out


def bucket_for(z: float) -> str | None:
    if pd.isna(z):
        return None
    for name, lo, hi in BUCKETS:
        if lo <= z < hi:
            return name
    return None
