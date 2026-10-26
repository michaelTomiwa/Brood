"""Free price data around event times.

Sources:
  dukascopy  forex, gold, oil and indices (tick data, no account needed)
  binance    crypto (1-minute bars, no account needed)
  csv        your own export, e.g. from MetaTrader 5 (columns: time + close/bid/price)

Each loader returns a pandas Series of prices indexed by UTC timestamp.
"""

from __future__ import annotations

import json
import lzma
import struct
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache")
USER_AGENT = "Mozilla/5.0 (brood news study)"

# Dukascopy stores prices as integers; divide by this to get the real price.
# Most FX pairs use 1e5; JPY pairs, metals, oil and indices use 1e3.
DUKASCOPY_POINT = {
    "EURUSD": 1e5, "GBPUSD": 1e5, "AUDUSD": 1e5, "NZDUSD": 1e5,
    "USDCAD": 1e5, "USDCHF": 1e5, "EURGBP": 1e5,
    "USDJPY": 1e3, "EURJPY": 1e3, "GBPJPY": 1e3,
    "XAUUSD": 1e3, "XAGUSD": 1e3,
    "LIGHTCMDUSD": 1e3, "BRENTCMDUSD": 1e3,          # WTI and Brent oil
    "USATECHIDXUSD": 1e3, "USA30IDXUSD": 1e3,        # NAS100 and US30
    "USA500IDXUSD": 1e3,                              # SPX500
}

# Prop-firm style names mapped to Dukascopy's names.
ALIASES = {
    "NAS100": "USATECHIDXUSD", "US100": "USATECHIDXUSD",
    "US30": "USA30IDXUSD", "SPX500": "USA500IDXUSD", "US500": "USA500IDXUSD",
    "USOIL": "LIGHTCMDUSD", "WTI": "LIGHTCMDUSD", "UKOIL": "BRENTCMDUSD",
    "GOLD": "XAUUSD", "SILVER": "XAGUSD",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _cached(name: str, fetch) -> bytes:
    path = CACHE_DIR / name
    if path.exists():
        return path.read_bytes()
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _hours_between(start: datetime, end: datetime) -> list[datetime]:
    hour = start.replace(minute=0, second=0, microsecond=0)
    hours = []
    while hour <= end:
        hours.append(hour)
        hour += timedelta(hours=1)
    return hours


def parse_bi5(raw: bytes, hour: datetime, point: float) -> pd.Series:
    """Decode one Dukascopy hour file into a Series of mid prices."""
    if not raw:
        return pd.Series(dtype=float)
    data = lzma.decompress(raw)
    times, mids = [], []
    for ms, ask, bid, _, _ in struct.iter_unpack(">IIIff", data):
        times.append(hour + timedelta(milliseconds=ms))
        mids.append((ask + bid) / 2 / point)
    return pd.Series(mids, index=pd.DatetimeIndex(times), dtype=float)


def dukascopy_ticks(symbol: str, start: datetime, end: datetime,
                    point: float | None = None) -> pd.Series:
    symbol = ALIASES.get(symbol.upper(), symbol.upper())
    point = point or DUKASCOPY_POINT.get(symbol)
    if point is None:
        raise ValueError(f"Unknown point size for {symbol}; pass --point")
    parts = []
    for hour in _hours_between(start, end):
        # Dukascopy months are zero-based.
        path = f"{symbol}/{hour.year}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
        raw = _cached(f"dukascopy/{path}",
                      lambda p=path: _get(f"https://datafeed.dukascopy.com/datafeed/{p}"))
        parts.append(parse_bi5(raw, hour, point))
    return _join(parts)


def binance_minutes(symbol: str, start: datetime, end: datetime) -> pd.Series:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    name = f"binance/{symbol.upper()}_{start_ms}_{end_ms}.json"
    url = (f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}"
           f"&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000")
    rows = json.loads(_cached(name, lambda: _get(url)))
    # Index each bar by its close time so "price at t" never peeks at the future.
    times = [datetime.fromtimestamp((r[0] + 60_000) / 1000, tz=timezone.utc) for r in rows]
    return pd.Series([float(r[4]) for r in rows], index=pd.DatetimeIndex(times), dtype=float)


def load_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.lower().strip("<>"): c for c in df.columns}
    time_col = next((cols[c] for c in ("time", "datetime", "date", "timestamp") if c in cols), None)
    price_col = next((cols[c] for c in ("close", "mid", "bid", "price") if c in cols), None)
    if time_col is None or price_col is None:
        raise ValueError(f"{path} needs a time column and a close/bid/price column")
    index = pd.to_datetime(df[time_col], utc=True)
    return pd.Series(df[price_col].astype(float).values, index=index).sort_index()


def prices_around(source: str, symbol: str, times, before: timedelta, after: timedelta,
                  csv_path: str | None = None, point: float | None = None) -> pd.Series:
    """Fetch prices covering [t - before, t + after] for every event time."""
    if source == "csv":
        if not csv_path:
            raise ValueError("--prices-csv is required with --source csv")
        return load_csv(csv_path)
    parts = []
    for t in times:
        t = pd.Timestamp(t).to_pydatetime()
        if source == "dukascopy":
            parts.append(dukascopy_ticks(symbol, t - before, t + after, point))
        elif source == "binance":
            parts.append(binance_minutes(symbol, t - before, t + after))
        else:
            raise ValueError(f"Unknown source: {source}")
    return _join(parts)


def _join(parts: list[pd.Series]) -> pd.Series:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    series = pd.concat(parts).sort_index()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    return series[~series.index.duplicated(keep="last")]
