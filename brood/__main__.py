"""Command line: python -m brood study ...  |  python -m brood demo"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .events import add_surprise, load_events
from .prices import prices_around
from .study import headline, odds_table, reactions


def run_study(events: pd.DataFrame, prices: pd.Series, horizons: list[int],
              entry_delay: int, out: str | None) -> pd.DataFrame:
    rows = reactions(events, prices, horizons, entry_delay)
    table = odds_table(rows, horizons)
    if table.empty:
        print("No events had price data. Check the dates, symbol and source.")
        return table
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.to_string(index=False))
    print()
    for line in headline(table, horizons[len(horizons) // 2]):
        print(line)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out.replace(".csv", "_events.csv"), index=False)
        table.to_csv(out, index=False)
        print(f"\nSaved {out} and {out.replace('.csv', '_events.csv')}")
    return table


def cmd_study(args) -> None:
    horizons = [int(h) for h in args.horizons.split(",")]
    events = add_surprise(load_events(args.events), basis=args.basis)
    if args.event:
        events = events[events["event"].str.contains(args.event, case=False)]
    before = timedelta(minutes=10)
    after = timedelta(minutes=args.entry_delay + max(horizons) + 5)
    print(f"Loading {args.symbol} prices for {len(events)} events from {args.source}...")
    prices = prices_around(args.source, args.symbol, events["datetime_utc"], before, after,
                           csv_path=args.prices_csv, point=args.point)
    run_study(events, prices, horizons, args.entry_delay, args.out)


def cmd_demo(args) -> None:
    """Fake data where a beat pushes price down, to show what the output looks like."""
    rng = np.random.default_rng(7)
    times = pd.date_range("2021-01-13 13:30", periods=48, freq="28D", tz="UTC")
    forecast = rng.normal(0.3, 0.1, len(times)).round(1)
    actual = (forecast + rng.normal(0, 0.15, len(times))).round(1)
    events = add_surprise(pd.DataFrame({
        "datetime_utc": times, "event": "DEMO CPI m/m",
        "actual": actual, "forecast": forecast, "previous": np.nan,
    }))
    parts = []
    for t, z in zip(times, events["surprise_z"].fillna(0)):
        minutes = pd.date_range(t - pd.Timedelta(minutes=15), periods=120, freq="1min")
        drift = np.where(minutes >= t, -0.08 * z, 0.0)
        noise = rng.normal(0, 0.03, len(minutes)).cumsum()
        parts.append(pd.Series(2000 * (1 + (drift + noise) / 100), index=minutes))
    print("DEMO ONLY: synthetic data, these numbers mean nothing about real markets.\n")
    run_study(events, pd.concat(parts), [5, 15, 60], 0, None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="brood")
    sub = parser.add_subparsers(dest="command", required=True)

    study = sub.add_parser("study", help="measure price reactions to news events")
    study.add_argument("--events", required=True, help="events CSV (see data/events/TEMPLATE.csv)")
    study.add_argument("--symbol", required=True, help="e.g. XAUUSD, NAS100, EURUSD, BTCUSDT")
    study.add_argument("--source", default="dukascopy", choices=["dukascopy", "binance", "csv"])
    study.add_argument("--prices-csv", help="price file when --source csv")
    study.add_argument("--event", help="only events whose name contains this text")
    study.add_argument("--horizons", default="5,15,60", help="minutes after entry, comma separated")
    study.add_argument("--entry-delay", type=int, default=0,
                       help="enter N minutes after the release (prop-firm news window)")
    study.add_argument("--basis", default="forecast", choices=["forecast", "previous"],
                       help="what 'surprise' is measured against")
    study.add_argument("--point", type=float, help="Dukascopy price divisor for unknown symbols")
    study.add_argument("--out", help="save the odds table to this CSV")
    study.set_defaults(func=cmd_study)

    demo = sub.add_parser("demo", help="run the study on fake data to see the output")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
