import lzma
import struct
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from brood.__main__ import main
from brood.events import add_surprise, load_events, parse_number
from brood.prices import parse_bi5
from brood.study import odds_table, price_at, reactions, wilson_interval


def test_parse_number_handles_calendar_formats():
    assert parse_number("0.3%") == pytest.approx(0.3)
    assert parse_number("256K") == 256_000
    assert parse_number("-1.2M") == pytest.approx(-1_200_000)
    assert np.isnan(parse_number(""))


def test_load_events_and_surprise(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text(
        "# comment\n"
        "datetime_utc,event,actual,forecast,previous\n"
        "2024-02-13 13:30,CPI,0.3%,0.2%,0.2%\n"
        "2024-03-12 12:30,CPI,0.4%,0.4%,0.3%\n"
        "2024-04-10 12:30,CPI,0.2%,0.4%,0.4%\n"
    )
    df = add_surprise(load_events(str(path)))
    assert list(df["surprise"].round(2)) == [0.1, 0.0, -0.2]
    assert df["bucket"].tolist() == ["small beat", "in line", "big miss"]


def test_price_at_ignores_stale_and_future_prices():
    idx = pd.to_datetime(["2024-01-01 12:00", "2024-01-01 12:10"], utc=True)
    prices = pd.Series([100.0, 110.0], index=idx)
    assert price_at(prices, pd.Timestamp("2024-01-01 12:03", tz="UTC")) == 100.0
    assert np.isnan(price_at(prices, pd.Timestamp("2024-01-01 12:07", tz="UTC")))
    assert np.isnan(price_at(prices, pd.Timestamp("2024-01-01 11:59", tz="UTC")))


def make_market(n=40, effect=-0.1, seed=1):
    """Events where a positive surprise moves price by `effect` % per sigma."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2022-01-01 13:30", periods=n, freq="7D", tz="UTC")
    events = add_surprise(pd.DataFrame({
        "datetime_utc": times, "event": "CPI",
        "actual": rng.normal(0, 1, n), "forecast": 0.0, "previous": np.nan,
    }))
    parts = []
    for t, z in zip(times, events["surprise_z"]):
        minutes = pd.date_range(t - pd.Timedelta(minutes=10), periods=90, freq="1min")
        move = np.where(minutes >= t, effect * z, 0.0)
        parts.append(pd.Series(100 * (1 + move / 100), index=minutes))
    return events, pd.concat(parts)


def test_beats_push_price_down_when_effect_is_negative():
    events, prices = make_market()
    rows = reactions(events, prices, [5, 15])
    table = odds_table(rows, [5, 15])
    big_beat = table[(table["surprise"] == "big beat") & (table["horizon_min"] == 15)].iloc[0]
    big_miss = table[(table["surprise"] == "big miss") & (table["horizon_min"] == 15)].iloc[0]
    assert big_beat["price_up_pct"] == 0.0
    assert big_miss["price_up_pct"] == 100.0
    assert big_beat["avg_move_pct"] < 0 < big_miss["avg_move_pct"]


def test_entry_delay_skips_the_initial_jump():
    # The whole move happens at the release, so entering 5 minutes later sees nothing.
    events, prices = make_market()
    rows = reactions(events, prices, [15], entry_delay=5)
    assert rows["move_15m_pct"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_wilson_interval_is_wide_for_small_samples():
    lo, hi = wilson_interval(4, 5)
    assert hi - lo > 0.4
    lo, hi = wilson_interval(400, 500)
    assert hi - lo < 0.1


def test_parse_bi5_round_trip():
    hour = datetime(2024, 1, 10, 13, tzinfo=timezone.utc)
    raw = lzma.compress(struct.pack(">IIIff", 1500, 2030500, 2030100, 1.0, 1.0),
                        format=lzma.FORMAT_ALONE)
    series = parse_bi5(raw, hour, 1e3)
    assert series.iloc[0] == pytest.approx(2030.3)
    assert series.index[0] == pd.Timestamp("2024-01-10 13:00:01.5", tz="UTC")


def test_demo_runs(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "DEMO ONLY" in out and "big beat" in out
