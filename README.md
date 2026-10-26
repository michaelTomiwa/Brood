# Brood

Step 1 of a news-driven trading bot: **measure the real odds** of how a market
moves after economic news, before risking any money.

For each release (US CPI, NFP, Fed decisions...) it compares the actual number
with the forecast, groups releases by how big the surprise was, and reports how
often price went up or down 5, 15 and 60 minutes later, with a 95% range that
shows how much to trust each number.

## Setup

```bash
pip install -r requirements.txt
python -m brood demo        # see the output format on fake data
python -m pytest            # run the tests
```

## Running a real study

1. **Collect event history** into a CSV like `data/events/TEMPLATE.csv`:
   `datetime_utc,event,actual,forecast,previous`. Economic-calendar sites
   (Forex Factory, Investing.com, Myfxbook) show past releases with the forecast.
   Aim for at least 5 years per event (about 60 CPI or NFP releases).
2. **Run it**. Price data downloads automatically and is cached in `data/cache/`:

```bash
# Gold around US CPI (Dukascopy tick data, free)
python -m brood study --events data/events/us_cpi.csv --symbol XAUUSD --out results/cpi_gold.csv

# NAS100, entering 5 minutes after the release (prop-firm news window)
python -m brood study --events data/events/us_cpi.csv --symbol NAS100 --entry-delay 5

# Bitcoin around NFP (Binance, free)
python -m brood study --events data/events/us_nfp.csv --symbol BTCUSDT --source binance

# Your own MT5 export (columns: time + close)
python -m brood study --events data/events/us_cpi.csv --symbol XAUUSD --source csv --prices-csv my_xauusd_m1.csv
```

Symbols: XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY, NAS100, US30, SPX500, USOIL,
UKOIL (Dukascopy), and any Binance pair such as BTCUSDT.

## Reading the results

| Column | Meaning |
|---|---|
| `surprise` | big miss / small miss / in line / small beat / big beat, scaled to that event's usual surprise size |
| `price_up_pct` | % of those releases where price was higher after `horizon_min` minutes |
| `range_95_low/high` | where the true odds likely sit; a wide range means not enough data |
| `avg_move_pct` | average signed move, which tells you whether the edge is worth the spread |

Odds near 50%, or a 95% range that crosses 50%, mean there is **no edge**.
Results based on fewer than about 30 events are a hint, not a strategy.

## Notes

- Many prop firms restrict trading a few minutes before and after high-impact
  news. Use `--entry-delay` to measure only what you would be allowed to trade.
- Spreads widen at the release, so real fills will be worse than these mid prices.
- `--basis previous` measures surprise against the last release when you have no
  forecast history. It is much weaker than using the forecast.
