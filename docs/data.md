# Preparing real data

The original MT5 CSVs are on another computer and have not been examined in this rebuild. Do not infer their timezone, instrument, timeframe or permissions from filenames alone.

For each dataset, record locally:

- Instrument, broker/provider, export date and covered period.
- Whether timestamps denote bar opens or closes, source timezone and daylight-saving behavior.
- Timeframe, gaps, duplicate timestamps and whether the last bar was still forming.
- Whether OHLC represents bid, ask or another price series; units of spread/volume if supplied.
- Source checksum and permission to use or redistribute the data.

The current loader accepts exactly `time,open,high,low,close`, with explicit UTC timestamps, positive finite prices and consistent OHLC bounds. Convert only after verifying the source semantics. MT5's native tab-delimited/date/time exports are not automatically interpreted. Keep the original file unchanged and record conversion decisions. Do not invent UTC by attaching a timezone to an unknown local timestamp.

Use a single instrument and timeframe per run. Supply at least 80 completed bars; that minimum exercises the software and is not a statistically sufficient research sample. Missing intervals are not synthesized. The loader checks ordering and OHLC consistency, not data provenance or a trading calendar.

Run from the repository root:

```sh
python -m trading_research --csv prices.csv --cost-bps 2 --output artifacts/real-data-01
```

The cost value is an example, not a broker estimate. Choose and record assumptions before reviewing results. CSVs and generated artifacts are ignored by Git. Publish data or reports only after checking contents and redistribution permissions; account information and credentials are unnecessary.
