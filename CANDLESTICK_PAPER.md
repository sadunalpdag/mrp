# Hammer / Morning Star — eight-hour paper observer

This opt-in feature is isolated from Crash Bounce, Bull Flag and every broker path. It does **not** submit actual Binance orders.

- Toggle: `CANDLESTICK_PAPER_ENABLED=1` on the existing Render `mrp` worker.
- Uses existing `api2` / `API2` / `COINBEACON_API_KEY`.
- Each quarter hour, scans CoinBeacon `/public/detectors/candlesticks` for Binance Futures USDT pairs on 15m, 1h, 4h and 1d, filtering for Hammer or Morning Star and excluding `setup` states.
- Accepts at most **10 different symbols** simultaneously. Once full, all detector calls stop until a symbol's full eight-hour review finishes. The same coin cannot fill two slots.
- Each accepted case opens a **paper-only** LONG at the detector snapshot price, with a 1% profit target; dedupe key is coin + timeframe + pattern name + completion candle anchor.
- No WebSocket polling and no ongoing OHLCV requests. After eight hours (plus time for the last closed 15m candle), retrieve CoinBeacon 15m OHLCV once per coin. Compute TP hit, close return, maximum upside and drawdown. Missing candles are **not** treated as losses or guessed; retry when data is available. A 15m candle cannot reveal intrabar execution order; the first TP candle's close is used as a conservative reporting timestamp.
- Writes signals and results into `HAMMER_MORNING_STAR` worksheet in the existing configured Google Sheets workbook, using `GOOGLE_SERVICE_ACCOUNT_JSON`. `CANDLESTICK_PAPER_SHEETS_ID` optionally overrides the workbook. Records persist in `DATA_DIR/candlestick_paper.sqlite3` across restarts.
- After midnight **Europe/Paris**, writes yesterday's compact compressed daily JSON into R2 at `candlestick-paper/daily/YYYY/MM/YYYY-MM-DD.json.gz`. Days containing still-active eight-hour cases are rewritten when their results finish. No candle history is stored in that nightly archive.
- Sends Telegram entries and outcomes through existing `BOT_TOKEN`/`CHAT_ID` when present.

Note: the CoinBeacon detector board can truncate after 150 rows; a shortfall is logged and not misreported as zero patterns.

## Before activating

Review API credits (4 detector requests per cycle until ten positions are full), verify `COINBEACON_API_KEY`, existing Sheets access and R2 credentials, then set `CANDLESTICK_PAPER_ENABLED=1` on the existing Render worker. Confirm service logs and the first Sheets row; no duplicate worker should be launched.
