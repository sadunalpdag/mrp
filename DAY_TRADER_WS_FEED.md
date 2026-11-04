# Day Trader research: Binance WebSocket-only closed candle feed

This is a **read-only data bridge**, not the original US-stock Day Trader Skill.
The upstream Day Trader Skill expects Yahoo 1m/5m candles, NYSE session hours
and (for its Time Machine) 60 days of historical data. Do not run its stock
entry/scanner/simulator unchanged on USDT-M futures.

## Data path

Existing \`ema.py\` Binance USD-M Futures socket -> confirmed \`@kline_15m\`
events with \`k.x=true\` -> \`DayTraderWsFeed.submit()\` (non-blocking queue) ->
one JSON file per symbol under \`$DATA_DIR/day_trader_ws/<SYMBOL>.json\`.

- **No new WebSocket connection, subscriptions or Binance REST calls.**
- **No data from REST, Binance daily archive, Coinbase or Bybit** enters this feed.
- Separate research file only; Crash Bounce V2 decisions, BUY/TP logic, risk
  settings, Telegram and Binance user data streams are **not changed**.
- Files contain only public market OHLCV; no API keys, account or orders.
- Per symbol keep up to 960 real closed 15m candles, timestamp-sorted,
  deduplicated. Missing candles are counted in \`gap_slots\`, not fabricated.
- The source collects data only when the underlying existing Crash scanner
  subscribes to that symbol. A stopped/disconnected socket, rotation out of
  the top selection, or Render disk loss creates gaps.
- WebSocket cannot replay arbitrary old candles: building the full five days
  from this feed requires five days of **actually received** closed bars. The
  existing V2 history cache may be fuller but is not included in the strict
  WS-only feed.
- Writes occur in a separate daemon thread; the WebSocket callback never
  waits for disk. An overflow or disk failure is logged and does not stop V2.

## Configuration

\`CRASH_DAY_TRADER_WS_ENABLED=1\` (default): enabled on \`ema.py\` startup.
\`CRASH_DAY_TRADER_WS_ENABLED=0\`: disable entirely.

The feed uses the pre-existing \`DATA_DIR\` location. To keep files across
deploys, the Render service must mount persistent storage there. The GitHub
\`render.yaml\` declares a disk, but confirm the **live** service configuration.

## JSON

\`\`\`json
{
  "schema": "crash_day_trader_ws_v1",
  "symbol": "BTCUSDT",
  "interval": "15m",
  "source": "BINANCE_USDT_M_WEBSOCKET",
  "closed_only": true,
  "updated_at": 1793800000.0,
  "candle_count": 1,
  "gap_slots": 0,
  "candles": [
    {
      "open_time": 1793799000000,
      "open": "70000",
      "high": "70200",
      "low": "69900",
      "close": "70100",
      "volume": "120",
      "close_time": 1793799899999
    }
  ]
}
\`\`\`

\`updated_at\` is Unix time in UTC seconds, candle timestamps UTC milliseconds.
A crypto-native Day Trader entry/risk/shadow replay can consume these records
without any new market-data HTTP requests. A 60-day historical replay cannot
be bootstrapped from a fresh WS connection.

## Tests

\`\`\`bash
python -m unittest test_day_trader_ws_feed test_crash_websocket -v
\`\`\`
