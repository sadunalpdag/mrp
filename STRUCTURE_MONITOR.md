# Structure Agent Test

`STRUCTURE_MONITOR_ENABLED=1` (default) enables an independent five-minute
rule-based observer in `ema.py`. It does not invoke an LLM, use broker methods,
change Crash candidate eligibility, or submit orders. `0` disables the observer
and its extra kline subscriptions.

The existing EMA Filter workbook receives one new worksheet, **Structure Agent
Test**. Other worksheets are not edited or deleted. Rows show the current top
50 active USDT perpetual contracts by 24-hour USDT quote turnover; unavailable
age or price data is not interpreted as eligibility. Contract age must be at
least 30 days. Only five days of candles are required, not 30 days of candles.
BTC/ETH changes are shown as context, not imposed as a trading filter.

The observer merges the existing history-cache files with closed WebSocket
bars, preferring Binance to alternate-venue candles. All 480 expected slots
must exist before structural analysis. Complete UTC 4-hour bars are aggregated
from 16 contiguous real 15-minute bars. A pivot is known only after two bars
have closed to its right.

Two independent LONG research hypotheses are monitored:

- Reversal: negative 24h change, prior descending confirmed 4h swings, first
  body-close above the last confirmed lower high, first 15m contact within 16
  bars at +/-0.20 ATR14, and bullish close above the level on that contact.
- Continuation: rising confirmed 4h swings, positive 24h change, >=1 ATR bullish
  impulse with >=1.5 times preceding 20-bar volume, 1-4 bearish pullback bars
  holding the impulse low, then a bullish close above the pullback highs.

At most five current candidates per hypothesis are selected, prioritizing
fresh confirmations then turnover rank. Indicative entry is the observed
ticker price, never a claimed historical fill. Stop is below the relevant
pullback low by 0.20 ATR. The illustrative target is 1.5 times price risk.
Known overhead pivot resistance must leave room for that target. Spread and
order-book depth are explicitly **unmeasured**, so confirmations are never
marked execution-eligible. There is no paper-fill/P&L simulator in this version.

Confirmed observations are deduplicated by symbol/method/candle timestamp and
retained locally for 30 days (maximum 2,000 records); the latest 100 are shown
below the diagnostic table. Restarting does not backdate a missed first touch.
State is `DATA_DIR/structure_agent_monitor.json`.

Missing bars propose one warmup every five minutes, with a rotating cursor and
at most three structure warmups outstanding in the existing history worker.
Due Crash jobs take precedence at submission. The existing archive checksum,
alternate-venue price validation, request pacing, IP cooldown and last-resort
REST budget remain enforced; no synthetic bars are inserted. A failed source
can leave a symbol waiting. Sheet/network failures remain in the observer's
daemon and do not stop the Crash loop. Logs distinguish `sheets_verified=1`
from local-only output and always include `execution=OFF`.
