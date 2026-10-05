# Crash-bounce Telegram alerts

The new scanner emits price plans for every currently trading Binance USDT-M
perpetual whose current rolling 24-hour ticker change is **-12% or lower**.
No previous-day drop, top-volume ranking, strategy score, session window or
RSI condition is used to exclude qualifying coins.

## Price plan

- Use the last 16 completed 15-minute candles (four hours).
- Reference low: the lowest low in these candles, or the currently observed
  price if it has already dropped below that low.
- Candidate entry: reference low plus 0.3%, rounded UP to the contract tick.
- Target: candidate entry plus 1%, rounded UP to the contract tick.
- These are conditional reference levels, not an executed entry, resting order
  or proven support. The message asks for the low to hold and a 15-minute
  reversal close. A tick-rounded target can be slightly above 1%.

The entry heuristic is new and has not been validated by the earlier backtests.
Targets are gross price changes, excluding fees and funding. No orders or
automatic TP orders are placed by this scanner.

## Timing, delivery and persistence

The scanner runs in its own daemon thread, starting before the existing symbol
initialization. It waits 60 seconds after each scan. Scans involving many
qualifying coins or slow HTTP responses can take longer; this is not a hard
60-second delivery guarantee. Exchange metadata refreshes hourly. The global
Binance REST cooldown and request slot helper are respected.

Each coin can generate at most one new signal in a rolling 24-hour window.
The cooldown, history and pending deliveries are atomically stored in
`DATA_DIR/crash_bounce_state.json`, including across restarts. Keep DATA_DIR
persistent on the deployment host. Corrupt state stops alerts until repaired
rather than resetting cooldowns and repeating alerts.

Failed Telegram deliveries retry the same plan, without creating a second
signal. Plans older than 15 minutes are marked stale instead of sending outdated
prices. A network interruption after Telegram accepts a message but before its
acknowledgment/persistence can still cause an at-least-once delivery duplicate.
Run one bot instance per DATA_DIR; cross-process deduplication is not provided.

Missing, stale or discontinuous candles are logged and retried next cycle.
There is no candidate count or volume cap. Set the existing BOT_TOKEN and CHAT_ID
environment variables for actual Telegram delivery.

## Existing strategies

Only CRASH_BOUNCE proactive Telegram messages are permitted. Other proactive
messages and document uploads are suppressed centrally. Replies/uploads
explicitly requested through existing Telegram commands remain available in a
thread-local command context; they do not allow background strategy messages.

Existing scans, PRE-SIGNAL logs, pump-watch data, performance tracking and Sheets
updates remain in the main loop. The previously disabled Fibonacci/run_parallel
scanner is not re-enabled. This does not restore other already disabled scanners
or alter legacy trade/command execution settings.

## Offline validation

```sh
python -m py_compile ema.py
python -m unittest discover -p 'test_crash_bounce.py' -v
```

Tests extract only the functions under test from the AST to avoid importing the
legacy bot's startup side effects. They mock Telegram and Binance; no live
messages or orders are sent by the tests.
