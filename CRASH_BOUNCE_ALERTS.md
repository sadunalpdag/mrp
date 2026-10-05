# Crash-only alerts
ema.py now runs only the foreground crash-bounce loop. Legacy strategy data collection, per-symbol price polling, account/trading routines, Google Sheets heartbeat and Telegram command handlers are not started. No orders are opened.

- Eligible: rank trading USDT-M perpetuals by current rolling 24h change, select the 20 biggest losers, then require change <= -12%. Coins on cooldown are not replaced with the 21st loser.
- One new signal per coin per rolling 24h, persisted in DATA_DIR/crash_bounce_state.json.
- Entry candidate: minimum of last 16 fully closed 15m lows and observed price, plus 0.3%, rounded up to tick. Gross TP: entry plus 1%, rounded up to tick.
- Levels are conditional, unbacktested reference levels; notifications do not verify a reversal or execution.
- One bulk 24h ticker call per scan, hourly exchangeInfo, 20 candles only for eligible coins without a current cooldown.
- Nominal scan start interval: 180 seconds; long scans and cooldowns delay it. Intraminute transient threshold crossings may be missed.
- Startup waits at least 120 seconds to let old process traffic age out.
- REST requests are serial, at least one second apart, with a conservative local weight budget of 120/minute. IP usage >=50% of the exchangeInfo minute limit pauses requests until the next minute.
- 429: at least 120s exponential cooldown. 403/418: at least 900s. Honor longer Retry-After and ban expiry from the error body, with a 5s margin.
- Cooldown saved atomically in DATA_DIR/crash_bounce_rest_cooldown.json and respected on restart. Invalid persistence fails closed.
- Pending Telegram deliveries expire after 15 minutes; failed acknowledgements can cause a retry duplicate.
- Use persistent DATA_DIR and one running instance. Stop old replicas before restart; another process sharing the public IP can still exhaust Binance's IP limit. No ban-free guarantee.
- Deploy/restart the service to activate a changed ema.py. Repository update alone does not prove deployment.

Offline validation:
python -m py_compile ema.py
python -m unittest discover -p 'test_crash*.py' -v

12 tests cover top-20 ranking, threshold, price rounding, daily deduplication, Telegram routing/retries, persistence failures, startup isolation, cooldown persistence, 429/418, endpoint restrictions, spacing and weight limits. No live Binance or Telegram calls in tests.
