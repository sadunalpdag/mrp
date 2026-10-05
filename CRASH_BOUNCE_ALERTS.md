# Crash-only alerts
ema.py now runs only the foreground crash-bounce loop. Legacy strategy data collection, per-symbol price polling, account/trading routines, Google Sheets heartbeat and Telegram command handlers are not started. No orders are opened.

- Eligible: rank trading USDT-M perpetuals by current rolling 24h change, select the 20 biggest losers, then require change <= -12%. Coins on cooldown are not replaced with the 21st loser.
- One new signal per coin per rolling 24h, persisted in DATA_DIR/crash_bounce_state.json.
- Entry candidate: minimum of last 16 fully closed 15m lows and observed price, plus 0.3%, rounded up to tick. Gross TP: entry plus 1%, rounded up to tick.
- Levels are conditional, unbacktested reference levels; notifications do not verify a reversal or execution.
- One bulk 24h ticker call per scan, hourly exchangeInfo, 64 candles only for eligible coins without a current cooldown.
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

21 tests cover top-20 ranking, threshold, price rounding, daily deduplication, Telegram routing/retries, persistence failures, startup isolation, cooldown persistence, 429/418, endpoint restrictions, spacing and weight limits. No live Binance or Telegram calls in tests.

## Virtual outcome tracking and research exports
- Every 180s bulk ticker snapshot also updates all pending/open crash signals, including coins outside the current top 20. This adds no Binance request.
- Model: sampled_limit_touch_v1. Only AFTER signal creation, an observed last price <= planned entry is a hypothetical limit fill at planned entry. This does not verify the reversal confirmation in the alert or actual execution.
- A later sample >= planned TP labels TP. The entry snapshot cannot also label TP.
- Entry wait expires after 72h: NO_ENTRY_OBSERVED. Once entered, another 72h is allowed for TP: otherwise TIMEOUT.
- Track observed TP hours, sampled favorable/adverse excursion, last observed return and gaps >540 seconds. TP rate denominator is TP + TIMEOUT; OPEN/WAIT_ENTRY/no-entry/legacy records are separately shown.
- Prices between samples are unknown; TIMEOUT means no sampled TP, not proof TP never occurred. Fees, funding, slippage and leverage are unmodeled.
- Pre-existing records are LEGACY_UNKNOWN. Historical results are not fabricated.
- New signal features include up to 64 fully closed 15m OHLCV candles, Wilder RSI14, EMA7/25 and slope, recent volume ratio, 1h return, 24h ticker fields and loser rank.
- UTC daily append-only JSONL stores top-20 selection, threshold/cooldown decisions, BTC/ETH context, new signals, price samples and results. Research price samples continue up to 6 days after each signal, including after TP, for alternative TP/SL comparisons.
- State retains 180 days of signals; JSONL observations and daily ZIP exports are retained without automatic deletion in DATA_DIR/crash_research. Ensure DATA_DIR is persistent and has capacity.
- Telegram report is scheduled 24h after first startup of this version, then 24h after each acknowledged report delivery. It runs during Binance cooldown too, while the process is alive.
- Each ZIP contains report.json, signals_and_outcomes.json, up to 8 UTC observation files covering the last 7 days, and README.txt with model/units/limitations. Caption shows new daily signals and rolling 7d outcome statistics.
- Pending export and schedule are durable. Failed Telegram upload retries the SAME export at most every 30min. A lost acknowledgement can cause a duplicate delivery.
- Exports over 49 MiB are retained locally and logged, not sent; export failure cannot disable market scanning.
- AI analysis should compare candidate rules on chronological holdout data, distinguish resolved from censored/open signals, account for observation gaps and execution costs, and never use future outcomes as entry features. This code collects data; it does not automatically find or deploy an optimized algorithm.
