# Crash-only alerts
ema.py now runs only the foreground crash-bounce loop. Legacy strategy data collection, per-symbol price polling, account/trading routines, Google Sheets heartbeat and Telegram command handlers are not started. No orders are opened.

- Eligible: rank trading USDT-M perpetuals by current rolling 24h change, select the 20 biggest losers, then require change <= -12%. Coins on cooldown are not replaced with the 21st loser.
- Repeated signals require an entry at least 2% below the preceding recorded candidate. Identical/higher levels are suppressed across restarts and days, persisted in DATA_DIR/crash_bounce_state.json.
- Entry candidate: minimum of last 16 fully closed 15m lows and observed price, plus 0.3%, rounded up to tick. Gross TP: entry plus 1%, rounded up to tick.
- Levels are conditional, unbacktested reference levels; notifications do not verify a reversal or execution.
- Prices and rolling 24h changes arrive over one Futures WebSocket. exchangeInfo refreshes daily; missing candle history uses a capped REST bootstrap.
- Nominal scan start interval: 180 seconds; long scans delay it. REST cooldown does not stop WebSocket scans. Threshold crossings between scans may be missed.
- Startup starts WebSocket immediately. Persisted REST cooldown applies only to market metadata and candle bootstrap.
- REST requests are serial, at least one second apart, with a conservative local weight budget of 120/minute. IP usage >=50% of the exchangeInfo minute limit pauses requests until the next minute.
- 429: at least 120s exponential cooldown. 403/418: at least 900s. Honor longer Retry-After and ban expiry from the error body, with a 5s margin.
- Cooldown saved atomically in DATA_DIR/crash_bounce_rest_cooldown.json and respected on restart. Invalid persistence fails closed.
- Pending Telegram deliveries expire after 15 minutes; failed acknowledgements can cause a retry duplicate.
- Use persistent DATA_DIR and one running instance. Stop old replicas before restart; another process sharing the public IP can still exhaust Binance's IP limit. No ban-free guarantee.
- Deploy/restart the service to activate a changed ema.py. Repository update alone does not prove deployment.

Offline validation:
python -m py_compile ema.py
python -m unittest discover -p 'test_crash*.py' -v

44 tests cover top-20 ranking, threshold, price rounding, daily deduplication, Telegram routing/retries, persistence failures, startup isolation, cooldown persistence, 429/418, endpoint restrictions, spacing and weight limits. Tests mock Binance and Telegram. A separate read-only live WebSocket smoke check received an array of 153 ticker updates; this does not validate deployed service behavior.

## Virtual outcome tracking and research exports
- Every 180s the fresh WebSocket ticker cache also updates all pending/open crash signals, including coins outside the current top 20. This adds no Binance request.
- Existing records retain sampled_limit_touch_v1. New records use sampled_upward_cross_v2: a later observed price crosses upward from below planned entry to at/above entry. This is a hypothetical fill at entry; gaps, slippage and actual execution are unknown. A new low below the reference before entry invalidates that candidate. Neither model verifies a 15m reversal candle.
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

## WebSocket transport
- Routed combined endpoint: wss://fstream.binance.com/market/stream?streams=!ticker@arr.
- Ticker arrays contain changed symbols only: updates merge into the cache. USDT-M events are accepted; Coin-M events are excluded. Ranking covers eligible symbols with fresh observed quotes.
- After connecting, allow 20 seconds of warmup. Quotes older than 120 seconds are excluded. Disconnect clears prices, preventing signals from stale quotes. There is no REST price fallback.
- Automatic reconnect uses backoff and restores subscriptions; WebSocket ping/pong is handled by websocket-client.
- Subscribe to 15m candles for the current top 20. Only closed candles are stored, deduplicated by open time, retaining 64 per symbol in crash_bounce_ws_candles.json.
- A missing/invalid candle cache may bootstrap 64 candles by REST. Budget: at most 24 attempts per rolling hour, with 30 minutes between attempts for a symbol. Valid cached candles work during REST cooldown.
- Metadata is cached in crash_bounce_market_cache.json, refreshed every 24 hours, and accepted for up to 7 days. Failed metadata refresh retries hourly. New listings can await the next metadata refresh.
- With no usable metadata and REST blocked, signals wait for metadata. If initial candle bootstrap is blocked, a continuously subscribed symbol can need approximately four hours to accumulate 16 closed candles.
- REST weight guards are protective pauses, not evidence of an actual HTTP ban. Actual HTTP limit responses are logged separately. Neither stops WebSocket price processing.
- WebSocket transport and daily research export schedule are unchanged; the revision update below replaces signal deduplication and the tracking model for new records. Shared-IP or connection restrictions can still affect operation; WebSocket does not guarantee ban-free service.

## Lower-entry revisions
- First qualification still requires a top-20 USDT-M coin with rolling 24h loss >=12%.
- After an alert, observe that coin for 72 hours from its latest candidate, including outside the top 20, above the threshold, and after a TP. A new qualifying lower candidate starts another 72h window.
- Entry remains last four hours' closed 15m low, including the current observed lower price, plus 0.3%, rounded up to tick. TP is entry +1% gross, rounded up.
- Send a revised candidate only when its rounded entry <= preceding recorded entry *0.98. The previous reference is durable; midnight does not reset the price floor. No new message for smaller changes.
- Messages number each coin's candidates per Europe/Paris calendar day (#1, #2, ...). Pending retries retain their number and level; new candidates cannot overwrite an unacknowledged delivery.
- A candidate notification is immediate, not an instruction to buy while falling. New virtual results wait for an observed upward crossing; breaking the reference low before entry labels INVALIDATED_BEFORE_ENTRY.
- Each revision has its own immutable level, parent signal ID, daily number and outcome. Already OPEN or TP records keep their own result; older records are not reinterpreted.
- The ZIP report includes invalidated candidates and rolling summaries per tracking model. Aggregate cohorts can contain both models; use the per-model results for comparisons.
- No automated orders. Prices are evaluated every 180 seconds; intraperiod crossings and low breaks can be missed. A separate reversal-confirmation subsystem is described below.
- Lost Telegram acknowledgements may still cause delivery duplicates; durable level deduplication cannot guarantee exactly-once delivery across a network failure.

## Reversal confirmation messages and outcomes
- Enabled only for candidates created by this version. Existing candidates/outcomes are preserved and do not receive retrospective confirmations.
- First message is labeled IZLEME ADAYI. After a fully closed green 15m candle that started at/after the candidate, observe two post-close price samples: one <= candle high, then one > candle high. The chosen green candle must have closed within the preceding 30 minutes.
- Current sampled price and available closed candle lows since the candidate must preserve its reference low. A detected lower low permanently invalidates confirmation for that candidate. A newer candidate supersedes an older unconfirmed candidate.
- Confirm only candidates whose initial Telegram delivery was acknowledged. Confirmation remains eligible up to 72h after candidate creation.
- Second message is labeled DONUS ONAYI, includes the parent daily number, old candidate entry, observed confirmation price, tick-rounded confirmation reference and gross +1% TP.
- Confirmation decisions are durably recorded before delivery. Retry the same message for at most 15 minutes. Suppress pending delivery when fresh price is unavailable, below reference low, or already at/above its TP. Lost acknowledgements can still cause delivery duplicates.
- Confirmation results use sampled_reversal_confirmation_v1, independently of candidate results: virtual entry reference is the tick-rounded confirmation price, TP requires a later sample, and timeout is 72h after confirmation. Track sampled favorable/adverse moves and coverage gaps.
- Reports retain original candidate outcomes and add confirmed_7d_cohort, confirmation_states, nested confirmation records, and confirmation events/results in JSONL. Telegram caption includes confirmation count and post-confirmation TP/open counts.
- This rule verifies a particular sampled reversal pattern; it does not guarantee price continuation, fills or profitability. In-progress lows and intraperiod breaks can be missed; 3-minute sampling can delay or miss confirmation. Fees, slippage, funding and leverage remain unmodeled.
- 44 offline tests cover candle close timing, upward high crossing, low invalidation, supersession, duplicate suppression, restart/retry, disk failure and independent confirmation outcomes alongside the earlier transport tests. No live messages or orders are sent by tests.
