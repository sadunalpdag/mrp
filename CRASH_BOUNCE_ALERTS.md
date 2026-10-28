# Reliability and tracking update — 27 October 2026

- Entry remains 4h reference low +0.3%, tick-rounded, with the existing upward-cross/dip-protection rule. The proposed lower entry was withdrawn by the user.
- New candidates, metadata and historical downloads accept only ASCII USDT symbols (A–Z, 0–9, underscore; max 40 characters). Unsupported persisted history jobs are removed once; old records are preserved. Already-open positions retain live trade monitoring.
- An independent watchdog checks every 5 seconds. A socket without an accepted fresh ticker for over 120 seconds is closed; the existing reconnect loop restores subscriptions with 5–60 second exponential backoff plus jitter. Stale prices are cleared. Logs show connection attempts, retry delay, ticker age and errors. This does not replace missing observations or guarantee uninterrupted connectivity.
- Active entry waits now allow 150 hours from candidate creation; OPEN records allow another 150 hours from actual virtual entry. Both timer and WS event paths use the same horizon. Previously terminal TIMEOUT/NO_ENTRY_OBSERVED records remain terminal.
- ZIP records/observation files cover 14 days to include the combined 300-hour window. Rolling headline statistics still describe the last 7 days; existing separate risk-scenario experiments retain their 5-day horizon.
- No real orders or credentials are used by these changes. Runtime deployment must be checked on Render after the push.

---

# Current deployment configuration — 17 October 2026

**Only Crash Bounce V2** is started by `ema.py`'s `main()`.
Fibonacci's worker is **disabled** (the historical module remains in the repository).

- Screening remains every 180 seconds: USDT-M top 20 losers, 24h drop at least 12%; a new candidate needs an entry at least 2% below the prior reference.
- At candidate creation, V2 checks **480 closed 15m candles**, RSI14 >= 20 and 4h low recovery >= 1%. If the cache is missing, it requests those candles with the existing paced Binance REST guard; missing data fail closed. Eligible candidates are sent to Telegram **before** a simulated entry, within the 90-second freshness window.
- Telegram attempts are durably recorded with API acceptance timestamp and `message_id`; failures and ambiguous timeouts are logged. `SENT` means API-accepted, not guaranteed device delivery. Unknown timeout delivery is not blindly retried.
- Only fresh, ordered Binance Futures `aggTrade` WebSocket events classify **new** candidate virtual entries, TP and pre-entry low breaks. A crossing that goes straight above TP without a separately observed entry is `MISSED_SIGNAL` / `ENTRY_ABOVE_TP`, **not a TP**. Events are buffered during data/bootstrap operations; buffer loss and timing gaps are marked.
- The daily ZIP and Telegram caption report missed signals separately. Older sampled models and their outcomes remain archived for research, without retroactive rewrites.
- No automatic orders are sent. Deploy/restart Render to pick up GitHub updates; a GitHub commit by itself does not verify that a worker is running.

---

## Continuous strength score — research-only

Crash Bounce V2 eligibility **does not change**: top-20 USDT-M losers, at least 12% 24h loss, 480 closed 15m candles, RSI14 >= 20 and 4h recovery >= 1%. A qualifying candidate is still sent to Telegram immediately. No Fibonacci, live order placement or score-based filtering.

The optional `continuous_7factor_v1` score takes values from 0 to 100 and uses **fractional** component points, not binary awards:

| Component | Max contribution | Proportional measurement |
|---|---:|---|
| Dip stability | 25 | Count of newly broken supports over last 4h and bars since last low |
| RSI momentum | 20 | RSI14 change over last hour, with overbought penalty |
| Volume confirmation | 20 | Last-hour volume / prior 12h median, plus volume on green 15m candles (proxy only, *not* actual buy/sell flow) |
| EMA25 improvement | 15 | EMA25 slope change and latest EMA25 slope |
| ATR / TP capacity | 10 | ATR14 as percent of current price; excessive volatility penalized |
| Selling slowdown | 5 | Last-hour return versus previous 4-hour decline rate |
| BTC context | 5 | BTCUSDT 24-hour futures change at candidate time; absent if not fresh |

Each component receives an interpolated share between 0 and its maximum, such as **Dip 13.7/25** or **RSI 8.9/20**. Available components are normalized to 100 and the **data coverage percentage** is always shown. For example, unavailable BTC means 95% coverage, not an invented BTC value. No score is computed on incomplete five-day candle history. The output is a research ranking, **not a calibrated probability of TP**.

Candidate Telegram alerts include the overall score, each component and coverage. `strength_score` (including measurements and score version) is kept on the signal and included in daily ZIP exports. Daily captions include average score, count scored and count at least 75 (a **descriptive reporting bin**, not a trade threshold).

Next step before using score to decide entries systematically: compare score bins against subsequent TP, timeout, pre-entry dip-break and ENTRY_ABOVE_TP labels with chronology held out, slippage/fees accounted for, and missing WebSocket observations separated. Scoring weights are preliminary, not statistically optimized.

---

## Historical design notes (superseded where they conflict with current V2)

# CRASH BOUNCE V2 ONLY (2026-10-15)

Current version: `five_day_entry_v2_only`. Telegram candidate, legacy RSI/EMA scoring and old confirmation notifications are suppressed. Initial candidates remain internal tracking records. A Telegram alert is sent **only after a sampled upward crossing** of the existing candidate entry, and only when a contiguous 480 fully closed 15-minute candle history is available and last closed candle RSI14 >=20 and (last closed close / last 4h candle low - 1)*100 >=1%. TP reference remains +1% (gross). An entry observation already at/above the TP is suppressed. Missing/late history fails closed and is logged as `V2_FILTERED`; there is no retroactive V2 alert or REST call at entry. The existing 24h-drop/top-20 selection, 4h-low +0.3% entry, 2% lower revision, market WebSocket, 3-minute scan, 5-day research cache, and 24h data ZIP report remain. All alerts remain virtual; no automatic orders.

Historical records retain their prior results and are not reinterpreted as V2. Note that existing offline tests which explicitly assert legacy 0/50/100 candidate messages must be updated for the changed notification contract. A GitHub push alone does not confirm Render deployment or runtime health. V2 is exploratory: the previously observed 95% two-hour TP figure is in-sample, not an independently validated expectation.

---

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
- Entry wait expires after 150h: NO_ENTRY_OBSERVED. Once entered, another 150h is allowed for TP: otherwise TIMEOUT.
- Track observed TP hours, sampled favorable/adverse excursion, last observed return and gaps >540 seconds. TP rate denominator is TP + TIMEOUT; OPEN/WAIT_ENTRY/no-entry/legacy records are separately shown.
- Prices between samples are unknown; TIMEOUT means no sampled TP, not proof TP never occurred. Fees, funding, slippage and leverage are unmodeled.
- Pre-existing records are LEGACY_UNKNOWN. Historical results are not fabricated.
- New signal features include up to 64 fully closed 15m OHLCV candles, Wilder RSI14, EMA7/25 and slope, recent volume ratio, 1h return, 24h ticker fields and loser rank.
- UTC daily append-only JSONL stores top-20 selection, threshold/cooldown decisions, BTC/ETH context, new signals, price samples and results. Research price samples continue up to 6 days after each signal, including after TP, for alternative TP/SL comparisons.
- State retains 180 days of signals; JSONL observations and daily ZIP exports are retained without automatic deletion in DATA_DIR/crash_research. Ensure DATA_DIR is persistent and has capacity.
- Telegram report is scheduled 24h after first startup of this version, then 24h after each acknowledged report delivery. It runs during Binance cooldown too, while the process is alive.
- Each ZIP contains report.json, signals_and_outcomes.json, up to 15 UTC observation files covering the last 14 days, and README.txt with model/units/limitations. Caption shows new daily signals and rolling 7d outcome statistics.
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
- After an alert, observe that coin for 150 hours from its latest candidate, including outside the top 20, above the threshold, and after a TP. A new qualifying lower candidate starts another 150h window.
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


## Two-hour health messages and optional night execution

Telegram health messages run independently of the scan thread, first after about
three minutes on a new installation, then every two hours after an acknowledged
send. The saved schedule survives restarts. Failed sends retry after five minutes;
ambiguous Telegram timeouts wait two hours to avoid flooding. Reports distinguish
fresh WebSocket data, completed scans, missing candles, scan failures and the last
acknowledged signal delivery. A healthy scan without eligible candidates is normal.
A frozen scan or stale stream is never reported as healthy. Render also receives
`[CRASH SCAN COMPLETE]` counters and `[CRASH HEARTBEAT]` delivery results.
A stopped Render process cannot send a heartbeat.

### Python file switches (current)

The following settings near the top of `ema.py` are the only production activation
controls. Both are shipped **0/off**. Changes take effect after restart/redeploy:

```python
AUTO_TRADING_ENABLED = 0       # 1: allow new night entries
MANUAL_TP_REPAIR_ENABLED = 0   # 1: inspect/repair missing TP on manual longs
AUTO_MAX_DAILY_ENTRIES = 5
```

`CRASH_NIGHT_TRADING` and Telegram open/close commands no longer activate anything.
The Telegram poller accepts only counter/status commands in the configured chat. Telegram also sends
signals, health reports and execution/TP errors; `TELEGRAM_OWNER_ID` is not needed
for sending those messages. Setting the entry flag to 0 stops new entries but
preserves supervision of previously tracked automatic TP orders. Persisted fault
locks remain effective even with the entry flag set to 1.

The broker reads **exactly** `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` from the
process environment. The screenshot contains those names, not their values, and
does not establish key validity, IP permissions or Futures trading permissions.
`CRASH_NIGHT_STATE_DIR` is optional; by default state is stored in `DATA_DIR/night_execution`. No Render persistent disk is required. Missing state starts the current-day counter at zero by user request. Manual TP repair queries live positions and both normal/algo orders after startup. Ephemeral storage may disappear on deploy, resetting the local counter. Existing exchange orders are not canceled.
Keep one service instance and never delete state to clear a limit/fault.

Execution behavior:

- Automatic entries run **24 hours**, without a night-only restriction. Daily counters still use the Paris calendar day.
- Existing Crash Bounce V2 rules remain unchanged. A newly acknowledged Telegram candidate queues a GTC limit entry (maximum queue age 120 seconds). No upward crossing is required for real orders; virtual research tracking keeps its existing crossing model. No historical catch-up.
- **10x**, at most **50 USDT notional** after rounding (approximately **5 USDT initial
  margin**, excluding fees). Margin mode is unchanged. Exchange filters may prevent
  small orders; quantities are never increased to satisfy minimums.
- At most **5 reserved entry attempts per Paris calendar day**. No-fill, rejected
  or uncertain attempts consume a slot conservatively. The quota resets at midnight,
  not at the start of the overnight window.
- At most **five occupied symbols**, including account positions and pending orders.
  Existing positions/orders on the target symbol prevent entry. Concurrent manual
  activity or another bot can change the account after the check; use a dedicated
  account/subaccount for this executor.
- One-way and Hedge long positions are supported; account mode is never changed.
- TP and price/quantity/minimum-notional filters are checked **before** entry.
  BUY LIMIT GTC at the candidate price waits when the market is higher and can fill at the limit or better when the market is lower;
  an entirely unfilled order stays open. After a partial fill the remainder is canceled before TP submission.
- After a fill, SELL LIMIT GTC (`reduceOnly` in one-way; `positionSide=LONG` in hedge) is submitted for the **filled quantity**,
  then queried to confirm its properties and status. Entry and TP are separate
  requests: atomic TP protection and TP execution are not guaranteed. No SL is added.
- TP failure/cancellation or uncertain entry results **halt additional entries**
  and queue a Telegram warning. Existing positions are not force-closed. Protection
  checks continue outside entry hours. Restarting does not clear the halt; account
  and state require manual review.
- Durable client IDs/phases are written before POST. Unknown responses and restart
  recovery query the saved ID rather than blindly resubmitting. A crash between
  persisting TP intent and submitting it can leave an unprotected position, followed
  by a halt requiring manual intervention. An alert does not mean it was closed.
- Unacknowledged execution notifications block additional entries and retry.
  Timeout retries can duplicate notifications, but do not intentionally repeat an
  exchange order. Signal-only and health messages continue.

Tests use fake exchange/Telegram adapters without real orders. Live fills, persistent
storage, account permissions and Telegram delivery require operator verification.


### Manual-position missing-TP repair

With `MANUAL_TP_REPAIR_ENABLED=1`, the executor inspects positive USDT perpetual
positions approximately every 30 seconds, including outside the night entry window.
One-way and Hedge long positions are supported; short positions are not modified.
This covers account longs not already owned by the automatic-entry tracker, rather
than inferring manual provenance from Binance (the exchange does not label a
position "manual"). It may include longs opened by other software on the account.

Both normal open orders and `/fapi/v1/openAlgoOrders` must be queried successfully.
An existing full reduce-only profitable LIMIT or conditional TAKE_PROFIT order is
preserved, including `closePosition` TP. Partial TP coverage, other closing orders,
SL/trailing-order conflicts and unknown order states produce warnings rather than
adding a conflicting order. No existing order is canceled or resized.

If there is no closing order, use actual average entry ×1.01, tick-size rounded up,
for the full actual long quantity. Recheck position and both order lists before
POST. If best bid already reached the target, warn without submitting a sale.
Repair orders use reduce-only LIMIT **GTX (maker-only)** to avoid deliberately
selling immediately if price jumps during the check. A maker rejection is reported
with code `-5022`. TP placement still cannot be atomic with the original manual entry.

Persist client order ID before sending. An uncertain response is query-only across
restarts, never a blind resubmission. For a known whitelist of definite rejection
codes, allow one later retry after at least 60 seconds and fresh checks (two attempts
maximum per observed position version). Authentication/filter/reduce-only failures
report sanitized Binance codes and Turkish explanations. Unknown errors do not
include raw signed URLs, keys or response messages. Identical warnings are throttled
to 30 minutes, and execution notifications remain in the persistent delivery queue.

A filled/canceled/expired prior TP with a remaining position is warned about rather
than automatically recreated. Changes to a previously attempted position also
require review. Same-price/same-size close-and-reopen between polls cannot reliably
be distinguished; the guard can conservatively hold instead of placing a new TP.
Offline tests cover these paths; no live orders or API credential verification were
performed as part of this change.

### Hedge-mode compatibility
Manual long TP repair and automatic long entries support BOTH (one-way) and LONG (hedge). The account mode is never changed. Hedge closing orders use SELL + positionSide=LONG without reduceOnly; one-way orders retain reduceOnly. Existing normal and conditional TP orders are matched to the same symbol and position side. SHORT positions/orders are not managed. Coin-specific failures identify the symbol and LONG side. If fetching the account position list fails, a separate account-query warning explicitly says TP absence was not established.

Position sizing remains 50 USDT notional at 10x (approximately 5 USDT margin); existing positions or pending orders on the candidate symbol block duplicate automatic entry.

### GTC candidate entries
A candidate acknowledged by Telegram immediately queues a BUY LIMIT GTC at its entry level. Quantity is calculated with a 50 USDT notional ceiling at that limit; a better fill may use less than 50 USDT. The predetermined candidate TP price is retained. New unfilled orders stay on Binance until filled or canceled, including across bot restarts; disabling new entries does not cancel them. Pending symbols count toward the five-symbol cap. Live orders do not follow virtual low-break invalidation.

The worker polls tracked orders every five seconds (plus API latency). NEW with no fill is normal pending state, not an error. On partial fill the bot cancels the remainder, queries final executed quantity, then places TP for that quantity. An unconfirmed cancellation stops new entries and alerts; TP protection cannot be guaranteed during an API outage. Unknown submit responses are queried, not blindly resubmitted. Missing-history daily hold has been removed; a fresh state starts at zero.

### Daily counter and Telegram correction
Missing local execution history now starts at 0, as explicitly requested. The old unverified-day hold is removed. A missing/ephemeral state store can therefore reset the used-entry counter on deploy; existing Binance positions and orders are still checked separately for the five-symbol cap. The Paris calendar day determines the counter: after midnight, previous-day counts/adjustments do not apply. No positions or pending orders are canceled by resetting the counter.

The configured-chat Telegram poller is enabled solely for `/sayac` (show status), `/sayac N` (set today's used-entry count, 0 through AUTO_MAX_DAILY_ENTRIES), and `/otomatik_durum`. It does not accept trading enable/disable commands; those remain controlled by ema.py. New attempts after a correction increment the corrected value. TELEGRAM_OWNER_ID is not required. Any human user in configured CHAT_ID can correct the counter, including all members of a configured group. Other chats and forwarded/edited/stale messages are rejected. This changes the used count, not the daily maximum.

### Occupied slot counting
Closing orders (reduceOnly, closePosition, SELL/LONG or BUY/SHORT) do not reserve additional entry slots. Count the unique union of active position symbols and non-closing pending-order symbols. Two open positions plus three distinct pending entry symbols still occupy five slots, even if the positions also have TP orders. Unknown order direction is conservatively treated as a possible entry. Existing orders on the candidate symbol still block entry to avoid collisions with residual closing orders. MAX_5_WITH_PENDING_ORDERS now reports active, additional pending-entry, total and excluded closing-order counts.
