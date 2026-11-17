# CoinBeacon AI Learning – isolated 8-hour research mode

This is a new **paper-only**, opt-in subsystem in the existing mrp worker. It never
places Binance orders and does not modify Crash Bounce or CB2's real-trading
executor. Do not activate real order placement based on Gemini output without
a separate reviewed implementation, exchange tests and explicit authorization.

## Data path

1. The AI intake uses the same CoinBeacon /public/setups data as CB2 but applies
   NO SQS, bias, archetype, ADX, ATR or MACD filtering. A candidate still needs
   a recent 15m timestamp and 480 consecutive CLOSED 15m CoinBeacon OHLC bars.
   No additional CoinBeacon /public/setups request is made; upstream API
   selection limits and candle-credit protection are still in effect.
2. Original input (the last 480 closed 15m OHLC bars, setup metadata and indicators)
   is durably inserted in a separate SQLite database and Gemini is called
   asynchronously. The first prediction is immutable after saving. Gemini may
   suggest a hypothetical LONG/SHORT/WAIT with LIMIT/STOP price, or WAIT.
3. A **single** Binance Futures public WebSocket carries closed 15m candle
   updates for all active symbols, with dynamic subscriptions. Observation starts
   with the first fully formed candle after the initial Gemini response, never
   on historical bars or an unobservable partial entry candle.
4. Eight hours after the prediction, the observer computes paper order trigger,
   +1% TP observation, favorable/adverse movement and data coverage. Entry and
   TP in the *same* 15m bar are labelled AMBIGUOUS, not counted as a confirmed
   TP; OHLC cannot prove intrabar event ordering or actual fills.
5. Missing 15m candles can be filled with **one public Binance REST GET** at
   expiry. WebSocket is primary; REST is used only if gaps exist. The actual
   8h candles, immutable prediction and measured outcome go to Gemini for
   a distinct final review. This does not train Gemini's weights.
6. Every accepted candidate and every state update is stored in persistent SQLite.
   A **single JSON per Paris-local reporting day (01:00–01:00)** is uploaded
   to the existing private Cloudflare R2 bucket at:
   coinbeacon-ai-learning/daily/YYYY/MM/gemini_learning_YYYY-MM-DD.json
   The file is refreshed when a late 8h result, AI review, or error arrives;
   a revision journal retries cloud failures without pretending a failed upload
   succeeded. Older per-case JSON objects remain historical, but no new ones
   are created by this version.

### Environment (Render)

- CB_AI_LEARNING_ENABLED=1  – opt in, default 0
- GEMINI_API_KEY=... – secret, must not be committed to GitHub
- CB_AI_GEMINI_MODEL=gemini-2.5-flash-lite – adjustable model ID
- CB_AI_DAILY_GEMINI_CALL_CAP=50 – daily request safety cap, counts initial+final
- CB_AI_MAX_ACTIVE=20 – bound concurrent cases
- CB_AI_REST_GAP_FILL_ENABLED=1 – one missing-data REST reconciliation at expiry
- DATA_DIR – existing persistent disk location
- Existing CRASH_REPORT_R2_* account, bucket and credentials – reused, private

Check the Gemini API free-tier availability/rate caps in the Google AI Studio
project; a model's 1M context window is **not** a daily free token quota.
This feature stores AI output, market data and timing; no API keys in archives.

### Operational considerations

- Feature is **OFF unless enabled**, including after merging; opening this PR
  does not deploy or activate trading. Production validation of the live WS,
  Gemini response format, available API quota, and R2 storage remains required.
- The legacy CB2 momentum observer must also be enabled
  (CB2_LIVE_MOMENTUM_ENABLED=1), since it supplies the already-fetched candles.
- Candidate selection may under-sample rejected/stale/low-SQS setups. Results
  must be analysed by selection criteria, not generalized to every coin.
- Coverage gaps produce explicit incomplete labels. R2 requires private
  credentials; without configuration, uploads stay queued.
- Gemini-generated entry plans are *hypotheses*; do not interpret candle touches
  as real Binance fills. Fees, slippage, latency and liquidation are excluded.
- Reusing archived cases for future prompt retrieval or model retraining is a
  **separate, testable feature**, not automatic training in this PR.
- Use case's initial/final prompt fields and candle time stamps for leakage-free
  offline evaluation and comparisons against a rule-only baseline.

Run tests offline: python -m unittest -v test_coinbeacon_ai_learning.py

## Bot Status Dashboard (J:O)

The existing EMA Filter workbook **Bot Status** tab retains A:B reports and
Crash D:I controls; AI Learning uses **J:O only**, without resetting existing
Crash approvals. J=label, K=current, L=new value, M=description,
N=result, O=approval. Set L and enter **UYGULA** in O; the sheet publisher
checks roughly every five minutes. It persists changes in
`DATA_DIR/coinbeacon_ai_settings.json` and starts the independent paper
observer when enabled and prerequisites are present. Never put API credentials
in the sheet; GEMINI_API_KEY stays in Render Environment.

| Dashboard setting | Default | Options |
| --- | --- | --- |
| AI Learning | OFF | AÇIK / KAPALI |
| Gemini calls/day | 100 | 1-200 |
| Max active cases | 30 | 1-100 |
| Tracking horizon | 8 hours | 1-24 hours, applied to new cases |
| Review cadence | 15 minutes | 15/30/60; every 15m candle still archived |
| AI stages | Initial + final | Initial + final or initial only |

OFF prevents new AI cases. Existing 8-hour WS observations/reviews are
allowed to finish when the worker is already running, so their labels are not
silently lost. The on/off setting does not affect any real Binance orders.
The 15-minute data collection granularity remains fixed for comparability.

## Daily JSON for later pattern/model research

The daily archive contains `schema`, Paris window bounds and `finalized`, a
`summary` (signal/coin counts, AI decisions, observed TP and failure labels,
coverage warnings, Gemini request attempts, usage metadata and learning tags),
`gemini_dispatches` and full `cases`. Each case retains its **480 pre-signal
closed 15m candles**, original CoinBeacon time, price, pattern, initial raw Gemini
response and immutable planned entry, post-prediction 8h candles, TP/MAE/MFE
and ambiguity/missing-bar flags, plus final Gemini review and errors. These
are training/evaluation *data*, not an automatically trained model. Do not
count `WAIT`, ambiguous candles, incomplete coverage or missing entry triggers
as successful TP trades. A late result updates the SAME day's object rather
than opening an additional JSON file.
