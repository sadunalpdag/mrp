# CoinBeacon AI Learning – isolated 8-hour research mode

This is a new **paper-only**, opt-in subsystem in the existing mrp worker. It never
places Binance orders and does not modify Crash Bounce or CB2's real-trading
executor. Do not activate real order placement based on Gemini output without
a separate reviewed implementation, exchange tests and explicit authorization.

## Data path

1. Existing /public/setups and CoinBeacon /public/ohlcv/binance-futures/SYMBOL/15m
   are reused by cb2_live_momentum; **no extra CoinBeacon request** is made by
   AI Learning. Candidates are SQS>=70, supported breakout archetypes, with a
   valid, fresh set of CLOSED 15m candles. They are included regardless of CB2
   momentum's separate ADX/MACD acceptance criteria.
2. Original input (at most last 96 closed bars, setup metadata and indicators)
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
6. The case is saved to the local isolated SQLite journal and uploaded to the
   existing private Cloudflare R2 bucket under:
   coinbeacon-ai-learning/YYYY/MM/DD/<signal-id>.json
   Failed upload remains queued locally for retries.

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
