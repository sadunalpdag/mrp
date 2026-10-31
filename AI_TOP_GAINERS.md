# Independent AI Top Gainers + hourly BTC (research only)

This module is isolated from Crash Bounce V2 filters, entry handling, TP orders and auto-trading. It does **not** place orders or change the bot's trading switches.

## Activation (same Render worker)

```ini
TOP_GAINERS_ENABLED=1
OPENAI_API_KEY=<configure in Render Environment, never commit>
# Optional: TOP_GAINERS_MODEL=gpt-5.4-mini-2026-03-17
```

The existing `ema.py` starts `ai_top_gainers.TopGainers` as a daemon research worker. It shares the existing Binance Futures ticker WebSocket cache and the centralized paced Binance public REST client. No new persistent worker is required; do not start a second instance of the bot.

## Schedule and inputs

- Every 4 hours at UTC 00/04/08/12/16/20 boundaries, rank USDT-M **active perpetual** instruments by changes between the current WS ticker snapshot and the snapshot taken one four-hour slot earlier. First slot after launch is a warmup with no ranking. Missing or stale symbols are not imputed. Therefore this is the *fresh, comparable WS subset*, not a guaranteed exchange-wide complete ranking.
- On the selected ten instruments, request up to 60 1h Futures klines per symbol via the shared rate limiter; analyze only closed bars.
- A single OpenAI Structured Output request evaluates the ten coins and may propose **LONG paper entry levels**. Prices must be between 90% and 100% of the market snapshot price. Missing/invalid entries are omitted. TP is calculated in Python as entry x 1.01.
- Every hour, an independent BTCUSDT AI forecast uses 1h closed candles and a current WS price, predicts UP/DOWN/SIDEWAYS over the next hour, and is compared against a later observed WS quote. A +/-0.2% band defines the neutral category.
- Every day from 21:00 Europe/Paris, send a one-off research summary. In a long-running process it will usually run in the first worker cycle after 21:00. Schedules are DST-safe for local daily summaries.

## Tracking and limitations

- Proposals are enrolled only after Telegram accepts the report, then sampled at ~1-minute intervals. `WAIT_ENTRY` -> `OPEN` requires a **later** observed price <= proposed entry. `OPEN` -> `TP_1PCT` requires a **subsequent** observed price >= entry x 1.01. Entries waiting 24h become `NO_ENTRY_24H`; positions not seeing TP within 24h from entry become `TIMEOUT_AFTER_ENTRY_24H`.
- These are **observed snapshot** paths, not tick-perfect fills. A 1-minute quote can miss fleeting touches or both sides of a candle. No fee, funding, spread, latency or slippage is modeled. Neither TP rate nor subjective AI score is a calibrated success probability.
- If the OpenAI key is absent or model results are invalid, the bot can send an unscored market report, but it generates no new entry proposal.
- BTC prediction accuracy is computed only when the next-hour quote is available in the allowed observation window; otherwise `NO_OBSERVATION` and no correct/incorrect score.
- Daily report covers the previous rolling 24 hours. TP success is reported **only among resolved paper entries** and unresolved observations are kept distinct.
- The full ledger is atomically written to `/data/ai_top_gainers/top_gainers_state.json` (assuming `DATA_DIR=/data`), and included in the normal daily Crash Bounce ZIP under `ai_market/top_gainers_state.json`. Back up ZIPs outside Render. No secret is saved in this ledger.
- To stop only this research service: `TOP_GAINERS_ENABLED=0`.

## Verification

Run `python -m unittest test_ai_top_gainers -v` offline. CI on this branch also compiles both Python entry files and runs the existing Crash Bounce tests. Production API calls, Telegram delivery and deployment must still be verified in Render after merge.

## Four analyst research (new)

The four-hour Top Gainers scan evaluates up to ten symbols in batched OpenAI requests: technical indicators including MACD/ADX and support/resistance; OI/funding and perp taker-volume imbalance proxy (not real trade-level CVD); a single optional OpenAI web search for timestamped cited news; and risk including a recent BTC hourly forecast where available. A separate final evaluator proposes scored paper LONG entries. This is not the TradingAgents third-party graph/framework.

The extra research uses a maximum of five OpenAI Responses API calls per four-hour report (three analyst calls, one optional web search, one final decision), shared paced public Binance REST requests, and **never** creates real orders. News is not counted as verified unless a URL and a plausibly recent model-reported date exist; publication dates are not independently verified. Model scores are opinions, not probabilities.

Render options: `TOP_GAINERS_NEWS_ENABLED=1` to search once per four-hour cycle (set 0 to disable), `TOP_GAINERS_SPOT_DELTA_ENABLED=0` to disable extra spot kline calls (set 1 to enable). See `ai_market_agents.py`. New results and estimated AI costs are included in the existing Top Gainers JSON and daily 21:00 Europe/Paris report. API search costs, model tokens and REST consumption are additional to the old version; the daily cost calculation is approximate, not an enforced budget cap.
