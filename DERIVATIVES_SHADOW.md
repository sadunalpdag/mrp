# Crash Bounce V2 derivatives shadow study

No V2 eligibility conditions, original strength score, entry price, TP or virtual transitions are changed. `AUTO_TRADING_ENABLED = 0` ships with this study. Existing order/TP maintenance remains available; this does not cancel previously placed orders. Regime features never enable trading or block a signal. No automatic promotion of a regime to a trading filter is implemented.

## Features and observation cutoff

Only successfully delivered new V2 candidates enqueue work. The initial message contains a PENDING/UNKNOWN block without delaying the signal. A separate signal-ID-linked Telegram update follows completion. Jobs use the candidate's original timestamp, not the collection time. All 1h/4h CVD windows use consecutive, closed 15m candles ending at the preceding UTC quarter-hour. CVD = sum(2 × taker-buy quote volume − total quote volume), in USDT. This is the net taker-volume change over the window, not an exchange-wide perpetual cumulative total. Binance spot and Binance USDT-M perp are kept separate. Same-symbol spot pairs only; an absent spot listing is UNKNOWN. Bybit rows and rows lacking taker volume cannot be used as Binance CVD.

OI change uses `sumOpenInterest` (base units), not the USD value, at exact 1h/4h endpoints. Funding is explicitly the last SETTLED rate available at or before signal time (up to 24h old), not a later current/predicted rate; `funding_time_ms` is saved. `as_of`, `window_end_ms`, `fetched_at`, data sources, window coverage and errors are saved. Any missing required regime input yields UNKNOWN with zero adjustment; missing measurements remain null, not fabricated zeroes. A genuine zero CVD is valid.

Rule precedence is exactly the supplied order: DELEVERAGING, OI_RISING_SELL_PRESSURE, POSSIBLE_ABSORPTION, SPOT_CONFIRMED_RECOVERY, NEUTRAL. In particular rising OI + negative perp delta overrides positive spot absorption. Shadow score = clipped(base score + adjustment), with +8 / -15 / +3 / +10 / 0 respectively. The base score is never overwritten.

## Load and persistence

One background job per minute, only for delivered signals. Perp candles reuse existing Binance cache/WS when complete; otherwise request 16 closed candles. Spot requests 16 closed candles, OI requests 17 observations, funding requests at most one day's history. At most four shared, paced public REST requests per job. Every request obeys existing public/private cooldown checks. No all-market OI polling or five-day spot download is introduced. REST failures yield partial/unknown snapshots and do not suppress V2 signals. This reduces load but does not guarantee the shared IP never hits limits.

Snapshots are stored in derivatives/snapshots.jsonl, observation JSONL, and each signal's derivatives_shadow object. The daily ZIP includes signals_and_outcomes.json, derivatives_shadow.csv and derivatives_summary.json. Pending persisted signal snapshots can be requeued after restart for up to one day; already completed snapshots are not refreshed with future data. Without persistent storage, a deploy can lose research state/settings. CSV empty cells represent unknown values.

## Evaluation

Review at least 30–50 prospective signals (summary marks 50). Groups report TP, entered/open/timeout, observed new-low proportions with observed denominators, observed MAE and coverage gaps. Open/waiting cases are not losses. MAE is a negative percentage relative to planned entry, observed only after virtual entry; disconnected trade streams can miss lows. These are virtual outcome statistics, not net trading profitability. Fees, funding payments, slippage and liquidation must be modeled separately before a profitability conclusion. Any future proposal involving a 20% worse failure rate must specify relative percent versus percentage points, matched follow-up windows and sufficient resolved observations; no such rule goes live automatically.

## Reports and settings

Daily reports run at 21:00 Europe/Paris using a DST-aware calendar slot, rather than waiting 24h of process lifetime. `/rapor` requests an immediate report through the configured Telegram chat even with automatic entry off. Text summary and ZIP are sent separately; failed ZIP sends retry after 30m. Missing pending ZIP files regenerate; oversized ZIPs log a text-only delivery. Telegram document captions are bounded. No direct Render/Telegram delivery was tested during implementation.

`/sembol 50` sets concurrent unique open/pending-entry symbols (range 1–50). TP closers remain excluded. `/limit` still controls daily entry attempts (1–50); `/tutar` remains total notional (1–500 USDT). These commands do not enable automatic entries. Saved values persist only while the state file survives.

References:
- https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data
- https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market
