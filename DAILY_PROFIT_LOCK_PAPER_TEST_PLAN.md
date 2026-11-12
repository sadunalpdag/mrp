# Daily Profit Lock — Paper Test Plan (approved 2026-11-12)

Status (2026-11-12): **paper-only monitor implemented for master deployment**. It reads signed Binance account and income history, stores a baseline from its first successful observation, and saves virtual snapshots. It has NO Binance POST/DELETE path, sends no closing orders, and does not block real entries. Live order-close feature is NOT implemented.

## User-approved rules
- Clock: Europe/Paris midnight-to-midnight, including DST. Default target = 5%, adjustable by authorized Telegram command.
- Maintain highest **validated, net-of-deposits/withdrawals USDT-M account equity** ever observed after activation. Equity means wallet balance plus unrealized P&L, NOT leveraged position notional. No daily drawdown can decrease this historical peak.
- At Paris 00:00, freeze next day's threshold as historical peak before that day's observation × (1 + target %). The target does **not** rise within the current day if a newer peak occurs; newer peaks apply starting the following day.
- Example: historical maximum = 1,000 USDT; following day reaches only 950. Threshold remains 1,050, not 997.50. If a session reaches 1,040 but not 1,050, the following day's threshold becomes 1,092.
- Once validated adjusted account equity reaches the day's threshold: save timestamp, account equity, each open position's quantity and observed market price **once**, and freeze the snapshot for that close cycle.
- Immediately disable ALL real buying and queued retries. Cancel and reconcile pending opening buys before reducing positions; fills may arrive during cancellation, so recheck positions.
- For each open LONG, reconcile bot-owned existing TP/algo orders and prepare a position-reducing SELL LIMIT GTC at the captured price (one-way reduceOnly; hedge proper positionSide). For SHORT, use symmetrical close where supported. Do **not** automatically lower the frozen exit level when the market falls. A limit order might fill immediately, partially, or never.
- Treat trigger reached separately from **realized** earnings. Fees, funding, execution slippage, unfilled orders and liquidation exposure mean 5% is NEVER guaranteed.
- Verify each individual fill and remaining position from fresh Binance exchange state. In uncertain order states, stop new buying, alert on Telegram, and recover idempotently; never double-close or reverse positions.
- Only cancel **verified owned** bot orders; unknown/manually created orders need separate safe treatment, not indiscriminate deletion.
- Keep real entry lock until the next Paris midnight **and** until all relevant positions are verified flat and orders confirmed cleared. If any position survives midnight, keep lock; after midnight release only once verified clear. HBREAK and CoinBeacon API2 paper monitoring continue.
- Changing settings or disabling future triggers cannot silently dismiss an already-triggered safety lock.
- Planned Telegram controls (not implemented): `/gunluk_kar_ac`, `/gunluk_kar_kapat`, `/gunluk_kar 5`, `/gunluk_kar_durum`. Restrict to the authorized account; show peak, frozen goal, trigger date, virtual fill status, pending orders, and realized/unrealized PnL.

## Offline acceptance scenarios
| Scenario | Expected |
|---|---|
| Past peak 1,000, next day max 950 | 1,050 goal remains |
| Past peak 1,000, today's high 1,040, no hit | Next day's goal 1,092 |
| Frozen goal 1,050, observed equity exactly 1,050 | Single trigger and immutable price/quantity snapshot |
| Equity falls after trigger | No automatic downward repricing |
| Some exits unfilled at midnight | No new real entries until verified all flat and cleared |
| All exits confirmed same day | Stay locked until Paris midnight |
| API timeout, old market quote, unknown fill | No new buys; alert and reconcile, not blind retry |
| Deposit or withdrawal | Transfer excluded from performance |
| Restart between snapshot and exits | Recover durable state, no duplicate order |
| Telegram toggles after trigger | Already triggered lock preserved |

## Implemented read-only runtime
- `crash_daily_profit_paper_live.py` samples USD-M account equity approximately every 5 minutes using the shared REST budget.
- Accounts in Binance multi-assets mode or with non-USDT margin are skipped until validated.
- Cashflows seen in income history since activation are removed from the paper equity series; this is a best-effort estimator, not a reconciled audited PnL statement.
- The initial historical high is **unknown**: the first validated equity sample becomes the initial peak. No previous-day high is invented.
- Trigger snapshots capture Binance position mark prices; revisiting such prices is an **indicative paper observation**, never a confirmed limit execution.
- Durable state: `<CRASH_NIGHT_STATE_DIR>/daily_profit_paper_state.json` or `<DATA_DIR>/night_execution/daily_profit_paper_state.json`.
- Owner-checked Telegram: `/gunluk_kar_durum`, `/gunluk_kar_ac`, `/gunluk_kar_kapat`, `/gunluk_kar 5`. Target changes apply next Paris day.
- Enabled in paper-only mode at worker startup when the normal Crash broker exists; failures skip only paper samples. No real trading safety change has been deployed.

## Remaining before real orders
Build an independently reviewed, restart-safe real closing engine with authenticated order ownership, precise fill reconciliation, partial fills, order replacement, price filters, and live dry-run test coverage. The currently deployed paper monitor must **not** be mistaken for a live profit lock.
