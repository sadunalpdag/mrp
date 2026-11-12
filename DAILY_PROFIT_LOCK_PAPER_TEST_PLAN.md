# Daily Profit Lock — Paper Test Plan (approved 2026-11-12)

Status: **approved design for offline/paper tests; NOT active on live Binance**.

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

## Implementation boundary
An independent offline module can test frozen daily goals and lock transitions. It must not call Binance order placement, Telegram, Sheets, or Render. **No main-branch deploy, no change to `ema.py` or `crash_night.py`, and no real order modification from this plan.** Before real trading rollout, build account-equity/transfer data validation, durable state persistence, price/precision filters, order reconciliation, partial-fill logic, and comprehensive isolated tests.
