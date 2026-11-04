# Shared Binance REST IP budget (Crash V2)

The former V2 market-data limiter capped only its own requests. It could
not see the concurrently running signed BinanceBroker calls. Binance
`X-MBX-USED-WEIGHT-1M` exposes the *total futures IP weight* and can
be larger than the market worker's local counter.

## New shared policy (single Render worker process)

- Market research (15m klines / BTC hourly / OI / spot) has an **IP soft
  cap of 400/minute** and is paced at a minimum **8 seconds** per call.
- Signed broker account/preflight calls have a soft cap **1400/minute**,
  paced a minimum **1.25 seconds** between requests.
- TP placement, BUY order status checks, and expiration cancellations
  have a separate **protect tier capped at 1750/minute**, paced at 0.25s.
- A response's **actual** `X-MBX-USED-WEIGHT-1M` is folded into the
  shared estimate, so load from other code using the same Render IP
  is visible at the next response.
- A 429/418 causes a shared cooldown across research *and* broker.
  Research cannot bypass a ban. No blind duplicate BUYs are retried.
- The budget is **process-wide**, not cross-service. Other Render
  tenants/containers on the same egress IP are not under this service's
  control. Binance may impose different product/endpoints or order
  count limits; **zero 429s cannot be guaranteed**.
- A single all-symbol `/fapi/v1/openOrders` GET costs 40 weight,
  versus 1 when scoped to a symbol. The safety preflight all-symbol
  checks remain (one per candidate); the guard accounts for their
  full weight. Cache/WS/Bybit/Coinbase priority stays unchanged.

The hard exchange limit can be 2400 requests-weight/minute, but the
service deliberately runs far below it and stops new research sooner.
When soft caps are reached, history work is delayed; no fake candles
are produced and V2 still needs 480 closed 15m bars.

Rate budget decisions are logged with `[CRASH IP BUDGET WAIT]`.
An account request deferred by the gate is **not** a real Binance 429;
the broker returns a safe local -1003 error and sends no request.
The existing active trading state, limits, TP tracking, 4H/Drive PR
and risk sizing remain unchanged.
