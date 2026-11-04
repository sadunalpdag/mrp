# Multi-exchange 15-minute history fallback

This change is **separate from** the 4H-breakout + Drive backup PR #125.
It changes only how missing *closed* research candles are obtained, not
Crash Bounce V2 signal rules, risk labels, BUY or TP order handling.

## Provider priority

1. Existing Binance WebSocket / persisted 15m candles.
2. Cached, checksum-verified Binance daily public archives
   (different host from the trading REST API).
3. Bybit **USDT linear futures** candles, missing timestamps only.
4. Coinbase Exchange **USD spot** candles, missing timestamps only.
5. Binance USDT-M REST candles **only if gaps remain**, and only when
   the Binance REST ban/cooldown guard allows it.

When a coin is unavailable on both Bybit and Coinbase, Binance REST is
attempted after the provider pacing windows allow an attempt. An active
Binance 418 ban is never bypassed. The bot does not create a proxy or
alternate Binance IP.

**Important:** Bybit and Coinbase market data can differ from Binance
perpetual futures. Alternative candles require a directly adjacent,
same-time Binance candle with <=3% close-price divergence. No anchor or
larger divergence means the fallback is rejected, leaving missing data
unfilled. No fabricated bars, no live orders from external exchanges.

## Provider pacing

- Bybit: max 12 requests in a rolling hour, >=180 seconds between
  requests, >=1800 seconds per symbol. Honor HTTP 429 Retry-After with
  300-3600 seconds of cooldown.
- Coinbase: max 12 requests in a rolling hour, >=180 seconds between
  requests, >=1800 seconds per symbol; unsupported products (HTTP 400/404)
  skipped for 24 hours. Honor HTTP 429 Retry-After cooldown.
- Binance: existing shared research limiter and ban guard remain in
  force; REST used only for unresolved gaps or a paced replacement of
  provisional Bybit/Coinbase candles. Max 192 contiguous 15m bars per call.
- The history collector already executes one queued history job per
  60 seconds, so new sources are not polled in parallel.
- Data is tagged `BINANCE_ARCHIVE`, `BYBIT_LINEAR`,
  `COINBASE_SPOT`, or `BINANCE_REST` for analysis and reports.
  Coinbase volume is spot base-volume with approximate quote-volume
  (base volume × closing price), not Binance futures volume.

## Operational caveats

Coinbase public Exchange candles use the product format `BTC-USD`,
`ETH-USD`, etc., and only exist for listed products. Binance-only
futures tickers will normally fall through to Binance REST. Provider
unavailability is not a signal to ignore Binance cooldowns.

If all providers are unavailable, the history remains incomplete and
the existing V2 validation continues to require 480 closed 15m candles.
The source-specific pacing state is process-local; only one Render worker
instance should run. The existing `/data` persisted research cache
continues to be used. A future hardening step could persist all venue
cooldowns and add a central cross-module IP budget for Binance account
and market endpoints; this PR does not claim to solve account REST bans.
