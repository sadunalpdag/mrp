# Crash research: Google Drive daily backup + 4H falling-trend research

Both features are isolated from the Crash Bounce V2 signal and order rules.
Neither is allowed to block Telegram reporting, BUY/TP handling, or the main scan.

## Daily Google Drive backup

The existing 21:00 Europe/Paris Crash ZIP already contains `failure_shadow.csv`,
`report.json`, and `signals_and_outcomes.json`. This change also includes
`breakout_4h.csv` (when its research worker is initialized). A separate daemon
uploads two files to `Crash_Bounce/YYYY-MM/`:

- `crash_YYYY-MM-DD_HHMMSS.zip` (full unmodified report)
- `risk_YYYY-MM-DD_HHMMSS.csv` (risk flags and paper outcomes)

The timestamp comes from the report's `period_end`, not upload time.
An outbox at `/data/crash_drive_backup/drive_backup_outbox.json` persists retries
independently of Telegram success. File lookup and update avoid duplicate uploads
when a completed remote write is followed by a local restart. Failed uploads use
exponential backoff capped at 6 hours, with Telegram notification on first failure
and success; no API secrets are printed.

Enable only **after configuring credentials** on Render:
```env
CRASH_DRIVE_BACKUP_ENABLED=1
```

For a personal My Drive use Google OAuth **offline refresh-token** credentials:
```env
GOOGLE_DRIVE_CLIENT_ID=...
GOOGLE_DRIVE_CLIENT_SECRET=...
GOOGLE_DRIVE_REFRESH_TOKEN=...
# Optional folder ID; omit to create Crash_Bounce at root:
GOOGLE_DRIVE_PARENT_FOLDER_ID=...
```
Use an OAuth client with Drive API enabled and the
`https://www.googleapis.com/auth/drive.file` scope, grant offline access, and
keep refresh token/client secret **only in Render environment**, never in GitHub.

Alternatively, for a **Shared Drive**, use an existing Google service account
with write permission to the target folder:
```env
GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON=<complete JSON or Render secret file path>
GOOGLE_DRIVE_PARENT_FOLDER_ID=<writable Shared Drive folder ID>
```
`GOOGLE_SERVICE_ACCOUNT_JSON` is an optional compatibility fallback.
For service accounts, a parent folder ID is mandatory; do not assume the service
account can upload into a consumer My Drive root or that it owns Drive storage.
If using OAuth for a pre-existing parent folder, it must be accessible under the
selected OAuth scope (for example by selecting it using an authorized Google flow).
The plugin connection in ChatGPT is independent of Render service credentials.

Missing credentials / insufficient folder access cause retry notifications,
not failed main-worker startup. The ZIP remains available in Telegram as before.
Only one bot instance should run, and the `/data` disk must be mounted to keep
the outbox and source ZIP across redeploys.

## 4H falling-trend breakout (research only)

`CRASH_4H_BREAKOUT_ENABLED=1` by default; set to `0` to disable.
The independent research daemon uses the existing Binance USDT-M Futures
WebSocket ticker snapshot and validates symbols against the existing
active-perpetual metadata. It selects up to 30 liquid symbols with 24h USDT
notional volume >= 10 million, prioritizing 24h changes -3% to -10%,
-10% to -30%, and +5% to +15%.

From up to 65 *closed* four-hour candles it requires at least three descending
swing highs (neighbor-confirmed pivot highs). The falling trend line extrapolates
the last two pivot highs; a **confirmed breakout** requires a closed 4H
crossing, >=1.5x the preceding 20-bar quote-volume average, and 4H RSI14 35–70.
Unconfirmed low-volume and overextended (>2% above line) moves are reported
separately. Near-line candidates use distance -2% to 0%, RSI 35–55, and
improving EMA25 slope. These are *unvalidated research heuristics*.

**WS-first / REST fallback discipline:**
- Fetch up to 65 *closed* 4H bars per candidate once at bootstrap, slowly:
  at most **one 4H REST request per 120 seconds** across the new scanner
  (65 bars = Binance Futures REST request weight 1).
- Subsequent 4H candles are consumed from `{symbol}@kline_4h` using the
  existing futures WebSocket connection. Only events with `k.x == true` and
  correct 4H alignment are accepted. Bars merge by open timestamp into the
  persisted 65-bar research history.
- To limit WS message/CPU load (Binance emits updates to open 4H bars every
  ~250ms), the 4H research channels are subscribed only in the final 3 minutes
  **before** a 4H close and first 3 minutes **after** it; up to 30 at once.
  They are unsubscribed for the rest of the four-hour interval.
- New closes have a **five-minute grace period** for WebSocket delivery. If
  the socket was disconnected, a candle is missed, bootstrap is not complete,
  or sequence continuity fails, the missing history is repaired via the paced
  shared REST client. The scanner never treats a gapped series as a trend
  breakout. One failed exchange request does not block the main V2 loop.
- The existing `_crash_public_get` shared rate limiter, ban/cooldown and
  endpoint guards remain in force. No dedicated REST client is added.
- Best-effort 15m confirmation uses existing WS 15m cache; no extra 15m REST.
- After the initial history, a healthy continuous WS connection ideally needs
  **zero daily REST requests** for the same 30 symbols; backup requests are
  demand-driven after gaps, unavailable WS data, or watchlist changes.
- 30 uncached candidates take **at least 60 minutes** to bootstrap. Events
  observed late or with incomplete coverage cannot be assumed to have been
  profitable or unprofitable.

The paper research ledger is written atomically under
`/data/crash_4h_breakout/breakout_4h_state.json` with event-time flags,
observations, maximum sampled rises, and separately completed 1/4/12/24h
horizons. A Telegram research summary is sent once per 4H block, at least
60 minutes after that block begins. Sparse WebSocket coverage can miss highs:
reported percentages are observed samples, **not exchange OHLC maxima or a
real-trading performance estimate**.

### What does not change

Crash Bounce V2 signal criteria, risk-tagged live BUY rejection, entry level,
TP = +1%, order manager, existing Telegram alerts, and 21:00 report schedule
remain unchanged. Drive authorization requires a separate Render configuration.
