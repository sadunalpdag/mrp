# Trend Breakout Shadow + Cloudflare Workers AI

Render automatically creates separate `trend_breakout_YYYY-MM-DD.json` and `.zip`
under the EXISTING private R2 bucket:

`<CRASH_REPORT_R2_PREFIX>/trend-breakout/YYYY/MM/DD/`

The JSON is a **7-day rolling snapshot** of observations sourced only from
existing Crash Bounce-followed 15-minute candles, **at most 20 symbols** per
15-minute research scan. It includes 24h/48h/72h status, first +1%/-1% touch,
maximum high/low movement and data gaps. It does **not** assert that a real
exchange order would be filled, and it does not trade.

The ZIP is for backups; the uncompressed JSON is for the AI Worker.

## One-time Cloudflare setup

1. In Cloudflare dashboard, confirm the existing private R2 bucket that
   the Render `CRASH_REPORT_R2_BUCKET` setting points to.
2. Install Wrangler (Node.js 20+): `npm install -g wrangler`, then
   `wrangler login`.
3. Edit this directory's `wrangler.toml`:
   replace `YOUR_EXISTING_PRIVATE_R2_BUCKET_NAME` with the actual existing
   bucket name. If Render uses a custom `CRASH_REPORT_R2_PREFIX`, set
   `R2_PREFIX` to that same prefix.
4. From **this directory**, deploy using `wrangler deploy`.
5. In Cloudflare Workers & Pages, select `trend-breakout-ai-review` and
   inspect Cron Trigger and logs. The scheduled job runs daily at 20:45 UTC.
   Cron times are UTC. The AI model may incur usage charges depending on plan.
6. Inspect generated reports at
   `<prefix>/trend-breakout/ai-reports/YYYY/MM/DD/trend_breakout_ai_YYYY-MM-DD.json`.

No R2 credentials are embedded in the Worker. It uses an R2 binding and a
Workers AI binding. No public HTTP endpoint and no trading API access.

For a simple deployment smoke test, upload a valid source JSON to the date
folder. Workers will skip existing report objects; remove a report object only
if a deliberate re-analysis is needed. If Render's daily upload fails, no
AI report is created until the input appears (retry on next schedule required).
Test with a nonproduction R2 bucket first if unsure.

### Event interpretation

- A confirmed channel breakout uses two successive closed 15-minute candles
  beyond a 0.15% buffer, calculated from cached 5-day history.
- `max_up_pct`, `max_down_pct` use following CLOSED 15-minute candle HIGH/LOW,
  not executable futures trades or price tick chronology.
- `horizons` of 24/48/72 hours may be null or flagged with a gap.
- Research stores only candle information already present in Crash Bounce
  subscriptions; if a symbol leaves that existing set, follow-up data may stop.
- The research thread cannot submit live orders or modify risk sizing.
