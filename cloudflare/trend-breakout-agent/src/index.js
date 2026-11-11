// Scheduled Cloudflare Workers AI reader for private R2 Trend Breakout JSON.
// Deliberately cannot access Binance/trading; exports research analysis to R2 only.
const MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

function parisDate(timestampMs) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/Paris", year: "numeric", month: "2-digit", day: "2-digit"
  }).formatToParts(new Date(timestampMs));
  const get = (name) => parts.find(p => p.type === name)?.value;
  return [get("year"), get("month"), get("day")].join("-");
}

function safeNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function mean(values) {
  if (!values.length) return null;
  return Number((values.reduce((a, b) => a + b, 0) / values.length).toFixed(4));
}
function median(values) {
  if (!values.length) return null;
  const s = values.slice().sort((a, b) => a - b);
  const n = Math.floor(s.length / 2);
  return s.length % 2 ? s[n] : Number(((s[n - 1] + s[n]) / 2).toFixed(4));
}

function summarizeGroup(events) {
  const result = {count: events.length};
  const ups = [], downs = [];
  let plus1 = 0, minus1 = 0, gaps = 0;
  for (const event of events) {
    const up = safeNumber(event.max_up_pct), down = safeNumber(event.max_down_pct);
    if (up !== null) ups.push(up);
    if (down !== null) downs.push(down);
    if (safeNumber(event.first_plus_1pct_at) !== null) plus1++;
    if (safeNumber(event.first_minus_1pct_at) !== null) minus1++;
    if (event.coverage_gaps > 0) gaps++;
  }
  result.first_plus_1pct_count = plus1;
  result.first_minus_1pct_count = minus1;
  result.max_up_mean_pct = mean(ups);
  result.max_down_median_pct = median(downs);
  result.events_with_gaps = gaps;
  result.horizons = {};
  for (const hours of [24, 48, 72]) {
    const closes = events.map(e => safeNumber(e.horizons?.[hours + "h"]?.close_pct))
      .filter(v => v !== null);
    result.horizons[hours + "h"] = {
      measured: closes.length, mean_pct: mean(closes), median_pct: median(closes)
    };
  }
  return result;
}

export function aggregate(source) {
  if (!source || source.schema_version !== "trend_breakout_shadow_v1" ||
      !Array.isArray(source.events)) throw new Error("INVALID_TREND_SCHEMA");
  // Bounds prevent excessive Workers AI token spending.
  const events = source.events.slice(0, 500).filter(e => e &&
    /^[A-Z0-9]{2,24}USDT$/.test(String(e.symbol || "")) &&
    ["YUKARI", "AŞAĞI"].includes(e.direction) &&
    safeNumber(e.reference_price) !== null && e.reference_price > 0);
  const byDirection = {}, byRisk = {};
  for (const direction of ["YUKARI", "AŞAĞI"])
    byDirection[direction] = summarizeGroup(events.filter(e => e.direction === direction));
  for (let count = 0; count <= 4; count++)
    byRisk[count] = summarizeGroup(events.filter(e => e.signal_context?.risk_count === count));
  const outliers = events
    .filter(e => safeNumber(e.max_down_pct) !== null)
    .sort((a, b) => a.max_down_pct - b.max_down_pct).slice(0, 8)
    .map(e => ({
      symbol: e.symbol, direction: e.direction,
      max_down_pct: e.max_down_pct, max_up_pct: safeNumber(e.max_up_pct),
      bars_observed: e.bars_observed, coverage_gaps: e.coverage_gaps
    }));
  return {
    sample_count: events.length, by_direction: byDirection,
    by_risk_count: byRisk, worst_adverse_moves: outliers,
    limitations: ["Research-only 15m candle OHLC, intrabar order unknown",
      "No live order fills, fees, funding, spread or liquidation simulation",
      "Same coin can have correlated breakout events",
      "Unmeasured horizon rows must never be treated as zero returns"]
  };
}

function objectPaths(prefix, day) {
  if (!ISO_DAY.test(day)) throw new Error("INVALID_DATE");
  if (!/^[a-zA-Z0-9/_-]+$/.test(prefix)) throw new Error("INVALID_PREFIX");
  const stem = day.replaceAll("-", "/");
  return {
    source: prefix + "/trend-breakout/" + stem + "/trend_breakout_" + day + ".json",
    report: prefix + "/trend-breakout/ai-reports/" + stem +
            "/trend_breakout_ai_" + day + ".json"
  };
}

async function analyzeDay(env, day) {
  const prefix = (env.R2_PREFIX || "crash-bounce").replace(/^\/+|\/+$/g, "");
  const {source, report} = objectPaths(prefix, day);
  if (await env.ARCHIVE_BUCKET.head(report)) return "EXISTS";
  const item = await env.ARCHIVE_BUCKET.get(source);
  if (!item) return "SOURCE_NOT_FOUND";
  if (item.size > 3_000_000) throw new Error("DAILY_JSON_TOO_LARGE");
  const content = await item.json();
  const statistics = aggregate(content);
  let analysis;
  if (statistics.sample_count === 0) {
    analysis = "Kırılım kaydı yok. İstatistiksel sonuç çıkarılamaz.";
  } else {
    const result = await env.AI.run(MODEL, {
      messages: [
        {role: "system", content:
          "You are a quantitative research analyst. Reply in Turkish, max 450 words. " +
          "Analyze only the supplied aggregates. Distinguish observation from proof, " +
          "don't invent sample counts, avoid trading recommendations and orders. " +
          "Highlight missing horizons, weak sample sizes, adverse moves, follow-up tests."},
        {role: "user", content: JSON.stringify(statistics)}
      ],
      max_tokens: 700
    });
    analysis = String(result?.response || "").slice(0, 9000);
    if (!analysis) throw new Error("AI_EMPTY_RESPONSE");
  }
  const reportData = {
    schema_version: "trend_breakout_ai_review_v1",
    source_key: source, sample_day_paris: day,
    produced_at: new Date().toISOString(),
    model: statistics.sample_count ? MODEL : "none_empty_sample",
    research_only: true, trading_execution: false,
    statistics, analysis_tr: analysis
  };
  await env.ARCHIVE_BUCKET.put(report, JSON.stringify(reportData), {
    httpMetadata: {contentType: "application/json"}
  });
  return "SAVED";
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil((async () => {
      const day = parisDate(event.scheduledTime || Date.now());
      const status = await analyzeDay(env, day);
      console.log("[TREND AI]", day, status);
    })());
  }
};
