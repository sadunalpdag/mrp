# BTC paper range study

New feature is opt-in and performs no Binance order placement. Existing BTC hourly direction + CB2/Crash Bounce real-order code is unchanged.

## Settings
- BTC_DIRECTION_OBSERVER_ENABLED=1: existing hourly observation thread (must be running).
- BTC_RANGE_PAPER_ENABLED=1: new paper study (default disabled).
- BTC_RANGE_AI_ENABLED=1 and GEMINI_API_KEY: optional initial 30-day Gemini reference study.
- BTC_RANGE_AI_HOURLY_ENABLED=1: optional hourly Gemini pre-breakout interpretation; incurs API costs.
- BTC_RANGE_GEMINI_MODEL: defaults to gemini-2.5-flash-lite.

## Logic
On the hourly observation, fetch only closed BTCUSDT 15-minute candles from Binance Futures. Build an initial 2,880-candle, 30-day reference with normalized OHLC, volume and descriptive historical results; refresh after 30 days. Store locally and in Cloudflare R2 under btc-range/references/YYYY-MM-DD.json and btc-range/reference_latest.json.
Use the last 96 x 15m bars to estimate support/resistance from low/high percentiles. Require repeated level touches, ADX <= 20, low slope, and range width 0.9–5%. A close within 0.30% above support can create one virtual limit long. Entry is support +0.10%, target resistance -0.10%, invalidation support -0.30%, and minimum estimated net 0.45% after 0.08% assumed roundtrip fees. One pending/open experiment at once.
Paper orders expire unfilled after six hours; filled positions close hypothetically at target, invalidation or a 24-hour timeout, using only later closed candles. Same-candle ambiguous target/stop is scored conservatively as invalidation. A same-candle target on entry is not counted as a win. This is a backtest approximation; gaps, fees and slippage make actual trades different.
Signal suggestion appears in the existing hourly Telegram notification, marked SANAL. Existing BTC_YON_ANALIZ sheet gains columns: Yatay rejim, Destek, Direnc, Sanal durum, Sanal net %. Results persist in the existing per-day btc-direction R2 JSON, with separately versioned 30-day research archive.

## Limitations and release
Support/resistance here is inferred from Binance OHLC candles, not CoinBeacon API; CoinBeacon level integration remains separate pending the relevant documented field. Reference Gemini explanations and hourly AI comments require explicit flags and have API usage costs. Deploying to Render and enabling flags are distinct actions; creating a PR alone does not make the feature live. Verify CI, R2, Sheets and resource limits before enabling.