# Gemini BTC smoke test (isolated)

This test is NOT wired to production startup yet and does not fetch CoinBeacon itself. It accepts a JSON file containing 16 **closed** BTCUSDT 15m CoinBeacon candles (array, or object with `candles` array). Validate candle provenance and timestamps before running. It never sends orders.

Install: `pip install google-genai`

Set `GEMINI_API_KEY` securely; optional `GEMINI_SMOKE_MODEL` defaults to `gemini-2.5-flash`. For Telegram, optionally set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (verify these match the existing worker's configuration). Run: `python scripts/btc_gemini_smoke.py btc_coinbeacon_15m.json`.

The script prints SUCCESS and token metadata on a nonempty model response; API errors or empty responses fail the test. If Telegram is configured it sends the outcome, but Telegram delivery failure is reported separately. Do not enable startup execution until the CoinBeacon closed-candle source, existing Telegram sender, retry budget, and rate limits are integrated and tested.
