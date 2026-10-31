# OpenAI signal review (shadow research)

Only newly delivered Crash Bounce V2 signals are evaluated. V2 filters, original strength scores, entry/TP prices, automatic-entry switches and order execution are unchanged. No new Binance requests are made by this module. This is not a trading agent with exchange tools.

## Render setup

In the existing service's Environment settings add:

```
OPENAI_API_KEY=<your OpenAI project API key>
CRASH_AI_ENABLED=1
CRASH_AI_MAX_DAILY=50
```

Do not commit or send the key in Telegram/chat. Redeploy/restart after changing environment variables. API billing/access must be enabled in the OpenAI project. With no key, the module logs OFF and the existing strategy continues. Set `CRASH_AI_ENABLED=0` to disable analysis independently of automatic orders.

Default model: `gpt-5.4-mini-2026-03-17`. Optional `CRASH_AI_MODEL` override; it must support Responses, JSON Schema and `reasoning.effort=none`. Cost is reported as unknown for unrecognized model IDs. No SDK installation is needed; the existing `requests` dependency is used.

## Flow

- Accepted Telegram signal -> frozen allowlisted numerical signal snapshot -> background queue.
- Wait at most 120 seconds for the existing OI/CVD worker, using exactly the same signal timestamp. Missing values remain null. Later prices, TP outcomes, account data, original V2 score and shadow score are excluded from the model input.
- One OpenAI Responses request per signal. No browser/search tools; `store=false`; output cap 1,000 tokens; at least 10 seconds between attempts. HTTP 429 pauses the AI queue for 5 minutes, not Binance or V2.
- Strict JSON plus local validation. Failure/refusal/incomplete response produces no score. Missing core entry/TP/RSI forces a null score. Model scoring remains subjective and experimental, not a probability.
- Separate Telegram AI message with score, data quality, supporting evidence, risks and missing data. Two-hour system status includes whether AI analysis is enabled.
- Result, exact sanitized input, input hash, model, prompt hash, token usage and estimated USD are saved in the signal state and `openai/ai_events.jsonl`. Daily ZIP adds `openai_reviews.csv`; full inputs/results are also in `signals_and_outcomes.json` and daily observations. Outcomes and MAE join by signal ID for prospective evaluation.

The worker is single-process. Keep one Render instance for this bot. Default daily cap is 50 attempts (Europe/Paris), independent of trading limits; configurable 1–100. Attempts are journaled before HTTP to avoid blind paid retries after timeouts/restarts. Pending, not-yet-attempted in-memory jobs can be lost on restart; older signals are not automatically re-enrolled. A failed/ambiguous Telegram delivery is recorded without retrying the OpenAI request. If local storage is erased during redeploy, its historical journal/counters are lost; regular report exports preserve research outside that instance. This module does not provision persistent storage.

## Cost estimate, verified 2026-10-31

Official GPT-5.4 Mini standard rates: $0.75 per million input tokens, $4.50 per million output tokens:
https://developers.openai.com/api/docs/models/gpt-5.4-mini

Assuming 2,000 input + 500 output tokens per review:

| Volume | Estimated API cost |
| --- | ---: |
| One signal | $0.00375 |
| 20 signals/day, 30 days | $2.25/month |
| 50 signals/day, 30 days | $5.625/month |

Actual usage varies. The 1,000-token output cap is a ceiling, not a target. Recorded estimate conservatively charges all input at the uncached price. Taxes, hosting, other API usage and future price changes are excluded. Configure project billing alerts separately; this is an attempt cap, not an account-wide dollar budget.

## Verification

Offline tests mock OpenAI and Telegram. They cover frozen allowlists, future-data exclusion, original score exclusion, missing-key behavior, duplicate/restart prevention, daily cap, timeout/refusal/incomplete responses, authorization/rate-limit errors, CSV export and continued real-order submission when the AI queue fails. A successful paid request and Render deployment must be checked after the key is configured.
