# Crash Bounce V2 — prospective risk-count order sizing

This experiment changes **new automated real BUY order sizing only**.
It leaves the existing three risk definitions, V2 eligibility, simulated
outcome tracking, target price +1%, leverage 10x, Telegram /tutar,
existing BUY orders and all TP orders unchanged.

| Signal-time risk flags | Real automatic BUY | Fraction of `/tutar` |
| --- | --- | --- |
| 0 of 3 | permitted | 100% |
| 1 of 3 | permitted | 100% |
| 2 of 3 | permitted | 50% |
| 3 of 3 | blocked | 0% |
| any of 3 unknown, or missing assessment | blocked | 0% |

The three independent booleans are `weak_volume`, `falling_rsi_ema`
and `repeat_lower_entry` as stored at candidate time in
`failure_shadow.flags`. Do not use eventual TP results when assessing.
Live order checks the individual flags twice (before market preflight and
again immediately before persisting/submitting the BUY). An unexpected or
incomplete assessment fails closed.

**Notional vs margin**: `/tutar 500` is a 500 USDT maximum order notional,
not margin. With exactly two flags, order notional is capped at 250 USDT.
At 10x leverage, that is about 25 USDT initial margin before fees.
Order quantity always rounds down according to Binance symbol filters.
If half-size is under the exchange minimum, **do not enlarge the order**;
report the rejected preflight and skip the BUY.

## Audit and research fields

- The original `failure_shadow` stored on each eligible new V2 signal
  includes `sizing_policy_version=risk_count_sizing_v1`,
  `position_fraction`, actual flag booleans, flag_count and known_flags.
  Earlier historical labels retain their original sizing-policy absence.
- A new real BUY reserves the planned size and flags in persistent
  `crash_night_state.json` **before** submitting to Binance:
  `risk_sizing_version`, `risk_flags`, `risk_flag_count`,
  `risk_position_fraction`, `risk_configured_usdt`,
  `risk_target_usdt` and `risk_effective_notional_usdt`. Persisted
  `phase`, quantities and `filled_qty` continue tracking execution/TP.
- Telegram V2 messages show full, half, blocked or incomplete.
- Daily ZIP `failure_shadow.csv` gains risk sizing version, fraction,
  action and corresponding actual live-order fields, matched by signal ID.
  `report.json` gets prospective **exclusive** cohorts
  `sizing_cohorts_prospective`: zero / one / two / three / incomplete flags,
  with separate paper outcomes, live order counts, cumulative target
  notionals and live order phases.
- Old unobserved risk sizing is **not** backfilled as a real live trade.
  Old virtual TP records are never called actual executions.
- Missing data and unknown outcomes are not treated as losses.
  This policy is experimental and does not establish TP probabilities.

**Safety:** Only new auto-entry decisions change. A Binance IP block still
prevents all required account-side order reconciliation; do not infer TP
coverage from public-price data alone.
