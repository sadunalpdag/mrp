"""Entry-time Crash Bounce V2 features. Public CLOSED 15m data; no orders."""
import math

STEP_MS = 900_000
HISTORY_CANDLES = 480
MIN_RECOVERY_PCT = 1.0
MIN_RSI14 = 20.0


def _rsi_wilder(closes):
    if len(closes) < 15:
        return None
    deltas = [b-a for a,b in zip(closes, closes[1:])]
    gain = sum(max(0.0, x) for x in deltas[:14]) / 14
    loss = sum(max(0.0, -x) for x in deltas[:14]) / 14
    for diff in deltas[14:]:
        gain = (13*gain + max(0.0, diff))/14
        loss = (13*loss + max(0.0, -diff))/14
    return 100.0 - 100.0 / (1.0 + gain / loss) if loss else (100.0 if gain else 50.0)


def assess_entry(history_rows, ws_rows, at, btc_change_24h=None):
    """Use 478-480 real CLOSED candles in the fixed 5-day timestamp window.

    One or two missing bars are allowed per user request; no synthetic candles
    are inserted. The most recent closed candle MUST be available and >=14 of
    the latest 16 time slots must be present. Older/more gaps fail closed.
    No unfinished or future data enters a decision.
    """
    end = int(at * 1000) // STEP_MS * STEP_MS
    start = end - HISTORY_CANDLES * STEP_MS
    candles = {}
    quote_volumes = {}
    provenance = {}
    for batch in (history_rows, ws_rows):
        if not isinstance(batch, list):
            continue
        for row in batch:
            try:
                opened = int(row[0]); closed = int(row[6])
                if not (start <= opened < end and opened % STEP_MS == 0 and
                        closed == opened + STEP_MS - 1 and closed < at * 1000):
                    continue
                o,h,l,c,v = (float(row[j]) for j in (1,2,3,4,5))
                if (not all(math.isfinite(x) for x in (o,h,l,c,v)) or
                        min(o,h,l,c) <= 0 or v < 0 or
                        l > min(o,c) or h < max(o,c) or l > h):
                    continue
                new_source = row[12] if len(row)>12 else 'BINANCE_CACHE_WS'
                if (opened in candles and provenance[opened] != 'BYBIT_LINEAR'
                        and new_source == 'BYBIT_LINEAR'):
                    continue  # Binance data always outranks a cached Bybit gap.
                provenance[opened] = new_source
                candles[opened] = (o,h,l,c,v)
                try:
                    quote = float(row[7])
                    quote_volumes[opened] = quote if math.isfinite(quote) and quote >= 0 else None
                except (IndexError, TypeError, ValueError, OverflowError):
                    quote_volumes[opened] = None
            except (IndexError, TypeError, ValueError, OverflowError):
                continue
    present = sum(start+i*STEP_MS in candles for i in range(HISTORY_CANDLES))
    recent_times = range(end-16*STEP_MS, end, STEP_MS)
    recent = [candles[t] for t in recent_times if t in candles]
    if (present < HISTORY_CANDLES-2 or end-STEP_MS not in candles
            or len(recent) < 14):
        return {"status":"MISSING_5D_HISTORY", "qualified":False,
                "candles":present, "expected_candles":HISTORY_CANDLES,
                "missing_candles":HISTORY_CANDLES-present,
                "recent_4h_observed":len(recent)}
    ordered = [candles[start+i*STEP_MS] for i in range(HISTORY_CANDLES)
               if start+i*STEP_MS in candles]
    closes = [c[3] for c in ordered]
    rsi = _rsi_wilder(closes)
    low4h = min(c[2] for c in recent)
    recovery = 100.0*(closes[-1]/low4h-1.0)
    five_day_return = 100.0*(closes[-1]/ordered[0][0]-1.0)
    qualified = rsi is not None and rsi >= MIN_RSI14 and recovery >= MIN_RECOVERY_PCT
    # Continuous strength is informational only, not an extra V2 acceptance gate.
    from crash_score_v3 import score_candidate
    strength_score = score_candidate(ordered, btc_change_24h=btc_change_24h)
    # Daily 96-bar partitions are meaningful only with all timestamps present.
    # For partial history, omit the 5-day context instead of shifting day edges.
    context = None
    if present == HISTORY_CANDLES:
        from crash_context import assess_context
        context = assess_context(ordered,
            [quote_volumes.get(start+i*STEP_MS) for i in range(HISTORY_CANDLES)],
            strength_score)
    return {"status":"STRONG_V2" if qualified else "FILTERED_V2",
            "five_day_context": context,
            "strength_score":strength_score,
            "qualified":qualified, "candles":present,
            "expected_candles":HISTORY_CANDLES,
            "missing_candles":HISTORY_CANDLES-present,
            "recent_4h_observed":len(recent),
            "candle_tolerance_version":"missing_up_to_2_real_closed_v1",
            "rsi14":round(rsi,4),
            "recovery_4h_pct":round(recovery,4),
            "low_4h":low4h, "return_5d_pct":round(five_day_return,4),
            "last_closed_open_ms":end-STEP_MS}
