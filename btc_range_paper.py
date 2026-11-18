"""BTC range paper research. Reads only public market data; NEVER calls a broker order API.

The feature is opt-in and is called only by the existing hourly BTC observer.
Immutable paper entry plans use closed candles. Their outcomes use *later* closed
15-minute candles, conservatively treating ambiguous TP/stop candles as losses.
"""
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

LOG = logging.getLogger("btc_range_paper")
FIFTEEN = 900_000
HOUR = 3_600_000
DAY = 86_400_000
REFERENCE_BARS = 30 * 24 * 4
RECENT_BARS = 48 * 4
FEE_ROUNDTRIP_PCT = 0.08  # Conservative assumed fees; no slippage guarantee.
REFERENCE_KEY = "btc-range/reference_latest.json"


def _folder():
    folder = Path(os.getenv("DATA_DIR") or (Path(__file__).parent / "data"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _fetch_15m(end_ms, count, priority_checker):
    """Contiguous closed 15m candles ending at end_ms (the exclusive time bound)."""
    from crash_rest_budget import BUDGET, estimated_weight
    first = end_ms - count * FIFTEEN
    bars = {}
    cursor = first
    while cursor < end_ms:
        if priority_checker():
            raise RuntimeError("LIVE_CRASH_PRIORITY")
        limit = min(1000, math.ceil((end_ms - cursor) / FIFTEEN))
        params = {"symbol": "BTCUSDT", "interval": "15m", "startTime": cursor,
                  "endTime": end_ms - 1, "limit": limit}
        path = "/fapi/v1/klines"
        BUDGET.reserve("research", estimated_weight(path, params))
        response = requests.get("https://fapi.binance.com" + path, params=params, timeout=15)
        BUDGET.observe(response.headers, response.status_code,
                       response.headers.get("Retry-After", 0))
        response.raise_for_status()
        received = response.json()
        if not received:
            raise RuntimeError("BTC_15M_EMPTY_HISTORY")
        for b in received:
            opened = int(b[0])
            if cursor <= opened < end_ms and opened + FIFTEEN <= end_ms:
                bars[opened] = [opened] + [float(b[i]) for i in (1, 2, 3, 4, 5)]
        new_cursor = int(received[-1][0]) + FIFTEEN
        if new_cursor <= cursor:
            raise RuntimeError("BTC_15M_CURSOR_NOT_ADVANCING")
        cursor = new_cursor
    expected = list(range(first, end_ms, FIFTEEN))
    if any(x not in bars for x in expected):
        raise RuntimeError("BTC_15M_HISTORY_GAP")
    return [bars[x] for x in expected]


def _percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    left = int(pos)
    return ordered[left] * (1 - pos + left) + ordered[min(left + 1, len(ordered) - 1)] * (pos - left)


def _touches(values, level, band=0.0025):
    """Separated touches instead of counting 4 adjacent candles as 4 tests."""
    n = 0
    last = -10
    for i, price in enumerate(values):
        if abs(price / level - 1) <= band and i - last >= 4:
            n += 1
            last = i
    return n


def detect_range(bars, adx):
    """A transparent heuristic, not a predictive probability or AI instruction."""
    if len(bars) < 96:
        return {"regime": "INSUFFICIENT_DATA"}
    window = bars[-96:]
    lows = [b[3] for b in window]
    highs = [b[2] for b in window]
    closes = [b[4] for b in window]
    support = _percentile(lows, 0.10)
    resistance = _percentile(highs, 0.90)
    close = closes[-1]
    if support <= 0 or resistance <= support:
        return {"regime": "INVALID"}
    width = 100 * (resistance / support - 1)
    slope = abs(100 * (sum(closes[-16:]) / sum(closes[:16]) - 1))
    touch_s = _touches(lows, support)
    touch_r = _touches(highs, resistance)
    regime = (0.9 <= width <= 5.0 and slope <= 1.2
              and adx is not None and adx <= 20
              and touch_s >= 2 and touch_r >= 2
              and support * .998 <= close <= resistance * 1.002)
    proximity = 100 * (close / support - 1)
    result = {
        "regime": "RANGE" if regime else "NO_RANGE",
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "width_pct": round(width, 3),
        "slope_pct": round(slope, 3),
        "support_touches": touch_s, "resistance_touches": touch_r,
        "support_distance_pct": round(proximity, 3),
        "last_closed_price": close,
    }
    if regime and 0 <= proximity <= 0.30:
        entry = support * 1.001
        target = resistance * .999
        invalidation = support * .997
        expected_net = 100 * (target / entry - 1) - FEE_ROUNDTRIP_PCT
        if entry <= close and expected_net >= .45 and target > entry > invalidation:
            result["candidate"] = {
                "side": "PAPER_LONG", "entry": round(entry, 2),
                "target": round(target, 2),
                "invalidation": round(invalidation, 2),
                "estimated_net_pct": round(expected_net, 3),
                "plan": "SUPPORT_TO_RESISTANCE",
            }
    return result


def _settle(plan, time_ms, exit_price, why):
    plan["status"] = "CLOSED" if why != "ENTRY_EXPIRED" else "EXPIRED"
    plan["exit_time_ms"] = time_ms
    plan["exit_price"] = round(exit_price, 2) if exit_price is not None else None
    plan["result"] = why
    if exit_price is not None:
        plan["net_pct"] = round(100 * (exit_price / plan["entry"] - 1)
                                - FEE_ROUNDTRIP_PCT, 4)


def evaluate_positions(records, bars):
    """Evaluate only candles that closed after a plan was recorded, idempotently."""
    changed_days = set()
    for record in records.values():
        plan = record.get("range_paper")
        if not plan or plan.get("status") in ("CLOSED", "EXPIRED"):
            continue
        for candle in bars:
            opened, o, high, low, close, volume = candle
            if opened < plan["created_ms"] or opened <= plan.get("last_checked_open_ms", -1):
                continue
            plan["last_checked_open_ms"] = opened
            changed_days.add(record["timestamp_paris"][:10])
            if plan["status"] == "PENDING":
                if opened >= plan["created_ms"] + 6 * HOUR:
                    _settle(plan, opened + FIFTEEN, None, "ENTRY_EXPIRED")
                    break
                # Limit fill must trade at the limit, not just gap below it.
                if not low <= plan["entry"] <= high:
                    continue
                plan["status"] = "OPEN"
                plan["filled_ms"] = opened + FIFTEEN
                plan["max_adverse_pct"] = round(100 * (min(low, plan["entry"]) /
                                                        plan["entry"] - 1), 4)
                # Target could have traded BEFORE entry inside this same candle;
                # never award a same-candle TP. Assume a stop if touched.
                if low <= plan["invalidation"]:
                    _settle(plan, opened + FIFTEEN,
                            min(o, plan["invalidation"]), "INVALIDATION")
                    break
                continue
            if plan["status"] == "OPEN":
                plan["max_adverse_pct"] = min(
                    plan.get("max_adverse_pct", 0),
                    round(100 * (low / plan["entry"] - 1), 4))
                # Conservative order of tests on same candle.
                if low <= plan["invalidation"]:
                    _settle(plan, opened + FIFTEEN,
                            min(o, plan["invalidation"]), "INVALIDATION")
                    break
                if high >= plan["target"]:
                    _settle(plan, opened + FIFTEEN, plan["target"], "TARGET")
                    break
                if opened + FIFTEEN >= plan["filled_ms"] + 24 * HOUR:
                    _settle(plan, opened + FIFTEEN, close, "TIMEOUT_24H")
                    break
    return changed_days


def _read_reference():
    path = _folder() / "btc_range_reference.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") == 1:
            return data
    except (OSError, ValueError, AttributeError):
        pass
    try:
        from btc_hourly_direction import _r2_client
        client, bucket = _r2_client()
        if client:
            obj = client.get_object(Bucket=bucket, Key=REFERENCE_KEY)
            data = json.loads(obj["Body"].read())
            if data.get("schema") == 1:
                _write_local_ref(data)
                return data
    except Exception:
        pass
    return None


def _write_local_ref(reference):
    path = _folder() / "btc_range_reference.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _save_reference(reference):
    _write_local_ref(reference)
    try:
        from btc_hourly_direction import _r2_client
        client, bucket = _r2_client()
        if client:
            body = json.dumps(reference, ensure_ascii=False).encode("utf-8")
            day = datetime.fromtimestamp(reference["created_ms"] / 1000,
                                         timezone.utc).strftime("%Y-%m-%d")
            client.put_object(Bucket=bucket,
                              Key=f"btc-range/references/{day}.json",
                              Body=body, ContentType="application/json")
            client.put_object(Bucket=bucket, Key=REFERENCE_KEY,
                              Body=body, ContentType="application/json")
    except Exception as exc:
        LOG.warning("[BTC RANGE R2 REF] %s", type(exc).__name__)


def historical_patterns(bars):
    """Simple labeled study of *past* support-near events vs next 4h outcomes."""
    total = up = down = flat = 0
    for i in range(96, len(bars) - 16, 4):
        window = bars[i - 96:i]
        support = _percentile([b[3] for b in window], .10)
        resistance = _percentile([b[2] for b in window], .90)
        if support <= 0 or not .009 <= resistance / support - 1 <= .05:
            continue
        previous = bars[i - 1][4]
        if not 0 <= previous / support - 1 <= .003:
            continue
        total += 1
        future = bars[i:i + 16]
        top = max(b[2] for b in future)
        bottom = min(b[3] for b in future)
        # Ambiguity is NOT a directional win; classification only.
        if bottom < support * .997:
            down += 1
        elif top > resistance * .999:
            up += 1
        else:
            flat += 1
    return {"historical_support_setups": total, "up_to_resistance_first_proxy": up,
            "broke_below_support_or_ambiguous": down,
            "neither_within_4h": flat,
            "warning": "Descriptive sample, not causal proof or an out-of-sample win rate."}


def _compact(bars):
    base = bars[0][4]
    # Every closed 15m candle in thirty days, expressed as basis points from
    # one anchor; volume in rounded units. No future observation is included.
    return [[int((b[0] - bars[0][0]) / FIFTEEN)]
            + [round(10_000 * (b[i] / base - 1)) for i in (1, 2, 3, 4)]
            + [round(b[5], 2)] for b in bars]


def _gemini(payload, instruction):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return {"status": "NO_API_KEY"}
    model = os.getenv("BTC_RANGE_GEMINI_MODEL", "gemini-2.5-flash-lite")
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,80}", model):
        raise ValueError("BTC_RANGE_GEMINI_MODEL_INVALID")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"systemInstruction": {"parts": [{"text": (
            "Only analyze supplied historical BTC closed candles. No live data, "
            "no future knowledge, no guarantees, no real trading orders. "
            "Return valid JSON in Turkish. Be cautious about overfitting.")}]},
              "contents": [{"role": "user", "parts": [
                  {"text": instruction + "\n" +
                   json.dumps(payload, separators=(",", ":"), allow_nan=False)}
              ]}], "generationConfig": {
                  "responseMimeType": "application/json",
                  "temperature": .1, "maxOutputTokens": 1300}},
        timeout=35)
    response.raise_for_status()
    parts = ((response.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    parsed = json.loads("".join(p.get("text", "") for p in parts))
    if not isinstance(parsed, dict):
        raise ValueError("BTC_RANGE_GEMINI_INVALID_JSON")
    return {"status": "COMPLETE", "analysis": parsed}


def ensure_reference(hour_ms, priority_checker, existing=None):
    reference = existing if existing is not None else _read_reference()
    if reference and hour_ms - reference["created_ms"] < 30 * DAY:
        if (os.getenv("BTC_RANGE_AI_ENABLED") == "1" and
                reference.get("gemini", {}).get("status") == "DISABLED"):
            # Reference is already stored with all candles; no Binance re-download.
            if not priority_checker():
                try:
                    reference["gemini"] = _gemini({
                        "anchor_price": reference["anchor_price"],
                        "starting_utc": reference["start_utc"],
                        "bar_interval_minutes": 15,
                        "candles": reference["compact_candles"],
                        "historical_patterns": reference["historical_patterns"]
                    }, "Analyze 30-day BTC pre-breakout candle/volume behavior: "
                       "support-resistance touches, candle bodies/wicks, false breaks, "
                       "volume contraction/expansion. Output patterns with direction, "
                       "evidence, invalidation, uncertainty, and risks; no future predictions.")
                except Exception as exc:
                    LOG.warning("[BTC RANGE GEMINI REF] %s", type(exc).__name__)
                    reference["gemini"] = {"status": "ERROR", "error": type(exc).__name__}
                _save_reference(reference)
        return reference
    candles = _fetch_15m(hour_ms, REFERENCE_BARS, priority_checker)
    reference = {
        "schema": 1, "created_ms": hour_ms,
        "start_utc": datetime.fromtimestamp(candles[0][0] / 1000, timezone.utc).isoformat(),
        "end_exclusive_utc": datetime.fromtimestamp(hour_ms / 1000, timezone.utc).isoformat(),
        "anchor_price": candles[0][4],
        "interval": "15m", "bar_count": len(candles),
        "compact_candles": _compact(candles),
        "historical_patterns": historical_patterns(candles),
        "gemini": {"status": "DISABLED"},
    }
    # Persist immediately: Gemini/API failure must not trigger repeat 30-day backfill.
    _save_reference(reference)
    if os.getenv("BTC_RANGE_AI_ENABLED") == "1" and not priority_checker():
        try:
            reference["gemini"] = _gemini({
                "anchor_price": reference["anchor_price"],
                "starting_utc": reference["start_utc"],
                "bar_interval_minutes": 15,
                "candles": reference["compact_candles"],
                "historical_patterns": reference["historical_patterns"]
            }, "Analyze pre-breakout candlestick and volume signatures in 30 days. "
               "Identify repeated support/resistance patterns, failed breakouts, "
               "directional biases, explicit evidence and invalidation. JSON keys: "
               "patterns (array), caveats (array), short_summary (string). "
               "Do not use nonexistent future outcomes for a live prediction.")
        except Exception as exc:
            LOG.warning("[BTC RANGE GEMINI REF] %s", type(exc).__name__)
            reference["gemini"] = {"status": "ERROR", "error": type(exc).__name__}
        _save_reference(reference)
    return reference


def _hourly_ai(bars, snapshot, reference, priority_checker):
    if os.getenv("BTC_RANGE_AI_HOURLY_ENABLED", "0") != "1":
        return {"status": "DISABLED"}
    if priority_checker():
        return {"status": "LIVE_CRASH_PRIORITY"}
    try:
        condensed = _compact(bars[-96:])
        return _gemini({
            "latest_closed_at_utc": datetime.fromtimestamp(
                (bars[-1][0] + FIFTEEN) / 1000, timezone.utc).isoformat(),
            "last_24h_15m_candles": condensed,
            "anchor_price": bars[-96][4],
            "range_detection": snapshot,
            "reference_patterns": (reference.get("gemini") or {}).get("analysis", {}),
            "historical_statistics": reference.get("historical_patterns", {})
        }, "Judge if a breakout is likely to be UP, DOWN or UNCERTAIN from "
           "pre-breakout candles only. Examine wick/body and volume changes, "
           "repeated range tests, traps and invalidation. JSON keys: "
           "direction (UP/DOWN/UNCERTAIN), evidence (array), risks (array), "
           "reason (string). This is observation only; no order instruction.")
    except Exception as exc:
        LOG.warning("[BTC RANGE GEMINI HOURLY] %s", type(exc).__name__)
        return {"status": "ERROR", "error": type(exc).__name__}


def step(records, hour_ms, priority_checker, hourly_bar_record):
    """Called by BTC direction observer after its own new hourly record exists."""
    candles = _fetch_15m(hour_ms, RECENT_BARS, priority_checker)
    changed = evaluate_positions(records, candles)
    current = hourly_bar_record
    if "range_observation" in current:
        return changed
    # If reference is unavailable, paper range is still calculated with a
    # visible evidence warning. No invented Gemini verdict.
    reference = ensure_reference(hour_ms, priority_checker)
    snapshot = detect_range(candles, current["indicators"]["adx_1h"])
    snapshot["reference_created_ms"] = reference["created_ms"]
    snapshot["reference_gemini_status"] = reference.get("gemini", {}).get("status")
    snapshot["ai"] = _hourly_ai(candles, snapshot, reference, priority_checker)
    current["range_observation"] = snapshot
    changed.add(current["timestamp_paris"][:10])
    active = any(r.get("range_paper", {}).get("status") in ("PENDING", "OPEN")
                 for r in records.values())
    if snapshot.get("candidate") and not active:
        plan = dict(snapshot["candidate"])
        plan.update({"created_ms": hour_ms, "status": "PENDING",
                     "model": "btc_range_paper_v1", "fees_roundtrip_pct": FEE_ROUNDTRIP_PCT,
                     "source": "CLOSED_15M_AND_1H", "last_checked_open_ms": hour_ms - 1})
        current["range_paper"] = plan
        current["range_observation"]["virtual_action"] = "PAPER_LIMIT_BUY"
    else:
        current["range_observation"]["virtual_action"] = "WAIT"
    return changed
