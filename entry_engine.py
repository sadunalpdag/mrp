"""
entry_engine.py — Professional-grade breakout entry engine.

This module is called AFTER a breakout has already been detected and confirmed
(e.g., 2 closed 15m candles above swing high for LONG, or below swing low for SHORT).

It is responsible ONLY for determining the safest, highest-probability entry point.
It does NOT open trades directly. It returns a structured entry decision dict.

Public API
----------
evaluate_breakout_entry(klines, direction, breakout_level, *, ...)
    Main entry point. Returns a decision dict.

Helper functions (also importable individually)
-----------------------------------------------
detect_retest(...)
detect_reaction(...)
is_overextended(...)
calculate_entry_zone(...)
calculate_stop_loss(...)
calculate_take_profit(...)
evaluate_entry_quality(...)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Inline helpers so this module has zero runtime dependencies on ema.py.
# These are intentionally self-contained.
# ---------------------------------------------------------------------------

def _atr(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Wilder's ATR."""
    tr = []
    for i in range(len(highs)):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
    if len(tr) < period:
        return [tr[-1]] * len(tr) if tr else [0.0]
    seed = sum(tr[:period]) / period
    result = [0.0] * (period - 1) + [seed]
    for i in range(period, len(tr)):
        result.append((result[-1] * (period - 1) + tr[i]) / period)
    return result


def _fibonacci_retracement(swing_low: float, swing_high: float) -> dict:
    """
    Return key Fibonacci retracement levels for a bullish impulse leg.

    For LONG: impulse went from swing_low → swing_high.
    For SHORT callers: pass (swing_high, swing_low) to get the UP-leg levels,
    then interpret them in reverse (they will be below the swing_high).
    """
    diff = swing_high - swing_low
    return {
        "0.0":   swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.500": swing_high - 0.500 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0":   swing_low,
    }


# ---------------------------------------------------------------------------
# Configuration defaults (callers can override via kwargs)
# ---------------------------------------------------------------------------

DEFAULT_OVEREXTENDED_ATR_MULT    = 2.0   # max ATR distance from breakout level
DEFAULT_OVEREXTENDED_PCT         = 0.03  # max % distance from breakout level (3 %)
DEFAULT_RETEST_TOLERANCE_PCT     = 0.003 # 0.3 % — how close to level counts as retest
DEFAULT_RETEST_LOOKBACK          = 8     # candles to look back for retest
DEFAULT_MIN_BODY_RATIO           = 0.40  # candle body / total range
DEFAULT_MAX_WICK_BODY_RATIO      = 2.0   # (upper+lower wick) / body
DEFAULT_FAKE_BREAKOUT_LOOKBACK   = 3     # candles after break to detect quick close-back
DEFAULT_RR_TP1                   = 2.0   # risk : reward for TP1
DEFAULT_RR_TP2                   = 3.0   # risk : reward for TP2


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def detect_retest(
    highs: list,
    lows: list,
    closes: list,
    level: float,
    direction: str,
    tolerance_pct: float = DEFAULT_RETEST_TOLERANCE_PCT,
    lookback: int = DEFAULT_RETEST_LOOKBACK,
) -> Tuple[bool, Optional[int], Optional[float]]:
    """
    Detect whether price has returned (retested) the broken level.

    For LONG: price broke above `level`; a retest means price pulled back and
    touched/approached the level from above.
    For SHORT: price broke below `level`; a retest means price bounced back up
    to the level from below.

    Parameters
    ----------
    highs, lows, closes : price arrays (most recent = last element)
    level               : the broken structural level (swing high for LONG,
                          swing low for SHORT)
    direction           : "LONG" or "SHORT"
    tolerance_pct       : how close to level counts as retest (default 0.3 %)
    lookback            : how many recent candles to examine

    Returns
    -------
    (retested: bool, candle_index: int|None, retest_price: float|None)
        candle_index is negative (e.g. -2 means two bars before last)
    """
    n = len(closes)
    if n < 3:
        return False, None, None

    window = min(lookback, n)

    if direction == "LONG":
        # Retest: low must reach AT or BELOW the broken level (wicked into support),
        # and close must hold AT or ABOVE the level (support held).
        for i in range(n - 1, n - window - 1, -1):
            if lows[i] <= level and closes[i] >= level:
                return True, i - n, lows[i]
    else:  # SHORT
        # Retest: high must reach AT or ABOVE the broken level (wicked into resistance),
        # and close must hold AT or BELOW the level (resistance held).
        for i in range(n - 1, n - window - 1, -1):
            if highs[i] >= level and closes[i] <= level:
                return True, i - n, highs[i]

    return False, None, None


def detect_reaction(
    opens: list,
    highs: list,
    lows: list,
    closes: list,
    retest_idx: int,
    direction: str,
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
    max_wick_body_ratio: float = DEFAULT_MAX_WICK_BODY_RATIO,
) -> Tuple[bool, str]:
    """
    Detect a strong directional reaction candle after a retest.

    For LONG: look for a bullish close (close > open) with a strong body
    in the candle immediately following the retest candle.
    For SHORT: look for a bearish close (close < open) with a strong body.

    Parameters
    ----------
    opens, highs, lows, closes : price arrays
    retest_idx  : negative index of the retest candle (e.g. -3)
    direction   : "LONG" or "SHORT"
    min_body_ratio       : minimum body / total range
    max_wick_body_ratio  : maximum wick-to-body ratio

    Returns
    -------
    (reacted: bool, reason: str)
    """
    # The reaction candle is the one after the retest
    reaction_idx = retest_idx + 1
    if reaction_idx >= 0:
        # retest_idx was the last bar or newer — use the last bar as reaction
        reaction_idx = -1

    try:
        o = float(opens[reaction_idx])
        h = float(highs[reaction_idx])
        l = float(lows[reaction_idx])
        c = float(closes[reaction_idx])
    except (IndexError, TypeError, ValueError):
        return False, "Candle data unavailable"

    total_range = h - l
    if total_range <= 0:
        return False, "Zero-range candle"

    body = abs(c - o)
    body_ratio = body / total_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    wick_body_ratio = (upper_wick + lower_wick) / body if body > 0 else float("inf")

    candle_quality_ok = (
        body_ratio >= min_body_ratio and wick_body_ratio <= max_wick_body_ratio
    )

    if direction == "LONG":
        bullish = c > o
        if bullish and candle_quality_ok:
            return True, f"Bullish reaction (body={body_ratio:.0%}, wick/body={wick_body_ratio:.1f})"
        if not bullish:
            return False, "Bearish candle after retest — no bullish reaction"
        return False, f"Weak candle quality (body={body_ratio:.0%}, wick/body={wick_body_ratio:.1f})"
    else:  # SHORT
        bearish = c < o
        if bearish and candle_quality_ok:
            return True, f"Bearish reaction (body={body_ratio:.0%}, wick/body={wick_body_ratio:.1f})"
        if not bearish:
            return False, "Bullish candle after retest — no bearish reaction"
        return False, f"Weak candle quality (body={body_ratio:.0%}, wick/body={wick_body_ratio:.1f})"


def is_overextended(
    current_price: float,
    breakout_level: float,
    direction: str,
    atr_value: float,
    atr_multiplier: float = DEFAULT_OVEREXTENDED_ATR_MULT,
    max_pct: float = DEFAULT_OVEREXTENDED_PCT,
) -> Tuple[bool, float]:
    """
    Determine whether price has moved too far from the breakout level.

    Two checks (either triggers overextension):
    1. ATR-based: distance > atr_value * atr_multiplier
    2. Percentage-based: distance > max_pct * breakout_level

    Parameters
    ----------
    current_price   : latest close
    breakout_level  : structural level that was broken
    direction       : "LONG" or "SHORT"
    atr_value       : current ATR value
    atr_multiplier  : how many ATRs before declaring overextension
    max_pct         : percentage distance threshold

    Returns
    -------
    (overextended: bool, distance_pct: float)
    """
    if breakout_level <= 0:
        return False, 0.0

    if direction == "LONG":
        distance = current_price - breakout_level
    else:
        distance = breakout_level - current_price

    distance_pct = distance / breakout_level

    atr_threshold = atr_value * atr_multiplier if atr_value > 0 else float("inf")

    overextended = distance > atr_threshold or distance_pct > max_pct
    return overextended, round(distance_pct, 4)


def calculate_entry_zone(
    breakout_level: float,
    swing_low: float,
    swing_high: float,
    direction: str,
) -> Tuple[float, float, dict]:
    """
    Calculate the primary and secondary (Fibonacci) entry zones.

    Primary zone  = around the broken level itself (±0.1 %).
    Secondary zone = Fibonacci 38.2 %–50 % retracement of the prior impulse.

    The returned entry_zone_low / entry_zone_high encompasses BOTH zones so
    callers can use it as a single unified acceptance range.

    Parameters
    ----------
    breakout_level : structural level that was broken (swing high for LONG,
                     swing low for SHORT)
    swing_low      : start of the impulse leg
    swing_high     : end of the impulse leg
    direction      : "LONG" or "SHORT"

    Returns
    -------
    (entry_zone_low, entry_zone_high, detail_dict)
        detail_dict contains fib_levels and the individual zones
    """
    fib = _fibonacci_retracement(swing_low, swing_high)

    if direction == "LONG":
        # Primary: just below/at the broken swing high (support-turned-resistance retested)
        primary_low  = breakout_level * 0.999
        primary_high = breakout_level * 1.001
        # Secondary: 38.2 %–50 % Fibonacci retracement pullback
        secondary_low  = fib["0.500"]
        secondary_high = fib["0.382"]
        # Unified zone
        entry_zone_low  = min(primary_low, secondary_low)
        entry_zone_high = max(primary_high, secondary_high)
    else:  # SHORT
        # Primary: just above/at the broken swing low (support retested as resistance)
        primary_low  = breakout_level * 0.999
        primary_high = breakout_level * 1.001
        # Secondary: 38.2 %–50 % bounce from the swing_low toward swing_high.
        # For a bearish impulse (swing_high → swing_low), a retracement goes UP:
        #   38.2% bounce = swing_low + 0.382 * (swing_high - swing_low)
        #   50.0% bounce = swing_low + 0.500 * (swing_high - swing_low)
        diff = swing_high - swing_low
        secondary_low  = swing_low + 0.382 * diff
        secondary_high = swing_low + 0.500 * diff
        entry_zone_low  = min(primary_low, secondary_low)
        entry_zone_high = max(primary_high, secondary_high)

    detail = {
        "fib_levels": {k: round(v, 6) for k, v in fib.items()},
        "primary_zone":   [round(primary_low, 6), round(primary_high, 6)],
        "secondary_zone": [round(secondary_low, 6), round(secondary_high, 6)],
    }

    return round(entry_zone_low, 6), round(entry_zone_high, 6), detail


def calculate_stop_loss(
    breakout_level: float,
    retest_low: Optional[float],
    retest_high: Optional[float],
    direction: str,
    atr_value: float,
    atr_buffer_mult: float = 0.3,
) -> float:
    """
    Calculate stop loss level.

    For LONG:
        Use retest_low if available; otherwise fall back to breakout_level − ATR buffer.
    For SHORT:
        Use retest_high if available; otherwise fall back to breakout_level + ATR buffer.

    Parameters
    ----------
    breakout_level  : structural level that was broken
    retest_low      : low of the retest candle (LONG) or None
    retest_high     : high of the retest candle (SHORT) or None
    direction       : "LONG" or "SHORT"
    atr_value       : current ATR value
    atr_buffer_mult : fraction of ATR to add as buffer beyond the structural level

    Returns
    -------
    stop_loss price (float)
    """
    buffer = atr_value * atr_buffer_mult

    if direction == "LONG":
        if retest_low is not None:
            sl = retest_low - buffer
        else:
            sl = breakout_level - buffer
    else:
        if retest_high is not None:
            sl = retest_high + buffer
        else:
            sl = breakout_level + buffer

    return round(sl, 6)


def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    direction: str,
    next_structure_level: Optional[float] = None,
    rr_tp1: float = DEFAULT_RR_TP1,
    rr_tp2: float = DEFAULT_RR_TP2,
) -> Tuple[float, float, float]:
    """
    Calculate TP1, TP2, and risk-reward ratio.

    Preference order:
    1. If next_structure_level is provided and yields RR ≥ rr_tp1, use it as TP1
       and project TP2 via rr_tp2.
    2. Otherwise fall back to pure RR-based targets.

    Parameters
    ----------
    entry_price           : intended entry price
    stop_loss             : calculated stop-loss price
    direction             : "LONG" or "SHORT"
    next_structure_level  : nearest swing high (LONG) or swing low (SHORT), optional
    rr_tp1, rr_tp2        : minimum risk:reward multiples for TP1 / TP2

    Returns
    -------
    (tp1, tp2, risk_reward_tp1)
    """
    risk = abs(entry_price - stop_loss)
    if risk <= 0:
        risk = entry_price * 0.005  # fallback: 0.5 % of price

    if direction == "LONG":
        tp1_rr = entry_price + risk * rr_tp1
        tp2_rr = entry_price + risk * rr_tp2
        if next_structure_level and next_structure_level > entry_price:
            structure_rr = (next_structure_level - entry_price) / risk
            if structure_rr >= rr_tp1:
                tp1 = next_structure_level
                tp2 = entry_price + risk * rr_tp2
            else:
                tp1, tp2 = tp1_rr, tp2_rr
        else:
            tp1, tp2 = tp1_rr, tp2_rr
    else:  # SHORT
        tp1_rr = entry_price - risk * rr_tp1
        tp2_rr = entry_price - risk * rr_tp2
        if next_structure_level and next_structure_level < entry_price:
            structure_rr = (entry_price - next_structure_level) / risk
            if structure_rr >= rr_tp1:
                tp1 = next_structure_level
                tp2 = entry_price - risk * rr_tp2
            else:
                tp1, tp2 = tp1_rr, tp2_rr
        else:
            tp1, tp2 = tp1_rr, tp2_rr

    rr_achieved = abs(tp1 - entry_price) / risk if risk > 0 else 0
    return round(tp1, 6), round(tp2, 6), round(rr_achieved, 2)


def evaluate_entry_quality(
    opens: list,
    highs: list,
    lows: list,
    closes: list,
    idx: int = -1,
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
    max_wick_body_ratio: float = DEFAULT_MAX_WICK_BODY_RATIO,
) -> Tuple[bool, float, float, str]:
    """
    Evaluate whether a specific candle meets the minimum quality bar for entry.

    Filters:
    - body_ratio   = |close − open| / (high − low)  ≥ min_body_ratio
    - wick_body_ratio = (upper_wick + lower_wick) / body  ≤ max_wick_body_ratio
    - Doji / spinning-top candles are rejected.

    Parameters
    ----------
    opens, highs, lows, closes : price arrays
    idx                : candle index to evaluate (default -1 = last)
    min_body_ratio     : minimum acceptable body ratio
    max_wick_body_ratio: maximum acceptable wick-to-body ratio

    Returns
    -------
    (passes, body_ratio, wick_body_ratio, reason_str)
    """
    try:
        o = float(opens[idx])
        h = float(highs[idx])
        l = float(lows[idx])
        c = float(closes[idx])
    except (IndexError, TypeError, ValueError):
        return False, 0.0, float("inf"), "Candle data unavailable"

    total_range = h - l
    if total_range <= 0:
        return False, 0.0, float("inf"), "Zero-range candle"

    body = abs(c - o)
    body_ratio = body / total_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    wick_body_ratio = (upper_wick + lower_wick) / body if body > 0 else float("inf")

    if body_ratio < min_body_ratio:
        reason = f"Body too small (body={body_ratio:.0%} < {min_body_ratio:.0%})"
        return False, round(body_ratio, 3), round(wick_body_ratio, 3), reason
    if wick_body_ratio > max_wick_body_ratio:
        reason = f"Wicks dominant (wick/body={wick_body_ratio:.1f} > {max_wick_body_ratio:.1f})"
        return False, round(body_ratio, 3), round(wick_body_ratio, 3), reason

    return (
        True,
        round(body_ratio, 3),
        round(wick_body_ratio, 3),
        f"Strong candle (body={body_ratio:.0%}, wick/body={wick_body_ratio:.1f})",
    )


def _detect_fake_breakout(
    closes: list,
    highs: list,
    lows: list,
    breakout_level: float,
    direction: str,
    lookback: int = DEFAULT_FAKE_BREAKOUT_LOOKBACK,
) -> bool:
    """
    Return True if a fake breakout is detected.

    A fake breakout occurs when price closes beyond the breakout level but
    then closes BACK inside (below for LONG, above for SHORT) within `lookback`
    candles, indicating the breakout had no follow-through.
    """
    n = len(closes)
    if n < lookback + 1:
        return False

    window = closes[-lookback:]

    # Find first candle that closed beyond the level
    broke_idx = None
    for i, c in enumerate(window):
        if direction == "LONG" and c > breakout_level:
            broke_idx = i
            break
        if direction == "SHORT" and c < breakout_level:
            broke_idx = i
            break

    if broke_idx is None or broke_idx >= len(window) - 1:
        return False

    # After the breakout candle, did price close back inside?
    for c in window[broke_idx + 1:]:
        if direction == "LONG" and c < breakout_level:
            return True
        if direction == "SHORT" and c > breakout_level:
            return True

    return False


def _score_confidence(
    retest_confirmed: bool,
    reaction_confirmed: bool,
    candle_quality_ok: bool,
    in_entry_zone: bool,
    volume_spike: bool,
    overextended: bool,
    fake_breakout: bool,
    entry_type: str,
) -> int:
    """Compute a 0-100 confidence score based on entry conditions."""
    score = 40  # base score: breakout already confirmed externally

    if fake_breakout or overextended:
        return 0

    if retest_confirmed:
        score += 20
    if reaction_confirmed:
        score += 15
    if candle_quality_ok:
        score += 10
    if in_entry_zone:
        score += 10
    if volume_spike:
        score += 10
    if entry_type == "RETEST_ENTRY":
        score += 5
    elif entry_type == "BREAKOUT_CHASE":
        score -= 15

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_breakout_entry(
    klines: list,
    direction: str,
    breakout_level: float,
    *,
    swing_low: Optional[float] = None,
    swing_high: Optional[float] = None,
    next_structure_level: Optional[float] = None,
    overextended_atr_mult: float = DEFAULT_OVEREXTENDED_ATR_MULT,
    overextended_pct: float = DEFAULT_OVEREXTENDED_PCT,
    retest_tolerance_pct: float = DEFAULT_RETEST_TOLERANCE_PCT,
    retest_lookback: int = DEFAULT_RETEST_LOOKBACK,
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
    max_wick_body_ratio: float = DEFAULT_MAX_WICK_BODY_RATIO,
    fake_breakout_lookback: int = DEFAULT_FAKE_BREAKOUT_LOOKBACK,
    rr_tp1: float = DEFAULT_RR_TP1,
    rr_tp2: float = DEFAULT_RR_TP2,
    volume_spike: bool = False,
) -> dict:
    """
    Determine the best entry point after a confirmed breakout.

    This function assumes the breakout has ALREADY been confirmed externally
    (e.g., 2 closed 15m candles above swing high for LONG).
    It only decides WHETHER and HOW to enter.

    Parameters
    ----------
    klines          : list of klines [[time, open, high, low, close, volume, ...], ...]
                      ordered oldest-first; must include at least 20 bars
    direction       : "LONG" or "SHORT"
    breakout_level  : the structural level that was broken
                      (swing high for LONG, swing low for SHORT)
    swing_low       : start of the prior impulse leg (used for Fibonacci zones)
                      auto-detected from klines if not provided
    swing_high      : end of the prior impulse leg
                      auto-detected from klines if not provided
    next_structure_level : nearest opposing structure level for TP1 targeting
    overextended_atr_mult : ATR multiplier threshold for overextension check
    overextended_pct      : percentage threshold for overextension check
    retest_tolerance_pct  : how close to level counts as a retest
    retest_lookback       : candles to look back for retest
    min_body_ratio        : minimum body/range ratio for candle quality
    max_wick_body_ratio   : max wick/body ratio for candle quality
    fake_breakout_lookback: candles to look back for fake-breakout detection
    rr_tp1, rr_tp2        : risk:reward multiples for take-profit levels
    volume_spike          : whether a volume spike was detected at breakout

    Returns
    -------
    dict with keys:
        signal          : "ENTRY_READY" | "WAIT_FOR_RETEST" | "FAKE_BREAKOUT" |
                          "OVEREXTENDED_NO_ENTRY" | "NO_ENTRY" | "INVALID_DATA"
        entry_type      : "RETEST_ENTRY" | "PULLBACK_ENTRY" | "BREAKOUT_CHASE" | None
        entry_zone      : [entry_zone_low, entry_zone_high]
        best_entry      : float  (mid-point of tightest entry zone)
        stop_loss       : float
        take_profit     : [tp1, tp2]
        risk_reward     : float
        reason          : str
        confidence      : int 0-100
    """
    _empty = {
        "signal": "INVALID_DATA",
        "entry_type": None,
        "entry_zone": None,
        "best_entry": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": None,
        "reason": "",
        "confidence": 0,
    }

    # ── Validate inputs ────────────────────────────────────────────────
    if not klines or len(klines) < 20:
        _empty["reason"] = "Insufficient kline data (need ≥ 20 bars)"
        return _empty

    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        _empty["reason"] = f"Invalid direction: {direction}"
        return _empty

    try:
        opens  = [float(k[1]) for k in klines]
        highs  = [float(k[2]) for k in klines]
        lows   = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
    except (IndexError, TypeError, ValueError) as exc:
        _empty["reason"] = f"Kline parse error: {exc}"
        return _empty

    current_close = closes[-1]

    # ── ATR ────────────────────────────────────────────────────────────
    atr_vals = _atr(highs, lows, closes)
    atr_value = atr_vals[-1] if atr_vals else current_close * 0.005

    # ── Auto-detect swing high / low if not supplied ──────────────────
    if swing_low is None or swing_high is None:
        lookback_auto = min(50, len(closes))
        _highs = highs[-lookback_auto:]
        _lows  = lows[-lookback_auto:]
        swing_high = swing_high if swing_high is not None else max(_highs)
        swing_low  = swing_low  if swing_low  is not None else min(_lows)

    # ── Fake-breakout check ────────────────────────────────────────────
    fake = _detect_fake_breakout(
        closes, highs, lows, breakout_level, direction, fake_breakout_lookback
    )
    if fake:
        return {
            "signal":      "FAKE_BREAKOUT",
            "entry_type":  None,
            "entry_zone":  None,
            "best_entry":  None,
            "stop_loss":   None,
            "take_profit": None,
            "risk_reward": None,
            "reason":      "Price closed back below/above breakout level — fake breakout",
            "confidence":  0,
        }

    # ── Overextension check ────────────────────────────────────────────
    overext, dist_pct = is_overextended(
        current_close, breakout_level, direction, atr_value,
        overextended_atr_mult, overextended_pct
    )
    if overext:
        return {
            "signal":      "OVEREXTENDED_NO_ENTRY",
            "entry_type":  None,
            "entry_zone":  None,
            "best_entry":  None,
            "stop_loss":   None,
            "take_profit": None,
            "risk_reward": None,
            "reason": (
                f"Price is {dist_pct:.1%} away from breakout level — "
                "wait for pullback before entering"
            ),
            "confidence":  0,
        }

    # ── Entry zone ────────────────────────────────────────────────────
    entry_zone_low, entry_zone_high, zone_detail = calculate_entry_zone(
        breakout_level, swing_low, swing_high, direction
    )

    # ── Retest detection ───────────────────────────────────────────────
    retest_found, retest_idx, retest_price = detect_retest(
        highs, lows, closes, breakout_level, direction,
        retest_tolerance_pct, retest_lookback
    )

    # ── Reaction detection (requires retest) ──────────────────────────
    reaction_ok = False
    reaction_reason = "No retest detected"
    if retest_found and retest_idx is not None:
        reaction_ok, reaction_reason = detect_reaction(
            opens, highs, lows, closes,
            retest_idx, direction,
            min_body_ratio, max_wick_body_ratio
        )

    # ── Candle quality of latest bar ──────────────────────────────────
    cq_ok, body_ratio, wick_ratio, cq_reason = evaluate_entry_quality(
        opens, highs, lows, closes,
        idx=-1, min_body_ratio=min_body_ratio, max_wick_body_ratio=max_wick_body_ratio
    )

    # ── Price position relative to entry zone ─────────────────────────
    in_entry_zone = entry_zone_low <= current_close <= entry_zone_high

    # ── Determine entry type and signal ───────────────────────────────
    reasons = []

    if retest_found and reaction_ok and cq_ok:
        # Best case: retest + reaction + quality all confirmed
        signal     = "ENTRY_READY"
        entry_type = "RETEST_ENTRY"
        reasons.append("Retest held, breakout confirmed, strong reaction")
        reasons.append(reaction_reason)
        reasons.append(cq_reason)
    elif retest_found and cq_ok:
        # Retest present, current candle is strong — acceptable as pullback entry
        signal     = "ENTRY_READY"
        entry_type = "PULLBACK_ENTRY"
        reasons.append("Price pulled back to breakout level")
        reasons.append(reaction_reason)
        reasons.append(cq_reason)
    elif in_entry_zone and volume_spike and cq_ok:
        # Price is in the zone with momentum — breakout chase (lower quality)
        signal     = "ENTRY_READY"
        entry_type = "BREAKOUT_CHASE"
        reasons.append("Strong momentum breakout with volume — chase acceptable")
        reasons.append(cq_reason)
    elif retest_found and (reaction_ok or in_entry_zone):
        # Retest/reaction seen but current candle is weak — wait for clean bar
        signal     = "WAIT_FOR_RETEST"
        entry_type = None
        reasons.append("Retest/reaction detected but current candle quality insufficient")
        reasons.append(cq_reason)
    elif retest_found:
        # Retest detected but no reaction yet and price outside entry zone
        signal     = "WAIT_FOR_RETEST"
        entry_type = None
        reasons.append("Retest detected — waiting for reaction confirmation")
        reasons.append(reaction_reason)
    elif in_entry_zone:
        signal     = "WAIT_FOR_RETEST"
        entry_type = None
        reasons.append("Price in entry zone but no confirmed retest / reaction yet")
    else:
        signal     = "NO_ENTRY"
        entry_type = None
        reasons.append("Price not in entry zone, no retest, conditions not met")

    # ── Retest anchor for stop loss ────────────────────────────────────
    retest_low_anchor  = retest_price if (retest_found and direction == "LONG")  else None
    retest_high_anchor = retest_price if (retest_found and direction == "SHORT") else None

    # ── Stop loss ─────────────────────────────────────────────────────
    stop_loss = calculate_stop_loss(
        breakout_level,
        retest_low_anchor,
        retest_high_anchor,
        direction,
        atr_value,
    )

    # ── Best entry price ──────────────────────────────────────────────
    # For LONG: enter as close to the breakout level as possible (cheapest price).
    # For SHORT: enter as close to the breakout level as possible (highest price).
    best_entry = round(breakout_level, 6)

    # ── Take profit ───────────────────────────────────────────────────
    tp1, tp2, rr = calculate_take_profit(
        best_entry, stop_loss, direction,
        next_structure_level=next_structure_level,
        rr_tp1=rr_tp1, rr_tp2=rr_tp2
    )

    # ── Confidence score ──────────────────────────────────────────────
    confidence = _score_confidence(
        retest_confirmed  = retest_found,
        reaction_confirmed= reaction_ok,
        candle_quality_ok = cq_ok,
        in_entry_zone     = in_entry_zone,
        volume_spike      = volume_spike,
        overextended      = overext,
        fake_breakout     = fake,
        entry_type        = entry_type or "",
    )

    return {
        "signal":      signal,
        "entry_type":  entry_type,
        "entry_zone":  [entry_zone_low, entry_zone_high],
        "best_entry":  best_entry,
        "stop_loss":   stop_loss,
        "take_profit": [tp1, tp2],
        "risk_reward": rr,
        "reason":      " | ".join(reasons),
        "confidence":  confidence,
        # ── diagnostic extras ──────────────────────────────────────────
        "_debug": {
            "current_close":   round(current_close, 6),
            "breakout_level":  round(breakout_level, 6),
            "atr":             round(atr_value, 6),
            "retest_found":    retest_found,
            "retest_price":    round(retest_price, 6) if retest_price else None,
            "reaction_ok":     reaction_ok,
            "candle_quality":  cq_ok,
            "in_entry_zone":   in_entry_zone,
            "overextended":    overext,
            "fake_breakout":   fake,
            "swing_high":      round(swing_high, 6),
            "swing_low":       round(swing_low, 6),
            "zone_detail":     zone_detail,
        },
    }
