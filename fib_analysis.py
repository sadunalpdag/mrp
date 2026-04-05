"""
fib_analysis.py — Signal-only Fibonacci breakout / fakeout analysis module.

Designed to plug into the existing Binance futures trading bot (ema.py).
Does NOT place orders, does NOT modify any existing strategy logic.

Public API
----------
analyze_fib_breakout_fakeout(klines, higher_tf_klines=None) -> dict

Possible signal values:
    LONG_CONTINUATION       – bullish impulse pullback into fib support
    SHORT_CONTINUATION      – bearish impulse bounce into fib resistance
    BULLISH_FAKE_BREAKOUT   – wick below fib support, closed back above (long setup)
    BEARISH_FAKE_BREAKOUT   – wick above fib resistance, closed back below (short setup)
    NO_TRADE                – conditions not met or confidence too low
    INVALID                 – insufficient data

Klines format expected (same as Binance API and ema.py):
    [[time_ms, open, high, low, close, volume, ...], ...]
"""

from ema import (
    fibonacci_levels,
    detect_liquidity_sweep,
    detect_volume_spike,
    check_ema_trend_alignment,
    detect_market_state,
    ema,
    atr_like,
)


# ---------------------------------------------------------------------------
# 1. Pivot detection
# ---------------------------------------------------------------------------

def detect_pivot_highs(highs, lows, closes, left=3, right=3):
    """
    Return confirmed pivot highs as a list of (index, price) tuples.

    A pivot high at index i is confirmed when:
      - highs[i] >= all highs in the `left` bars before it
      - highs[i] >= all highs in the `right` bars after it

    Using confirmed pivots (not just raw max) prevents chasing noise and
    ensures the swing was actually respected by subsequent price action.
    """
    pivots = []
    for i in range(left, len(highs) - right):
        is_pivot = all(highs[i] >= highs[i - j] for j in range(1, left + 1)) and \
                   all(highs[i] >= highs[i + j] for j in range(1, right + 1))
        if is_pivot:
            pivots.append((i, highs[i]))
    return pivots


def detect_pivot_lows(highs, lows, closes, left=3, right=3):
    """
    Return confirmed pivot lows as a list of (index, price) tuples.

    A pivot low at index i is confirmed when:
      - lows[i] <= all lows in the `left` bars before it
      - lows[i] <= all lows in the `right` bars after it
    """
    pivots = []
    for i in range(left, len(lows) - right):
        is_pivot = all(lows[i] <= lows[i - j] for j in range(1, left + 1)) and \
                   all(lows[i] <= lows[i + j] for j in range(1, right + 1))
        if is_pivot:
            pivots.append((i, lows[i]))
    return pivots


# ---------------------------------------------------------------------------
# 2. Impulse swing detection
# ---------------------------------------------------------------------------

def detect_impulse_swing(highs, lows, closes, min_move_pct=2.0, left=3, right=3):
    """
    Identify the latest valid impulse leg using confirmed pivot highs and lows.

    An impulse leg is a directional move of at least `min_move_pct` percent
    between the most recent confirmed pivot high and the most recent confirmed
    pivot low.  The direction is determined by which pivot is more recent.

    Returns a dict or None if no valid impulse is found:
        {
            "direction":      "UP" | "DOWN",
            "swing_high":     float,
            "swing_low":      float,
            "swing_high_idx": int,
            "swing_low_idx":  int,
            "move_pct":       float
        }
    """
    pivot_highs = detect_pivot_highs(highs, lows, closes, left, right)
    pivot_lows  = detect_pivot_lows(highs, lows, closes, left, right)

    if not pivot_highs or not pivot_lows:
        return None

    last_ph_idx, last_ph_price = pivot_highs[-1]
    last_pl_idx, last_pl_price = pivot_lows[-1]

    if last_pl_price <= 0:
        return None

    move_pct = abs(last_ph_price - last_pl_price) / last_pl_price * 100

    if move_pct < min_move_pct:
        return None

    if last_ph_idx > last_pl_idx:
        # Swing low formed first, then swing high → bullish impulse
        direction = "UP"
    else:
        # Swing high formed first, then swing low → bearish impulse
        direction = "DOWN"

    return {
        "direction":      direction,
        "swing_high":     last_ph_price,
        "swing_low":      last_pl_price,
        "swing_high_idx": last_ph_idx,
        "swing_low_idx":  last_pl_idx,
        "move_pct":       round(move_pct, 2),
    }


# ---------------------------------------------------------------------------
# 3. Candle quality filter
# ---------------------------------------------------------------------------

def classify_candle_quality(open_p, high_p, low_p, close_p, min_body_ratio=0.4):
    """
    Analyse a single candle for breakout / reversal confirmation quality.

    Rules:
        - STRONG:   body_ratio >= 0.6  (decisive directional candle)
        - MODERATE: body_ratio >= min_body_ratio (acceptable confirmation)
        - WEAK:     body_ratio < min_body_ratio (doji / spinning-top; avoid)

    Returns a dict with per-candle metrics and a quality label.
    """
    total_range = high_p - low_p
    if total_range < 1e-12:
        return {
            "body_ratio":        0.0,
            "upper_wick_ratio":  0.0,
            "lower_wick_ratio":  0.0,
            "direction":         "NEUTRAL",
            "quality":           "WEAK",
        }

    body             = abs(close_p - open_p)
    upper_wick       = high_p - max(open_p, close_p)
    lower_wick       = min(open_p, close_p) - low_p

    body_ratio        = body / total_range
    upper_wick_ratio  = upper_wick / total_range
    lower_wick_ratio  = lower_wick / total_range
    direction         = "BULL" if close_p >= open_p else "BEAR"

    if body_ratio >= 0.6:
        quality = "STRONG"
    elif body_ratio >= min_body_ratio:
        quality = "MODERATE"
    else:
        quality = "WEAK"

    return {
        "body_ratio":        round(body_ratio, 3),
        "upper_wick_ratio":  round(upper_wick_ratio, 3),
        "lower_wick_ratio":  round(lower_wick_ratio, 3),
        "direction":         direction,
        "quality":           quality,
    }


# ---------------------------------------------------------------------------
# 4. Breakout type classification
# ---------------------------------------------------------------------------

def classify_breakout_type(
    current_open, current_high, current_low, current_close,
    prev_close, level, direction, has_volume_spike, vol_ratio
):
    """
    Classify a candle's interaction with a Fibonacci level.

    direction: "UP"  → we are checking for a breakout / fake-breakout above the level
               "DOWN"→ we are checking for a breakout / fake-breakout below the level

    Returns (breakout_type, candle_info) where breakout_type is one of:
        "REAL_BREAKOUT"   – closed beyond level with adequate body + (ideally) volume
        "FAKE_BREAKOUT"   – wick beyond level but closed back inside (liquidity sweep)
        "WEAK_BREAKOUT"   – closed beyond level but body is too small to trust
        "NO_BREAKOUT"     – candle did not interact meaningfully with the level
    """
    candle = classify_candle_quality(
        current_open, current_high, current_low, current_close
    )

    if direction == "UP":
        wick_above  = current_high > level
        close_above = current_close > level

        if not wick_above:
            return "NO_BREAKOUT", candle
        if wick_above and not close_above:
            # High swept above level but price closed back below → fake breakout
            return "FAKE_BREAKOUT", candle
        # Closed above the level
        if candle["quality"] in ("STRONG", "MODERATE"):
            return "REAL_BREAKOUT", candle
        return "WEAK_BREAKOUT", candle

    else:  # direction == "DOWN"
        wick_below  = current_low < level
        close_below = current_close < level

        if not wick_below:
            return "NO_BREAKOUT", candle
        if wick_below and not close_below:
            # Low swept below level but price closed back above → fake breakout
            return "FAKE_BREAKOUT", candle
        # Closed below the level
        if candle["quality"] in ("STRONG", "MODERATE"):
            return "REAL_BREAKOUT", candle
        return "WEAK_BREAKOUT", candle


# ---------------------------------------------------------------------------
# 5. Retest confirmation
# ---------------------------------------------------------------------------

def detect_retest(closes, highs, lows, level, direction, tolerance_pct=0.3, lookback=5):
    """
    Detect whether price has come back to retest a broken level within the last
    `lookback` bars (excluding the current bar).

    direction: "UP"  → we broke above the level and expect a pullback retest from above
               "DOWN"→ we broke below the level and expect a bounce retest from below

    A valid retest satisfies:
        - The bar's wick touched the level (within tolerance)
        - The bar's close stayed on the breakout side (not back inside)

    Returns (retest_confirmed: bool, bar_index: int | None).
    """
    tolerance = level * tolerance_pct / 100
    search_start = max(0, len(closes) - lookback - 1)

    if direction == "UP":
        for i in range(search_start, len(closes) - 1):
            # Low touched the level from above, closed at or above it
            if lows[i] <= level + tolerance and closes[i] >= level - tolerance:
                return True, i
    else:
        for i in range(search_start, len(closes) - 1):
            # High touched the level from below, closed at or below it
            if highs[i] >= level - tolerance and closes[i] <= level + tolerance:
                return True, i

    return False, None


# ---------------------------------------------------------------------------
# 6. Multi-timeframe context extractor
# ---------------------------------------------------------------------------

def get_higher_tf_context(higher_tf_klines):
    """
    Extract trend and key-level context from a higher timeframe candle series.

    Expected input format: same as lower TF klines
        [[time_ms, open, high, low, close, volume, ...], ...]

    Returns a dict:
        {
            "trend":          "UP" | "DOWN" | None,
            "market_state":   str,
            "key_level_high": float | None,   # most recent HTF pivot high
            "key_level_low":  float | None,   # most recent HTF pivot low
        }
    """
    if not higher_tf_klines or len(higher_tf_klines) < 50:
        return {
            "trend":          None,
            "market_state":   "UNKNOWN",
            "key_level_high": None,
            "key_level_low":  None,
        }

    htf_closes = [float(k[4]) for k in higher_tf_klines]
    htf_highs  = [float(k[2]) for k in higher_tf_klines]
    htf_lows   = [float(k[3]) for k in higher_tf_klines]

    trend        = check_ema_trend_alignment(htf_closes)
    market_state = detect_market_state(htf_closes, htf_highs, htf_lows)

    # Use looser pivot parameters on HTF (fewer bars available)
    htf_ph = detect_pivot_highs(htf_highs, htf_lows, htf_closes, left=2, right=2)
    htf_pl = detect_pivot_lows(htf_highs, htf_lows, htf_closes, left=2, right=2)

    key_high = htf_ph[-1][1] if htf_ph else None
    key_low  = htf_pl[-1][1] if htf_pl else None

    return {
        "trend":          trend,
        "market_state":   market_state,
        "key_level_high": key_high,
        "key_level_low":  key_low,
    }


# ---------------------------------------------------------------------------
# 7. Main analysis function
# ---------------------------------------------------------------------------

def analyze_fib_breakout_fakeout(klines, higher_tf_klines=None):
    """
    Signal-only Fibonacci breakout / fakeout analysis.

    Integrates:
        - Pivot-confirmed swing detection for accurate Fibonacci anchoring
        - Breakout vs fake-breakout classification (close, wick, body ratio)
        - Volume confirmation via detect_volume_spike()
        - Trend filter via check_ema_trend_alignment()
        - Market context via detect_market_state()
        - Candle quality filter (body ratio, wick/body imbalance)
        - Optional retest confirmation
        - Optional multi-timeframe context (higher_tf_klines)

    Args:
        klines:            List of OHLCV klines for the signal timeframe.
        higher_tf_klines:  Optional higher TF klines (e.g. 4H) for context.

    Returns:
        dict with keys:
            signal          – LONG_CONTINUATION | SHORT_CONTINUATION |
                              BULLISH_FAKE_BREAKOUT | BEARISH_FAKE_BREAKOUT |
                              NO_TRADE | INVALID
            confidence      – int 0–100
            reason          – str summary of contributing factors
            entry_zone      – [low, high] price range or None
            invalidation    – price level that would negate the setup or None
            tp_zone         – [near_tp, far_tp] or None
            fib_levels      – dict of Fibonacci levels keyed by ratio string
            market_state    – str from detect_market_state()
            trend_alignment – "UP" | "DOWN" | None
            impulse         – dict describing the detected impulse swing
            candle_quality  – dict from classify_candle_quality()
            volume_spike    – bool
            volume_ratio    – float
            htf_context     – dict from get_higher_tf_context()
    """

    # --- Guard: minimum data -----------------------------------------------
    if not klines or len(klines) < 50:
        return {
            "signal":          "INVALID",
            "confidence":      0,
            "reason":          "Insufficient kline data (minimum 50 bars required)",
            "entry_zone":      None,
            "invalidation":    None,
            "tp_zone":         None,
            "fib_levels":      None,
            "market_state":    "UNKNOWN",
            "trend_alignment": None,
            "impulse":         None,
            "candle_quality":  None,
            "volume_spike":    False,
            "volume_ratio":    1.0,
            "htf_context":     get_higher_tf_context(higher_tf_klines),
        }

    # --- Extract OHLCV arrays -----------------------------------------------
    opens  = [float(k[1]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    current_open  = opens[-1]
    current_high  = highs[-1]
    current_low   = lows[-1]
    current_close = closes[-1]
    prev_close    = closes[-2]

    # --- Market-level filters ------------------------------------------------
    market_state            = detect_market_state(closes, highs, lows)
    trend_align             = check_ema_trend_alignment(closes)
    has_volume_spike, vol_ratio = detect_volume_spike(klines)

    # --- Higher timeframe context --------------------------------------------
    htf_context    = get_higher_tf_context(higher_tf_klines)
    # Prefer HTF trend when available; fall back to current-TF trend
    effective_trend = htf_context["trend"] if htf_context["trend"] else trend_align

    # --- Impulse swing detection (pivot-confirmed) ---------------------------
    impulse = detect_impulse_swing(highs, lows, closes, min_move_pct=2.0)
    if not impulse:
        return {
            "signal":          "NO_TRADE",
            "confidence":      0,
            "reason":          "No valid impulse swing found; cannot anchor Fibonacci levels",
            "entry_zone":      None,
            "invalidation":    None,
            "tp_zone":         None,
            "fib_levels":      None,
            "market_state":    market_state,
            "trend_alignment": effective_trend,
            "impulse":         None,
            "candle_quality":  None,
            "volume_spike":    has_volume_spike,
            "volume_ratio":    round(vol_ratio, 3),
            "htf_context":     htf_context,
        }

    swing_high = impulse["swing_high"]
    swing_low  = impulse["swing_low"]

    # --- Fibonacci levels ----------------------------------------------------
    fib_lvls = fibonacci_levels(swing_high, swing_low)

    # --- ATR-based zone tolerance -------------------------------------------
    atr_vals    = atr_like(highs, lows, closes)
    current_atr = atr_vals[-1] if atr_vals else (current_close * 0.005)
    zone_tol    = current_atr * 0.5  # within half-ATR counts as "at" a level

    # --- Find which Fibonacci level price is currently interacting with ------
    key_fib_names  = ["0.236", "0.382", "0.500", "0.618", "0.786"]
    tested_level_name  = None
    tested_level_price = None

    for name in key_fib_names:
        price = fib_lvls[name]
        # Level is "tested" if wick or close is within half-ATR of it
        if (abs(current_close - price) <= zone_tol or
                abs(current_low - price) <= zone_tol or
                abs(current_high - price) <= zone_tol):
            tested_level_name  = name
            tested_level_price = price
            break  # Take the first (highest) matching level

    # --- Candle quality of the current bar -----------------------------------
    candle_info = classify_candle_quality(
        current_open, current_high, current_low, current_close
    )

    # --- Setup detection -----------------------------------------------------
    confidence = 0
    reasons    = []
    signal     = "NO_TRADE"
    entry_zone = invalidation = tp_zone = None

    if impulse["direction"] == "UP":
        # -----------------------------------------------------------------
        # BULLISH IMPULSE — two scenarios:
        #   (A) Pullback continuation long (price in 38.2–61.8% support zone)
        #   (B) Bullish fake breakout at fib support (wick below, close above)
        # -----------------------------------------------------------------
        in_pullback_zone = fib_lvls["0.618"] <= current_close <= fib_lvls["0.382"]
        sweep_dir, sweep_level = detect_liquidity_sweep(highs, lows, closes)

        if sweep_dir == "UP" and tested_level_price is not None:
            # --- Scenario B: Bullish liquidity sweep at fib support ----------
            bo_type, candle_info = classify_breakout_type(
                current_open, current_high, current_low, current_close,
                prev_close, tested_level_price, "DOWN",
                has_volume_spike, vol_ratio,
            )

            if bo_type == "FAKE_BREAKOUT":
                signal      = "BULLISH_FAKE_BREAKOUT"
                confidence += 35
                reasons.append(
                    f"Bullish liquidity sweep below Fibonacci {tested_level_name} level"
                )

                # Volume: an illiquid wick (low volume) strengthens fake-breakout read;
                # a volume spike on the close bar confirms the reversal push.
                if not has_volume_spike:
                    confidence += 15
                    reasons.append("Low-volume sweep confirms illiquid wick (not real selling)")
                elif vol_ratio >= 1.2:
                    confidence += 10
                    reasons.append("Volume spike on reversal bar")

                # Trend alignment
                if effective_trend == "UP":
                    confidence += 20
                    reasons.append("Trend aligned UP — continuation long favored")
                elif effective_trend == "DOWN":
                    confidence -= 15
                    reasons.append("Against DOWN trend — reduced conviction")

                # Market context: fakeouts are most reliable in range markets
                if market_state == "RANGE":
                    confidence += 10
                    reasons.append("Range market: fake-breakout setups are primary")
                elif market_state == "STRONG_TREND":
                    confidence += 5
                    reasons.append("Strong trend context supports continuation")

                # Candle quality on the reversal bar
                if candle_info["quality"] == "STRONG":
                    confidence += 15
                    reasons.append("Strong bullish reversal candle")
                elif candle_info["quality"] == "MODERATE":
                    confidence += 8
                    reasons.append("Moderate reversal candle")
                else:
                    confidence -= 10
                    reasons.append("Weak candle quality — wait for confirmation")

                # Retest of the swept level adds reliability
                retest_confirmed, _ = detect_retest(
                    closes, highs, lows, tested_level_price, "UP"
                )
                if retest_confirmed:
                    confidence += 10
                    reasons.append("Post-sweep level retest confirmed")

                entry_zone  = [round(tested_level_price, 6),
                                round(tested_level_price * 1.002, 6)]
                invalidation = round(sweep_level * 0.998, 6)
                tp_zone      = [round(fib_lvls["0.236"], 6),
                                round(fib_lvls["0.0"], 6)]

        elif in_pullback_zone and candle_info["direction"] == "BULL":
            # --- Scenario A: Pullback continuation long at fib support -------
            signal      = "LONG_CONTINUATION"
            confidence += 30
            reasons.append(
                "Price pulling back into Fibonacci support zone (38.2–61.8%) "
                "of bullish impulse"
            )

            if has_volume_spike:
                confidence += 15
                reasons.append("Volume spike on bounce from support")

            if effective_trend == "UP":
                confidence += 20
                reasons.append("Trend aligned UP")
            elif effective_trend == "DOWN":
                confidence -= 20
                reasons.append("Against DOWN trend — signal degraded")

            if market_state == "STRONG_TREND":
                confidence += 15
                reasons.append("Strong trend market favors Fibonacci continuation")
            elif market_state == "RANGE":
                confidence -= 5
                reasons.append("Range market reduces continuation reliability")

            if candle_info["quality"] == "STRONG":
                confidence += 15
                reasons.append("Strong bullish candle confirming bounce")
            elif candle_info["quality"] == "MODERATE":
                confidence += 8
                reasons.append("Moderate bullish candle")
            else:
                confidence -= 10
                reasons.append("Weak candle — wait for better confirmation")

            retest_confirmed, _ = detect_retest(
                closes, highs, lows, fib_lvls["0.500"], "UP"
            )
            if retest_confirmed:
                confidence += 10
                reasons.append("50% level retest confirmed")

            entry_zone   = [round(fib_lvls["0.618"], 6), round(fib_lvls["0.382"], 6)]
            invalidation = round(fib_lvls["0.786"], 6)
            tp_zone      = [round(fib_lvls["0.236"], 6), round(fib_lvls["0.0"], 6)]

    else:  # impulse["direction"] == "DOWN"
        # -----------------------------------------------------------------
        # BEARISH IMPULSE — two scenarios:
        #   (A) Short continuation (price bouncing into 38.2–61.8% resistance)
        #   (B) Bearish fake breakout at fib resistance (wick above, close below)
        # -----------------------------------------------------------------
        # fib_lvls["0.618"] < fib_lvls["0.382"] in price (higher ratio = lower price),
        # so the correct range check is [0.618_price, 0.382_price].
        in_bounce_zone = fib_lvls["0.618"] <= current_close <= fib_lvls["0.382"]
        sweep_dir, sweep_level = detect_liquidity_sweep(highs, lows, closes)

        if sweep_dir == "DOWN" and tested_level_price is not None:
            # --- Scenario B: Bearish liquidity sweep at fib resistance -------
            bo_type, candle_info = classify_breakout_type(
                current_open, current_high, current_low, current_close,
                prev_close, tested_level_price, "UP",
                has_volume_spike, vol_ratio,
            )

            if bo_type == "FAKE_BREAKOUT":
                signal      = "BEARISH_FAKE_BREAKOUT"
                confidence += 35
                reasons.append(
                    f"Bearish liquidity sweep above Fibonacci {tested_level_name} level"
                )

                if not has_volume_spike:
                    confidence += 15
                    reasons.append("Low-volume sweep confirms illiquid wick (not real buying)")
                elif vol_ratio >= 1.2:
                    confidence += 10
                    reasons.append("Volume spike on reversal bar")

                if effective_trend == "DOWN":
                    confidence += 20
                    reasons.append("Trend aligned DOWN — continuation short favored")
                elif effective_trend == "UP":
                    confidence -= 15
                    reasons.append("Against UP trend — reduced conviction")

                if market_state == "RANGE":
                    confidence += 10
                    reasons.append("Range market: fake-breakout setups are primary")
                elif market_state == "STRONG_TREND":
                    confidence += 5
                    reasons.append("Strong trend context supports continuation")

                if candle_info["quality"] == "STRONG":
                    confidence += 15
                    reasons.append("Strong bearish reversal candle")
                elif candle_info["quality"] == "MODERATE":
                    confidence += 8
                    reasons.append("Moderate reversal candle")
                else:
                    confidence -= 10
                    reasons.append("Weak candle quality — wait for confirmation")

                retest_confirmed, _ = detect_retest(
                    closes, highs, lows, tested_level_price, "DOWN"
                )
                if retest_confirmed:
                    confidence += 10
                    reasons.append("Post-sweep level retest confirmed")

                entry_zone   = [round(tested_level_price * 0.998, 6),
                                 round(tested_level_price, 6)]
                invalidation = round(sweep_level * 1.002, 6)
                tp_zone      = [round(fib_lvls["0.786"], 6),
                                 round(fib_lvls["1.0"], 6)]

        elif in_bounce_zone and candle_info["direction"] == "BEAR":
            # --- Scenario A: Short continuation at fib resistance -----------
            signal      = "SHORT_CONTINUATION"
            confidence += 30
            reasons.append(
                "Price bouncing into Fibonacci resistance zone (38.2–61.8%) "
                "of bearish impulse"
            )

            if has_volume_spike:
                confidence += 15
                reasons.append("Volume spike on rejection at resistance")

            if effective_trend == "DOWN":
                confidence += 20
                reasons.append("Trend aligned DOWN")
            elif effective_trend == "UP":
                confidence -= 20
                reasons.append("Against UP trend — signal degraded")

            if market_state == "STRONG_TREND":
                confidence += 15
                reasons.append("Strong trend market favors Fibonacci continuation short")
            elif market_state == "RANGE":
                confidence -= 5
                reasons.append("Range market reduces continuation reliability")

            if candle_info["quality"] == "STRONG":
                confidence += 15
                reasons.append("Strong bearish candle confirming rejection")
            elif candle_info["quality"] == "MODERATE":
                confidence += 8
                reasons.append("Moderate bearish candle")
            else:
                confidence -= 10
                reasons.append("Weak candle — wait for better confirmation")

            retest_confirmed, _ = detect_retest(
                closes, highs, lows, fib_lvls["0.500"], "DOWN"
            )
            if retest_confirmed:
                confidence += 10
                reasons.append("50% level retest confirmed")

            entry_zone   = [round(fib_lvls["0.618"], 6), round(fib_lvls["0.382"], 6)]
            invalidation = round(fib_lvls["0.236"], 6)
            tp_zone      = [round(fib_lvls["0.786"], 6), round(fib_lvls["1.0"], 6)]

    # --- Final confidence clamping & minimum threshold ----------------------
    confidence = max(0, min(100, confidence))

    # Downgrade to NO_TRADE if evidence is insufficient
    if signal not in ("INVALID",) and confidence < 30:
        signal = "NO_TRADE"
        reasons.append("Confidence below threshold (30) — no valid signal")

    # --- Return structured result dict --------------------------------------
    return {
        "signal":          signal,
        "confidence":      confidence,
        "reason":          "; ".join(reasons) if reasons else "No setup conditions met",
        "entry_zone":      entry_zone,
        "invalidation":    invalidation,
        "tp_zone":         tp_zone,
        "fib_levels":      {k: round(v, 6) for k, v in fib_lvls.items()},
        "market_state":    market_state,
        "trend_alignment": effective_trend,
        "impulse": {
            "direction":  impulse["direction"],
            "swing_high": round(swing_high, 6),
            "swing_low":  round(swing_low, 6),
            "move_pct":   impulse["move_pct"],
        },
        "candle_quality": candle_info,
        "volume_spike":   has_volume_spike,
        "volume_ratio":   round(vol_ratio, 3),
        "htf_context":    htf_context,
    }
