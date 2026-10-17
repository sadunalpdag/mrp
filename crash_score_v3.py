"""Crash Bounce continuous 0-100 research score.

Every feature is derived from the same 480 *closed* 15m candles used by V2,
plus a point-in-time 24h BTC futures ticker when available. These weights are
hypotheses for *ranking*, not win probabilities and not order-entry gates.
"""
import math
import statistics

VERSION = "continuous_7factor_v1"
WEIGHTS = {
    "dip_stability": 25.0,
    "rsi_momentum": 20.0,
    "volume_confirmation": 20.0,
    "ema25_improvement": 15.0,
    "atr_tp_capacity": 10.0,
    "selling_slowdown": 5.0,
    "btc_context": 5.0,
}


def _clip(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def _lerp(x, x0, x1):
    return _clip((x - x0) / (x1 - x0))


def _ema(values, period):
    value = sum(values[:period]) / period
    result = [None] * (period - 1) + [value]
    alpha = 2 / (period + 1)
    for price in values[period:]:
        value = value + alpha * (price - value)
        result.append(value)
    return result


def _rsi_sequence(closes, period=14):
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(0., x) for x in deltas]
    losses = [max(0., -x) for x in deltas]
    gain = sum(gains[:period]) / period
    loss = sum(losses[:period]) / period
    result = [None] * period
    def value(g, l):
        return 100 - 100 / (1 + g / l) if l else (100. if g else 50.)
    result.append(value(gain, loss))
    for i in range(period, len(deltas)):
        gain = (gain * (period - 1) + gains[i]) / period
        loss = (loss * (period - 1) + losses[i]) / period
        result.append(value(gain, loss))
    return result


def _atr_pct(highs, lows, closes):
    tr = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]),
                      abs(lows[i]-closes[i-1])))
    atr = sum(tr[-14:]) / 14
    return 100 * atr / closes[-1]


def _atr_capacity(atr_pct):
    # A +1% gross TP needs movement, but extreme volatility adds execution risk.
    return min(_lerp(atr_pct, 0.35, 1.50), 1.,
               1 - _lerp(atr_pct, 4.0, 9.0))


def score_candidate(candles, btc_change_24h=None):
    """Produce fractional component scores and underlying observable measurements.

    `candles` must be 480 already-validated chronological closed OHLCV tuples.
    Missing external BTC data are *excluded* from denominator rather than
    silently awarded 0/5 or 5/5; coverage is always reported.
    """
    if len(candles) != 480:
        raise ValueError("Need 480 fully closed candles")
    opens = [float(c[0]) for c in candles]
    highs = [float(c[1]) for c in candles]
    lows = [float(c[2]) for c in candles]
    closes = [float(c[3]) for c in candles]
    volumes = [float(c[4]) for c in candles]
    if any(not all(math.isfinite(x) for x in c) for c in candles):
        raise ValueError("Nonfinite candle input")
    if min(closes) <= 0 or min(volumes) < 0:
        raise ValueError("Invalid candle input")

    # 1. Number of new lows vs preceding 4h and age of the last 4h low.
    previous_support = min(lows[-32:-16])
    new_lows = 0
    for low in lows[-16:]:
        if low < previous_support:
            new_lows += 1
            previous_support = low
    low_index = min(range(16), key=lambda i: lows[-16+i])
    bars_since_low = 15 - low_index
    dip_fraction = (0.65 * (1 - _lerp(new_lows, 0, 8))
                    + 0.35 * _lerp(bars_since_low, 0, 12))

    # 2. Recent RSI movement, not simply the current RSI threshold.
    rsi = _rsi_sequence(closes)
    rsi_now, rsi_1h_ago = rsi[-1], rsi[-5]
    rsi_delta = rsi_now - rsi_1h_ago
    rsi_fraction = (0.75 * _lerp(rsi_delta, -2.0, 9.0)
                    + 0.25 * (1 - _lerp(rsi_now, 48.0, 75.0)))

    # 3. Relative volume and the share in green candles over the last 1h.
    baseline = statistics.median(volumes[-52:-4])  # previous 12 hours
    latest_volume = sum(volumes[-4:]) / 4
    vol_ratio = latest_volume / baseline if baseline > 0 else None
    last_vol = sum(volumes[-4:])
    green_vol_ratio = (sum(volumes[i] for i in range(-4, 0)
                           if closes[i] > opens[i]) / last_vol if last_vol > 0 else None)
    vol_fraction = (0.6 * _lerp(vol_ratio, 0.6, 2.4) +
                    0.4 * _lerp(green_vol_ratio, 0.2, 0.8)
                    if vol_ratio is not None and green_vol_ratio is not None else None)

    # 4. Is EMA25 down-slope flattening or turning positive?
    ema25 = _ema(closes, 25)
    slope_now = 100 * (ema25[-1] / ema25[-2] - 1)
    slope_1h_ago = 100 * (ema25[-5] / ema25[-6] - 1)
    slope_change = slope_now - slope_1h_ago
    ema_fraction = (0.7 * _lerp(slope_change, -0.08, 0.15)
                    + 0.3 * _lerp(slope_now, -0.35, 0.15))

    # 5. ATR14 as percentage of price for a 1% gross TP.
    atr_pct = _atr_pct(highs, lows, closes)
    atr_fraction = _atr_capacity(atr_pct)

    # 6. Is the decline accelerating or slowing? Normalize previous 4h to 1h.
    return_1h = 100 * (closes[-1] / closes[-5] - 1)
    previous_4h_per_hour = 25 * (closes[-5] / closes[-21] - 1)
    deceleration = return_1h - previous_4h_per_hour
    slowdown_fraction = _lerp(deceleration, -1.0, 2.0)

    fractions = {
        "dip_stability": dip_fraction,
        "rsi_momentum": rsi_fraction,
        "volume_confirmation": vol_fraction,
        "ema25_improvement": ema_fraction,
        "atr_tp_capacity": atr_fraction,
        "selling_slowdown": slowdown_fraction,
    }
    btc_value = None
    try:
        parsed = float(btc_change_24h)
        if btc_change_24h is not None and math.isfinite(parsed):
            btc_value = parsed
    except (ValueError, TypeError, OverflowError):
        pass
    fractions["btc_context"] = _lerp(btc_value, -3.0, 2.0) if btc_value is not None else None

    points = {name:(round(_clip(fraction)*WEIGHTS[name], 2)
                    if fraction is not None else None)
              for name, fraction in fractions.items()}
    available = sum(WEIGHTS[k] for k in points if points[k] is not None)
    raw = sum(points[k] for k in points if points[k] is not None)
    # Preserve missing data transparently. Do not pretend full coverage.
    score = 100 * raw / available if available else None
    return {
        "version": VERSION, "score": round(score, 1) if score is not None else None,
        "raw_points": round(raw, 2), "available_points": available,
        "coverage_pct": round(available, 1),
        "components": {k: {"points": v, "max_points": WEIGHTS[k]}
                       for k,v in points.items()},
        "metrics": {
            "new_lows_4h":new_lows,
            "bars_since_low_15m":bars_since_low,
            "rsi14":round(rsi_now, 3),
            "rsi_delta_1h":round(rsi_delta, 3),
            "volume_ratio_vs_prior_12h":round(vol_ratio,3) if vol_ratio is not None else None,
            "green_volume_share_1h":round(green_vol_ratio,3) if green_vol_ratio is not None else None,
            "ema25_slope_pct":round(slope_now,4),
            "ema25_slope_change_1h_pp":round(slope_change,4),
            "atr14_pct":round(atr_pct,3),
            "return_1h_pct":round(return_1h,3),
            "previous_4h_return_per_hour_pct":round(previous_4h_per_hour,3),
            "btc_24h_change_pct":round(btc_value,3) if btc_value is not None else None,
        }
    }
