"""
EMA ULTRA — ema.py

detect_distribution_pattern()
==============================
Pump sonrası blow-off top / MM distribution / exit liquidity yapısını tespit eder.

Pattern şartları:
  * Son 6-12 mum içinde güçlü impulse (%12+ veya config ile ayarlanabilir)
  * Tepe mumunda anlamlı upper wick
  * Hacim spike (ortalamanın üstünde)
  * Sonraki 1-3 mumda kırmızı follow-through
  * Peak'ten belirgin rejection

Return:
  {
    "detected":     bool,
    "score":        int,
    "state":        "DISTRIBUTION" | "NONE",
    "bias":         "LONG_EXIT" | "SHORT_BIAS" | "NEUTRAL",
    "action":       "NO_LONG" | "WAIT_PULLBACK" | "LOOK_FOR_SHORT_CONFIRMATION",
    "impulse_pct":  float,
    "peak_price":   float,
    "last_price":   float,
    "rejection_pct": float,
    "wick_ratio":   float,
    "volume_ratio": float,
    "reason":       str,
  }

Entegrasyon:
  * Distribution varsa yeni long açma
  * Breakout long sinyali varsa baskıla
  * State akışı: OVEREXTENDED → DISTRIBUTION  veya
                 BREAKOUT_PENDING → FAKE_BREAKOUT_LONG

Örnek log:
  🚫 DISTRIBUTION DETECTED — ENJUSDT
  Bias: LONG_EXIT | State: OVEREXTENDED → DISTRIBUTION
  Peak: 0.07317 | Last: 0.05285
  Impulse: %64.9 | Rejection: %27.8
  WickRatio: 2.8 | VolRatio: 3.9
  Action: NO_LONG / LOOK_FOR_SHORT_CONFIRMATION
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (all callers can override via kwargs)
# ---------------------------------------------------------------------------

DEFAULT_IMPULSE_LOOKBACK         = 10    # candles to scan for the impulse leg
DEFAULT_FOLLOW_THROUGH_LOOKBACK  = 3     # candles after peak to check for red close
DEFAULT_VOLUME_AVG_LOOKBACK      = 20    # bars used to build the baseline avg volume

DEFAULT_MIN_IMPULSE_PCT          = 0.12  # 12 % minimum pump to qualify
DEFAULT_MIN_WICK_RATIO           = 1.5   # upper_wick / body on the peak candle
DEFAULT_MIN_VOLUME_RATIO         = 1.5   # peak-candle volume / rolling-avg volume
DEFAULT_MIN_REJECTION_PCT        = 0.03  # 3 % drop from peak to current price


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    """Return a zeroed-out result dict for error / no-signal paths."""
    return {
        "detected":      False,
        "score":         0,
        "state":         "NONE",
        "bias":          "NEUTRAL",
        "action":        "WAIT_PULLBACK",
        "impulse_pct":   0.0,
        "peak_price":    0.0,
        "last_price":    0.0,
        "rejection_pct": 0.0,
        "wick_ratio":    0.0,
        "volume_ratio":  0.0,
        "reason":        "",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_distribution_pattern(
    klines: list,
    symbol: str = "",
    *,
    impulse_lookback: int = DEFAULT_IMPULSE_LOOKBACK,
    follow_through_lookback: int = DEFAULT_FOLLOW_THROUGH_LOOKBACK,
    volume_avg_lookback: int = DEFAULT_VOLUME_AVG_LOOKBACK,
    min_impulse_pct: float = DEFAULT_MIN_IMPULSE_PCT,
    min_wick_ratio: float = DEFAULT_MIN_WICK_RATIO,
    min_volume_ratio: float = DEFAULT_MIN_VOLUME_RATIO,
    min_rejection_pct: float = DEFAULT_MIN_REJECTION_PCT,
) -> dict:
    """
    Detect a pump-followed blow-off top / MM distribution / exit liquidity
    structure in the supplied kline history.

    Parameters
    ----------
    klines : list of klines [[time, open, high, low, close, volume, ...], ...]
             ordered oldest-first.  Needs at least
             impulse_lookback + follow_through_lookback + 5 bars.
    symbol : ticker string used only for logging (e.g. "ENJUSDT")
    impulse_lookback        : candles to look back when searching for the peak
                              (6-12 recommended; default 10)
    follow_through_lookback : candles AFTER the peak to check for red close
                              (1-3 recommended; default 3)
    volume_avg_lookback     : bars before the impulse window used to build the
                              rolling volume baseline (default 20)
    min_impulse_pct         : minimum percentage rise to qualify as a pump
                              (default 0.12 = 12 %)
    min_wick_ratio          : upper_wick / body ratio on the peak candle
                              (default 1.5)
    min_volume_ratio        : peak-candle volume / avg-volume threshold
                              (default 1.5)
    min_rejection_pct       : minimum % drop from peak to current price
                              (default 0.03 = 3 %)

    Returns
    -------
    dict — see module docstring for full key / value specification.

    Integration notes
    -----------------
    * If ``detected`` is True, suppress any new LONG signal.
    * State machine transitions supported:
        - OVEREXTENDED  → DISTRIBUTION
        - BREAKOUT_PENDING → FAKE_BREAKOUT_LONG
    """
    result = _empty_result()

    # ── Validate input ────────────────────────────────────────────────────
    min_bars = impulse_lookback + follow_through_lookback + 5
    if not klines or len(klines) < min_bars:
        result["reason"] = f"Insufficient kline data (need ≥ {min_bars} bars)"
        return result

    try:
        opens   = [float(k[1]) for k in klines]
        highs   = [float(k[2]) for k in klines]
        lows    = [float(k[3]) for k in klines]
        closes  = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
    except (IndexError, TypeError, ValueError) as exc:
        result["reason"] = f"Kline parse error: {exc}"
        return result

    n          = len(closes)
    last_price = closes[-1]

    # ── Step 1: Locate the peak inside the impulse scan window ───────────
    # The scan window excludes the last `follow_through_lookback` candles so
    # that the red follow-through candles are guaranteed to fall AFTER the peak.
    scan_start = max(0, n - impulse_lookback - follow_through_lookback)
    scan_end   = n - follow_through_lookback
    if scan_end <= scan_start:
        result["reason"] = "Scan window too narrow"
        return result

    window_highs   = highs[scan_start:scan_end]
    peak_idx_local = window_highs.index(max(window_highs))
    peak_idx       = scan_start + peak_idx_local   # absolute index
    peak_price     = highs[peak_idx]

    # ── Step 2: Measure the impulse leg ──────────────────────────────────
    # Base = lowest low/close from scan_start up to and including the peak candle.
    impulse_lows   = lows[scan_start : peak_idx + 1]
    impulse_closes = closes[scan_start : peak_idx + 1]
    base_price     = min(min(impulse_lows), min(impulse_closes))

    if base_price <= 0:
        result["reason"] = "Zero base price — cannot compute impulse"
        return result

    impulse_pct = (peak_price - base_price) / base_price   # e.g. 0.649 for 64.9 %

    # ── Step 3: Upper wick ratio on the peak candle ───────────────────────
    peak_open       = opens[peak_idx]
    peak_close      = closes[peak_idx]
    peak_body       = abs(peak_close - peak_open)
    peak_upper_wick = peak_price - max(peak_open, peak_close)
    # A doji (zero-body) candle is treated as an extreme wick — use None to signal
    # "unmeasurable" and handle it explicitly below rather than propagating infinity.
    wick_ratio: float | None = (
        round(peak_upper_wick / peak_body, 2) if peak_body > 0 else None
    )

    # ── Step 4: Volume spike at the peak candle ───────────────────────────
    vol_start  = max(0, scan_start - volume_avg_lookback)
    vol_window = volumes[vol_start:scan_start]
    avg_volume = (sum(vol_window) / len(vol_window)) if vol_window else 1.0
    peak_volume  = volumes[peak_idx]
    volume_ratio = peak_volume / avg_volume if avg_volume > 0 else 1.0

    # ── Step 5: Red follow-through after the peak ────────────────────────
    ft_start = peak_idx + 1
    ft_end   = min(n, ft_start + follow_through_lookback)
    ft_pairs = list(zip(opens[ft_start:ft_end], closes[ft_start:ft_end]))
    red_count          = sum(1 for o, c in ft_pairs if c < o)
    has_follow_through = len(ft_pairs) > 0 and red_count >= 1

    # ── Step 6: Rejection from the peak ──────────────────────────────────
    rejection_pct = (peak_price - last_price) / peak_price if peak_price > 0 else 0.0

    # ── Scoring (max 100) ─────────────────────────────────────────────────
    score          = 0
    reasons_met    = []
    reasons_missed = []

    # Impulse — mandatory, weight 30
    if impulse_pct >= min_impulse_pct:
        score += 30
        reasons_met.append(f"Impulse {impulse_pct:.1%}")
    else:
        reasons_missed.append(
            f"Impulse too weak ({impulse_pct:.1%} < {min_impulse_pct:.1%})"
        )

    # Upper-wick ratio — weight 20
    wick_qualifies = wick_ratio is not None and wick_ratio >= min_wick_ratio
    if wick_qualifies:
        score += 20
        reasons_met.append(f"WickRatio {wick_ratio:.1f}")
    elif wick_ratio is None:
        # Doji peak candle: zero body implies entire range is wick — counts as extreme wick
        score += 20
        reasons_met.append("WickRatio N/A (doji peak — full wick)")
        wick_qualifies = True
    else:
        reasons_missed.append(f"Wick too small (ratio={wick_ratio:.1f})")

    # Volume spike — weight 20
    if volume_ratio >= min_volume_ratio:
        score += 20
        reasons_met.append(f"VolRatio {volume_ratio:.1f}")
    else:
        reasons_missed.append(f"No volume spike (ratio={volume_ratio:.1f})")

    # Red follow-through — weight 15
    if has_follow_through:
        score += 15
        reasons_met.append(f"RedFollowThrough({red_count}/{len(ft_pairs)})")
    else:
        reasons_missed.append("No red follow-through after peak")

    # Rejection from peak — weight 15
    if rejection_pct >= min_rejection_pct:
        score += 15
        reasons_met.append(f"Rejection {rejection_pct:.1%}")
    else:
        reasons_missed.append(f"Rejection too small ({rejection_pct:.1%})")

    # ── Detection decision ────────────────────────────────────────────────
    # Require: impulse qualifies AND at least 2 of the 4 secondary conditions.
    impulse_ok   = impulse_pct >= min_impulse_pct
    secondary_ok = sum([
        wick_qualifies,
        volume_ratio >= min_volume_ratio,
        has_follow_through,
        rejection_pct >= min_rejection_pct,
    ])
    detected = impulse_ok and secondary_ok >= 2

    # ── State / bias / action ─────────────────────────────────────────────
    if detected:
        state = "DISTRIBUTION"
        if rejection_pct >= 0.15:
            bias   = "SHORT_BIAS"
            action = "LOOK_FOR_SHORT_CONFIRMATION"
        elif rejection_pct >= 0.05:
            bias   = "LONG_EXIT"
            action = "NO_LONG"
        else:
            bias   = "LONG_EXIT"
            action = "WAIT_PULLBACK"
    else:
        state  = "NONE"
        bias   = "NEUTRAL"
        action = "WAIT_PULLBACK"

    # ── Reason string ─────────────────────────────────────────────────────
    reason_parts = []
    if reasons_met:
        reason_parts.append("✅ " + " | ".join(reasons_met))
    if reasons_missed:
        reason_parts.append("❌ " + " | ".join(reasons_missed))
    reason = " — ".join(reason_parts) if reason_parts else "No signal"

    # ── Logging ───────────────────────────────────────────────────────────
    if detected:
        sym_tag    = f" — {symbol}" if symbol else ""
        prev_state = "OVEREXTENDED" if impulse_pct >= 0.30 else "BREAKOUT_PENDING"
        action_log = (
            "NO_LONG / LOOK_FOR_SHORT_CONFIRMATION"
            if action == "LOOK_FOR_SHORT_CONFIRMATION"
            else action
        )
        logger.warning(
            "\U0001f6ab DISTRIBUTION DETECTED%s\n"
            "  Bias: %s | State: %s \u2192 DISTRIBUTION\n"
            "  Peak: %.5f | Last: %.5f\n"
            "  Impulse: %%%.1f | Rejection: %%%.1f\n"
            "  WickRatio: %s | VolRatio: %.1f\n"
            "  Action: %s",
            sym_tag,
            bias,
            prev_state,
            peak_price,
            last_price,
            impulse_pct * 100,
            rejection_pct * 100,
            f"{wick_ratio:.1f}" if wick_ratio is not None else "N/A",
            volume_ratio,
            action_log,
        )

    return {
        "detected":      detected,
        "score":         score,
        "state":         state,
        "bias":          bias,
        "action":        action,
        "impulse_pct":   round(impulse_pct, 4),
        "peak_price":    round(peak_price, 6),
        "last_price":    round(last_price, 6),
        "rejection_pct": round(rejection_pct, 4),
        "wick_ratio":    wick_ratio,
        "volume_ratio":  round(volume_ratio, 2),
        "reason":        reason,
    }
