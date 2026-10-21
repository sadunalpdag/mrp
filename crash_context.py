"""Point-in-time research diagnostics; never an acceptance or order gate."""
import math
from collections import Counter

VERSION = "five_day_context_v1"


def assess_context(candles, quote_volumes, strength):
    """Consume the 480 validated closed candles selected by crash_v2."""
    closes = [c[3] for c in candles]
    high = max(c[1] for c in candles)
    low = min(c[2] for c in candles)
    daily = [100 * (closes[(i+1)*96-1] / candles[i*96][0] - 1)
             for i in range(5)]
    previous = 100 * (closes[-97] / candles[0][0] - 1)
    last = 100 * (closes[-1] / closes[-97] - 1)
    complete_volume = len(quote_volumes) == 480 and all(
        v is not None and math.isfinite(v) and v >= 0 for v in quote_volumes)
    baseline = sum(quote_volumes[:-96]) / 4 if complete_volume else 0
    volume_ratio = sum(quote_volumes[-96:]) / baseline if baseline > 0 else None
    m = strength.get("metrics", {})
    green, rsi = m.get("green_volume_share_1h"), m.get("rsi_delta_1h")
    known = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (green, rsi))
    flag = green < .2 and rsi < 0 if known else None
    # Continuous, uncalibrated research index. Not a loss probability.
    pressure = (100 * (1-green) * min(1, max(0, -rsi/10))) if known else None
    regime = ("PULLBACK_AFTER_RISE" if previous > 0 else "CONTINUING_DECLINE") if last < 0 else "RECOVERY_OR_FLAT"
    return {
        "version": VERSION, "blocks_delivery": False,
        "return_5d_pct": round(100*(closes[-1]/candles[0][0]-1), 4),
        "previous_4d_return_pct": round(previous, 4),
        "return_24h_pct": round(last, 4),
        "drawdown_from_5d_high_pct": round(100*(closes[-1]/high-1), 4),
        "range_position_5d_pct": round(100*(closes[-1]-low)/(high-low), 4) if high > low else None,
        "daily_returns_pct": [round(v, 4) for v in daily],
        "declining_24h_blocks": sum(v < 0 for v in daily),
        "quote_volume_ratio_24h_vs_previous_4d": round(volume_ratio, 4) if volume_ratio is not None else None,
        "quote_volume_complete": complete_volume,
        "regime": regime,
        "selling_pressure_index": round(pressure, 1) if pressure is not None else None,
        "weak_green_and_falling_rsi": flag,
        "reason": "GREEN_VOLUME_LT_20_AND_RSI_FALLING" if flag else (
            "RULE_NOT_TRIGGERED" if flag is False else "MISSING_MOMENTUM_DATA"),
    }


def format_context(context):
    if not context:
        return ""
    regimes = {"PULLBACK_AFTER_RISE": "Yükseliş sonrası geri çekilme",
               "CONTINUING_DECLINE": "Önceki düşüş devam ediyor",
               "RECOVERY_OR_FLAT": "Son gün toparlanma/yatay"}
    ratio = context.get("quote_volume_ratio_24h_vs_previous_4d")
    volume = f"{ratio:.2f} kat" if ratio is not None else "veri yok"
    pressure = context.get("selling_pressure_index")
    value = f"{pressure:.1f}/100" if pressure is not None else "veri yok"
    flag = context.get("weak_green_and_falling_rsi")
    warning = ("Zayıf yeşil hacim + düşen RSI" if flag else
               "Birleşik uyarı yok" if flag is False else "Momentum verisi eksik")
    return (f"5g bağlam: {regimes.get(context.get('regime'), 'veri yok')}\n"
            f"Son gün USDT hacmi / önceki 4 gün ort.: {volume}\n"
            f"Satış baskısı araştırma endeksi: {value}\n"
            f"{warning}. Deneyseldir; olasılık veya sinyal engeli değildir.\n")


def summarize_context(records):
    """Mutually exclusive cohorts; older/unavailable diagnostics stay separate."""
    groups = {name: [] for name in ("flagged", "not_flagged", "unknown")}
    for record in records:
        context = (record.get("v2_features") or {}).get("five_day_context") or {}
        flag = context.get("weak_green_and_falling_rsi") if context.get("version") == VERSION else None
        groups["flagged" if flag is True else "not_flagged" if flag is False else "unknown"].append(record)
    result = {}
    for name, items in groups.items():
        counts = Counter(r.get("tracking_status", "UNKNOWN") for r in items)
        entered = sum(r.get("entered_at") is not None for r in items)
        result[name] = {"signals": len(items), "entered": entered,
                        "statuses": dict(counts),
                        "tp_fraction_of_entries_pct": 100*counts["TP"]/entered if entered else None,
                        "records_with_gaps": sum(r.get("coverage_gaps", 0) > 0 for r in items)}
    return {"version": VERSION, "blocks_delivery": False, "groups": result,
            "note": "Prospective descriptive cohorts; open trades unresolved, not a causal filter backtest."}
