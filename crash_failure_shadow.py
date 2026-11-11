"""Research TP-risk flags and user-requested automatic BUY exclusion.

Only historical closed V2 candles and preceding signals are used.
V2 qualification, Telegram delivery, virtual tracking and existing TP orders
remain unchanged. This is an experimental risk restriction, not a probability.
"""
import csv
import io
import math
from collections import Counter

VERSION = "tp_nonclosure_entry_gate_v2_atr"
LEGACY_VERSION = "tp_nonclosure_entry_gate_v1"
SIZING_VERSION = "risk_count_sizing_v2_four_flags"
LEGACY_SIZING_VERSION = "risk_count_sizing_v1"
FLAG_NAMES = ("weak_volume", "falling_rsi_ema", "repeat_lower_entry", "low_atr_15m")
ATR_MIN_PCT = 1.3
REPEAT_LOOKBACK_SECONDS = 72 * 3600


def position_fraction(record):
    """0/1 full; 2 half; 3 thirty percent; 4 blocked. Missing blocks.

    Inspect the stored signal-time flag booleans, never outcome or claimed
    flag_count. Only new-version assessments are valid for new BUY. Legacy assessments
    fail closed rather than assuming that missing ATR is safe.
    """
    data = record.get("failure_shadow")
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return None
    flags = data.get("flags")
    if not isinstance(flags, dict) or any(type(flags.get(k)) is not bool for k in FLAG_NAMES):
        return None
    count = sum(flags[k] for k in FLAG_NAMES)
    return 0.0 if count == 4 else 0.3 if count == 3 else 0.5 if count == 2 else 1.0


def number(value, minimum=None, maximum=None):
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(x) or (minimum is not None and x < minimum) or (
            maximum is not None and x > maximum):
        return None
    return x


def assess(record, earlier_records):
    """Research flags at candidate time, without using subsequent prices/results."""
    strength = (record.get("v2_features") or {}).get("strength_score") or {}
    metrics = strength.get("metrics") or {}
    green = number(metrics.get("green_volume_share_1h"), 0, 1)
    volume = number(metrics.get("volume_ratio_vs_prior_12h"), 0)
    rsi_change = number(metrics.get("rsi_delta_1h"), -100, 100)
    ema_change = number(metrics.get("ema25_slope_change_1h_pp"))
    atr_pct = number(metrics.get("atr14_pct"), 0)
    low_atr = (atr_pct < ATR_MIN_PCT if atr_pct is not None else None)
    # Both volume measures must be known; missing volume is NOT a clean signal.
    weak_volume = (green <= 0.20 and volume < 1.5
                   if green is not None and volume is not None else None)
    falling_momentum = (rsi_change <= -5.0 and ema_change <= -0.08
                        if rsi_change is not None and ema_change is not None else None)

    created = number(record.get("created_at"), 0)
    current_entry = number(record.get("entry"), 0)
    prior = None
    if created is not None:
        candidates = []
        for other in earlier_records:
            if (other.get("signal_id") == record.get("signal_id") or
                    other.get("symbol") != record.get("symbol") or
                    other.get("v2_status") != "STRONG_V2"):
                continue
            prior_at = number(other.get("created_at"), 0)
            prior_entry = number(other.get("entry"), 0)
            if (prior_at is not None and prior_entry is not None and prior_entry > 0
                    and 0 < created - prior_at <= REPEAT_LOOKBACK_SECONDS):
                candidates.append((prior_at, prior_entry, other.get("signal_id")))
        if candidates:
            prior = max(candidates, key=lambda item: item[0])
    decline = (100 * (current_entry / prior[1] - 1)
               if prior and current_entry is not None and current_entry > 0 else None)
    repeat_lower = (decline <= -2 if decline is not None
                    else False if created is not None and current_entry is not None
                    else None)

    flags = {
        "weak_volume": weak_volume,
        "falling_rsi_ema": falling_momentum,
        "repeat_lower_entry": repeat_lower,
        "low_atr_15m": low_atr,
    }
    flagged = sum(value is True for value in flags.values())
    known = sum(value is not None for value in flags.values())
    return {
        "version": VERSION,
        "observed_at": created,
        "blocks_v2": False,
        "blocks_live_entry": flagged == len(FLAG_NAMES) or known < len(FLAG_NAMES),
        "sizing_policy_version": SIZING_VERSION,
        "position_fraction": (None if known < len(FLAG_NAMES)
                              else 0.0 if flagged == 4 else 0.3 if flagged == 3
                              else 0.5 if flagged == 2 else 1.0),
        "flags": flags,
        "flag_count": flagged,
        "known_flags": known,
        "metrics": {
            "green_volume_share_1h": green,
            "volume_ratio_vs_prior_12h": volume,
            "rsi_delta_1h": rsi_change,
            "ema25_slope_change_1h_pp": ema_change,
            "atr14_pct": atr_pct,
            "atr_min_pct": ATR_MIN_PCT,
            "repeat_prior_entry_drop_pct": round(decline, 3) if decline is not None else None,
            "repeat_prior_signal_id": str(prior[2]) if prior and prior[2] else None,
        },
        "note": "Exploratory warning only, not TP probability; only past data.",
    }


def entry_block_reason(record):
    """New BUYs blocked on four true flags or incomplete/old assessment."""
    data = record.get("failure_shadow")
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return "RISK_ASSESSMENT_MISSING"
    flags = data.get("flags")
    if not isinstance(flags, dict):
        return "RISK_ASSESSMENT_INVALID"
    if any(type(flags.get(name)) is not bool for name in FLAG_NAMES):
        return "RISK_METRICS_INCOMPLETE"
    if all(flags[name] is True for name in FLAG_NAMES):
        return "RISK_4_OF_4"
    return None


def format_warning(result):
    """Risk notice for Telegram; never suppress V2 signal itself."""
    if not isinstance(result, dict) or result.get("version") != VERSION:
        return ("⏸️ 4'LÜ RİSK ETİKETİ HESAPLANAMADI: otomatik gerçek alış devre dışı.\n"
                "V2 sanal takibi devam eder.\n")
    flags = result.get("flags") or {}
    names = {
        "weak_volume": "zayıf yeşil mum hacmi",
        "falling_rsi_ema": "RSI ve EMA eğimi kötüleşiyor",
        "repeat_lower_entry": "aynı coinde daha düşük yeni sinyal",
        "low_atr_15m": f"15dk ATR %{ATR_MIN_PCT} altında",
    }
    hits = [label for key, label in names.items() if flags.get(key) is True]
    known = sum(type(flags.get(key)) is bool for key in names)
    missing = f" | eksik veri {len(names)-known}/{len(names)}" if known < len(names) else ""
    fraction = None if known < len(names) else (
        0.0 if len(hits) == 4 else 0.3 if len(hits) == 3
        else 0.5 if len(hits) == 2 else 1.0)
    action = ("⏸️ EKSİK VERİ: otomatik GERÇEK ALIŞ YAPILMAYACAK.\n"
              if fraction is None else
              "⛔ 4/4 RİSK: otomatik GERÇEK ALIŞ YAPILMAYACAK.\n"
              if fraction == 0 else
              "⚠️ 3/4 RİSK: otomatik pozisyon tutarının %30'u; TP hedefi %0,5.\n"
              if fraction == 0.3 else
              "⚖️ 2/4 RİSK: otomatik pozisyon tutarının %50'si; TP hedefi %0,5.\n"
              if fraction == 0.5 else
              "✅ 0–1/4 RİSK: otomatik pozisyon tutarının %100'ü; TP hedefi %1.\n")
    label = " | ".join(hits) if hits else "belirgin risk işareti yok"
    return (f"🔎 TP RİSK ETİKETLERİ: {len(hits)}/4 — {label}{missing}\n"
            + action + "Deneysel; TP olasılığı değildir. Sinyal ve sanal takip devam eder.\n")

def summarize(records, executor_orders=None):
    """Compare prospective, same-model cohorts without counting OPEN as loss.

    Cohort groups overlap; counts cannot be summed. Missing flags are unknown,
    not unflagged. Backfilled features from later snapshots are never used.
    """
    selected = [record for record in records
                if (record.get("tracking_model") == "ws_aggtrade_upcross_v3"
                    and record.get("v2_status") == "STRONG_V2"
                    and record.get("v2_sent") is True
                    and (record.get("failure_shadow") or {}).get("version") in (VERSION, LEGACY_VERSION))]
    def has_flag(record, name):
        return (record["failure_shadow"].get("flags") or {}).get(name)
    groups = {
        "all": selected,
        "weak_volume": [r for r in selected if has_flag(r, "weak_volume") is True],
        "falling_rsi_ema": [r for r in selected if has_flag(r, "falling_rsi_ema") is True],
        "repeat_lower_entry": [r for r in selected if has_flag(r, "repeat_lower_entry") is True],
        "low_atr_15m": [r for r in selected if has_flag(r, "low_atr_15m") is True],
        "two_or_more": [r for r in selected if r["failure_shadow"].get("flag_count", 0) >= 2],
        "none_known": [r for r in selected if r["failure_shadow"].get("known_flags") == len(FLAG_NAMES)
                       and r["failure_shadow"].get("flag_count") == 0],
        "missing_some_data": [r for r in selected if r["failure_shadow"].get("known_flags", 0) < len(FLAG_NAMES)],
    }
    executor_orders = executor_orders or {}
    result = {}
    for name, items in groups.items():
        counts = Counter(item.get("tracking_status", "UNKNOWN") for item in items)
        closed = counts["TP"] + counts["TIMEOUT"]
        result[name] = {
            "signals": len(items),
            "entered": counts["TP"] + counts["TIMEOUT"] + counts["OPEN"],
            "tp": counts["TP"],
            "timeout": counts["TIMEOUT"],
            "open_unresolved": counts["OPEN"],
            "waiting_entry": counts["WAIT_ENTRY"],
            "other_or_no_entry": len(items) - sum(counts[x] for x in
                ("TP", "TIMEOUT", "OPEN", "WAIT_ENTRY")),
            "resolved_tp_pct": round(100 * counts["TP"] / closed, 1) if closed else None,
        }
    # Only new signal-time sizing decisions belong to the prospective policy
    # cohort; old records are kept in the original risk-only cohort above.
    prospective = [r for r in selected
                   if (r.get("failure_shadow") or {}).get("sizing_policy_version") == SIZING_VERSION]
    exclusive = {
        "zero_flags_full": [r for r in prospective if r["failure_shadow"].get("known_flags")==4
                            and r["failure_shadow"].get("flag_count")==0],
        "one_flag_full": [r for r in prospective if r["failure_shadow"].get("known_flags")==4
                           and r["failure_shadow"].get("flag_count")==1],
        "two_flags_half": [r for r in prospective if r["failure_shadow"].get("known_flags")==4
                            and r["failure_shadow"].get("flag_count")==2],
        "three_flags_thirty": [r for r in prospective if r["failure_shadow"].get("known_flags")==4
                               and r["failure_shadow"].get("flag_count")==3],
        "four_flags_blocked": [r for r in prospective if r["failure_shadow"].get("known_flags")==4
                               and r["failure_shadow"].get("flag_count")==4],
        "incomplete_blocked": [r for r in prospective if r["failure_shadow"].get("known_flags",0)<4],
    }
    sizing = {}
    for name, items in exclusive.items():
        status = Counter(r.get("tracking_status") for r in items)
        orders = [executor_orders[r["signal_id"]] for r in items
                  if r.get("signal_id") in executor_orders
                  and executor_orders[r["signal_id"]].get("risk_sizing_version") == SIZING_VERSION]
        sizing[name] = {
            "signals": len(items),
            "paper_tp": status["TP"],
            "paper_timeout": status["TIMEOUT"],
            "paper_open": status["OPEN"],
            "real_orders_created": len(orders),
            "real_target_usdt_total": round(sum(float(o.get("risk_target_usdt",0))
                                                for o in orders), 4),
            "real_order_phases": dict(Counter(o.get("phase","UNKNOWN") for o in orders)),
        }
    return {
        "version": VERSION,
        "sizing_policy_version": SIZING_VERSION,
        "v2_gating": False,
        "live_entry_gate": "0/1 full; 2 half; 3 thirty percent; 4 or unknown blocked",
        "groups": result,
        "sizing_cohorts_prospective": sizing,
        "notes": "Cohorts overlap. OPEN is unresolved, not a loss. Resolved TP excludes OPEN. "
                 "Only new 4-flag labels qualify for prospective sizing; 0/1 risk flags full, two half, three 30 percent, four or unknown block real BUY. "
                 "Cohorts are selection-biased and not calibrated TP probabilities.",
    }


def export_csv(records, executor_orders=None):
    """CSV research observations tied to immutable candidate-time flags."""
    fields = [
        "signal_id", "symbol", "created_at", "tracking_model", "tracking_status",
        "entered_at", "finished_at", "score", "green_volume_share_1h",
        "volume_ratio_vs_prior_12h", "rsi_delta_1h",
        "ema25_slope_change_1h_pp", "repeat_prior_entry_drop_pct", "atr14_pct",
        "weak_volume", "falling_rsi_ema", "repeat_lower_entry", "low_atr_15m", "flag_count",
        "known_flags", "risk_sizing_version", "risk_position_fraction",
        "risk_position_action", "real_order_phase", "real_configured_usdt",
        "real_target_usdt", "real_effective_notional_usdt",
        "real_order_quantity", "real_filled_quantity",
        "max_adverse_pct", "max_favorable_pct", "tp_hours",
    ]
    executor_orders = executor_orders or {}
    buffer = io.StringIO()
    out = csv.DictWriter(buffer, fieldnames=fields)
    out.writeheader()
    for record in records:
        shadow = record.get("failure_shadow") or {}
        if shadow.get("version") not in (VERSION, LEGACY_VERSION):
            continue
        metrics, flags = shadow.get("metrics") or {}, shadow.get("flags") or {}
        strength = (record.get("v2_features") or {}).get("strength_score") or {}
        values = {k: record.get(k) for k in fields}
        fraction = shadow.get("position_fraction") if shadow.get("sizing_policy_version") in (SIZING_VERSION, LEGACY_SIZING_VERSION) else None
        values.update(score=strength.get("score"),
                      flag_count=shadow.get("flag_count"),
                      known_flags=shadow.get("known_flags"),
                      risk_sizing_version=shadow.get("sizing_policy_version"),
                      risk_position_fraction=fraction,
                      risk_position_action=("UNKNOWN" if fraction is None else
                                            "BLOCK" if fraction == 0 else
                                            "HALF" if fraction == 0.5 else
                                             "THIRTY" if fraction == 0.3 else "FULL"))
        order = executor_orders.get(record.get("signal_id")) or {}
        if order.get("risk_sizing_version") in (SIZING_VERSION, LEGACY_SIZING_VERSION):
            values.update(real_order_phase=order.get("phase"),
                          real_configured_usdt=order.get("risk_configured_usdt"),
                          real_target_usdt=order.get("risk_target_usdt"),
                          real_effective_notional_usdt=order.get("risk_effective_notional_usdt"),
                          real_order_quantity=order.get("quantity"),
                          real_filled_quantity=order.get("filled_qty"))
        for k in metrics:
            if k in values:
                values[k] = metrics[k]
        for k in flags:
            if k in values:
                values[k] = flags[k]
        out.writerow(values)
    return buffer.getvalue()
