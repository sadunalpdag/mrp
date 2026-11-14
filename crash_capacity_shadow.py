"""Paper-only tracking for Crash Bounce signals rejected by live capacity gates.

Copies the already-validated V2 entry trajectory into an independent virtual
track with the risk-tier TP. Never calls an exchange, sends an order or alters
real executor state. Trade prints are observations, not guaranteed fills.
"""
import math
from crash_live_v2 import MODEL, advance_trade
from crash_policy import TRACKING_SECONDS, POST_ENTRY_TRACKING_SECONDS

REASONS = frozenset(("OPEN_LIMIT_REACHED", "PENDING_LIMIT_REACHED"))
ACTIVE = frozenset(("WAIT_ENTRY", "OPEN"))


def attach(record, reason, planned, target_usdt, tp_pct, observed_at):
    """Attach once by original signal, preserving its existing V2 paper result."""
    if reason not in REASONS or record.get("capacity_virtual") is not None:
        return False
    status = record.get("tracking_status")
    if status not in ACTIVE or record.get("tracking_model") != MODEL:
        return False
    try:
        entry = float(planned["price"])
        tp = float(planned["tp"])
        bottom = float(record["reference_low"])
        created = float(record["created_at"])
        observed_at = float(observed_at)
        if not (all(math.isfinite(x) for x in (entry, tp, bottom, created, observed_at))
                and entry > 0 and tp > entry and bottom > 0 and created <= observed_at):
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    shadow = {
        "tracking_model": MODEL,
        "tracking_status": status,
        "entry": entry, "tp": tp, "reference_low": bottom,
        "created_at": created,
        "previous_watch_price": record.get("previous_watch_price", record.get("last_price", entry)),
        "ws_last_trade_id": record.get("ws_last_trade_id"),
        "ws_last_event_at": record.get("ws_last_event_at", created),
        "samples": 0, "coverage_gaps": 0, "max_gap_seconds": 0,
        "reason": reason, "risk_tp_pct": str(tp_pct),
        "planned_notional_usdt": str(target_usdt),
        "attached_at": observed_at, "result_source": "WS_AGGTRADE_PAPER_ONLY",
    }
    if status == "OPEN":
        # A qualifying V2 virtual entry has already been observed. Reuse its
        # exact entry timestamp and trade ID, never backdate new samples.
        if record.get("entered_at") is None or record.get("entry_trade_id") is None:
            return False
        shadow["entered_at"] = record["entered_at"]
        shadow["entry_trade_id"] = record["entry_trade_id"]
        shadow["max_favorable_pct"] = record.get("max_favorable_pct", 0)
        shadow["max_adverse_pct"] = record.get("max_adverse_pct", 0)
    record["capacity_virtual"] = shadow
    return True


def advance(record, price, observed_at, trade_id, force_gap=False):
    """Read a shared WS aggTrade without touching live orders or V2 status."""
    shadow = record.get("capacity_virtual")
    if not isinstance(shadow, dict) or shadow.get("tracking_status") not in ACTIVE:
        return None
    if force_gap:
        shadow["ws_last_event_at"] = observed_at - 60
        shadow["coverage_gaps"] = int(shadow.get("coverage_gaps", 0)) + 1
    return advance_trade(shadow, price, observed_at, trade_id)


def expire(record, now):
    """Resolve only the independent paper horizon; no made-up TP fills."""
    shadow = record.get("capacity_virtual")
    if not isinstance(shadow, dict):
        return None
    status = shadow.get("tracking_status")
    if status not in ACTIVE:
        return None
    start = float(shadow["created_at"] if status == "WAIT_ENTRY" else shadow["entered_at"])
    deadline = start + (TRACKING_SECONDS if status == "WAIT_ENTRY" else POST_ENTRY_TRACKING_SECONDS)
    if now < deadline:
        return None
    result = "NO_ENTRY_OBSERVED" if status == "WAIT_ENTRY" else "TIMEOUT"
    shadow["tracking_status"] = result
    shadow["finished_at"] = deadline
    if result == "TIMEOUT" and shadow.get("last_observed_price"):
        shadow["observed_return_pct"] = 100 * (
            float(shadow["last_observed_price"]) / float(shadow["entry"]) - 1)
    return result


def summarize(records):
    """Resolved TP denominator excludes pending and never-filled signals."""
    items = [r["capacity_virtual"] for r in records
             if isinstance(r, dict) and isinstance(r.get("capacity_virtual"), dict)
             and r["capacity_virtual"].get("reason") in REASONS]
    counts = {status: sum(r.get("tracking_status") == status for r in items)
              for status in ("WAIT_ENTRY", "OPEN", "TP", "TIMEOUT", "NO_ENTRY_OBSERVED",
                             "INVALIDATED_BEFORE_ENTRY", "MISSED_SIGNAL")}
    resolved = counts["TP"] + counts["TIMEOUT"]
    tp_minutes = [(r["finished_at"] - r["entered_at"]) / 60 for r in items
                  if r.get("tracking_status") == "TP" and r.get("entered_at") is not None]
    mae = [r.get("max_adverse_pct") for r in items
           if r.get("tracking_status") in ("TP", "TIMEOUT", "OPEN")
           and isinstance(r.get("max_adverse_pct"), (int, float))]
    return {
        "total": len(items),
        "open_limit": sum(r.get("reason") == "OPEN_LIMIT_REACHED" for r in items),
        "pending_limit": sum(r.get("reason") == "PENDING_LIMIT_REACHED" for r in items),
        **{k.lower(): v for k, v in counts.items()},
        "tp_rate_resolved_pct": round(100 * counts["TP"] / resolved, 1) if resolved else None,
        "avg_tp_minutes": round(sum(tp_minutes) / len(tp_minutes), 1) if tp_minutes else None,
        "worst_mae_pct": round(min(mae), 2) if mae else None,
    }
