"""Cached, retrospective Crash V2 checks. Never creates signals or orders.

This path only annotates historical candidates that expired before the five-day
history arrived. The decision uses the ORIGINAL candidate time and closed bars.
"""
import math

RETRY_SECONDS = 300
HISTORY_HORIZON_SECONDS = 24 * 3600


def reassess_expired(state, now, validate, log):
    """Update research-only snapshots. Return count of changed historical records.

    NEVER change live eligibility, delivery, tracking, or execution fields.
    Real BUYs remain restricted to the independent original 120s entry window.
    """
    from crash_failure_shadow import assess
    history = state.get("history") or []
    symbols = state.get("symbols") or {}
    changed = 0
    for record in history:
        if not isinstance(record, dict):
            continue
        if (record.get("v2_status") != "MISSING_5D_HISTORY"
                or record.get("v2_reason") != "HISTORY_NOT_READY_IN_TIME"
                or record.get("v2_delivery_phase") != "CANDIDATE"):
            continue
        created = record.get("created_at")
        if (isinstance(created, bool) or not isinstance(created, (float, int))
                or not math.isfinite(created) or not 0 <= now-created < HISTORY_HORIZON_SECONDS):
            continue
        previous = record.get("retrospective_v2") or {}
        if previous.get("status") in ("STRONG_V2", "FILTERED_V2"):
            continue
        if now < previous.get("checked_at", 0) + RETRY_SECONDS:
            continue
        try:
            result = validate(record, created)
            if not isinstance(result, dict):
                continue
            status = result.get("status")
            if status not in ("MISSING_5D_HISTORY", "STRONG_V2", "FILTERED_V2"):
                continue
            retrospective = {
                "status": status,
                "candles": result.get("candles"),
                "missing_reason": result.get("missing_reason"),
                "checked_at": now,
                "research_only": True,
            }
            if (status in ("STRONG_V2", "FILTERED_V2")
                    and isinstance(result.get("strength_score"), dict)):
                # Risk columns and power score are historical observations,
                # also valid for FILTERED candidates. They never change the
                # original signal status, delivery, or real BUY eligibility.
                # All calculations use the ORIGINAL candidate time, not now.
                retrospective["strength_score"] = result["strength_score"]
                research_record = dict(record, v2_status=status,
                                       v2_features=result)
                risk = assess(research_record, history)
                record["failure_shadow"] = risk
                retrospective["known_risk_flags"] = risk["known_flags"]
            record["retrospective_v2"] = retrospective
            latest = symbols.get(record.get("symbol"))
            if isinstance(latest, dict) and latest.get("signal_id") == record.get("signal_id"):
                latest.update(record)
            changed += 1
            if status != "MISSING_5D_HISTORY":
                log(f"[CRASH RETROSPECTIVE V2] {record.get('symbol')}; "
                    f"original_as_of={created}; status={status}; "
                    "research_only=1; no Telegram; no BUY")
        except Exception as error:
            log(f"[CRASH RETROSPECTIVE ERR] {record.get('symbol')}; "
                f"reason={type(error).__name__}")
    return changed
