"""Display-only, signal-time risk backfill after delayed five-day history.

A late assessment is NEVER written into the live 'failure_shadow' field:
NightExecutor and Telegram admission read only that original safety snapshot.
No order, alert, or V2 status transition happens in this module.
"""
import math

from crash_failure_shadow import FLAG_NAMES, VERSION, assess

_NEXT_CHECK = {}
BACKFILL_INTERVAL_SECONDS = 300
MAX_SIGNAL_AGE_SECONDS = 24 * 3600


def _known_flags(value):
    if not isinstance(value, dict) or value.get("version") != VERSION:
        return 0
    flags = value.get("flags")
    return (sum(type(flags.get(name)) is bool for name in FLAG_NAMES)
            if isinstance(flags, dict) else 0)


def refresh_late_risk(state, now, validate, save, log, limit=8):
    """Backfill missing labels from CLOSED candles at original candidate time.

    Bounded, read-only-data work; independent of the live execution gate.
    Called under the existing state lock, not from the history worker thread.
    """
    updated = 0
    checked = 0
    history = state.get("history") or []
    for record in history:
        if checked >= limit:
            break
        if not isinstance(record, dict) or record.get("v2_status") not in (
                "STRONG_V2", "MISSING_5D_HISTORY"):
            continue
        signal_id = record.get("signal_id")
        created = record.get("created_at")
        if (not isinstance(signal_id, str) or not signal_id
                or isinstance(created, bool) or not isinstance(created, (int, float))
                or not math.isfinite(created) or not 0 <= now-created < MAX_SIGNAL_AGE_SECONDS):
            continue
        # Original, complete signal-time assessment stays the authoritative one.
        if _known_flags(record.get("failure_shadow")) == len(FLAG_NAMES):
            continue
        if _known_flags(record.get("late_failure_shadow")) == len(FLAG_NAMES):
            continue
        if now < _NEXT_CHECK.get(signal_id, 0):
            continue
        _NEXT_CHECK[signal_id] = now + BACKFILL_INTERVAL_SECONDS
        checked += 1
        try:
            result = validate(record, created)
            if not isinstance(result, dict) or result.get("status") != "STRONG_V2":
                continue
            # Validate at 'created', NEVER at now, to prevent future data leakage.
            shadow = assess(dict(record, v2_features=result), history)
            known = _known_flags(shadow)
            if known <= _known_flags(record.get("late_failure_shadow")):
                continue
            record["late_failure_shadow"] = shadow
            record["late_risk_backfilled_at"] = now
            record["late_risk_signal_time"] = created
            latest = (state.get("symbols") or {}).get(record.get("symbol"))
            if isinstance(latest, dict) and latest.get("signal_id") == signal_id:
                latest["late_failure_shadow"] = shadow
                latest["late_risk_backfilled_at"] = now
                latest["late_risk_signal_time"] = created
            log(f"[CRASH LATE RISK] {record.get('symbol')}; known={known}/3; "
                "original_signal_time=1; retrospective_only=1; orders=0")
            updated += 1
        except Exception as error:
            log(f"[CRASH LATE RISK WAIT] {record.get('symbol')}; "
                f"reason={type(error).__name__}")
    if updated:
        save(state)
    if len(_NEXT_CHECK) > 2048:
        for key, until in list(_NEXT_CHECK.items()):
            if until <= now:
                _NEXT_CHECK.pop(key, None)
    return updated
