"""Shared Crash Bounce tracking horizon and new-candidate symbol policy."""
import re
import math

# Unfilled candidates expire 24h after creation; virtual OPEN trades retain
# the pre-existing 150h post-entry TP/drawdown observation window.
TRACKING_SECONDS = 24 * 3600
POST_ENTRY_TRACKING_SECONDS = 150 * 3600


def is_active_candidate(record, now):
    """Still waiting for entry (<24h), or entered and still awaiting TP."""
    if not isinstance(record, dict):
        return False
    created = record.get("created_at")
    if (not isinstance(created, (int, float)) or isinstance(created, bool)
            or not math.isfinite(created) or created > now):
        return False
    if record.get("tracking_status") == "OPEN":
        # Keep even older OPEN records until the tracking loop confirms their
        # post-entry timeout, never pruning them before timeout assessment.
        return True
    return 0 <= now-created < TRACKING_SECONDS

def prune_active_state(state, now):
    """Discard expired Crash V2 candidates, not real exchange order state.

    The executor owns actual open orders separately. Shared five-day OHLCV
    caches are intentionally retained to avoid repeatedly downloading bars.
    """
    old_history = state.get("history") or []
    old_symbols = state.get("symbols") or {}
    state["history"] = [rec for rec in old_history if is_active_candidate(rec, now)]
    state["symbols"] = {s: rec for s, rec in old_symbols.items() if is_active_candidate(rec, now)}
    return (len(old_history)-len(state["history"]) +
            len(old_symbols)-len(state["symbols"]))


def supported_symbol(symbol):
    """Only ASCII USDT symbols; never normalize an exchange symbol."""
    return (isinstance(symbol, str) and symbol.endswith('USDT')
            and re.fullmatch(r'[A-Z0-9_]{1,40}', symbol) is not None)
