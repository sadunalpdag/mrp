"""Shared Crash Bounce tracking horizon and new-candidate symbol policy."""
import re
import math

# A candidate is followed for no more than 24h from its ORIGINAL creation.
TRACKING_SECONDS = 24 * 3600

def prune_active_state(state, now):
    """Discard expired Crash V2 candidates, not real exchange order state.

    The executor owns actual open orders separately. Shared five-day OHLCV
    caches are intentionally retained to avoid repeatedly downloading bars.
    """
    def fresh(record):
        if not isinstance(record, dict):
            return False
        at = record.get("created_at")
        return (isinstance(at, (int, float)) and not isinstance(at, bool)
                and math.isfinite(at) and 0 <= now-at < TRACKING_SECONDS)

    old_history = state.get("history") or []
    old_symbols = state.get("symbols") or {}
    state["history"] = [rec for rec in old_history if fresh(rec)]
    state["symbols"] = {s: rec for s, rec in old_symbols.items() if fresh(rec)}
    return (len(old_history)-len(state["history"]) +
            len(old_symbols)-len(state["symbols"]))


def supported_symbol(symbol):
    """Only ASCII USDT symbols; never normalize an exchange symbol."""
    return (isinstance(symbol, str) and symbol.endswith('USDT')
            and re.fullmatch(r'[A-Z0-9_]{1,40}', symbol) is not None)
