"""Shared LOW PRIORITY Binance REST gateway for paper-only HBREAK and API2.

Crash Bounce historical candle research has a larger weight allowance; live
Binance order/account/TP reconciliation has much larger reserved headroom.
Only public Binance futures data calls use this gateway. CoinBeacon paid
queries do not consume Binance REQUEST_WEIGHT but paper workers defer those
too while the gateway is unavailable, avoiding wasted paid credits.

No secrets, trading actions, order placement or retroactive paper fills.
"""
import threading
import time
from urllib.parse import urlsplit

from crash_rest_budget import BUDGET, PAPER_CEILING, RateDeferred, estimated_weight

_GATE_LOCK = threading.RLock()
_CRASH_URGENT = None
URL_HOST = "fapi.binance.com"


class PaperRestDeferred(RuntimeError):
    def __init__(self, reason, wait=0):
        self.reason = str(reason)
        self.wait = max(0, float(wait))
        super().__init__(f"PAPER_REST_DEFERRED_{self.reason}")


def set_crash_priority_checker(checker):
    """Register a read-only process-local callback from the crash worker."""
    if checker is not None and not callable(checker):
        raise TypeError("checker must be callable or None")
    global _CRASH_URGENT
    with _GATE_LOCK:
        _CRASH_URGENT = checker


def _crash_busy():
    with _GATE_LOCK:
        checker = _CRASH_URGENT
    if checker is None:
        return False
    try:
        # If the crash health callback itself fails, preserve its priority.
        return bool(checker())
    except Exception:
        return True


def paper_rest_available():
    """Cheap preflight; paid paper scans may be deferred before using credits."""
    if _crash_busy():
        return False
    stats = BUDGET.stats()
    return (stats["blocked_until"] <= time.time()
            and max(stats["server_weight"], stats["reserved_weight"]) < PAPER_CEILING)


def paper_get(http, url, params=None, timeout=12):
    """Get one public futures market endpoint, never an order or signed API.

    Declined requests send zero HTTP calls. Paper workers must treat
    PaperRestDeferred as data postponed, not an invalid candle or loss.
    """
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != URL_HOST
            or parts.path not in ("/fapi/v1/klines", "/fapi/v1/ticker/price")
            or parts.query or parts.fragment or parts.username or parts.password):
        raise ValueError("PAPER_ONLY_PUBLIC_FUTURES_GET_ALLOWED")
    if _crash_busy():
        raise PaperRestDeferred("FRESH_CRASH_SIGNAL")
    weight = estimated_weight(parts.path, params or {}, "GET")
    try:
        BUDGET.reserve("paper", weight)
    except RateDeferred as exc:
        raise PaperRestDeferred(exc.reason, exc.seconds) from exc
    # Crash priority may have changed while waiting for the paced budget.
    if _crash_busy():
        raise PaperRestDeferred("FRESH_CRASH_SIGNAL")
    result = http.get(url, params=params, timeout=timeout)
    # Capture any exchange-side IP weight/ban, even for a failed paper GET.
    BUDGET.observe(getattr(result, "headers", {}),
                   getattr(result, "status_code", 0),
                   (getattr(result, "headers", {}) or {}).get("Retry-After", 0))
    return result
