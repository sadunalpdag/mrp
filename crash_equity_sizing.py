"""Daily equity-proportional trade NOTIONAL, based on the existing paper observer.

This module NEVER calls Binance, never sends orders and never changes trading
state itself. The executor alone commits one immutable size per Paris day.
"""
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
MAX_STALENESS_SECONDS = 600
MIN_USDT = Decimal("1")
MAX_USDT = Decimal("1000")


def decimal(value):
    try:
        num = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("INVALID_EQUITY_NUMBER") from exc
    if not num.is_finite() or num <= 0:
        raise ValueError("INVALID_EQUITY_NUMBER")
    return num


def read_equity(path, now):
    """Return (equity, Paris_day, epoch_seconds) or None if not fresh.

    Strictly require observed equity to be recent. No balance extrapolation,
    no use of all-time high, and no setting a day from stale 23:59 data.
    """
    try:
        with Path(path).open("r", encoding="utf8") as handle:
            payload = json.load(handle)
        if payload.get("mode") != "PAPER_ONLY":
            return None
        sampled = datetime.fromisoformat(payload["sampled_at"])
        if sampled.tzinfo is None:
            return None
        observed_at = sampled.timestamp()
        if not 0 <= float(now) - observed_at <= MAX_STALENESS_SECONDS:
            return None
        eq = decimal(payload["last_equity"])
        return eq, datetime.fromtimestamp(observed_at, PARIS).date().isoformat(), observed_at
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def proportional_amount(previous_amount, previous_equity, current_equity):
    """Round the new entire-order budget to the nearest integer (HALF_UP).

    500 USDT x 1.03 = 515, x 0.97 = 485.
    Clamps to the executor's preexisting 1–1000 USDT entry budget range.
    """
    value = decimal(previous_amount) * decimal(current_equity) / decimal(previous_equity)
    integer = value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return min(MAX_USDT, max(MIN_USDT, integer))
