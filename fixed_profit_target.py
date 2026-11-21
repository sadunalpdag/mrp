"""Persisted, order-free fixed profit target state machine.

Inputs must be validated USDT-M account equity with deposits/withdrawals removed.
This module never submits/cancels orders and cannot activate live trading by itself.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")


def _money(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("INVALID_EQUITY_OR_PERCENT") from exc
    if not result.is_finite():
        raise ValueError("NON_FINITE_VALUE")
    return result


def _day(instant):
    if not isinstance(instant, datetime) or instant.tzinfo is None:
        raise ValueError("TIMEZONE_AWARE_INSTANT_REQUIRED")
    return instant.astimezone(PARIS).date().isoformat()


def initial_state(equity, percent, instant):
    equity, pct = _money(equity), _money(percent)
    if equity <= 0 or not Decimal("0.5") <= pct <= Decimal("20"):
        raise ValueError("INVALID_BASE_OR_PERCENT")
    return {
        "version": 1,
        "phase": "WAIT_TARGET",
        "day": _day(instant),
        "reference": str(equity),
        "target": str(equity * (Decimal("1") + pct / 100)),
        "pct": str(pct),
        "triggered_day": None,
        "completed_day": None,
        "completed": False,
    }


def observe(state, equity, instant):
    """Returns (new_state, trigger_now), no order side effects.

    A missed goal never moves, even after midnight; active close locks cannot
    expire at midnight or reset on restart.
    """
    equity = _money(equity)
    if equity <= 0:
        raise ValueError("INVALID_EQUITY")
    new = dict(state)
    today = _day(instant)
    if new["phase"] == "WAIT_TARGET":
        if equity >= _money(new["target"]):
            new.update(phase="CLOSING", triggered_day=today, completed=False)
            return new, True
        new["day"] = today
        return new, False
    if new["phase"] in ("CLOSING", "WAIT_REBASE", "WAIT_MIDNIGHT"):
        return new, False
    raise ValueError("INVALID_PHASE")


def closing_verified(state, positions_flat, owned_entries_cleared,
                     closing_orders_cleared, instant):
    """The broker adapter must independently verify all three conditions."""
    new = dict(state)
    if new["phase"] != "CLOSING":
        return new
    if not (positions_flat and owned_entries_cleared and closing_orders_cleared):
        return new
    new["phase"] = "WAIT_REBASE"
    new["completed"] = True
    new["completed_day"] = _day(instant)
    return new


def roll_after_midnight(state, closing_equity, percent, instant,
                        positions_flat, owned_entries_cleared,
                        closing_orders_cleared):
    """Advance immediately after broker-confirmed closure, without midnight wait.

    WAIT_MIDNIGHT is accepted for migration of already completed cycles.
    """
    new = dict(state)
    if new["phase"] not in ("WAIT_REBASE", "WAIT_MIDNIGHT"):
        return new
    if not (positions_flat and owned_entries_cleared and closing_orders_cleared):
        return new
    equity, pct = _money(closing_equity), _money(percent)
    if equity <= 0 or not Decimal("0.5") <= pct <= Decimal("20"):
        raise ValueError("INVALID_NEW_BASE_OR_PERCENT")
    return initial_state(equity, pct, instant)
