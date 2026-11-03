"""Authoritative, read-only checks used when a bot TP becomes inactive.

Never infer a closed futures position from a cancelled TP or from a WebSocket
delta. Every decision must use successful, complete Binance REST snapshots.
No orders are placed or cancelled here.
"""
from decimal import Decimal, InvalidOperation


def _rows(value, name):
    if not isinstance(value, list):
        raise ValueError(name + "_NOT_LIST")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError(name + "_INVALID_ROW")
    return value


def _amount(position):
    try:
        value = Decimal(str(position["positionAmt"]))
    except (KeyError, TypeError, InvalidOperation) as error:
        raise ValueError("POSITION_AMOUNT_UNKNOWN") from error
    if not value.is_finite():
        raise ValueError("POSITION_AMOUNT_INVALID")
    return value


def _active_positions(broker):
    return [p for p in _rows(broker.positions(), "POSITIONS") if _amount(p) != 0]


def _open_orders(broker):
    return _rows(broker.open_entry_orders(), "OPEN_ORDERS")


def confirmed_flat(broker, symbol):
    """True only if no position or pending normal/algo order can reopen symbol.

    Two position/order passes narrow (but cannot eliminate) exchange race
    windows. A later preflight is still mandatory before any new order.
    """
    for _ in range(2):
        if any(p.get("symbol") == symbol for p in _active_positions(broker)):
            return False
        if any(o.get("symbol") == symbol for o in _open_orders(broker)):
            return False
        if any(o.get("symbol") == symbol for o in
               _rows(broker.closing_orders(symbol), "CLOSING_ORDERS")):
            return False
    return True


def account_safe_to_resume(broker, tracked):
    """Fail closed on all pending entries, incomplete bot states, or uncovered positions.

    This only authorizes *clearing an old TP halt* following an explicit user
    /otomatik_ac command; it does not submit orders.
    """
    terminal = {"DONE", "NO_FILL", "CLOSED_EXTERNALLY"}
    if any(o.get("phase") not in terminal | {"PROTECTED"} for o in tracked):
        return False
    from crash_manual_tp import tp_coverage

    # Repeat the full account check to reduce TOCTOU windows.
    for _ in range(2):
        active = _active_positions(broker)
        standard = _open_orders(broker)
        algo_fetch = getattr(broker, "open_algo_orders", None)
        if not callable(algo_fetch):
            return False
        algo = _rows(algo_fetch(), "OPEN_ALGO_ORDERS")
        all_orders = standard + algo
        if any(not broker.is_closing_order(o) for o in all_orders):
            return False

        by_symbol = {}
        for p in active:
            symbol, side = p.get("symbol"), p.get("positionSide", "BOTH")
            qty = _amount(p)
            if not symbol or side not in ("BOTH", "LONG") or qty <= 0:
                return False
            entry = Decimal(str(p.get("entryPrice", "0")))
            if not entry.is_finite() or entry <= 0:
                return False
            closing = _rows(broker.closing_orders(symbol), "CLOSING_ORDERS")
            if tp_coverage(closing, symbol, entry, qty, side) != "full":
                return False
            by_symbol[(symbol, side)] = qty

        for o in tracked:
            if o.get("phase") == "PROTECTED":
                key = (o.get("symbol"), o.get("position_side", "BOTH"))
                if by_symbol.get(key, Decimal("0")) <= 0:
                    return False
    return True
