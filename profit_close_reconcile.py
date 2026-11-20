"""Pure Binance closing-cycle reconciliation for fixed profit lock.

No network calls. No side effects. Unknown/manual positions and orders are
explicitly reported instead of being silently cancelled or declared filled.
"""
from decimal import Decimal, InvalidOperation


def amount(value):
    try:
        x = Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError("BAD_EXCHANGE_NUMBER") from exc
    if not x.is_finite():
        raise ValueError("BAD_EXCHANGE_NUMBER")
    return x


def normalize_positions(rows):
    if not isinstance(rows, list):
        raise ValueError("INVALID_POSITION_SNAPSHOT")
    result = {}
    for p in rows:
        if not isinstance(p, dict) or "symbol" not in p:
            raise ValueError("INVALID_POSITION_ROW")
        qty = amount(p["positionAmt"])
        if qty == 0:
            continue
        side = p.get("positionSide", "BOTH")
        if side not in ("BOTH", "LONG", "SHORT"):
            raise ValueError("INVALID_POSITION_SIDE")
        if side == "LONG" and qty < 0 or side == "SHORT" and qty > 0:
            raise ValueError("INCONSISTENT_POSITION_DIRECTION")
        key = (p["symbol"], side)
        if key in result:
            raise ValueError("DUPLICATE_POSITION_KEY")
        result[key] = qty
    return result


def opening_order(order):
    """Require explicit closing properties, not a best-effort order-type guess."""
    if order.get("reduceOnly") in (True, "true") or order.get("closePosition") in (True, "true"):
        return False
    side = order.get("positionSide", "BOTH")
    if side == "LONG" and order.get("side") == "SELL":
        return False
    if side == "SHORT" and order.get("side") == "BUY":
        return False
    return True


def reconcile_close(snapshot, current_positions, normal_orders, algo_orders,
                    owned_entry_ids, owned_exit_ids):
    """Return verifiable close conditions, never assume missing order = fill.

    Caller must fetch FRESH positions AND both order lists from Binance;
    missing/unparseable data raises and therefore cannot unlock entries.
    """
    positions = normalize_positions(current_positions)
    if not isinstance(snapshot, list) or not isinstance(normal_orders, list) or not isinstance(algo_orders, list):
        raise ValueError("INCOMPLETE_CLOSE_SNAPSHOT")
    expected = {(p["symbol"], p["position_side"]) for p in snapshot}
    if len(expected) != len(snapshot):
        raise ValueError("DUPLICATE_CLOSE_SNAPSHOT")
    # Any position on the account blocks a global all-flat claim.
    remaining = {str(k): str(v) for k, v in positions.items()}
    owned_pending = []
    unknown_open = []
    pending_exits = []
    for o in normal_orders + algo_orders:
        if not isinstance(o, dict) or "symbol" not in o:
            raise ValueError("INVALID_ORDER_ROW")
        client_id = o.get("clientOrderId") or o.get("clientAlgoId")
        if client_id in owned_entry_ids and opening_order(o):
            owned_pending.append((o["symbol"], str(client_id)))
        elif client_id in owned_exit_ids:
            pending_exits.append((o["symbol"], str(client_id)))
        elif opening_order(o):
            unknown_open.append((o["symbol"], str(client_id or "UNKNOWN")))
    return {
        "positions_flat": not positions,
        "owned_entries_cleared": not owned_pending,
        "closing_orders_cleared": not pending_exits,
        "all_clear": not positions and not owned_pending and not pending_exits and not unknown_open,
        "remaining_positions": remaining,
        "owned_pending_entries": owned_pending,
        "pending_exits": pending_exits,
        "unknown_opening_orders": unknown_open,
        "expected_symbols": sorted(expected),
    }
