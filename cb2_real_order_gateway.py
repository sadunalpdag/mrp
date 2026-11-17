"""Independent CB2 live LIMIT entry preflight; no order submission.

This module deliberately NEVER imports or mutates the Crash executor.
It validates a fresh 15m three-filter signal and computes a marketable
LIMIT BUY at the current Binance best ask (rounded UP to tick size).
A future independent executor must own durable idempotent BUY/TP and
independent durable BUY/TP reconciliation before live activation.
"""
import json
import math
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
D = Decimal
MAX_AGE_SECONDS = 120


class PreflightBlocked(ValueError):
    pass


def _dec(value):
    try:
        number = D(str(value))
    except (ValueError, ArithmeticError, TypeError):
        raise PreflightBlocked("INVALID_NUMBER") from None
    if not number.is_finite():
        raise PreflightBlocked("NON_FINITE_NUMBER")
    return number


def _utc_seconds(value):
    from datetime import timezone
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise PreflightBlocked("INVALID_SIGNAL_TIME")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timestamp without UTC offset")
        return stamp.astimezone(timezone.utc).timestamp()
    except (ValueError, OverflowError):
        raise PreflightBlocked("INVALID_SIGNAL_TIME") from None


def validate_fresh_momentum(item, now):
    if not isinstance(item, dict) or not item.get("id") or not item.get("symbol"):
        raise PreflightBlocked("INVALID_CANDIDATE")
    if item.get("filters_passed") is not True:
        raise PreflightBlocked("FILTER_NOT_APPROVED")
    data = item.get("indicators")
    if not isinstance(data, dict):
        raise PreflightBlocked("INDICATORS_MISSING")
    if not (_dec(item.get("sqs")) >= 70
            and _dec(data.get("adx14")) >= 20
            and _dec(data.get("change24_pct")) >= 10
            and _dec(data.get("macd_histogram")) > 0):
        raise PreflightBlocked("THREE_FILTERS_NOT_MET")
    from crash_policy import supported_symbol
    if not supported_symbol(item["symbol"]):
        raise PreflightBlocked("INVALID_SYMBOL")
    # A 15-minute polling interval does not authorize buying a stale signal.
    observed = _utc_seconds(item.get("observed_at"))
    if not 0 <= now - observed <= MAX_AGE_SECONDS:
        raise PreflightBlocked("SIGNAL_TOO_OLD")
    return item["symbol"]


def read_crash_snapshot(root, now):
    """Read live Crash settings and ledger WITHOUT modifying Crash files."""
    path = Path(root) / "crash_night_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raise PreflightBlocked("CRASH_STATE_UNAVAILABLE") from None
    if not isinstance(state, dict) or not isinstance(state.get("orders"), dict):
        raise PreflightBlocked("CRASH_STATE_INVALID")
    if state.get("halt"):
        raise PreflightBlocked("CRASH_HALTED")
    if state.get("paused") is not False:
        raise PreflightBlocked("CRASH_AUTO_PAUSED")
    settings = state.get("trade_settings")
    if not isinstance(settings, dict):
        raise PreflightBlocked("CRASH_SETTINGS_MISSING")
    values = {}
    for key in ("max_daily", "max_symbols", "max_open_positions", "max_pending_entries"):
        try:
            values[key] = int(settings[key])
        except (ValueError, TypeError, KeyError):
            raise PreflightBlocked("CRASH_LIMIT_INVALID") from None
        if not 1 <= values[key] <= 50:
            raise PreflightBlocked("CRASH_LIMIT_INVALID")
    amount = _dec(settings.get("position_usdt"))
    if not D("1") <= amount <= D("1000"):
        raise PreflightBlocked("CRASH_AMOUNT_INVALID")
    date = datetime.fromtimestamp(now, PARIS).date().isoformat()
    orders = list(state["orders"].values())
    used = sum(1 for order in orders if isinstance(order, dict)
               and order.get("day") == date
               and order.get("entry_never_sent") is not True)
    override = state.get("daily_counter_override") or {}
    if override.get("day") == date:
        try:
            used = max(0, int(override["count"]) + used - int(override["raw_at_set"]))
        except (ValueError, TypeError, KeyError):
            raise PreflightBlocked("CRASH_COUNTER_INVALID") from None
    values.update(crash_used=used, day=date, amount_usdt=amount)
    return values


def gate_shared_capacity(snapshot, cb2_orders, active_positions, opening_orders,
                         current_symbol):
    """One-way CB2 preflight; does NOT make Crash count CB2 future entries."""
    if not isinstance(snapshot, dict) or not isinstance(cb2_orders, (list, tuple)):
        raise PreflightBlocked("CAPACITY_MISSING")
    if not isinstance(active_positions, (set, list, tuple)) or not isinstance(opening_orders, (set, list, tuple)):
        raise PreflightBlocked("LIVE_CAPACITY_MISSING")
    active = set(active_positions)
    pending = set(opening_orders)
    if current_symbol in active or current_symbol in pending:
        raise PreflightBlocked("SYMBOL_ALREADY_ACTIVE")
    own = [o for o in cb2_orders if isinstance(o, dict)
           and o.get("day") == snapshot["day"]
           and o.get("entry_never_sent") is not True]
    # Operator explicitly permitted CB2 to exceed the Crash-only daily 50
    # counter; retain aggregate usage for visibility, not an entry gate.
    # Crash's own limit remains unchanged.
    if len(active) >= snapshot["max_open_positions"]:
        raise PreflightBlocked("MAX_OPEN_POSITIONS")
    if len(pending) >= snapshot["max_pending_entries"]:
        raise PreflightBlocked("MAX_PENDING_ENTRIES")
    if len(active | pending) >= snapshot["max_symbols"]:
        raise PreflightBlocked("MAX_SYMBOLS")
    return True


def plan_ask_limit(symbol, ask_price, exchange_info, position_usdt, tp_pct="0.01"):
    """Produce quote-time LIMIT BUY + expected TP. Do not submit an order."""
    if not isinstance(exchange_info, dict) or exchange_info.get("symbol") != symbol:
        raise PreflightBlocked("EXCHANGE_INFO_SYMBOL_MISMATCH")
    if exchange_info.get("status") != "TRADING" or exchange_info.get("contractType") != "PERPETUAL":
        raise PreflightBlocked("SYMBOL_NOT_TRADING")
    ask = _dec(ask_price)
    amount = _dec(position_usdt)
    if ask <= 0 or not 1 <= amount <= 1000:
        raise PreflightBlocked("INVALID_ASK_OR_AMOUNT")
    if _dec(tp_pct) != D("0.01"):
        raise PreflightBlocked("TP_MUST_BE_ONE_PERCENT")
    filters = {f["filterType"]: f for f in exchange_info.get("filters", []) if isinstance(f, dict) and "filterType" in f}
    try:
        pf = filters["PRICE_FILTER"]
        lot = filters["LOT_SIZE"]
        tick, step = _dec(pf["tickSize"]), _dec(lot["stepSize"])
        if tick <= 0 or step <= 0:
            raise PreflightBlocked("INVALID_SYMBOL_FILTERS")
        entry = (ask / tick).to_integral_value(rounding=ROUND_UP) * tick
        target = (entry * D("1.01") / tick).to_integral_value(rounding=ROUND_UP) * tick
        qty = (amount / entry / step).to_integral_value(rounding=ROUND_DOWN) * step
        minimum = _dec(filters.get("MIN_NOTIONAL", {}).get("notional", "0"))
        if (qty <= 0 or qty < _dec(lot["minQty"]) or qty > _dec(lot["maxQty"])
                or entry < _dec(pf["minPrice"]) or entry > _dec(pf["maxPrice"])
                or target > _dec(pf["maxPrice"]) or qty * entry < minimum):
            raise PreflightBlocked("BINANCE_SYMBOL_FILTER_FAILED")
    except (KeyError, TypeError):
        raise PreflightBlocked("SYMBOL_FILTER_MISSING") from None
    return {"symbol": symbol, "side": "BUY", "type": "LIMIT", "timeInForce": "GTC",
            "price": format(entry, "f"), "quantity": format(qty, "f"),
            "target_tp": format(target, "f"), "leverage": 10,
            "mode": "PREFLIGHT_ONLY_NO_BINANCE_ORDER"}


def make_read_only_proposal(item, now, crash_root, cb2_orders, active_symbols,
                            pending_symbols, ask_price, exchange_info):
    """A fully validated plan; fail closed when account/settings are unavailable."""
    symbol = validate_fresh_momentum(item, now)
    snapshot = read_crash_snapshot(crash_root, now)
    gate_shared_capacity(snapshot, cb2_orders, active_symbols, pending_symbols, symbol)
    proposal = plan_ask_limit(symbol, ask_price, exchange_info,
                              snapshot["amount_usdt"])
    proposal["daily_usage_before"] = snapshot["crash_used"] + sum(
        1 for order in cb2_orders if isinstance(order, dict)
        and order.get("day") == snapshot["day"]
        and order.get("entry_never_sent") is not True)
    proposal["max_daily"] = snapshot["max_daily"]
    proposal["signal_id"] = item["id"]
    return proposal
