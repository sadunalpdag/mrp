"""Read-only daily observed equity, BTC comparison and verified order results.

Works only from the local paper account observer, public WS ticker snapshots and
persisted executor order states. It NEVER signs Binance requests or changes orders.
Missing historical samples are not backfilled or invented.
"""
import json
import math
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
FILE_NAME = "crash_daily_performance.json"
OBSERVER = "daily_profit_lock_paper.json"
MAX_AGE_SECONDS = 3600
KEEP_DAYS = 60
HEADERS = [
    "Tarih (Paris)", "İlk gözlenen sermaye USDT", "Gözlenen günlük zirve USDT",
    "Son gözlenen sermaye USDT", "Sermaye değişimi USDT",
    "Sermaye değişimi %", "BTC başlangıç USDT (eşzamanlı)",
    "BTC son USDT (eşzamanlı)", "BTC değişimi % (eşzamanlı)",
    "Sermaye değişimi % (BTC aralığı)", "Fark (yüzde puan)",
    "İlk örnek (Paris)", "Son örnek (Paris)", "Günün veri durumu",
    "Gerçek dolan giriş", "Gerçek TP (borsa onaylı)", "Haricen kapanan",
    "Gerçek açık / TP belirsiz", "Gerçek TP payı % (sonuçlanan)",
    "Not",
]


def _decimal(value):
    try:
        result = Decimal(str(value))
        if not result.is_finite() or result <= 0:
            return None
        return result
    except (InvalidOperation, TypeError, ValueError):
        return None


def _rounded(value, places="0.01"):
    return float(value.quantize(Decimal(places), rounding=ROUND_HALF_UP))


def _percent(start, end):
    base, last = _decimal(start), _decimal(end)
    if base is None or last is None:
        return None
    return _rounded((last / base - 1) * 100)


def _observer(directory, now):
    try:
        with (Path(directory) / OBSERVER).open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("mode") != "PAPER_ONLY":
            return None
        instant = datetime.fromisoformat(data["sampled_at"])
        if instant.tzinfo is None:
            return None
        stamp = instant.timestamp()
        if not 0 <= now - stamp <= MAX_AGE_SECONDS:
            return None
        equity = _decimal(data.get("last_equity"))
        if equity is None:
            return None
        return instant.astimezone(PARIS), equity
    except (OSError, ValueError, OverflowError, TypeError, KeyError):
        return None


def _btc_price(tickers):
    for ticker in tickers or []:
        if isinstance(ticker, dict) and ticker.get("symbol") == "BTCUSDT":
            return _decimal(ticker.get("lastPrice"))
    return None


def _load(path):
    try:
        with path.open(encoding="utf-8") as handle:
            obj = json.load(handle)
        if obj.get("version") != 1 or not isinstance(obj.get("days"), dict):
            return {}
        return {key: val for key, val in obj["days"].items()
                if isinstance(key, str) and isinstance(val, dict)}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _store(path, days):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump({"version": 1, "days": days}, handle, ensure_ascii=False,
                  sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _order_counts(executor, date):
    """Only DONE means a confirmed TP fill; NO_FILL is not a trade."""
    if executor is None:
        return None
    state = getattr(executor, "state", None)
    if not isinstance(state, dict) or not isinstance(state.get("orders"), dict):
        return None
    filled = tp = external = active = 0
    for order in state["orders"].values():
        if not isinstance(order, dict) or order.get("day") != date:
            continue
        phase = order.get("phase")
        fill_quantity = _decimal(order.get("filled_qty"))
        actual_fill = (fill_quantity is not None or phase in
                       ("DONE", "CLOSED_EXTERNALLY", "PROTECTED", "TP_UNKNOWN"))
        if actual_fill:
            filled += 1
        if phase == "DONE":
            tp += 1
        elif phase == "CLOSED_EXTERNALLY":
            external += 1
        elif phase in ("PROTECTED", "TP_UNKNOWN") and actual_fill:
            active += 1
    resolved = tp + external
    ratio = round(100 * tp / resolved, 2) if resolved else None
    return (filled, tp, external, active, ratio)


def update(directory, tickers, executor, now):
    """Record a verified observation and return a sheet-ready table, or None.

    Bot/BTC comparison uses a paired first sample and paired current sample,
    not BTC's rolling 24h change. Missing past data stays missing.
    """
    found = _observer(directory, now)
    if found is None:
        return None
    observed, equity = found
    today = observed.date().isoformat()
    path = Path(directory) / FILE_NAME
    days = _load(path)
    instant = observed.timestamp()
    stamp = observed.isoformat()
    current = days.get(today)
    changed = False
    just_created = current is None
    if current is None:
        current = {
            "first_equity": str(equity), "peak": str(equity), "low": str(equity),
            "last_equity": str(equity), "first_at": stamp, "last_at": stamp,
            "last_ts": instant,
        }
        days[today] = current
        changed = True
    new_observation = instant > float(current.get("last_ts", 0))
    if just_created or new_observation:
        if new_observation:
            changed = True
        current["last_ts"] = instant
        current["last_at"] = stamp
        current["last_equity"] = str(equity)
        peak = _decimal(current.get("peak"))
        current["peak"] = str(max(peak or equity, equity))
        low = _decimal(current.get("low"))
        # Old persisted days have no reliable historical minimum: do not
        # invent it from the last observation. New days collect it from start.
        if low is not None:
            current["low"] = str(min(low, equity))
        price = _btc_price(tickers)
        if price is not None:
            if not current.get("btc_first"):
                current["btc_first"] = str(price)
                current["paired_equity_first"] = str(equity)
                current["paired_first_at"] = stamp
                changed = True
            current["btc_last"] = str(price)
            current["paired_equity_last"] = str(equity)
            changed = True
    for key, day in days.items():
        if key < today and not day.get("closed"):
            day["closed"] = True
            changed = True
    if len(days) > KEEP_DAYS:
        for key in sorted(days)[:-KEEP_DAYS]:
            del days[key]
        changed = True
    if changed:
        _store(path, days)
    rows = [HEADERS]
    for key in sorted(days, reverse=True):
        day = days[key]
        first, peak, last = (_decimal(day.get(name)) for name in
                             ("first_equity", "peak", "last_equity"))
        if first is None or peak is None or last is None:
            continue
        delta = _rounded(last - first)
        pct = _percent(first, last)
        btc_pct = _percent(day.get("btc_first"), day.get("btc_last"))
        paired_pct = _percent(day.get("paired_equity_first"),
                              day.get("paired_equity_last"))
        diff = round(paired_pct - btc_pct, 2) if (
            paired_pct is not None and btc_pct is not None) else "VERİ YOK"
        end = datetime.fromisoformat(day["last_at"]).astimezone(PARIS)
        complete = bool(day.get("closed"))
        coverage = ("ÖNCEKİ GÜN / SON ÖRNEK " if complete else
                    "DEVAM EDİYOR / SON ÖRNEK ") + end.strftime("%H:%M")
        if complete and end.hour != 23:
            coverage = "EKSİK GÜN SONU / SON ÖRNEK " + end.strftime("%H:%M")
        orders = _order_counts(executor, key)
        rows.append([
            key, _rounded(first), _rounded(peak), _rounded(last),
            delta, pct, _rounded(_decimal(day["btc_first"])) if day.get("btc_first") else "VERİ YOK",
            _rounded(_decimal(day["btc_last"])) if day.get("btc_last") else "VERİ YOK",
            btc_pct if btc_pct is not None else "VERİ YOK",
            paired_pct if paired_pct is not None else "VERİ YOK", diff,
            datetime.fromisoformat(day["first_at"]).astimezone(PARIS).strftime("%d.%m %H:%M"),
            end.strftime("%d.%m %H:%M"), coverage,
            *(orders if orders is not None else ("VERİ YOK",) * 5),
            "Gözlenen futures özsermayesi; transfer etkisi ayrıştırılmadı. Eksik saatler tahmin edilmedi.",
        ])
    return rows
