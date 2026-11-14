"""Append-only Bot Status daily account history, 01:05 Europe/Paris.

Reads existing observed equity/BTC history, verified executor order events and
a low-priority, signed GET of Binance Futures account income. NEVER submits or
modifies orders. Missing observations are explicitly identified.
"""
import json
import os
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
ARCHIVE = "crash_bot_status_daily.json"
LEDGER = "crash_daily_performance.json"
START_ROW = 40
HEADERS = [
    "Tarih (Fransa)", "Günlük zirve USDT", "Günlük dip USDT",
    "Önceki gün zirvesi USDT", "Gün sonu - önceki zirve USDT",
    "Gün sonu - önceki zirve %", "Günlük dip - önceki zirve USDT",
    "Günlük dip - önceki zirve %", "BTC değişim % (gözlenen)",
    "Gerçek açılan (bot)", "Gerçek kapanan (bot)",
    "Binance gerçekleşen net K/Z USDT", "Sermaye değişimi USDT",
    "Gün başı sermaye USDT", "Gün sonu sermaye USDT",
    "Veri kapsamı",
]


def _money(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _round(number):
    return float(number.quantize(Decimal("0.01"))) if number is not None else "VERİ YOK"


def _pct(start, end):
    if start is None or end is None or start <= 0:
        return "VERİ YOK"
    return _round((end - start) / start * Decimal("100"))


def _read(path):
    try:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, allow_nan=False, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def _date_of(value):
    ts = _money(value)
    if ts is None or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(float(ts), PARIS).date().isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _counts(executor, target):
    """Only actual fills/confirmed flats with known event timestamps count.

    Prior records without execution timestamps are unknown, never assigned to
    their entry-submission date as though that were their execution date.
    """
    state = getattr(executor, "state", None)
    orders = state.get("orders") if isinstance(state, dict) else None
    if not isinstance(orders, dict):
        return "VERİ YOK", "VERİ YOK", "emir durum kayıtları yok"
    opened = closed = 0
    missing = False
    for order in list(orders.values()):
        if not isinstance(order, dict):
            continue
        phase = order.get("phase")
        filled = _money(order.get("filled_qty"))
        actual = filled is not None and filled > 0 or phase in (
            "DONE", "PROTECTED", "TP_UNKNOWN", "CLOSED_EXTERNALLY")
        if not actual:
            continue
        opened_date = _date_of(order.get("entry_filled_at"))
        if opened_date == target:
            opened += 1
        elif opened_date is None and order.get("day") == target:
            missing = True
        if phase == "DONE":
            closed_date = _date_of(order.get("tp_filled_at"))
        elif phase == "CLOSED_EXTERNALLY":
            closed_date = _date_of(order.get("flat_confirmed_at"))
        else:
            closed_date = None
        if closed_date == target:
            closed += 1
        elif phase in ("DONE", "CLOSED_EXTERNALLY") and closed_date is None:
            # Historical terminal records cannot be assigned to a day.
            if order.get("day") == target:
                missing = True
    if missing:
        return "VERİ YOK", "VERİ YOK", "eski işlemlerde kapanış/dolum zamanı eksik"
    return opened, closed, "bot emir kayıtları; manuel işlemler dahil değil"


def _income(executor, target):
    """Account-wide realized PNL plus commissions/funding, read-only.

    Binance USD-M /fapi/v1/income returns account-wide events (not bot-only).
    Do not report partial pages as complete or pretend an API error means zero.
    """
    broker = getattr(executor, "broker", None)
    if not callable(getattr(broker, "request", None)):
        return "VERİ YOK", "Binance gelir API bağlantısı yok"
    start = datetime.combine(datetime.fromisoformat(target).date(), time.min, PARIS)
    end = start + timedelta(days=1)
    params = {"startTime": int(start.timestamp() * 1000),
              "endTime": int(end.timestamp() * 1000) - 1, "limit": 1000}
    try:
        events = broker.request("GET", "/fapi/v1/income", params)
        if not isinstance(events, list) or len(events) >= 1000:
            return "VERİ YOK", "gelir geçmişi eksik veya sayfalanmalı"
        total = Decimal(0)
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("INVALID_INCOME_RECORD")
            if event.get("incomeType") not in (
                    "REALIZED_PNL", "COMMISSION", "FUNDING_FEE", "COMMISSION_REBATE"):
                continue
            if event.get("asset") != "USDT":
                return "VERİ YOK", "USDT dışı gelir kaydı var"
            value = _money(event.get("income"))
            if value is None:
                raise ValueError("INVALID_INCOME_VALUE")
            total += value
        return _round(total), "Binance hesap geneli; PNL + komisyon + fonlama"
    except Exception:
        return "VERİ YOK", "Binance gelir sorgusu doğrulanamadı"


def _row(days, target, executor, income_value, income_note):
    day = days.get(target)
    if not isinstance(day, dict):
        return None
    before = (datetime.fromisoformat(target).date() - timedelta(days=1)).isoformat()
    previous = days.get(before, {})
    if not isinstance(previous, dict):
        previous = {}
    peak = _money(day.get("peak"))
    low = _money(day.get("low"))
    prev_peak = _money(previous.get("peak"))
    first = _money(day.get("first_equity"))
    last = _money(day.get("last_equity"))
    btc_first = _money(day.get("btc_first"))
    btc_last = _money(day.get("btc_last"))
    opened, closed, count_note = _counts(executor, target)
    issues = []
    if low is None or prev_peak is None:
        issues.append("eski/dip/önceki zirve verisi eksik")
    try:
        end_sample = datetime.fromisoformat(day["last_at"]).astimezone(PARIS)
        if end_sample.hour != 23:
            issues.append("gün sonu örneği eksik: " + end_sample.strftime("%H:%M"))
    except (ValueError, TypeError, KeyError):
        issues.append("son örnek saati yok")
    if count_note != "bot emir kayıtları; manuel işlemler dahil değil":
        issues.append(count_note)
    issues.append(income_note)
    return [
        target, _round(peak), _round(low), _round(prev_peak),
        _round(last-prev_peak) if last is not None and prev_peak is not None else "VERİ YOK",
        _pct(prev_peak, last),
        _round(low-prev_peak) if low is not None and prev_peak is not None else "VERİ YOK",
        _pct(prev_peak, low), _pct(btc_first, btc_last),
        opened, closed, income_value,
        _round(last-first) if first is not None and last is not None else "VERİ YOK",
        _round(first), _round(last), " | ".join(issues),
    ]


def report(directory, now, executor):
    """Return headers + archived days (oldest first). Capture yesterday >=01:05.

    The worker's existing 5-minute Sheets tick schedules the first eligible
    write after 01:05 Paris time. Once captured, a date never shifts rows.
    Missing Binance income is retried no more often than every 30 minutes.
    """
    local = datetime.fromtimestamp(now, PARIS)
    root = Path(directory)
    archive_path = root / ARCHIVE
    data = _read(archive_path)
    if data.get("version") != 1 or not isinstance(data.get("days"), dict):
        data = {"version": 1, "days": {}, "last_income_try": {}}
    rows_by_day = data["days"]
    if (local.hour, local.minute) >= (1, 5):
        target = (local.date() - timedelta(days=1)).isoformat()
        ledger = _read(root / LEDGER).get("days", {})
        if isinstance(ledger, dict) and isinstance(ledger.get(target), dict):
            row = rows_by_day.get(target)
            last_try = float(data.get("last_income_try", {}).get(target, 0))
            retry_income = (isinstance(row, list) and
                            len(row) == len(HEADERS) and
                            row[11] == "VERİ YOK" and now-last_try >= 1800)
            if row is None or retry_income:
                income_value, income_note = _income(executor, target)
                new_row = _row(ledger, target, executor, income_value, income_note)
                if new_row is not None:
                    rows_by_day[target] = new_row
                    data.setdefault("last_income_try", {})[target] = now
                    _save(archive_path, data)
    return [HEADERS] + [
        rows_by_day[key] for key in sorted(rows_by_day)
        if isinstance(rows_by_day[key], list) and len(rows_by_day[key]) == len(HEADERS)
    ]
