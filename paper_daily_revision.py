"""Daily 18:45 Europe/Paris archival reconciliation for PAPER CB2.

Runs inside each existing paper monitor (using its SQLite connection). Public
Binance daily ZIPs only, no exchange REST, no orders and no real-trade changes.
"""
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
LOG = logging.getLogger("paper_daily_revision")
TIME_OF_DAY_MINUTES = 18 * 60 + 45
MAX_REVISIT_SECONDS = 240
MAX_TRACK_PASSES = 12
LOOKBACK_DAYS = 7


def due(last_day, now=None):
    local = (now or datetime.now(PARIS)).astimezone(PARIS)
    minute = local.hour * 60 + local.minute
    return minute >= TIME_OF_DAY_MINUTES and str(last_day or "") != local.date().isoformat()


def _iso_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(PARIS).date().isoformat()
    except (ValueError, TypeError):
        return None


def _stale_open(trade, now=None):
    """OPEN still missing yesterday's final archived 15m candle."""
    if trade.get("status") not in ("OPEN", "OPEN_24H"):
        return False
    today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    opened = trade.get("opened_at")
    checked = trade.get("last_candle_at")
    try:
        opened_day = datetime.fromisoformat(opened.replace("Z", "+00:00")).astimezone(timezone.utc).date()
        if opened_day > yesterday:
            return False  # Today's signals cannot have yesterday's candles.
        last_day = (datetime.fromisoformat(checked.replace("Z", "+00:00"))
                    .astimezone(timezone.utc).date()) if checked else None
        return last_day is None or last_day < yesterday
    except (ValueError, TypeError, AttributeError):
        return True


def _daily_sheet(system, totals, days, book_id):
    import gspread
    from google.oauth2.service_account import Credentials
    cred = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not cred or not book_id:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS_OR_ID_MISSING")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    auth = (Credentials.from_service_account_info(json.loads(cred), scopes=scopes)
            if cred.startswith("{") else Credentials.from_service_account_file(cred, scopes=scopes))
    book = gspread.authorize(auth).open_by_key(book_id)
    title = "CB2_REVIZYON_1845"
    try:
        ws = book.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=title, rows=150, cols=9)
    stamp = datetime.now(PARIS).isoformat(timespec="seconds")
    data = [
        [system.upper() + " | ARSIV GUNLUK REVIZYON | 18:45 FRANSA"],
        ["Son kontrol Europe/Paris", stamp, "Veri kaynagi", "Binance Futures public daily 15m ZIP"],
        ["Durum", "Kapanmamis sinyaller TP basarisi sayilmaz; yayimlanmayan mumlar BEKLIYOR kalir"],
        ["Toplam kayit", totals["total"], "TP +%1", totals["tp"],
         "Acik/izlenen", totals["open"], "Eksik/bekleyen", totals["pending"]],
        ["Gecmis gun tamamlanan mum", totals["new_bars"], "Arsiv tarama turu", totals["passes"]],
        ["GUNLUK OZET | SON 7 FRANSA GUNU"],
        ["Tarih (Fransa)", "Toplam sinyal", "TP +%1", "Acik veya kismi",
         "Arsiv bekleyen", "Gozlem/Watch", "Not"],
    ]
    for day in sorted(days, reverse=True)[:LOOKBACK_DAYS]:
        row = days[day]
        data.append([day, row["total"], row["tp"], row["open"],
                     row["pending"], row["watch"], "Sanal OHLC; kesin gercek islem degil"])
    if ws.row_count < len(data):
        ws.add_rows(len(data) - ws.row_count + 20)
    ws.update(range_name="A1", values=data, value_input_option="RAW")
    if ws.row_count > len(data):
        ws.batch_clear([f"A{len(data)+1}:I{ws.row_count}"])
    return title


def _day_group(rows, *, cb2=False):
    grouped = defaultdict(lambda: {"total": 0, "tp": 0, "open": 0, "pending": 0, "watch": 0})
    for row in rows:
        day = _iso_date(row.get("observed_at") if cb2 else row.get("opened_at"))
        if day is None:
            continue
        bucket = grouped[day]
        bucket["total"] += 1
        if cb2:
            status = str(row.get("status") or "")
            outcome = (row.get("entry_scenarios") or {}).get("next_15m_open") or {}
            watch = status in ("WATCH", "WATCH_EXPIRED", "UNCONFIRMED_OI") or str(row.get("family", "")).endswith("_watch")
            if watch:
                bucket["watch"] += 1
            else:
                bucket["tp"] += int(outcome.get("status") == "TP_1_PERCENT")
                bucket["open"] += int(outcome.get("status") == "OPEN_IN_OBSERVED_WINDOW")
                bucket["pending"] += int(status != "COMPLETE_48H")
        else:
            status = row.get("status")
            bucket["tp"] += int(status == "TP_1_PERCENT")
            bucket["open"] += int(status == "OPEN")
            bucket["pending"] += int(status == "OPEN" and (
                row.get("data_status") != "OK" or _stale_open(row)))
    return dict(grouped)


def revise_cb2(db, market):
    import coinbeacon_six_paper as c
    import cb2_signal_research as research
    from paper_archive_15m import klines
    today = datetime.now(PARIS).date().isoformat()
    if not due(c.meta(db, "paper_revision_1845_day")):
        return False
    before = sum(int(x.get("observed_candles") or 0) for x in c.all_trades(db))
    start = time.monotonic()
    passes = 0
    for _ in range(MAX_TRACK_PASSES):
        if time.monotonic() - start >= MAX_REVISIT_SECONDS:
            break
        c.reopen_24h(db)
        market.begin_cycle()
        checked, new_bars = c.track(db, market)
        passes += 1
        if not new_bars:
            break
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    checked_signals = research.analyze(
        db, klines, market.http, c.now_iso(), force=True,
        max_checks=1000, since_ms=cutoff, max_seconds=240)
    signals = research.export(db)["signals"]
    trades = c.all_trades(db)
    live = [x for x in trades if x.get("status") in ("OPEN", "OPEN_24H")]
    watch = [x for x in signals if x.get("status") in
             ("WATCH", "WATCH_EXPIRED", "UNCONFIRMED_OI") or
             str(x.get("family") or "").endswith("_watch")]
    next_open = [(x.get("entry_scenarios") or {}).get("next_15m_open") or {}
                 for x in signals if x not in watch]
    totals = {
        "total": len(signals),
        "tp": sum(x.get("status") == "TP_1_PERCENT" for x in next_open),
        "open": len(live),
        "pending": sum(x.get("status") not in ("COMPLETE_48H", "WATCH", "WATCH_EXPIRED", "UNCONFIRMED_OI")
                       for x in signals if x not in watch)
                   + sum(_stale_open(x) for x in live),
        "new_bars": sum(int(x.get("observed_candles") or 0) for x in trades) - before,
        "passes": passes,
    }
    c.sheet_sync(db)
    book_id = (os.getenv("COINBEACON_SIX_SHEET_ID") or os.getenv("GOOGLE_SHEETS_ID")
               or os.getenv("GOOGLE_SHEET_ID"))
    _daily_sheet("cb2", totals, _day_group(signals, cb2=True), book_id)
    # Force the standard cumulative JSON upload on this cycle after revision.
    c.put_meta(db, "json_upload_at", "0")
    c.put_meta(db, "paper_revision_1845_day", today)
    LOG.info("[CB2 REVISION 18:45] day=%s checked_signals=%d totals=%s",
             today, checked_signals, totals)
    return True
