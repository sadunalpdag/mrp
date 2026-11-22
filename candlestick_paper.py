"""Isolated CoinBeacon Hammer/Morning Star 8h paper observer. Never places orders."""
import gzip
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

LOG = logging.getLogger("candlestick_paper")
API = "https://api.coinbeacon.io"
PARIS = ZoneInfo("Europe/Paris")
FRAMES = ("15m", "1h", "4h", "1d")
BAR = 900_000
HORIZON = 8 * 3_600_000
CAP = 10
_STARTED = False
HEADERS = ["ID", "Coin", "Pattern", "TF", "Entry Paris", "Entry UTC",
           "Entry", "TP +1%", "8h deadline UTC", "Status", "TP reached",
           "TP time UTC", "8h close", "8h return %", "Max up %",
           "Max down %", "Bars", "Source", "Notes"]


def _db():
    root = Path(os.getenv("DATA_DIR", "/data"))
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(root / "candlestick_paper.sqlite3"), timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        id TEXT PRIMARY KEY, symbol TEXT NOT NULL, pattern TEXT NOT NULL,
        tf TEXT NOT NULL, anchor_ms INTEGER NOT NULL, entry_ms INTEGER NOT NULL,
        entry REAL NOT NULL, target REAL NOT NULL, deadline_ms INTEGER NOT NULL,
        state TEXT NOT NULL, tp_ms INTEGER, close_8h REAL, ret_pct REAL,
        up_pct REAL, down_pct REAL, bars INTEGER, notes TEXT DEFAULT '',
        sheet_dirty INTEGER DEFAULT 1)""")
    con.execute("""CREATE TABLE IF NOT EXISTS archive_days (
        day TEXT PRIMARY KEY, done INTEGER DEFAULT 0)""")
    con.commit()
    return con


def _api(path, params=None):
    key = (os.getenv("api2") or os.getenv("API2")
           or os.getenv("COINBEACON_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("COINBEACON_KEY_MISSING")
    r = requests.get(API + path, params=params, headers={"X-API-Key": key}, timeout=25)
    if r.status_code == 429:
        raise RuntimeError("COINBEACON_RATE_LIMIT")
    r.raise_for_status()
    return r.json()


def _number(value):
    try:
        f = float(value)
        return f if math.isfinite(f) and f > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _ms(value):
    try:
        val = int(float(value))
        return val if val > 1_000_000_000_000 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _pattern(row, tf, now):
    if not isinstance(row, dict) or row.get("exchange") != "binance-futures" or row.get("tf") != tf:
        return None
    item = row.get("item")
    if not isinstance(item, dict):
        return None
    raw = re.sub(r"[^a-z]", "", str(item.get("name") or "").lower())
    name = {"hammer": "Hammer", "morningstar": "Morning Star"}.get(raw)
    if not name or str(item.get("direction") or "bull").lower() not in ("bull", "bullish"):
        return None
    if str(item.get("state") or "").lower() == "setup":
        return None
    symbol = str(row.get("symbol") or "")
    if not re.fullmatch(r"[A-Z0-9]{2,30}USDT", symbol):
        return None
    entry = _number(row.get("price"))
    # Prefer the pattern's own completed candle, avoiding changing score/price as identity.
    candles = item.get("candles")
    anchor = _ms(item.get("completedAt"))
    if anchor is None and isinstance(candles, list) and candles:
        anchor = _ms(candles[-1].get("openTime")) if isinstance(candles[-1], dict) else None
    if anchor is None:
        anchor = _ms(row.get("computedAt"))
    if not entry or not anchor or anchor > now + 60_000:
        return None
    ident = hashlib.sha256(f"{symbol}|{tf}|{name}|{anchor}".encode()).hexdigest()[:24]
    return ident, symbol, name, entry, anchor


def _notify(message):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": message[:1900]}, timeout=12)
        r.raise_for_status()
    except requests.RequestException:
        LOG.warning("Candlestick Telegram delivery failed")


def _ts(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def _result(db, rec, now):
    if now < rec["deadline_ms"] + BAR:
        return False
    start = ((rec["entry_ms"] + BAR - 1) // BAR) * BAR
    # Only fully closed candles that ended before the original 8h deadline.
    end = (rec["deadline_ms"] // BAR) * BAR
    if end <= start:
        return False
    data = _api(f"/public/ohlcv/binance-futures/{rec['symbol']}/15m",
                {"from": start, "to": end - 1, "limit": 40})
    if data.get("exchange") != "binance-futures" or data.get("symbol") != rec["symbol"] or data.get("timeframe") != "15m":
        raise ValueError("OHLCV_WRONG_MARKET")
    bars = []
    for row in data.get("bars", []):
        try:
            t = int(row["openTime"])
            e = int(row["closeTime"])
            hi, lo, close = (_number(row.get(k)) for k in ("high", "low", "close"))
            if start <= t < end and e == t + BAR - 1 and hi and lo and close and hi >= max(lo, close):
                bars.append((t, hi, lo, close))
        except (TypeError, ValueError, KeyError):
            continue
    bars = sorted({b[0]: b for b in bars}.values())
    expected = (end - start) // BAR
    if len(bars) != expected or any(b[0] != start + i * BAR for i, b in enumerate(bars)):
        raise ValueError("OHLCV_INCOMPLETE")
    first_tp = next((b[0] + BAR - 1 for b in bars if b[1] >= rec["target"]), None)
    hi = max(b[1] for b in bars)
    lo = min(b[2] for b in bars)
    last = bars[-1][3]
    entry = rec["entry"]
    db.execute("""UPDATE signals SET state='DONE', tp_ms=?, close_8h=?, ret_pct=?,
                  up_pct=?, down_pct=?, bars=?, notes=?, sheet_dirty=1 WHERE id=?""",
               (first_tp, last, round((last / entry - 1) * 100, 5),
                round((hi / entry - 1) * 100, 5),
                round((lo / entry - 1) * 100, 5), len(bars),
                "TP after entry assessed with closed 15m bars; first TP timestamp is candle close proxy",
                rec["id"]))
    db.commit()
    _notify(f"📊 CANDLESTICK 8H SANAL SONUÇ\n{rec['symbol']} {rec['pattern']} {rec['tf']}\n"
            f"TP %1: {'EVET' if first_tp else 'HAYIR'} | 8h getiri: {(last / entry - 1)*100:+.2f}%\n"
            f"En yüksek: {(hi / entry - 1)*100:+.2f}% | En düşük: {(lo / entry - 1)*100:+.2f}%")
    return True


def _sheet(db):
    if not db.execute("SELECT 1 FROM signals WHERE sheet_dirty=1 LIMIT 1").fetchone():
        return
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        from crash_sheets_dashboard import DEFAULT_EMA_FILTER_ID
        sid = (os.getenv("CANDLESTICK_PAPER_SHEETS_ID")
               or os.getenv("CRASH_SHEETS_TARGET_ID") or DEFAULT_EMA_FILTER_ID).strip()
        if not sid:
            return
        info = json.loads(raw) if raw.startswith("{") else None
        creds = (Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
                 if info else Credentials.from_service_account_file(raw, scopes=["https://www.googleapis.com/auth/spreadsheets"]))
        book = gspread.authorize(creds).open_by_key(sid)
        try:
            tab = book.worksheet("HAMMER_MORNING_STAR")
        except gspread.WorksheetNotFound:
            tab = book.add_worksheet(title="HAMMER_MORNING_STAR", rows=1200, cols=len(HEADERS))
        rows = [HEADERS]
        for x in db.execute("SELECT * FROM signals ORDER BY entry_ms DESC LIMIT 1000"):
            rows.append([x["id"], x["symbol"], x["pattern"], x["tf"],
                         datetime.fromtimestamp(x["entry_ms"]/1000, PARIS).isoformat(),
                         _ts(x["entry_ms"]), x["entry"], x["target"], _ts(x["deadline_ms"]),
                         x["state"], "YES" if x["tp_ms"] else ("NO" if x["state"] == "DONE" else ""),
                         _ts(x["tp_ms"]) if x["tp_ms"] else "",
                         x["close_8h"] or "", x["ret_pct"] if x["ret_pct"] is not None else "",
                         x["up_pct"] if x["up_pct"] is not None else "",
                         x["down_pct"] if x["down_pct"] is not None else "",
                         x["bars"] or "", "CoinBeacon", x["notes"] or ""])
        tab.update(range_name="A1", values=rows)
        db.execute("UPDATE signals SET sheet_dirty=0")
        db.commit()
    except Exception as exc:
        LOG.warning("Candlestick Sheets sync failed: %s", type(exc).__name__)


def _archive(db, now):
    # Archive yesterday and recent days at/after Paris midnight. An archive
    # stays mutable until all cases from the date finish their full 8h review.
    today = datetime.fromtimestamp(now/1000, PARIS).date()
    for days_ago in (1, 2):
        day = (today - timedelta(days=days_ago)).isoformat()
        state = db.execute("SELECT done FROM archive_days WHERE day=?", (day,)).fetchone()
        if state and state["done"]:
            continue
        start_day = datetime.fromisoformat(day).replace(tzinfo=PARIS)
        begin = int(start_day.timestamp()*1000)
        finish = int((start_day + timedelta(days=1)).timestamp()*1000)
        rows = db.execute("SELECT * FROM signals WHERE entry_ms>=? AND entry_ms<?",
                          (begin, finish)).fetchall()
        if not rows and days_ago == 2:
            continue
        records = [{k: x[k] for k in (
            "id","symbol","pattern","tf","entry_ms","entry","target",
            "deadline_ms","state","tp_ms","close_8h","ret_pct",
            "up_pct","down_pct","bars")} for x in rows]
        try:
            from btc_hourly_direction import _r2_client
            client, bucket = _r2_client()
            if not client:
                return
            payload = json.dumps({"schema":"candlestick_paper_v1","day_paris":day,
                                  "records":records}, separators=(",",":"), allow_nan=False).encode()
            key = f"candlestick-paper/daily/{day[:4]}/{day[5:7]}/{day}.json.gz"
            client.put_object(Bucket=bucket, Key=key, Body=gzip.compress(payload),
                              ContentType="application/json", ContentEncoding="gzip")
            done = int(all(x["state"] == "DONE" for x in rows))
            db.execute("INSERT OR REPLACE INTO archive_days(day,done) VALUES (?,?)", (day,done))
            db.commit()
        except Exception as exc:
            LOG.warning("Candlestick R2 archive failed: %s", type(exc).__name__)


def step():
    now = int(time.time()*1000)
    with _db() as db:
        for rec in db.execute("SELECT * FROM signals WHERE state='ACTIVE' AND deadline_ms<=? ORDER BY deadline_ms", (now,)).fetchall():
            try:
                _result(db, rec, now)
            except Exception as exc:
                LOG.warning("Candlestick 8h result pending for %s: %s", rec["symbol"], type(exc).__name__)
        _sheet(db)
        _archive(db, now)
        # Freeze ALL detector calls once ten unique coins are active.
        active = {x["symbol"] for x in db.execute("SELECT symbol FROM signals WHERE state='ACTIVE'")}
        if len(active) >= CAP:
            LOG.info("Candlestick capacity reached: %d; scan paused", len(active))
            return
        for tf in FRAMES:
            if len(active) >= CAP:
                break
            try:
                data = _api("/public/detectors/candlesticks",
                            {"exchange":"binance-futures","timeframe":tf,"quote":"USDT"})
                if data.get("kind") != "candlesticks" or data.get("timeframe") != tf:
                    raise ValueError("DETECTOR_WRONG_BOARD")
                board = data.get("items", [])
                if not isinstance(board, list):
                    raise ValueError("DETECTOR_WRONG_ITEMS")
                if int(data.get("total") or len(board)) > len(board):
                    LOG.warning("Candlestick board truncated tf=%s count=%s total=%s",
                                tf, len(board), data.get("total"))
                for row in board:
                    if len(active) >= CAP:
                        break
                    hit = _pattern(row, tf, now)
                    if not hit:
                        continue
                    ident, symbol, name, entry, anchor = hit
                    if symbol in active or db.execute("SELECT 1 FROM signals WHERE id=?", (ident,)).fetchone():
                        continue
                    # Do not create a historical entry from a stale board snapshot.
                    if now - anchor > 24*3_600_000:
                        continue
                    db.execute("""INSERT INTO signals (id,symbol,pattern,tf,anchor_ms,
                                  entry_ms,entry,target,deadline_ms,state)
                                  VALUES (?,?,?,?,?,?,?,?,?,'ACTIVE')""",
                               (ident,symbol,name,tf,anchor,now,entry,entry*1.01,now+HORIZON))
                    db.commit()
                    active.add(symbol)
                    _notify(f"🟢 CANDLESTICK SANAL LONG\n{symbol} | {name} | {tf}\n"
                            f"Giriş: {entry:.9g} | TP %1: {entry*1.01:.9g}\n"
                            f"8 saat takip | Aktif coin: {len(active)}/10")
            except Exception as exc:
                LOG.warning("Candlestick scan %s failed: %s", tf, type(exc).__name__)
        _sheet(db)


def start():
    global _STARTED
    if _STARTED or os.getenv("CANDLESTICK_PAPER_ENABLED", "0") != "1":
        return False
    if not (os.getenv("api2") or os.getenv("API2") or os.getenv("COINBEACON_API_KEY")):
        LOG.warning("Candlestick observer disabled: no API key")
        return False
    _STARTED = True
    def loop():
        while True:
            try:
                step()
            except Exception as exc:
                LOG.warning("Candlestick observer cycle error: %s", type(exc).__name__)
            time.sleep(max(15, 900 - (time.time() % 900)))
    threading.Thread(target=loop, daemon=True, name="candlestick-paper").start()
    return True
