"""Isolated Bull Flag paper observer. No broker/order imports. Opt-in only."""
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

LOG = logging.getLogger("bull_flag_paper")
API = "https://api.coinbeacon.io"
FRAMES = ("15m", "1h", "4h", "1d")
BAR_MS = 900000
HORIZON_MS = 8 * 3600000
_MAX_NEW = 8
_STARTED = False


def _db():
    root = Path(os.getenv("DATA_DIR", "/data"))
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "bull_flag_paper.sqlite3", timeout=15)
    db.execute("""CREATE TABLE IF NOT EXISTS flags(
        id TEXT PRIMARY KEY, symbol TEXT, tf TEXT, first_ms INTEGER,
        last_ms INTEGER, status TEXT, support REAL, resistance REAL,
        source TEXT, paper_entry REAL, paper_target REAL, paper_ms INTEGER,
        result TEXT, payload TEXT, archived INTEGER DEFAULT 0)""")
    db.execute("CREATE TABLE IF NOT EXISTS scan_notice(day TEXT PRIMARY KEY, message TEXT)")
    db.commit()
    return db


def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _api(path, params=None):
    key = (os.getenv("api2") or os.getenv("API2") or
           os.getenv("COINBEACON_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("COINBEACON_KEY_MISSING")
    r = requests.get(API + path, params=params, headers={"X-API-Key": key}, timeout=35)
    if r.status_code == 429:
        raise RuntimeError("COINBEACON_RATE_LIMIT")
    r.raise_for_status()
    return r.json()


def _bars(symbol, before_ms, count=480):
    # Explicit 'to' freezes the observation. Do not include a developing bar.
    data = _api("/public/ohlcv/binance-futures/%s/15m" % symbol,
                {"to": before_ms - 1, "limit": count})
    if data.get("exchange") != "binance-futures" or data.get("symbol") != symbol:
        raise ValueError("OHLCV_WRONG_MARKET")
    out = []
    for row in data.get("bars", []):
        try:
            t = int(row["openTime"])
            end = int(row["closeTime"])
            vals = [float(row[k]) for k in ("open", "high", "low", "close", "volume")]
            if (t % BAR_MS or end != t + BAR_MS - 1 or end >= before_ms or
                any(not math.isfinite(x) for x in vals) or min(vals[:4]) <= 0
                or vals[4] < 0 or vals[1] < max(vals[0], vals[3])
                or vals[2] > min(vals[0], vals[3])):
                continue
            out.append([t] + vals)
        except (KeyError, TypeError, ValueError):
            continue
    out = sorted({x[0]: x for x in out}.values())
    if len(out) > count:
        out = out[-count:]
    complete = (len(out) == count and
                all(b[0] - a[0] == BAR_MS for a, b in zip(out, out[1:])))
    return out, complete


def _telegram(message):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        LOG.warning("Bull Flag Telegram not configured")
        return False
    try:
        r = requests.post("https://api.telegram.org/bot%s/sendMessage" % token,
                          data={"chat_id": chat, "text": message[:2000]}, timeout=12)
        r.raise_for_status()
        return bool(r.json().get("ok"))
    except (requests.RequestException, ValueError):
        LOG.warning("Bull Flag Telegram delivery failed")
        return False


def _flag(row, tf):
    item = row.get("item")
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or item.get("type") or item.get("pattern") or item.get("patternType") or "")
    if re.sub(r"[^a-z]", "", name.lower()) != "bullflag":
        return None
    sym = str(row.get("symbol") or "")
    if (not re.fullmatch(r"[A-Z0-9]{2,30}USDT", sym) or
        row.get("exchange") != "binance-futures" or row.get("tf") != tf):
        return None
    # Pattern geometry is preferred. If absent, no fabricated API level.
    support = _number(item.get("support") or item.get("supportPrice"))
    resistance = _number(item.get("resistance") or item.get("resistancePrice"))
    source = "COINBEACON_PATTERN" if support and resistance and support < resistance else "UNAVAILABLE"
    if source == "UNAVAILABLE":
        support = resistance = None
    return sym, support, resistance, source, item


def _identity(sym, tf, item, db):
    # Stable detector pivot geometry, if published. Never hash changing score/price.
    pivots = item.get("pivots") or item.get("points")
    if isinstance(pivots, list) and pivots:
        anchor = json.dumps(pivots, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((sym + tf + "BullFlag" + anchor).encode()).hexdigest()[:24]
    # Without stable pivot metadata, retain one observational instance for 48h.
    cutoff = int(time.time() * 1000) - 48 * 3600000
    old = db.execute("SELECT id FROM flags WHERE symbol=? AND tf=? AND first_ms>? "
                     "ORDER BY first_ms DESC LIMIT 1", (sym, tf, cutoff)).fetchone()
    if old:
        return old[0]
    return hashlib.sha256((sym + tf + str(int(time.time() // 172800))).encode()).hexdigest()[:24]


def _archive(record, db):
    if record["archived"]:
        return
    try:
        from btc_hourly_direction import _r2_client
        client, bucket = _r2_client()
        if client is None:
            return
        payload = json.loads(record["payload"])
        key = "bull-flag-paper/%s/%s.json.gz" % (
            datetime.fromtimestamp(record["first_ms"]/1000, timezone.utc).strftime("%Y-%m"),
            record["id"])
        packed = gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())
        client.put_object(Bucket=bucket, Key=key, Body=packed,
                          ContentType="application/json", ContentEncoding="gzip")
        db.execute("UPDATE flags SET archived=1 WHERE id=?", (record["id"],))
        db.commit()
    except Exception as exc:
        LOG.warning("Bull Flag R2 archive failed: %s", type(exc).__name__)


def _sheet(db):
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    # Write to the same EMA Filter workbook as the working Crash dashboard.
    from crash_sheets_dashboard import DEFAULT_EMA_FILTER_ID
    sid = (os.getenv("BULL_FLAG_SHEETS_ID") or os.getenv("CRASH_SHEETS_TARGET_ID") or DEFAULT_EMA_FILTER_ID).strip()
    if not raw or not sid:
        print("[BULL FLAG SHEETS] not configured", flush=True)
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            info = None
        creds = (Credentials.from_service_account_info(info, scopes=scopes)
                 if isinstance(info, dict)
                 else Credentials.from_service_account_file(raw, scopes=scopes))
        book = gspread.authorize(creds).open_by_key(sid)
        try:
            tab = book.worksheet("BULL_FLAG_PAPER")
        except gspread.WorksheetNotFound:
            tab = book.add_worksheet(title="BULL_FLAG_PAPER", rows=1500, cols=15)
        rows = [["ID", "Coin", "TF", "First UTC", "Last UTC", "Support", "Resistance",
                 "Level source", "Paper entry", "Paper target", "Status", "Result",
                 "History bars", "Future bars"]]
        for rec in db.execute("SELECT * FROM flags ORDER BY first_ms DESC LIMIT 1000"):
            record = dict(zip([c[0] for c in db.execute("SELECT * FROM flags LIMIT 0").description], rec))
            p = json.loads(record["payload"])
            rows.append([record["id"], record["symbol"], record["tf"],
                         datetime.fromtimestamp(record["first_ms"]/1000, timezone.utc).isoformat(),
                         datetime.fromtimestamp(record["last_ms"]/1000, timezone.utc).isoformat(),
                         record["support"] or "", record["resistance"] or "",
                         record["source"], record["paper_entry"] or "", record["paper_target"] or "",
                         record["status"], record["result"] or "",
                         len(p.get("history", [])), len(p.get("future", []))])
        tab.update(range_name="A1", values=rows)
        print("[BULL FLAG SHEETS] updated rows=%d target_suffix=%s" % (len(rows)-1, sid[-8:]), flush=True)
    except Exception as exc:
        print("[BULL FLAG SHEETS] update failed: %s" % type(exc).__name__, flush=True)


def _later(db, now_ms):
    db.row_factory = sqlite3.Row
    records = db.execute("SELECT * FROM flags WHERE archived=0 LIMIT 40").fetchall()
    for rec in records:
        obj = dict(rec)
        payload = json.loads(obj["payload"])
        # Completed cases can retry R2 upload after temporary outages.
        if obj["status"] == "COMPLETE":
            _archive(obj, db)
            continue
        # Fetch future only at the end of 8h; compact 32 closed 15m candles.
        if now_ms < obj["first_ms"] + HORIZON_MS + BAR_MS:
            continue
        try:
            future, _ = _bars(obj["symbol"], obj["first_ms"] + HORIZON_MS + BAR_MS, 40)
            future = [x for x in future if obj["first_ms"] <= x[0] <
                      obj["first_ms"] + HORIZON_MS][:32]
            payload["future"] = future
            payload["future_complete"] = len(future) == 32 and all(
                b[0] - a[0] == BAR_MS for a, b in zip(future, future[1:]))
            # Resolve paper outcome conservatively from future closed bars only.
            result = obj["result"]
            entry, target = obj["paper_entry"], obj["paper_target"]
            if entry and target and obj["support"]:
                for bar in future:
                    if bar[0] < (obj["paper_ms"] or 0):
                        continue
                    if bar[2] >= target and bar[3] < obj["support"] * .997:
                        result = "AMBIGUOUS_INVALIDATED"
                        break
                    if bar[3] < obj["support"] * .997:
                        result = "INVALIDATED"
                        break
                    if bar[2] >= target:
                        result = "TARGET"
                        break
                result = result or "TIMEOUT"
            else:
                result = result or "NO_ENTRY"
            payload["result"] = result
            db.execute("UPDATE flags SET result=?,status=?,payload=? WHERE id=?",
                       (result, "COMPLETE", json.dumps(payload, separators=(",", ":")), obj["id"]))
            db.commit()
            _telegram("📊 BULL FLAG SANAL SONUÇ\n%s %s: %s" %
                      (obj["symbol"], obj["tf"], result))
            _archive({**obj, "payload": json.dumps(payload), "archived": 0}, db)
        except Exception as exc:
            LOG.warning("Bull Flag 8h followup unavailable: %s", type(exc).__name__)


def step():
    now_ms = int(time.time()*1000)
    with _db() as db:
        # Existing paper cases are processed even if a detector snapshot is truncated.
        _later(db, now_ms)
        new_count = 0
        for tf in FRAMES:
            try:
                data = _api("/public/detectors/patterns",
                            {"exchange": "binance-futures", "timeframe": tf, "quote": "USDT"})
                if data.get("kind") != "patterns" or data.get("timeframe") != tf:
                    raise ValueError("PATTERN_BOARD_WRONG_MARKET")
                board = data.get("items", [])
                if not isinstance(board, list):
                    raise ValueError("PATTERN_ITEMS_NOT_LIST")
                # CoinBeacon detector boards are hard-capped at 150 and cannot be paged.
                # Report truncation instead of presenting missing Bull Flags as zero signals.
                total = data.get("total")
                try:
                    total_count = int(total) if total is not None else len(board)
                except (TypeError, ValueError):
                    total_count = len(board)
                truncated = total_count > len(board)
                print("[BULL FLAG COVERAGE] tf=%s returned=%d total=%d truncated=%s" %
                      (tf, len(board), total_count, truncated), flush=True)
                if truncated:
                    day = datetime.now(timezone.utc).date().isoformat()
                    notice_key = day + ":" + tf
                    if not db.execute("SELECT 1 FROM scan_notice WHERE day=?", (notice_key,)).fetchone():
                        message = ("⚠️ BULL FLAG API KAPSAMI\\n%s: %d/%d formasyon erişilebilir. "
                                   "CoinBeacon API 150 kayıtla sınırlı; kalan formasyonlar bu taramada görülemiyor." %
                                   (tf, len(board), total_count))
                        _telegram(message)
                        db.execute("INSERT OR IGNORE INTO scan_notice(day,message) VALUES (?,?)",
                                   (notice_key, message))
                        db.commit()
                names = {}
                for row in board:
                    if not isinstance(row, dict):
                        continue
                    item = row.get("item")
                    source = item if isinstance(item, dict) else row
                    name = str(source.get("name") or source.get("type") or source.get("pattern") or source.get("patternType") or "UNKNOWN")
                    names[name] = names.get(name, 0) + 1
                print("[BULL FLAG BOARD] tf=%s total=%d kinds=%s first_keys=%s" %
                      (tf, len(board), str(sorted(names.items(), key=lambda x: -x[1])[:8]),
                       str(sorted(board[0].keys()) if board and isinstance(board[0], dict) else [])), flush=True)
                for row in board:
                    hit = _flag(row, tf)
                    if hit is None:
                        continue
                    sym, support, resistance, source, item = hit
                    ident = _identity(sym, tf, item, db)
                    old = db.execute("SELECT id,status FROM flags WHERE id=?", (ident,)).fetchone()
                    if old:
                        db.execute("UPDATE flags SET last_ms=? WHERE id=?", (now_ms, ident))
                        # A new paper entry needs valid levels, and another closed-bar observation.
                        if old[1] == "WATCH" and support and resistance:
                            bars, ok = _bars(sym, (now_ms // BAR_MS)*BAR_MS, 2)
                            if ok and len(bars) >= 2:
                                prev, cur = bars[-2:]
                                close = cur[4]
                                if (support <= cur[3] <= support*1.003 and
                                    close > prev[4] and close > support and
                                    resistance*.999 > close*1.005):
                                    db.execute("UPDATE flags SET status=?,paper_entry=?,"
                                               "paper_target=?,paper_ms=? WHERE id=?",
                                               ("PAPER_LONG", close, resistance*.999, now_ms, ident))
                                    _telegram("🟢 BULL FLAG SANAL LONG\n%s %s\n"
                                              "Giriş: %.8g | Destek: %.8g | Direnç: %.8g" %
                                              (sym, tf, close, support, resistance))
                        continue
                    if new_count >= _MAX_NEW:
                        continue
                    # Archive the actual context as of first detection, before any outcome.
                    end = (now_ms // BAR_MS)*BAR_MS
                    bars, complete = _bars(sym, end, 480)
                    payload = {"schema":"bull_flag_paper_v1", "id":ident, "symbol":sym,
                               "tf":tf, "observed_ms":now_ms, "pattern":item,
                               "support":support, "resistance":resistance,
                               "levels_source":source, "history":bars,
                               "history_complete":complete, "future":[],
                               "future_complete":False}
                    # Without CoinBeacon geometric levels, use an explicit candle-derived
                    # range estimate (never label it as a CoinBeacon detector level).
                    if source == "UNAVAILABLE" and complete and len(bars) >= 32:
                        support = min(b[3] for b in bars[-32:])
                        resistance = max(b[2] for b in bars[-32:])
                        if support > 0 and resistance > support * 1.009:
                            source = "CANDLE_RANGE_ESTIMATE"
                            payload["support"] = support
                            payload["resistance"] = resistance
                            payload["levels_source"] = source
                    status = "WATCH" if source != "UNAVAILABLE" and complete else "DATA_PENDING"
                    db.execute("INSERT INTO flags(id,symbol,tf,first_ms,last_ms,status,"
                               "support,resistance,source,payload) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (ident,sym,tf,now_ms,now_ms,status,support,resistance,source,
                                json.dumps(payload, separators=(",", ":"))))
                    db.commit()
                    new_count += 1
                    _telegram("🚩 YENİ BULL FLAG (SANAL)\n%s %s\n"
                              "Destek: %s | Direnç: %s\nVeri: %s" %
                              (sym, tf, support, resistance, status))
            except Exception as exc:
                print("[BULL FLAG SCAN] %s unavailable: %s" % (tf, type(exc).__name__), flush=True)
        _sheet(db)
        print("[BULL FLAG SCAN] cycle complete new=%s" % new_count, flush=True)


def start():
    # Keep the existing ema.py entrypoint; use the combined research worker.
    try:
        from bull_pattern_research import start as start_combined
    except SyntaxError as exc:
        LOG.error("Bull pattern module syntax: file=%s line=%s message=%s", exc.filename, exc.lineno, exc.msg)
        raise
    return start_combined()


def start_legacy():
    global _STARTED
    if _STARTED or os.getenv("BULL_FLAG_PAPER_ENABLED", "0") != "1":
        return False
    if not (os.getenv("api2") or os.getenv("API2") or os.getenv("COINBEACON_API_KEY")):
        LOG.warning("Bull Flag disabled: missing CoinBeacon key")
        return False
    _STARTED = True
    def loop():
        while True:
            try:
                step()
            except Exception as exc:
                print("[BULL FLAG WORKER] error: %s" % type(exc).__name__, flush=True)
            time.sleep(max(30, 1800 - (time.time() % 1800)))
    threading.Thread(target=loop, name="bull-flag-paper", daemon=True).start()
    return True
