"""Six independent CoinBeacon paper-only experiments.

Uses CoinBeacon key in Render env 'api2'. Read-only public market endpoints only:
CoinBeacon GET, Binance futures public GET. No trading/broker imports or POSTs.
HBREAK's SQLite DB is opened read-only. All experiment outputs isolated.
"""
import csv
import hashlib
import io
import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

LOG = logging.getLogger("coinbeacon_six_paper")
if not LOG.handlers:
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("[CB-API2] %(levelname)s %(message)s"))
    LOG.addHandler(stream)
LOG.setLevel(logging.INFO)
LOG.propagate = False

PARIS = ZoneInfo("Europe/Paris")
BASE = Path(os.getenv("DATA_DIR", "/data"))
DB = BASE / "coinbeacon_six_paper.sqlite3"
HBREAK_DB = BASE / "hbreak_paper.sqlite3"
API = "https://api.coinbeacon.io"
BINANCE = "https://fapi.binance.com"
SHEETS = {
    "rvol": "CB2_01_RVOL",
    "retest": "CB2_02_RETEST",
    "trend": "CB2_03_TREND",
    "squeeze": "CB2_04_SQUEEZE",
    "volume": "CB2_05_VOLUME",
    "oi_squeeze": "CB2_06_OI_SQUEEZE",
}
FIELDS = (
    "id strategy symbol status signal_at opened_at source_id source_archetype "
    "sqs rvol trend_1h trend_4h oi_change_pct short_liq_usd "
    "entry tp tp_at tp_minutes lowest max_drawdown_pct "
    "last_price checked_at data_status observed_candles last_candle_at "
    "age_hours note"
).split()
COSTS = {"trend": 5, "squeeze": 5, "volume": 5, "liquidations": 1, "oi": 10}
INTERVALS = {"trend": 3600, "squeeze": 3600, "volume": 1800,
             "liquidations": 1200}
RESERVE = 400
MAX_OI_PER_UTC_DAY = 24
MAX_NEW_SYMBOLS_PER_CYCLE = 8
MAX_TRACK_SYMBOLS_PER_CYCLE = 12
MAX_SHEET_ROWS = 8000


def now_ms():
    return int(time.time() * 1000)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_to_ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def num(value, default=None):
    try:
        v = float(value)
        return v if v == v and abs(v) < float("inf") else default
    except (ValueError, TypeError):
        return default


def stamp_ms(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat(timespec="seconds")


def key_for(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def initialize(db):
    db.execute("CREATE TABLE IF NOT EXISTS trades(id TEXT PRIMARY KEY, strategy TEXT NOT NULL, data TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS trades_by_strategy ON trades(strategy)")
    db.commit()
    if not meta(db, "start_ms"):
        put_meta(db, "start_ms", str(now_ms()))
        LOG.info("Forward-only paper experiments initialized; no historic HBREAK fills imported")


def meta(db, k, default=""):
    r = db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def put_meta(db, k, v):
    db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES (?,?)", (k, str(v)))
    db.commit()


def all_trades(db, strategy=None):
    if strategy:
        records = db.execute("SELECT data FROM trades WHERE strategy=?", (strategy,)).fetchall()
    else:
        records = db.execute("SELECT data FROM trades").fetchall()
    return [json.loads(r[0]) for r in records]


def save(db, trade):
    db.execute("INSERT OR REPLACE INTO trades(id,strategy,data) VALUES (?,?,?)",
               (trade["id"], trade["strategy"], json.dumps(trade, allow_nan=False)))
    db.commit()


def first_full_5m(opened):
    return ((iso_to_ms(opened) + 299999) // 300000) * 300000


def read_hbreak_new(db):
    """Read HBREAK trades without ever writing to the original DB."""
    if not HBREAK_DB.exists():
        return []
    since = int(meta(db, "start_ms"))
    try:
        with sqlite3.connect("file:" + str(HBREAK_DB) + "?mode=ro", uri=True, timeout=2) as source:
            items = [json.loads(row[0]) for row in source.execute("SELECT data FROM trades")]
        return sorted((x for x in items if iso_to_ms(x["opened_at"]) >= since
                       and x.get("archetype") in ("pattern_breakout", "trendline_break", "breakout_retest")),
                      key=lambda x: x["opened_at"])
    except (sqlite3.Error, ValueError, KeyError) as exc:
        LOG.warning("Read-only HBREAK source unavailable (%s)", type(exc).__name__)
        return []


def new_trade(db, strategy, symbol, source_id, signal_at, price, *, watch=False, **extra):
    """Only hypothetical fills observed at module execution time, never hindsight prices."""
    if strategy not in SHEETS or not symbol.endswith("USDT") or num(price, 0) <= 0:
        return False
    identifier = key_for("CB2", strategy, source_id)
    if db.execute("SELECT 1 FROM trades WHERE id=?", (identifier,)).fetchone():
        return False
    opened = now_iso()
    trade = dict.fromkeys(FIELDS)
    trade.update(id=identifier, strategy=strategy, symbol=symbol,
                 status="WATCH" if watch else "OPEN", source_id=source_id,
                 signal_at=signal_at, opened_at=opened,
                 entry=None if watch else float(price),
                 tp=None if watch else float(price) * 1.01,
                 lowest=None if watch else float(price),
                 max_drawdown_pct=None, data_status="WATCH" if watch else "PENDING_DATA",
                 observed_candles=0, checked_at=opened, age_hours=0,
                 cursor=first_full_5m(opened), **extra)
    save(db, trade)
    return True


class MarketClient:
    def __init__(self, http):
        self.http = http
        self.key = (os.getenv("api2") or os.getenv("API2") or "").strip()
        self.daily_left = None
        self.minute_left = None
        self.usage_checked = 0
        self.public_candle_cache = {}
        self.price_cache = {}
        self.new_symbols = set()

    def begin_cycle(self):
        self.public_candle_cache.clear()
        self.price_cache.clear()
        self.new_symbols.clear()

    def usage(self):
        if not self.key:
            return False
        if time.time() - self.usage_checked < 590 and self.daily_left is not None:
            return True
        self.usage_checked = time.time()
        try:
            r = self.http.get(API + "/public/usage", headers={"X-API-Key": self.key}, timeout=12)
            r.raise_for_status()
            result = r.json()
            self.daily_left = num((result.get("today") or {}).get("creditsLeft"))
            self.minute_left = num((result.get("thisMinute") or {}).get("creditsLeft"))
            if self.daily_left is None or self.minute_left is None:
                LOG.warning("api2 quota status missing; extra paid scans paused")
                return False
            LOG.info("api2 shared-account credits left=%d; minute left=%d (reserve=%d)",
                     self.daily_left, self.minute_left, RESERVE)
            return True
        except Exception as exc:
            self.daily_left = None
            LOG.warning("api2 usage unavailable (%s); extra paid scans paused",
                        type(exc).__name__)
            return False

    def cb(self, path, cost, params=None):
        if not self.usage() or self.daily_left is None:
            return None
        # Spend conservatively: preserve HBREAK's 5-credit /10m budget.
        if self.daily_left < RESERVE + cost or self.minute_left is None or self.minute_left < cost + 5:
            return None
        try:
            r = self.http.get(API + path, params=params,
                              headers={"X-API-Key": self.key}, timeout=15)
            if r.status_code == 429:
                LOG.warning("api2 quota 429; pausing paid queries")
                self.daily_left = None
                return None
            r.raise_for_status()
            self.daily_left -= cost
            self.minute_left -= cost
            return r.json()
        except Exception as exc:
            LOG.warning("api2 request %s failed (%s)", path, type(exc).__name__)
            return None

    def klines(self, symbol, tf, start=None, limit=60, track=False):
        if not symbol.endswith("USDT") or not symbol.isalnum() or len(symbol) > 20:
            return []
        if not track:
            if symbol not in self.new_symbols and len(self.new_symbols) >= MAX_NEW_SYMBOLS_PER_CYCLE:
                return []
            self.new_symbols.add(symbol)
        cache_key = (symbol, tf, start, limit)
        if cache_key in self.public_candle_cache:
            return self.public_candle_cache[cache_key]
        try:
            params = {"symbol": symbol, "interval": tf, "limit": limit}
            if start is not None:
                params["startTime"] = int(start)
            r = self.http.get(BINANCE + "/fapi/v1/klines", params=params, timeout=12)
            r.raise_for_status()
            candles = [x for x in r.json() if int(x[6]) < now_ms()]
            self.public_candle_cache[cache_key] = candles
            return candles
        except Exception as exc:
            LOG.warning("Binance public candles %s %s (%s)", symbol, tf, type(exc).__name__)
            return []

    def price(self, symbol):
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        if symbol not in self.new_symbols and len(self.new_symbols) >= MAX_NEW_SYMBOLS_PER_CYCLE:
            return None
        self.new_symbols.add(symbol)
        try:
            r = self.http.get(BINANCE + "/fapi/v1/ticker/price", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            price = num(r.json().get("price"))
            self.price_cache[symbol] = price
            return price
        except Exception:
            return None


def rvol_from_candles(c):
    if len(c) < 24:
        return None
    prior = [float(x[7]) for x in c[-21:-1]]
    return float(c[-1][7]) / (sum(prior) / len(prior)) if sum(prior) > 0 else None


def bullish_closed_break(c):
    """Current CLOSED 15m candle closes above previous 20-candle resistance."""
    if len(c) < 23:
        return False
    last = c[-1]
    previous_high = max(float(x[2]) for x in c[-21:-1])
    return float(last[4]) > previous_high and float(last[4]) > float(last[1])


def close_higher(c, pct=0.002):
    if not c:
        return False
    op, cl = float(c[-1][1]), float(c[-1][4])
    return op > 0 and cl >= op * (1 + pct)


def recent_duplicate(db, strategy, symbol, seconds):
    cutoff = now_ms() - seconds * 1000
    return any(x["symbol"] == symbol and iso_to_ms(x["opened_at"]) > cutoff
               for x in all_trades(db, strategy))


def evaluate_hbreak(db, market, trend_data):
    candidates = read_hbreak_new(db)
    matrix = {x.get("symbol"): x.get("scores") or {} for x in
              ((trend_data or {}).get("items") or []) if isinstance(x, dict)}
    counts = defaultdict(int)
    for row in candidates:
        symbol = row.get("coin") or ""
        rvol = num(row.get("rvol"))
        sqs = num(row.get("sqs"))
        if not symbol.endswith("USDT") or sqs is None or sqs < 50:
            continue
        source_id = row["id"]
        # Paper entry = current public futures ticker, never historical HBREAK entry.
        eligible = []
        if rvol is not None and rvol >= 1.5:
            eligible.append(("rvol", {}))
        if row.get("archetype") == "breakout_retest":
            eligible.append(("retest", {}))
        scores = matrix.get(symbol) or {}
        h1 = num((scores.get("1h") or {}).get("score"))
        h4 = num((scores.get("4h") or {}).get("score"))
        if h1 is not None and h4 is not None and h1 > 0 and h4 > 0:
            eligible.append(("trend", {"trend_1h": h1, "trend_4h": h4}))
        if not eligible:
            continue
        # Do not read a stale signal hours later and enter at a new price.
        signal_age = now_ms() - iso_to_ms(row["opened_at"])
        if signal_age < 0 or signal_age > 20 * 60 * 1000:
            continue
        fresh = [p for p in eligible if not db.execute(
            "SELECT 1 FROM trades WHERE id=?", (key_for("CB2", p[0], source_id),)
        ).fetchone()]
        if not fresh:
            continue
        price = market.price(symbol)
        if not price:
            continue
        for strategy, extra in fresh:
            if new_trade(db, strategy, symbol, source_id, row["opened_at"], price,
                         source_archetype=row["archetype"], sqs=sqs, rvol=rvol, **extra):
                counts[strategy] += 1
    return dict(counts)


def due(db, name, interval):
    return time.time() >= float(meta(db, "next_" + name, "0"))


def set_due(db, name, interval):
    put_meta(db, "next_" + name, int(time.time()) + interval)


def run_paid_board(db, market, name, path, params=None):
    if not due(db, name, INTERVALS[name]):
        return None
    result = market.cb(path, COSTS[name], params)
    if result is not None:
        set_due(db, name, INTERVALS[name])
    return result


def screen_rows(result):
    if not result:
        return []
    if (result.get("meta") or {}).get("stale"):
        return []
    return [r for r in result.get("rows", []) if isinstance(r, dict)
            and r.get("quote") == "USDT" and (r.get("symbol") or "").endswith("USDT")]


def check_watch(db, market):
    changed = 0
    watches = sorted((x for x in all_trades(db, "squeeze") if x["status"] == "WATCH"),
                     key=lambda x: x.get("checked_at") or x["opened_at"])
    for t in watches[:MAX_NEW_SYMBOLS_PER_CYCLE]:
        if now_ms() - iso_to_ms(t["opened_at"]) > 12 * 3600 * 1000:
            t["status"], t["data_status"] = "WATCH_EXPIRED", "OK"
            t["checked_at"] = now_iso()
            save(db, t)
            continue
        candles = market.klines(t["symbol"], "15m", limit=50)
        t["checked_at"] = now_iso()
        save(db, t)
        if not bullish_closed_break(candles):
            continue
        # Breakout must occur AFTER the watch was added.
        if int(candles[-1][6]) < iso_to_ms(t["opened_at"]):
            continue
        price = market.price(t["symbol"])
        if not price:
            continue
        t.update(status="OPEN", data_status="PENDING_DATA", entry=price,
                 tp=price * 1.01, lowest=price,
                 opened_at=now_iso(), checked_at=now_iso(),
                 observed_candles=0, max_drawdown_pct=None)
        t["cursor"] = first_full_5m(t["opened_at"])
        t["rvol"] = rvol_from_candles(candles)
        t["note"] = "15m closed candle broke 20-bar resistance after squeeze WATCH"
        save(db, t)
        changed += 1
    return changed


def evaluate_squeeze(db, market, result):
    count = 0
    for row in screen_rows(result):
        symbol = row["symbol"]
        if recent_duplicate(db, "squeeze", symbol, 12 * 3600):
            continue
        candles = market.klines(symbol, "15m", limit=50)
        if len(candles) < 24:
            continue
        # Store WATCH, not a paper trade. No false entry on the screen alone.
        if new_trade(db, "squeeze", symbol,
                     "squeeze_" + symbol + "_" + str(now_ms() // 3600000),
                     now_iso(), float(candles[-1][4]), watch=True,
                     rvol=rvol_from_candles(candles),
                     note="squeeze preset; waiting for subsequent 15m bullish breakout"):
            count += 1
    return count


def evaluate_volume(db, market, result):
    count = 0
    for row in screen_rows(result):
        symbol = row["symbol"]
        if recent_duplicate(db, "volume", symbol, 3600):
            continue
        candles = market.klines(symbol, "15m", limit=50)
        if not close_higher(candles) or len(candles) < 24:
            continue
        rvol = rvol_from_candles(candles)
        if rvol is None or rvol < 2:
            continue
        price = market.price(symbol)
        if not price:
            continue
        event_key = "volume_" + symbol + "_" + str(int(candles[-1][0]))
        if new_trade(db, "volume", symbol, event_key, stamp_ms(int(candles[-1][6])),
                     price, rvol=round(rvol, 3),
                     note="volume-surge preset + bullish closed futures 15m candle"):
            count += 1
    return count


def liquidation_candidates(board):
    if not board or board.get("stale"):
        return []
    since = now_ms() - 30 * 60 * 1000
    shorts = defaultdict(float)
    for row in board.get("tape") or []:
        if row.get("side") == "short" and num(row.get("ts"), 0) >= since:
            shorts[row.get("symbol")] += num(row.get("notionalUsd"), 0)
    for row in (board.get("cascades") or {}).get("active") or []:
        if row.get("side") == "short" and num(row.get("lastEventAt"), 0) >= since:
            shorts[row.get("symbol")] = max(shorts[row.get("symbol")],
                                             num(row.get("shortUsd"), 0))
    return sorted(((s, amount) for s, amount in shorts.items()
                   if isinstance(s, str) and s.endswith("USDT") and amount >= 25000),
                  key=lambda p: p[1], reverse=True)


def evaluate_oi(db, market, board):
    if not board:
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    counter = "oi_spent_" + today
    used = int(meta(db, counter, "0"))
    if used >= MAX_OI_PER_UTC_DAY or not due(db, "oi", 3600):
        return 0
    for symbol, short_amount in liquidation_candidates(board):
        if recent_duplicate(db, "oi_squeeze", symbol, 3 * 3600):
            continue
        candles = market.klines(symbol, "15m", limit=50)
        if not close_higher(candles):
            continue
        if not market.usage() or market.daily_left is None or market.daily_left < RESERVE + 10:
            return 0
        # One OI lookup per hour and never more than 24 per UTC day.
        data = market.cb("/public/open-interest/" + symbol + "/history", 10,
                         {"period": "15m"})
        if data is None:
            return 0
        put_meta(db, counter, used + 1)
        set_due(db, "oi", 3600)
        points = data.get("points") or []
        if data.get("symbol") != symbol or len(points) < 5:
            return 0
        points = sorted(points, key=lambda p: num(p.get("time"), 0))
        last_time = num(points[-1].get("time"), 0) * 1000
        if now_ms() - last_time > 3600000:
            return 0
        first = num(points[-5].get("oi"))
        last = num(points[-1].get("oi"))
        if not first or last is None:
            return 0
        change = (last / first - 1) * 100
        if change > -0.2:
            return 0
        price = market.price(symbol)
        if not price:
            return 0
        return int(new_trade(
            db, "oi_squeeze", symbol,
            "oisq_" + symbol + "_" + str(int(candles[-1][0])),
            now_iso(), price, oi_change_pct=round(change, 3),
            short_liq_usd=round(short_amount, 2),
            note="short liquidations >=$25k/30m + bullish 15m + OI contracts decline"
        ))
    return 0


def track(db, market):
    # One public 5m request per symbol can update multiple virtual strategies.
    open_rows = [r for r in all_trades(db) if r["status"] == "OPEN"]
    groups = defaultdict(list)
    for t in open_rows:
        groups[t["symbol"]].append(t)
    symbols = sorted(groups, key=lambda s: min(
        iso_to_ms(t.get("checked_at") or t["opened_at"]) for t in groups[s]))
    checked = 0
    bars_total = 0
    for symbol in symbols[:MAX_TRACK_SYMBOLS_PER_CYCLE]:
        active = groups[symbol]
        start = min(int(t.get("cursor") or first_full_5m(t["opened_at"]))
                    for t in active)
        candles = market.klines(symbol, "5m", start=start, limit=200, track=True)
        if not candles:
            # Missing candles is not evidence of 0% drawdown.
            continue
        for t in active:
            cursor = max(first_full_5m(t["opened_at"]), int(t.get("cursor") or 0))
            processed = 0
            for candle in candles:
                if int(candle[0]) < cursor:
                    continue
                processed += 1
                low, high = float(candle[3]), float(candle[2])
                t["lowest"] = min(float(t["lowest"]), low)
                t["last_candle_at"] = stamp_ms(int(candle[6]))
                t["last_price"] = float(candle[4])
                if high >= float(t["tp"]):
                    t["status"] = "TP_1_PERCENT"
                    t["tp_at"] = t["last_candle_at"]
                    t["tp_minutes"] = round((iso_to_ms(t["tp_at"]) -
                                              iso_to_ms(t["opened_at"])) / 60000, 2)
                    if low < float(t["entry"]):
                        t["note"] = ((t.get("note") or "") +
                                     "; intrabar TP/low order unknown (low first assumed)")
                    break
            if processed:
                # Never skip unprocessed candles after TP or a page of history.
                processed_last = [c for c in candles if int(c[0]) >= cursor][processed - 1]
                t["cursor"] = int(processed_last[0]) + 300000
                t["observed_candles"] = int(t.get("observed_candles") or 0) + processed
                t["max_drawdown_pct"] = round(min(0, (float(t["lowest"]) /
                                                       float(t["entry"]) - 1) * 100), 4)
                t["data_status"] = "OK"
                checked += 1
                bars_total += processed
            elif not t.get("observed_candles"):
                t["data_status"] = "PENDING_DATA"
            t["age_hours"] = round((now_ms() - iso_to_ms(t["opened_at"])) / 3600000, 2)
            t["checked_at"] = now_iso()
            if t["status"] == "OPEN" and t["age_hours"] >= 24:
                t["status"] = "OPEN_24H"
            save(db, t)
    # OPEN_24H must remain monitored indefinitely (no stop loss).
    return checked, bars_total


def reopen_24h(db):
    # Keep 24h-old positions eligible for ongoing tracking, with no extra order.
    for t in all_trades(db):
        if t["status"] == "OPEN_24H":
            t["status"] = "OPEN"
            save(db, t)


def sheet_sync(db):
    import gspread
    from google.oauth2.service_account import Credentials
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sid = (os.getenv("COINBEACON_SIX_SHEET_ID") or
           os.getenv("GOOGLE_SHEETS_ID") or os.getenv("GOOGLE_SHEET_ID") or "").strip()
    if not raw or not sid:
        raise ValueError("SHEETS_CREDENTIALS_OR_ID_MISSING")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    auth = (Credentials.from_service_account_info(json.loads(raw), scopes=scopes)
            if raw.startswith("{")
            else Credentials.from_service_account_file(raw, scopes=scopes))
    workbook = gspread.authorize(auth).open_by_key(sid)
    for strategy, tab in SHEETS.items():
        data = sorted(all_trades(db, strategy), key=lambda r: r["opened_at"])
        if len(data) + 1 > MAX_SHEET_ROWS:
            LOG.warning("Sheets %s has %d rows: export trimmed; full JSON remains in R2",
                        tab, len(data))
            data = data[-(MAX_SHEET_ROWS - 1):]
        values = [FIELDS] + [[str(r.get(k) if r.get(k) is not None else "")
                               for k in FIELDS] for r in data]
        try:
            ws = workbook.worksheet(tab)
        except gspread.WorksheetNotFound:
            ws = workbook.add_worksheet(title=tab, rows=1000, cols=len(FIELDS))
        if ws.row_count < len(values):
            ws.add_rows(len(values) - ws.row_count + 100)
        ws.update(range_name="A1", values=values)
    LOG.info("Google Sheets synced 6 isolated CB2 tabs")


def json_archive(db, current=None):
    """One separate cumulative JSON per Paris day, encrypted/private R2 bucket."""
    current = current or datetime.now(PARIS)
    if current.hour == 0 and current.minute < 10:
        return
    # Snapshot every Paris day at/after 21:00; full history stays cumulative.
    if current.hour < 21:
        return
    date = current.date().isoformat()
    if meta(db, "json_day") == date:
        return
    from crash_report_storage import r2_configured, R2_ENV_NAMES
    if not r2_configured():
        raise ValueError("R2_NOT_CONFIGURED")
    import boto3
    all_rows = all_trades(db)
    stats = {strategy: {
        "total": sum(t["strategy"] == strategy for t in all_rows),
        "tp": sum(t["strategy"] == strategy and t["status"] == "TP_1_PERCENT"
                  for t in all_rows),
        "open": sum(t["strategy"] == strategy and t["status"] in ("OPEN", "OPEN_24H")
                    for t in all_rows),
        "watch": sum(t["strategy"] == strategy and t["status"] == "WATCH"
                     for t in all_rows),
    } for strategy in SHEETS}
    payload = {
        "date_paris": date, "as_of_utc": now_iso(), "mode": "PAPER_ONLY",
        "api_key_env": "api2", "source": "CoinBeacon + public Binance futures",
        "daily_cumulative": True, "strategy_tabs": SHEETS,
        "statistics": stats, "trades": all_rows,
    }
    env = os.environ
    account, bucket, access, secret = [env[n].strip() for n in R2_ENV_NAMES]
    client = boto3.client("s3", endpoint_url="https://" + account + ".r2.cloudflarestorage.com",
                          aws_access_key_id=access, aws_secret_access_key=secret,
                          region_name="auto")
    folder = current.strftime("%Y/%m/%d")
    key = "coinbeacon-api2/" + folder + "/coinbeacon_six_" + date + ".json"
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body,
                      ContentType="application/json")
    if int(client.head_object(Bucket=bucket, Key=key)["ContentLength"]) != len(body):
        raise ValueError("R2_JSON_UPLOAD_SIZE_MISMATCH")
    put_meta(db, "json_day", date)
    LOG.info("Private Cloudflare R2 JSON saved key=%s rows=%d", key, len(all_rows))


def cycle(db, market):
    market.begin_cycle()
    market.usage()
    # A previously verified matrix may be reused between hourly requests,
    # but must never be treated as fresh beyond 90 minutes.
    trend = run_paid_board(db, market, "trend", "/public/trend-scanner",
                           {"exchange": "binance-futures", "sort": "1h",
                            "dir": "desc", "quote": "USDT"})
    if trend:
        put_meta(db, "trend_cache", json.dumps({"at": time.time(), "data": trend}))
    if not trend:
        try:
            cached = json.loads(meta(db, "trend_cache", "{}"))
            if time.time() - cached.get("at", 0) <= 5400:
                trend = cached.get("data")
        except (ValueError, TypeError):
            pass
    counts = evaluate_hbreak(db, market, trend)
    squeeze = run_paid_board(db, market, "squeeze", "/public/screener",
                             {"preset": "squeeze"})
    if squeeze:
        counts["squeeze_watch"] = evaluate_squeeze(db, market, squeeze)
    counts["squeeze_triggered"] = check_watch(db, market)
    volume = run_paid_board(db, market, "volume", "/public/screener",
                            {"preset": "volume-surge"})
    if volume:
        counts["volume"] = evaluate_volume(db, market, volume)
    liq = run_paid_board(db, market, "liquidations", "/public/liquidations/board")
    if liq:
        counts["oi_squeeze"] = evaluate_oi(db, market, liq)
    reopen_24h(db)
    checked, bars = track(db, market)
    LOG.info("Cycle paper_new=%s tracked=%d candle_bars=%d, total=%d",
             counts, checked, bars, len(all_trades(db)))
    try:
        sheet_sync(db)
    except Exception as exc:
        LOG.warning("Sheets sync failed (%s): %s", type(exc).__name__, str(exc)[:140])
    try:
        json_archive(db)
    except Exception as exc:
        LOG.warning("R2 JSON failed (%s): %s", type(exc).__name__, str(exc)[:100])


def run():
    BASE.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB, timeout=10) as db, requests.Session() as http:
        initialize(db)
        client = MarketClient(http)
        LOG.info("6 paper strategies started; api2_configured=%s, orders=DISABLED",
                 bool(client.key))
        while True:
            try:
                cycle(db, client)
            except Exception as exc:
                LOG.warning("Paper cycle failed (%s): %s", type(exc).__name__, str(exc)[:170])
            time.sleep(max(15, 600 - int(time.time()) % 600))


def start():
    thread = threading.Thread(target=run, name="coinbeacon-six-paper-only", daemon=True)
    thread.start()
    return thread
