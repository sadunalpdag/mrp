"""CB2 live momentum paper journal (read-only, NO Binance orders).

Reuses the existing CoinBeacon /public/setups response. It is deliberately
separate from the six original experiments and Crash Bounce's executor.
"""
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

LOG = logging.getLogger("cb2_live_momentum")
PARIS = ZoneInfo("Europe/Paris")
FIFTEEN_MS = 900000
ALLOWED = {"pattern_breakout", "trendline_break", "breakout_retest"}
SHEET = "CB2_MOMENTUM_CANLI"
MAX_NEW_CANDLES_PER_CYCLE = 12
MAX_TRACK_CANDLES_PER_CYCLE = 8
MAX_EXTRA_CREDITS_PER_UTC_DAY = 350
MAX_QUOTES_PER_UTC_DAY = 90
MAX_ROWS = 3000


def _now_ms():
    return int(time.time() * 1000)


def _number(x):
    try:
        n = float(x)
        return n if n == n and abs(n) < float("inf") else None
    except (TypeError, ValueError, OverflowError):
        return None


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


def _day(ms):
    return datetime.fromtimestamp(ms / 1000, PARIS).date().isoformat()


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS cb2_momentum_live
                  (id TEXT PRIMARY KEY, payload TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS cb2_momentum_meta
                  (k TEXT PRIMARY KEY, v TEXT NOT NULL)""")
    db.commit()


def _read(db, key, default=""):
    row = db.execute("SELECT v FROM cb2_momentum_meta WHERE k=?", (key,)).fetchone()
    return row[0] if row else default


def _write(db, key, value):
    db.execute("INSERT OR REPLACE INTO cb2_momentum_meta VALUES (?,?)", (key, str(value)))
    db.commit()


def _save(db, item):
    db.execute("INSERT OR REPLACE INTO cb2_momentum_live VALUES (?,?)",
               (item["id"], json.dumps(item, allow_nan=False, ensure_ascii=False)))
    db.commit()


def records(db):
    return [json.loads(x[0]) for x in db.execute(
        "SELECT payload FROM cb2_momentum_live ORDER BY rowid DESC LIMIT ?", (MAX_ROWS,))]


def _ema(values, period):
    k = 2.0 / (period + 1)
    last = values[0]
    out = []
    for value in values:
        last = value * k + last * (1 - k)
        out.append(last)
    return out


def _adx(bars, period=14):
    """Wilder ADX14 with warm-up; uses CLOSED candles only."""
    trs, ups, downs = [], [], []
    for prev, curr in zip(bars, bars[1:]):
        high, low = curr["high"], curr["low"]
        ph, pl, pc = prev["high"], prev["low"], prev["close"]
        up, down = high - ph, pl - low
        trs.append(max(high-low, abs(high-pc), abs(low-pc)))
        ups.append(up if up > down and up > 0 else 0.0)
        downs.append(down if down > up and down > 0 else 0.0)
    if len(trs) < 2*period:
        return None
    tr, plus, minus = sum(trs[:period]), sum(ups[:period]), sum(downs[:period])
    dxs = []
    for i in range(period-1, len(trs)):
        if i >= period:
            tr = tr - tr / period + trs[i]
            plus = plus - plus / period + ups[i]
            minus = minus - minus / period + downs[i]
        pdi, mdi = (100*plus/tr if tr else 0), (100*minus/tr if tr else 0)
        dxs.append(100*abs(pdi-mdi)/(pdi+mdi) if pdi+mdi else 0)
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = ((period-1)*adx + dx)/period
    return adx


def _closed_bars(response, at_ms):
    if not isinstance(response, dict) or response.get("timeframe") != "15m" \
       or response.get("exchange") != "binance-futures":
        return []
    out = []
    for raw in response.get("bars") or []:
        try:
            opened = int(raw["openTime"])
            closed = int(raw["closeTime"])
            prices = {key: _number(raw.get(key)) for key in ("open", "high", "low", "close")}
            if any(value is None or value <= 0 for value in prices.values()):
                continue
            if closed > at_ms or opened + FIFTEEN_MS - 1 != closed:
                continue
            out.append(dict(open_ms=opened, close_ms=closed, **prices))
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda b: b["open_ms"])
    clean = []
    for bar in out:
        if clean and bar["open_ms"] == clean[-1]["open_ms"]:
            clean[-1] = bar
        else:
            clean.append(bar)
    return clean


def indicators(bars, at_ms):
    """500 requested bars allow 480 fully closed bars after dropping forming bar."""
    bars = [b for b in bars if b["close_ms"] <= at_ms]
    if len(bars) < 480:
        return None, "INSUFFICIENT_480_CLOSED_BARS"
    bars = bars[-480:]
    if any(b["open_ms"] - a["open_ms"] != FIFTEEN_MS for a,b in zip(bars,bars[1:])):
        return None, "MISSING_15M_CANDLES"
    if at_ms - bars[-1]["close_ms"] > 30*60000:
        return None, "STALE_15M_CACHE"
    closes = [b["close"] for b in bars]
    e12, e26 = _ema(closes,12), _ema(closes,26)
    macd = [a-b for a,b in zip(e12,e26)]
    histogram = macd[-1] - _ema(macd,9)[-1]
    adx = _adx(bars)
    if adx is None or closes[-97] <= 0:
        return None, "INDICATOR_UNAVAILABLE"
    change24 = 100*(closes[-1]/closes[-97]-1)
    return {
        "adx14": round(adx,4),
        "macd_histogram": round(histogram,12),
        "change24_pct": round(change24,4),
        "candle_close_utc": _iso(bars[-1]["close_ms"]),
    }, "OK"


def _capacity_shadow(at_ms):
    """Read saved Crash daily counter only: NOT an exchange slot check."""
    basedir = os.getenv("CRASH_NIGHT_STATE_DIR", "").strip()
    base = Path(basedir) if basedir else Path(os.getenv("DATA_DIR", "/data"))/"night_execution"
    try:
        state = json.loads((base/"crash_night_state.json").read_text())
        settings = state.get("trade_settings") or {}
        maximum = int(settings.get("max_daily", 50))
        today = _day(at_ms)
        orders = state.get("orders") or {}
        iterable = orders.values() if isinstance(orders, dict) else orders
        raw = sum(x.get("day") == today and not x.get("entry_never_sent",False)
                  for x in iterable if isinstance(x,dict))
        override = state.get("daily_counter_override") or {}
        count = max(0, int(override["count"])+raw-int(override["raw_at_set"])) \
            if override.get("day") == today else raw
        return {"status": "DAILY_LIMIT_REACHED" if count >= maximum else "DAILY_LIMIT_AVAILABLE",
                "used": count, "limit": maximum,
                "note": "Read-only saved Crash counter; current Binance open/pending capacity not verified"}
    except (OSError, ValueError, TypeError, KeyError):
        return {"status": "CAPACITY_UNKNOWN",
                "note": "Crash local counter unavailable; Binance capacity not queried"}


def _api_bars(db, market, symbol, day, cache=None):
    # Reuse a single call when the same symbol has several CB2 setups.
    if cache is not None and symbol in cache:
        return cache[symbol]
    key = "credits_" + day
    used = int(_read(db,key,"0"))
    if used >= MAX_EXTRA_CREDITS_PER_UTC_DAY:
        return None, "CB2_DAILY_BUDGET_EXHAUSTED"
    # MarketClient.cb checks shared-account quota and preserves 400 credits
    # for the existing CoinBeacon scanners. Never bypass that protection.
    result = market.cb("/public/ohlcv/binance-futures/" + symbol + "/15m",
                       1, {"limit": 500})
    if result is None:
        return None, "COINBEACON_QUOTA_OR_REQUEST_UNAVAILABLE"
    _write(db,key,used+1)
    if result.get("symbol") != symbol or result.get("exchange") != "binance-futures":
        return None, "OHLCV_WRONG_SYMBOL"
    output = result, "OK"
    if cache is not None:
        cache[symbol] = output
    return output


def _quote_now(db, market, symbol, utc_day):
    """Fresh CoinBeacon batch quote. Never substitute stale setup snapshot."""
    key = "quotes_" + utc_day
    used = int(_read(db, key, "0"))
    if used >= MAX_QUOTES_PER_UTC_DAY:
        return None, "QUOTE_BUDGET_EXHAUSTED"
    pair = "binance-futures:" + symbol
    response = market.cb("/public/market/quotes/batch", 5, {"m": pair})
    if response is None:
        return None, "QUOTE_QUOTA_OR_REQUEST_UNAVAILABLE"
    _write(db, key, used+1)
    if not isinstance(response, dict):
        return None, "QUOTE_BAD_RESPONSE"
    for row in response.get("quotes") or []:
        if row.get("m") == pair and row.get("venueId") == "binance-futures":
            p = _number(row.get("price"))
            if p is None or p <= 0:
                return None, "QUOTE_NO_PRICE"
            # Batch quotes are live/cache snapshots; age is recorded, not guessed.
            asof = _number(response.get("asOf"))
            if asof is None or abs(_now_ms() - asof) > 60 * 1000:
                return None, "QUOTE_STALE"
            return (p, int(asof)), "OK"
    return None, "QUOTE_SYMBOL_MISSING"


def _notify(item):
    """Alerts mean observed paper candidates, NEVER a submitted Binance BUY."""
    import requests
    token = os.getenv("BOT_TOKEN","").strip()
    chat = os.getenv("CHAT_ID","").strip()
    if not token or not chat:
        return False
    data = item["indicators"]
    msg = ("📣 CB2 ÜÇLÜ FİLTRE | SANAL LONG\n"
           + item["symbol"] + " | " + item["archetype"] + "\n"
           + "SQS " + str(item["sqs"]) + " | ADX14 " + str(data["adx14"]) + "\n"
           + "24s %" + str(data["change24_pct"]) + " | MACD " + str(data["macd_histogram"]) + "\n"
           + "Saat (Fransa): " + datetime.fromtimestamp(
               item["observed_ms"]/1000,PARIS).strftime("%d.%m %H:%M") + "\n"
           + "Durum: " + item["capacity"]["status"] + " | %1 TP sanal izleme\n"
           + "Binance'e GERÇEK EMİR GÖNDERİLMEDİ.")
    try:
        response=requests.post("https://api.telegram.org/bot"+token+"/sendMessage",
                               data={"chat_id":chat,"text":msg},timeout=10)
        return bool(response.ok and response.json().get("ok"))
    except (requests.RequestException,ValueError):
        return False


def _update_paper(item, bars, now_ms):
    if item["status"] not in ("PAPER_PENDING","PAPER_OPEN","DATA_GAP"):
        return False
    entry_ms = item["entry_bar_ms"]
    previous = item.get("last_bar_ms")
    if previous is None:
        previous = entry_ms - FIFTEEN_MS
    eligible = [x for x in bars if x["open_ms"] >= entry_ms and
                x["close_ms"] <= now_ms and x["open_ms"] > previous]
    changed=False
    for bar in eligible:
        expected = (item.get("last_bar_ms") or (entry_ms-FIFTEEN_MS)) + FIFTEEN_MS
        if bar["open_ms"] > expected:
            item["data_gap"] = True
        if item.get("entry_price") is None:
            # No paper fill unless a real-time quote was captured at approval.
            item["status"]="PENDING_PRICE"
            item["data_note"]="MISSING_LIVE_ENTRY_QUOTE"
            break
        item["lowest_price"]=min(item["lowest_price"],bar["low"])
        item["max_drawdown_pct"]=round(100*(item["lowest_price"]/item["entry_price"]-1),4)
        item["last_bar_ms"]=bar["open_ms"]
        item["last_checked_at"]=_iso(now_ms)
        item["status"]="PAPER_OPEN"
        changed=True
        if bar["high"] >= item["tp_price"]:
            item["status"]="TP_1_PERCENT"
            item["tp_at_candle_close"]=_iso(bar["close_ms"])
            item["tp_minutes_upper_bound"]=round((bar["close_ms"]-entry_ms+1)/60000,2)
            item["ohlc_intrabar_order_uncertain"]=bar["low"] < item["entry_price"]
            break
    if (item["status"] == "PAPER_OPEN" and now_ms >= entry_ms + 48*3600000
            and item.get("last_bar_ms",0) >= entry_ms + 48*3600000-FIFTEEN_MS):
        item["status"]="INCOMPLETE_48H" if item.get("data_gap") else "OPEN_48H"
        changed=True
    return changed


def cycle(db, market, setups, observed_iso):
    """Store all candidates, evaluate three filters, track filtered paper outcomes.

    The caller owns the existing paid setups fetch; no second setups GET.
    """
    initialize(db)
    from cb2_independent_signals import _fresh_setups
    now = int(datetime.fromisoformat(observed_iso.replace("Z","+00:00")).timestamp()*1000)
    utc_day = datetime.fromtimestamp(now/1000,timezone.utc).date().isoformat()
    new, passed, updates = 0, 0, 0
    candle_cache = {}
    if isinstance(setups,dict):
        for row, kind, event_ms in _fresh_setups(setups, observed_iso):
            sqs = _number(row.get("sqs"))
            if sqs is None or sqs < 70 or kind not in ALLOWED:
                continue
            sym = row["symbol"]
            identifier=hashlib.sha256(
                (sym+"|"+kind+"|"+str(event_ms)).encode()).hexdigest()[:24]
            if db.execute("SELECT 1 FROM cb2_momentum_live WHERE id=?", (identifier,)).fetchone():
                continue
            item = {
                "id":identifier,"symbol":sym,"archetype":kind,"sqs":sqs,
                "observed_ms":now,"observed_at":_iso(now),
                "source_signal_at":_iso(event_ms),"reference_price":_number(row.get("price")),
                "status":"PENDING_CANDLES","filters_passed":None,
                "indicators":None,"data_note":"WAITING_COINBEACON_OHLCV",
                "capacity":None,"entry_bar_ms":None,
                "entry_price":None,"tp_price":None,"data_gap":False,
                "last_bar_ms":None,"last_candle_request_ms":0,
            }
            _save(db,item)
            new+=1
    candidates=[x for x in records(db) if x["status"]=="PENDING_CANDLES"]
    for item in sorted(candidates,key=lambda x:x["observed_ms"])[:MAX_NEW_CANDLES_PER_CYCLE]:
        # Only evaluate candles available at ORIGINAL signal time, not future data.
        # Missing candles are retried, but when historical signal-time data are
        # no longer available we leave the record pending, never peek ahead.
        if now-item["observed_ms"] > 6*3600000:
            item["status"]="INDICATORS_UNAVAILABLE"
            item["data_note"]="SIGNAL_TIME_CANDLE_HISTORY_EXPIRED"
            _save(db,item)
            continue
        result, reason = _api_bars(db,market,item["symbol"],utc_day,candle_cache)
        if result is None:
            item["data_note"]=reason
            _save(db,item)
            if "BUDGET" in reason or "QUOTA" in reason:
                break
            continue
        bars=_closed_bars(result,item["observed_ms"])
        ind, reason=indicators(bars,item["observed_ms"])
        if ind is None:
            item["data_note"]=reason
            _save(db,item)
            continue
        item["indicators"]=ind
        item["data_note"]="OK"
        item["filters_passed"]=(ind["adx14"]>=20
             and ind["change24_pct"]>=10 and ind["macd_histogram"]>0)
        item["status"]="PENDING_PRICE" if item["filters_passed"] else "REJECTED_FILTER"
        passed+=int(item["filters_passed"])
        _save(db,item)
    # Capture a current quote immediately after indicator approval.
    # No 0.5%/1% pullback and no fabricated next-15m-open entry.
    for item in [x for x in records(db) if x["status"] == "PENDING_PRICE"]:
        if now - item["observed_ms"] >= FIFTEEN_MS:
            item["status"]="QUOTE_EXPIRED_NO_ENTRY"
            item["data_note"]="QUOTE_NOT_AVAILABLE_WITHIN_15M_OF_FIRST_OBSERVATION"
            _save(db,item)
            continue
        quote, reason = _quote_now(db,market,item["symbol"],utc_day)
        if quote is None:
            item["data_note"]=reason
            _save(db,item)
            if "QUOTA" in reason or "BUDGET" in reason:
                break
            continue
        price, price_ms=quote
        item["entry_price"]=price
        item["tp_price"]=round(price*1.01,12)
        item["entered_at"]=_iso(price_ms)
        # The entry candle is only partly observable using 15m OHLCV.
        # Begin TP analysis at the first FULL candle after quote time;
        # never claim a TP occurred before the paper entry.
        item["entry_bar_ms"]=((price_ms//FIFTEEN_MS)+1)*FIFTEEN_MS
        item["lowest_price"]=price
        item["capacity"]=_capacity_shadow(price_ms)
        item["status"]="PAPER_OPEN"
        item["data_note"]="LIVE_QUOTE_ENTRY_FIRST_PARTIAL_BAR_EXCLUDED"
        _save(db,item)
        item["notified"]=bool(_notify(item))
        _save(db,item)
    # Intraday follow-up uses budgeted closed OHLCV; late/missing bars remain
    # eligible for the existing next-day published Binance ZIP archive analysis.
    pending=[x for x in records(db)
             if x["status"] in ("PAPER_OPEN","DATA_GAP")
             and now-x.get("last_candle_request_ms",0)>=FIFTEEN_MS]
    for item in sorted(pending,key=lambda x:x.get("last_candle_request_ms",0))[:MAX_TRACK_CANDLES_PER_CYCLE]:
        result,reason=_api_bars(db,market,item["symbol"],utc_day,candle_cache)
        item["last_candle_request_ms"]=now
        if result is None:
            item["data_note"]=reason
            _save(db,item)
            if "BUDGET" in reason or "QUOTA" in reason:
                break
            continue
        bars=_closed_bars(result,now)
        updates+=int(_update_paper(item,bars,now))
        _save(db,item)
    LOG.info("[CB2 MOMENTUM] new=%d pass=%d updated=%d tracked=%d",
             new,passed,updates,len(records(db)))
    return {"new":new,"passed":passed,"updates":updates}


def export(db):
    initialize(db)
    items=records(db)
    now=_now_ms()
    days=defaultdict(lambda: {"candidates":0,"passed":0,"tp":0,
                               "paper_open":0,"blocked_daily":0,
                               "missing_indicators":0})
    for x in items:
        d=days[_day(x["observed_ms"])]
        d["candidates"]+=1
        d["passed"]+=int(x.get("filters_passed") is True)
        d["tp"]+=int(x["status"]=="TP_1_PERCENT")
        d["paper_open"]+=int(x["status"] in ("PENDING_PRICE","PAPER_OPEN","DATA_GAP"))
        d["blocked_daily"]+=int((x.get("capacity") or {}).get("status")=="DAILY_LIMIT_REACHED")
        d["missing_indicators"]+=int(x["status"] in ("PENDING_CANDLES","INDICATORS_UNAVAILABLE"))
    return {"generated_at_utc":_iso(now),"mode":"LIVE_SIGNAL_PAPER_ONLY_NO_BINANCE_ORDERS",
            "rules":{"adx14_min":20,"change24_pct_min":10,"macd_histogram_gt":0,
                     "sqs_min":70,"tp_pct":1.0,"candle_tf":"15m"},
            "note":"OHLC paper fills not exchange executions; daily Crash capacity snapshot is read-only",
            "by_paris_day":dict(sorted(days.items(),reverse=True)),
            "count":len(items),"signals":items}


def sheet_sync(db):
    """Standalone tab; existing CB2 and Crash Sheets tabs are unchanged."""
    import gspread
    from google.oauth2.service_account import Credentials
    raw=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON","").strip()
    sheet_id=(os.getenv("COINBEACON_SIX_SHEET_ID") or
              os.getenv("GOOGLE_SHEETS_ID") or os.getenv("GOOGLE_SHEET_ID") or "").strip()
    if not raw or not sheet_id:
        raise ValueError("CB2_MOMENTUM_SHEETS_NOT_CONFIGURED")
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
    creds=Credentials.from_service_account_info(json.loads(raw),scopes=scopes) \
        if raw.startswith("{") else Credentials.from_service_account_file(raw,scopes=scopes)
    workbook=gspread.authorize(creds).open_by_key(sheet_id)
    data=export(db)
    fields=["Gün (Paris)","Coin","Formasyon","Durum","Sinyal UTC","SQS",
            "ADX14","24s %","MACD histogram","Filtre","Sanal Giriş",
            "TP +%1","TP zamanı","Maks. düşüş %","Alış limiti gölge",
            "Crash kullanılan/limit","Telegram","Veri durumu","Kaynak ID"]
    values=[["CB2 CANLI ÜÇLÜ FİLTRE | SADECE SANAL TAKİP | GERÇEK EMİR YOK"],
            ["Veri UTC: "+data["generated_at_utc"]],
            ["Tarih","Aday","Geçen","TP","Sanal açık/bekleyen","Crash günlük limite takılan","İndikatör bekleyen"]]
    for day,summary in list(data["by_paris_day"].items())[:30]:
        values.append([day,summary["candidates"],summary["passed"],summary["tp"],
                       summary["paper_open"],summary["blocked_daily"],summary["missing_indicators"]])
    values.append([])
    values.append(fields)
    for x in data["signals"][:2000]:
        metrics=x.get("indicators") or {}
        cap=x.get("capacity") or {}
        values.append([_day(x["observed_ms"]),x["symbol"],x["archetype"],x["status"],
                       x["observed_at"],x["sqs"],metrics.get("adx14"),
                       metrics.get("change24_pct"),metrics.get("macd_histogram"),
                       x.get("filters_passed"),x.get("entry_price"),x.get("tp_price"),
                       x.get("tp_at_candle_close"),x.get("max_drawdown_pct"),
                       cap.get("status"),str(cap.get("used",""))+"/"+str(cap.get("limit","")),
                       x.get("notified",False),x.get("data_note"),x["id"]])
    try:
        ws=workbook.worksheet(SHEET)
    except gspread.WorksheetNotFound:
        ws=workbook.add_worksheet(title=SHEET,rows=max(1000,len(values)+100),cols=len(fields))
    if ws.row_count<len(values):
        ws.add_rows(len(values)-ws.row_count+100)
    ws.update(range_name="A1",values=values,value_input_option="RAW")
    LOG.info("[CB2 MOMENTUM SHEETS] rows=%d tab=%s",len(values),SHEET)


def r2_upload(db, client, bucket, current):
    """Separate cumulative JSON; caller supplies its already-authenticated R2 client."""
    data=export(db)
    date=current.date().isoformat()
    key="cb2/live-momentum/"+current.strftime("%Y/%m/%d")+"/cb2_momentum_"+date+".json"
    body=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    client.put_object(Bucket=bucket,Key=key,Body=body,ContentType="application/json")
    LOG.info("[CB2 MOMENTUM R2] key=%s count=%d",key,data["count"])
    return key
