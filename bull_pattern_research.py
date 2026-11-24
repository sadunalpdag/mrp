"""Bull Flag + Bull Pennant research only; no broker/order imports."""
import gzip
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import bull_flag_paper as legacy

LOG = logging.getLogger("bull_pattern_research")
FRAMES = ("15m", "1h", "4h", "1d")
STEP_MS = 900_000
EIGHT = 8 * 3600_000
DAY = 24 * 3600_000
MAX_NEW_PER_SCAN = 12
_STARTED = False


def database():
    root = Path(os.getenv("DATA_DIR", "/data"))
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "bull_pattern_research.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS patterns(
      id TEXT PRIMARY KEY, symbol TEXT NOT NULL, tf TEXT NOT NULL,
      kind TEXT NOT NULL, first_ms INTEGER NOT NULL, last_ms INTEGER NOT NULL,
      status TEXT NOT NULL, entry REAL, entry_ms INTEGER, target REAL,
      outcome_8h TEXT, outcome_24h TEXT, archived INTEGER DEFAULT 0,
      payload TEXT NOT NULL)""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_patterns_pending ON patterns(first_ms,archived)")
    db.commit()
    return db


def identity(row, item, tf):
    # Prefer provider ID; otherwise use stable pole geometry, never changing price/score.
    provider_id = row.get("id") or item.get("id") or item.get("patternId")
    if provider_id is not None and str(provider_id).strip():
        anchor = str(provider_id)
    else:
        flag = item.get("flag") or {}
        a, b = flag.get("poleStart") or {}, flag.get("poleEnd") or {}
        if not a.get("time") or not b.get("time"):
            return None  # do not invent identity or suppress separate formations
        anchor = str(a["time"]) + ":" + str(b["time"])
    sym = row.get("symbol")
    kind = str(item.get("type", "")).lower().replace(" ", "-")
    return hashlib.sha256(("%s|%s|%s|%s" % (sym, tf, kind, anchor)).encode()).hexdigest()[:32]


def level(obj):
    if isinstance(obj, (int, float, str)):
        return legacy._number(obj)
    if isinstance(obj, dict):
        for k in ("price", "value", "current", "endPrice", "lastPrice"):
            v = legacy._number(obj.get(k))
            if v:
                return v
    return None


def extract(row, tf):
    if not isinstance(row, dict) or not isinstance(row.get("item"), dict):
        return None
    item = row["item"]
    kind = str(item.get("type", "")).lower().replace(" ", "-")
    sym = row.get("symbol", "")
    if kind not in ("bull-flag", "bull-pennant") or not isinstance(sym, str) or not sym.endswith("USDT"):
        return None
    if row.get("exchange") != "binance-futures" or row.get("tf") != tf:
        return None
    lower, upper = level(item.get("lower")), level(item.get("upper"))
    status = str(item.get("status") or "unknown").lower()
    if status not in ("forming", "near", "broken", "retest", "failed"):
        status = "unknown"
    flag = item.get("flag") or {}
    position = flag.get("position")
    if not isinstance(position, (int, float)) or not 0 <= position <= 1:
        position = None
    return sym, kind, status, lower, upper, position, item


def _closed(now):
    return now // STEP_MS * STEP_MS


def _outcome(bars, entry, target, entry_ms):
    if not entry or not target or not entry_ms:
        return "NO_ENTRY"
    for b in bars:
        if b[0] < entry_ms:
            continue
        if b[2] >= target:
            return "TP_1_PERCENT"
    return "NO_TP"


def _store(db, obj):
    db.execute("UPDATE patterns SET last_ms=?,status=?,entry=?,entry_ms=?,target=?,"
               "outcome_8h=?,outcome_24h=?,archived=?,payload=? WHERE id=?",
               (obj["last_ms"],obj["status"],obj["entry"],obj["entry_ms"],
                obj["target"],obj["outcome_8h"],obj["outcome_24h"],obj["archived"],
                json.dumps(obj["payload"], separators=(",", ":"), allow_nan=False),obj["id"]))
    db.commit()


def _r2(db, obj):
    if obj["archived"] or not obj["outcome_24h"]:
        return
    try:
        from btc_hourly_direction import _r2_client
        client, bucket = _r2_client()
        if client is None:
            return
        stamp = datetime.fromtimestamp(obj["first_ms"]/1000, timezone.utc).strftime("%Y-%m")
        key = "bull-patterns-learning/%s/%s.json.gz" % (stamp,obj["id"])
        body = gzip.compress(json.dumps(obj["payload"], separators=(",", ":"), allow_nan=False).encode())
        client.put_object(Bucket=bucket,Key=key,Body=body,
                          ContentType="application/json",ContentEncoding="gzip")
        obj["archived"] = 1
        obj["payload"]["r2_key"] = key
        _store(db,obj)
    except Exception as exc:
        LOG.warning("Bull patterns R2 pending: %s",type(exc).__name__)


def followups(db, now):
    for row in db.execute("SELECT * FROM patterns WHERE archived=0 ORDER BY first_ms LIMIT 100").fetchall():
        obj = dict(row)
        obj["payload"] = json.loads(obj["payload"])
        age = now - obj["first_ms"]
        if age >= EIGHT + STEP_MS and not obj["outcome_8h"]:
            try:
                bars, _ = legacy._bars(obj["symbol"], obj["first_ms"]+EIGHT+STEP_MS,40)
                bars = [b for b in bars if obj["first_ms"] <= b[0] < obj["first_ms"]+EIGHT]
                obj["payload"]["candles_8h"] = bars
                obj["payload"]["complete_8h"] = len(bars) == 32 and all(
                    b[0] - a[0] == STEP_MS for a, b in zip(bars, bars[1:]))
                if obj["payload"]["complete_8h"]:
                    obj["outcome_8h"] = _outcome(bars,obj["entry"],obj["target"],obj["entry_ms"])
                    _store(db,obj)
            except Exception as exc:
                LOG.warning("Bull patterns 8h followup: %s",type(exc).__name__)
        if age >= DAY + STEP_MS and not obj["outcome_24h"]:
            try:
                # Fetch all closed 15m bars after discovery; 96 bars for 24 hours.
                bars, complete = legacy._bars(obj["symbol"], obj["first_ms"]+DAY+STEP_MS,110)
                bars = [b for b in bars if obj["first_ms"] <= b[0] < obj["first_ms"]+DAY]
                obj["payload"]["candles_24h"] = bars
                obj["payload"]["complete_24h"] = len(bars) == 96 and all(
                    b[0]-a[0] == STEP_MS for a,b in zip(bars,bars[1:]))
                if obj["payload"]["complete_24h"]:
                    obj["outcome_24h"] = _outcome(bars,obj["entry"],obj["target"],obj["entry_ms"])
                    obj["payload"]["result_24h"] = obj["outcome_24h"]
                    _store(db,obj)
            except Exception as exc:
                LOG.warning("Bull patterns 24h followup: %s",type(exc).__name__)
        if obj["outcome_24h"]:
            _r2(db,obj)


def _sheet(db):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        from crash_sheets_dashboard import DEFAULT_EMA_FILTER_ID
        raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON","").strip()
        sid = (os.getenv("BULL_FLAG_SHEETS_ID") or os.getenv("CRASH_SHEETS_TARGET_ID")
               or DEFAULT_EMA_FILTER_ID).strip()
        if not raw or not sid:
            print("[BULL PATTERN SHEETS] missing credentials or sheet ID",flush=True)
            LOG.warning("Bull patterns Sheets not configured")
            return
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            info = None
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = (Credentials.from_service_account_info(info,scopes=scopes) if isinstance(info,dict)
                 else Credentials.from_service_account_file(raw,scopes=scopes))
        book = gspread.authorize(creds).open_by_key(sid)
        try:
            sheet = book.worksheet("BULL_PATTERNS_LEARNING")
        except gspread.WorksheetNotFound:
            sheet = book.add_worksheet(title="BULL_PATTERNS_LEARNING",rows=1500,cols=18)
        values = [["ID","Coin","Pattern","TF","First UTC","Last UTC","Status",
                   "Position","Lower","Upper","Entry +1% target","Paper entry",
                   "TP 8h","TP 24h","Observations","24h closed bars"]]
        for row in db.execute("SELECT * FROM patterns ORDER BY first_ms DESC LIMIT 1000"):
            r=dict(row); p=json.loads(r["payload"]); latest=p.get("latest",{})
            values.append([r["id"],r["symbol"],r["kind"],r["tf"],
                datetime.fromtimestamp(r["first_ms"]/1000,timezone.utc).isoformat(),
                datetime.fromtimestamp(r["last_ms"]/1000,timezone.utc).isoformat(),
                r["status"],latest.get("position",""),latest.get("lower",""),
                latest.get("upper",""),r["target"] or "",r["entry"] or "",
                r["outcome_8h"] or "",r["outcome_24h"] or "",
                len(p.get("observations",[])),len(p.get("candles_24h",[]))])
        sheet.update(range_name="A1",values=values)
        print("[BULL PATTERN SHEETS] updated rows=%d" % (len(values)-1),flush=True)
        LOG.info("Bull patterns Sheets updated rows=%d",len(values)-1)
    except Exception as exc:
        print("[BULL PATTERN SHEETS ERROR] %s: %s" % (type(exc).__name__,str(exc)[:130]),flush=True)
        LOG.warning("Bull patterns sheet update failed: %s",type(exc).__name__)


def step():
    now = int(time.time()*1000)
    print("[BULL PATTERN SCAN] begin at_ms=%d" % now,flush=True)
    db=database()
    try:
        followups(db,now)
        fresh=0
        for tf in FRAMES:
            try:
                data=legacy._api("/public/detectors/patterns",
                    {"exchange":"binance-futures","timeframe":tf,"quote":"USDT",
                     "type":"bull-flag,bull-pennant"})
                if data.get("kind")!="patterns" or data.get("timeframe")!=tf:
                    raise ValueError("INVALID_COINBEACON_BOARD")
                items=data.get("items",[])
                print("[BULL PATTERN API] tf=%s kind=%s total=%s returned=%s" % (tf,data.get("kind"),data.get("total"),len(items) if isinstance(items,list) else "invalid"),flush=True)
                if not isinstance(items,list):
                    raise ValueError("INVALID_PATTERN_ITEMS")
                if int(data.get("total",len(items)))>len(items):
                    LOG.warning("Bull patterns coverage truncated tf=%s %s/%s",
                                tf,len(items),data.get("total"))
                for raw in items:
                    got=extract(raw,tf)
                    if not got:
                        continue
                    sym,kind,status,lower,upper,position,item=got
                    ident=identity(raw,item,tf)
                    if not ident:
                        continue
                    found=db.execute("SELECT * FROM patterns WHERE id=?",(ident,)).fetchone()
                    if not found and fresh>=MAX_NEW_PER_SCAN:
                        continue
                    latest={"at_ms":now,"status":status,"position":position,
                            "lower":lower,"upper":upper,
                            "measuredTarget":item.get("measuredTarget"),
                            "flag":item.get("flag")}
                    if found:
                        obj=dict(found);obj["payload"]=json.loads(obj["payload"])
                        # Preserve every 15m observation, but avoid duplicate sampling after restart.
                        obs=obj["payload"].setdefault("observations",[])
                        if not obs or _closed(obs[-1]["at_ms"])!=_closed(now):
                            obs.append(latest)
                        obj["payload"]["latest"]=latest
                        obj["last_ms"]=now
                        obj["status"]=status
                        # Paper entry requires confirmed 15m reversal near support or
                        # confirmed close beyond upper boundary, never status alone.
                        if not obj["entry"] and lower and upper and upper>lower:
                            bars,_=legacy._bars(sym,_closed(now),2)
                            if len(bars)==2:
                                prev,cur=bars
                                close=cur[4]
                                bounce=(position is not None and position<=.20 and
                                        cur[3]<=lower*1.003 and close>prev[4] and close>lower)
                                breakout=(status in ("broken","retest") and
                                          close>upper and prev[4]<=upper)
                                if (bounce and upper > close * 1.012) or breakout:
                                    obj["entry"]=close
                                    obj["entry_ms"]=cur[0]+STEP_MS
                                    obj["target"]=close*1.01
                                    obj["payload"]["entry_reason"]="SUPPORT_REBOUND" if bounce else "BREAKOUT"
                        _store(db,obj)
                        continue
                    bars,complete=[],False
                    payload={"schema":"bull_patterns_learning_v1","id":ident,
                             "symbol":sym,"tf":tf,"kind":kind,
                             "first_ms":now,"original_pattern":item,
                             "latest":latest,"observations":[latest],
                             "historical_15m":bars,"historical_complete":complete,
                             "candles_8h":[],"candles_24h":[],"mode":"PAPER_ONLY"}
                    db.execute("INSERT OR IGNORE INTO patterns(id,symbol,tf,kind,first_ms,last_ms,status,payload)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (ident,sym,tf,kind,now,now,status,json.dumps(payload,separators=(",",":"))))
                    db.commit();fresh+=1
                    print("[BULL PATTERN NEW] %s %s %s saved=%s" % (sym,tf,kind,ident),flush=True)
                    legacy._telegram("🚩 YENİ SANAL FORMASYON\n%s %s %s\n"
                        "Durum: %s | Konum: %s\nAlt: %s | Üst: %s" %
                        (sym,tf,kind,status,position,lower,upper))
            except Exception as exc:
                print("[BULL PATTERN SCAN ERROR] tf=%s error=%s detail=%s" % (tf,type(exc).__name__,str(exc)[:160]),flush=True)
                LOG.warning("Bull patterns scan %s failed: %s",tf,type(exc).__name__)
        _sheet(db)
        print("[BULL PATTERN SCAN] cycle finished new=%s" % fresh,flush=True)
        LOG.info("Bull patterns cycle finished new=%s",fresh)
    finally:
        db.close()


def start():
    global _STARTED
    if _STARTED or os.getenv("BULL_FLAG_PAPER_ENABLED","0")!="1":
        return False
    if not (os.getenv("api2") or os.getenv("API2") or os.getenv("COINBEACON_API_KEY")):
        return False
    _STARTED=True
    def run():
        while True:
            try:
                step()
            except Exception:
                LOG.exception("Bull pattern observer error")
            time.sleep(max(30,900-(time.time()%900)))
    threading.Thread(target=run,name="bull-pattern-research",daemon=True).start()
    return True
