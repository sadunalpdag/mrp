"""Read-only, cumulative TP and non-TP indicator archive for CB2/HBREAK.

Binance PUBLIC daily 15m ZIPs only; never Binance REST, never broker/orders.
Each result's technical indicators use CLOSED candles preceding the signal.
"""
import json
import math
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from paper_archive_15m import klines

PARIS=ZoneInfo("Europe/Paris")
ROOT=Path(os.getenv("DATA_DIR","/data"))
HEADERS=["Sistem","Tarih Paris","Strateji","Coin","Sinyal UTC",
         "Giris USDT","TP USDT","TP mum UTC","TP dk (ust sinir)",
         "Maks dusus %","RSI14","EMA20","EMA50","EMA200",
         "EMA20>EMA50","ATR14 %","ADX14","RVOL20","Hacim USDT",
         "MACD histogram","Bollinger band genisligi %",
         "1s degisim %","4s degisim %","24s degisim %",
         "15m mum sayisi","Kapsam","Intrabar belirsizlik","Kaynak ID"]
MAX_PER_CYCLE=0  # 0 = no fixed signal-count cap; use elapsed-time budget instead.
MAX_WORK_SECONDS=480  # Cap each background pass so paper monitoring is never monopolized.
LOOKBACK_BARS=480
WINDOW_DAYS=3


def _number(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError,OverflowError):
        return None


def _date(value):
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError,ValueError,OverflowError):
        return None


def _ema(data, period):
    if len(data)<period:
        return None
    current=sum(data[:period])/period
    alpha=2/(period+1)
    for value in data[period:]:
        current=(value-current)*alpha+current
    return current


def _rsi(close, period=14):
    if len(close)<period+2:
        return None
    gains=[max(b-a,0) for a,b in zip(close,close[1:])]
    losses=[max(a-b,0) for a,b in zip(close,close[1:])]
    up=sum(gains[:period])/period
    down=sum(losses[:period])/period
    for u,d in zip(gains[period:],losses[period:]):
        up=(up*(period-1)+u)/period
        down=(down*(period-1)+d)/period
    return 100 if down==0 else 100-100/(1+up/down)


def _atr_adx(rows, period=14):
    if len(rows)<2*period+1:
        return None,None
    high=[float(x[2]) for x in rows]
    low=[float(x[3]) for x in rows]
    close=[float(x[4]) for x in rows]
    tr=[]; plus=[]; minus=[]
    for i in range(1,len(rows)):
        up=high[i]-high[i-1]
        down=low[i-1]-low[i]
        plus.append(up if up>down and up>0 else 0)
        minus.append(down if down>up and down>0 else 0)
        tr.append(max(high[i]-low[i],abs(high[i]-close[i-1]),
                      abs(low[i]-close[i-1])))
    def smooth(vals):
        out=[]; a=sum(vals[:period])
        out.append(a)
        for v in vals[period:]:
            a=a-a/period+v
            out.append(a)
        return out
    trs,ps,ms=map(smooth,(tr,plus,minus))
    dx=[]
    for t,p,m in zip(trs,ps,ms):
        pos=100*p/t if t else 0
        neg=100*m/t if t else 0
        dx.append(100*abs(pos-neg)/(pos+neg) if pos+neg else 0)
    adx=sum(dx[:period])/period
    for x in dx[period:]:
        adx=(adx*(period-1)+x)/period
    return trs[-1]/period,adx


def indicators(candles, signal_ms):
    # No look-ahead; a bar must have closed STRICTLY BEFORE signal time.
    rows=[c for c in candles if int(c[6])<signal_ms]
    rows.sort(key=lambda x:int(x[0]))
    if len(rows)<LOOKBACK_BARS:
        return {"coverage":len(rows),"status":"PENDING_480_CLOSED_BARS"}
    rows=rows[-LOOKBACK_BARS:]
    # Yesterday's last candle must not be mistaken for a fresh signal-time
    # indicator when today's Binance daily ZIP has not been published.
    if signal_ms-int(rows[-1][6])>1800000:
        return {"coverage":len(rows),"status":"LATEST_SIGNAL_CANDLE_NOT_PUBLISHED",
                "last_closed_candle_utc":datetime.fromtimestamp(
                    int(rows[-1][6])/1000,timezone.utc).isoformat()}
    # Stop on any archive gap: never stitch unknown time into technical indicators.
    if any(int(b[0])-int(a[0])!=900000 for a,b in zip(rows,rows[1:])):
        return {"coverage":len(rows),"status":"ARCHIVE_GAP"}
    close=[float(c[4]) for c in rows]
    high=[float(c[2]) for c in rows]
    low=[float(c[3]) for c in rows]
    volume=[float(c[7]) for c in rows] # Binance quote asset volume, USDT
    if any(not math.isfinite(v) for v in close+high+low+volume) or min(close)<=0:
        return {"coverage":len(rows),"status":"INVALID_CANDLE"}
    atr,adx=_atr_adx(rows)
    ema20=_ema(close,20);ema50=_ema(close,50);ema200=_ema(close,200)
    # O(N) seeded EMA recurrence: identical to recomputing all prefixes,
    # but avoids a quadratic MACD calculation across thousands of signals.
    hist=None
    if len(close)>=34:
        a12=2/13
        a26=2/27
        ef=sum(close[:12])/12
        for price in close[12:26]:
            ef=(price-ef)*a12+ef
        es=sum(close[:26])/26
        macds=[ef-es]
        for price in close[26:]:
            ef=(price-ef)*a12+ef
            es=(price-es)*a26+es
            macds.append(ef-es)
        signal=_ema(macds,9)
        hist=macds[-1]-signal if signal is not None else None
    mean=sum(close[-20:])/20
    sigma=math.sqrt(sum((x-mean)**2 for x in close[-20:])/20)
    vmean=sum(volume[-21:-1])/20
    output={
        "status":"READY","coverage":len(rows),
        "last_closed_candle_utc":datetime.fromtimestamp(int(rows[-1][6])/1000,timezone.utc).isoformat(),
        "rsi14":_rsi(close),"ema20":ema20,"ema50":ema50,"ema200":ema200,
        "ema20_gt_ema50":ema20>ema50,
        "atr14":atr,"atr14_pct":100*atr/close[-1] if atr is not None else None,
        "adx14":adx,"rvol20":volume[-1]/vmean if vmean>0 else None,
        "quote_volume_15m_usdt":volume[-1],
        "macd_histogram":hist,
        "bollinger_width20_pct":400*sigma/mean if mean>0 else None,
        "return_1h_pct":100*(close[-1]/close[-5]-1),
        "return_4h_pct":100*(close[-1]/close[-17]-1),
        "return_24h_pct":100*(close[-1]/close[-97]-1),
    }
    return {k:(round(v,7) if isinstance(v,float) else v) for k,v in output.items()}


def candidates(system, records, now=None, days=WINDOW_DAYS):
    current=now or datetime.now(PARIS)
    cutoff=(current.astimezone(PARIS).date()-timedelta(days=days-1)
            if days and days>0 else None)
    items=[]
    for item in records:
        if system=="hbreak":
            raw_date=item.get("opened_at")
            tp=item.get("tp_at")
            entry=item.get("entry")
            entry_at=item.get("opened_at")
            details={"status":item.get("status"),"entry_price":entry,
                     "tp_at_candle_close":tp,
                     "tp_minutes_upper_bound":None,
                     "max_drawdown_pct_observed":item.get("max_drawdown_pct"),
                     "intrabar_order_uncertain":True}
            family=item.get("archetype")
            symbol=item.get("coin")
        else:
            # CB2 observational backtest, not the older 'legacy' paper trades.
            if not str(item.get("source") or "").startswith("coinbeacon-api2"):
                continue
            family=item.get("strategy") or item.get("family")
            # Include watch and incomplete observations as separate categories.
            # Never mark them as failed fills: some had no executable entry.
            details=(item.get("entry_scenarios") or {}).get("next_15m_open") or {}
            raw_date=item.get("observed_at")
            entry_at=raw_date
            entry=details.get("entry_price")
            tp=details.get("tp_at_candle_close")
            symbol=item.get("symbol")
        started=_date(raw_date)
        if not started or (cutoff and started.astimezone(PARIS).date()<cutoff):
            continue
        if started.astimezone(PARIS).date()>current.astimezone(PARIS).date():
            continue
        if not symbol:
            continue
        observed_entry=_number(entry)
        if observed_entry is None and system=="cb2":
            observed_entry=_number(item.get("reference_price"))
        status=details.get("status")
        if system=="cb2" and (str(family).endswith("_watch") or
                              item.get("status") in ("WATCH","WATCH_EXPIRED","UNCONFIRMED_OI")):
            outcome="WATCH_ONLY"
        elif status:
            outcome=status
        elif system=="hbreak":
            outcome=item.get("status") or "PENDING_ARCHIVE"
        else:
            outcome="PENDING_ARCHIVE"
        identifier=str(item.get("id") or "")
        if not identifier:
            continue
        scenario_map={name:case.get("status")
                      for name,case in (item.get("entry_scenarios") or {}).items()
                      if isinstance(case,dict)}
        items.append({"id":identifier,"system":system,
                      "strategy":family,"symbol":symbol,
                      "outcome_status":outcome,
                      "source_status":item.get("status"),
                      "source_data_status":item.get("data_status"),
                      "reference_price":_number(item.get("reference_price")),
                      "reference_status":item.get("reference_status"),
                      "source_id":item.get("source_id"),
                      "source_signal_at":item.get("source_signal_at") or item.get("triggered_at"),
                      "source_sqs":item.get("sqs"),"source_rvol":item.get("rvol"),
                      "source_archetype":item.get("archetype"),
                      "scenario_statuses":scenario_map,
                      "signal_at":entry_at,"entry_price":observed_entry,
                      "tp_at":tp,
                      "tp_price":(item.get("tp") if system=="hbreak" else
                                  observed_entry*1.01 if observed_entry is not None else None),
                      "result":details,
                      "date_paris":started.astimezone(PARIS).date().isoformat()})
    # TP comes first to fill the existing TP-only tab as quickly as possible.
    # OPEN / PENDING / WATCH remain in the same cumulative JSON.
    return sorted(items,key=lambda x:(x["outcome_status"]!="TP_1_PERCENT",
                                     x["signal_at"],x["id"]))


def _cache_path(system):
    return ROOT / ("paper_tp_indicators_"+system+".json")


def _load(system):
    try:
        data=json.loads(_cache_path(system).read_text(encoding="utf-8"))
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError,TypeError):
        return {}


def _save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(temp,path)


def _upload(system,payload,at):
    """Private R2 JSON. Failed uploads remain local, to retry next cycle."""
    from crash_report_storage import r2_configured,R2_ENV_NAMES
    if not r2_configured():
        return None
    import boto3
    account,bucket,key,secret=(os.environ[n].strip() for n in R2_ENV_NAMES)
    stamp=at.astimezone(PARIS).strftime("%Y/%m/%d")
    name=at.astimezone(PARIS).strftime("%Y-%m-%d")
    path=system+"/tp-analysis/"+stamp+"/"+system+"_3day_tp_indicators_"+name+".json"
    body=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    client=boto3.client("s3",endpoint_url="https://"+account+".r2.cloudflarestorage.com",
                        aws_access_key_id=key,aws_secret_access_key=secret,region_name="auto")
    client.put_object(Bucket=bucket,Key=path,Body=body,ContentType="application/json")
    if client.head_object(Bucket=bucket,Key=path)["ContentLength"]!=len(body):
        raise ValueError("R2_UPLOAD_NOT_VERIFIED")
    return path


def _values(item):
    i=item.get("indicators") or {}
    s=item["signal"]
    r=s.get("result") or {}
    values=[s["system"],s["date_paris"],s["strategy"],s["symbol"],
            s["signal_at"],s["entry_price"],s["tp_price"],s["tp_at"],
            r.get("tp_minutes_upper_bound"),
            r.get("max_drawdown_pct_observed"),i.get("rsi14"),
            i.get("ema20"),i.get("ema50"),i.get("ema200"),
            i.get("ema20_gt_ema50"),i.get("atr14_pct"),i.get("adx14"),
            i.get("rvol20"),i.get("quote_volume_15m_usdt"),
            i.get("macd_histogram"),i.get("bollinger_width20_pct"),
            i.get("return_1h_pct"),i.get("return_4h_pct"),i.get("return_24h_pct"),
            i.get("coverage"),i.get("status"),r.get("intrabar_order_uncertain"),
            s["id"]]
    return [v if v is not None else "" for v in values]


def _sheet(system,results,auth,book_id):
    """Dedicated TP-only tab; preserve all other monitoring sheets as-is."""
    import gspread
    from google.oauth2.service_account import Credentials
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
    creds=Credentials.from_service_account_info(json.loads(auth),scopes=scopes) if auth.startswith("{") else Credentials.from_service_account_file(auth,scopes=scopes)
    book=gspread.authorize(creds).open_by_key(book_id)
    title="HBREAK_TP_ANALIZ" if system=="hbreak" else "CB2_TP_ANALIZ"
    try:
        ws=book.worksheet(title)
    except gspread.WorksheetNotFound:
        ws=book.add_worksheet(title=title,rows=1500,cols=len(HEADERS))
    data=[HEADERS]+[_values(entry) for entry in results]
    if ws.row_count<len(data):
        ws.add_rows(len(data)-ws.row_count+10)
    if ws.col_count<len(HEADERS):
        ws.add_cols(len(HEADERS)-ws.col_count)
    ws.update(range_name="A1",values=data,value_input_option="RAW")
    # Do not clear somebody else's historic rows; only this tab is owned.
    if ws.row_count>len(data):
        ws.batch_clear(["A"+str(len(data)+1)+":AB"+str(ws.row_count)])
    return title


def run(system,records,http,book_id=None,now=None,logger=None):
    """Bounded archive enrichment. Call once per 15m paper cycle."""
    at=now or datetime.now(timezone.utc)
    selected=candidates(system,records,at)
    cache=_load(system)
    keep={x["id"] for x in selected}
    cache={k:v for k,v in cache.items() if k in keep}
    attempted=0
    for signal in selected:
        sid=signal["id"]
        old=cache.get(sid) or {}
        if old.get("indicators",{}).get("status")=="READY":
            continue
        if attempted>=MAX_PER_CYCLE:
            break
        if time.time()-float(old.get("last_attempt",0))<3600:
            continue
        observed=_date(signal["signal_at"])
        if observed is None:
            continue
        attempted+=1
        # Fetch 5+ days BEFORE the event. Published public ZIP only.
        start_ms=int((observed-timedelta(days=6)).timestamp()*1000)
        try:
            bars=klines(signal["symbol"],"15m",start=start_ms,
                        limit=800,http=http)
            enriched=indicators(bars,int(observed.timestamp()*1000))
        except Exception as error:
            enriched={"status":"ARCHIVE_READ_ERROR","reason":type(error).__name__}
        cache[sid]={"signal":signal,"indicators":enriched,
                    "last_attempt":time.time()}
    for signal in selected:
        if signal["id"] not in cache:
            cache[signal["id"]]={"signal":signal,"indicators":{"status":"PENDING_ENRICHMENT"}}
        else:
            cache[signal["id"]]["signal"]=signal
    _save(_cache_path(system),cache)
    enriched_rows=[cache[s["id"]] for s in selected]
    tp_only=[x for x in enriched_rows
             if x["signal"]["outcome_status"]=="TP_1_PERCENT"]
    complete=sum(x["indicators"].get("status")=="READY" for x in enriched_rows)
    by_date={}
    for item in enriched_rows:
        day=item["signal"]["date_paris"]
        stats=by_date.setdefault(day,{"signals":0,"tp_observed":0,
                                      "open_or_pending":0,"indicators_ready":0})
        stats["signals"]+=1
        stats["tp_observed"]+=item["signal"]["outcome_status"]=="TP_1_PERCENT"
        stats["open_or_pending"]+=item["signal"]["outcome_status"]!="TP_1_PERCENT"
        stats["indicators_ready"]+=item["indicators"].get("status")=="READY"
    payload={"schema_version":"paper_tp_indicators_v1","system":system,
             "generated_at_utc":at.astimezone(timezone.utc).isoformat(),
             "period_days":WINDOW_DAYS,"mode":"PAPER_ONLY",
             "source":"Binance futures public 15m daily ZIP",
             "no_binance_rest":True,"signal_time_no_lookahead":True,
             "tp_label":"15m OHLC hypothetical TP, not exchange fill",
             "warnings":["A TP on the entry candle is ambiguous about intrabar order.",
                         "Indicator coverage requires 480 contiguous PRE-signal candles.",
                         "Published daily archives may be delayed; missing values never guessed."],
             "totals":{"signal_records":len(selected),"tp_records":len(tp_only),
                       "comparison_records":len(selected)-len(tp_only),
                       "indicator_complete":complete},
             "daily":by_date,"records":enriched_rows}
    directory=ROOT/"paper_tp_analysis"
    directory.mkdir(parents=True,exist_ok=True)
    local=directory/(system+"_3day_latest.json")
    _save(local,payload)
    creds=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON","").strip()
    if creds and book_id:
        try:
            title=_sheet(system,tp_only,creds,book_id)
            if logger: logger("["+system.upper()+" TP ANALYSIS SHEETS] tab="+title+
                              " tp="+str(len(tp_only))+" ready="+str(complete))
        except Exception as error:
            if logger: logger("["+system.upper()+" TP ANALYSIS SHEETS ERR] "+type(error).__name__)
    marker=directory/(system+"_last_r2_at.json")
    previous=_load_upload(marker)
    if time.time()-previous>=3600:
        try:
            key=_upload(system,payload,at)
            if key:
                _save(marker,{"at":time.time()})
                if logger: logger("["+system.upper()+" TP ANALYSIS R2] "+key)
        except Exception as error:
            if logger: logger("["+system.upper()+" TP ANALYSIS R2 ERR] "+type(error).__name__)
    if logger:
        logger("["+system.upper()+" TP ANALYSIS] signals="+str(len(selected))+
               " tp="+str(len(tp_only))+
               " ready="+str(complete)+" checked="+str(attempted))
    return payload


def _load_upload(marker):
    try:
        return float(json.loads(marker.read_text(encoding="utf-8")).get("at",0))
    except (OSError,ValueError,TypeError):
        return 0

# Research must never block the signal-capture thread or live executor.
_WORKERS={"hbreak":threading.Lock(),"cb2":threading.Lock()}


def launch(system,records,book_id=None,logger=None):
    lock=_WORKERS[system]
    if not lock.acquire(blocking=False):
        return False
    snapshot=list(records)
    def worker():
        import requests
        try:
            with requests.Session() as session:
                run(system,snapshot,session,book_id=book_id,logger=logger)
        except Exception as error:
            if logger:
                logger("["+system.upper()+" TP ANALYSIS WORKER ERR] "+type(error).__name__)
        finally:
            lock.release()
    try:
        threading.Thread(target=worker,name=system+"-tp-research",
                         daemon=True).start()
    except Exception:
        lock.release()
        raise
    return True
