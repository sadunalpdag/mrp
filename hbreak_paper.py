"""Isolated CoinBeacon HBREAK 15m LONG paper monitor. No exchange order functions."""
import os,time,json,sqlite3,threading,hashlib,io,csv,zipfile,logging
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import requests

LOG=logging.getLogger("hbreak")
if not LOG.handlers:
    _stream=logging.StreamHandler()
    _stream.setFormatter(logging.Formatter("[HBREAK] %(levelname)s %(message)s"))
    LOG.addHandler(_stream)
LOG.setLevel(logging.INFO)
LOG.propagate=False
BASE=Path(os.getenv("DATA_DIR","/data"))
DB=BASE/"hbreak_paper.sqlite3"
SHEET="HBREAK_15M_LONG_TEST"
FIELDS="id opened_at coin archetype triggered_at sqs rvol taker_flow_pct liquidity_usd_24h trend_direction multi_tf_confirmed rsi14_15m atr14_15m atr_pct volume_15m_usdt entry tp status tp_at low_since_entry max_drawdown_pct checked_at".split()
URL="https://api.coinbeacon.io/public/setups"
PARAMS={"exchange":"binance-futures","timeframe":"15m","archetype":"breakout_watch,pattern_breakout,trendline_break,breakout_retest","bias":"bullish","minScore":"50"}
PARIS=ZoneInfo("Europe/Paris")
def utc(): return datetime.now(timezone.utc)
def stamp(): return utc().isoformat(timespec="seconds")
def initialize(db):
    db.execute("create table if not exists trades (id text primary key, data text not null)")
    db.execute("create table if not exists meta (k text primary key, v text not null)")
    db.commit()
def mget(db,k):
    r=db.execute("select v from meta where k=?",(k,)).fetchone()
    return r[0] if r else ""
def mset(db,k,v):
    db.execute("insert or replace into meta values (?,?)",(k,str(v)));db.commit()
def rows(db):
    return [json.loads(r[0]) for r in db.execute("select data from trades")]
def save(db,v):
    db.execute("insert or replace into trades values (?,?)",(v["id"],json.dumps(v)));db.commit()
def get_klines(http,symbol,tf,limit=100,start=None):
    p={"symbol":symbol,"interval":tf,"limit":limit}
    if start is not None:p["startTime"]=int(start)
    resp=http.get("https://fapi.binance.com/fapi/v1/klines",params=p,timeout=12);resp.raise_for_status()
    t=int(time.time()*1000)
    return [v for v in resp.json() if int(v[6])<t]
def rsi_atr(data):
    if len(data)<40:return None
    closes=[float(v[4]) for v in data]; highs=[float(v[2]) for v in data];lows=[float(v[3]) for v in data]
    gains=[max(closes[i]-closes[i-1],0) for i in range(1,len(closes))]
    losses=[max(closes[i-1]-closes[i],0) for i in range(1,len(closes))]
    g=sum(gains[:14])/14;l=sum(losses[:14])/14
    for x,y in zip(gains[14:],losses[14:]):g=(g*13+x)/14;l=(l*13+y)/14
    rsi=100 if l==0 else 100-100/(1+g/l)
    tr=[max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1])) for i in range(1,len(closes))]
    atr=sum(tr[:14])/14
    for x in tr[14:]:atr=(atr*13+x)/14
    return round(rsi,3),round(atr,10),round(100*atr/closes[-1],3),round(float(data[-1][7]),2)
def enroll(db,http,setups):
    existing={x["id"] for x in rows(db)}
    added=0
    for s in setups:
        try:
            a=s.get("archetype") or {}
            if a.get("key") not in ("pattern_breakout","trendline_break","breakout_retest"):continue
            if s.get("tf")!="15m" or s.get("bias")!="bullish" or float(s.get("sqs") or 0)<50:continue
            age=int(a.get("barsSinceTrigger") or 0)
            if age<0 or age>2:continue
            sym=s["symbol"];price=float(s.get("price") or 0)
            if price<=0 or not sym.endswith("USDT"):continue
            # computedAt is epoch milliseconds at the last CLOSED 4h candle.
            trigger=int(s["computedAt"])//900000-age
            key=hashlib.sha256(f"15m|{sym}|{a['key']}|{trigger}".encode()).hexdigest()[:24]
            if key in existing:continue
            vals=rsi_atr(get_klines(http,sym,"15m",100))
            if vals is None:continue
            part=s.get("participation") or {}
            obj=dict.fromkeys(FIELDS)
            obj.update(id=key,opened_at=stamp(),coin=sym,archetype=a["key"],
               triggered_at=datetime.fromtimestamp(trigger*900,timezone.utc).isoformat(),
               sqs=s.get("sqs"),rvol=part.get("rvol"),taker_flow_pct=part.get("takerFlowPct"),
               liquidity_usd_24h=part.get("liquidityUsd24h"),trend_direction=s.get("trendDirection"),
               multi_tf_confirmed=s.get("multiTfConfirmed"),rsi14_15m=vals[0],atr14_15m=vals[1],
               atr_pct=vals[2],volume_15m_usdt=vals[3],entry=price,tp=price*1.01,
               status="OPEN",low_since_entry=price,max_drawdown_pct=0,checked_at=stamp(),
               cursor=int(utc().timestamp()*1000))
            save(db,obj);existing.add(key);added+=1
        except Exception as e:LOG.warning("HBREAK candidate skip %s %s",s.get("symbol"),str(e)[:100])
    return added
def track(db,http):
    # Hard cap reduces Binance REST load. Round-robin on oldest checked.
    pending=sorted((x for x in rows(db) if x["status"]=="OPEN"),key=lambda v:v.get("checked_at",""))[:8]
    for t in pending:
        try:
            start=int(t.get("cursor") or int(datetime.fromisoformat(t["opened_at"]).timestamp()*1000))
            candles=get_klines(http,t["coin"],"5m",limit=1000,start=start)
            for c in candles:
                if int(c[6])<int(datetime.fromisoformat(t["opened_at"]).timestamp()*1000):continue
                t["low_since_entry"]=min(t["low_since_entry"],float(c[3]))
                if float(c[2])>=t["tp"]:
                    t["status"]="TP_1_PERCENT"
                    t["tp_at"]=datetime.fromtimestamp(int(c[6])/1000,timezone.utc).isoformat()
                    break
            if candles:t["cursor"]=int(candles[-1][0])+1
            t["max_drawdown_pct"]=round((t["low_since_entry"]/t["entry"]-1)*100,4)
            t["checked_at"]=stamp();save(db,t)
        except Exception as e:LOG.warning("HBREAK track %s %s",t["coin"],str(e)[:120])
def sheet_sync(db):
    import gspread
    from google.oauth2.service_account import Credentials
    cred=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON","").strip()
    sheet=os.getenv("GOOGLE_SHEETS_ID",os.getenv("GOOGLE_SHEET_ID","")).strip()
    if not cred or not sheet:raise ValueError("Missing existing Sheets credentials or id")
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
    if cred.startswith("{"):auth=Credentials.from_service_account_info(json.loads(cred),scopes=scopes)
    else:auth=Credentials.from_service_account_file(cred,scopes=scopes)
    sh=gspread.authorize(auth).open_by_key(sheet)
    try:ws=sh.worksheet(SHEET)
    except gspread.WorksheetNotFound:ws=sh.add_worksheet(title=SHEET,rows=1000,cols=len(FIELDS))
    vals=[FIELDS]+[[str(x.get(k) if x.get(k) is not None else "") for k in FIELDS] for x in sorted(rows(db),key=lambda r:r["opened_at"])]
    if len(vals)>ws.row_count:ws.add_rows(len(vals)-ws.row_count+100)
    ws.update(range_name="A1",values=vals)
    # Clear stale trailing rows after deletions (none expected in normal operation).
def archive(db):
    local=utc().astimezone(PARIS)
    if local.hour<21:return
    day=local.date().isoformat()
    if mget(db,"zip_day")==day:return
    data=rows(db);BASE.mkdir(exist_ok=True,parents=True)
    path=BASE/f"hbreak_15m_long_{day}.zip"
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=FIELDS,extrasaction="ignore");writer.writeheader();writer.writerows(data)
    stats={"date":day,"mode":"PAPER_ONLY","total":len(data),"open":sum(x["status"]=="OPEN" for x in data),"tp":sum(x["status"]=="TP_1_PERCENT" for x in data)}
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("trades.json",json.dumps(data,ensure_ascii=False))
        z.writestr("trades.csv",out.getvalue().encode("utf-8-sig"))
        z.writestr("summary.json",json.dumps(stats))
    from crash_report_storage import upload_private_report
    # Keep the archive in a distinct R2 prefix; no modification of existing Crash ZIPs.
    env=dict(os.environ);env["CRASH_REPORT_R2_PREFIX"]="hbreak"
    result=upload_private_report(str(path),time.time(),env=env)
    mset(db,"zip_day",day)
    LOG.info("HBREAK R2 archive uploaded key=%s",result["key"])
def step(db,http):
    bucket=str(int(time.time()//600))
    if mget(db,"bucket_15m")!=bucket:
        key=os.getenv("hbreak") or os.getenv("HBREAK") or os.getenv("COINBEACON_API_KEY")
        if not key:raise ValueError("Missing Render hbreak CoinBeacon key")
        resp=http.get(URL,params=PARAMS,headers={"X-API-Key":key},timeout=20)
        if resp.status_code==429:raise ValueError("CoinBeacon 429, skipping retry this cycle")
        resp.raise_for_status();result=resp.json()
        if not isinstance(result.get("setups"),list):raise ValueError("CoinBeacon response missing setups")
        # Mark the bucket after successful fetch, before costly downstream processing.
        mset(db,"bucket_15m",bucket)
        LOG.info("HBREAK fetched %s setup rows; new %s",len(result["setups"]),enroll(db,http,result["setups"]))
    track(db,http)
    try:sheet_sync(db)
    except Exception as e:LOG.warning("HBREAK sheet %s",str(e)[:180])
    try:archive(db)
    except Exception as e:LOG.warning("HBREAK archive %s",str(e)[:180])
def run():
    BASE.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(DB) as db,requests.Session() as http:
        initialize(db)
        while True:
            try:step(db,http)
            except Exception as e:LOG.warning("HBREAK cycle %s",str(e)[:180])
            time.sleep(max(15,600-int(time.time())%600))
def start():
    threading.Thread(target=run,name="hbreak-paper-only",daemon=True).start()
