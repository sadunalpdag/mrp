import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE=os.path.join(DATA_DIR,"state.json")
PARAM_FILE=os.path.join(DATA_DIR,"params.json")
AI_SIGNALS_FILE=os.path.join(DATA_DIR,"ai_signals.json")
AI_ANALYSIS_FILE=os.path.join(DATA_DIR,"ai_analysis.json")
AI_RL_FILE=os.path.join(DATA_DIR,"ai_rl_log.json")
SIM_POS_FILE=os.path.join(DATA_DIR,"sim_positions.json")
SIM_CLOSED_FILE=os.path.join(DATA_DIR,"sim_closed.json")
LOG_FILE=os.path.join(DATA_DIR,"log.txt")

BOT_TOKEN=os.getenv("BOT_TOKEN")
CHAT_ID=os.getenv("CHAT_ID")
BINANCE_KEY=os.getenv("BINANCE_API_KEY")
BINANCE_SECRET=os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI="https://fapi.binance.com"

SAVE_LOCK=threading.Lock(); PRECISION_CACHE={}; TREND_LOCK={}; SIM_QUEUE=[]

def safe_load(p,d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:return json.load(f)
    except:pass
    return d
def safe_save(p,d):
    try:
        with SAVE_LOCK:
            tmp=p+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:print("[SAVE ERR]",e)
def log(m):print(m,flush=True);open(LOG_FILE,"a",encoding="utf-8").write(f"{datetime.now(timezone.utc).isoformat()} {m}\n")
def now_ts_ms():return int(datetime.now(timezone.utc).timestamp()*1000)
def now_local_iso():return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()
def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID:return
    try:requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except:pass

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    h={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r=requests.post(url,headers=h,timeout=10) if m=="POST" else requests.get(url,headers=h,timeout=10)
    if r.status_code!=200:raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE:return PRECISION_CACHE[sym]
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        PRECISION_CACHE[sym]={"stepSize":float(lot.get("stepSize","1")),"tickSize":float(pricef.get("tickSize","0.01"))}
    except:PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001}
    return PRECISION_CACHE[sym]

# ✅ v13.6 Precision Legacy TP/SL
def futures_set_tp_sl(sym,dir,qty,entry,tp_pct,sl_pct):
    try:
        f=get_symbol_filters(sym);tick=float(f["tickSize"])
        pos="LONG" if dir=="UP" else "SHORT";side="SELL" if dir=="UP" else "BUY"
        if dir=="UP":
            tp_raw=entry*(1+tp_pct);sl_raw=entry*(1-sl_pct)
            tp=math.floor(tp_raw/tick)*tick;sl=math.ceil(sl_raw/tick)*tick
            if sl>=entry:sl=entry-tick
        else:
            tp_raw=entry*(1-tp_pct);sl_raw=entry*(1+sl_pct)
            tp=math.ceil(tp_raw/tick)*tick;sl=math.floor(sl_raw/tick)*tick
            if sl<=entry:sl=entry+tick
        dec=0
        if "." in str(tick):dec=len(str(tick).split(".")[1].rstrip("0"))
        fmt=f"{{:.{dec}f}}"
        for t,p in [("TAKE_PROFIT_MARKET",tp),("STOP_MARKET",sl)]:
            pay={"symbol":sym,"side":side,"type":t,"stopPrice":fmt.format(p),
                 "quantity":f"{qty}","workingType":"MARK_PRICE","closePosition":"true",
                 "reduceOnly":"false","positionSide":pos,"timestamp":now_ts_ms()}
            _signed_request("POST","/fapi/v1/order",pay)
        msg=f"✅ TP/SL {sym} {dir} TP={fmt.format(tp)} SL={fmt.format(sl)}"
        tg_send(msg);log(msg)
    except Exception as e:
        tg_send(f"⚠️ TP/SL ERR {sym} {e}");log(f"[TP/SL ERR]{sym}{e}")
def ema(vals,n):k=2/(n+1);e=[vals[0]];[e.append(v*k+e[-1]*(1-k)) for v in vals[1:]];return e
def rsi(vals,period=14):
    if len(vals)<period+2:return[50]*len(vals)
    d=np.diff(vals);g=np.maximum(d,0);l=-np.minimum(d,0)
    ag=np.mean(g[:period]);al=np.mean(l[:period]);out=[50]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period;al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0;out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(h,l,c,p=14):
    tr=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) if i>0 else h[0]-l[0] for i in range(len(h))]
    a=[sum(tr[:p])/p]
    for i in range(p,len(tr)):a.append((a[-1]*(p-1)+tr[i])/p)
    return [0]*(len(h)-len(a))+a

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_v):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55+diff*20+((rsi_v-50)/50)*15+(atr_v/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    if p>=75:return"ULTRA","🟩"
    if p>=68:return"PREMIUM","🟦"
    if p>=60:return"NORMAL","🟨"
    return None,""

def heartbeat_and_status_check():
    global STATE
    now=time.time()
    if now-STATE.get("last_api_check",0)<600:return
    STATE["last_api_check"]=now;safe_save(STATE_FILE,STATE)
    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift=abs(now_ts_ms()-st)
        ping_ok=requests.get(BINANCE_FAPI+"/fapi/v1/ping",timeout=5).status_code==200
        key_ok=True
        try:_=_signed_request("GET","/fapi/v2/account",{"timestamp":now_ts_ms()})
        except:key_ok=False
        hb=f"✅ HEARTBEAT drift={int(drift)}ms ping={ping_ok} key={key_ok}" if all([ping_ok,key_ok,drift<1500]) else f"⚠️ HEARTBEAT ping={ping_ok} key={key_ok} drift={int(drift)}"
        tg_send(hb);log(hb)
    except Exception as e:
        tg_send(f"❌ HEARTBEAT {e}");log(f"[HBERR]{e}")
    # status
    live=fetch_open_positions_real() if 'fetch_open_positions_real' in globals() else {"long_count":0,"short_count":0}
    msg=f"📊 STATUS bar:{STATE.get('bar_index',0)} auto:{'✅' if STATE.get('auto_trade_active',1) else '🟥'} long:{live['long_count']} short:{live['short_count']}"
    tg_send(msg);log(msg)
STATE_DEFAULT={"bar_index":0,"last_report":0,"auto_trade_active":True,"last_api_check":0}
STATE=safe_load(STATE_FILE,STATE_DEFAULT)

def main():
    tg_send("🚀 EMA ULTRA v15.9.10 (StatusLegacy) başladı")
    log("[START] v15.9.10")
    while True:
        try:
            STATE["bar_index"]+=1
            safe_save(STATE_FILE,STATE)
            heartbeat_and_status_check()  # her 10 dk bir rapor
            time.sleep(30)
        except Exception as e:
            log(f"[LOOP ERR] {e}")
            time.sleep(10)

if __name__=="__main__":
    main()