import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.56 — Hedge Mode Full Fix
#  - Mean Reversion sistemi parallel ve hedge uyumlu (positionSide eklendi)
#  - reduceOnly tamamen kaldırıldı
#  - Tüm stratejiler (EARLY / MACD / UTSTC / FVG / PULLBACK) aktif
# ==============================================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE  = os.path.join(DATA_DIR,"state.json")
LOG_FILE    = os.path.join(DATA_DIR,"log.txt")

BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
getcontext().prec = 28

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

def safe_load(p,d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return d

def safe_save(p,d):
    try:
        with SAVE_LOCK:
            tmp=p+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:
        log(f"[SAVE ERR]{e}")

def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def ema(vals,n):
    k=2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k+e[-1]*(1-k))
    return e

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else: tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)): a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except: pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                       params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except: return None

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                       params={"symbol":sym,"interval":it,"limit":lim},
                       timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms(): r=r[:-1]
        return r
    except: return []

def auto_init_symbols():
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}"); symbols=[]
    symbols.sort()
    return symbols
# ===================== Mean Reversion System =====================

MEAN_REV_FILE = os.path.join(DATA_DIR,"mean_reversion_positions.json")
MEAN_REV_POS  = safe_load(MEAN_REV_FILE,[])

def detect_mean_reversion_signal(sym):
    kl=futures_get_klines(sym,"1h",120)
    if len(kl)<60: return None
    closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
    e99=ema(closes,99); last=closes[-1]; dist=(last-e99[-1])/e99[-1]*100
    if dist>5: direction="DOWN"
    elif dist<-5: direction="UP"
    else: return None
    return {"symbol":sym,"dir":direction,"entry":last,"distance":dist,"ema99":e99[-1],"time":now_local_iso()}

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r=(requests.post(url,headers=headers,timeout=10) if m=="POST" else requests.get(url,headers=headers,timeout=10))
    if r.status_code!=200: raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        return {"stepSize":float(lot.get("stepSize","1")),"tickSize":float(pricef.get("tickSize","0.01"))}
    except: return {"stepSize":0.0001,"tickSize":0.0001}

def _round_to_step(v, step): return round(round(v/step)*step,12) if step>0 else v
def calc_order_qty(sym, entry_price, usd=250.0):
    f=get_symbol_filters(sym); raw=usd/max(entry_price,1e-12)
    return _round_to_step(raw, f["stepSize"])

MEAN_REV_USD_SIZE,MEAN_REV_MAX_BUY,MEAN_REV_MAX_SELL=250.0,3,3
MEAN_REV_EXIT_TRIG,MEAN_REV_EXIT_MAX,MEAN_REV_CONFIRM,MEAN_REV_INTERVAL=8.0,30.0,2,120

def _count_live_positions():
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[MEAN-REV POSRISK ERR]{e}"); return 0,0
    long_cnt=sum(1 for p in acc if float(p.get("positionAmt",0))>0)
    short_cnt=sum(1 for p in acc if float(p.get("positionAmt",0))<0)
    return long_cnt,short_cnt

def _mr_save(): safe_save(MEAN_REV_FILE,MEAN_REV_POS)

def mean_reversion_distance_pct(sym):
    kl=futures_get_klines(sym,"1h",120)
    if not kl: return 0.0,None,None,None
    highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]; closes=[float(k[4]) for k in kl]
    c_now=closes[-1]; ma99=ema(closes,99)[-1]
    stv=(np.mean(highs[-10:])+np.mean(lows[-10:]))/2
    d_ma=abs(c_now-ma99)/max(ma99,1e-9)*100; d_st=abs(c_now-stv)/max(stv,1e-9)*100
    return max(d_ma,d_st),c_now,ma99,stv
def open_mean_reversion(sym,direction):
    dist,price,_,_=mean_reversion_distance_pct(sym)
    if price is None: return False
    long_cnt,short_cnt=_count_live_positions()
    if direction=="UP" and long_cnt>=MEAN_REV_MAX_BUY: return False
    if direction=="DOWN" and short_cnt>=MEAN_REV_MAX_SELL: return False
    qty=calc_order_qty(sym,price,MEAN_REV_USD_SIZE)
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    try:
        res=_signed_request("POST","/fapi/v1/order",{
            "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
            "positionSide":pos_side,"timestamp":now_ts_ms()})
        entry=float(res.get("avgPrice") or res.get("price") or price)
        MEAN_REV_POS.append({"symbol":sym,"dir":direction,"qty":qty,"entry":entry,
                             "open_time":now_local_iso(),"open_dist_pct":dist,"confirm":0})
        _mr_save()
        tg_send(f"📘 Mean-Reversion — No TP\n{sym} {direction} qty:{qty}\nEntry:{entry:.12f}\nDist:{dist:.2f}%")
        log(f"[MEAN-REV OPEN]{sym}{direction}entry={entry}dist={dist:.2f}%")
        return True
    except Exception as e:
        tg_send(f"❌ MEAN-REV OPEN ERR {sym} {direction}\n{e}")
        log(f"[MEAN-REV OPEN ERR]{sym}{e}")
        return False

def close_mean_reversion(sym,direction,reason):
    side="SELL" if direction=="UP" else "BUY"
    pos_side="LONG" if direction=="UP" else "SHORT"
    try:
        pos=next((p for p in MEAN_REV_POS if p["symbol"]==sym and p["dir"]==direction),None)
        if not pos: return
        qty=pos["qty"]
        _signed_request("POST","/fapi/v1/order",{
            "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
            "positionSide":pos_side,"timestamp":now_ts_ms()})
        tg_send(f"⚠️ {sym} Mean-Reversion Exit — {reason}")
        log(f"[MEAN-REV CLOSE]{sym}{direction}{reason}")
    except Exception as e:
        tg_send(f"❌ MEAN-REV CLOSE ERR {sym}\n{e}")
        log(f"[MEAN-REV CLOSE ERR]{sym}{e}")
    MEAN_REV_POS[:]=[p for p in MEAN_REV_POS if not(p["symbol"]==sym and p["dir"]==direction)]
    _mr_save()

def mean_reversion_watcher():
    while True:
        try:
            if not MEAN_REV_POS: time.sleep(MEAN_REV_INTERVAL); continue
            for pos in list(MEAN_REV_POS):
                sym,dir=pos["symbol"],pos["dir"]
                dist,_,_,_=mean_reversion_distance_pct(sym)
                if dist>=MEAN_REV_EXIT_MAX:
                    close_mean_reversion(sym,dir,f"fiyat ortalamadan uzaklaştı (%{dist:.2f}) [HARD]"); continue
                if dist>=MEAN_REV_EXIT_TRIG:
                    pos["confirm"]=pos.get("confirm",0)+1
                else: pos["confirm"]=0
                if pos["confirm"]>=MEAN_REV_CONFIRM:
                    close_mean_reversion(sym,dir,f"fiyat ortalamadan uzaklaştı (%{dist:.2f})")
            _mr_save()
        except Exception as e: log(f"[MEAN-REV WATCH ERR]{e}")
        time.sleep(MEAN_REV_INTERVAL)

def mean_reversion_loop(symbols):
    log("[MEAN-REV] Loop başlatıldı.")
    while True:
        try:
            long_cnt,short_cnt=_count_live_positions()
            for sym in symbols:
                sig=detect_mean_reversion_signal(sym)
                if not sig: continue
                direction=sig["dir"]
                if direction=="UP" and long_cnt<MEAN_REV_MAX_BUY:
                    if open_mean_reversion(sym,direction): long_cnt+=1
                elif direction=="DOWN" and short_cnt<MEAN_REV_MAX_SELL:
                    if open_mean_reversion(sym,direction): short_cnt+=1
            time.sleep(300)
        except Exception as e:
            log(f"[MEAN-REV LOOP ERR]{e}")
            time.sleep(60)

def main():
    tg_send("🚀 EMA ULTRA v15.9.56 aktif — Hedge Mode Full Fix")
    log("[START] EMA ULTRA v15.9.56 FULL")
    symbols=auto_init_symbols()
    threading.Thread(target=mean_reversion_loop,args=(symbols,),daemon=True).start()
    threading.Thread(target=mean_reversion_watcher,daemon=True).start()
    while True:
        try:
            check_telegram_commands()
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]
            sigs=run_parallel(symbols,bar_i)
            for sig in sigs:
                ai_log_signal(sig); queue_sim_variants(sig)
                update_directional_limits(); execute_real_trade(sig)
            process_sim_queue_and_open_due(); process_sim_closes()
            auto_report_if_due(); heartbeat_and_status_check({})
            _cleanup_trend_lock_expired(); safe_save(STATE_FILE,STATE)
            time.sleep(30)
        except Exception as e:
            log(f"[MAIN LOOP ERR]{e}")
            time.sleep(10)

if __name__=="__main__": main()