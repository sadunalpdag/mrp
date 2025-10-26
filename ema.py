# ==============================================================================
# 📘 EMA ULTRA v15.6 — Real-Only Instant-Scalp
# ==============================================================================
#  - Sadece ULTRA işlemler açar (PREMIUM/NORMAL atlanır)
#  - Simülasyon, pending ve duplicate sim guard kaldırıldı
#  - 1 bar (yaklaşık 30 sn) sonra sinyal direkt olarak işleme döner
#  - Dynamic AutoTrade: limit dolunca durur, düşünce yeniden açılır
#  - Angle + %10 Volatility filter aktif
# ==============================================================================

import os, json, time, requests, hmac, hashlib, threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE          = os.path.join(DATA_DIR, "state.json")
PARAM_FILE          = os.path.join(DATA_DIR, "params.json")
AI_SIGNALS_FILE     = os.path.join(DATA_DIR, "ai_signals.json")
AI_ANALYSIS_FILE    = os.path.join(DATA_DIR, "ai_analysis.json")
AI_RL_FILE          = os.path.join(DATA_DIR, "ai_rl_log.json")
LOG_FILE            = os.path.join(DATA_DIR, "log.txt")

BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()

# ---------- helpers ----------
def safe_load(p,d): 
    try:
        if os.path.exists(p): 
            with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except: pass
    return d
def safe_save(p,d):
    try:
        with SAVE_LOCK:
            tmp=p+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
    except: pass
def log(m): 
    print(m,flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {m}\n")
    except: pass
def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_local_iso(): return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try: requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except: pass
def tg_send_file(p,c):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p): return
    try:
        with open(p,"rb") as f:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":c},
                files={"document":(os.path.basename(p),f)},timeout=20)
    except: pass

# ---------- binance ----------
def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    h={"X-MBX-APIKEY":BINANCE_KEY}
    u=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r=requests.post(u,headers=h,timeout=10) if m=="POST" else requests.get(u,headers=h,timeout=10)
    if r.status_code!=200: raise RuntimeError(r.text)
    return r.json()
def futures_get_price(s):
    try:return float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",params={"symbol":s},timeout=5).json()["price"])
    except:return None
def futures_24h_change(s):
    try:return float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",params={"symbol":s},timeout=5).json()["priceChangePercent"])
    except:return 0.0
def futures_get_klines(s,interval,limit):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
            params={"symbol":s,"interval":interval,"limit":limit},timeout=10).json()
        now_ms=int(datetime.now(timezone.utc).timestamp()*1000)
        if r and int(r[-1][6])>now_ms: r=r[:-1]
        return r
    except:return []
def get_symbol_filters(s):
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        x=next((i for i in info["symbols"] if i["symbol"]==s),None)
        lot=next((f for f in x["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in x["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        return {"stepSize":float(lot.get("stepSize","1")),"tickSize":float(pricef.get("tickSize","0.01"))}
    except:return {"stepSize":1.0,"tickSize":0.01}
def round_nearest(x,step): return round(round(x/step)*step,8)
def adjust_precision(s,v,mode="price"):
    f=get_symbol_filters(s);st=f["tickSize"] if mode=="price" else f["stepSize"]
    v=max(round_nearest(v,st),st);return float(f"{v:.8f}")
def calc_order_qty(s,price,notional):
    raw=notional/max(price,1e-12)
    return adjust_precision(s,raw,"qty")
def open_market_position(s,dir,qty):
    side="BUY" if dir=="UP" else "SELL"
    pos="LONG" if dir=="UP" else "SHORT"
    payload={"symbol":s,"side":side,"type":"MARKET","quantity":f"{qty}","positionSide":pos,"timestamp":now_ts_ms()}
    res=_signed_request("POST","/fapi/v1/order",payload)
    avg=res.get("avgPrice") or res.get("price") or None
    p=float(avg) if avg else futures_get_price(s)
    return {"symbol":s,"qty":qty,"entry":p,"side":pos}
def futures_set_tp_sl(s,dir,qty,entry,tp,sl):
    pos="LONG" if dir=="UP" else "SHORT";close="SELL" if dir=="UP" else "BUY"
    tp_p=entry*(1+tp) if dir=="UP" else entry*(1-tp)
    sl_p=entry*(1-sl) if dir=="UP" else entry*(1+sl)
    tp_s=adjust_precision(s,tp_p,"price");sl_s=adjust_precision(s,sl_p,"price")
    for t,p in [("TAKE_PROFIT_MARKET",tp_s),("STOP_MARKET",sl_s)]:
        payload={"symbol":s,"side":close,"type":t,"stopPrice":f"{p:.8f}",
                 "quantity":f"{qty}","positionSide":pos,"workingType":"MARK_PRICE",
                 "reduceOnly":"true","timestamp":now_ts_ms()}
        try:_signed_request("POST","/fapi/v1/order",payload)
        except Exception as e: log(f"[TP/SL ERR]{s}{e}")
def fetch_open_positions_real():
    r={"long":{}, "short":{}, "long_count":0, "short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            sym=p["symbol"];amt=float(p["positionAmt"])
            if amt>0:r["long"][sym]=amt
            elif amt<0:r["short"][sym]=abs(amt)
        r["long_count"]=len(r["long"]);r["short_count"]=len(r["short"])
    except:pass
    return r

# ---------- params ----------
PARAM_DEFAULT={"SCALP_TP_PCT":0.006,"SCALP_SL_PCT":0.20,"TRADE_SIZE_USDT":250.0,
               "MAX_BUY":30,"MAX_SELL":30,"ANGLE_MIN":0.0001}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
STATE=safe_load(STATE_FILE,{"bar_index":0,"last_report":0,"auto_trade_active":True})
AI_SIGNALS=safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS=safe_load(AI_ANALYSIS_FILE,[])
AI_RL=safe_load(AI_RL_FILE,[])

# ---------- indicators ----------
def ema(v,n):
    k=2/(n+1);e=[v[0]]
    for x in v[1:]:e.append(x*k+e[-1]*(1-k))
    return e
def rsi(v,n=14):
    if len(v)<n+1:return [50]*len(v)
    d=np.diff(v);g=np.maximum(d,0);l=-np.minimum(d,0)
    ag,al=np.mean(g[:n]),np.mean(l[:n])
    out=[50]*n
    for i in range(n,len(d)):
        ag=(ag*(n-1)+g[i])/n;al=(al*(n-1)+l[i])/n
        rs=ag/al if al>0 else 0;out.append(100-100/(1+rs))
    return [50]*(len(v)-len(out))+out
def atr_like(h,l,c,n=14):
    tr=[]
    for i in range(len(h)):
        if i==0:tr.append(h[i]-l[i])
        else:tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:n])/n]
    for i in range(n,len(tr)):a.append((a[-1]*(n-1)+tr[i])/n)
    return [0]*(len(h)-len(a))+a
def calc_power(e7,e7p,e7p2,atr,p,r):
    diff=abs(e7-e7p)/(atr*0.6) if atr>0 else 0
    base=55+diff*20+((r-50)/50)*15+(atr/p)*200
    return min(100,max(0,base))
def tier_from_power(p):
    if p>=75:return"ULTRA","🟩"
    elif p>=68:return"PREMIUM","🟦"
    elif p>=60:return"NORMAL","🟨"
    return None,""

# ---------- signals ----------
def build_scalp_signal(sym,kl,bar):
    c=[float(k[4]) for k in kl];chg=futures_24h_change(sym)
    if abs(chg)>=10:return None
    e7=ema(c,7);s_now=e7[-1]-e7[-4];s_prev=e7[-2]-e7[-5]
    d="UP" if s_prev<0 and s_now>0 else "DOWN" if s_prev>0 and s_now<0 else None
    if not d:return None
    if abs(s_now-s_prev)<PARAM["ANGLE_MIN"]:return None
    h=[float(k[2]) for k in kl];l=[float(k[3]) for k in kl]
    atr_v=atr_like(h,l,c)[-1];r=rsi(c)[-1]
    pwr=calc_power(e7[-1],e7[-2],e7[-5],atr_v,c[-1],r)
    tier,emo=tier_from_power(pwr)
    if tier!="ULTRA":return None
    price=futures_get_price(sym); 
    if not price:return None
    tp=adjust_precision(sym,price*(1+(PARAM["SCALP_TP_PCT"] if d=="UP" else -PARAM["SCALP_TP_PCT"])),"price")
    sl=adjust_precision(sym,price*(1- (PARAM["SCALP_SL_PCT"] if d=="UP" else -PARAM["SCALP_SL_PCT"])),"price")
    return{"symbol":sym,"dir":d,"tier":tier,"emoji":emo,"entry":price,"tp":tp,"sl":sl,
           "power":pwr,"rsi":r,"atr":atr_v,"chg24h":chg,"born_bar":bar}

def scan_symbol(sym,bar):
    kl=futures_get_klines(sym,"1h",200)
    return build_scalp_signal(sym,kl,bar) if len(kl)>=60 else None
def run_parallel(symbols,bar):
    out=[]
    with ThreadPoolExecutor(max_workers=5)as ex:
        futs=[ex.submit(scan_symbol,s,bar)for s in symbols]
        for f in as_completed(futs):
            try:r=f.result()
            except:r=None
            if r:out.append(r)
    return out

# ---------- auto trade logic ----------
def approve_and_trade(bar):
    live=fetch_open_positions_real()
    if STATE.get("auto_trade_active",True):
        if live["long_count"]>=PARAM["MAX_BUY"] or live["short_count"]>=PARAM["MAX_SELL"]:
            STATE["auto_trade_active"]=False
            tg_send(f"🚫 AutoTrade durduruldu — limit aşıldı (long:{live['long_count']} short:{live['short_count']})")
    else:
        if live["long_count"]<PARAM["MAX_BUY"] and live["short_count"]<PARAM["MAX_SELL"]:
            STATE["auto_trade_active"]=True
            tg_send(f"✅ AutoTrade yeniden aktif (long:{live['long_count']} short:{live['short_count']})")

def trade_ultra(sig):
    s,d=sig["symbol"],sig["dir"]
    if not STATE.get("auto_trade_active",True):return
    live=fetch_open_positions_real()
    if (d=="UP" and s in live["long"]) or (d=="DOWN" and s in live["short"]):return
    qty=calc_order_qty(s,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    try:
        opened=open_market_position(s,d,qty)
        futures_set_tp_sl(s,d,qty,opened["entry"],PARAM["SCALP_TP_PCT"],PARAM["SCALP_SL_PCT"])
        tg_send(f"✅ REAL {s} {d} qty:{qty} entry:{opened['entry']:.6f}")
        AI_RL.append({"time":now_local_iso(),"symbol":s,"dir":d,"entry":opened["entry"],"power":sig["power"]})
        safe_save(AI_RL_FILE,AI_RL)
    except Exception as e:
        tg_send(f"❌ OPEN ERR {s} {e}");log(f"[OPEN ERR]{s}{e}")

# ---------- report ----------
def auto_report():
    now=time.time()
    if now-STATE.get("last_report",0)<14400:return
    for p in[AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE]:
        tg_send_file(p,f"📊 AutoBackup {os.path.basename(p)}")
    tg_send("🕐 4h backup done")
    STATE["last_report"]=now;safe_save(STATE_FILE,STATE)

# ---------- main ----------
def main():
    tg_send("🚀 EMA ULTRA v15.6 başladı (Real-Only Instant-Scalp)")
    info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo").json()
    syms=[s["symbol"] for s in info["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"];syms.sort()
    while True:
        try:
            STATE["bar_index"]+=1;bar=STATE["bar_index"]
            sigs=run_parallel(syms,bar)
            for s in sigs:
                tg_send(f"{s['emoji']} {s['symbol']} {s['dir']} Pow:{s['power']:.1f}")
                AI_SIGNALS.append({"time":now_local_iso(),"symbol":s["symbol"],"dir":s["dir"],"power":s["power"]})
                safe_save(AI_SIGN
safe_save(AI_SIGNALS_FILE, AI_SIGNALS)

                # 1 bar (yaklaşık 30 sn) sonra doğrudan işlem aç
                time.sleep(30)
                trade_ultra(s)

            # her bar sonunda trade durumu ve rapor kontrolü
            approve_and_trade(bar)
            auto_report()
            safe_save(STATE_FILE, STATE)

            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR] {e}")
            time.sleep(10)

# run
if __name__ == "__main__":
    main()