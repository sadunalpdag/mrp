# ==============================================================================
# 📘 EMA ULTRA v15.1 — SCALP ONLY + APPROVE + DualMode + RL + AutoReport
# ==============================================================================
# - SCALP ONLY (EMA7 slope reversal)
# - APPROVE_BARS bekleme sistemi
# - DualMode Trade Logic:
#     * ULTRA sinyaller: gerçek Binance (AutoTrade)
#     * PREMIUM / NORMAL sinyaller: sadece simülasyon
# - Precision Fix: stepSize / tickSize
# - 4h AutoReport + RL Log + SafeSave + Status raporu
# ==============================================================================

import os, json, time, math, requests, hmac, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE = os.path.join(DATA_DIR, "state.json")
PARAM_FILE = os.path.join(DATA_DIR, "params.json")
OPEN_POS_FILE = os.path.join(DATA_DIR, "open_positions.json")
CLOSED_TRADES_FILE = os.path.join(DATA_DIR, "closed_trades.json")
AI_RL_FILE = os.path.join(DATA_DIR, "ai_rl_log.json")
LOG_FILE = os.path.join(DATA_DIR, "log.txt")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI = "https://fapi.binance.com"

AUTO_TRADE = os.getenv("AUTO_TRADE","1") == "1"
SIMULATE   = os.getenv("SIMULATE","1") == "1"
SAVE_LOCK = threading.Lock()

def safe_load(path, default):
    try:
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return default

def safe_save(path, data):
    try:
        with SAVE_LOCK:
            tmp = path + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
    except Exception as e:
        print(f"[SAVE ERR] {e}", flush=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_local_iso(): return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(text):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":text},timeout=10)
    except: pass

def tg_send_file(path, caption):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(path): return
    try:
        with open(path,"rb") as f:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                          data={"chat_id":CHAT_ID,"caption":caption},
                          files={"document":(os.path.basename(path),f)},timeout=20)
    except: pass

def _signed_request(method, path, payload):
    query = "&".join([f"{k}={payload[k]}" for k in payload])
    sig = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + query + "&signature=" + sig
    r = requests.post(url, headers=headers, timeout=10) if method=="POST" else requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(symbol):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",params={"symbol":symbol},timeout=5).json()
        return float(r["price"])
    except: return None

def futures_get_klines(symbol, interval, limit):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/klines",params={"symbol":symbol,"interval":interval,"limit":limit},timeout=10).json()
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        if r and int(r[-1][6])>now_ms: r = r[:-1]
        return r
    except: return []

def get_symbol_filters(symbol):
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s = next((x for x in info["symbols"] if x["symbol"]==symbol),None)
        lot = next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef = next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        return {"stepSize":float(lot.get("stepSize","1")),"tickSize":float(pricef.get("tickSize","0.01"))}
    except: return {"stepSize":1.0,"tickSize":0.01}

def round_nearest(x, step):
    return round(round(x/step)*step, 12) if step>0 else x

def adjust_precision(symbol, value, mode="price"):
    f = get_symbol_filters(symbol)
    step = f["tickSize"] if mode=="price" else f["stepSize"]
    adj = round_nearest(value, step)
    return float(f"{adj:.12f}")

def calc_order_qty(symbol, entry_price, notional_usdt):
    raw = notional_usdt / max(entry_price,1e-12)
    return adjust_precision(symbol, raw, "qty")

def open_market_position(symbol, direction, qty):
    side = "BUY" if direction=="UP" else "SELL"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    payload = {"symbol":symbol,"side":side,"type":"MARKET","quantity":f"{qty}","positionSide":pos_side,"timestamp":now_ts_ms()}
    res = _signed_request("POST","/fapi/v1/order",payload)
    fills = res.get("avgPrice") or res.get("price")
    price = adjust_precision(symbol,float(fills) if fills else futures_get_price(symbol),"price")
    return {"symbol":symbol,"positionSide":pos_side,"qty":qty,"entry_price":price}

def futures_set_tp_sl(symbol, direction, qty, entry_price, tp_pct, sl_pct):
    pos_side = "LONG" if direction=="UP" else "SHORT"
    close_side = "SELL" if direction=="UP" else "BUY"
    tp = adjust_precision(symbol, entry_price*(1+tp_pct if direction=="UP" else 1-tp_pct))
    sl = adjust_precision(symbol, entry_price*(1-sl_pct if direction=="UP" else 1+sl_pct))
    for t,p in [("TAKE_PROFIT_MARKET",tp),("STOP_MARKET",sl)]:
        try:
            payload={"symbol":symbol,"side":close_side,"type":t,"stopPrice":f"{p:.12f}","quantity":f"{qty}",
                     "positionSide":pos_side,"workingType":"MARK_PRICE","reduceOnly":"true","timestamp":now_ts_ms()}
            _signed_request("POST","/fapi/v1/order",payload)
        except Exception as e:
            tg_send(f"⚠ TP/SL ERR {symbol} {e}"); log(f"[TP/SL ERR] {symbol} {e}")

def fetch_open_positions_real():
    out={"long":{}, "short":{}, "long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            sym=p["symbol"]; amt=float(p["positionAmt"])
            if amt>0: out["long"][sym]=amt
            elif amt<0: out["short"][sym]=abs(amt)
        out["long_count"]=len(out["long"]); out["short_count"]=len(out["short"])
    except Exception as e: log(f"[FETCH POS ERR] {e}")
    return out

PARAM_DEFAULT={"SCALP_TP_PCT":0.006,"SCALP_SL_PCT":0.20,"TRADE_SIZE_USDT":250.0,"MAX_BUY":30,"MAX_SELL":30,"APPROVE_BARS":1,"ULTRA_ONLY_TRADE":True}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
STATE=safe_load(STATE_FILE,{"bar_index":0,"last_report":0,"last_scalp_seen":{},"pending":[]})
OPEN_POS=safe_load(OPEN_POS_FILE,[])
CLOSED_LOG=safe_load(CLOSED_TRADES_FILE,[])

def ema(v,n):
    k=2/(n+1); e=[v[0]]
    for x in v[1:]: e.append(x*k+e[-1]*(1-k))
    return e

def rsi(v,p=14):
    if len(v)<p+1: return [50]*len(v)
    d=np.diff(v); g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:p]); al=np.mean(l[:p]); rs=ag/al if al>0 else 0
    out=[50]*p
    for i in range(p,len(d)):
        ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+l[i])/p; rs=ag/al if al>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(v)-len(out))+out

def atr_like(h,l,c,p=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else: tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:p])/p]
    for i in range(p,len(tr)): a.append((a[-1]*(p-1)+tr[i])/p)
    return [0]*(len(h)-len(a))+a

def calc_power(e7,e7p,e7p2,atr,price,rsi):
    diff=abs(e7-e7p)/(atr*0.6) if atr>0 else 0
    base=55+diff*20+((rsi-50)/50)*15+(atr/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    if p>=75:return"ULTRA","🟩"
    elif p>=68:return"PREMIUM","🟦"
    elif p>=60:return"NORMAL","🟨"
    return None,""

def build_scalp_signal(sym,kl,bar_i):
    closes=[float(k[4])for k in kl]; e7=ema(closes,7)
    if len(e7)<6:return None
    s_now=e7[-1]-e7[-4]; s_prev=e7[-2]-e7[-5]
    if s_prev<0 and s_now>0: d="UP"
    elif s_prev>0 and s_now<0: d="DOWN"
    else:return None
    h=[float(k[2])for k in kl]; lo=[float(k[3])for k in kl]
    atr=atr_like(h,lo,closes)[-1]; r=rsi(closes)[-1]
    pwr=calc_power(e7[-1],e7[-2],e7[-5],atr,closes[-1],r)
    tier,em=tier_from_power(pwr)
    if not tier:return None
    entry=futures_get_price(sym); 
    if not entry:return None
    tp=entry*(1+PARAM["SCALP_TP_PCT"]) if d=="UP" else entry*(1-PARAM["SCALP_TP_PCT"])
    sl=entry*(1-PARAM["SCALP_SL_PCT"]) if d=="UP" else entry*(1+PARAM["SCALP_SL_PCT"])
    return{"symbol":sym,"dir":d,"tier":tier,"emoji":em,"entry":entry,"tp":tp,"sl":sl,"power":pwr,"rsi":r,"born_bar":bar_i}

def run_parallel(symbols,bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=5)as ex:
        futs=[ex.submit(build_scalp_signal,s,futures_get_klines(s,"1h",200),bar_i)for s in symbols]
        for f in as_completed(futs):
            try:r=f.result()
            except:r=None
            if r:out.append(r)
    return out

def record_sim_open(sig,bar_i):
    OPEN_POS.append({"symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
                     "entry":sig["entry"],"tp":sig["tp"],"sl":sig["sl"],
                     "time_open":now_local_iso(),"bar_open":bar_i,"closed":False})
    safe_save(OPEN_POS_FILE,OPEN_POS)

def check_sim_closes():
    global OPEN_POS,CLOSED_LOG
    price_cache={}
    still=[]
    for p in OPEN_POS:
        if p.get("closed"):continue
        sym=p["symbol"]
        if sym not in price_cache:price_cache[sym]=futures_get_price(sym)
        last=price_cache[sym]; 
        if last is None:still.append(p);continue
        hit=None
        if p["dir"]=="UP":
            if last>=p["tp"]:hit="WIN"
            elif last<=p["sl"]:hit="LOSS"
        else:
            if last<=p["tp"]:hit="WIN"
            elif last>=p["sl"]:hit="LOSS"
        if hit:
            CLOSED_LOG.append({"symbol":sym,"dir":p["dir"],"tier":p["tier"],
                               "entry":p["entry"],"exit_price":last,"result":hit,"time_close":now_local_iso()})
            tg_send(f"📘 CLOSE {sym} {p['dir']} ({p['tier']}) [sim] -> {hit}")
        else: still.append(p)
    OPEN_POS=still; safe_save(OPEN_POS_FILE,OPEN_POS); safe_save(CLOSED_TRADES_FILE,CLOSED_LOG)

def approve_and_execute(bar_i):
    if not STATE["pending"]:return
    live=fetch_open_positions_real(); still=[]
    for p in STATE["pending"]:
        if bar_i<p["approve_at_bar"]:still.append(p);continue
        is_ultra=(p["tier"]=="ULTRA"and PARAM.get("ULTRA_ONLY_TRADE",True))
        sym=p["symbol"];d=p["dir"]
        if is_ultra:
            if d=="UP"and(live["long_count"]>=PARAM["MAX_BUY"]or sym in live["long"]):continue
            if d=="DOWN"and(live["short_count"]>=PARAM["MAX_SELL"]or sym in live["short"]):continue
        elif any(x for x in OPEN_POS if x["symbol"]==sym and x["dir"]==d and not x["closed"]):continue
        if is_ultra and AUTO_TRADE:
            qty=calc_order_qty(sym,p["entry"],PARAM["TRADE_SIZE_USDT"])
            try:op=open_market_position(sym,d,qty)
            except Exception as e:tg_send(f"❌ OPEN ERR {sym} {e}");continue
            futures_set_tp_sl(sym,d,qty,op["entry_price"],PARAM["SCALP_TP_PCT"],PARAM["SCALP_SL_PCT"])
            tg_send(f"✅ OPENED {sym} {d} ({p['tier']}) qty:{qty}")
            rl=safe_load(AI_RL_FILE,[]);rl.append({"time":now_local_iso(),"symbol":sym,"dir":d,"entry":op["entry_price"],"power":p["power"]});safe_save(AI_RL_FILE,rl)
        elif SIMULATE:record_sim_open(p,bar_i);tg_send(f"📒 SIM TRADE {sym} {d} ({p['tier']})")
    STATE["pending"]=still

def maybe_auto_report():
    now=time.time()
    if now-STATE.get("last_report",0)<14400:return
    for f in[AI_RL_FILE,CLOSED_TRADES_FILE]:
        tg_send_file(f,f"📊 AutoBackup {os.path.basename(f)}")
    tg_send("🕐 4 saatlik yedek gönderildi.");STATE["last_report"]=now;safe_save(STATE_FILE,STATE)

def main():
    tg_send("🚀 EMA ULTRA v15.1 başladı (DualMode + Precision + AutoReport)")
    info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo").json()
    syms=[s["symbol"]for s in info["symbols"]if s["quoteAsset"]=="USDT"and s["status"]=="TRADING"];syms.sort()
    while True:
        try:
            STATE["bar_index"]+=1;b=STATE["bar_index"]
            sigs=run_parallel(syms,b)
            for s in sigs:
                key=f"{s['symbol']}_{s['dir']}"
                if STATE["last_scalp_seen"].get(key)==b:continue
                STATE["last_scalp_seen"][key]=b
                tg_send(f"{s['emoji']} {s['tier']} {s['symbol']} {s['dir']} Pow:{s['power']:.1f}")
                s["approve_at_bar"]=b+PARAM["APPROVE_BARS"];STATE["pending"].append(s)
            approve_and_execute(b);check_sim_closes();maybe_auto_report();safe_save(STATE_FILE,STATE)
            time.sleep(30)
        except Exception as e:log(f"[LOOP ERR] {e}");time.sleep(10)

if __name__=="__main__":main()
