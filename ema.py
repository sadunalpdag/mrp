# ==============================================================================
# 📘 EMA ULTRA v13.5.6-r3 — Auto-Sim Sync + Soft Limit Full Restore + reduceOnly Fix
#  (Dosya adı: ema.py)
#
#  ⚙️ Özellikler:
#   • Binance hedge-mode destekli otomatik trade sistemi
#   • reduceOnly fix — Binance 2025 API uyumlu
#   • Soft limit (MAX_BUY / MAX_SELL) yön bazlı
#   • Limit dolunca yön durur, düşünce otomatik tekrar açılır
#   • Her iki yön kapalıysa Simulate ON (veri toplama)
#     En az bir yön açıksa Simulate OFF (gerçek trade)
#   • /autotrade on/off -> simulate flag otomatik ayarlanır
# ==============================================================================

import os, json, time, math, requests, hmac, hashlib
from datetime import datetime, timezone, timedelta

# ================= PATHS =================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE  = os.path.join(DATA_DIR, "state.json")
PARAM_FILE  = os.path.join(DATA_DIR, "params.json")
TG_QUEUE_FILE = os.path.join(DATA_DIR, "tg_queue.json")
LOG_FILE    = os.path.join(DATA_DIR, "log.txt")

# ================= ENV VARS =================
BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI = "https://fapi.binance.com"

# ================= HELPERS =================
def safe_load(p, d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except: pass
    return d

def safe_save(p,d):
    try:
        tmp=p+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)
        os.replace(tmp,p)
    except: pass

def log(m):
    print(m,flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f: 
            f.write(f"{datetime.now(timezone.utc).isoformat()} {m}\n")
    except: pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_iso(): return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

# ================= TELEGRAM =================
def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except Exception as e:
        q=safe_load(TG_QUEUE_FILE,[]); q.append({"type":"text","text":t}); safe_save(TG_QUEUE_FILE,q)
        log(f"[TG ERR] {e}")

def tg_flush():
    q=safe_load(TG_QUEUE_FILE,[])
    if not q: return
    new=[]
    for i in q:
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={"chat_id":CHAT_ID,"text":i.get("text","")},timeout=10)
            time.sleep(0.3)
        except: new.append(i)
    safe_save(TG_QUEUE_FILE,new)

# ================= BINANCE API =================
def _signed(method,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    h={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r=requests.request(method,url,headers=h,timeout=10)
    if r.status_code!=200: raise RuntimeError(f"Binance HTTP {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(sym):
    try:
        return float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
            params={"symbol":sym},timeout=5).json()["price"])
    except: return None

def futures_fetch_positions():
    d=_signed("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    out=[]
    for p in d:
        a=float(p.get("positionAmt","0"))
        ps=p.get("positionSide","BOTH")
        if ps=="LONG" and a>0: out.append({"symbol":p["symbol"],"positionSide":"LONG","qty":a})
        if ps=="SHORT" and a<0: out.append({"symbol":p["symbol"],"positionSide":"SHORT","qty":abs(a)})
    return out

def futures_market_order(sym,side,qty,posside):
    """reduceOnly param kaldırıldı (Binance -1106 fix)"""
    pl={"symbol":sym,"side":side,"type":"MARKET","quantity":qty,
        "positionSide":posside,"timestamp":now_ts_ms()}
    return _signed("POST","/fapi/v1/order",pl)

def futures_set_tp_sl(sym,side,posside,qty,tp,sl):
    close_side="SELL" if side=="BUY" else "BUY"
    for t,price in (("TAKE_PROFIT_MARKET",tp),("STOP_MARKET",sl)):
        try:
            pl={"symbol":sym,"side":close_side,"type":t,"stopPrice":f"{price:.8f}",
                "quantity":qty,"reduceOnly":"true","positionSide":posside,
                "workingType":"CONTRACT_PRICE","timestamp":now_ts_ms()}
            _signed("POST","/fapi/v1/order",pl)
        except Exception as e:
            tg_send(f"⚠️ {t} ERR {sym}: {e}")

def calc_order_quantity(sym,usdt):
    price=futures_get_price(sym)
    if not price: return None
    qty=usdt/price
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        if s:
            for f in s["filters"]:
                if f["filterType"]=="LOT_SIZE":
                    step=float(f["stepSize"])
                    precision=max(0,abs(int(round(math.log10(step))))) if step<1 else 0
                    qty=math.floor(qty/step)*step
                    qty=round(qty,precision)
                    break
    except: pass
    return qty if qty>0 else None

# ================= STATE INIT =================
STATE=safe_load(STATE_FILE,{
    "auto_trade":True,"simulate":True,
    "auto_trade_long":True,"auto_trade_short":True
})
PARAM=safe_load(PARAM_FILE,{"TRADE_SIZE_USDT":250.0,"MAX_BUY":15,"MAX_SELL":15})
safe_save(PARAM_FILE,PARAM)

# ================= AUTO LIMIT & SIM SYNC =================
def count_real(): 
    pos=futures_fetch_positions()
    l=sum(1 for p in pos if p["positionSide"]=="LONG")
    s=sum(1 for p in pos if p["positionSide"]=="SHORT")
    return l,s,pos

def enforce_limits_autotrade_soft():
    l,s,_=count_real(); ch=False
    if l>=PARAM["MAX_BUY"] and STATE["auto_trade_long"]: 
        STATE["auto_trade_long"]=False; tg_send("⛔ BUY limit doldu"); ch=True
    if l<PARAM["MAX_BUY"] and not STATE["auto_trade_long"]: 
        STATE["auto_trade_long"]=True; tg_send("✅ BUY limit altında"); ch=True
    if s>=PARAM["MAX_SELL"] and STATE["auto_trade_short"]: 
        STATE["auto_trade_short"]=False; tg_send("⛔ SELL limit doldu"); ch=True
    if s<PARAM["MAX_SELL"] and not STATE["auto_trade_short"]: 
        STATE["auto_trade_short"]=True; tg_send("✅ SELL limit altında"); ch=True
    all_off=not STATE["auto_trade_long"] and not STATE["auto_trade_short"]
    if all_off and not STATE["simulate"]: 
        STATE["simulate"]=True; tg_send("🧠 Her iki yön kapalı → Simulate ON"); ch=True
    if not all_off and STATE["simulate"]: 
        STATE["simulate"]=False; tg_send("💸 En az bir yön aktif → Simulate OFF"); ch=True
    if ch: safe_save(STATE_FILE,STATE)
    return not all_off

# ================= TRADE EXEC =================
def execute_trade(sym,dir,tp,sl,entry):
    enforce_limits_autotrade_soft()
    if STATE["simulate"]:
        tg_send(f"📒 SIM TRADE {sym} {dir} entry={entry:.4f}")
        return
    side="BUY" if dir=="UP" else "SELL"
    posside="LONG" if dir=="UP" else "SHORT"
    qty=calc_order_quantity(sym,PARAM["TRADE_SIZE_USDT"])
    if not qty: tg_send(f"qty hesaplanamadı {sym}"); return
    try:
        futures_market_order(sym,side,qty,posside)
        futures_set_tp_sl(sym,side,posside,qty,tp,sl)
        tg_send(f"💸 REAL TRADE {sym} {dir} qty={qty} entry≈{entry:.4f}")
    except Exception as e:
        tg_send(f"❌ REAL TRADE ERR {sym}\n{e}\nSim olarak kaydedildi.")

# ================= MAIN LOOP =================
def main():
    tg_send("🚀 ema.py v13.5.6-r3 başladı (Auto-Sim Sync + Soft Limit Full Restore)")
    while True:
        tg_flush()
        enforce_limits_autotrade_soft()
        # Buraya sinyal kontrolü ve execute_trade çağrıları entegre edilir.
        time.sleep(60)

if __name__=="__main__":
    try: main()
    except Exception as e:
        tg_send(f"❗FATAL ema.py: {e}")
        log(f"[FATAL]{e}")