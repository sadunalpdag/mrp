# ==============================================================
# 📘 EMA ULTRA v15.9.39 — FULL REAL STRATEGY PACK (REAL-ONLY)
#  - Strategies: EARLY | E200S | UT-STC | A+FVG  (hepsi REAL)
#  - TP Safe v3: TAKE_PROFIT_MARKET (closePosition=true, no qty)
#  - SL yok, reduceOnly yok
#  - Precision-safe (tickSize/stepSize), Telegram heartbeat
# ==============================================================
import os, json, time, math, threading, requests, hmac, hashlib
from datetime import datetime, timezone, timedelta
import numpy as np

# ---------- Paths / Files ----------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE        = os.path.join(DATA_DIR,"state.json")
PARAM_FILE        = os.path.join(DATA_DIR,"params.json")
AI_SIGNALS_FILE   = os.path.join(DATA_DIR,"ai_signals.json")
AI_RL_FILE        = os.path.join(DATA_DIR,"ai_rl_log.json")
SIM_CLOSED_FILE   = os.path.join(DATA_DIR,"sim_closed.json")
LOG_FILE          = os.path.join(DATA_DIR,"log.txt")

# ---------- Env / API ----------
BOT_TOKEN      = os.getenv("BOT_TOKEN","")
CHAT_ID        = os.getenv("CHAT_ID","")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY","")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY","")
BINANCE_FAPI   = "https://fapi.binance.com"

# ---------- Globals ----------
SAVE_LOCK = threading.Lock()
PRECISION_CACHE = {}
LAST_HEARTBEAT  = 0
HB_INTERVAL_SEC = 600   # 10 dk heartbeat
SCAN_SLEEP_SEC  = 20

# ---------- Params / State ----------
PARAM_DEFAULT = {
    # trade & risk
    "TRADE_SIZE_USDT": 250.0,
    "TP_USD_LOW": 1.60,
    "TP_USD_HIGH": 2.00,
    "TP_USD_STEP_FINE": 0.01,
    "TP_USD_STEP_COARSE": 0.10,

    # limits
    "MAX_LONG": 30,
    "MAX_SHORT": 30,

    # indicators
    "EARLY_FAST_EMA": 3,
    "EARLY_SLOW_EMA": 7,
    "EARLY_ATR_SPIKE": 0.10,  # atr[-1] >= atr[-2]*(1+ratio)

    "E200_RSI_LONG": 55.0,
    "E200_RSI_SHORT": 45.0,

    # UT-STC
    "ST_ATR_PERIOD": 10,
    "ST_MULTIPLIER": 2.0,
    "STC_LEN": 80,
    "STC_FAST": 27,
    "STC_SLOW": 50,
    "STC_GREEN": 25,
    "STC_RED": 75,

    # A+FVG
    "RANGE_TF": "15m",
    "ENTRY_TF": "5m",

    # scan universe
    "QUOTE": "USDT"
}
STATE_DEFAULT = {"bar":0, "last_report":0}

def safe_load(p, d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return d

def safe_save(p, d):
    try:
        with SAVE_LOCK:
            tmp=p+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:
        print("[SAVE ERR]",e,flush=True)

PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
AI_SIGNALS = safe_load(AI_SIGNALS_FILE, [])
AI_RL      = safe_load(AI_RL_FILE, [])
SIM_CLOSED = safe_load(SIM_CLOSED_FILE, [])

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

def now_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_iso_local():
    # Türkiye UTC+3
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":t}, timeout=10)
    except: pass

# ---------- Binance REST ----------
def _signed_request(method, path, payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    hdr={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r = (requests.post(url, headers=hdr, timeout=10) if method=="POST"
         else requests.get(url, headers=hdr, timeout=10))
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE: return PRECISION_CACHE[sym]
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        PRECISION_CACHE[sym]={
            "stepSize": float(lot.get("stepSize","1")),
            "tickSize": float(pricef.get("tickSize","0.01")),
            "minPrice": float(pricef.get("minPrice","0.00000001")),
            "maxPrice": float(pricef.get("maxPrice","100000000"))
        }
    except Exception as e:
        log(f"[PREC WARN] {sym} {e}")
        PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001,"minPrice":1e-8,"maxPrice":9e8}
    return PRECISION_CACHE[sym]

def adj_step(v, step): 
    if step<=0: return v
    return round(round(v/step)*step, 12)

def qfmt_by_tick(tick):
    s=f"{tick:.12f}".rstrip("0")
    dec = 0 if "." not in s else len(s.split(".")[1])
    return f"{{:.{dec}f}}"

def futures_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except: return None

def futures_klines(sym, interval, limit):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                       params={"symbol":sym,"interval":interval,"limit":limit},timeout=10).json()
        if r and int(r[-1][6])>now_ms(): r=r[:-1]
        return r
    except: return []

# ---------- Indicators ----------
def ema(arr, n):
    if len(arr)==0: return []
    k=2/(n+1)
    out=[arr[0]]
    for v in arr[1:]:
        out.append(v*k + out[-1]*(1-k))
    return out

def sma(arr, n):
    if len(arr)<n: return [sum(arr)/len(arr)]*len(arr)
    out=[]
    s=sum(arr[:n]); out.extend([s/n]*(n-1))
    for i in range(n-1, len(arr)):
        if i==n-1: out.append(s/n)
        else:
            s += arr[i]-arr[i-n]
            out.append(s/n)
    return out

def rsi(closes, period=14):
    if len(closes)<period+2: return [50.0]*len(closes)
    d=np.diff(closes)
    g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:period]); al=np.mean(l[:period])
    out=[50.0]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period
        al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0
        out.append(100-100/(1+rs))
    return [50.0]*(len(closes)-len(out))+out

def true_range(h,l,c_prev):
    return max(h-l, abs(h-c_prev), abs(l-c_prev))

def atr(h, l, c, period=14):
    if len(c)==0: return []
    trs=[]
    for i in range(len(c)):
        if i==0: trs.append(h[i]-l[i])
        else: trs.append(true_range(h[i],l[i],c[i-1]))
    a=[sum(trs[:period])/period]
    for i in range(period, len(trs)):
        a.append((a[-1]*(period-1)+trs[i])/period)
    return [a[0]]*(len(trs)-len(a))+a

# --- Supertrend (basic) ---
def supertrend(h, l, c, atr_period=10, mult=2.0):
    _atr = atr(h,l,c,atr_period)
    if len(_atr)==0: return [None]*len(c)
    hl2 = [(h[i]+l[i])/2 for i in range(len(c))]
    ub = [hl2[i] + mult * _atr[i] for i in range(len(c))]
    lb = [hl2[i] - mult * _atr[i] for i in range(len(c))]
    st=[None]*len(c)
    final_ub=[0]*len(c); final_lb=[0]*len(c)
    for i in range(len(c)):
        if i==0:
            final_ub[i]=ub[i]; final_lb[i]=lb[i]; st[i]=1
        else:
            final_ub[i]=ub[i] if (ub[i]<final_ub[i-1] or c[i-1]>final_ub[i-1]) else final_ub[i-1]
            final_lb[i]=lb[i] if (lb[i]>final_lb[i-1] or c[i-1]<final_lb[i-1]) else final_lb[i-1]
            st[i] = 1 if c[i] > final_ub[i-1] else (-1 if c[i] < final_lb[i-1] else st[i-1])
            if st[i]==1 and final_lb[i]<final_lb[i-1]: final_lb[i]=final_lb[i-1]
            if st[i]==-1 and final_ub[i]>final_ub[i-1]: final_ub[i]=final_ub[i-1]
    return st  # +1 uptrend, -1 downtrend

# --- STC (approx): MACD(fast,slow) → %K on MACD via stochastic over length ---
def stc_like(closes, fast=27, slow=50, length=80):
    if len(closes)<slow+length+5: return [50.0]*len(closes)
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd = [ema_fast[i]-ema_slow[i] for i in range(len(closes))]
    stc=[50.0]*len(closes)
    for i in range(length, len(closes)):
        window = macd[i-length+1:i+1]
        lo=min(window); hi=max(window)
        val = 50.0 if hi==lo else 100*(macd[i]-lo)/(hi-lo)
        stc[i]=val
    return stc
# ---------- Utility ----------
def power_metric(e_now, e_prev, a, price, rsi_v):
    # görsel amaçlı, 0-100 (kullanıma hazır)
    base = 55
    diff = abs(e_now-e_prev)/(a*0.6) if (a and a>0) else 0
    base += diff*20 + ((rsi_v-50)/50)*15 + ((a/price)*200 if price>0 and a else 0)
    return max(0, min(100, base))

def ai_log_signal(sig):
    AI_SIGNALS.append(sig)
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)

# ---------- Universe ----------
def all_symbols_usdt():
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        syms=[s["symbol"] for s in info["symbols"] if s.get("quoteAsset")==PARAM["QUOTE"] and s.get("status")=="TRADING"]
        syms.sort()
        return syms
    except: return ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]

# ---------- TP Safe v3 ----------
def tp_price_from_usd(direction, entry, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd, 1e-12)
    return (entry*(1+tp_pct) if direction=="UP" else entry*(1-tp_pct)), tp_pct

def set_take_profit_market(sym, direction, entry_exec):
    """
    TAKE_PROFIT_MARKET + closePosition=true + positionSide
    - side: LONG kapat => SELL, SHORT kapat => BUY
    - stopPrice: tick’e yuvarla, min/max kontrol
    - quantity GÖNDERME! (aksi halde -1106)
    """
    f = get_symbol_filters(sym)
    tick = f["tickSize"]; fmt = qfmt_by_tick(tick)
    minp, maxp = f["minPrice"], f["maxPrice"]
    pos_side = ("LONG" if direction=="UP" else "SHORT")
    side     = ("SELL" if direction=="UP" else "BUY")
    trade_usd = PARAM["TRADE_SIZE_USDT"]

    # coarse scan → fine scan
    coarse = [round(x,2) for x in np.arange(PARAM["TP_USD_LOW"], PARAM["TP_USD_HIGH"]+1e-9, PARAM["TP_USD_STEP_COARSE"])]
    fine   = [round(x,2) for x in np.arange(PARAM["TP_USD_LOW"], PARAM["TP_USD_HIGH"]+1e-9, PARAM["TP_USD_STEP_FINE"])]

    def try_once(tp_usd):
        stop, tp_pct = tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd)
        if not (minp <= stop <= maxp): 
            log(f"[TP RANGE] {sym} skip ${tp_usd} stop={stop}")
            return False, None, None
        stop = adj_step(stop, tick)   # tick’e yuvarla
        payload = {
            "symbol": sym,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": fmt.format(stop),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "positionSide": pos_side,
            "timestamp": now_ms()
        }
        try:
            _signed_request("POST","/fapi/v1/order", payload)
            log(f"[TP OK] {sym} tp_usd={tp_usd} stop={fmt.format(stop)} side={side} ps={pos_side}")
            return True, tp_usd, tp_pct
        except Exception as e:
            log(f"[TP FAIL] {sym} tp=${tp_usd} err={e}")
            return False, None, None

    for v in coarse:
        ok,u,p = try_once(v)
        if ok: return True,u,p
    for v in fine:
        ok,u,p = try_once(v)
        if ok: return True,u,p
    log(f"[NO TP] {sym} 1.6–2.0$ aralığında uygun TP bulunamadı.")
    return False, None, None

# ---------- Open REAL Market ----------
def calc_order_qty(sym, entry, usd):
    step = get_symbol_filters(sym)["stepSize"]
    raw = usd/max(entry,1e-12)
    return adj_step(raw, step)

def open_market(sym, direction, entry_ref):
    side = "BUY" if direction=="UP" else "SELL"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    qty = calc_order_qty(sym, entry_ref, PARAM["TRADE_SIZE_USDT"])
    payload = {"symbol":sym, "side":side, "type":"MARKET", "quantity":f"{qty}",
               "positionSide":pos_side, "timestamp":now_ms()}
    res = _signed_request("POST","/fapi/v1/order", payload)
    fill = res.get("avgPrice") or res.get("price") or futures_price(sym)
    return float(fill), qty, pos_side

# ---------- Guards / Live Snapshot ----------
def live_pos_counts():
    out={"long":0,"short":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ms()})
        for p in acc:
            amt=float(p.get("positionAmt","0"))
            if amt>0: out["long"] += 1
            elif amt<0: out["short"] += 1
    except Exception as e:
        log(f"[POSRISK ERR] {e}")
    return out

def can_open(direction):
    live = live_pos_counts()
    if direction=="UP"   and live["long"] >= PARAM["MAX_LONG"]:  return False
    if direction=="DOWN" and live["short"]>= PARAM["MAX_SHORT"]: return False
    return True

# ---------- Strategies ----------
def build_EARLY(sym):
    # 1h bars
    kl = futures_klines(sym,"1h",200)
    if len(kl)<60: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    f=PARAM["EARLY_FAST_EMA"]; s=PARAM["EARLY_SLOW_EMA"]
    efast=ema(c,f); eslow=ema(c,s)
    # cross on confirmed bar
    up = (efast[-2]>eslow[-2] and efast[-3]<=eslow[-3])
    dn = (efast[-2]<eslow[-2] and efast[-3]>=eslow[-3])
    if not (up or dn): return None
    atrs = atr(h,l,c,14)
    if len(atrs)<2: return None
    if not (atrs[-1] >= atrs[-2]*(1+PARAM["EARLY_ATR_SPIKE"])): return None
    direction = "UP" if up else "DOWN"
    r = rsi(c)[-1]
    powv = power_metric(eslow[-1],eslow[-2],atrs[-1],c[-1],r)
    return {"tag":"EARLY","symbol":sym,"dir":direction,"entry_ref":c[-1],
            "rsi":r,"atr":atrs[-1],"power":powv}

def build_E200S(sym):
    # 1h bars
    kl = futures_klines(sym,"1h",200)
    if len(kl)<60: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    e20=ema(c,20); e200=ema(c,200)
    r=rsi(c)[-1]; a=atr(h,l,c,14)[-1]
    if e20[-1]>e200[-1] and r>PARAM["E200_RSI_LONG"]:
        return {"tag":"E200S","symbol":sym,"dir":"UP","entry_ref":c[-1],"rsi":r,"atr":a,"power":power_metric(e20[-1],e20[-2],a,c[-1],r)}
    if e20[-1]<e200[-1] and r<PARAM["E200_RSI_SHORT"]:
        return {"tag":"E200S","symbol":sym,"dir":"DOWN","entry_ref":c[-1],"rsi":r,"atr":a,"power":power_metric(e20[-1],e20[-2],a,c[-1],r)}
    return None

def build_UT_STC(sym):
    # 15m bars
    kl = futures_klines(sym,"15m",200)
    if len(kl)<60: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    st = supertrend(h,l,c,PARAM["ST_ATR_PERIOD"],PARAM["ST_MULTIPLIER"])
    s  = stc_like(c, PARAM["STC_FAST"], PARAM["STC_SLOW"], PARAM["STC_LEN"])
    if st[-1]==1 and s[-1]<PARAM["STC_GREEN"] and s[-1]>s[-2]:
        a=atr(h,l,c,14)[-1]; r=rsi(c)[-1]
        return {"tag":"UT-STC","symbol":sym,"dir":"UP","entry_ref":c[-1],"rsi":r,"atr":a,"stc":s[-1],
                "power":power_metric(c[-1],c[-2],a,c[-1],r)}
    if st[-1]==-1 and s[-1]>PARAM["STC_RED"] and s[-1]<s[-2]:
        a=atr(h,l,c,14)[-1]; r=rsi(c)[-1]
        return {"tag":"UT-STC","symbol":sym,"dir":"DOWN","entry_ref":c[-1],"rsi":r,"atr":a,"stc":s[-1],
                "power":power_metric(c[-1],c[-2],a,c[-1],r)}
    return None

def detect_fvg_5m(sym):
    # 5m bars
    kl=futures_klines(sym,"5m",200)
    if len(kl)<10: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    # 3-candle FVG (i-2, i-1, i) — bullish gap: l[i] > h[i-2]
    i=len(c)-1
    bull = l[i] > h[i-2]
    bear = h[i] < l[i-2]
    if not (bull or bear): return None
    direction = "UP" if bull else "DOWN"
    a=atr(h,l,c,14)[-1]; r=rsi(c)[-1]
    return {"tag":"A+FVG","symbol":sym,"dir":direction,"entry_ref":c[-1],
            "rsi":r,"atr":a,"power":power_metric(c[-1],c[-2],a,c[-1],r)}

def build_signals_for(sym):
    # Hepsini dener, ilk uygunları toplar
    res=[]
    for builder in (build_EARLY, build_E200S, build_UT_STC, detect_fvg_5m):
        try:
            s=builder(sym)
            if s: res.append(s)
        except Exception as e:
            log(f"[SIG ERR] {sym} {builder.__name__} {e}")
    return res

# ---------- Execute REAL trade ----------
def execute_real(sig):
    sym=sig["symbol"]; direction=sig["dir"]; tag=sig["tag"]
    if not can_open(direction): 
        log(f"[BLOCK] {sym} {direction} limits")
        return
    entry_ref = sig["entry_ref"]
    try:
        entry_exec, qty, ps = open_market(sym, direction, entry_ref)
        tp_ok, tp_usd, tp_pct = set_take_profit_market(sym, direction, entry_exec)

        # Telegram
        prefix=f"✅ REAL {tag}"
        if tp_ok:
            tg_send(f"{prefix} {sym} {direction}\nEntry:{entry_exec:.12f}\nTP:{tp_usd:.2f}$ ({(tp_pct or 0)*100:.3f}%)\n"
                    f"rsi:{sig.get('rsi'):.2f} atr:{sig.get('atr'):.5f} pow:{sig.get('power',0):.1f}\n{now_iso_local()}")
        else:
            tg_send(f"{prefix} {sym} {direction}\nEntry:{entry_exec:.12f}\nTP:YOK (scan fail)\n{now_iso_local()}")

        # RL log
        AI_RL.append({
            "time":now_iso_local(),"symbol":sym,"dir":direction,"tag":tag,
            "entry":entry_exec,"qty":qty,"tp_ok":tp_ok,"tp_usd_used":tp_usd,
            "rsi":sig.get("rsi"),"atr":sig.get("atr"),"power":sig.get("power")
        })
        safe_save(AI_RL_FILE, AI_RL)

    except Exception as e:
        log(f"[OPEN ERR] {sym} {e}")
# ---------- Heartbeat / Status ----------
def heartbeat():
    global LAST_HEARTBEAT
    now=time.time()
    if now - LAST_HEARTBEAT < HB_INTERVAL_SEC: return
    LAST_HEARTBEAT = now
    ok_ping=False; key_ok=True; drift=0
    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift = abs(now_ms()-st)
        ok_ping = requests.get(BINANCE_FAPI+"/fapi/v1/ping",timeout=5).status_code==200
        try: _signed_request("GET","/fapi/v2/account",{"timestamp":now_ms()})
        except: key_ok=False
    except Exception as e:
        log(f"[HB ERR] {e}")
    msg=(f"✅ HEARTBEAT drift={int(drift)}ms ping={ok_ping} key={key_ok}"
         if ok_ping and key_ok and drift<1500 else
         f"⚠️ HEARTBEAT ping={ok_ping} key={key_ok} drift={int(drift)}")
    tg_send(msg); log(msg)

def status_snapshot():
    live=live_pos_counts()
    msg=(f"📊 STATUS bar:{STATE.get('bar',0)} "
         f"long:{live['long']} short:{live['short']} "
         f"signals_total:{len(AI_SIGNALS)} rl:{len(AI_RL)}")
    tg_send(msg); log(msg)

# ---------- Main Loop ----------
def main():
    tg_send("🚀 EMA ULTRA v15.9.39 — FULL REAL PACK (TP Safe v3, SL OFF, reduceOnly OFF) aktif")
    log("[START] v15.9.39 REAL")
    try:
        symbols = all_symbols_usdt()
    except:
        symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]

    while True:
        try:
            STATE["bar"]=STATE.get("bar",0)+1
            # tarama
            for sym in symbols:
                sigs = build_signals_for(sym)
                for s in sigs:
                    s["time"]=now_iso_local()
                    ai_log_signal(s)
                    execute_real(s)

            # heartbeat
            heartbeat()

            # durum kaydı
            if STATE["bar"]%10==0: status_snapshot()

            safe_save(STATE_FILE, STATE)
            time.sleep(SCAN_SLEEP_SEC)

        except Exception as e:
            log(f"[LOOP ERR] {e}")
            time.sleep(5)

# ---------- Entrypoint ----------
if __name__=="__main__":
    main()