import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.28 — High Sensitivity + 6h TrendLock Cooldown + Smart TP + Decimal Fix
#  - stop=0 hatası düzeltildi (Decimal/tick doğrulama)
#  - High sensitivity: ANGLE_MIN↓, ATR_SPIKE_RATIO=0.10
#  - EARLY & REAL: Power 65–75, TradeSize 250 USDT
#  - TP only:
#       * Büyük fiyatlı semboller: 1.6→2.0 USD (0.1 → 0.01 tarama) → STOP_MARKET fallback
#       * Mikro fiyatlı semboller: %0.5 → %1.0 aralığında tarama (0.05% adım)
#       * minPrice / tickSize hizalama
#  - SL yok, reduceOnly yok
#  - TrendLock: Açılışta set; kapanışta hemen kalkmaz → 6 saat cooldown
#  - Telegram: yalnız gerçek fill (EARLY/REAL) + heartbeat
# ==============================================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE       = os.path.join(DATA_DIR,"state.json")
PARAM_FILE       = os.path.join(DATA_DIR,"params.json")
AI_SIGNALS_FILE  = os.path.join(DATA_DIR,"ai_signals.json")
AI_ANALYSIS_FILE = os.path.join(DATA_DIR,"ai_analysis.json")
AI_RL_FILE       = os.path.join(DATA_DIR,"ai_rl_log.json")
SIM_POS_FILE     = os.path.join(DATA_DIR,"sim_positions.json")
SIM_CLOSED_FILE  = os.path.join(DATA_DIR,"sim_closed.json")
LOG_FILE         = os.path.join(DATA_DIR,"log.txt")

BOT_TOKEN      = os.getenv("BOT_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
PRECISION_CACHE = {}
TREND_LOCK = {}
TREND_LOCK_TIME = {}
TRENDLOCK_EXPIRY_SEC = 6 * 3600
SIM_QUEUE = []

getcontext().prec = 28  # Decimal hassasiyet

# ========== SAFE I/O + UTILITIES ==========

def safe_load(p,d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
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
        print("[SAVE ERR]",e,flush=True)

def log(msg):
    print(msg,flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())
def now_local_iso(): return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except: pass

def tg_send_file(p,cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p): return
    try:
        with open(p,"rb") as f:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":cap},
                files={"document":(os.path.basename(p),f)},timeout=30)
    except: pass

# ========== BINANCE CORE ==========

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r = (requests.post(url,headers=headers,timeout=10) if m=="POST" else requests.get(url,headers=headers,timeout=10))
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE:
        return PRECISION_CACHE[sym]
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        PRECISION_CACHE[sym]={
            "stepSize":float(lot.get("stepSize","1")),
            "tickSize":float(pricef.get("tickSize","0.01")),
            "minPrice":float(pricef.get("minPrice","0.00000001")),
            "maxPrice":float(pricef.get("maxPrice","100000000"))
        }
    except Exception as e:
        log(f"[PREC WARN]{sym}{e}")
        PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001,"minPrice":0.00000001,"maxPrice":99999999}
    return PRECISION_CACHE[sym]

# ========== DECIMAL HELPERS (STOP=0 fix) ==========

def _decimals_from_tick(tick_str):
    try:
        d=Decimal(str(tick_str))
        return max(0,-d.as_tuple().exponent)
    except:
        s=str(tick_str)
        if "." in s: return len(s.split(".")[1])
        return 0

def round_to_tick(sym, price_float):
    f=get_symbol_filters(sym)
    t=Decimal(str(f["tickSize"]))
    p=Decimal(str(price_float))
    if t<=0: return float(p)
    q=(p/t).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    out=(q*t)
    return float(out)

def format_price_by_tick(sym, price_float):
    f=get_symbol_filters(sym)
    dec=_decimals_from_tick(str(f["tickSize"]))
    p_dec=Decimal(str(price_float)).quantize(Decimal(f"1e-{dec}"), rounding=ROUND_HALF_UP)
    if p_dec==Decimal("-0"): p_dec=Decimal("0")
    return f"{float(p_dec):.{dec}f}"
def adjust_precision(sym,v,kind="qty"):
    f=get_symbol_filters(sym)
    step=f["stepSize"] if kind=="qty" else f["tickSize"]
    if step<=0: return v
    return round(round(v/step)*step,12)

def calc_order_qty(sym,entry,usd):
    raw = usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except: return None

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",params={"symbol":sym,"interval":it,"limit":lim},timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms(): r=r[:-1]
        return r
    except: return []

# ================== INDICATORS ==================
def ema(vals,n):
    k=2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k+e[-1]*(1-k))
    return e

def rsi(vals,period=14):
    if len(vals)<period+2: return [50]*len(vals)
    d=np.diff(vals); g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:period]); al=np.mean(l[:period])
    out=[50]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0; out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else: tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)): a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_val):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55+diff*20+((rsi_val-50)/50)*15+(atr_v/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    if 65<=p<75: return "REAL","🟩"
    if p>=75: return "ULTRA","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""

# ==============================================================

PARAM_DEFAULT={"SCALP_TP_PCT":0.006,"SCALP_SL_PCT":0.20,"TRADE_SIZE_USDT":250.0,
               "MAX_BUY":30,"MAX_SELL":30,"ANGLE_MIN":0.00002,
               "FAST_EMA_PERIOD":3,"SLOW_EMA_PERIOD":7,"ATR_SPIKE_RATIO":0.10}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
if not isinstance(PARAM,dict): PARAM=PARAM_DEFAULT

STATE_DEFAULT={"bar_index":0,"last_report":0,"auto_trade_active":True,
               "last_api_check":0,"long_blocked":False,"short_blocked":False}
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

AI_SIGNALS=safe_load(AI_SIGNALS_FILE,[]); AI_ANALYSIS=safe_load(AI_ANALYSIS_FILE,[])
AI_RL=safe_load(AI_RL_FILE,[]); SIM_POSITIONS=safe_load(SIM_POS_FILE,[])
SIM_CLOSED=safe_load(SIM_CLOSED_FILE,[])

# ==============================================================
# 📈 SIGNALS + SIM + TP/SL + TRENDLOCK + REAL TRADE + MAIN LOOP
# ==============================================================

def build_scalp_signal(sym, kl, bar_i):
    if len(kl)<60: return None
    try: chg=float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                params={"symbol":sym},timeout=5).json()["priceChangePercent"])
    except: chg=0.0
    if abs(chg)>=10.0: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    e7=ema(closes,7)

    s_now=e7[-2]-e7[-5]; s_prev=e7[-3]-e7[-6]
    if abs(s_now-s_prev)<PARAM["ANGLE_MIN"]: return None
    if   s_prev<0 and s_now>0: direction="UP"
    elif s_prev>0 and s_now<0: direction="DOWN"
    else: return None

    atr_v=atr_like(highs,lows,closes)[-1]
    r_val=rsi(closes)[-1]
    pwr=calc_power(e7[-1],e7[-2],e7[-5],atr_v,closes[-1],r_val)
    tier,emoji=tier_from_power(pwr)
    if not tier: return None

    entry=closes[-1]
    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {"symbol":sym,"dir":direction,"tier":tier,"emoji":emoji,"entry":entry,
            "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atr_v,
            "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":False}

def build_early_signal(sym, kl, bar_i):
    if len(kl)<60: return None
    try: chg=float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                params={"symbol":sym},timeout=5).json()["priceChangePercent"])
    except: chg=0.0
    if abs(chg)>=10.0: return None

    closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
    ema_fast=ema(closes,PARAM["FAST_EMA_PERIOD"]); ema_slow=ema(closes,PARAM["SLOW_EMA_PERIOD"])
    up_cross=(ema_fast[-2]>ema_slow[-2]) and (ema_fast[-3]<=ema_slow[-3])
    dn_cross=(ema_fast[-2]<ema_slow[-2]) and (ema_fast[-3]>=ema_slow[-3])
    if not (up_cross or dn_cross): return None

    atrs=atr_like(highs,lows,closes)
    if len(atrs)<2 or not (atrs[-1]>=atrs[-2]*(1+PARAM["ATR_SPIKE_RATIO"])): return None

    direction="UP" if up_cross else "DOWN"; entry=closes[-1]; r_val=rsi(closes)[-1]
    pwr=calc_power(ema_slow[-1],ema_slow[-2],
                   ema_slow[-5] if len(ema_slow)>=6 else ema_slow[-2],
                   atrs[-1],entry,r_val)
    tier,emoji=tier_from_power(pwr); 
    if not tier: tier,emoji="EARLY","⚡️"
    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])
    return {"symbol":sym,"dir":direction,"tier":tier,"emoji":"⚡️","entry":entry,
            "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atrs[-1],
            "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":True}

def run_parallel(symbols,bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(lambda s: build_scalp_signal(s,futures_get_klines(s,"1h",200),bar_i) or
                                     build_early_signal(s,futures_get_klines(s,"1h",200),bar_i), sym) for sym in symbols]
        for f in as_completed(futs):
            try: sig=f.result()
            except: sig=None
            if sig: out.append(sig)
    return out

# =============================================================
# TRENDLOCK HANDLING
# =============================================================
def _set_trend_lock(sym, direction):
    TREND_LOCK[sym]=direction; TREND_LOCK_TIME[sym]=now_ts_s()
    log(f"[TRENDLOCK SET] {sym} {direction}")

def _unlock_trend_for(sym, delay_unlock=False):
    if delay_unlock:
        TREND_LOCK_TIME[sym]=now_ts_s()
        log(f"[TRENDLOCK DELAY CLEAR] {sym} (6h cooldown başlatıldı)")
        return
    TREND_LOCK.pop(sym,None); TREND_LOCK_TIME.pop(sym,None)
    log(f"[TRENDLOCK CLEAR] {sym}")

def _cleanup_trend_lock_expired():
    now_s=now_ts_s()
    expired=[sym for sym,t in TREND_LOCK_TIME.items() if now_s-t>=TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        _unlock_trend_for(sym)
        log(f"[TRENDLOCK TIMEOUT] {sym} (6h cooldown bitti)")

# =============================================================
# SMART TP ONLY
# =============================================================
def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd,1e-12)
    return (entry_exec*(1+tp_pct) if direction=="UP" else entry_exec*(1-tp_pct)), tp_pct

def futures_set_tp_only(sym, direction, qty, entry_exec, tp_low_usd=1.6, tp_high_usd=2.0):
    try:
        f=get_symbol_filters(sym)
        tick=f["tickSize"]; minp=f["minPrice"]; maxp=f["maxPrice"]
        pos_side="LONG" if direction=="UP" else "SHORT"; side="SELL" if direction=="UP" else "BUY"
        trade_usd=PARAM.get("TRADE_SIZE_USDT",250.0)
        usd_based = entry_exec>0.2  # düşük fiyatlı semboller fallback
        ok=False; tp_usd_used=None; tp_pct_used=None

        def try_once(tp_price_candidate, order_type, tp_usd_used=None, tp_pct_used=None):
            price=round_to_tick(sym,tp_price_candidate)
            if price<minp: price=round_to_tick(sym,minp)
            if price>maxp: price=round_to_tick(sym,maxp)
            stop_str=format_price_by_tick(sym,price)
            if float(stop_str)<=0:
                price=round_to_tick(sym,max(minp,1e-12))
                stop_str=format_price_by_tick(sym,price)
                if float(stop_str)<=0: 
                    log(f"[TP GUARD] {sym} stop hesap 0 oldu; minp jump başarısız")
                    return False,None,None
            payload={"symbol":sym,"side":side,"type":order_type,"stopPrice":stop_str,
                     "quantity":f"{qty}","workingType":"MARK_PRICE","closePosition":"true",
                     "positionSide":pos_side,"timestamp":now_ts_ms()}
            try:
                _signed_request("POST","/fapi/v1/order",payload)
                log(f"[TP OK] {sym} {order_type} stop={stop_str} qty={qty}")
                return True,tp_usd_used,tp_pct_used
            except Exception as e:
                log(f"[TP FAIL] {sym} {order_type} stop={stop_str} err={e}")
                return False,None,None

        if usd_based:
            for tp_usd in [round(x,1) for x in np.arange(tp_low_usd,tp_high_usd+0.001,0.1)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,tp_usd_used,tp_pct_used=try_once(tp_price,"TAKE_PROFIT_MARKET",tp_usd,tp_pct)
                if ok: return True,tp_usd_used,tp_pct_used
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd,tp_high_usd+0.0001,0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,tp_usd_used,tp_pct_used=try_once(tp_price,"TAKE_PROFIT_MARKET",tp_usd,tp_pct)
                if ok: return True,tp_usd_used,tp_pct_used
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd,tp_high_usd+0.0001,0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,tp_usd_used,tp_pct_used=try_once(tp_price,"STOP_MARKET",tp_usd,tp_pct)
                if ok: return True,tp_usd_used,tp_pct_used
        else:
            for tp_pct in np.arange(0.005,0.0105,0.0005):
                tp_price = entry_exec*(1+tp_pct if direction=="UP" else 1-tp_pct)
                ok,_,_=try_once(tp_price,"TAKE_PROFIT_MARKET",None,tp_pct)
                if ok: return True,None,tp_pct
        log(f"[NO TP] {sym} uygun TP bulunamadı.")
        return False,None,None
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False,None,None

# =============================================================
# REAL TRADE
# =============================================================
def open_market_position(sym, direction, qty):
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()})
    fill=res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    return {"symbol":sym,"dir":direction,"qty":qty,"entry":float(fill),"pos_side":pos_side}

def _duplicate_or_locked(sym,direction):
    if TREND_LOCK.get(sym)==direction:
        log(f"[TRENDLOCK HIT] {sym} {direction}"); return True
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except: acc=[]
    if direction=="UP":
        if any(float(p["positionAmt"])>0 and p["symbol"]==sym for p in acc): return True
    else:
        if any(float(p["positionAmt"])<0 and p["symbol"]==sym for p in acc): return True
    return False

def _can_direction(direction):
    if not STATE.get("auto_trade_active",True): return False
    if direction=="UP" and STATE.get("long_blocked",False): return False
    if direction=="DOWN" and STATE.get("short_blocked",False): return False
    return True

def execute_real_trade(sig):
    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig["power"]; is_early=bool(sig.get("early",False))
    early_ok=is_early and 65<=pwr<75; real_ok=(not is_early) and sig.get("tier")=="REAL" and 65<=pwr<75
    if not (early_ok or real_ok): return
    if not _can_direction(direction): return
    if _duplicate_or_locked(sym,direction): return
    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0: log(f"[QTY ERR]{sym}"); return
    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened.get("entry") or futures_get_price(sym)
        if not entry_exec or entry_exec<=0: log(f"[OPEN FAIL]{sym}"); return
        tp_ok,tp_usd_used,tp_pct_used=futures_set_tp_only(sym,direction,qty,entry_exec)
        _set_trend_lock(sym,direction)
        prefix=("⚡️ EARLY" if is_early else "✅ REAL")
        if tp_ok:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\nPower:{pwr:.2f}\nEntry:{entry_exec}\n"
                    f"TP hedefi:{tp_usd_used or tp_pct_used:.4f}\ntime:{now_local_iso()}")
        else:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\nPower:{pwr:.2f}\nEntry:{entry_exec}\nTP: YOK\n"
                    f"time:{now_local_iso()}")
        AI_RL.append({"time":now_local_iso(),"symbol":sym,"dir":direction,"entry":entry_exec,
                      "tp_usd_used":tp_usd_used,"tp_pct_used":tp_pct_used,"tp_ok":tp_ok,
                      "power":pwr,"born_bar":sig.get("born_bar"),"early":is_early})
        safe_save(AI_RL_FILE,AI_RL)
    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")

# =============================================================
# MAIN LOOP
# =============================================================
def main():
    tg_send("🚀 EMA ULTRA v15.9.28 aktif (Smart TP, Decimal Fix, 6h TrendLock Cooldown)")
    log("[START] EMA ULTRA v15.9.28 FULL")
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}"); symbols=[]
    symbols.sort()

    while True:
        try:
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]
            sigs=run_parallel(symbols,bar_i)
            for sig in sigs:
                execute_real_trade(sig)
            _cleanup_trend_lock_expired()
            safe_save(STATE_FILE,STATE)
            time.sleep(30)
        except Exception as e:
            log(f"[LOOP ERR]{e}"); time.sleep(10)

if __name__=="__main__":
    main()