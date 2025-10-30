import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.31 — PEMA (Slope Confirmed) + EARLY Mode
#  - PEMA: Price–EMA7 Cross + EMA7 slope yön onayı (📗 BUY / 📕 SELL)
#  - EARLY: ATR_SPIKE_RATIO = 0.03
#  - Power band (65–75) => yalnız bu bantta gerçek trade
#  - Smart TP: USD 1.6–2.0 + mikro fiyatlarda % fallback + stop=0 guard
#  - 6h TrendLock cooldown (kapanışta başlar) + 6h auto-timeout
#  - Heartbeat(10 dk), Auto-Backup(4 saat), SIM approve(30/60/90/120 dk)
#  - Telegram: /status /report /set KEY VALUE /export
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
TREND_LOCK = {}          # { "SYMBOL": "UP"/"DOWN" }
TREND_LOCK_TIME = {}     # { "SYMBOL": last_set_ts }
TRENDLOCK_EXPIRY_SEC = 6 * 3600
SIM_QUEUE = []

getcontext().prec = 28  # Decimal hassasiyet

# ===================== SAFE I/O & LOG =====================

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

# ===================== TIME & TG HELPERS =====================

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())
def now_local_iso():
    # Türkiye (UTC+3)
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":t},
            timeout=10
        )
    except: pass

def tg_send_file(p, cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p): return
    try:
        with open(p,"rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":cap},
                files={"document":(os.path.basename(p),f)},
                timeout=30
            )
    except: pass

# ===================== BINANCE CORE =====================

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

# ===================== DECIMAL/TICK HELPERS =====================

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
# ===================== INDICATORS =====================

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
        rs=ag/al if al>0 else 0
        out.append(100-100/(1+rs))
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

# ===================== PRICE / KLINES HELPERS =====================

def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                       params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                       params={"symbol":sym,"interval":it,"limit":lim},
                       timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms():
            r=r[:-1]
        return r
    except:
        return []

# ===================== SIGNAL BUILDERS =====================

def build_pema_signal(sym, kl, bar_i):
    """
    Price–EMA7 Cross + EMA7 Slope Confirmed:
      - BUY: close[-2]<ema7[-2] & close[-1]>ema7[-1] & slope>0  → 📗 PEMA BUY
      - SELL: close[-2]>ema7[-2] & close[-1]<ema7[-1] & slope<0 → 📕 PEMA SELL
      - |24h chg| < 10
      - Power/RSI/ATR hesap, tier fallback "PEMA"
    """
    if len(kl)<60: return None
    try:
        chg=float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                               params={"symbol":sym},timeout=5).json()["priceChangePercent"])
    except: chg=0.0
    if abs(chg)>=10.0: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    e7=ema(closes,7)

    c_prev,c_now=closes[-2],closes[-1]
    e_prev,e_now=e7[-2],e7[-1]
    slope=e_now-e_prev

    direction=None; tag=""
    if c_prev<e_prev and c_now>e_now and slope>0:
        direction="UP"; tag="📗 PEMA BUY"
    elif c_prev>e_prev and c_now<e_now and slope<0:
        direction="DOWN"; tag="📕 PEMA SELL"
    else:
        return None

    atr_v=atr_like(highs,lows,closes)[-1]
    r_val=rsi(closes)[-1]
    pwr=calc_power(e_now,e_prev,e7[-5] if len(e7)>=6 else e_prev,atr_v,c_now,r_val)
    tier,emoji=tier_from_power(pwr)
    if not tier: tier,emoji="PEMA","🟪"

    entry=c_now
    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":emoji,"entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atr_v,
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":False,
        "kind":"PEMA","tag":tag
    }

def build_scalp_signal(sym, kl, bar_i):
    if len(kl)<60: return None
    try:
        chg=float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                               params={"symbol":sym},timeout=5).json()["priceChangePercent"])
    except: chg=0.0
    if abs(chg)>=10.0: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    e7=ema(closes,7)

    s_now  = e7[-2]-e7[-5]
    s_prev = e7[-3]-e7[-6]
    if abs(s_now-s_prev) < PARAM["ANGLE_MIN"]:
        return None

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

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":emoji,"entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atr_v,
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":False,
        "kind":"SCALP","tag":"🟩 REAL SCALP"
    }

def build_early_signal(sym, kl, bar_i):
    if len(kl)<60: return None
    try:
        chg=float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                               params={"symbol":sym},timeout=5).json()["priceChangePercent"])
    except: chg=0.0
    if abs(chg)>=10.0: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    fper=PARAM.get("FAST_EMA_PERIOD",3)
    sper=PARAM.get("SLOW_EMA_PERIOD",7)
    ema_fast=ema(closes,fper)
    ema_slow=ema(closes,sper)

    up_cross = (ema_fast[-2] > ema_slow[-2]) and (ema_fast[-3] <= ema_slow[-3])
    dn_cross = (ema_fast[-2] < ema_slow[-2]) and (ema_fast[-3] >= ema_slow[-3])
    if not (up_cross or dn_cross): return None

    atrs=atr_like(highs,lows,closes)
    if len(atrs)<2: return None
    if not (atrs[-1] >= atrs[-2]*(1.0 + PARAM.get("ATR_SPIKE_RATIO",0.03))):
        return None

    direction="UP" if up_cross else "DOWN"
    entry=closes[-1]
    r_val=rsi(closes)[-1]
    pwr=calc_power(
        ema_slow[-1], ema_slow[-2],
        ema_slow[-5] if len(ema_slow)>=6 else ema_slow[-2],
        atrs[-1], entry, r_val
    )
    tier,emoji=tier_from_power(pwr)
    if not tier: tier,emoji="EARLY","⚡️"

    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":"⚡️","entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":True,
        "kind":"EARLY","tag":"⚡️ EARLY"
    }

def scan_symbol(sym,bar_i):
    kl=futures_get_klines(sym,"1h",200)
    if len(kl)<60: return []
    res=[]
    s1=build_scalp_signal(sym,kl,bar_i);  s2=build_early_signal(sym,kl,bar_i);  s3=build_pema_signal(sym,kl,bar_i)
    if s1: res.append(s1)
    if s2: res.append(s2)
    if s3: res.append(s3)
    return res

def run_parallel(symbols,bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(scan_symbol,s,bar_i) for s in symbols]
        for f in as_completed(futs):
            try: sigs=f.result()
            except: sigs=[]
            if sigs: out.extend(sigs)
    return out

# ===================== RL ENRICH / SIM ENGINE =====================

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE,[])
AI_RL         = safe_load(AI_RL_FILE,[])
SIM_POSITIONS = safe_load(SIM_POS_FILE,[])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE,[])

def enrich_with_ai_context(pos):
    best=None
    for s in reversed(AI_SIGNALS):
        if s.get("symbol")!=pos.get("symbol"): continue
        e_sig=s.get("entry"); e_pos=pos.get("entry")
        if not e_sig or not e_pos: continue
        if abs(e_sig-e_pos)/max(e_sig,1e-12) < 0.002:
            best=s; break
    if best:
        for k in ("rsi","atr","chg24h","born_bar","tier","power","early","kind","tag"):
            if k in best: pos[k]=best.get(k)
    return pos

def queue_sim_variants(sig):
    delays=[(30*60,"approve_30m",30),(60*60,"approve_1h",60),(90*60,"approve_1h30",90),(120*60,"approve_2h",120)]
    now_s=now_ts_s()
    for secs,label,mins in delays:
        SIM_QUEUE.append({
            "symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
            "entry":sig["entry"],"tp":sig["tp"],"sl":sig["sl"],"power":sig["power"],
            "created_ts":now_s,"open_after_ts":now_s+secs,
            "approve_delay_min":mins,"approve_label":label,
            "status":"PENDING","early":bool(sig.get("early",False)),
            "kind":sig.get("kind",""),"tag":sig.get("tag","")
        })
    safe_save(SIM_POS_FILE,SIM_QUEUE)

def process_sim_queue_and_open_due():
    global SIM_POSITIONS
    now_s=now_ts_s()
    remain=[]; opened=False
    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            SIM_POSITIONS.append({**q,"status":"OPEN","open_ts":now_s,"open_time":now_local_iso()})
            opened=True
            log(f"[SIM OPEN] {q['symbol']} {q['dir']} approve={q['approve_delay_min']}m kind={q.get('kind')}")
        else:
            remain.append(q)
    SIM_QUEUE[:]=remain
    if opened: safe_save(SIM_POS_FILE,SIM_POSITIONS)

def _unlock_trend_for(sym, delay_unlock=False):
    if delay_unlock:
        TREND_LOCK_TIME[sym]=now_ts_s()
        log(f"[TRENDLOCK DELAY CLEAR] {sym} (6h cooldown started)")
        return
    TREND_LOCK.pop(sym,None); TREND_LOCK_TIME.pop(sym,None)
    log(f"[TRENDLOCK CLEAR] {sym}")

def process_sim_closes():
    global SIM_POSITIONS
    if not SIM_POSITIONS: return
    still=[]; changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")!="OPEN": continue
        last=futures_get_price(pos["symbol"])
        if last is None:
            still.append(pos); continue
        hit=None
        if pos["dir"]=="UP":
            if last>=pos["tp"]: hit="TP"
            elif last<=pos["sl"]: hit="SL"
        else:
            if last<=pos["tp"]: hit="TP"
            elif last>=pos["sl"]: hit="SL"
        if hit:
            close_time=now_local_iso()
            gain_pct=((last/pos["entry"]-1.0)*100.0 if pos["dir"]=="UP" else (pos["entry"]/last-1.0)*100.0)
            SIM_CLOSED.append({
                **enrich_with_ai_context(dict(pos)),
                "status":"CLOSED","close_time":close_time,
                "exit_price":last,"exit_reason":hit,"gain_pct":gain_pct
            })
            _unlock_trend_for(pos["symbol"], delay_unlock=True)  # kapanışta cooldown
            changed=True
            log(f"[SIM CLOSE] {pos['symbol']} {pos['dir']} {hit} {gain_pct:.3f}% approve={pos.get('approve_delay_min')}m kind={pos.get('kind')}")
        else:
            still.append(pos)
    SIM_POSITIONS=still
    if changed:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE,SIM_CLOSED)

# ===================== TRENDLOCK UTILS =====================

def _set_trend_lock(sym, direction):
    TREND_LOCK[sym]=direction
    TREND_LOCK_TIME[sym]=now_ts_s()
    log(f"[TRENDLOCK SET] {sym} {direction}")

def _cleanup_trend_lock_expired():
    now_s=now_ts_s()
    expired=[sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        _unlock_trend_for(sym)
        log(f"[TRENDLOCK TIMEOUT] {sym} (6h cooldown bitti)")
# ===================== SMART TP (USD/% + Decimal Fix) =====================

def adjust_precision(sym,v,kind="qty"):
    f=get_symbol_filters(sym)
    step=f["stepSize"] if kind=="qty" else f["tickSize"]
    if step<=0: return v
    return round(round(v/step)*step,12)

def calc_order_qty(sym,entry,usd):
    raw = usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd,1e-12)
    return (entry_exec*(1+tp_pct) if direction=="UP" else entry_exec*(1-tp_pct)), tp_pct

def futures_set_tp_only(sym, direction, qty, entry_exec, tp_low_usd=1.6, tp_high_usd=2.0):
    try:
        f=get_symbol_filters(sym)
        minp=f["minPrice"]; maxp=f["maxPrice"]
        pos_side="LONG" if direction=="UP" else "SHORT"; side="SELL" if direction=="UP" else "BUY"
        trade_usd=PARAM.get("TRADE_SIZE_USDT",250.0)
        usd_based = entry_exec>0.2  # mikro fiyatlarda % fallback

        def try_once(tp_price_candidate, order_type, tp_usd_used=None, tp_pct_used=None):
            price=round_to_tick(sym,tp_price_candidate)
            if price<minp: price=round_to_tick(sym,minp)
            if price>maxp: price=round_to_tick(sym,maxp)
            stop_str=format_price_by_tick(sym,price)
            if float(stop_str)<=0:
                price=round_to_tick(sym,max(minp,1e-12))
                stop_str=format_price_by_tick(sym,price)
                if float(stop_str)<=0:
                    log(f"[TP GUARD] {sym} stop=0 minp jump failed")
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
            for tp_usd in [round(x,1) for x in np.arange(tp_low_usd, tp_high_usd+0.001, 0.1)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,"TAKE_PROFIT_MARKET",tp_usd,tp_pct)
                if ok: return True,u,p
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,"TAKE_PROFIT_MARKET",tp_usd,tp_pct)
                if ok: return True,u,p
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,"STOP_MARKET",tp_usd,tp_pct)
                if ok: return True,u,p
        else:
            for tp_pct in [round(x,4) for x in np.arange(0.0050, 0.0100+0.0001, 0.0005)]:  # %0.5 → %1.0
                tp_price = entry_exec*(1+tp_pct if direction=="UP" else 1-tp_pct)
                ok,u,p=try_once(tp_price,"TAKE_PROFIT_MARKET",None,tp_pct)
                if ok: return True,u,p

        log(f"[NO TP] {sym} uygun TP bulunamadı.")
        return False,None,None
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False,None,None

# ===================== GUARDS / HEARTBEAT / REPORT =====================

def update_directional_limits():
    live={"long":{}, "short":{},"long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            amt=float(p["positionAmt"]); sym=p["symbol"]
            if amt>0: live["long"][sym]=amt
            elif amt<0: live["short"][sym]=abs(amt)
        live["long_count"]=len(live["long"])
        live["short_count"]=len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")

    STATE["long_blocked"]  = (live["long_count"]  >= PARAM["MAX_BUY"])
    STATE["short_blocked"] = (live["short_count"] >= PARAM["MAX_SELL"])
    STATE["auto_trade_active"] = not (STATE["long_blocked"] and STATE["short_blocked"])
    safe_save(STATE_FILE,STATE)
    return live

def heartbeat_and_status_check(live_positions_snapshot):
    now=time.time()
    if now-STATE.get("last_api_check",0)<600:
        return
    STATE["last_api_check"]=now
    safe_save(STATE_FILE,STATE)
    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift=abs(now_ts_ms()-st)
        ping_ok=requests.get(BINANCE_FAPI+"/fapi/v1/ping",timeout=5).status_code==200
        key_ok=True
        try: _=_signed_request("GET","/fapi/v2/account",{"timestamp":now_ts_ms()})
        except: key_ok=False
        hb = (f"✅ HEARTBEAT drift={int(drift)}ms ping={ping_ok} key={key_ok}"
              if ping_ok and key_ok and drift<1500 else
              f"⚠️ HEARTBEAT ping={ping_ok} key={key_ok} drift={int(drift)}")
        tg_send(hb); log(hb)
    except Exception as e:
        tg_send(f"❌ HEARTBEAT {e}"); log(f"[HBERR]{e}")

    msg=(f"📊 STATUS bar:{STATE.get('bar_index',0)} "
         f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
         f"long_blocked:{STATE.get('long_blocked')} "
         f"short_blocked:{STATE.get('short_blocked')} "
         f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
         f"sim_closed:{len(SIM_CLOSED)}")
    tg_send(msg); log(msg)

def ai_log_signal(sig):
    AI_SIGNALS.append({
        "time":now_local_iso(),"symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
        "chg24h":sig["chg24h"],"power":sig["power"],"rsi":sig.get("rsi"),"atr":sig.get("atr"),
        "tp":sig["tp"],"sl":sig["sl"],"entry":sig["entry"],"born_bar":sig.get("born_bar"),
        "early":bool(sig.get("early",False)),"kind":sig.get("kind",""),"tag":sig.get("tag","")
    })
    safe_save(AI_SIGNALS_FILE,AI_SIGNALS)

def ai_update_analysis_snapshot():
    snapshot={
        "time":now_local_iso(),
        "ultra_signals_total": sum(1 for x in AI_SIGNALS if x.get("tier")=="ULTRA"),
        "real_signals_total":  sum(1 for x in AI_SIGNALS if x.get("tier")=="REAL"),
        "normal_signals_total":sum(1 for x in AI_SIGNALS if x.get("tier")=="NORMAL"),
        "early_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="EARLY"),
        "pema_signals_total":  sum(1 for x in AI_SIGNALS if x.get("kind")=="PEMA"),
        "scalp_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="SCALP"),
        "sim_open_count":len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
        "sim_closed_count":len(SIM_CLOSED)
    }
    AI_ANALYSIS.append(snapshot); safe_save(AI_ANALYSIS_FILE,AI_ANALYSIS)

def auto_report_if_due():
    now_now=time.time()
    if now_now-STATE.get("last_report",0) < 14400:
        return
    ai_update_analysis_snapshot()
    for fpath in [AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,SIM_POS_FILE,SIM_CLOSED_FILE,PARAM_FILE,STATE_FILE]:
        try:
            if os.path.exists(fpath) and os.path.getsize(fpath)>20*1024*1024:
                with open(fpath,"r",encoding="utf-8") as f: raw=f.read()
                tail=raw[-int(len(raw)*0.2):]
                with open(fpath,"w",encoding="utf-8") as f: f.write(tail)
        except: pass
        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")
    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"]=now_now; safe_save(STATE_FILE,STATE)

# ===================== TELEGRAM COMMANDS =====================

STATE_DEFAULT={
    "bar_index":0, "last_report":0, "auto_trade_active":True,
    "last_api_check":0, "long_blocked":False, "short_blocked":False,
    "tg_update_offset":0
}
PARAM_DEFAULT={
    "SCALP_TP_PCT":0.006, "SCALP_SL_PCT":0.20, "TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30, "MAX_SELL":30,
    "ANGLE_MIN":0.00002, "FAST_EMA_PERIOD":3, "SLOW_EMA_PERIOD":7,
    "ATR_SPIKE_RATIO":0.03, "SCALP_APPROVE_BARS":0
}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
if not isinstance(PARAM,dict): PARAM=PARAM_DEFAULT
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

def _tg_get_updates():
    if not BOT_TOKEN: return []
    try:
        url=f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params={"timeout":0,"offset":STATE.get("tg_update_offset",0)}
        r=requests.get(url,params=params,timeout=10).json()
        return r.get("result",[])
    except: return []

def _tg_set_offset(new_off):
    STATE["tg_update_offset"]=new_off
    safe_save(STATE_FILE,STATE)

def _cmd_status():
    live=update_directional_limits()
    tg_send(
        f"📊 /status bar:{STATE.get('bar_index')} "
        f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
        f"long:{live.get('long_count',0)} short:{live.get('short_count',0)} "
        f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
        f"sim_closed:{len(SIM_CLOSED)}"
    )

def _cmd_report():
    ai_update_analysis_snapshot()
    tg_send_file(AI_SIGNALS_FILE,"📄 ai_signals.json")
    tg_send_file(AI_ANALYSIS_FILE,"📄 ai_analysis.json")
    tg_send_file(AI_RL_FILE,"📄 ai_rl_log.json")
    tg_send_file(SIM_POS_FILE,"📄 sim_positions.json")
    tg_send_file(SIM_CLOSED_FILE,"📄 sim_closed.json")

def _cmd_set(args):
    try:
        key=args[0]; val=" ".join(args[1:])
        if val.lower() in ("true","false"):
            v = (val.lower()=="true")
        else:
            try:
                v=float(val)
                if v.is_integer(): v=int(v)
            except:
                v=val
        PARAM[key]=v
        safe_save(PARAM_FILE,PARAM)
        tg_send(f"✅ /set {key} = {v}")
    except Exception as e:
        tg_send(f"❌ /set hata: {e}")

def _cmd_export():
    for fpath in [PARAM_FILE,STATE_FILE,AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,SIM_POS_FILE,SIM_CLOSED_FILE,LOG_FILE]:
        tg_send_file(fpath, f"📦 {os.path.basename(fpath)}")

def check_telegram_commands():
    if not BOT_TOKEN or not CHAT_ID: return
    updates=_tg_get_updates()
    if not updates: return
    for up in updates:
        _tg_set_offset(up["update_id"]+1)
        msg=up.get("message") or up.get("edited_message")
        if not msg: continue
        chat_id = str(msg.get("chat",{}).get("id"))
        if chat_id != str(CHAT_ID):  # tek chat filtre
            continue
        text=msg.get("text","").strip()
        if not text.startswith("/"): continue
        parts=text.split(); cmd=parts[0].lower(); args=parts[1:]
        if cmd=="/status": _cmd_status()
        elif cmd=="/report": _cmd_report()
        elif cmd=="/set" and args: _cmd_set(args)
        elif cmd=="/export": _cmd_export()
        else:
            tg_send("Komutlar: /status, /report, /set KEY VALUE, /export")

# ===================== REAL / EARLY / PEMA TRADE =====================

def open_market_position(sym, direction, qty):
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })
    fill = res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    return {"symbol":sym,"dir":direction,"qty":qty,"entry":float(fill),"pos_side":pos_side}

def _duplicate_or_locked(sym, direction):
    if TREND_LOCK.get(sym)==direction:
        log(f"[TRENDLOCK HIT] {sym} {direction}")
        return True
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[POSRISK ERR]{e}"); acc=[]
    if direction=="UP":
        if sym in [p["symbol"] for p in acc if float(p["positionAmt"])>0]:
            log(f"[DUP-LONG] {sym}"); return True
    else:
        if sym in [p["symbol"] for p in acc if float(p["positionAmt"])<0]:
            log(f"[DUP-SHORT] {sym}"); return True
    return False

def _can_direction(direction):
    if not STATE.get("auto_trade_active", True): return False
    if direction=="UP" and STATE.get("long_blocked",False):  return False
    if direction=="DOWN" and STATE.get("short_blocked",False): return False
    return True

def execute_real_trade(sig):
    # Opsiyonel scalp onayı
    approve_bars = int(PARAM.get("SCALP_APPROVE_BARS",0))
    if approve_bars>0 and (STATE.get("bar_index",0) - sig.get("born_bar",0) < approve_bars):
        return

    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig["power"]
    kind=sig.get("kind","")
    is_early = (kind=="EARLY")
    is_pema  = (kind=="PEMA")

    # Gerçek trade koşulu: power 65–75
    if not (65 <= pwr < 75): return
    if not _can_direction(direction): return
    if _duplicate_or_locked(sym,direction): return

    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return

    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened.get("entry") or futures_get_price(sym)
        if not entry_exec or entry_exec<=0:
            log(f"[OPEN FAIL] {sym} entry alınamadı."); return

        tp_ok, tp_usd_used, tp_pct_used = futures_set_tp_only(
            sym,direction,qty,entry_exec,tp_low_usd=1.6,tp_high_usd=2.0
        )

        _set_trend_lock(sym, direction)

        # Telegram etiketi
        if is_early: prefix="⚡️ EARLY"
        elif is_pema: prefix=sig.get("tag","🟪 PEMA")
        else: prefix="🟩 REAL"

        if tp_ok:
            tp_line = (f"TP hedefi:{tp_usd_used:.2f}$" if tp_usd_used is not None
                       else f"TP hedefi:%{(tp_pct_used or 0)*100:.2f}")
            tp_pct_show = (tp_pct_used or (tp_usd_used or 0)/max(PARAM.get('TRADE_SIZE_USDT',250.0),1e-12))*100
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"Power:{pwr:.2f}\n"
                    f"Entry:{entry_exec:.12f}\n"
                    f"{tp_line} ({tp_pct_show:.3f}%)\n"
                    f"time:{now_local_iso()}")
        else:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"Power:{pwr:.2f}\n"
                    f"Entry:{entry_exec:.12f}\n"
                    f"TP: YOK (USD/% tarama başarısız)\n"
                    f"time:{now_local_iso()}")

        AI_RL.append({
            "time":now_local_iso(),"symbol":sym,"dir":direction,"entry":entry_exec,
            "tp_usd_used":tp_usd_used,"tp_pct_used":tp_pct_used,"tp_ok":tp_ok,
            "power":pwr,"born_bar":sig.get("born_bar"),
            "early":is_early,"kind":kind,"tag":sig.get("tag","")
        })
        safe_save(AI_RL_FILE,AI_RL)

    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")

# ===================== MAIN LOOP =====================

def auto_init_symbols():
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}"); symbols=[]
    symbols.sort(); return symbols

def main():
    tg_send("🚀 EMA ULTRA v15.9.31 aktif (PEMA Slope Confirmed, EARLY ATR=0.03, Smart TP, 6h TrendLock)")
    log("[START] EMA ULTRA v15.9.31 FULL")

    symbols=auto_init_symbols()

    while True:
        try:
            # Telegram komutlarını dinle
            check_telegram_commands()

            # bar index
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]

            # 1) Sinyal tarama
            sigs=run_parallel(symbols,bar_i)

            # 2) Sinyal kayıt + SIM approve + Gerçek trade
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)
                update_directional_limits()
                execute_real_trade(sig)

            # 3) SIM open/close
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4) 4 saatlik auto-backup
            auto_report_if_due()

            # 5) Heartbeat (10 dk)
            heartbeat_and_status_check({})

            # 6) TrendLock cooldown temizliği
            _cleanup_trend_lock_expired()

            # 7) state save & sleep
            safe_save(STATE_FILE,STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ===================== ENTRY =====================

if __name__=="__main__":
    main()