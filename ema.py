# ==============================================================
# 📘 EMA ULTRA v15.9.33 — EARLY + E200S (REAL, No SL, Full Data)
#  - EARLY: EMA3–EMA7 cross + ATR spike (default 0.03)
#  - E200S: 5m EMA50/200 trend + 1m MACD hist spike + RSI filtresi
#  - REAL: Her iki strateji gerçek emir açar (TrendLock + DuplicateGuard)
#  - TP only: USD 1.6→2.0 (0.1→0.01) → STOP_MARKET fallback
#  - SL yok, reduceOnly yok
#  - TrendLock: 6 saat sonra auto-expire + close'da unlock
#  - Telegram: sadece gerçek fill + heartbeat
#  - SIM: kayıt ve bağlam zenginliği (ATR, RSI, EMA’lar, slope, vb.)
# ==============================================================

import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE        = os.path.join(DATA_DIR,"state.json")
PARAM_FILE        = os.path.join(DATA_DIR,"params.json")
AI_SIGNALS_FILE   = os.path.join(DATA_DIR,"ai_signals.json")
AI_ANALYSIS_FILE  = os.path.join(DATA_DIR,"ai_analysis.json")
AI_RL_FILE        = os.path.join(DATA_DIR,"ai_rl_log.json")
SIM_POS_FILE      = os.path.join(DATA_DIR,"sim_positions.json")
SIM_CLOSED_FILE   = os.path.join(DATA_DIR,"sim_closed.json")
LOG_FILE          = os.path.join(DATA_DIR,"log.txt")

BOT_TOKEN      = os.getenv("BOT_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
PRECISION_CACHE = {}
TREND_LOCK = {}          # { "SYMBOL": "UP"/"DOWN" }
TREND_LOCK_TIME = {}     # { "SYMBOL": last_set_ts }
TRENDLOCK_EXPIRY_SEC = 6 * 3600  # 6 saat

SIM_QUEUE = []           # analiz amaçlı approve varyant kuyruğu (REAL olsa da loglar)

def safe_load(p, d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return d

def safe_save(p, d):
    try:
        with SAVE_LOCK:
            tmp = p + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:
        print("[SAVE ERR]", e, flush=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())
def now_local_iso():
    # Türkiye (UTC+3) gösterimi
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":t},
            timeout=10
        )
    except:
        pass

def tg_send_file(p,cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p):
        return
    try:
        with open(p,"rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":cap},
                files={"document":(os.path.basename(p),f)},
                timeout=30
            )
    except:
        pass

def _signed_request(method, path, payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r = (requests.post(url,headers=headers,timeout=10) if method=="POST"
         else requests.get(url,headers=headers,timeout=10))
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    """
    exchangeInfo -> tickSize / stepSize / minPrice / maxPrice (cache)
    """
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
        PRECISION_CACHE[sym]={
            "stepSize":0.0001,
            "tickSize":0.0001,
            "minPrice":0.00000001,
            "maxPrice":99999999
        }
    return PRECISION_CACHE[sym]

def adjust_precision(sym, v, kind="qty"):
    f=get_symbol_filters(sym)
    step=f["stepSize"] if kind=="qty" else f["tickSize"]
    if step<=0:
        return v
    return round(round(v/step)*step, 12)

def calc_order_qty(sym, entry, usd_size):
    raw = usd_size / max(entry,1e-12)
    return adjust_precision(sym, raw, "qty")

def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                       params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym, it, lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                       params={"symbol":sym,"interval":it,"limit":lim},
                       timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms():
            r=r[:-1]  # son bar tamamlanmamışsa düş
        return r
    except:
        return []

# ================== INDICATORS ==================
def ema(vals, n):
    k=2/(n+1)
    e=[vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+2:
        return [50]*len(vals)
    d=np.diff(vals)
    g=np.maximum(d,0)
    l=-np.minimum(d,0)
    ag=np.mean(g[:period])
    al=np.mean(l[:period])
    out=[50]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period
        al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0:
            tr.append(h[i]-l[i])
        else:
            tr.append(max(
                h[i]-l[i],
                abs(h[i]-c[i-1]),
                abs(l[i]-c[i-1])
            ))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_val):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55 + diff*20 + ((rsi_val-50)/50)*15 + (atr_v/price)*200
    return min(100,max(0,base))

# ================== PARAM / STATE ==================
PARAM_DEFAULT = {
    # Genel
    "TRADE_SIZE_USDT": 250.0,
    "MAX_BUY": 30,
    "MAX_SELL": 30,

    # EARLY
    "FAST_EMA_PERIOD": 3,
    "SLOW_EMA_PERIOD": 7,
    "ATR_SPIKE_RATIO": 0.03,    # early için daha hassas
    "EARLY_POWER_MIN": 65.0,    # 65–75 bandı önerilir
    "EARLY_POWER_MAX": 75.0,

    # E200S
    "E200S_RSI_MAX_LONG": 60.0,  # long onayı için
    "E200S_RSI_MIN_SHORT": 40.0, # short onayı için
    "E200S_HIST_UP_FACTOR": 1.30,
    "E200S_HIST_DN_FACTOR": 0.70,
    "E200S_POWER_MIN": 60.0,

    # TP USD taraması
    "TP_USD_LOW": 1.6,
    "TP_USD_HIGH": 2.0
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index": 0,
    "last_report": 0,
    "auto_trade_active": True,
    "last_api_check": 0,
    "long_blocked": False,
    "short_blocked": False
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
for k,v in STATE_DEFAULT.items():
    STATE.setdefault(k,v)

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE, [])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE, [])
AI_RL         = safe_load(AI_RL_FILE, [])
SIM_POSITIONS = safe_load(SIM_POS_FILE, [])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE, [])
# ===================== CONTEXT PACK =====================
def _safe(v, default=None):
    return default if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))) else v

def build_context_pack(sym, closes, highs, lows, extra=None):
    extra = extra or {}
    e7 = ema(closes, 7)
    e50 = ema(closes, 50) if len(closes) >= 50 else [closes[-1]]
    e200 = ema(closes, 200) if len(closes) >= 200 else [closes[-1]]
    slope7 = _safe(e7[-1]-e7[-2], 0.0) if len(e7)>=2 else 0.0
    ctx = {
        "symbol": sym,
        "close": _safe(closes[-1], 0.0),
        "ema7": _safe(e7[-1], 0.0),
        "ema7_prev": _safe(e7[-2] if len(e7)>=2 else e7[-1], 0.0),
        "ema50": _safe(e50[-1], 0.0),
        "ema200": _safe(e200[-1], 0.0),
        "slope7": _safe(slope7, 0.0),
        "hl_range": _safe(highs[-1]-lows[-1], 0.0),
        "time": now_local_iso()
    }
    ctx.update(extra)
    return ctx

def attach_context(sig, closes, highs, lows, extra=None):
    sig["context"] = build_context_pack(sig["symbol"], closes, highs, lows, extra)
    return sig

# ================== EARLY SIGNAL (1h EMA3–EMA7 + ATR spike) ==================
def build_early_signal(sym, bar_i):
    kl = futures_get_klines(sym, "1h", 200)
    if len(kl) < 60:
        return None
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    fper=PARAM.get("FAST_EMA_PERIOD",3)
    sper=PARAM.get("SLOW_EMA_PERIOD",7)
    ema_fast=ema(closes,fper)
    ema_slow=ema(closes,sper)

    # cross (confirmed bar)
    up_cross = (ema_fast[-2] > ema_slow[-2]) and (ema_fast[-3] <= ema_slow[-3])
    dn_cross = (ema_fast[-2] < ema_slow[-2]) and (ema_fast[-3] >= ema_slow[-3])
    if not (up_cross or dn_cross):
        return None

    atrs=atr_like(highs,lows,closes)
    if len(atrs)<2:
        return None
    if not (atrs[-1] >= atrs[-2]*(1.0 + PARAM.get("ATR_SPIKE_RATIO",0.03))):
        return None

    direction = "UP" if up_cross else "DOWN"
    entry = closes[-1]
    r_val = rsi(closes)[-1]
    pwr = calc_power(
        ema_slow[-1], ema_slow[-2],
        ema_slow[-5] if len(ema_slow)>=6 else ema_slow[-2],
        atrs[-1], entry, r_val
    )

    # güç bandı
    if not (PARAM.get("EARLY_POWER_MIN",65) <= pwr < PARAM.get("EARLY_POWER_MAX",75)):
        return None

    sig = {
        "symbol": sym, "dir": direction, "tier": "REAL",
        "kind": "EARLY", "tag": f"⚡️ EARLY {'BUY' if direction=='UP' else 'SELL'}",
        "entry": entry, "tp": None, "sl": None,  # TP sonra set edilecek
        "power": pwr, "rsi": r_val, "atr": atrs[-1],
        "chg24h": 0.0, "time": now_local_iso(),
        "born_bar": bar_i, "early": True
    }
    # context
    extra = {"atr": atrs[-1], "rsi": r_val, "power": pwr}
    sig = attach_context(sig, closes, highs, lows, extra)
    return sig

# ================== E200S SIGNAL (5m EMA50/200 + 1m MACD + RSI) ==================
def build_e200s_signal(sym):
    kl5 = futures_get_klines(sym, "5m", 400)
    if len(kl5) < 220:
        return None

    closes = [float(k[4]) for k in kl5]
    highs  = [float(k[2]) for k in kl5]
    lows   = [float(k[3]) for k in kl5]
    ema50  = ema(closes, 50)
    ema200 = ema(closes, 200)
    rsi_v  = rsi(closes, 14)[-1]

    trend = "UP" if ema50[-1] > ema200[-1] else "DOWN"

    # MACD histogram (1m)
    k1 = futures_get_klines(sym, "1m", 200)
    if len(k1) < 30:
        return None
    c1 = [float(k[4]) for k in k1]
    ema12 = ema(c1, 12); ema26 = ema(c1, 26)
    macd_hist = [ema12[i] - ema26[i] for i in range(len(c1))]
    hist_now, hist_prev = macd_hist[-1], macd_hist[-2]

    direction = None
    if (trend=="UP"
        and hist_now > hist_prev * PARAM.get("E200S_HIST_UP_FACTOR",1.30)
        and rsi_v < PARAM.get("E200S_RSI_MAX_LONG",60.0)):
        direction = "UP"
    elif (trend=="DOWN"
        and hist_now < hist_prev * PARAM.get("E200S_HIST_DN_FACTOR",0.70)
        and rsi_v > PARAM.get("E200S_RSI_MIN_SHORT",40.0)):
        direction = "DOWN"
    if not direction:
        return None

    # güç eşiği
    pwr = min(100, max(55, 70 + abs(hist_now - hist_prev)*80))
    if pwr < PARAM.get("E200S_POWER_MIN",60.0):
        return None

    entry = closes[-1]
    sig = {
        "symbol": sym, "dir": direction, "tier": "REAL",
        "kind": "E200S", "tag": f"📘 E200S {'BUY' if direction=='UP' else 'SELL'}",
        "entry": entry, "tp": None, "sl": None,  # TP sonra set edilecek
        "power": pwr, "rsi": rsi_v,
        "atr": np.std(closes[-20:]) / max(entry,1e-12),
        "chg24h": 0.0, "time": now_local_iso(),
        "born_bar": 0, "early": False
    }
    extra = {"rsi": rsi_v, "power": pwr}
    sig = attach_context(sig, closes, highs, lows, extra)
    return sig

# ================== SCAN ==================
def scan_symbol(sym, bar_i):
    res=[]
    try:
        s1 = build_early_signal(sym, bar_i)
        if s1: res.append(s1)
    except Exception as e:
        log(f"[EARLY ERR]{sym} {e}")
    try:
        s2 = build_e200s_signal(sym)
        if s2: res.append(s2)
    except Exception as e:
        log(f"[E200S ERR]{sym} {e}")
    return res

def run_parallel(symbols, bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(scan_symbol, s, bar_i) for s in symbols]
        for f in as_completed(futs):
            try:
                sigs=f.result()
            except:
                sigs=[]
            if sigs:
                out.extend(sigs)
    return out

# ================== SIM QUEUE (analiz için) ==================
def queue_sim_variants(sig):
    delays=[(30*60,"approve_30m",30),(60*60,"approve_1h",60),
            (90*60,"approve_1h30",90),(120*60,"approve_2h",120)]
    now_s=now_ts_s()
    for secs,label,mins in delays:
        SIM_QUEUE.append({
            "symbol":sig["symbol"], "dir":sig["dir"], "tier":sig["tier"],
            "entry":sig["entry"], "tp":sig.get("tp"), "sl":None,
            "power":sig.get("power"), "created_ts":now_s,
            "open_after_ts":now_s+secs, "approve_delay_min":mins, "approve_label":label,
            "status":"PENDING", "early":bool(sig.get("early",False)),
            "kind": sig.get("kind",""), "tag": sig.get("tag",""),
            "context": sig.get("context",{})
        })
    safe_save(SIM_POS_FILE, SIM_QUEUE)

def process_sim_queue_and_open_due():
    # REAL öncelikli; sim sadece analiz
    now_s=now_ts_s()
    remain=[]
    opened=False
    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            SIM_POSITIONS.append({**q,"status":"OPEN","open_ts":now_s,"open_time":now_local_iso()})
            opened=True
            log(f"[SIM OPEN] {q['symbol']} {q['dir']} approve={q['approve_delay_min']}m")
        else:
            remain.append(q)
    SIM_QUEUE[:] = remain
    if opened:
        safe_save(SIM_POS_FILE, SIM_POSITIONS)

def process_sim_closes():
    # SL yok; sim tarafında TP varsa kapatırız
    if not SIM_POSITIONS:
        return
    still=[]; changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")!="OPEN":
            continue
        last=futures_get_price(pos["symbol"])
        if last is None:
            still.append(pos); continue
        hit=None
        if pos["dir"]=="UP":
            if pos.get("tp") and last>=pos["tp"]: hit="TP"
        else:
            if pos.get("tp") and last<=pos["tp"]: hit="TP"
        if hit:
            close_time=now_local_iso()
            gain_pct=((last/pos["entry"]-1.0)*100.0 if pos["dir"]=="UP"
                      else (pos["entry"]/last-1.0)*100.0)
            SIM_CLOSED.append({
                **pos, "status":"CLOSED","close_time":close_time,
                "exit_price":last,"exit_reason":hit,"gain_pct":gain_pct
            })
            # close olduğunda trendlock kaldır
            _unlock_trend_for(pos["symbol"])
            changed=True
            log(f"[SIM CLOSE] {pos['symbol']} {pos['dir']} {hit} {gain_pct:.3f}% approve={pos.get('approve_delay_min')}")
        else:
            still.append(pos)
    # Temizle
    open_list=[p for p in still if p.get("status")=="OPEN"]
    SIM_POSITIONS[:] = open_list
    if changed:
        safe_save(SIM_POS_FILE, SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE, SIM_CLOSED)

# ================== TP ONLY (USD 1.6→2.0; 0.1→0.01; STOP_MARKET) ==================
def _fmt_by_tick(tick):
    if "." in str(tick):
        dec=len(str(tick).split(".")[1].rstrip("0"))
    else:
        dec=0
    return f"{{:.{dec}f}}"

def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd,1e-12)
    return (entry_exec*(1+tp_pct) if direction=="UP" else entry_exec*(1-tp_pct)), tp_pct

def futures_set_tp_only(sym, direction, qty, entry_exec, tp_low_usd, tp_high_usd):
    try:
        f=get_symbol_filters(sym)
        tick=f["tickSize"]
        minp=f.get("minPrice",0.0); maxp=f.get("maxPrice",9e9)
        pos_side="LONG" if direction=="UP" else "SHORT"
        side    ="SELL" if direction=="UP" else "BUY"
        fmt=_fmt_by_tick(tick)
        trade_usd=PARAM.get("TRADE_SIZE_USDT",250.0)

        def try_once(tp_usd, order_type):
            tp_price, tp_pct = _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd)
            if tp_price <= 0 or tp_price < minp or tp_price > maxp:
                log(f"[TP RANGE] {sym} skip ${tp_usd} price={tp_price}")
                return False, None, None
            payload={
                "symbol":sym,"side":side,"type":order_type,
                "stopPrice":fmt.format(adjust_precision(sym, tp_price, "price")),
                "quantity":f"{qty}",
                "workingType":"MARK_PRICE","closePosition":"true",
                "positionSide":pos_side,"timestamp":now_ts_ms()
            }
            try:
                _signed_request("POST","/fapi/v1/order",payload)
                log(f"[TP OK] {sym} {order_type} tp=${tp_usd} stop={fmt.format(tp_price)} qty={qty}")
                return True, tp_usd, tp_pct
            except Exception as e:
                log(f"[TP FAIL] {sym} {order_type} tp=${tp_usd} err={e}")
                return False, None, None

        # 0.1 step
        for tp_usd in [round(x,1) for x in np.arange(tp_low_usd, tp_high_usd+0.001, 0.1)]:
            ok,u,p=try_once(tp_usd,"TAKE_PROFIT_MARKET")
            if ok: return True,u,p
        # 0.01 step
        for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
            ok,u,p=try_once(tp_usd,"TAKE_PROFIT_MARKET")
            if ok: return True,u,p
        # STOP_MARKET fallback
        for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
            ok,u,p=try_once(tp_usd,"STOP_MARKET")
            if ok: return True,u,p

        log(f"[NO TP] {sym} 1.6–2.0$ aralığında geçerli TP bulunamadı.")
        return False, None, None
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False, None, None

# ================== GUARDS & TRADE ==================
def _unlock_trend_for(sym):
    if sym in TREND_LOCK:
        TREND_LOCK.pop(sym, None)
    if sym in TREND_LOCK_TIME:
        TREND_LOCK_TIME.pop(sym, None)
    log(f"[TRENDLOCK CLEAR] {sym}")

def _set_trend_lock(sym, direction):
    TREND_LOCK[sym]=direction
    TREND_LOCK_TIME[sym]=now_ts_s()
    log(f"[TRENDLOCK SET] {sym} {direction}")

def _duplicate_or_locked(sym, direction):
    # TrendLock
    if TREND_LOCK.get(sym)==direction:
        log(f"[TRENDLOCK HIT] {sym} {direction}")
        return True
    # Duplicate guard
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[POSRISK ERR]{e}")
        acc=[]
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

def open_market_position(sym, direction, qty):
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })
    fill = res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    return {"symbol":sym,"dir":direction,"qty":qty,"entry":float(fill),"pos_side":pos_side}

def execute_real_trade(sig):
    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig.get("power",0.0)

    if not _can_direction(direction):
        return
    if _duplicate_or_locked(sym, direction):
        return

    qty=calc_order_qty(sym, sig["entry"], PARAM.get("TRADE_SIZE_USDT",250.0))
    if not qty or qty<=0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return

    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened.get("entry") or futures_get_price(sym)
        if not entry_exec or entry_exec<=0:
            log(f"[OPEN FAIL] {sym} entry alınamadı."); return

        # TP set (USD scan 1.6→2.0)
        tp_ok, tp_usd_used, tp_pct_used = futures_set_tp_only(
            sym, direction, qty, entry_exec,
            PARAM.get("TP_USD_LOW",1.6), PARAM.get("TP_USD_HIGH",2.0)
        )

        _set_trend_lock(sym, direction)

        # Telegram (yalnız gerçek fill)
        prefix = sig.get("tag","✅ REAL")
        if tp_ok:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"Power:{pwr:.2f}\n"
                    f"Entry:{entry_exec:.12f}\n"
                    f"TP hedefi:{tp_usd_used:.2f}$ ({(tp_pct_used or 0)*100:.3f}%)\n"
                    f"time:{now_local_iso()}")
        else:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"Power:{pwr:.2f}\n"
                    f"Entry:{entry_exec:.12f}\n"
                    f"TP: YOK (1.6–2.0$ tarama başarısız)\n"
                    f"time:{now_local_iso()}")

        # RL log
        AI_RL.append({
            "time":now_local_iso(),"symbol":sym,"dir":direction,"entry":entry_exec,
            "tp_usd_used":tp_usd_used,"tp_pct_used":tp_pct_used,"tp_ok":tp_ok,
            "power":pwr,"born_bar":sig.get("born_bar"),"early":bool(sig.get("early",False)),
            "kind":sig.get("kind",""),"tag":sig.get("tag",""),
            "context":sig.get("context",{})
        })
        safe_save(AI_RL_FILE,AI_RL)

    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")
# ================== LOGS / ANALYSIS ==================
def ai_log_signal(sig):
    AI_SIGNALS.append({
        "time":now_local_iso(),
        "symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
        "kind":sig.get("kind",""),"tag":sig.get("tag",""),
        "early":bool(sig.get("early",False)),
        "entry":sig["entry"],"tp":sig.get("tp"),"sl":None,
        "power":sig.get("power"),"rsi":sig.get("rsi"),
        "atr":sig.get("atr"),"chg24h":sig.get("chg24h"),
        "born_bar":sig.get("born_bar"),
        "context":sig.get("context",{})
    })
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)

def ai_update_analysis_snapshot():
    snapshot={
        "time":now_local_iso(),
        "e200s_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="E200S"),
        "early_signals_total":  sum(1 for x in AI_SIGNALS if x.get("kind")=="EARLY"),
        "sim_open_count":len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
        "sim_closed_count":len(SIM_CLOSED)
    }
    AI_ANALYSIS.append(snapshot)
    safe_save(AI_ANALYSIS_FILE, AI_ANALYSIS)

# ================== Directional Limits & Heartbeat ==================
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

def _cleanup_trend_lock_expired():
    now_s=now_ts_s()
    expired=[sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        _unlock_trend_for(sym)
        log(f"[TRENDLOCK TIMEOUT] {sym} (6h)")

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

# ================== Auto backup (4h) ==================
def auto_report_if_due():
    now_now=time.time()
    if now_now-STATE.get("last_report",0) < 14400:
        return
    ai_update_analysis_snapshot()
    for fpath in [AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,SIM_POS_FILE,SIM_CLOSED_FILE]:
        try:
            if os.path.exists(fpath) and os.path.getsize(fpath)>10*1024*1024:
                with open(fpath,"r",encoding="utf-8") as f: raw=f.read()
                tail=raw[-int(len(raw)*0.2):]
                with open(fpath,"w",encoding="utf-8") as f: f.write(tail)
        except: pass
        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")
    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"]=now_now; safe_save(STATE_FILE,STATE)

# ================== MAIN LOOP ==================
def main():
    tg_send("🚀 EMA ULTRA v15.9.33 aktif (EARLY + E200S, REAL, No SL, TP 1.6–2.0$)")
    log("[START] EMA ULTRA v15.9.33 FULL")

    # USDT sembolleri
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}")
        symbols=[]
    symbols.sort()

    while True:
        try:
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]

            # 1) Tara
            sigs = run_parallel(symbols, bar_i)

            # 2) İşle
            for sig in sigs:
                # Kayıt
                ai_log_signal(sig)
                # SIM approve varyantları (analiz için)
                queue_sim_variants(sig)

                # Canlı pozisyon snapshot
                live=update_directional_limits()

                # REAL trade
                execute_real_trade(sig)

            # SIM open due / closes
            process_sim_queue_and_open_due()
            process_sim_closes()

            # Yedek / Rapor
            auto_report_if_due()

            # Heartbeat + status
            live=update_directional_limits()
            heartbeat_and_status_check(live)

            # TrendLock expiry cleanup (6h)
            _cleanup_trend_lock_expired()

            # State save & sleep
            safe_save(STATE_FILE,STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ================== ENTRY ==================
if __name__=="__main__":
    main()
