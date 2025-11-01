import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================
# 📘 EMA ULTRA v15.9.33a — EARLY + E200S REAL Mode (No SL, TP Safe)
#  - TP Safe Patch (uses price instead of stopPrice)
#  - EARLY cross (EMA3–EMA7 + ATR 0.03)
#  - E200S (EMA50/200 + MACD1m + RSI5m)
#  - REAL mode active for both strategies
#  - TP auto scan 1.6→2.0 USD or 0.5%
#  - No SL, no reduceOnly, TrendLock 6 h
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE       = os.path.join(DATA_DIR, "state.json")
PARAM_FILE       = os.path.join(DATA_DIR, "params.json")
AI_SIGNALS_FILE  = os.path.join(DATA_DIR, "ai_signals.json")
AI_ANALYSIS_FILE = os.path.join(DATA_DIR, "ai_analysis.json")
AI_RL_FILE       = os.path.join(DATA_DIR, "ai_rl_log.json")
SIM_POS_FILE     = os.path.join(DATA_DIR, "sim_positions.json")
SIM_CLOSED_FILE  = os.path.join(DATA_DIR, "sim_closed.json")
LOG_FILE         = os.path.join(DATA_DIR, "log.txt")

BOT_TOKEN      = os.getenv("BOT_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
PRECISION_CACHE = {}
TREND_LOCK = {}
TREND_LOCK_TIME = {}
TRENDLOCK_EXPIRY_SEC = 6 * 3600  # 6 saat

def safe_load(p, d):
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return d

def safe_save(p, d):
    try:
        with SAVE_LOCK:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p)
    except Exception as e:
        print("[SAVE ERR]", e, flush=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_local_iso(): return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": t}, timeout=10)
    except: pass

def _signed_request(m, path, payload):
    q = "&".join([f"{k}={payload[k]}" for k in payload])
    sig = hmac.new(BINANCE_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + q + "&signature=" + sig
    r = requests.post(url, headers=headers, timeout=10) if m=="POST" else requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

# ================= Indicators =================
def ema(vals, n):
    k = 2/(n+1); e = [vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+2: return [50]*len(vals)
    d=np.diff(vals); g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:period]); al=np.mean(l[:period])
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
        if i==0: tr.append(h[i]-l[i])
        else: tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def calc_power(e_now,e_prev,atr_v,price,rsi_val):
    diff = abs(e_now - e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base = 55 + diff*20 + ((rsi_val-50)/50)*15 + (atr_v/price)*200
    return min(100, max(0, base))

def tier_from_power(p):
    if 65<=p<75: return "REAL","🟩"
    if p>=75: return "ULTRA","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""

# ================= Parameters =================
PARAM_DEFAULT={
  "TRADE_SIZE_USDT":250.0,
  "ANGLE_MIN":0.00002,
  "FAST_EMA_PERIOD":3,
  "SLOW_EMA_PERIOD":7,
  "ATR_SPIKE_RATIO":0.03,   # daha hassas EARLY
  "TP_LOW_USD":1.6,
  "TP_HIGH_USD":2.0,
  "SCALP_TP_PCT":0.005,
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
# ================= Missing helpers (define if not present) =================
if 'PRECISION_CACHE' not in globals():
    PRECISION_CACHE = {}

def _fmt_by_tick(tick):
    if "." in str(tick):
        dec=len(str(tick).split(".")[1].rstrip("0"))
    else:
        dec=0
    return f"{{:.{dec}f}}"

if 'get_symbol_filters' not in globals():
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
            PRECISION_CACHE[sym]={
                "stepSize":0.0001,"tickSize":0.0001,
                "minPrice":0.00000001,"maxPrice":99999999
            }
        return PRECISION_CACHE[sym]

if 'adjust_precision' not in globals():
    def adjust_precision(sym, v, kind="qty"):
        f=get_symbol_filters(sym)
        step=f["stepSize"] if kind=="qty" else f["tickSize"]
        if step<=0: return v
        return round(round(v/step)*step, 12)

if 'futures_get_klines' not in globals():
    def futures_get_klines(sym, it, lim):
        try:
            r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                           params={"symbol":sym,"interval":it,"limit":lim},
                           timeout=10).json()
            if r and int(r[-1][6])>int(datetime.now(timezone.utc).timestamp()*1000):
                r=r[:-1]
            return r
        except:
            return []

if 'futures_get_price' not in globals():
    def futures_get_price(sym):
        try:
            r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                           params={"symbol":sym},timeout=5).json()
            return float(r["price"])
        except:
            return None

# ================= Global stores (init if missing) =================
if 'STATE' not in globals(): STATE = {}
if 'PARAM' not in globals(): PARAM = {}
if 'AI_SIGNALS' not in globals():
    AI_SIGNALS = []
    try:
        from pathlib import Path
        DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
        AI_SIGNALS = json.load(open(os.path.join(DATA_DIR,"ai_signals.json"), "r", encoding="utf-8"))
    except: pass

if 'AI_RL' not in globals():
    AI_RL = []
    try:
        AI_RL = json.load(open(os.path.join(DATA_DIR,"ai_rl_log.json"), "r", encoding="utf-8"))
    except: pass

if 'SIM_POSITIONS' not in globals():
    SIM_POSITIONS = []
    try:
        SIM_POSITIONS = json.load(open(os.path.join(DATA_DIR,"sim_positions.json"), "r", encoding="utf-8"))
    except: pass

if 'SIM_CLOSED' not in globals():
    SIM_CLOSED = []
    try:
        SIM_CLOSED = json.load(open(os.path.join(DATA_DIR,"sim_closed.json"), "r", encoding="utf-8"))
    except: pass

if 'TREND_LOCK' not in globals(): TREND_LOCK = {}
if 'TREND_LOCK_TIME' not in globals(): TREND_LOCK_TIME = {}

# ================= Context helpers =================
def _safe(v, default=None):
    try:
        if v is None: return default
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return default
        return v
    except:
        return default

def build_context_pack(sym, closes, highs, lows, extra=None):
    extra = extra or {}
    e7   = ema(closes, 7)
    e50  = ema(closes, 50) if len(closes)>=50 else [closes[-1]]
    e200 = ema(closes,200) if len(closes)>=200 else [closes[-1]]
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
    ctx.update(extra or {})
    return ctx

def attach_context(sig, closes, highs, lows, extra=None):
    sig["context"] = build_context_pack(sig["symbol"], closes, highs, lows, extra)
    return sig

# ================= EARLY signal (1h EMA3–EMA7 + ATR spike) =================
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

    up_cross = (ema_fast[-2] > ema_slow[-2]) and (ema_fast[-3] <= ema_slow[-3])
    dn_cross = (ema_fast[-2] < ema_slow[-2]) and (ema_fast[-3] >= ema_slow[-3])
    if not (up_cross or dn_cross):
        return None

    atrs=atr_like(highs,lows,closes)
    if len(atrs)<2: return None
    if not (atrs[-1] >= atrs[-2]*(1.0 + PARAM.get("ATR_SPIKE_RATIO",0.03))):
        return None

    direction = "UP" if up_cross else "DOWN"
    entry = closes[-1]
    r_val = rsi(closes)[-1]
    pwr = calc_power(ema_slow[-1], ema_slow[-2], atrs[-1], entry, r_val)

    # Power band (65–75)
    if not (65 <= pwr < 75):
        return None

    sig = {
        "symbol": sym, "dir": direction, "tier": "REAL",
        "kind": "EARLY", "tag": f"⚡️ EARLY {'BUY' if direction=='UP' else 'SELL'}",
        "entry": entry, "tp": None, "sl": None,
        "power": pwr, "rsi": r_val, "atr": atrs[-1],
        "chg24h": 0.0, "time": now_local_iso(),
        "born_bar": bar_i, "early": True
    }
    extra = {"atr": atrs[-1], "rsi": r_val, "power": pwr}
    sig = attach_context(sig, closes, highs, lows, extra)
    return sig

# ================= E200S signal (5m EMA50/200 + 1m MACD hist + RSI) =================
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

    k1 = futures_get_klines(sym, "1m", 200)
    if len(k1) < 30:
        return None
    c1 = [float(k[4]) for k in k1]
    ema12 = ema(c1, 12); ema26 = ema(c1, 26)
    macd_hist = [ema12[i]-ema26[i] for i in range(len(c1))]
    hist_now, hist_prev = macd_hist[-1], macd_hist[-2]

    direction=None
    if (trend=="UP"
        and hist_now > hist_prev * 1.30
        and rsi_v < 60.0):
        direction="UP"
    elif (trend=="DOWN"
        and hist_now < hist_prev * 0.70
        and rsi_v > 40.0):
        direction="DOWN"
    if not direction:
        return None

    pwr = min(100, max(55, 70 + abs(hist_now - hist_prev)*80))
    if pwr < 60.0:
        return None

    entry = closes[-1]
    sig = {
        "symbol": sym, "dir": direction, "tier": "REAL",
        "kind": "E200S", "tag": f"📘 E200S {'BUY' if direction=='UP' else 'SELL'}",
        "entry": entry, "tp": None, "sl": None,
        "power": pwr, "rsi": rsi_v,
        "atr": np.std(closes[-20:]) / max(entry,1e-12),
        "chg24h": 0.0, "time": now_local_iso(),
        "born_bar": 0, "early": False
    }
    extra = {"rsi": rsi_v, "power": pwr}
    sig = attach_context(sig, closes, highs, lows, extra)
    return sig

# ================= Scanner =================
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

# ================= TP SAFE (price only, no stopPrice) =================
def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd, 1e-12)
    return (entry_exec*(1+tp_pct) if direction=="UP" else entry_exec*(1-tp_pct)), tp_pct

def futures_set_tp_only_safe(sym, direction, qty, entry_exec, tp_low_usd, tp_high_usd):
    """
    TP Safe Patch:
      * Only 'price' is sent (no stopPrice)
      * Validate minPrice/maxPrice
      * Align to tickSize
    """
    try:
        f=get_symbol_filters(sym)
        tick=f["tickSize"]; minp=f.get("minPrice",0.0); maxp=f.get("maxPrice",9e9)
        pos_side="LONG" if direction=="UP" else "SHORT"
        side    ="SELL" if direction=="UP" else "BUY"
        fmt=_fmt_by_tick(tick)
        trade_usd=PARAM.get("TRADE_SIZE_USDT",250.0)

        def try_once(tp_usd, order_type):
            tp_price, tp_pct = _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd)
            tp_price = max(tp_price, minp*1.001)
            tp_price = min(tp_price, maxp*0.999)
            tp_price = adjust_precision(sym, tp_price, "price")
            if tp_price <= 0:
                log(f"[TP RANGE] {sym} skip ${tp_usd} price={tp_price}")
                return False, None, None
            payload={
                "symbol":sym,"side":side,"type":order_type,
                "price":fmt.format(tp_price),
                "quantity":f"{qty}",
                "positionSide":pos_side,"timestamp":now_ts_ms()
            }
            try:
                _signed_request("POST","/fapi/v1/order",payload)
                log(f"[TP OK] {sym} {order_type} tp=${tp_usd} price={fmt.format(tp_price)} qty={qty}")
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
        # STOP_MARKET fallback (price field only)
        for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
            ok,u,p=try_once(tp_usd,"STOP_MARKET")
            if ok: return True,u,p

        log(f"[NO TP] {sym} 1.6–2.0$ aralığında geçerli TP bulunamadı.")
        return False, None, None
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False, None, None

# ================= Guards & REAL trade =================
TRENDLOCK_EXPIRY_SEC = 6*3600

def _unlock_trend_for(sym):
    if sym in TREND_LOCK: TREND_LOCK.pop(sym, None)
    if sym in TREND_LOCK_TIME: TREND_LOCK_TIME.pop(sym, None)
    log(f"[TRENDLOCK CLEAR] {sym}")

def _set_trend_lock(sym, direction):
    TREND_LOCK[sym]=direction
    TREND_LOCK_TIME[sym]=int(datetime.now(timezone.utc).timestamp())
    log(f"[TRENDLOCK SET] {sym} {direction}")

def _duplicate_or_locked(sym, direction):
    if TREND_LOCK.get(sym)==direction:
        log(f"[TRENDLOCK HIT] {sym} {direction}")
        return True
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

def calc_order_qty(sym, entry, usd_size):
    raw = usd_size / max(entry,1e-12)
    return adjust_precision(sym, raw, "qty")

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

        # TP Safe (price only)
        tp_ok, tp_usd_used, tp_pct_used = futures_set_tp_only_safe(
            sym, direction, qty, entry_exec,
            PARAM.get("TP_LOW_USD",1.6), PARAM.get("TP_HIGH_USD",2.0)
        )

        _set_trend_lock(sym, direction)

        # Telegram
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
# ================ Logs & Analysis ================
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
    try:
        AI_ANALYSIS = safe_load(AI_ANALYSIS_FILE, [])
    except:
        AI_ANALYSIS = []
    snapshot={
        "time":now_local_iso(),
        "e200s_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="E200S"),
        "early_signals_total":  sum(1 for x in AI_SIGNALS if x.get("kind")=="EARLY"),
    }
    AI_ANALYSIS.append(snapshot)
    safe_save(AI_ANALYSIS_FILE, AI_ANALYSIS)

# ================ Directional Limits & Heartbeat ================
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

    STATE["long_blocked"]  = (live["long_count"]  >= 30)
    STATE["short_blocked"] = (live["short_count"] >= 30)
    STATE["auto_trade_active"] = not (STATE["long_blocked"] and STATE["short_blocked"])
    safe_save(STATE_FILE,STATE)
    return live

def _cleanup_trend_lock_expired():
    now_s=int(datetime.now(timezone.utc).timestamp())
    expired=[sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        if sym in TREND_LOCK: TREND_LOCK.pop(sym, None)
        if sym in TREND_LOCK_TIME: TREND_LOCK_TIME.pop(sym, None)
        log(f"[TRENDLOCK TIMEOUT] {sym} (6h)")

def heartbeat_and_status_check(live_positions_snapshot):
    now=time.time()
    last=STATE.get("last_api_check",0)
    if now-last<600: return
    STATE["last_api_check"]=now
    safe_save(STATE_FILE,STATE)

    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift=abs(int(datetime.now(timezone.utc).timestamp()*1000)-st)
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
         f"short_blocked:{STATE.get('short_blocked')}")
    tg_send(msg); log(msg)

# ================ Auto backup (4h) ================
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

# ================ Main Loop ================
def main():
    tg_send("🚀 EMA ULTRA v15.9.33a aktif (EARLY+E200S REAL, No SL, TP Safe)")
    log("[START] EMA ULTRA v15.9.33a FULL")

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

            # 1) Taramalar
            sigs = run_parallel(symbols, bar_i)

            # 2) Log + REAL trade
            for sig in sigs:
                ai_log_signal(sig)
                execute_real_trade(sig)

            # 3) Rutin işler
            auto_report_if_due()
            live=update_directional_limits()
            heartbeat_and_status_check(live)
            _cleanup_trend_lock_expired()

            safe_save(STATE_FILE,STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

if __name__=="__main__":
    main()
