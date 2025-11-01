# ==============================================================
# 📘 EMA ULTRA v15.9.42 — ALL-REAL + 30/30 Guard + Stable TP
#  - Strategies: EARLY(EMA3/7+ATR spike), EMA200S, UT-STC, A+FVG
#  - All REAL trades, parallel SIM logs (approve variants)
#  - TP only (1.6→2.0 USD, 0.1→0.01 scan) via TAKE_PROFIT_MARKET
#  - No SL, no reduceOnly
#  - Max concurrent: 30 LONG + 30 SHORT (separate)
#  - TrendLock: 6h auto-expire, and unlock on close
#  - Telegram: only real fills & 5-min STATUS heartbeat
# ==============================================================

import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
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
TREND_LOCK = {}          # { "SYMBOL": "UP"/"DOWN" }
TREND_LOCK_TIME = {}     # { "SYMBOL": last_set_ts }
TRENDLOCK_EXPIRY_SEC = 6 * 3600  # 6 saat

# Max limits per direction
MAX_LONG_TRADES  = 30
MAX_SHORT_TRADES = 30

# Status heartbeat (sec)
STATUS_INTERVAL_SEC = 300

# ---------- Safe IO ----------
def safe_load(p, dflt):
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return dflt

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
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())
def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": t},
            timeout=10
        )
    except: pass

def tg_send_file(p, cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p):
        return
    try:
        with open(p, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id": CHAT_ID, "caption": cap},
                files={"document": (os.path.basename(p), f)},
                timeout=30
            )
    except: pass

# ---------- Signed request ----------
def _signed_request(method, path, payload):
    q = "&".join([f"{k}={payload[k]}" for k in payload])
    sig = hmac.new(BINANCE_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + q + "&signature=" + sig
    r = (requests.post(url, headers=headers, timeout=10) if method=="POST"
         else requests.get(url, headers=headers, timeout=10))
    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

# ---------- Exchange Info / Precision ----------
def get_symbol_filters(sym):
    if sym in PRECISION_CACHE:
        return PRECISION_CACHE[sym]
    try:
        info = requests.get(BINANCE_FAPI + "/fapi/v1/exchangeInfo", timeout=10).json()
        s = next((x for x in info["symbols"] if x["symbol"] == sym), None)
        lot = next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"), {})
        pricef = next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"), {})
        PRECISION_CACHE[sym] = {
            "stepSize": float(lot.get("stepSize","1")),
            "tickSize": float(pricef.get("tickSize","0.01")),
            "minPrice": float(pricef.get("minPrice","0.00000001")),
            "maxPrice": float(pricef.get("maxPrice","100000000"))
        }
    except Exception as e:
        log(f"[PREC WARN]{sym} {e}")
        PRECISION_CACHE[sym] = {"stepSize": 0.0001, "tickSize": 0.0001, "minPrice":1e-8, "maxPrice":9e8}
    return PRECISION_CACHE[sym]

def adjust_step(v, step):
    if step <= 0: return v
    return round(math.floor((v + 1e-15)/step)*step, 12)

def adjust_price(sym, p):
    f = get_symbol_filters(sym)
    tick = f["tickSize"]
    ap = adjust_step(p, tick)
    ap = max(ap, f["minPrice"])
    ap = min(ap, f["maxPrice"])
    return float(f"{ap:.12f}")

def adjust_qty(sym, q):
    f = get_symbol_filters(sym)
    return float(f"{adjust_step(q, f['stepSize']):.12f}")

def calc_order_qty(sym, entry, usd):
    if entry <= 0: return 0.0
    raw = usd / entry
    return adjust_qty(sym, raw)

def futures_get_price(sym):
    try:
        r = requests.get(BINANCE_FAPI + "/fapi/v1/ticker/price",
                         params={"symbol": sym}, timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym, it, lim):
    try:
        r = requests.get(BINANCE_FAPI + "/fapi/v1/klines",
                         params={"symbol": sym, "interval": it, "limit": lim},
                         timeout=10).json()
        if r and int(r[-1][6]) > now_ts_ms():
            r = r[:-1]
        return r
    except:
        return []

# ---------- Indicators ----------
def ema(vals, n):
    if not vals: return []
    k = 2/(n+1)
    e = [vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def sma(vals, n):
    if len(vals) < n: return [sum(vals)/len(vals)]*len(vals)
    out = []
    run = sum(vals[:n])
    out.extend([run/n]*(n))
    for i in range(n, len(vals)):
        run += vals[i] - vals[i-n]
        out.append(run/n)
    return out

def rsi(vals, period=14):
    if len(vals) < period+2: return [50]*len(vals)
    d = np.diff(vals)
    g = np.maximum(d,0); l = -np.minimum(d,0)
    ag = np.mean(g[:period]); al = np.mean(l[:period])
    out=[50]*period
    for i in range(period, len(d)):
        ag = (ag*(period-1)+g[i])/period
        al = (al*(period-1)+l[i])/period
        rs = ag/al if al>0 else 0
        out.append(100 - 100/(1+rs))
    return [50]*(len(vals)-len(out)) + out

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else:
            tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period, len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a)) + a

def stc_like(vals, length=80, fast=27, slow=50):
    # lightweight Schaff-like oscillator in [0,100]
    if len(vals) < max(length,slow)+5:
        return [50]*len(vals)
    fast_ema = ema(vals, fast)
    slow_ema = ema(vals, slow)
    macd = [f-s for f,s in zip(fast_ema, slow_ema)]
    lo = []
    for i in range(len(macd)):
        left = max(0, i-length+1)
        window = macd[left:i+1]
        mn, mx = (min(window), max(window)) if window else (0,1)
        v = 50 if mx==mn else (macd[i]-mn)/(mx-mn)*100
        lo.append(v)
    return lo

def slope(x):
    # simple last-diff slope
    if len(x) < 3: return 0.0
    return x[-2]-x[-3]

# ---------- Power (kept stable) ----------
def calc_power(e_now, e_prev, e_prev2, atr_v, price, rsi_val):
    diff = abs(e_now - e_prev) / (atr_v*0.6) if atr_v>0 else 0
    base = 55 + diff*20 + ((rsi_val-50)/50)*15 + (atr_v/max(price,1e-12))*200
    return float(min(100, max(0, base)))

def tier_from_power(p):
    if 65 <= p < 75: return "REAL", "🟩"
    if p >= 75:     return "ULTRA", "🟦"
    if p >= 60:     return "NORMAL","🟨"
    return None, ""

# ---------- Params & State ----------
PARAM_DEFAULT = {
    "TRADE_SIZE_USDT": 250.0,
    "EARLY_FAST_EMA": 3,
    "EARLY_SLOW_EMA": 7,
    "EARLY_ATR_SPIKE_RATIO": 0.08,   # (düşürüldü)
    "RSI_LEN": 14,
    "SCALP_TP_PCT_REF": 0.006,       # log amaçlı
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index": 0,
    "last_status_ts": 0,
    "last_report": 0,
    "auto_trade_active": True,
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
for k,v in STATE_DEFAULT.items():
    STATE.setdefault(k, v)

AI_SIGNALS  = safe_load(AI_SIGNALS_FILE, [])
AI_ANALYSIS = safe_load(AI_ANALYSIS_FILE, [])
AI_RL       = safe_load(AI_RL_FILE, [])
SIM_POSITIONS = safe_load(SIM_POS_FILE, [])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE, [])

# ---------- Guards & live snapshot ----------
def fetch_live_positions_snapshot():
    live = {"long":{}, "short":{}, "long_count":0, "short_count":0}
    try:
        acc = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            amt = float(p["positionAmt"]); sym = p["symbol"]
            if amt > 0:  live["long"][sym]  = amt
            elif amt < 0: live["short"][sym] = abs(amt)
        live["long_count"]  = len(live["long"])
        live["short_count"] = len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")
    STATE["auto_trade_active"] = True
    safe_save(STATE_FILE, STATE)
    return live
# =============== Strategy builders ===============

def kline_arrays(kl):
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    return closes, highs, lows

def build_signal_EARLY(sym, kl, bar_i):
    if len(kl) < 60: return None
    closes, highs, lows = kline_arrays(kl)
    fper = PARAM.get("EARLY_FAST_EMA",3)
    sper = PARAM.get("EARLY_SLOW_EMA",7)
    ema_f = ema(closes, fper)
    ema_s = ema(closes, sper)
    # confirmed cross on closed bar
    up_cross = (ema_f[-2] > ema_s[-2]) and (ema_f[-3] <= ema_s[-3])
    dn_cross = (ema_f[-2] < ema_s[-2]) and (ema_f[-3] >= ema_s[-3])
    if not (up_cross or dn_cross):
        return None
    atrs = atr_like(highs, lows, closes)
    if len(atrs) < 2: return None
    if not (atrs[-1] >= atrs[-2]*(1.0 + PARAM.get("EARLY_ATR_SPIKE_RATIO",0.08))):
        return None
    direction = "UP" if up_cross else "DOWN"
    entry = closes[-1]
    r_val = rsi(closes, PARAM.get("RSI_LEN",14))[-1]
    pwr = calc_power(ema_s[-1], ema_s[-2], ema_s[-5] if len(ema_s)>=6 else ema_s[-2],
                     atrs[-1], entry, r_val)
    # force REAL band 65–75 (ama tüm stratejiler REAL açacak; guard trade tarafında)
    tier, emoji = tier_from_power(pwr)
    return {
        "strategy":"EARLY","symbol":sym,"dir":direction,"tier":tier or "REAL",
        "emoji":"⚡️","entry":entry,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "time":now_local_iso(),"born_bar":bar_i,"early":True
    }

def build_signal_EMA200S(sym, kl, bar_i):
    if len(kl) < 220: return None
    closes, highs, lows = kline_arrays(kl)
    e200 = sma(closes, 200)
    e20  = ema(closes, 20)
    last = closes[-1]
    # Trend with 200MA as filter + 20 slope
    up = last > e200[-2] and slope(e20) > 0
    dn = last < e200[-2] and slope(e20) < 0
    if not (up or dn): return None
    direction = "UP" if up else "DOWN"
    atrs = atr_like(highs, lows, closes)
    r_val = rsi(closes, PARAM.get("RSI_LEN",14))[-1]
    pwr = calc_power(e20[-1], e20[-2], e20[-5] if len(e20)>=6 else e20[-2],
                     atrs[-1], last, r_val)
    return {
        "strategy":"EMA200S","symbol":sym,"dir":direction,"tier":"REAL",
        "emoji":"📈" if up else "📉","entry":last,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "time":now_local_iso(),"born_bar":bar_i,"early":False
    }

def ut_supertrend_like(closes, highs, lows, atr_len=10, factor=2):
    # very light supertrend approximation (dir only)
    atrs = atr_like(highs, lows, closes, period=atr_len)
    dir_arr=[0]*len(closes)
    trend_up = True
    for i in range(2, len(closes)):
        band_up   = (highs[i]+lows[i])/2 - factor*atrs[i]
        band_down = (highs[i]+lows[i])/2 + factor*atrs[i]
        # crude flip rule:
        if closes[i] > band_down: trend_up = True
        elif closes[i] < band_up: trend_up = False
        dir_arr[i] = 1 if trend_up else -1
    return dir_arr

def build_signal_UT_STC(sym, kl, bar_i):
    if len(kl) < 320: return None
    closes, highs, lows = kline_arrays(kl)
    stc = stc_like(closes, length=80, fast=27, slow=50)
    ut1 = ut_supertrend_like(closes, highs, lows, atr_len=10, factor=2)
    ut2 = ut_supertrend_like(closes, highs, lows, atr_len=300, factor=2)
    # BOTH UT filters agree (approx)
    if ut1[-2] == 1 and ut2[-2] == 1 and stc[-2] < 25 and (stc[-2]-stc[-3])>0:
        direction="UP"
    elif ut1[-2] == -1 and ut2[-2] == -1 and stc[-2] > 75 and (stc[-2]-stc[-3])<0:
        direction="DOWN"
    else:
        return None
    last = closes[-1]
    atrs = atr_like(highs, lows, closes)
    r_val = rsi(closes, PARAM.get("RSI_LEN",14))[-1]
    pwr = calc_power(closes[-1], closes[-2], closes[-5] if len(closes)>=6 else closes[-2],
                     atrs[-1], last, r_val)
    return {
        "strategy":"UT-STC","symbol":sym,"dir":direction,"tier":"REAL",
        "emoji":"🧭","entry":last,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "time":now_local_iso(),"born_bar":bar_i,"early":False
    }

def has_fvg_bullish(closes, highs, lows):
    # Simple 3-candle FVG: H[1] < L[3]
    if len(closes) < 5: return False
    h1 = highs[-3]; l3 = lows[-1]
    return h1 < l3

def has_fvg_bearish(closes, highs, lows):
    if len(closes) < 5: return False
    l1 = lows[-3]; h3 = highs[-1]
    return l1 > h3

def build_signal_A_PLUS_FVG(sym, kl, bar_i):
    if len(kl) < 30: return None
    closes, highs, lows = kline_arrays(kl)
    # trend by 50/200 EMA
    e50 = ema(closes, 50); e200 = ema(closes, 200)
    long_ok  = e50[-2] > e200[-2] and has_fvg_bullish(closes, highs, lows)
    short_ok = e50[-2] < e200[-2] and has_fvg_bearish(closes, highs, lows)
    if not (long_ok or short_ok):
        return None
    direction = "UP" if long_ok else "DOWN"
    last = closes[-1]
    atrs = atr_like(highs, lows, closes)
    r_val = rsi(closes, PARAM.get("RSI_LEN",14))[-1]
    pwr = calc_power(e50[-1], e50[-2], e50[-5] if len(e50)>=6 else e50[-2],
                     atrs[-1], last, r_val)
    return {
        "strategy":"A+FVG","symbol":sym,"dir":direction,"tier":"REAL",
        "emoji":"🅰️","entry":last,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "time":now_local_iso(),"born_bar":bar_i,"early":False
    }

# =============== TP-Only Engine (Stable) ===============

def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd, 1e-9)
    if direction == "UP":
        return entry_exec*(1+tp_pct), tp_pct
    else:
        return entry_exec*(1-tp_pct), tp_pct

def futures_set_tp_only(sym, direction, qty, entry_exec,
                        tp_low_usd=1.6, tp_high_usd=2.0):
    """
    TAKE_PROFIT_MARKET with stopPrice, workingType=MARK_PRICE, closePosition=true
    No 'price' field. (prevents code -1106)
    stopPrice>0 rounded to tickSize (prevents code -4006)
    """
    try:
        f = get_symbol_filters(sym)
        trade_usd = PARAM.get("TRADE_SIZE_USDT", 250.0)

        def place(tp_usd):
            stop_raw, tp_pct = _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd)
            stop_price = adjust_price(sym, stop_raw)
            if stop_price <= 0 or stop_price < f["minPrice"] or stop_price > f["maxPrice"]:
                log(f"[TP RANGE] {sym} skip ${tp_usd} stop={stop_price}")
                return False, None, None
            payload = {
                "symbol": sym,
                "side": "SELL" if direction=="UP" else "BUY",
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": f"{stop_price}",
                "workingType": "MARK_PRICE",
                "closePosition": "true",
                "timestamp": now_ts_ms()
            }
            # IMPORTANT: no 'price', no 'reduceOnly'
            _signed_request("POST","/fapi/v1/order", payload)
            log(f"[TP OK] {sym} TAKE_PROFIT_MARKET tp=${tp_usd} stop={stop_price} qty={qty}")
            return True, tp_usd, tp_pct

        # 0.1 scan
        cur = tp_low_usd
        while cur <= tp_high_usd + 1e-9:
            ok, u, p = place(round(cur,1))
            if ok: return True, u, p
            cur += 0.1

        # 0.01 scan
        cur = tp_low_usd
        while cur <= tp_high_usd + 1e-9:
            ok, u, p = place(round(cur,2))
            if ok: return True, u, p
            cur += 0.01

        log(f"[NO TP] {sym} 1.6–2.0$ aralığında TP yerleşmedi.")
        return False, None, None

    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False, None, None

# =============== Execution, Guards, Status ===============

def _duplicate_or_locked(sym, direction, live):
    # TrendLock
    if TREND_LOCK.get(sym) == direction:
        log(f"[TRENDLOCK HIT] {sym} {direction}")
        return True
    # Duplicate (by live positions)
    if direction=="UP" and sym in live.get("long", {}):  return True
    if direction=="DOWN" and sym in live.get("short", {}): return True
    return False

def _can_open_direction(direction, live):
    if direction=="UP" and live.get("long_count",0)  >= MAX_LONG_TRADES:  return False
    if direction=="DOWN" and live.get("short_count",0)>= MAX_SHORT_TRADES: return False
    return True

def _set_trend_lock(sym, direction):
    TREND_LOCK[sym] = direction
    TREND_LOCK_TIME[sym] = now_ts_s()
    log(f"[TRENDLOCK SET] {sym} {direction}")

def _cleanup_trend_lock_expired():
    now_s = now_ts_s()
    expired = [sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        TREND_LOCK.pop(sym, None)
        TREND_LOCK_TIME.pop(sym, None)
        log(f"[TRENDLOCK TIMEOUT] {sym} (6h)")

def open_market_position(sym, direction, qty):
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res = _signed_request("POST","/fapi/v1/order",{
        "symbol": sym, "side": side, "type":"MARKET",
        "quantity": f"{qty}", "positionSide": pos_side,
        "timestamp": now_ts_ms()
    })
    fill = res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    return {"symbol": sym, "dir": direction, "qty": qty, "entry": float(fill), "pos_side": pos_side}

def ai_log_signal(sig):
    AI_SIGNALS.append({
        "time":now_local_iso(),"symbol":sig["symbol"],"dir":sig["dir"],"tier":sig.get("tier","REAL"),
        "power":sig.get("power"),"rsi":sig.get("rsi"),"atr":sig.get("atr"),
        "entry":sig.get("entry"),"born_bar":sig.get("born_bar"),
        "strategy":sig.get("strategy"),"early":bool(sig.get("early",False))
    })
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)

def rl_log_trade(sym, direction, entry_exec, tp_ok, tp_usd_used, tp_pct_used, pwr, born_bar, strat, early):
    AI_RL.append({
        "time":now_local_iso(),"symbol":sym,"dir":direction,"entry":entry_exec,
        "tp_usd_used":tp_usd_used,"tp_pct_used":tp_pct_used,"tp_ok":tp_ok,
        "power":pwr,"born_bar":born_bar,"strategy":strat,"early":early
    })
    safe_save(AI_RL_FILE, AI_RL)

def status_to_telegram(live, last_trade_text=""):
    msg = (f"🧭 STATUS\n"
           f"Long = {live.get('long_count',0)} / {MAX_LONG_TRADES}\n"
           f"Short = {live.get('short_count',0)} / {MAX_SHORT_TRADES}\n")
    if last_trade_text:
        msg += f"Last = {last_trade_text}\n"
    total = live.get('long_count',0) + live.get('short_count',0)
    msg += f"Total = {total} / {MAX_LONG_TRADES+MAX_SHORT_TRADES}"
    tg_send(msg); log(msg)

def execute_real_trade(sig, live):
    sym = sig["symbol"]; direction = sig["dir"]; pwr = sig.get("power",0.0)
    # open guard
    if not _can_open_direction(direction, live): return False
    if _duplicate_or_locked(sym, direction, live): return False

    qty = calc_order_qty(sym, sig["entry"], PARAM.get("TRADE_SIZE_USDT",250.0))
    if qty <= 0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return False

    opened = open_market_position(sym, direction, qty)
    entry_exec = opened.get("entry") or futures_get_price(sym)
    if not entry_exec or entry_exec <= 0:
        log(f"[OPEN FAIL] {sym} entry alınamadı."); return False

    tp_ok, tp_usd_used, tp_pct_used = futures_set_tp_only(
        sym, direction, qty, entry_exec, tp_low_usd=1.6, tp_high_usd=2.0
    )

    _set_trend_lock(sym, direction)

    # Telegram — only real fill
    prefix = f"✅ REAL {sig.get('strategy','?')}"
    if tp_ok:
        tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                f"Power:{pwr:.2f}\nEntry:{entry_exec:.12f}\n"
                f"TP hedefi:{(tp_usd_used or 0):.2f}$ ({(tp_pct_used or 0)*100:.3f}%)\n"
                f"time:{now_local_iso()}")
        last_txt = f"{sym} {direction} @ {entry_exec:.6f} ({(tp_usd_used or 0):.2f}$ TP)"
    else:
        tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                f"Power:{pwr:.2f}\nEntry:{entry_exec:.12f}\n"
                f"TP: YOK (1.6–2.0$ tarama başarısız)\n"
                f"time:{now_local_iso()}")
        last_txt = f"{sym} {direction} @ {entry_exec:.6f} (TP yok)"

    # RL log
    rl_log_trade(sym, direction, entry_exec, tp_ok, tp_usd_used, tp_pct_used,
                 pwr, sig.get("born_bar"), sig.get("strategy"), sig.get("early",False))

    # refresh live & status
    live2 = fetch_live_positions_snapshot()
    status_to_telegram(live2, last_txt)
    return True

# =============== SIM Side (parallel bookkeeping) ===============

def queue_sim_variants(sig):
    # keep same delayed approve variants for analysis
    now_s = now_ts_s()
    for secs, label, mins in [(30*60,"approve_30m",30),(60*60,"approve_1h",60),
                              (90*60,"approve_1h30",90),(120*60,"approve_2h",120)]:
        SIM_POSITIONS.append({
            "symbol":sig["symbol"], "dir":sig["dir"], "tier":sig.get("tier","REAL"),
            "entry":sig["entry"], "power":sig.get("power"), "rsi":sig.get("rsi"),
            "atr":sig.get("atr"), "strategy":sig.get("strategy"),
            "created_ts":now_s, "open_after_ts":now_s+secs,
            "approve_delay_min":mins, "approve_label":label,
            "status":"PENDING", "early":bool(sig.get("early",False))
        })
    safe_save(SIM_POS_FILE, SIM_POSITIONS)

def process_sim_queue_and_open_due():
    if not SIM_POSITIONS: return
    now_s = now_ts_s()
    changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")=="PENDING" and pos["open_after_ts"]<=now_s:
            pos["status"]="OPEN"; pos["open_ts"]=now_s; pos["open_time"]=now_local_iso(); changed=True
            log(f"[SIM OPEN] {pos['symbol']} {pos['dir']} approve={pos['approve_delay_min']}m early={pos.get('early')}")
    if changed: safe_save(SIM_POS_FILE, SIM_POSITIONS)

def process_sim_closes():
    if not SIM_POSITIONS: return
    still=[]; changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")!="OPEN":
            still.append(pos); continue
        last=futures_get_price(pos["symbol"])
        if last is None: still.append(pos); continue
        hit=None
        if pos["dir"]=="UP":
            # sim TP: referans olarak entry * (1 + 0.006)
            if last >= pos["entry"]*(1+PARAM.get("SCALP_TP_PCT_REF",0.006)): hit="TP"
        else:
            if last <= pos["entry"]*(1-PARAM.get("SCALP_TP_PCT_REF",0.006)): hit="TP"
        if hit:
            close_time=now_local_iso()
            gain_pct = ((last/pos["entry"]-1.0)*100.0 if pos["dir"]=="UP"
                        else (pos["entry"]/last-1.0)*100.0)
            SIM_CLOSED.append({
                **pos, "status":"CLOSED", "close_time":close_time,
                "exit_price": last, "exit_reason": hit, "gain_pct": gain_pct
            })
            # unlock trend when closed
            TREND_LOCK.pop(pos["symbol"], None)
            TREND_LOCK_TIME.pop(pos["symbol"], None)
            changed=True
            log(f"[SIM CLOSE] {pos['symbol']} {pos['dir']} {hit} {gain_pct:.3f}% approve={pos.get('approve_delay_min')}m")
        else:
            still.append(pos)
    if changed:
        # keep OPEN + PENDING + others
        SIM_POSITIONS[:] = still
        safe_save(SIM_POS_FILE, SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE, SIM_CLOSED)
# =============== Scan orchestration ===============

def scan_symbol(sym, bar_i):
    kl_h1 = futures_get_klines(sym, "1h", 300)
    if len(kl_h1) < 60: return []
    res=[]
    s1 = build_signal_EARLY(sym, kl_h1, bar_i)
    if s1: res.append(s1)
    s2 = build_signal_EMA200S(sym, kl_h1, bar_i)
    if s2: res.append(s2)

    # UT-STC & A+FVG prefer H1 too (can be 5m+ but standardize)
    s3 = build_signal_UT_STC(sym, kl_h1, bar_i)
    if s3: res.append(s3)
    s4 = build_signal_A_PLUS_FVG(sym, kl_h1, bar_i)
    if s4: res.append(s4)
    return res

def run_parallel(symbols, bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(scan_symbol, s, bar_i) for s in symbols]
        for f in as_completed(futs):
            try: sigs=f.result()
            except: sigs=[]
            if sigs: out.extend(sigs)
    return out

# =============== Status heartbeat scheduler ===============
def heartbeat_if_due():
    now_s = now_ts_s()
    if now_s - STATE.get("last_status_ts", 0) >= STATUS_INTERVAL_SEC:
        live = fetch_live_positions_snapshot()
        status_to_telegram(live, "")
        STATE["last_status_ts"] = now_s
        safe_save(STATE_FILE, STATE)

# =============== MAIN LOOP ===============

def main():
    tg_send("🚀 EMA ULTRA v15.9.42 aktif (ALL-REAL, 30/30 Guard, Stable TP)")
    log("[START] EMA ULTRA v15.9.42")

    # universe: USDT pairs
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo", timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}")
        symbols=[]
    symbols.sort()

    while True:
        try:
            # bar tick
            STATE["bar_index"] = STATE.get("bar_index",0)+1
            bar_i = STATE["bar_index"]

            # 1) scan
            sigs = run_parallel(symbols, bar_i)

            # 2) process signals
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)     # sim bookkeeping

                # live snapshot for guards
                live = fetch_live_positions_snapshot()

                # ALL strategies open REAL (subject to 30/30 & guards)
                execute_real_trade(sig, live)

            # 3) SIM queue & closes
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4) Heartbeat status (5m)
            heartbeat_if_due()

            # 5) TrendLock expiry
            _cleanup_trend_lock_expired()

            # 6) persist
            safe_save(STATE_FILE, STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# =============== ENTRYPOINT ===============
if __name__ == "__main__":
    main()