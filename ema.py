# ema.py
# ==============================================================================
# 📘 EMA ULTRA v15.9.48 — FULL (PEMA removed, New Strategies + Smart TP v42 style)
#  - Strategies (ALL REAL + SIM log): EARLY, UT/STC, EMA20/200+MACD, FVG
#  - TP only (no SL). Binance Futures TAKE_PROFIT_MARKET with closePosition=true
#  - Smart TP: USD 1.6–2.0 scan + micro-price % fallback + stop>0 guard
#  - No reduceOnly (by request)
#  - TrendLock 6h (per symbol & direction) — cleans on close & auto-timeout 6h
#  - Directional max open count: 30 (BUY vs SELL tracked separately)
#  - Heartbeat 10m, Auto-Backup 4h, Telegram: /status /report /export /set
# ==============================================================================

import os, json, time, math, hmac, hashlib, threading, random
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta

import requests
import numpy as np

# --------------------- ENV / PATHS ---------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE        = os.path.join(DATA_DIR, "state.json")
PARAM_FILE        = os.path.join(DATA_DIR, "params.json")
AI_SIGNALS_FILE   = os.path.join(DATA_DIR, "ai_signals.json")
AI_ANALYSIS_FILE  = os.path.join(DATA_DIR, "ai_analysis.json")
SIM_POS_FILE      = os.path.join(DATA_DIR, "sim_positions.json")
SIM_CLOSED_FILE   = os.path.join(DATA_DIR, "sim_closed.json")
LOG_FILE          = os.path.join(DATA_DIR, "log.txt")
BACKUP_DIR        = os.path.join(DATA_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

# --------------------- TELEGRAM ------------------------
BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TG_CHAT_ID", "")
TG_API    = f"https://api.telegram.org/bot{BOT_TOKEN}"

# --------------------- BINANCE -------------------------
BINANCE_FAPI = "https://fapi.binance.com"
API_KEY    = os.getenv("BINANCE_FUTURES_KEY", "")
API_SECRET = os.getenv("BINANCE_FUTURES_SECRET", "")

# --------------------- GLOBALS -------------------------
getcontext().prec = 28

PARAM = {
    "INTERVAL": "1h",
    "SCAN_LIMIT": 200,
    "HEARTBEAT_MIN": 10,
    "BACKUP_HOURS": 4,

    # EARLY
    "EARLY_FAST": 3,
    "EARLY_SLOW": 7,
    "EARLY_ATR_SPIKE_RATIO": 0.03,

    # UT/STC
    "UT_KEY": 2.0,
    "UT_ATR1": 1.0,
    "UT_ATR_LONG": 300.0,   # UT #2 için "çok geniş" varyant
    "STC_LEN": 80,
    "STC_FAST": 27,
    "STC_SLOW": 50,
    "STC_GREEN": 25.0,
    "STC_RED": 75.0,

    # EMA20/200 + MACD
    "EMA_FAST": 12, "EMA_SLOW": 26, "MACD_SIGNAL": 9,
    "EMA_TREND_FAST": 20, "EMA_TREND_SLOW": 200,

    # FVG
    "FVG_LOOKBACK": 4,   # 3-bar FVG araması, +1 tolerans

    # POWER band (sende real filtre yok istedin; ama sinyal kalitesi için rapora yazıyoruz)
    "POWER_MIN": 0.0,

    # TP — v42 uyumlu davranış
    "USD_TP_MIN": 1.6,
    "USD_TP_MAX": 2.0,
    "TP_CANDIDATES": [1.6, 1.7, 1.8, 1.9, 2.0],
    "TP_PCT_FALLBACK": 0.006,   # %0.6 gibi mikro çiftlerde fallback

    # Max open per direction
    "MAX_OPEN_PER_DIR": 30,

    # TrendLock
    "TRENDLOCK_SEC": 6 * 3600,

    # Approve bars (0 = devre dışı, real doğrudan)
    "APPROVE_BARS": 0,

    # Qty/position sizing (örnek)
    "BASE_QTY_USDT": 250.0,  # sabit notional
    "LEVERAGE": 5,

    # Symbols (boşsa auto-init)
    "SYMBOLS": ""
}

STATE = {
    "bar_index": 0,
    "last_heartbeat": 0,
    "last_backup": 0,
    "trendlocks": {},                 # { "SYMBOL:UP": ts, "SYMBOL:DOWN": ts }
    "dir_open_count": {},             # { "SYMBOL:UP": int, "SYMBOL:DOWN": int }
    "filters_cache": {},              # exchangeInfo cache
    "open_positions": {}              # { "SYMBOL": {"side":"BUY/SELL","entry":..., "qty":...} } (quick map)
}
POSITION_MODE_CACHE = {"dual": None, "ts": 0}
AI_SIGNALS = []     # son sinyaller
AI_ANALYSIS = {}    # özet/istatistik
SIM_POSITIONS = []  # sim açık
SIM_CLOSED = []     # sim kapalı

# --------------------- UTILS ---------------------------
def now_ts():
    return int(time.time())

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def log(msg):
    line = f"{now_iso()} | {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def ensure_loaded():
    global PARAM, STATE, AI_SIGNALS, AI_ANALYSIS, SIM_POSITIONS, SIM_CLOSED
    PARAM = load_json(PARAM_FILE, PARAM)
    STATE = load_json(STATE_FILE, STATE)
    AI_SIGNALS = load_json(AI_SIGNALS_FILE, [])
    AI_ANALYSIS= load_json(AI_ANALYSIS_FILE, {})
    SIM_POSITIONS= load_json(SIM_POS_FILE, [])
    SIM_CLOSED   = load_json(SIM_CLOSED_FILE, [])

def persist_all():
    save_json(PARAM_FILE, PARAM)
    save_json(STATE_FILE, STATE)
    save_json(AI_SIGNALS_FILE, AI_SIGNALS)
    save_json(AI_ANALYSIS_FILE, AI_ANALYSIS)
    save_json(SIM_POS_FILE, SIM_POSITIONS)
    save_json(SIM_CLOSED_FILE, SIM_CLOSED)

# --------------------- TELEGRAM -------------------------
def tg_send(text):
    if not BOT_TOKEN or not CHAT_ID: 
        return
    try:
        requests.post(f"{TG_API}/sendMessage", json={"chat_id": CHAT_ID, "text": text}, timeout=7)
    except Exception as e:
        log(f"[TG ERR]{e}")

def _tg_get_updates(offset=None):
    if not BOT_TOKEN: return []
    try:
        params = {"timeout": 0}
        if offset is not None: params["offset"] = offset
        r = requests.get(f"{TG_API}/getUpdates", params=params, timeout=7).json()
        return r.get("result", [])
    except Exception as e:
        log(f"[TG getUpdates ERR]{e}")
        return []

def _tg_set_offset(offset):
    STATE["tg_offset"] = offset
    save_json(STATE_FILE, STATE)

# --------------------- BINANCE AUTH ---------------------
def _sign(params: dict):
    query = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + f"&signature={sig}"

def _headers():
    return {"X-MBX-APIKEY": API_KEY}

# --------------------- BINANCE HELPERS ------------------
def fapi_get(path, params=None, timeout=7):
    url = f"{BINANCE_FAPI}{path}"
    try:
        r = requests.get(url, params=params or {}, headers=_headers(), timeout=timeout)
        return r.json()
    except Exception as e:
        log(f"[GET ERR]{path} {e}")
        return {}

def fapi_signed(method, path, params=None, timeout=7):
    url = f"{BINANCE_FAPI}{path}"
    p = params.copy() if params else {}
    p["timestamp"] = int(time.time() * 1000)
    data = _sign(p)
    try:
        if method == "GET":
            r = requests.get(url, headers=_headers(), params=p, timeout=timeout)
        elif method == "POST":
            r = requests.post(url, headers=_headers(), data=data, timeout=timeout)
        elif method == "DELETE":
            r = requests.delete(url, headers=_headers(), data=data, timeout=timeout)
        else:
            raise ValueError("bad method")
        return r.json()
    except Exception as e:
        log(f"[SIGNED {method} ERR]{path} {e}")
        return {}

def futures_get_klines(symbol, interval, limit):
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines",
                         params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=7)
        return r.json()
    except Exception as e:
        log(f"[KLINES ERR]{symbol} {e}")
        return []

def get_24h(symbol):
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr",
                         params={"symbol":symbol}, timeout=5).json()
        return float(r.get("priceChangePercent", 0.0))
    except:
        return 0.0

def get_filters(symbol):
    # cache exchangeInfo filters
    cache = STATE["filters_cache"].get(symbol)
    if cache: return cache
    info = fapi_get("/fapi/v1/exchangeInfo")
    for s in info.get("symbols", []):
        if s.get("symbol") == symbol:
            filters = {f["filterType"]: f for f in s.get("filters", [])}
            STATE["filters_cache"][symbol] = filters
            save_json(STATE_FILE, STATE)
            return filters
    return {}

def tick_round(value, tick_size):
    if tick_size <= 0: return value
    q = Decimal(value) / Decimal(tick_size)
    return float((q.to_integral_value(rounding="ROUND_FLOOR")) * Decimal(tick_size))

def price_round_to_filters(symbol, px):
    fs = get_filters(symbol)
    minPrice = float(fs.get("PRICE_FILTER", {}).get("minPrice", "0") or "0")
    tickSize = float(fs.get("PRICE_FILTER", {}).get("tickSize", "0") or "0")
    if tickSize > 0:
        px = tick_round(px, tickSize)
    if minPrice > 0 and px < minPrice:
        px = minPrice
    return px

def lot_round_to_filters(symbol, qty):
    fs = get_filters(symbol)
    stepSize = float(fs.get("LOT_SIZE", {}).get("stepSize", "0") or "0")
    minQty   = float(fs.get("LOT_SIZE", {}).get("minQty", "0") or "0")
    if stepSize > 0:
        q = Decimal(qty) / Decimal(stepSize)
        qty = float((q.to_integral_value(rounding="ROUND_FLOOR")) * Decimal(stepSize))
    if minQty > 0 and qty < minQty:
        qty = minQty
    return qty
def is_dual_position_mode(force_refresh=False):
    now = now_ts()
    if not force_refresh and POSITION_MODE_CACHE["dual"] is not None:
        # refresh every 10 minutes
        if now - POSITION_MODE_CACHE["ts"] < 600:
            return POSITION_MODE_CACHE["dual"]
    try:
        r = fapi_signed("GET", "/fapi/v1/positionSide/dual")
        dual = str(r.get("dualSidePosition", "false")).lower() == "true"
        POSITION_MODE_CACHE["dual"] = dual
        POSITION_MODE_CACHE["ts"] = now
        return dual
    except Exception as e:
        log(f"[POSITION MODE ERR]{e}")
        # preserve previous cache; default to False if unknown
        if POSITION_MODE_CACHE["dual"] is None:
            POSITION_MODE_CACHE["dual"] = False
        return POSITION_MODE_CACHE["dual"]

def maybe_apply_position_side(params, side):
    """Append positionSide when account is in dual-side (hedge) mode."""
    try:
        if is_dual_position_mode():
            params["positionSide"] = "LONG" if side == "BUY" else "SHORT"
    except Exception as e:
        log(f"[POSITION SIDE WARN]{e}")
    return params
def mark_price(symbol):
    try:
        r = fapi_get("/fapi/v1/premiumIndex", {"symbol":symbol})
        return float(r.get("markPrice"))
    except:
        return None

# --------------------- INDICATORS -----------------------
def ema(arr, length):
    if length <= 1: return arr[:]
    out = []
    k = 2 / (length + 1)
    prev = arr[0]
    out.append(prev)
    for i in range(1, len(arr)):
        v = arr[i] * k + prev * (1 - k)
        out.append(v)
        prev = v
    return out

def rsi(close, length=14):
    if len(close) < length + 1: 
        return [50.0]*len(close)
    gains = [0.0]
    losses= [0.0]
    for i in range(1,len(close)):
        diff = close[i] - close[i-1]
        gains.append(max(diff,0.0))
        losses.append(max(-diff,0.0))
    avg_gain = sum(gains[1:length+1]) / length
    avg_loss = sum(losses[1:length+1]) / length
    rsis = [50.0]*(length)
    for i in range(length+1, len(close)+1):
        gain = gains[i-1]
        loss = losses[i-1]
        avg_gain = (avg_gain*(length-1) + gain)/length
        avg_loss = (avg_loss*(length-1) + loss)/length
        rs  = (avg_gain/avg_loss) if avg_loss>0 else 999999
        rsi_val = 100 - (100/(1+rs))
        rsis.append(rsi_val)
    return rsis[:len(close)]

def atr_like(high, low, close, length=14):
    if len(close) < 2: return [0.0]*len(close)
    trs = [0.0]
    for i in range(1,len(close)):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        trs.append(tr)
    # RMA
    out=[]
    alpha = 1/length
    prev = sum(trs[1:length+1])/length if len(trs)>length else sum(trs)/max(1,len(trs))
    out = [0.0]*length + [prev]
    for i in range(length+1, len(trs)):
        prev = (1-alpha)*prev + alpha*trs[i]
        out.append(prev)
    if len(out) < len(trs):
        out += [prev]*(len(trs)-len(out))
    return out[:len(close)]

def macd_line(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = [ema_fast[i]-ema_slow[i] for i in range(len(close))]
    signal_line = ema(macd, signal)
    hist = [macd[i]-signal_line[i] for i in range(len(close))]
    return macd, signal_line, hist

def stc(close, fast=27, slow=50, length=80):
    # basit STC benzeri: MACD -> RSI of MACD -> stochastic benzeri normalize
    macd_v, sig, _ = macd_line(close, fast, slow, 9)
    # normalize macd to 0-100 using a rolling window 'length'
    out=[50.0]*len(close)
    for i in range(length, len(close)):
        win = macd_v[i-length+1:i+1]
        mn, mx = min(win), max(win)
        if mx>mn:
            out[i] = (macd_v[i]-mn)/(mx-mn)*100.0
        else:
            out[i] = 50.0
    return out

# --------------------- SMART TP v42 STYLE ---------------
def _tp_candidates_usd():
    # 1.6 → 2.0 USD aralığı + klasik listeden
    s = set(PARAM.get("TP_CANDIDATES", [1.6,1.7,1.8,1.9,2.0]))
    mn = float(PARAM.get("USD_TP_MIN", 1.6))
    mx = float(PARAM.get("USD_TP_MAX", 2.0))
    x = mn
    while x <= mx+1e-9:
        s.add(round(x, 2))
        x += 0.1
    return sorted(s)

def _calc_notional_qty(symbol, entry):
    # Sabit USDT notional / kaldıraç'a göre positionQty:
    base = float(PARAM.get("BASE_QTY_USDT", 250.0))
    lev  = max(1, int(PARAM.get("LEVERAGE", 5)))
    # Binance: position notional = qty * price; kaldıraç marjin gereksinimi, qty hesabını etkilemez
    qty = base / entry
    qty = lot_round_to_filters(symbol, qty)
    return qty

def build_tp_stop(symbol, side, entry_price):
    """
    v42’de sorunsuz çalışan kurgu:
      - TAKE_PROFIT_MARKET + closePosition=true
      - 'price' GÖNDERME! (-1106 hatasını önler)
      - stopPrice = entry ± (USD hedef / qty)  --> fiyat adımına yuvarla
      - stopPrice > 0 guard
      - mikro fiyatlarda % fallback (TP_PCT_FALLBACK)
    """
    entry = float(entry_price)
    qty   = _calc_notional_qty(symbol, entry)
    if qty <= 0:
        return None  # qty hesaplanamadı

    # USD aralığında tarama
    marks = mark_price(symbol) or entry
    fs = get_filters(symbol)
    tick = float(fs.get("PRICE_FILTER", {}).get("tickSize", "0") or "0")
    minP = float(fs.get("PRICE_FILTER", {}).get("minPrice", "0") or "0")

    # fiyat başına 1 USD kazanç için gereken fark (qty*Δprice ≈ USD)
    # Δprice ≈ USD / qty
    for usd in _tp_candidates_usd():
        dpx = usd / max(qty, 1e-12)
        if side == "BUY":
            tp_price = entry + dpx
        else:
            tp_price = entry - dpx
        tp_price = price_round_to_filters(symbol, tp_price)

        if tp_price and tp_price > 0 and (minP == 0 or tp_price >= minP):
            return {
                "stopPrice": tp_price,   # TAKE_PROFIT_MARKET 'stopPrice'
                "closePosition": True
            }

    # Fallback: % TP (mikro çiftlerde)
    pct = float(PARAM.get("TP_PCT_FALLBACK", 0.006))
    if side == "BUY":
        tp_price = entry * (1+pct)
    else:
        tp_price = entry * (1-pct)
    tp_price = price_round_to_filters(symbol, tp_price)
    if tp_price <= 0:
        return None

    return {
        "stopPrice": tp_price,
        "closePosition": True
    }
# ===================== SIGNAL BUILDERS =====================
def _extract_ohlc(kl):
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    opens =[float(k[1]) for k in kl]
    return opens, highs, lows, closes

def build_signal_early(symbol, kl, bar_i):
    if len(kl) < 60: return None
    try:
        chg = get_24h(symbol)
    except:
        chg = 0.0
    opens, highs, lows, closes = _extract_ohlc(kl)
    e_fast = ema(closes, int(PARAM["EARLY_FAST"]))
    e_slow = ema(closes, int(PARAM["EARLY_SLOW"]))
    atr_v  = atr_like(highs, lows, closes)[-1]

    cross_up   = e_fast[-2] < e_slow[-2] and e_fast[-1] > e_slow[-1]
    cross_down = e_fast[-2] > e_slow[-2] and e_fast[-1] < e_slow[-1]

    # ATR spike (son bar TR / ortalama ATR ~) basit: son bar range / atr
    last_range = highs[-1]-lows[-1]
    spike_ok   = atr_v>0 and (last_range/atr_v) >= (1.0 + float(PARAM["EARLY_ATR_SPIKE_RATIO"]))

    if spike_ok and cross_up:
        direction="UP"
    elif spike_ok and cross_down:
        direction="DOWN"
    else:
        return None

    entry = closes[-1]
    return {
        "symbol": symbol, "dir": direction, "entry": entry,
        "kind": "EARLY", "emoji": "🟩", "power": 0.0, "rsi": 0.0, "atr": atr_v,
        "chg24h": chg, "time": now_iso(), "born_bar": bar_i, "tag": "EARLY"
    }

def _ut_signal(opens, highs, lows, closes, key=2.0, atr_mult=1.0):
    # ta.supertrend benzeri bir zarf: basic bands via ATR
    atr_v = atr_like(highs, lows, closes)
    mid = ema(closes, 10)
    up  = [mid[i] + atr_mult*atr_v[i]*key for i in range(len(closes))]
    dn  = [mid[i] - atr_mult*atr_v[i]*key for i in range(len(closes))]
    # buy cuando cierra encima de 'up' o cruza 'dn'? Basit kural:
    buy  = closes[-2] <= up[-2] and closes[-1] > up[-1]
    sell = closes[-2] >= dn[-2] and closes[-1] < dn[-1]
    if buy:  return "BUY"
    if sell: return "SELL"
    return None

def build_signal_ut_stc(symbol, kl, bar_i):
    if len(kl)<120: return None
    opens, highs, lows, closes = _extract_ohlc(kl)
    # UT #1 (dar), UT #2 (çok geniş) — biri sinyal verirse
    ut1 = _ut_signal(opens, highs, lows, closes, key=float(PARAM["UT_KEY"]), atr_mult=float(PARAM["UT_ATR1"]))
    ut2 = _ut_signal(opens, highs, lows, closes, key=float(PARAM["UT_KEY"]), atr_mult=float(PARAM["UT_ATR_LONG"]))
    ut  = ut1 or ut2
    if not ut: return None

    stc_val = stc(closes, fast=int(PARAM["STC_FAST"]), slow=int(PARAM["STC_SLOW"]), length=int(PARAM["STC_LEN"]))[-1]
    stc_prev= stc(closes, fast=int(PARAM["STC_FAST"]), slow=int(PARAM["STC_SLOW"]), length=int(PARAM["STC_LEN"]))[-2]
    up_slope   = stc_val > stc_prev
    down_slope = stc_val < stc_prev

    direction=None
    if ut=="BUY" and stc_val < float(PARAM["STC_GREEN"]) and up_slope:
        direction="UP"
    elif ut=="SELL" and stc_val > float(PARAM["STC_RED"]) and down_slope:
        direction="DOWN"
    else:
        return None

    entry = closes[-1]
    return {
        "symbol": symbol, "dir": direction, "entry": entry,
        "kind": "UT_STC", "emoji": "🟩", "power": 0.0, "rsi": 0.0,
        "atr": atr_like(highs, lows, closes)[-1],
        "chg24h": get_24h(symbol), "time": now_iso(), "born_bar": bar_i, "tag": "UT/STC"
    }

def build_signal_ema_macd(symbol, kl, bar_i):
    if len(kl)<200: return None
    opens, highs, lows, closes = _extract_ohlc(kl)
    e20  = ema(closes, int(PARAM["EMA_TREND_FAST"]))
    e200 = ema(closes, int(PARAM["EMA_TREND_SLOW"]))
    macd_v, sig, hist = macd_line(closes, fast=int(PARAM["EMA_FAST"]), slow=int(PARAM["EMA_SLOW"]), signal=int(PARAM["MACD_SIGNAL"]))
    up_trend   = e20[-1] > e200[-1]
    down_trend = e20[-1] < e200[-1]
    macd_up    = macd_v[-1] > sig[-1]
    macd_down  = macd_v[-1] < sig[-1]

    direction=None
    if up_trend and macd_up:
        direction="UP"
    elif down_trend and macd_down:
        direction="DOWN"
    else:
        return None

    entry=closes[-1]
    return {
        "symbol":symbol, "dir":direction, "entry":entry,
        "kind":"EMA_MACD", "emoji":"🟩", "power":0.0, "rsi":rsi(closes)[-1],
        "atr": atr_like(highs, lows, closes)[-1],
        "chg24h": get_24h(symbol), "time": now_iso(), "born_bar": bar_i, "tag":"EMA20/200+MACD"
    }

def _find_fvg(closes, highs, lows, lookback=4):
    # Basit 3-bar FVG: bull için L1 > H3, bear için H1 < L3 (index -3,-2,-1)
    i = len(closes)-1
    for offs in range(0, lookback):
        i3 = i - offs
        if i3-2 < 0: break
        h1, l1 = highs[i3-2], lows[i3-2]
        h3, l3 = highs[i3],   lows[i3]
        # Bull FVG: l1 > h3 (gap yukarı)
        if l1 > h3:
            return ("UP", (h3, l1))  # gap bandı (lower, upper)
        # Bear FVG: h1 < l3 (gap aşağı)
        if h1 < l3:
            return ("DOWN", (h1, l3))
    return (None, None)

def build_signal_fvg(symbol, kl, bar_i):
    if len(kl)<30: return None
    opens, highs, lows, closes = _extract_ohlc(kl)
    direction, band = _find_fvg(closes, highs, lows, lookback=int(PARAM["FVG_LOOKBACK"]))
    if not direction: return None
    entry = closes[-1]

    # basit teyit: fiyat gap yönünde kapanış yapsın
    if direction=="UP" and closes[-1] <= band[0]:
        return None
    if direction=="DOWN" and closes[-1] >= band[1]:
        return None

    return {
        "symbol":symbol, "dir":direction, "entry":entry,
        "kind":"FVG", "emoji":"🟩", "power":0.0, "rsi":rsi(closes)[-1],
        "atr": atr_like(highs, lows, closes)[-1],
        "chg24h": get_24h(symbol), "time": now_iso(), "born_bar": bar_i, "tag":"FVG"
    }

def scan_symbol(symbol, bar_i):
    kl = futures_get_klines(symbol, PARAM["INTERVAL"], PARAM["SCAN_LIMIT"])
    if not isinstance(kl, list) or len(kl) < 60:
        return []
    out=[]
    s1 = build_signal_early(symbol, kl, bar_i)
    s2 = build_signal_ut_stc(symbol, kl, bar_i)
    s3 = build_signal_ema_macd(symbol, kl, bar_i)
    s4 = build_signal_fvg(symbol, kl, bar_i)
    for s in (s1, s2, s3, s4):
        if s: out.append(s)
    return out

# ===================== GUARDS / TRENDLOCK / COUNTS ======
def _key_dir(symbol, direction):
    return f"{symbol}:{direction}"

def _trendlock_active(symbol, direction):
    k = _key_dir(symbol, direction)
    ts = STATE["trendlocks"].get(k)
    if not ts: return False
    # timeout
    if now_ts() - ts > int(PARAM["TRENDLOCK_SEC"]):
        del STATE["trendlocks"][k]
        save_json(STATE_FILE, STATE)
        return False
    return True

def _trendlock_set(symbol, direction):
    STATE["trendlocks"][_key_dir(symbol, direction)] = now_ts()
    save_json(STATE_FILE, STATE)

def _dir_open_count(symbol, direction):
    return int(STATE["dir_open_count"].get(_key_dir(symbol, direction), 0))

def _dir_inc(symbol, direction):
    k = _key_dir(symbol, direction)
    STATE["dir_open_count"][k] = _dir_open_count(symbol, direction) + 1
    save_json(STATE_FILE, STATE)

def _dir_dec(symbol, direction):
    k = _key_dir(symbol, direction)
    c = max(0, _dir_open_count(symbol, direction)-1)
    STATE["dir_open_count"][k] = c
    save_json(STATE_FILE, STATE)

def _can_open(symbol, direction):
    # TrendLock engeli
    if _trendlock_active(symbol, direction):
        log(f"[TRENDLOCK HIT] {symbol} {direction}")
        return False
    # Max open per direction
    if _dir_open_count(symbol, direction) >= int(PARAM["MAX_OPEN_PER_DIR"]):
        log(f"[MAX DIR OPEN HIT] {symbol} {direction}")
        return False
    return True

# ===================== SIM ==============================
def sim_open(sig):
    pos = {
        "symbol": sig["symbol"],
        "side":   "BUY" if sig["dir"]=="UP" else "SELL",
        "entry":  sig["entry"],
        "qty":    _calc_notional_qty(sig["symbol"], sig["entry"]),
        "time":   now_iso(),
        "status": "OPEN",
        "kind":   sig["kind"],
        "tag":    sig["tag"]
    }
    SIM_POSITIONS.append(pos)
    save_json(SIM_POS_FILE, SIM_POSITIONS)
    return pos

def sim_close(symbol, side, reason="TP"):
    # basit: ilk açık eşleşeni kapat
    for p in SIM_POSITIONS:
        if p["symbol"]==symbol and p["status"]=="OPEN" and p["side"]==side:
            p["status"]="CLOSED"
            p["close_time"]=now_iso()
            SIM_CLOSED.append(p.copy())
            save_json(SIM_CLOSED_FILE, SIM_CLOSED)
            save_json(SIM_POS_FILE, SIM_POSITIONS)
            return True
    return False

# ===================== TELEGRAM CMDS =====================
def _cmd_status():
    lines=[]
    lines.append("📊 STATUS")
    lines.append(f"bar_index={STATE.get('bar_index',0)}  heartbeat_min={PARAM['HEARTBEAT_MIN']}")
    lines.append(f"TrendLocks: {len(STATE['trendlocks'])}")
    # Direction counts
    if STATE["dir_open_count"]:
        for k,v in STATE["dir_open_count"].items():
            lines.append(f"{k} → {v}")
    # Open (quick)
    lines.append("Open map: " + json.dumps(STATE.get("open_positions", {}))[:300])
    tg_send("\n".join(lines))

def _cmd_report():
    wins = len([x for x in SIM_CLOSED if x.get("reason","TP")=="TP"])
    total= len(SIM_CLOSED)
    tg_send(f"📈 SIM Closed: {total} (TP={wins})")

def _cmd_set(args):
    # /set KEY VALUE
    if len(args) < 2:
        tg_send("Kullanım: /set KEY VALUE")
        return
    key = args[0]
    val = " ".join(args[1:])
    # numara gibi parse
    try:
        if "." in val:
            PARAM[key]=float(val)
        else:
            PARAM[key]=int(val)
    except:
        PARAM[key]=val
    save_json(PARAM_FILE, PARAM)
    tg_send(f"Set {key}={PARAM[key]}")

def _cmd_export():
    # Basit export özet
    pth = os.path.join(BACKUP_DIR, f"export_{int(time.time())}.json")
    blob = {
        "state": STATE, "param": PARAM,
        "sim_open": SIM_POSITIONS, "sim_closed": SIM_CLOSED,
        "ai_signals": AI_SIGNALS, "ai_analysis": AI_ANALYSIS
    }
    save_json(pth, blob)
    tg_send("Export hazır (sunucudaki backup klasöründe).")
# ===================== REAL TRADE EXEC ===================
def set_leverage(symbol, lev=5):
    try:
        fapi_signed("POST","/fapi/v1/leverage", {"symbol":symbol,"leverage":lev})
    except Exception as e:
        log(f"[LEV ERR]{symbol} {e}")

def place_tp_market(symbol, side, entry_price):
    """
    TAKE_PROFIT_MARKET + closePosition=true
    - 'price' PARAMETRESİ GÖNDERME (-1106 önlenir)
    - stopPrice zorunlu, 0’dan büyük olmalı
    - reduceOnly YOK (özel istek)
    """
    tp = build_tp_stop(symbol, side, entry_price)
    if not tp or tp["stopPrice"]<=0:
        log(f"[TP BUILD FAIL] {symbol} side={side} entry={entry_price}")
        return {"error":"TP_BUILD_FAIL"}

    params = {
        "symbol": symbol,
        "side":   "SELL" if side=="BUY" else "BUY",  # kapanış yönü
        "type":   "TAKE_PROFIT_MARKET",
        "timeInForce": "GTC",
        "stopPrice": f"{tp['stopPrice']:.16f}",
        "closePosition": "true",
        "workingType": "MARK_PRICE",  # mark/contract price; istersen "MARK_PRICE"
        # !!! price GÖNDERME !!!
        "timestamp": int(time.time()*1000)
    }
    params = maybe_apply_position_side(params, side)
    # İmza manuel:
    data = _sign({k:v for k,v in params.items() if k!="timestamp"} | {"timestamp":params["timestamp"]})
    try:
        r = requests.post(f"{BINANCE_FAPI}/fapi/v1/order", headers=_headers(), data=data, timeout=7).json()
        if 'orderId' in r:
            log(f"[TP OK]{symbol} side={side} stopPrice={tp['stopPrice']}")
            return r
        else:
            log(f"[TP FAIL]{symbol} resp={r}")
            return {"error":"TP_FAIL","resp":r}
    except Exception as e:
        log(f"[TP REQ ERR]{symbol} {e}")
        return {"error":"TP_REQUEST_ERR","exc":str(e)}

def open_market(symbol, direction):
    # side & qty
    side = "BUY" if direction=="UP" else "SELL"
    px = mark_price(symbol)
    if not px: 
        log(f"[OPEN FAIL] no mark price {symbol}")
        return None
    qty = _calc_notional_qty(symbol, px)
    if qty <= 0:
        log(f"[OPEN FAIL] qty<=0 {symbol}")
        return None

    set_leverage(symbol, int(PARAM.get("LEVERAGE",5)))

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": f"{qty:.8f}",
        # reduceOnly YOK!
        "timestamp": int(time.time()*1000)
    }
    params = maybe_apply_position_side(params, side)
    data = _sign({k:v for k,v in params.items() if k!="timestamp"} | {"timestamp":params["timestamp"]})
    try:
        r = requests.post(f"{BINANCE_FAPI}/fapi/v1/order", headers=_headers(), data=data, timeout=7).json()
        if "orderId" in r:
            entry = px
            STATE["open_positions"][symbol] = {"side":side, "entry":entry, "qty":qty, "time":now_iso()}
            save_json(STATE_FILE, STATE)
            log(f"[OPEN OK]{symbol} {side} qty={qty} entry~{entry}")
            # hemen TP kur
            place_tp_market(symbol, side, entry)
            return r
        else:
            log(f"[OPEN FAIL]{symbol} {r}")
            return None
    except Exception as e:
        log(f"[OPEN REQ ERR]{symbol} {e}")
        return None

def close_all_for_symbol(symbol):
    # market close (failsafe)
    pos = STATE["open_positions"].get(symbol)
    if not pos: return
    original_side = pos["side"]
    side = "SELL" if original_side=="BUY" else "BUY"
    qty = float(pos["qty"])
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": f"{qty:.8f}",
        "timestamp": int(time.time()*1000)
    }
    params = maybe_apply_position_side(params, original_side)
    data = _sign({k:v for k,v in params.items() if k!="timestamp"} | {"timestamp":params["timestamp"]})
    try:
        r = requests.post(f"{BINANCE_FAPI}/fapi/v1/order", headers=_headers(), data=data, timeout=7).json()
        log(f"[FORCE CLOSE]{symbol} resp={r}")
    except Exception as e:
        log(f"[FORCE CLOSE ERR]{symbol} {e}")
    # cleanup
    try:
        direction = "UP" if pos["side"]=="BUY" else "DOWN"
        _dir_dec(symbol, direction)
        _trendlock_set(symbol, direction)  # kapanışta trendlock başlasın
        del STATE["open_positions"][symbol]
        save_json(STATE_FILE, STATE)
    except:
        pass

# ===================== DISPATCH =========================
def execute_signal(sig):
    symbol = sig["symbol"]
    direction = sig["dir"]        # "UP"/"DOWN"
    side = "BUY" if direction=="UP" else "SELL"

    # Approve bars (varsayılan 0 = direkt)
    approve = int(PARAM.get("APPROVE_BARS", 0))
    if approve>0 and (STATE.get("bar_index",0) - sig.get("born_bar",0)) < approve:
        return

    # Guardlar
    if not _can_open(symbol, direction):
        return

    # REAL OPEN
    if open_market(symbol, direction):
        _dir_inc(symbol, direction)
        tg_send(f"✅ REAL {symbol} {direction} {sig['kind']} entry≈{sig['entry']:.6f}")

    # SIM de açık olarak tut
    sim_open(sig)

# ===================== INIT / SYMBOLS ===================
def auto_init_symbols():
    sy = PARAM.get("SYMBOLS","").strip()
    if sy:
        return sorted([s.strip().upper() for s in sy.split(",") if s.strip()])
    # fallback: top usd volume?
    try:
        tick = fapi_get("/fapi/v1/ticker/24hr", {})
        # USDT perpetual’lardan ilk 40
        arr = [t["symbol"] for t in tick if t.get("symbol","").endswith("USDT")]
        arr = sorted(arr)[:40]
        return arr
    except:
        return ["BTCUSDT","ETHUSDT","BNBUSDT"]

# ===================== HEARTBEAT/BACKUP =================
def maybe_heartbeat():
    last = int(STATE.get("last_heartbeat",0))
    if now_ts() - last >= PARAM["HEARTBEAT_MIN"]*60:
        tg_send(f"💓 Heartbeat v15.9.48 | open={len(STATE['open_positions'])} sim_open={len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])}")
        STATE["last_heartbeat"]=now_ts()
        save_json(STATE_FILE, STATE)

def maybe_backup():
    last = int(STATE.get("last_backup",0))
    if now_ts() - last >= PARAM["BACKUP_HOURS"]*3600:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob = {
            "state": STATE, "param": PARAM, "sim_open": SIM_POSITIONS,
            "sim_closed": SIM_CLOSED, "ai_signals": AI_SIGNALS, "ai_analysis": AI_ANALYSIS
        }
        save_json(os.path.join(BACKUP_DIR, f"backup_{stamp}.json"), blob)
        STATE["last_backup"]=now_ts()
        save_json(STATE_FILE, STATE)

# ===================== LOOP =============================
def main():
    ensure_loaded()
    tg_send("🚀 EMA ULTRA v15.9.48 aktif (PEMA removed | EARLY+UT/STC+EMA20/200+MACD+FVG | Smart TP v42 | no SL | no reduceOnly)")
    log("[START] EMA ULTRA v15.9.48 FULL")

    symbols = auto_init_symbols()
    log(f"[SYMBOLS] {len(symbols)} loaded")
    # --- HEDGE MODE'u etkinleştir ---
    try:
        fapi_signed("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true"})
        log("[INIT] Hedge Mode aktif edildi (dualSidePosition=true)")
    except Exception as e:
        log(f"[DUAL MODE SET ERR]{e}")

    while True:
        try:
            # Telegram komutları
            updates = _tg_get_updates(STATE.get("tg_offset"))
            if updates:
                for up in updates:
                    _tg_set_offset(up["update_id"]+1)
                    msg = up.get("message") or up.get("edited_message")
                    if not msg: continue
                    if str(msg.get("chat",{}).get("id")) != str(CHAT_ID): 
                        continue
                    text = (msg.get("text") or "").strip()
                    if not text.startswith("/"): 
                        continue
                    parts = text.split()
                    cmd = parts[0].lower()
                    args= parts[1:]
                    if cmd == "/status": _cmd_status()
                    elif cmd=="/report": _cmd_report()
                    elif cmd=="/set":    _cmd_set(args)
                    elif cmd=="/export": _cmd_export()
                    else:
                        tg_send("Komutlar: /status /report /set KEY VALUE /export")

            # bar tick
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            save_json(STATE_FILE, STATE)

            # tarama
            all_sigs=[]
            for sym in symbols:
                sigs = scan_symbol(sym, STATE["bar_index"])
                if sigs:
                    for s in sigs:
                        AI_SIGNALS.append(s)
                        # hemen çalıştır
                        execute_signal(s)
                all_sigs += sigs
            # analitik-lite
            AI_ANALYSIS["last_scan"] = now_iso()
            AI_ANALYSIS["signals_in_bar"] = len(all_sigs)
            save_json(AI_SIGNALS_FILE, AI_SIGNALS)
            save_json(AI_ANALYSIS_FILE, AI_ANALYSIS)

            # bakım işleri
            maybe_heartbeat()
            maybe_backup()

            # 60s bekle (1h bar olsa bile arada mark fiyat, komut vs.)
            time.sleep(60)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(5)

# ===================== ENTRY ============================
if __name__ == "__main__":
    main()
