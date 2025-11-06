import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.62 — Kıvanç Confirm + EARLY + SCALP + UT/STC + MACD + FVG
#   (PEMA Disabled) + MR Closer Only + Guard + Hedge
# ------------------------------------------------------------------------------
# - Kıvanç Confirm: SuperTrend + EMA9/EMA30 Cross (4 LONG + 4 SHORT limiti)
# - EARLY: EMA3/EMA7 + ATR spike
# - SCALP: EMA7 reversal + mini ATR filtresi
# - UT/STC: (UT benzeri trend filtresi) + MACD/RSI momentum onayı
# - MACD Trend: MACD line & signal kesişimleri
# - FVG Break: Son gap bölgesinin kırılımı yönüne işlem
# - Mean-Reversion: YENİ İŞLEM AÇMAZ, sadece açık MR pozisyonlarını kapatır
# - PEMA: TAMAMEN DEVRE DIŞI
# - Hedge uyumlu; Duplicate-Guard (TrendLock) + 6 saat cooldown
# ==============================================================================

# ---- Dosya yolları & temel ayarlar
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE       = os.path.join(DATA_DIR,"state.json")
PARAM_FILE       = os.path.join(DATA_DIR,"params.json")
LOG_FILE         = os.path.join(DATA_DIR,"log.txt")

BOT_TOKEN        = os.getenv("BOT_TOKEN")
CHAT_ID          = os.getenv("CHAT_ID")
BINANCE_KEY      = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET   = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI     = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
getcontext().prec = 28

# ---- TrendLock (duplicate guard)
TREND_LOCK = {}
TREND_LOCK_TIME = {}
TRENDLOCK_EXPIRY_SEC = 6 * 3600

# ---- Utils
def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

def safe_load(p, dflt):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return dflt

def safe_save(p, obj):
    try:
        with SAVE_LOCK:
            tmp = p + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p)
    except Exception as e:
        log(f"[SAVE ERR] {e}")

def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":t},
            timeout=10
        )
    except: pass

# ---- Indicators
def ema(vals, n):
    if not vals: return []
    k = 2/(n+1)
    e = [vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals) < period+2: return [50]*len(vals)
    d = np.diff(vals)
    g = np.maximum(d,0); l = -np.minimum(d,0)
    ag = np.mean(g[:period]); al = np.mean(l[:period])
    out = [50]*period
    for i in range(period, len(d)):
        ag = (ag*(period-1) + g[i]) / period
        al = (al*(period-1) + l[i]) / period
        rs = ag/al if al>0 else 0
        out.append(100 - 100/(1+rs))
    return [50]*(len(vals)-len(out)) + out

def macd(vals, fast=12, slow=26, signal=9):
    ef = ema(vals, fast); es = ema(vals, slow)
    macd_line = np.array(ef) - np.array(es)
    signal_ln = ema(macd_line.tolist(), signal)
    hist = macd_line - np.array(signal_ln)
    return macd_line.tolist(), signal_ln, hist.tolist()

def atr_like(highs, lows, closes, period=14):
    tr=[]
    for i in range(len(highs)):
        if i==0: tr.append(highs[i]-lows[i])
        else:
            tr.append(max(highs[i]-lows[i],
                          abs(highs[i]-closes[i-1]),
                          abs(lows[i]-closes[i-1])))
    if len(tr) < period: return [0]*len(highs)
    a = [sum(tr[:period])/period]
    for i in range(period, len(tr)):
        a.append((a[-1]*(period-1) + tr[i]) / period)
    return [0]*(len(highs)-len(a)) + a

# ---- SuperTrend & EMA Cross (Kıvanç Confirm için)
def supertrend_last(highs, lows, closes, period=10, mult=3.0):
    atr = atr_like(highs, lows, closes, period)
    mid = (np.array(highs) + np.array(lows)) / 2.0
    upper = mid + mult * np.array(atr)
    lower = mid - mult * np.array(atr)
    dir_up = True
    for i in range(1, len(closes)):
        if closes[i] > upper[i - 1]:
            dir_up = True
        elif closes[i] < lower[i - 1]:
            dir_up = False
        if dir_up:
            upper[i] = max(upper[i], closes[i])
        else:
            lower[i] = min(lower[i], closes[i])
    st_val = upper[-1] if dir_up else lower[-1]
    st_dir = "UP" if dir_up else "DOWN"
    return st_val, st_dir
# ---- Binance helpers
def _signed_request(method, path, params):
    q = "&".join([f"{k}={params[k]}" for k in params])
    sig = hmac.new(BINANCE_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + q + "&signature=" + sig
    if method == "POST":
        r = requests.post(url, headers=headers, timeout=10)
    else:
        r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(sym):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                         params={"symbol":sym}, timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym, interval, limit):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                         params={"symbol":sym,"interval":interval,"limit":limit},
                         timeout=10).json()
        # Son bar kapansın:
        if r and int(r[-1][6]) > now_ts_ms():
            r = r[:-1]
        return r
    except:
        return []

PRECISION_CACHE = {}
def get_symbol_filters(sym):
    if sym in PRECISION_CACHE: return PRECISION_CACHE[sym]
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo", timeout=10).json()
        s = next((x for x in info["symbols"] if x["symbol"]==sym), None)
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
        PRECISION_CACHE[sym] = {"stepSize":0.0001,"tickSize":0.0001,"minPrice":1e-8,"maxPrice":1e8}
    return PRECISION_CACHE[sym]

def _decimals_from_tick(tick_str):
    try:
        d = Decimal(str(tick_str))
        return max(0, -d.as_tuple().exponent)
    except:
        s = str(tick_str)
        if "." in s: return len(s.split(".")[1])
        return 0

def round_to_tick(sym, price_float):
    f = get_symbol_filters(sym)
    t = Decimal(str(f["tickSize"]))
    p = Decimal(str(price_float))
    if t <= 0: return float(p)
    q = (p/t).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(q*t)

def format_price_by_tick(sym, price_float):
    f = get_symbol_filters(sym)
    dec = _decimals_from_tick(str(f["tickSize"]))
    p_dec = Decimal(str(price_float)).quantize(Decimal(f"1e-{dec}"), rounding=ROUND_HALF_UP)
    if p_dec == Decimal("-0"): p_dec = Decimal("0")
    return f"{float(p_dec):.{dec}f}"

def calc_order_qty(sym, entry_price, usd):
    f = get_symbol_filters(sym)
    step = f["stepSize"]
    raw = usd / max(entry_price,1e-12)
    # stepSize'a yuvarla (banker rounding yerine ROUND_HALF_UP)
    step_dec = Decimal(str(step))
    raw_dec = Decimal(str(raw))
    q = (raw_dec/step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step_dec
    return float(q)

def auto_init_symbols():
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo", timeout=10).json()
        symbols = [s["symbol"] for s in info["symbols"]
                   if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR] {e}")
        symbols = []
    symbols.sort()
    return symbols

# ---- Param & State
STATE_DEFAULT = {"bar_index": 0}
PARAM_DEFAULT = {"TRADE_SIZE_USDT": 250.0}

PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict): PARAM = PARAM_DEFAULT
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

def _cleanup_trend_lock_expired():
    now_s = now_ts_s()
    expired = [sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        TREND_LOCK.pop(sym, None); TREND_LOCK_TIME.pop(sym, None)
        log(f"[TRENDLOCK TIMEOUT] {sym}")

# ===================== Kıvanç Confirm (SuperTrend + EMA Cross) =================
def ema_cross_dir(closes, fast=9, slow=30):
    if len(closes) < slow+3: return None
    ef = ema(closes, fast); es = ema(closes, slow)
    up = (ef[-1] > es[-1]) and (ef[-2] <= es[-2])
    dn = (ef[-1] < es[-1]) and (ef[-2] >= es[-2])
    if up: return "UP"
    if dn: return "DOWN"
    return None

def build_kivanc_confirm_signal(sym, kl, bar_i):
    if len(kl) < 120:
        return None
    closes = [float(k[4]) for k in kl]
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]

    st_val, st_dir = supertrend_last(highs, lows, closes, period=10, mult=3.0)
    cross_dir = ema_cross_dir(closes, fast=9, slow=30)
    if not cross_dir or not st_dir: return None
    if cross_dir != st_dir: return None

    entry = closes[-1]
    r_val = rsi(closes)[-1] if len(closes) >= 16 else 50.0
    atr_v = atr_like(highs, lows, closes)[-1]
    pwr   = 60 + (abs(entry - st_val) / max(entry, 1e-9)) * 100.0 + (r_val - 50) / 2.0

    direction = cross_dir
    tag = "✅ KIVANC CONFIRM " + ("BUY" if direction=="UP" else "SELL")

    # Bilgi amaçlı
    tp = entry * (1.006 if direction=="UP" else 0.994)
    sl = entry * (0.80  if direction=="UP" else 1.20)

    return {
        "symbol": sym, "dir": direction, "tier": "KIVANC", "emoji":"✅",
        "entry": entry, "tp": tp, "sl": sl, "power": pwr, "rsi": r_val, "atr": atr_v,
        "time": now_local_iso(), "born_bar": bar_i, "early": False,
        "kind": "KIVANC_CONFIRM", "tag": tag
    }
# ===================== Diğer Stratejiler (PEMA devre dışı) ====================
# EARLY: EMA3/EMA7 + ATR spike oranı
def build_early_signal(sym, kl, bar_i):
    if len(kl) < 50: return None
    closes = [float(k[4]) for k in kl]
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    e3 = ema(closes,3); e7 = ema(closes,7)
    atrv = atr_like(highs,lows,closes,14)[-1]
    if atrv <= 0: return None
    # Spike oranı (son bar gerçek gövdesi / ATR)
    body = abs(closes[-1]-closes[-2])
    spike_ratio = body / max(atrv,1e-9)
    # Koşullar
    if e3[-1] > e7[-1] and e3[-2] <= e7[-2] and spike_ratio >= 0.03:
        return {"symbol":sym,"dir":"UP","entry":closes[-1],"kind":"EARLY","tag":"⚡ EARLY BUY","born_bar":bar_i}
    if e3[-1] < e7[-1] and e3[-2] >= e7[-2] and spike_ratio >= 0.03:
        return {"symbol":sym,"dir":"DOWN","entry":closes[-1],"kind":"EARLY","tag":"⚡ EARLY SELL","born_bar":bar_i}
    return None

# SCALP: EMA7 reversal + mini ATR filtresi
def build_scalp_signal(sym, kl, bar_i):
    if len(kl) < 40: return None
    closes = [float(k[4]) for k in kl]
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    e7 = ema(closes,7)
    atrv = atr_like(highs,lows,closes,14)[-1]
    if atrv <= 0: return None
    # Reversal: fiyat EMA7'yi aşağıdan yukarı/ yukarıdan aşağı keser + gövde ATR'in min %10'u
    body = abs(closes[-1]-closes[-2])
    if closes[-2] <= e7[-2] and closes[-1] > e7[-1] and body >= 0.10*atrv:
        return {"symbol":sym,"dir":"UP","entry":closes[-1],"kind":"SCALP","tag":"🎯 SCALP BUY","born_bar":bar_i}
    if closes[-2] >= e7[-2] and closes[-1] < e7[-1] and body >= 0.10*atrv:
        return {"symbol":sym,"dir":"DOWN","entry":closes[-1],"kind":"SCALP","tag":"🎯 SCALP SELL","born_bar":bar_i}
    return None

# UT/STC benzeri: SuperTrend yönü + MACD/RSI momentum onayı
def build_ut_stc_signal(sym, kl, bar_i):
    if len(kl) < 120: return None
    closes = [float(k[4]) for k in kl]
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    st_val, st_dir = supertrend_last(highs, lows, closes, period=10, mult=3.0)
    macd_line, macd_sig, _ = macd(closes,12,26,9)
    r = rsi(closes,14)[-1]
    # Momentum onayı: MACD line > signal & RSI>52 (UP), tersi DOWN
    if st_dir=="UP" and macd_line[-1] > macd_sig[-1] and r>52:
        return {"symbol":sym,"dir":"UP","entry":closes[-1],"kind":"UT_STC","tag":"🟢 UT/STC BUY","born_bar":bar_i}
    if st_dir=="DOWN" and macd_line[-1] < macd_sig[-1] and r<48:
        return {"symbol":sym,"dir":"DOWN","entry":closes[-1],"kind":"UT_STC","tag":"🔴 UT/STC SELL","born_bar":bar_i}
    return None

# MACD Trend: Sade MACD line/signal kesişimi
def build_macd_trend_signal(sym, kl, bar_i):
    if len(kl) < 40: return None
    closes = [float(k[4]) for k in kl]
    line, sig, _ = macd(closes,12,26,9)
    up = line[-1] > sig[-1] and line[-2] <= sig[-2]
    dn = line[-1] < sig[-1] and line[-2] >= sig[-2]
    if up: return {"symbol":sym,"dir":"UP","entry":closes[-1],"kind":"MACD","tag":"📈 MACD BUY","born_bar":bar_i}
    if dn: return {"symbol":sym,"dir":"DOWN","entry":closes[-1],"kind":"MACD","tag":"📉 MACD SELL","born_bar":bar_i}
    return None

# FVG Break: Son 10 bar içinde oluşan boşluk (gap) bölgesinin kırılması
def _find_last_fvg(highs, lows):
    # Basit FVG: up-gap (low[i] > high[i-1]) veya down-gap (high[i] < low[i-1])
    for i in range(len(highs)-1, 0, -1):
        if lows[i] > highs[i-1]:
            return ("UPGAP", highs[i-1], lows[i])   # [top, bottom] değil; gap aralığı (high[i-1]..low[i])
        if highs[i] < lows[i-1]:
            return ("DOWNGAP", highs[i], lows[i-1])
    return (None, None, None)

def build_fvg_break_signal(sym, kl, bar_i):
    if len(kl) < 30: return None
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    closes = [float(k[4]) for k in kl]
    kind, a, b = _find_last_fvg(highs[-10:], lows[-10:])
    if not kind: return None
    c = closes[-1]
    if kind=="UPGAP":
        # Gap bölgesine geri dönüş sonrası yukarı kırılım
        if c > b:  # gap üst sınırı yeniden aşıldı
            return {"symbol":sym,"dir":"UP","entry":c,"kind":"FVG","tag":"🪟 FVG BREAK BUY","born_bar":bar_i}
    if kind=="DOWNGAP":
        if c < a:
            return {"symbol":sym,"dir":"DOWN","entry":c,"kind":"FVG","tag":"🪟 FVG BREAK SELL","born_bar":bar_i}
    return None

# ====== Mean-Reversion (Sadece Kapatıcı) ======================================
MEAN_REV_FILE        = os.path.join(DATA_DIR,"mean_reversion_positions.json")
MEAN_REV_POS         = safe_load(MEAN_REV_FILE, [])
MEAN_REV_EXIT_DIST   = 15.0    # %15 ve üstü, 2 onayla kapanış
MEAN_REV_EXIT_MAX    = 30.0    # güvenlik hard cap
MEAN_REV_CONFIRM     = 2
MEAN_REV_INTERVAL    = 120     # watcher aralığı (s)

def _mr_save(): safe_save(MEAN_REV_FILE, MEAN_REV_POS)

def mean_reversion_distance_pct(sym):
    kl = futures_get_klines(sym, "1h", 120)
    if not kl: return 0.0, None, None, None
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    closes = [float(k[4]) for k in kl]
    c_now  = closes[-1]
    ma99   = ema(closes, 99)[-1] if len(closes) >= 99 else np.mean(closes[-50:])
    stv    = (np.mean(highs[-10:]) + np.mean(lows[-10:])) / 2.0
    d_ma = abs(c_now - ma99) / max(ma99, 1e-9) * 100.0
    d_st = abs(c_now - stv ) / max(stv , 1e-9) * 100.0
    return max(d_ma, d_st), c_now, ma99, stv

def close_market(sym, direction, qty):
    side = "SELL" if direction=="UP" else "BUY"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    _signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })

def close_mean_reversion(sym, direction, reason):
    try:
        pos = next((p for p in MEAN_REV_POS if p["symbol"]==sym and p["dir"]==direction), None)
        if not pos: return
        qty = pos["qty"]
        close_market(sym, direction, qty)
        tg_send(f"⚠️ {sym} Mean-Reversion Exit — {reason}")
        log(f"[MEAN-REV CLOSE] {sym} {direction} {reason}")
    except Exception as e:
        tg_send(f"❌ MEAN-REV CLOSE ERR {sym}\n{e}")
        log(f"[MEAN-REV CLOSE ERR] {sym} {e}")
    MEAN_REV_POS[:] = [p for p in MEAN_REV_POS if not (p["symbol"]==sym and p["dir"]==direction)]
    _mr_save()
    TREND_LOCK.pop(sym, None); TREND_LOCK_TIME.pop(sym, None)
    log(f"[MR TRENDLOCK CLEAR] {sym}")

def mean_reversion_watcher():
    while True:
        try:
            if not MEAN_REV_POS:
                time.sleep(MEAN_REV_INTERVAL); continue
            for pos in list(MEAN_REV_POS):
                sym, direction = pos["symbol"], pos["dir"]
                dist, _, _, _ = mean_reversion_distance_pct(sym)

                if dist >= MEAN_REV_EXIT_MAX:
                    close_mean_reversion(sym, direction, f"fiyat ortalamadan uzaklaştı (%{dist:.2f}) [HARD]")
                    continue

                if dist >= MEAN_REV_EXIT_DIST:
                    pos["confirm"] = pos.get("confirm",0) + 1
                else:
                    pos["confirm"] = 0

                if pos.get("confirm",0) >= MEAN_REV_CONFIRM:
                    close_mean_reversion(sym, direction, f"fiyat ortalamadan uzaklaştı (%{dist:.2f})")
            _mr_save()
        except Exception as e:
            log(f"[MEAN-REV WATCH ERR] {e}")
        time.sleep(MEAN_REV_INTERVAL)
# ===================== Trade Helpers & Executors ===============================
def open_market(sym, direction, qty):
    side = "BUY" if direction=="UP" else "SELL"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    res = _signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })
    return float(res.get("avgPrice") or res.get("price") or futures_get_price(sym) or 0.0)

def _count_positions_all():
    try:
        acc = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[POS COUNT ERR] {e}")
        return 0, 0
    long_cnt  = sum(1 for p in acc if float(p.get("positionAmt",0)) > 0)
    short_cnt = sum(1 for p in acc if float(p.get("positionAmt",0)) < 0)
    return long_cnt, short_cnt

def _calc_qty(sym, entry_ref):
    usd = PARAM.get("TRADE_SIZE_USDT", 250.0)
    q = calc_order_qty(sym, entry_ref, usd)
    return q if q and q>0 else 0.0

# ---- Kıvanç Confirm: 4 LONG + 4 SHORT limiti
def execute_kivanc_trade(sig):
    sym = sig["symbol"]; direction = sig["dir"]
    # Pozisyon limit kontrolü (sadece Kıvanç için)
    long_cnt, short_cnt = _count_positions_all()
    if direction == "UP" and long_cnt >= 4:
        log(f"[KIVANC LIMIT] Maksimum 4 LONG açık, {sym} atlandı.")
        return
    if direction == "DOWN" and short_cnt >= 4:
        log(f"[KIVANC LIMIT] Maksimum 4 SHORT açık, {sym} atlandı.")
        return
    # Duplicate guard
    if TREND_LOCK.get(sym) == direction:
        log(f"[KIVANC GUARD] {sym} {direction} aktif, atlandı.")
        return
    entry_ref = sig.get("entry") or futures_get_price(sym)
    if not entry_ref: return
    qty = _calc_qty(sym, entry_ref)
    if qty<=0: return
    try:
        fill = open_market(sym, direction, qty)
        TREND_LOCK[sym] = direction; TREND_LOCK_TIME[sym] = now_ts_s()
        tg_send(f"✅ KIVANC CONFIRM OPEN\n{sym} {direction} qty:{qty}\nEntry:{fill:.12f}")
        log(f"[KIVANC OPEN] {sym} {direction} entry={fill}")
    except Exception as e:
        tg_send(f"❌ KIVANC OPEN ERR {sym}\n{e}")
        log(f"[KIVANC OPEN ERR] {sym} {e}")

# ---- Diğer stratejiler: limit yok (TREND_LOCK sembol&yön bazlı korur)
def execute_generic_trade(sig):
    sym = sig["symbol"]; direction = sig["dir"]; kind = sig.get("kind","GEN")
    if TREND_LOCK.get(sym) == direction:
        log(f"[{kind} GUARD] {sym} {direction} aktif, atlandı.")
        return
    entry_ref = sig.get("entry") or futures_get_price(sym)
    if not entry_ref: return
    qty = _calc_qty(sym, entry_ref)
    if qty<=0: return
    try:
        fill = open_market(sym, direction, qty)
        TREND_LOCK[sym] = direction; TREND_LOCK_TIME[sym] = now_ts_s()
        tg_send(f"🟦 {kind} OPEN\n{sym} {direction} qty:{qty}\nEntry:{fill:.12f}\n{sig.get('tag','')}")
        log(f"[{kind} OPEN] {sym} {direction} entry={fill}")
    except Exception as e:
        tg_send(f"❌ {kind} OPEN ERR {sym}\n{e}")
        log(f"[{kind} OPEN ERR] {sym} {e}")

# ===================== Sinyal Taraması ========================================
def scan_symbol(sym, bar_i):
    kl = futures_get_klines(sym, "1h", 200)  # Kıvanç ve çoğu strateji 1H
    if len(kl) < 60: return []

    res = []

    # Kıvanç Confirm
    s_kiv = build_kivanc_confirm_signal(sym, kl, bar_i)
    if s_kiv: res.append(s_kiv)

    # PEMA DEVRE DIŞI — hiçbir çağrı YOK

    # Diğer stratejiler (aktif)
    s_early = build_early_signal(sym, kl, bar_i)
    if s_early: res.append(s_early)

    s_scalp = build_scalp_signal(sym, kl, bar_i)
    if s_scalp: res.append(s_scalp)

    s_ut = build_ut_stc_signal(sym, kl, bar_i)
    if s_ut: res.append(s_ut)

    s_macd = build_macd_trend_signal(sym, kl, bar_i)
    if s_macd: res.append(s_macd)

    s_fvg = build_fvg_break_signal(sym, kl, bar_i)
    if s_fvg: res.append(s_fvg)

    return res

# ===================== Main ====================================================
def main():
    tg_send("🚀 EMA ULTRA v15.9.62 aktif — KC + EARLY + SCALP + UT/STC + MACD + FVG | PEMA OFF | MR Closer")
    log("[START] EMA ULTRA v15.9.62")

    symbols = auto_init_symbols()

    # MR: sadece kapatıcı izleyici
    threading.Thread(target=mean_reversion_watcher, daemon=True).start()

    while True:
        try:
            STATE["bar_index"] = STATE.get("bar_index", 0) + 1
            bar_i = STATE["bar_index"]

            # Sinyal tarama
            sigs = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = [ex.submit(scan_symbol, s, bar_i) for s in symbols]
                for f in as_completed(futs):
                    try:
                        r = f.result()
                    except Exception as e:
                        log(f"[SCAN ERR] {e}")
                        r = []
                    if r: sigs.extend(r)

            # İşlemler
            for sig in sigs:
                kind = sig.get("kind","GEN")
                if kind == "KIVANC_CONFIRM":
                    execute_kivanc_trade(sig)      # 4L/4S limitli
                else:
                    execute_generic_trade(sig)     # limit yok, TREND_LOCK aktif

            # Guard timeout
            _cleanup_trend_lock_expired()

            # Persist
            safe_save(STATE_FILE, STATE)

            time.sleep(30)

        except Exception as e:
            log(f"[MAIN LOOP ERR] {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
