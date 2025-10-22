# ==============================================================
#  📘 EMA ULTRA v12.6.3 — JSONPersistent + 3-Bar Scalping + StateFix
#  Binance Futures PUBLIC data only — NO API keys needed
#  EMA+ATR+RSI+ADX + SCALP + AI Adaptive + Angle Momentum
#  JSON-safe storage + RenderFix + Daily ZIP backup to Telegram
# ==============================================================

import os, json, io, time, math, requests
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# ================= RENDER SAFE PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
if not DATA_DIR:
    DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === Folder structure ===
LOG_FILE    = os.path.join(DATA_DIR, "log.txt")
STATE_FILE  = os.path.join(DATA_DIR, "state.json")
OPEN_JSON   = os.path.join(DATA_DIR, "open_positions.json")
CLOSED_JSON = os.path.join(DATA_DIR, "closed_trades.json")
AI_LOG_JSON = os.path.join(DATA_DIR, "ai_updates.json")

LEARN_DIR   = os.path.join(DATA_DIR, "data_learning")
PARAM_HISTORY_DIR = os.path.join(LEARN_DIR, "params_history")
WEEK_MODEL_FILE   = os.path.join(LEARN_DIR, "week_model.json")
WEEKEND_MODEL_FILE= os.path.join(LEARN_DIR, "weekend_model.json")

# ================= TELEGRAM (env vars) =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ================= TIME HELPERS =================
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)
def now_ist(): return now_ist_dt().isoformat()
def today_ist_date(): return now_ist_dt().strftime("%Y-%m-%d")
def now_ist_hhmm(): return now_ist_dt().strftime("%H:%M")
def is_weekend_ist(): return now_ist_dt().weekday() in (5, 6)

# ================= LOGGING =================
def ensure_dir(path):
    try: os.makedirs(path, exist_ok=True)
    except: pass

def log(msg):
    print(msg, flush=True)
    try:
        ensure_dir(os.path.dirname(LOG_FILE) or ".")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_ist()} - {msg}\n")
    except: pass

def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] env eksik"); return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        log(f"[TG] {text.splitlines()[0][:64]} ...")
    except Exception as e:
        log(f"[TG] send error: {e}")

def send_tg_document(filename, bytes_content):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] env eksik"); return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    files = {'document': (filename, bytes_content)}
    data = {'chat_id': CHAT_ID, 'caption': filename}
    try:
        requests.post(url, data=data, files=files, timeout=60)
        log(f"[TG] document sent: {filename}")
    except Exception as e:
        log(f"[TG] doc error: {e}")

# ================= JSON HELPERS =================
def safe_load_json(path, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"[ERR] load_json {path}: {e}")
    return {} if default is None else default

def safe_save_json(path, data):
    try:
        ensure_dir(os.path.dirname(path) or ".")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"[ERR] save_json {path}: {e}")

# İlk dizin kurulumu
ensure_dir(DATA_DIR)
ensure_dir(LEARN_DIR)
ensure_dir(PARAM_HISTORY_DIR)
log("[INIT] ensured data folders")
time.sleep(3)
log("[WAIT] Render disk mount için 3sn bekleme eklendi.")

# ================= BASE SETTINGS =================
INTERVAL_1H, INTERVAL_4H, INTERVAL_1D = "1h", "4h", "1d"
LIMIT_1H, LIMIT_4H, LIMIT_1D = 500, 300, 250
ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14
VOL_MA_PERIOD = 20
VOL_SPIKE_MULT = 1.8
CROSS_CONFIRM_BARS = 1
SCAN_INTERVAL = 300
SLEEP_BETWEEN = 0.12
DAILY_SUMMARY_ENABLED = True
DAILY_SUMMARY_TIME = "23:59"
LEARN_DAYS = 14
AI_PREDICT_ENABLE = True
AI_PNL_THRESHOLD  = 0.30
AI_MIN_CONF       = 0.10

PARAM = {
    "POWER_NORMAL_MIN": 60.0,
    "POWER_PREMIUM_MIN": 68.0,
    "POWER_ULTRA_MIN": 75.0,
    "ATR_BOOST_PCT": 0.004,
    "ATR_BOOST_ADD": 5.0,
    "ADX_BASE": 25.0,
    "ADX_MAX_ADD": 10.0,
    "VOL_MAX_ADD": 7.0,
    "SCALP_TP_PCT": 0.006,
    "SCALP_SL_PCT": 0.10,
    "CROSS_TP_PCT": 0.010,
    "CROSS_SL_PCT": 0.030,
    "SCALP_COOLDOWN_BARS": 3
}

# ================= INDICATORS =================
def ema(vals, length):
    k = 2 / (length + 1)
    e = [vals[0]]
    for i in range(1, len(vals)):
        e.append(vals[i] * k + e[-1] * (1 - k))
    return e

def sma(vals, period):
    if len(vals) < period: return [sum(vals)/len(vals)]*len(vals)
    out=[]; s=sum(vals[:period])
    out.extend([s/period]*(period-1)); out.append(s/period)
    for i in range(period, len(vals)):
        s += vals[i] - vals[i-period]
        out.append(s/period)
    return out
# ================= MORE INDICATORS =================
def atr_series(highs, lows, closes, period=14):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i]-lows[i])
        else:
            pc = closes[i-1]
            trs.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    if len(trs) < period: return [0]*len(trs)
    a = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        a.append((a[-1]*(period-1) + trs[i]) / period)
    return [0]*(len(trs)-len(a)) + a

def rsi(vals, period=14):
    if len(vals) < period + 1: return [50]*len(vals)
    deltas = [vals[i] - vals[i-1] for i in range(1, len(vals))]
    gains  = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50]*period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else 0
        rsis.append(100 - 100/(1 + rs))
    return [50]*(len(vals)-len(rsis)) + rsis

def adx_series(highs, lows, closes, period=14):
    if len(highs) < period + 2: return [0]*len(highs), [0]*len(highs), [0]*len(highs)
    tr, plus_dm, minus_dm = [], [], []
    for i in range(len(highs)):
        if i == 0:
            tr.append(highs[i]-lows[i]); plus_dm.append(0); minus_dm.append(0)
        else:
            up = highs[i] - highs[i-1]
            dn = lows[i-1] - lows[i]
            plus_dm.append(up if (up>dn and up>0) else 0)
            minus_dm.append(dn if (dn>up and dn>0) else 0)
            pc = closes[i-1]
            tr.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    tr_s   = [sum(tr[:period])]
    plus_s = [sum(plus_dm[:period])]
    minus_s= [sum(minus_dm[:period])]
    for i in range(period, len(highs)):
        tr_s.append(tr_s[-1] - tr_s[-1]/period + tr[i])
        plus_s.append(plus_s[-1] - plus_s[-1]/period + plus_dm[i])
        minus_s.append(minus_s[-1] - minus_s[-1]/period + minus_dm[i])
    di_plus = [0]*(period) + [ (plus_s[i-period]/tr_s[i-period])*100 if tr_s[i-period]!=0 else 0 for i in range(period, len(tr_s)) ]
    di_minus= [0]*(period) + [ (minus_s[i-period]/tr_s[i-period])*100 if tr_s[i-period]!=0 else 0 for i in range(period, len(tr_s)) ]
    dx = []
    for i in range(len(di_plus)):
        dplus = di_plus[i]; dminus = di_minus[i]
        den = (dplus + dminus)
        dx.append( (abs(dplus - dminus)/den*100) if den != 0 else 0 )
    adx = []
    first = sum(dx[period:period*2]) / period if len(dx) >= period*2 else 0
    adx = [0]*(period*2) + [first] if len(dx) >= period*2 else [0]*len(dx)
    for i in range(period*2+1, len(dx)):
        adx.append((adx[-1]*(period-1) + dx[i]) / period)
    need = len(highs) - len(adx)
    adx = [0]*need + adx if need>0 else adx[:len(highs)]
    di_plus = di_plus[:len(highs)]
    di_minus= di_minus[:len(highs)]
    return adx, di_plus, di_minus

# ================= ANGLE HELPERS =================
def slope_angle_deg(slope, atr_now, eps=1e-9):
    """slope/ATR normalizasyonu ile açı (derece)"""
    if atr_now <= eps: return 0.0
    return math.degrees(math.atan((slope/atr_now)))

def angle_between_deg(s_prev, s_now, atr_now, eps=1e-9):
    """İki eğim arasındaki açı (derece) — m1=s_prev/ATR, m2=s_now/ATR"""
    if atr_now <= eps: return 0.0
    m1 = s_prev/atr_now; m2 = s_now/atr_now
    denom = 1.0 + (m1*m2)
    if abs(denom) < eps: return 90.0
    return math.degrees(math.atan(abs(m2 - m1)/denom))

# ================= POWER / LABEL HELPERS =================
def rsi_divergence(last_close, prev_close, rsi_now, rsi_prev):
    if last_close < prev_close and rsi_now > rsi_prev:  return "Bullish"
    if last_close > prev_close and rsi_now < rsi_prev:  return "Bearish"
    return "Neutral"

def atr_boost(atr_now, price):
    atr_pct = (atr_now / price) if price > 0 else 0.0
    return (PARAM["ATR_BOOST_ADD"] if atr_pct >= PARAM["ATR_BOOST_PCT"] else 0.0), atr_pct

def tier_color(power):
    if power >= PARAM["POWER_ULTRA_MIN"]:   return "ULTRA", "🟩"
    if power >= PARAM["POWER_PREMIUM_MIN"]: return "PREMIUM", "🟦"
    if power >= PARAM["POWER_NORMAL_MIN"]:  return "NORMAL", "🟨"
    return "NONE", ""

def power_base(s_prev, s_now, atr_now, price, rsi_now):
    slope_comp = abs(s_now - s_prev) / (atr_now * 0.6) if atr_now > 0 else 0.0
    rsi_comp   = (rsi_now - 50) / 50.0
    atr_comp   = (atr_now / price) * 100.0 if price > 0 else 0.0
    base = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    return max(0.0, min(100.0, base))

def power_with_adx_vol(base_power, adx_now, vol_now, vol_ma):
    add = 0.0
    if adx_now > PARAM["ADX_BASE"]:
        add += min(PARAM["ADX_MAX_ADD"], (adx_now - PARAM["ADX_BASE"]) / 15.0 * PARAM["ADX_MAX_ADD"])
    mult = (vol_now / vol_ma) if (vol_ma and vol_ma>0) else 1.0
    if mult >= 1.0:
        if mult >= VOL_SPIKE_MULT:
            add += min(PARAM["VOL_MAX_ADD"], (mult - 1.0) / (VOL_SPIKE_MULT - 1.0) * PARAM["VOL_MAX_ADD"])
        else:
            add += (mult - 1.0) * (PARAM["VOL_MAX_ADD"] / (VOL_SPIKE_MULT - 1.0))
    return max(0.0, min(100.0, base_power + add)), mult

def power_with_angle(base_power, ang_now, ang_change):
    bonus = min(6.0, max(0.0, (abs(ang_now) / 75.0) * 6.0))   # 75° ⇒ ~+6
    penalty = min(5.0, (ang_change / 45.0) * 5.0)             # 45° ⇒ ~-5
    return max(0.0, min(100.0, base_power + bonus - penalty))

# ================= BINANCE PUBLIC ENDPOINTS =================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v12.6.3", "Accept": "application/json"})
BASE = "https://fapi.binance.com"

def get_futures_symbols(retries=3, delay=3.0):
    for i in range(retries):
        try:
            r = SESSION.get(BASE + "/fapi/v1/exchangeInfo", timeout=15)
            data = r.json()
            syms = [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
            if syms: return syms
        except Exception as e:
            log(f"exchangeInfo err try {i+1}/{retries}: {e}")
        time.sleep(delay)
    return []

def get_klines(symbol, interval, limit=500):
    url = BASE + "/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
            # gelecekte kapanacak barı at
            if data and int(data[-1][6]) > now_ms: data = data[:-1]
            return data
    except Exception as e:
        log(f"klines err {symbol} {interval}: {e}")
    return []

def get_last_price(symbol):
    try:
        url = f"{BASE}/fapi/v1/ticker/price?symbol={symbol}"
        r = SESSION.get(url, timeout=8).json()
        return float(r["price"])
    except:
        return None

# ================= TREND HELPERS =================
def ema_trend_dir(closes):
    e7, e25 = ema(closes,7), ema(closes,25)
    return "UP" if e7[-1] > e25[-1] else "DOWN"

def trend_alignment(signal_dir, closes_4h, closes_1d):
    d4 = ema_trend_dir(closes_4h)
    d1 = ema_trend_dir(closes_1d)
    match = (signal_dir == d4) and (signal_dir == d1)
    return match, d4, d1

# ================== LITE AI PREDICT (JSON-ONLY) ==================
def rolling_stats_expected_pnl(direction_is_up, is_scalp, aligned, ang_now, ang_change, last_n=80):
    """
    Pkl/model kullanmadan, CLOSED_JSON içinden benzer sinyallerin ortalama PnL'ini ve güvenini hesaplar.
    - basit/sağlam: JSON-only, yeniden başlatmalara dayanıklı
    """
    try:
        data = safe_load_json(CLOSED_JSON, default=[])
        if not isinstance(data, list) or not data:
            return None, None

        # filtre: tip + yön + alignment'a yakın
        filt = []
        for r in reversed(data[-last_n:]):  # son N işlem
            if str(r.get("type","")).upper() == ("SCALP" if is_scalp else "CROSS") and \
               str(r.get("direction","")).upper() == ("UP" if direction_is_up else "DOWN"):
                # açı benzerliği (yumuşak)
                a_now = float(r.get("ang_now", 0.0)); a_ch = float(r.get("ang_change", 0.0))
                # |fark| toplamı 50° altında ise benzer say
                if abs(a_now - ang_now) + abs(a_ch - ang_change) <= 50.0:
                    filt.append(float(r.get("pnl", 0.0)))
        if not filt:
            return None, None

        avg = float(np.mean(filt))           # % cinsinden beklenen pnl
        n   = len(filt)
        # güven: örnek sayısına bağlı basit skala (0-1)
        conf = max(0.05, min(1.0, n/40.0))   # 40 örnek ~ 1.0

        return avg/100.0, conf  # model arayüzüne uyum için 0-1
    except Exception as e:
        log(f"[LITE_AI_ERR] {e}")
        return None, None

# ================= ENGINES =================
def run_cross_engine_confirmed(sym, kl1, kl4, kl1d):
    closes = [float(k[4]) for k in kl1]
    ema7, ema25 = ema(closes,7), ema(closes,25)

    cross_dir = None
    # her zaman kapanmış barı esas al (re-run'da tekrar sinyal olmasın)
    bar_close_ms = int(float(kl1[-2][6]))

    if CROSS_CONFIRM_BARS == 1:
        prev_diff    = ema7[-4] - ema25[-4]
        cross_diff   = ema7[-3] - ema25[-3]
        confirm_diff = ema7[-2] - ema25[-2]
        if prev_diff < 0 and cross_diff > 0 and confirm_diff > 0: cross_dir = "UP"
        if prev_diff > 0 and cross_diff < 0 and confirm_diff < 0: cross_dir = "DOWN"
    else:
        prev_diff = ema7[-3] - ema25[-3]
        curr_diff = ema7[-2] - ema25[-2]
        if prev_diff < 0 and curr_diff > 0: cross_dir = "UP"
        if prev_diff > 0 and curr_diff < 0: cross_dir = "DOWN"

    if not cross_dir: return None

    highs = [float(k[2]) for k in kl1]
    lows  = [float(k[3]) for k in kl1]
    vols  = [float(k[5]) for k in kl1]
    atr_now = atr_series(highs, lows, closes, ATR_PERIOD)[-1]
    rsi_all = rsi(closes, RSI_PERIOD)
    rsi_now, rsi_prev = rsi_all[-1], rsi_all[-2]

    # Eğimi 3 bar yerine cross için 4 bar koruyoruz (daha kararlı)
    s_now  = ema7[-2] - ema7[-5]
    s_prev = ema7[-3] - ema7[-6]
    price  = closes[-1]

    # Açı metrikleri
    ang_now = slope_angle_deg(s_now, atr_now)
    ang_prev = slope_angle_deg(s_prev, atr_now)
    ang_change = angle_between_deg(s_prev, s_now, atr_now)

    adx_all, _, _ = adx_series(highs, lows, closes, ADX_PERIOD)
    adx_now = adx_all[-1]
    vol_ma = sma(vols, VOL_MA_PERIOD)[-1]

    base = power_base(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr_after = base + boost
    pwr_final, vol_mult = power_with_adx_vol(pwr_after, adx_now, vols[-1], vol_ma)
    pwr_final = power_with_angle(pwr_final, ang_now, ang_change)

    div = rsi_divergence(closes[-1], closes[-2], rsi_now, rsi_prev)

    closes4  = [float(k[4]) for k in kl4]
    closes1d = [float(k[4]) for k in kl1d]
    aligned, d4, d1 = trend_alignment(cross_dir, closes4, closes1d)
    trend_match = 1 if aligned else 0

    entry = price
    tp = entry * (1 + PARAM["CROSS_TP_PCT"] if cross_dir == "UP" else 1 - PARAM["CROSS_TP_PCT"])
    sl = entry * (1 - PARAM["CROSS_SL_PCT"] if cross_dir == "UP" else 1 + PARAM["CROSS_SL_PCT"])

    # JSON-only "lite AI" tahmin (opsiyonel)
    ai_pred, ai_conf = (None, None)
    if AI_PREDICT_ENABLE:
        dir_is_up = (cross_dir == "UP")
        ai_pred, ai_conf = rolling_stats_expected_pnl(dir_is_up, is_scalp=False, aligned=aligned,
                                                      ang_now=ang_now, ang_change=ang_change)
        if ai_pred is not None and ai_conf is not None:
            if (ai_pred*100.0) < AI_PNL_THRESHOLD or ai_conf < AI_MIN_CONF:
                return None

    return {
        "symbol": sym, "type": "CROSS", "dir": cross_dir,
        "entry": entry, "tp": tp, "sl": sl,
        "power": pwr_final, "rsi": rsi_now, "div": div,
        "atr": atr_now, "atr_pct": atr_pct, "bar_close_ms": bar_close_ms,
        "aligned": aligned, "trend4h": d4, "trend1d": d1, "confirmed": CROSS_CONFIRM_BARS==1,
        "adx": adx_now, "vol_mult": vol_mult,
        "ang_now": ang_now, "ang_prev": ang_prev, "ang_change": ang_change,
        "ai_pred": ai_pred, "ai_conf": ai_conf
    }

def run_scalp_engine(sym, kl1, kl4, kl1d):
    closes1 = [float(k[4]) for k in kl1]
    ema7_1  = ema(closes1, 7)
    if len(ema7_1) < 5: return None  # 3-bar için min 5 örnek

    # 🔄 Yeni: 3-bar momentum penceresi (daha duyarlı scalp)
    s_now  = ema7_1[-2] - ema7_1[-5+2]  # eşdeğer: ema7[-2] - ema7[-3]
    s_now  = ema7_1[-1] - ema7_1[-3]
    s_prev = ema7_1[-2] - ema7_1[-4]

    # yön dönüşü
    slope_dir = "UP" if (s_prev < 0 and s_now > 0) else ("DOWN" if (s_prev > 0 and s_now < 0) else None)
    if not slope_dir: return None

    closes4  = [float(k[4]) for k in kl4]
    closes1d = [float(k[4]) for k in kl1d]
    d4 = ema_trend_dir(closes4)
    if slope_dir != d4: return None
    d1 = ema_trend_dir(closes1d)
    aligned = (slope_dir == d4) and (slope_dir == d1)
    trend_match = 1 if aligned else 0

    highs1 = [float(k[2]) for k in kl1]
    lows1  = [float(k[3]) for k in kl1]
    vols1  = [float(k[5]) for k in kl1]
    atr_now = atr_series(highs1, lows1, closes1, ATR_PERIOD)[-1]
    rsi_now = rsi(closes1, RSI_PERIOD)[-1]
    price   = closes1[-1]

    ang_now = slope_angle_deg(s_now, atr_now)
    ang_prev = slope_angle_deg(s_prev, atr_now)
    ang_change = angle_between_deg(s_prev, s_now, atr_now)

    adx_all, _, _ = adx_series(highs1, lows1, closes1, ADX_PERIOD)
    adx_now = adx_all[-1]
    vol_ma  = sma(vols1, VOL_MA_PERIOD)[-1]

    base = power_base(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr_after = base + boost
    pwr_final, vol_mult = power_with_adx_vol(pwr_after, adx_now, vols1[-1], vol_ma)
    pwr_final = power_with_angle(pwr_final, ang_now, ang_change)

    rsi_prev = rsi(closes1, RSI_PERIOD)[-2]
    div = rsi_divergence(closes1[-1], closes1[-2], rsi_now, rsi_prev)

    if pwr_final < PARAM["POWER_PREMIUM_MIN"]: return None

    entry = price
    tp = entry * (1 + PARAM["SCALP_TP_PCT"] if slope_dir == "UP" else 1 - PARAM["SCALP_TP_PCT"])
    sl = entry * (1 - PARAM["SCALP_SL_PCT"] if slope_dir == "UP" else 1 + PARAM["SCALP_SL_PCT"])

    ai_pred, ai_conf = (None, None)
    if AI_PREDICT_ENABLE:
        dir_is_up = (slope_dir == "UP")
        ai_pred, ai_conf = rolling_stats_expected_pnl(dir_is_up, is_scalp=True, aligned=aligned,
                                                      ang_now=ang_now, ang_change=ang_change)
        if ai_pred is not None and ai_conf is not None:
            if (ai_pred*100.0) < AI_PNL_THRESHOLD or ai_conf < AI_MIN_CONF:
                return None

    return {
        "symbol": sym, "type": "SCALP", "dir": slope_dir,
        "entry": entry, "tp": tp, "sl": sl,
        "power": pwr_final, "rsi": rsi_now, "div": div,
        "atr": atr_now, "atr_pct": atr_pct,
        "aligned": aligned, "trend4h": d4, "trend1d": d1,
        "adx": adx_now, "vol_mult": vol_mult,
        "ang_now": ang_now, "ang_prev": ang_prev, "ang_change": ang_change,
        "ai_pred": ai_pred, "ai_conf": ai_conf
    }
# ================= JSON LIST HELPERS =================
def load_json_list(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        log(f"[ERR] load_json_list {path}: {e}")
    return []

def save_json_list(path, data_list):
    try:
        ensure_dir(os.path.dirname(path) or ".")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"[ERR] save_json_list {path}: {e}")

def append_json_list(path, item):
    data = load_json_list(path)
    data.append(item)
    save_json_list(path, data)

# ================= DAILY SUMMARY (JSON-ONLY) =================
def build_daily_summary_for(date_str, state):
    rows = []
    try:
        closed = load_json_list(CLOSED_JSON)
        for r in closed:
            tc = str(r.get("time_close",""))
            if tc.startswith(date_str):
                rows.append(r)
    except Exception as e:
        log(f"[ERR] daily_summary load: {e}")

    total = len(rows)
    wins = sum(1 for r in rows if str(r.get("result","")).upper()=="TP")
    pnl_sum = sum(float(r.get("pnl",0) or 0) for r in rows)
    bars_sum= sum(int(r.get("bars",0) or 0) for r in rows)
    winrate = (wins/total*100.0) if total else 0.0
    avg_pnl = (pnl_sum/total) if total else 0.0
    avg_bars= (bars_sum/total) if total else 0.0
    dc = state.get("daily_counters", {})
    align_rate = (dc.get("cross_align_match",0)/dc.get("cross_align_total",1)*100.0) if dc.get("cross_align_total",0)>0 else 0.0

    return {
        "date": date_str, "rows": rows,
        "winrate": winrate, "avg_pnl": avg_pnl, "avg_bars": avg_bars,
        "open_cnt": len(state.get("open_positions", [])),
        "dc": dc, "align_rate": align_rate
    }

# ================= BACKUP ZIP (JSON ONLY) =================
import zipfile

def create_and_send_backup():
    """data/ altındaki tüm .json dosyalarını ZIP’e alır ve Telegram’a gönderir."""
    try:
        base = DATA_DIR
        files_to_zip = []
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".json"):
                    files_to_zip.append(os.path.join(root, f))

        if not files_to_zip:
            log("[ZIP] Eklenecek JSON dosyası bulunamadı.")
            return

        zip_name = f"backup_{today_ist_date()}.zip"
        zip_path = os.path.join(base, zip_name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                arcname = os.path.relpath(file_path, base)
                zipf.write(file_path, arcname)
        log(f"[ZIP] {len(files_to_zip)} JSON eklendi → {zip_path}")

        with open(zip_path, "rb") as f:
            send_tg_document(os.path.basename(zip_path), f.read())
    except Exception as e:
        log(f"[ZIP_ERR] {e}")

# ================= STATE & DAILY TICK =================
def ensure_state(st):
    st = st or {}
    st.setdefault("open_positions", [])   # list of dict
    st.setdefault("last_daily_summary_date", "")
    t = today_ist_date()
    st.setdefault("daily_counters", {
        "date": t,
        "cross_normal": 0, "cross_premium": 0, "cross_ultra": 0,
        "scalp_premium": 0, "scalp_ultra": 0,
        "cross_align_total": 0, "cross_align_match": 0
    })
    st.setdefault("last_cross_seen", {})   # { "BTCUSDT_UP":  bar_close_ms }
    st.setdefault("last_scalp_seen", {})   # { "BTCUSDT_DOWN": bar_idx }
    return st

def roll_daily_counters_if_needed(state):
    t = today_ist_date()
    if state["daily_counters"].get("date") != t:
        state["daily_counters"] = {
            "date": t,
            "cross_normal": 0, "cross_premium": 0, "cross_ultra": 0,
            "scalp_premium": 0, "scalp_ultra": 0,
            "cross_align_total": 0, "cross_align_match": 0
        }

def maybe_daily_summary_and_send_json(state):
    if not DAILY_SUMMARY_ENABLED: return
    target_date = state.get("daily_counters", {}).get("date", today_ist_date())

    if now_ist_hhmm() >= DAILY_SUMMARY_TIME and state.get("last_daily_summary_date") != target_date:
        rep = build_daily_summary_for(target_date, state)
        dc = rep["dc"]
        send_tg(
            "📊 GÜNLÜK RAPOR (İstanbul)\n"
            f"📅 {rep['date']}\n"
            f"Signals → CROSS: 🟨{dc.get('cross_normal',0)} | 🟦{dc.get('cross_premium',0)} | 🟩{dc.get('cross_ultra',0)}\n"
            f"           SCALP: 🟦{dc.get('scalp_premium',0)} | 🟩{dc.get('scalp_ultra',0)}\n"
            f"Alignment (CROSS) → {rep['align_rate']:.1f}%\n"
            f"Winrate: {rep['winrate']:.1f}% | AvgPnL: {rep['avg_pnl']:.2f}% | AvgBars: {rep['avg_bars']:.1f}\n"
            f"Açık Pozisyon: {rep['open_cnt']}"
        )

        # Gün sonu JSON dosyalarını gönder
        for p in [OPEN_JSON, CLOSED_JSON, STATE_FILE, AI_LOG_JSON, WEEK_MODEL_FILE, WEEKEND_MODEL_FILE]:
            try:
                if p and os.path.exists(p):
                    with open(p, "rb") as f:
                        send_tg_document(os.path.basename(p), f.read())
            except Exception as e:
                log(f"[SEND_JSON_ERR] {p}: {e}")

        # ZIP yedekle ve gönder (sadece JSON’lar)
        create_and_send_backup()

        state["last_daily_summary_date"] = target_date

# ================= MAIN LOOP =================
def main():
    send_tg("🚀 EMA ULTRA v12.6.3 (JSONPersistent + 3-Bar Scalping + StateFix) started")
    log("🚀 v12.6.3 started")

    # State yükle / kalıcı tut
    raw_state = safe_load_json(STATE_FILE)
    if not raw_state:
        log("[STATE] Yeni state başlatıldı.")
    else:
        log("[STATE] Eski state yüklendi.")
    state = ensure_state(raw_state)

    # Semboller
    symbols = get_futures_symbols()
    if not symbols:
        send_tg("❌ Binance sembol listesi boş döndü. 3 sn sonra tekrar deniyorum…")
        time.sleep(3)
        symbols = get_futures_symbols()
        if not symbols:
            send_tg("⚠️ Hala boş. Render’da Restart/Manual Redeploy deneyebilirsin.")
            return
    log(f"OK, {len(symbols)} sembol taranacak...")

    bar = 0
    while True:
        bar += 1
        for sym in symbols:
            kl1  = get_klines(sym, INTERVAL_1H, limit=LIMIT_1H)
            kl4  = get_klines(sym, INTERVAL_4H, limit=LIMIT_4H)
            kl1d = get_klines(sym, INTERVAL_1D, limit=LIMIT_1D)
            if not kl1 or len(kl1)<120 or not kl4 or len(kl4)<50 or not kl1d or len(kl1d)<50:
                time.sleep(SLEEP_BETWEEN)
                continue

            # === CROSS (confirmed 1-bar) ===
            cross = run_cross_engine_confirmed(sym, kl1, kl4, kl1d)
            if cross:
                cross_key = f"{sym}_{cross['dir']}"
                if state["last_cross_seen"].get(cross_key) != cross["bar_close_ms"]:
                    tier, color = tier_color(cross["power"])
                    if tier != "NONE":
                        state["daily_counters"]["cross_align_total"] += 1
                        if cross["aligned"]:
                            state["daily_counters"]["cross_align_match"] += 1
                        if tier=="ULTRA":   state["daily_counters"]["cross_ultra"] += 1
                        elif tier=="PREMIUM": state["daily_counters"]["cross_premium"] += 1
                        elif tier=="NORMAL":  state["daily_counters"]["cross_normal"] += 1

                        align_tag = "✅ 4h/1D Trend Match" if cross["aligned"] else f"⚠️ Counter-Trend (4h {cross['trend4h']}, 1D {cross['trend1d']})"
                        boost_tag = " | ATR Boosted" if cross["atr_pct"] >= PARAM["ATR_BOOST_PCT"] else ""
                        confirm_tag = "Confirmed: ✅ (1-bar)" if cross["confirmed"] else "Confirmed: —"
                        adx_tag = f"ADX: {cross['adx']:.1f}"
                        vol_tag = f"VOL Spike x{cross['vol_mult']:.2f}" if cross["vol_mult"] >= VOL_SPIKE_MULT else f"VOL x{cross['vol_mult']:.2f}"
                        ai_tag = (f" | AI: {cross['ai_pred']*100:.2f}% exp, conf {cross['ai_conf']:.2f}" if (cross.get('ai_pred') is not None and cross.get('ai_conf') is not None) else "")
                        header_map = {"ULTRA":"ULTRA CROSS","PREMIUM":"PREMIUM CROSS","NORMAL":"CROSS SIGNAL"}
                        header = f"{color} {header_map[tier]}: {sym} ({INTERVAL_1H})"

                        send_tg(
                            f"{header}\n"
                            f"Entry: {cross['entry']:.4f} | TP: {cross['tp']:.4f} | SL: {cross['sl']:.4f}\n"
                            f"Direction: {cross['dir']}\n"
                            f"{align_tag}\n"
                            f"RSI: {cross['rsi']:.1f} | Power: {cross['power']:.1f}{boost_tag}\n"
                            f"Divergence: {cross['div']}\n"
                            f"Açı: {cross['ang_now']:+.1f}° | ΔAçı: {cross['ang_change']:.1f}°\n"
                            f"{confirm_tag} | {adx_tag} | {vol_tag}{ai_tag}\n"
                            f"ATR({ATR_PERIOD}): {cross['atr']:.6f} ({cross['atr_pct']*100:.2f}%)\n"
                            f"Time: {now_ist()}"
                        )

                        # Open state JSON’a yaz
                        open_rec = {
                            "symbol": sym, "type": "CROSS", "direction": cross["dir"],
                            "entry": float(cross["entry"]), "tp": float(cross["tp"]), "sl": float(cross["sl"]),
                            "power": float(cross["power"]), "rsi": float(cross["rsi"]), "divergence": cross["div"],
                            "ang_now": float(cross["ang_now"]), "ang_change": float(cross["ang_change"]),
                            "atr_pct": float(cross["atr_pct"]), "adx": float(cross["adx"]),
                            "vol_mult": float(cross["vol_mult"]),
                            "time_open": now_ist(), "bar": bar
                        }
                        state["open_positions"].append(open_rec)
                        append_json_list(OPEN_JSON, open_rec)

                    state["last_cross_seen"][cross_key] = cross["bar_close_ms"]

            # === SCALP (trend uyumlu + cooldown, 3-bar momentum) ===
            scalp = run_scalp_engine(sym, kl1, kl4, kl1d)
            if scalp:
                scalp_key = f"{sym}_{scalp['dir']}"
                last_idx = state["last_scalp_seen"].get(scalp_key)
                if last_idx is None or (bar - last_idx) > PARAM["SCALP_COOLDOWN_BARS"]:
                    tier, color = tier_color(scalp["power"])
                    if tier in ("PREMIUM","ULTRA"):
                        align_tag = "✅ 4h/1D Trend Match" if scalp["aligned"] else f"ℹ️ 4h OK, 1D {scalp['trend1d']}"
                        boost_tag = " ⚡ ATR Boost" if scalp["atr_pct"] >= PARAM["ATR_BOOST_PCT"] else ""
                        adx_tag   = f"ADX: {scalp['adx']:.1f}"
                        vol_tag   = f"VOL Spike x{scalp['vol_mult']:.2f}" if scalp['vol_mult'] >= VOL_SPIKE_MULT else f"VOL x{scalp['vol_mult']:.2f}"
                        ai_tag = (f" | AI: {scalp['ai_pred']*100:.2f}% exp, conf {scalp['ai_conf']:.2f}" if (scalp.get('ai_pred') is not None and scalp.get('ai_conf') is not None) else "")
                        header = f"{color} {'ULTRA' if tier=='ULTRA' else 'PREMIUM'} SCALP: {sym} ({INTERVAL_1H})"

                        send_tg(
                            f"{header}\n"
                            f"Entry: {scalp['entry']:.4f} | TP: {scalp['tp']:.4f} | SL: {scalp['sl']:.4f}\n"
                            f"Direction: {scalp['dir']} (4h trend uyumlu)\n"
                            f"{align_tag}\n"
                            f"RSI: {scalp['rsi']:.1f} | Power: {scalp['power']:.1f}{boost_tag}\n"
                            f"Açı: {scalp['ang_now']:+.1f}° | ΔAçı: {scalp['ang_change']:.1f}°\n"
                            f"{adx_tag} | {vol_tag}{ai_tag}\n"
                            f"Time: {now_ist()}"
                        )

                        state["last_scalp_seen"][scalp_key] = bar

                        open_rec = {
                            "symbol": sym, "type": "SCALP", "direction": scalp["dir"],
                            "entry": float(scalp["entry"]), "tp": float(scalp["tp"]), "sl": float(scalp["sl"]),
                            "power": float(scalp["power"]), "rsi": float(scalp["rsi"]), "divergence": scalp["div"],
                            "ang_now": float(scalp["ang_now"]), "ang_change": float(scalp["ang_change"]),
                            "atr_pct": float(scalp["atr_pct"]), "adx": float(scalp["adx"]),
                            "vol_mult": float(scalp["vol_mult"]),
                            "time_open": now_ist(), "bar": bar
                        }
                        state["open_positions"].append(open_rec)
                        append_json_list(OPEN_JSON, open_rec)

            time.sleep(SLEEP_BETWEEN)

        # === TP/SL takibi ===
        still_open=[]
        for t in state["open_positions"]:
            lp = get_last_price(t["symbol"])
            if lp is None:
                still_open.append(t); continue
            hit_tp = (lp >= t["tp"]) if t["direction"]=="UP" else (lp <= t["tp"])
            hit_sl = (lp <= t["sl"]) if t["direction"]=="UP" else (lp >= t["sl"])
            if not (hit_tp or hit_sl):
                still_open.append(t); continue

            res = "TP" if hit_tp else "SL"
            pnl = (lp - t["entry"])/t["entry"]*100 if t["direction"]=="UP" else (t["entry"] - lp)/t["entry"]*100
            bars_open = bar - t["bar"]

            send_tg(
                f"📘 {res} | {t['symbol']} {t['type']} {t['direction']}\n"
                f"Entry: {t['entry']:.4f}  Exit: {lp:.4f}\n"
                f"PnL: {pnl:.2f}%  Bars: {bars_open}"
            )

            closed_row = {
                "symbol": t["symbol"], "type": t["type"], "direction": t["direction"],
                "entry": float(t["entry"]), "exit": float(lp), "result": res,
                "pnl": float(f"{pnl:.4f}"), "bars": int(bars_open),
                "power": float(t.get("power",0.0)), "rsi": float(t.get("rsi",50.0)),
                "divergence": t.get("divergence",""),
                "ang_now": float(t.get("ang_now",0.0)), "ang_change": float(t.get("ang_change",0.0)),
                "atr_pct": float(t.get("atr_pct",0.0)), "adx": float(t.get("adx",0.0)),
                "vol_mult": float(t.get("vol_mult",1.0)),
                "time_open": t["time_open"], "time_close": now_ist(),
                "trend_match": int(1)  # açılışta hizalıydı
            }
            append_json_list(CLOSED_JSON, closed_row)

        state["open_positions"] = still_open

        # Günlük yönetim
        maybe_daily_summary_and_send_json(state)
        roll_daily_counters_if_needed(state)

        # State kaydet
        safe_save_json(STATE_FILE, state)
        log(f"Scan done | Open: {len(state['open_positions'])} | Params: CrossTP {PARAM['CROSS_TP_PCT']:.4f}, ScalpTP {PARAM['SCALP_TP_PCT']:.4f}, ATR%≥{PARAM['ATR_BOOST_PCT']*100:.2f}")
        time.sleep(SCAN_INTERVAL)

# ================= MAIN =================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        try:
            send_tg(f"❗ Bot hata verdi: {e}")
        except:
            pass
