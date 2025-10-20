# ==============================================================
#  📘 EMA ULTRA v12.2 — Full System + AI Param Archive
#  Binance Futures EMA+ATR+RSI+ADX+AI Learning (Render-ready)
# ==============================================================

import os, json, csv, io, time, requests
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# ================= ORTAM / DOSYA YOLLARI =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

DATA_DIR   = os.getenv("DATA_DIR", "/data")
LEARN_DIR  = os.getenv("LEARN_DIR", os.path.join(DATA_DIR, "data_learning"))
STATE_FILE = os.getenv("STATE_FILE", os.path.join(DATA_DIR, "alerts.json"))
OPEN_CSV   = os.getenv("OPEN_CSV",   os.path.join(DATA_DIR, "open_positions.csv"))
CLOSED_CSV = os.getenv("CLOSED_CSV", os.path.join(DATA_DIR, "closed_trades.csv"))
LOG_FILE   = os.getenv("LOG_FILE",   os.path.join(DATA_DIR, "log.txt"))

PARAM_HISTORY_DIR = os.path.join(LEARN_DIR, "params_history")
WEEK_MODEL_FILE   = os.path.join(LEARN_DIR, "week_model.json")
WEEKEND_MODEL_FILE= os.path.join(LEARN_DIR, "weekend_model.json")
WEEK_MODEL_PKL    = os.path.join(LEARN_DIR, "week_model.pkl")
WEEKEND_MODEL_PKL = os.path.join(LEARN_DIR, "weekend_model.pkl")

# ================= GENEL AYARLAR =================
INTERVAL_1H, INTERVAL_4H, INTERVAL_1D = "1h", "4h", "1d"
LIMIT_1H, LIMIT_4H, LIMIT_1D = 500, 300, 250

EMA_1H = (7, 25, 99)    # cross motoru 1h
ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14
VOL_MA_PERIOD = 20
VOL_SPIKE_MULT = 1.8

CROSS_CONFIRM_BARS = 1           # 1-bar confirmed cross
SCAN_INTERVAL = 300              # 5 dk
SLEEP_BETWEEN = 0.12
DAILY_SUMMARY_ENABLED = True
DAILY_SUMMARY_TIME = "23:59"     # IST

LEARN_DAYS = 14  # 14 gün öğren sonra uygula

# ================= RUNTIME PARAMS (AI ile nazikçe değişir) =================
PARAM = {
    "POWER_NORMAL_MIN": 60.0,
    "POWER_PREMIUM_MIN": 68.0,
    "POWER_ULTRA_MIN": 75.0,

    "ATR_BOOST_PCT": 0.004,      # ATR% eşiği
    "ATR_BOOST_ADD": 5.0,        # power'a bonus

    "ADX_BASE": 25.0,
    "ADX_MAX_ADD": 10.0,
    "VOL_MAX_ADD": 7.0,

    "SCALP_TP_PCT": 0.006,       # %0.6
    "SCALP_SL_PCT": 0.10,        # %10
    "CROSS_TP_PCT": 0.010,       # %1.0
    "CROSS_SL_PCT": 0.030,       # %3.0

    "SCALP_COOLDOWN_BARS": 3     # tekrarsızlaşma (aynı yön)
}

# ================= ZAMAN / LOG / TG =================
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)
def now_ist(): return now_ist_dt().isoformat()
def today_ist_date(): return now_ist_dt().strftime("%Y-%m-%d")
def now_ist_hhmm(): return now_ist_dt().strftime("%H:%M")
def is_weekend_ist(): return now_ist_dt().weekday() in (5, 6)

def log(msg):
    print(msg, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_ist()} - {msg}\n")
    except:
        pass

def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] env eksik"); return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        log(f"[TG] {text.splitlines()[0][:60]} ...")
    except Exception as e:
        log(f"[TG] send error: {e}")

def send_tg_document(filename, bytes_content):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] env eksik"); return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    files = {'document': (filename, bytes_content)}
    data = {'chat_id': CHAT_ID, 'caption': filename}
    try:
        requests.post(url, data=data, files=files, timeout=30)
        log(f"[TG] document sent: {filename}")
    except Exception as e:
        log(f"[TG] doc error: {e}")

# ================= DOSYA / STATE / CSV =================
def safe_load_json(path, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return {} if default is None else default

def safe_save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"[ERR] save_json {path}: {e}")

# === GÜVENLİ DİZİN VE CSV OLUŞTURMA ===
def ensure_dir(path):
    try: os.makedirs(path, exist_ok=True)
    except: pass

def ensure_csv(path, headers):
    """Dosya yoksa dizini ve CSV dosyasını otomatik oluşturur."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            log(f"[INIT] CSV oluşturuldu: {path}")
    except Exception as e:
        log(f"[ERR] CSV oluşturulamadı: {path} → {e}")

# İlk kurulum güvenliği
ensure_dir(DATA_DIR)
ensure_dir(LEARN_DIR)
ensure_dir(PARAM_HISTORY_DIR)
ensure_csv(OPEN_CSV,   ["symbol","type","direction","entry","tp","sl","power","rsi","divergence","time_open"])
ensure_csv(CLOSED_CSV, ["symbol","type","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close"])
log("[INIT] ensured /data folders and CSV files")

# ================= İNDİKATÖRLER =================
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

# ================= POWER / ETİKETLER =================
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

# ================= BINANCE =================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v12.2", "Accept": "application/json"})
BASE = "https://fapi.binance.com"

def get_futures_symbols():
    try:
        r = SESSION.get(BASE + "/fapi/v1/exchangeInfo", timeout=15)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except:
        return []

def get_klines(symbol, interval, limit=500):
    url = BASE + "/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
            if int(data[-1][6]) > now_ms: data = data[:-1]
            return data
    except: pass
    return []

def get_last_price(symbol):
    try:
        url = f"{BASE}/fapi/v1/ticker/price?symbol={symbol}"
        r = SESSION.get(url, timeout=8).json()
        return float(r["price"])
    except: return None

# ================= TREND HELPERS =================
def ema_trend_dir(closes):
    e7, e25 = ema(closes,7), ema(closes,25)
    return "UP" if e7[-1] > e25[-1] else "DOWN"

def trend_alignment(signal_dir, closes_4h, closes_1d):
    d4 = ema_trend_dir(closes_4h)
    d1 = ema_trend_dir(closes_1d)
    match = (signal_dir == d4) and (signal_dir == d1)
    return match, d4, d1

# ================= ENGINES =================
def run_cross_engine_confirmed(sym, kl1, kl4, kl1d):
    closes = [float(k[4]) for k in kl1]
    ema7, ema25 = ema(closes,7), ema(closes,25)

    cross_dir = None
    if CROSS_CONFIRM_BARS == 1:
        prev_diff    = ema7[-3] - ema25[-3]
        cross_diff   = ema7[-2] - ema25[-2]
        confirm_diff = ema7[-1] - ema25[-1]
        if prev_diff < 0 and cross_diff > 0 and confirm_diff > 0: cross_dir = "UP"
        if prev_diff > 0 and cross_diff < 0 and confirm_diff < 0: cross_dir = "DOWN"
        bar_close_ms = int(kl1[-1][6])
    else:
        prev_diff = ema7[-2] - ema25[-2]
        curr_diff = ema7[-1] - ema25[-1]
        if prev_diff < 0 and curr_diff > 0: cross_dir = "UP"
        if prev_diff > 0 and curr_diff < 0: cross_dir = "DOWN"
        bar_close_ms = int(kl1[-1][6])

    if not cross_dir: return None

    highs = [float(k[2]) for k in kl1]
    lows  = [float(k[3]) for k in kl1]
    vols  = [float(k[5]) for k in kl1]
    atr_now = atr_series(highs, lows, closes, ATR_PERIOD)[-1]
    rsi_all = rsi(closes, RSI_PERIOD)
    rsi_now, rsi_prev = rsi_all[-1], rsi_all[-2]
    s_now  = ema7[-1] - ema7[-4]
    s_prev = ema7[-2] - ema7[-5]
    price  = closes[-1]

    adx_all, _, _ = adx_series(highs, lows, closes, ADX_PERIOD)
    adx_now = adx_all[-1]
    vol_ma = sma(vols, VOL_MA_PERIOD)[-1]

    base = power_base(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr_after_boost = base + boost
    pwr_final, vol_mult = power_with_adx_vol(pwr_after_boost, adx_now, vols[-1], vol_ma)

    div = rsi_divergence(closes[-1], closes[-2], rsi_now, rsi_prev)

    closes4  = [float(k[4]) for k in kl4]
    closes1d = [float(k[4]) for k in kl1d]
    aligned, d4, d1 = trend_alignment(cross_dir, closes4, closes1d)

    entry = price
    tp = entry * (1 + PARAM["CROSS_TP_PCT"] if cross_dir == "UP" else 1 - PARAM["CROSS_TP_PCT"])
    sl = entry * (1 - PARAM["CROSS_SL_PCT"] if cross_dir == "UP" else 1 + PARAM["CROSS_SL_PCT"])

    return {
        "symbol": sym, "type": "CROSS", "dir": cross_dir,
        "entry": entry, "tp": tp, "sl": sl,
        "power": pwr_final, "rsi": rsi_now, "div": div,
        "atr": atr_now, "atr_pct": atr_pct, "bar_close_ms": bar_close_ms,
        "aligned": aligned, "trend4h": d4, "trend1d": d1, "confirmed": CROSS_CONFIRM_BARS==1,
        "adx": adx_now, "vol_mult": vol_mult
    }

def run_scalp_engine(sym, kl1, kl4, kl1d):
    closes1 = [float(k[4]) for k in kl1]
    ema7_1  = ema(closes1, 7)
    if len(ema7_1) < 6: return None
    s_now  = ema7_1[-1] - ema7_1[-4]
    s_prev = ema7_1[-2] - ema7_1[-5]
    slope_dir = "UP" if (s_prev < 0 and s_now > 0) else ("DOWN" if (s_prev > 0 and s_now < 0) else None)
    if not slope_dir: return None

    closes4  = [float(k[4]) for k in kl4]
    closes1d = [float(k[4]) for k in kl1d]
    d4 = ema_trend_dir(closes4)
    if slope_dir != d4: return None
    d1 = ema_trend_dir(closes1d)
    aligned = (slope_dir == d4) and (slope_dir == d1)

    highs1 = [float(k[2]) for k in kl1]
    lows1  = [float(k[3]) for k in kl1]
    vols1  = [float(k[5]) for k in kl1]
    atr_now = atr_series(highs1, lows1, closes1, ATR_PERIOD)[-1]
    rsi_now = rsi(closes1, RSI_PERIOD)[-1]
    price   = closes1[-1]

    adx_all, _, _ = adx_series(highs1, lows1, closes1, ADX_PERIOD)
    adx_now = adx_all[-1]
    vol_ma  = sma(vols1, VOL_MA_PERIOD)[-1]

    base = power_base(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr_after_boost = base + boost
    pwr_final, vol_mult = power_with_adx_vol(pwr_after_boost, adx_now, vols1[-1], vol_ma)

    rsi_prev = rsi(closes1, RSI_PERIOD)[-2]
    div = rsi_divergence(closes1[-1], closes1[-2], rsi_now, rsi_prev)

    if pwr_final < PARAM["POWER_PREMIUM_MIN"]: return None

    entry = price
    tp = entry * (1 + PARAM["SCALP_TP_PCT"] if slope_dir == "UP" else 1 - PARAM["SCALP_TP_PCT"])
    sl = entry * (1 - PARAM["SCALP_SL_PCT"] if slope_dir == "UP" else 1 + PARAM["SCALP_SL_PCT"])

    return {
        "symbol": sym, "type": "SCALP", "dir": slope_dir,
        "entry": entry, "tp": tp, "sl": sl,
        "power": pwr_final, "rsi": rsi_now, "div": div,
        "atr": atr_now, "atr_pct": atr_pct,
        "aligned": aligned, "trend4h": d4, "trend1d": d1,
        "adx": adx_now, "vol_mult": vol_mult
    }
# ================= GÜNLÜK ÖZET / CSV GÖNDER =================
def build_daily_summary_for(date_str, state):
    rows = []
    try:
        with open(CLOSED_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("time_close","").startswith(date_str):
                    rows.append(row)
    except FileNotFoundError:
        pass
    total = len(rows)
    wins = sum(1 for r in rows if r.get("result")=="TP")
    pnl_sum = sum(float(r.get("pnl","0") or 0) for r in rows)
    bars_sum= sum(int(r.get("bars","0") or 0) for r in rows)

    winrate = (wins/total*100.0) if total else 0.0
    avg_pnl = (pnl_sum/total) if total else 0.0
    avg_bars= (bars_sum/total) if total else 0.0
    dc = state.get("daily_counters", {})
    align_rate = (dc["cross_align_match"]/dc["cross_align_total"]*100.0) if dc.get("cross_align_total",0)>0 else 0.0

    return {
        "date": date_str, "rows": rows,
        "winrate": winrate, "avg_pnl": avg_pnl, "avg_bars": avg_bars,
        "open_cnt": len(state.get("open_positions", [])),
        "dc": dc, "align_rate": align_rate
    }

def daily_csv_bytes(rows, date_str):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["symbol","type","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close"])
    for r in rows:
        w.writerow([r.get("symbol"), r.get("type"), r.get("direction"), r.get("entry"), r.get("exit"),
                    r.get("result"), r.get("pnl"), r.get("bars"), r.get("power"), r.get("rsi"),
                    r.get("divergence"), r.get("time_open"), r.get("time_close")])
    return out.getvalue().encode("utf-8"), f"closed_trades_{date_str}.csv"

# ================= AI LEARNING: DATA & MODEL =================
def clamp(v, lo, hi): return max(lo, min(hi, v))

def apply_model_to_params(model):
    win = model.get("winrate", 0.0)
    avg_bars = model.get("avg_bars", 0.0)
    avg_pnl  = model.get("avg_pnl", 0.0)

    if win >= 70:
        PARAM["CROSS_TP_PCT"] = clamp(PARAM["CROSS_TP_PCT"] + 0.0003, 0.006, 0.015)
        PARAM["SCALP_TP_PCT"] = clamp(PARAM["SCALP_TP_PCT"] + 0.0002, 0.004, 0.012)
        PARAM["CROSS_SL_PCT"] = clamp(PARAM["CROSS_SL_PCT"] - 0.002, 0.02, 0.06)
    elif win < 50:
        PARAM["POWER_NORMAL_MIN"]  = clamp(PARAM["POWER_NORMAL_MIN"] + 2, 55, 72)
        PARAM["POWER_PREMIUM_MIN"] = clamp(PARAM["POWER_PREMIUM_MIN"] + 2, 60, 80)
        PARAM["CROSS_TP_PCT"] = clamp(PARAM["CROSS_TP_PCT"] - 0.0005, 0.004, 0.012)
        PARAM["SCALP_TP_PCT"] = clamp(PARAM["SCALP_TP_PCT"] - 0.0004, 0.003, 0.010)

    if avg_bars and avg_bars > 10:
        PARAM["CROSS_TP_PCT"] = clamp(PARAM["CROSS_TP_PCT"] - 0.0003, 0.004, 0.012)
        PARAM["SCALP_TP_PCT"] = clamp(PARAM["SCALP_TP_PCT"] - 0.0002, 0.003, 0.010)

    if avg_pnl > 0.8:
        PARAM["ATR_BOOST_PCT"] = clamp(PARAM["ATR_BOOST_PCT"] - 0.0003, 0.0025, 0.006)
    elif avg_pnl < 0.0:
        PARAM["ATR_BOOST_PCT"] = clamp(PARAM["ATR_BOOST_PCT"] + 0.0003, 0.0025, 0.010)

def model_path_for_today():
    return WEEKEND_MODEL_FILE if is_weekend_ist() else WEEK_MODEL_FILE

def load_model_meta(path):
    m = safe_load_json(path, default={})
    if not m:
        m = {"days":0,"active":False,"winrate":0.0,"avg_pnl":0.0,"avg_bars":0.0,"samples":0}
    return m

def collect_training_df(days_back=14, weekend=False):
    try:
        df = pd.read_csv(CLOSED_CSV)
    except FileNotFoundError:
        return pd.DataFrame()

    if df.empty: return df
    df["time_close_dt"] = pd.to_datetime(df["time_close"], errors="coerce")
    df = df.dropna(subset=["time_close_dt"]).copy()
    cutoff = now_ist_dt() - timedelta(days=days_back)
    df = df[df["time_close_dt"] >= cutoff]
    df["weekday"] = df["time_close_dt"].dt.weekday
    if weekend: df = df[df["weekday"].isin([5,6])]
    else:       df = df[~df["weekday"].isin([5,6])]
    if df.empty: return df

    df["type_is_scalp"] = (df["type"].astype(str).str.upper()=="SCALP").astype(int)
    df["dir_is_up"]     = (df["direction"].astype(str).str.upper()=="UP").astype(int)
    div = df["divergence"].astype(str).str.lower().fillna("neutral")
    df["div_bull"] = (div=="bullish").astype(int)
    df["div_bear"] = (div=="bearish").astype(int)

    df["pnl"]   = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0.0)
    df["rsi"]   = pd.to_numeric(df["rsi"], errors="coerce").fillna(50.0)
    df["bars"]  = pd.to_numeric(df["bars"], errors="coerce").fillna(0)

    if "trend_match" not in df.columns: df["trend_match"] = 0
    if "atr_pct" not in df.columns:     df["atr_pct"] = 0.0

    feats = ["type_is_scalp","dir_is_up","power","rsi","div_bull","div_bear","weekday","bars","trend_match","atr_pct"]
    df = df.dropna(subset=["pnl"]).copy()
    return df[feats + ["pnl"]]

def train_random_forest(df):
    if df is None or df.empty or len(df) < 40:
        return None, None
    feats = [c for c in df.columns if c != "pnl"]
    X = df[feats].values
    y = df["pnl"].values
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    importances = dict(zip(feats, model.feature_importances_))
    return model, importances

def save_param_snapshot(model_meta, weekend=False):
    os.makedirs(PARAM_HISTORY_DIR, exist_ok=True)
    tag = "weekend" if weekend else "week"
    ts = now_ist_dt().strftime("%Y-%m-%d_%H%M%S")
    fname = os.path.join(PARAM_HISTORY_DIR, f"{ts}_{tag}_params.json")
    snapshot = {"timestamp": now_ist(), "model_type": tag, "ai_meta": model_meta, "params": PARAM}
    safe_save_json(fname, snapshot)
    log(f"[AI] Param snapshot saved → {fname}")

def ai_learning_update_and_apply():
    os.makedirs(LEARN_DIR, exist_ok=True)
    weekend = is_weekend_ist()
    json_path = WEEKEND_MODEL_FILE if weekend else WEEK_MODEL_FILE
    pkl_path  = WEEKEND_MODEL_PKL  if weekend else WEEK_MODEL_PKL

    model_meta = load_model_meta(json_path)
    df = collect_training_df(days_back=14, weekend=weekend)

    metrics = {"winrate":0.0,"avg_pnl":0.0,"avg_bars":0.0,"samples":0}
    rf_model, importances = None, None
    if df is not None and not df.empty:
        feats = [c for c in df.columns if c!="pnl"]
        X = df[feats].values
        y = df["pnl"].values
        rf_model, importances = train_random_forest(df)
        metrics["samples"] = len(df)
        metrics["winrate"] = float((y>0).mean()*100.0)
        metrics["avg_pnl"] = float(np.mean(y))
        metrics["avg_bars"]= float(np.mean(df["bars"]))

    model_meta["days"]     = model_meta.get("days",0) + 1
    model_meta["samples"]  = metrics["samples"]
    model_meta["winrate"]  = metrics["winrate"]
    model_meta["avg_pnl"]  = metrics["avg_pnl"]
    model_meta["avg_bars"] = metrics["avg_bars"]
    if not model_meta.get("active") and model_meta["days"] >= LEARN_DAYS:
        model_meta["active"] = True

    if rf_model is not None:
        try:
            dump(rf_model, pkl_path)
            model_meta["has_model"] = True
            if importances:
                top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                model_meta["top_features"] = [{"name":k, "imp":float(v)} for k,v in top]
        except Exception as e:
            log(f"[AI] model save error: {e}")
            model_meta["has_model"] = False
    else:
        model_meta["has_model"] = False

    safe_save_json(json_path, model_meta)

    if model_meta.get("active"):
        apply_model_to_params(model_meta)

    tf_label = "Haftasonu" if weekend else "Haftaiçi"
    feat_txt = ""
    if model_meta.get("top_features"):
        feat_txt = "\n".join([f"{i+1}️⃣ {f['name']} ({f['imp']*100:.1f}%)" for i,f in enumerate(model_meta["top_features"])])
    send_tg(
        f"🧠 AI Model Update [{tf_label}] | Days: {model_meta['days']}\n"
        f"Samples: {model_meta['samples']} | Winrate: {model_meta['winrate']:.1f}% | "
        f"AvgPnL: {model_meta['avg_pnl']:.2f}% | AvgBars: {model_meta['avg_bars']:.1f}\n"
        f"Active: {'✅' if model_meta.get('active') else '⏳'} | "
        f"Params → CrossTP {PARAM['CROSS_TP_PCT']:.4f} SL {PARAM['CROSS_SL_PCT']:.3f} | "
        f"ScalpTP {PARAM['SCALP_TP_PCT']:.4f} SL {PARAM['SCALP_SL_PCT']:.3f} | "
        f"Power {PARAM['POWER_NORMAL_MIN']:.0f}/{PARAM['POWER_PREMIUM_MIN']:.0f}/{PARAM['POWER_ULTRA_MIN']:.0f}"
        + (f"\nTop Features:\n{feat_txt}" if feat_txt else "")
    )
    save_param_snapshot(model_meta, weekend)

# ================= STATE =================
def ensure_state(st):
    st.setdefault("last_slope_dir", {})
    st.setdefault("open_positions", [])
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

# ================= GÜNLÜK ÖZET & LEARNING TETİK =================
def maybe_daily_summary_and_learn(state):
    if not DAILY_SUMMARY_ENABLED: return
    if now_ist_hhmm() >= DAILY_SUMMARY_TIME and state.get("last_daily_summary_date") != today_ist_date():
        rep = build_daily_summary_for(today_ist_date(), state)
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
        if rep["rows"]:
            csv_bytes, fname = daily_csv_bytes(rep["rows"], rep["date"])
            send_tg_document(fname, csv_bytes)
        ai_learning_update_and_apply()
        state["last_daily_summary_date"] = today_ist_date()

# ================= ANA DÖNGÜ =================
def main():
    send_tg("🚀 EMA ULTRA v12.2 AI Engine started… (Render)")
    log("🚀 v12.2 başladı (Full + AI Param Archive)")
    state = ensure_state(safe_load_json(STATE_FILE))

    symbols = get_futures_symbols()
    if not symbols:
        send_tg("❌ Binance sembol listesi boş döndü.")
        time.sleep(10)
        symbols = get_futures_symbols()
        if not symbols:
            send_tg("⚠️ Tekrar deniyorum; hala boşsa redeploy yapmayı deneyebilirsin.")
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
                continue

            # === CROSS (confirmed) ===
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
                        header_map = {"ULTRA":"ULTRA CROSS","PREMIUM":"PREMIUM CROSS","NORMAL":"CROSS SIGNAL"}
                        header = f"{color} {header_map[tier]}: {sym} ({INTERVAL_1H})"

                        send_tg(
                            f"{header}\n"
                            f"Entry: {cross['entry']:.4f} | TP: {cross['tp']:.4f} | SL: {cross['sl']:.4f}\n"
                            f"Direction: {cross['dir']}\n"
                            f"{align_tag}\n"
                            f"RSI: {cross['rsi']:.1f} | Power: {cross['power']:.1f}{boost_tag}\n"
                            f"Divergence: {cross['div']}\n"
                            f"{confirm_tag} | {adx_tag} | {vol_tag}\n"
                            f"ATR({ATR_PERIOD}): {cross['atr']:.6f} ({cross['atr_pct']*100:.2f}%)\n"
                            f"Time: {now_ist()}"
                        )
                        with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([sym, "CROSS", cross["dir"], f"{cross['entry']:.4f}", f"{cross['tp']:.4f}", f"{cross['sl']:.4f}",
                                                    f"{cross['power']:.1f}", f"{cross['rsi']:.1f}", cross["div"], now_ist()])
                        state["open_positions"].append({
                            "symbol": sym, "type": "CROSS", "dir": cross["dir"],
                            "entry": cross["entry"], "tp": cross["tp"], "sl": cross["sl"],
                            "power": cross["power"], "rsi": cross["rsi"], "div": cross["div"],
                            "open": now_ist(), "bar": bar
                        })
                    state["last_cross_seen"][cross_key] = cross["bar_close_ms"]

            # === SCALP (trend uyumlu + cooldown) ===
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
                        vol_tag   = f"VOL Spike x{scalp['vol_mult']:.2f}" if scalp["vol_mult"] >= VOL_SPIKE_MULT else f"VOL x{scalp['vol_mult']:.2f}"
                        header = f"{color} {'ULTRA' if tier=='ULTRA' else 'PREMIUM'} SCALP: {sym} ({INTERVAL_1H})"
                        send_tg(
                            f"{header}\n"
                            f"Entry: {scalp['entry']:.4f} | TP: {scalp['tp']:.4f} | SL: {scalp['sl']:.4f}\n"
                            f"Direction: {scalp['dir']} (4h trend uyumlu)\n"
                            f"{align_tag}\n"
                            f"RSI: {scalp['rsi']:.1f} | Power: {scalp['power']:.1f}{boost_tag}\n"
                            f"{adx_tag} | {vol_tag}\n"
                            f"Time: {now_ist()}"
                        )
                        if tier=="ULTRA": state["daily_counters"]["scalp_ultra"] += 1
                        else:             state["daily_counters"]["scalp_premium"] += 1

                        state["last_scalp_seen"][scalp_key] = bar
                        with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([sym, "SCALP", scalp["dir"], f"{scalp['entry']:.4f}", f"{scalp['tp']:.4f}", f"{scalp['sl']:.4f}",
                                                    f"{scalp['power']:.1f}", f"{scalp['rsi']:.1f}", scalp["div"], now_ist()])
                        state["open_positions"].append({
                            "symbol": sym, "type": "SCALP", "dir": scalp["dir"],
                            "entry": scalp["entry"], "tp": scalp["tp"], "sl": scalp["sl"],
                            "power": scalp["power"], "rsi": scalp["rsi"], "div": scalp["div"],
                            "open": now_ist(), "bar": bar
                        })

            time.sleep(SLEEP_BETWEEN)

        # === Açık pozisyonları yönet (TP/SL) ===
        still_open=[]
        for t in state["open_positions"]:
            lp = get_last_price(t["symbol"])
            if lp is None:
                still_open.append(t); continue
            hit_tp = (lp >= t["tp"]) if t["dir"]=="UP" else (lp <= t["tp"])
            hit_sl = (lp <= t["sl"]) if t["dir"]=="UP" else (lp >= t["sl"])
            if not (hit_tp or hit_sl):
                still_open.append(t); continue

            res = "TP" if hit_tp else "SL"
            pnl = (lp - t["entry"])/t["entry"]*100 if t["dir"]=="UP" else (t["entry"] - lp)/t["entry"]*100
            bars_open = bar - t["bar"]
            send_tg(
                f"📘 {res} | {t['symbol']} {t['type']} {t['dir']}\n"
                f"Entry: {t['entry']:.4f}  Exit: {lp:.4f}\n"
                f"PnL: {pnl:.2f}%  Bars: {bars_open}\n"
                f"From: {t['type']}"
            )
            with open(CLOSED_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([t["symbol"], t["type"], t["dir"], f"{t['entry']:.4f}", f"{lp:.4f}", res, f"{pnl:.2f}", bars_open,
                                        f"{t['power']:.1f}", f"{t['rsi']:.1f}", t["div"], t["open"], now_ist()])
        state["open_positions"] = still_open

        # Günlük yönetim
        roll_daily_counters_if_needed(state)
        maybe_daily_summary_and_learn(state)

        # Kaydet
        safe_save_json(STATE_FILE, state)
        log(f"Scan done | Open: {len(state['open_positions'])} | Params: CrossTP {PARAM['CROSS_TP_PCT']:.4f}, ScalpTP {PARAM['SCALP_TP_PCT']:.4f}, ATR%≥{PARAM['ATR_BOOST_PCT']*100:.2f}")
        time.sleep(SCAN_INTERVAL)

# ================= MAIN =================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        send_tg(f"❗ Bot hata verdi: {e}")
