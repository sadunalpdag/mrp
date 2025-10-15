import os
import json
import time
import random
from datetime import datetime, timezone
import requests

# ========= AYARLAR =========
EMA_7, EMA_25, EMA_99 = 7, 25, 99
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]

# Varsayılan interval bazlı eşikler
DEFAULT_THRESHOLDS = {
    "1h": {"ATR_MIN_PCT": 0.0035, "ATR_SLOPE_MULT": 0.6},  # 0.35% ve 0.6×ATR
    "4h": {"ATR_MIN_PCT": 0.0025, "ATR_SLOPE_MULT": 0.5},  # 0.25% ve 0.5×ATR
    "1d": {"ATR_MIN_PCT": 0.0015, "ATR_SLOPE_MULT": 0.4},  # 0.15% ve 0.4×ATR
}

# Global fallback (env ile verilebilir)
GLOBAL_ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
GLOBAL_ATR_MIN_PCT = float(os.getenv("ATR_MIN_PCT", "0.003"))
GLOBAL_ATR_SLOPE_MULT = float(os.getenv("ATR_SLOPE_MULT", "0.5"))

SLEEP_BETWEEN = 0.25
SCAN_INTERVAL = 600
BASE_BACKOFF = 1.0
MAX_BACKOFF = 32.0
MAX_TRIES_PER_CALL = 6

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

ALERTS_FILE = os.getenv("ALERTS_FILE", "/data/alerts.json")
if not os.path.isdir("/data"):
    ALERTS_FILE = "alerts.json"

LOG_FILE = os.getenv("LOG_FILE", "log.txt")
# ===========================

# ---------- Yardımcılar ----------
def nowiso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {msg}\n")
    except Exception:
        pass

def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("Telegram env değişkenleri eksik (BOT_TOKEN/CHAT_ID). Mesaj gönderilmedi.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        log(f"[TG] {text}")
    except Exception as e:
        log(f"Telegram hatası: {e}")

def safe_load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"alerts.json okunamadı: {e}")
    return {}

def safe_save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"alerts.json yazılamadı: {e}")

def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals

def slope_value(series, lookback=3):
    if len(series) < (lookback + 1):
        return 0.0
    return series[-1] - series[-lookback]

# ---- ATR Hesabı ----
def true_ranges(highs, lows, closes):
    trs = []
    for i in range(len(highs)):
        h, l = highs[i], lows[i]
        if i == 0:
            trs.append(h - l)
        else:
            prev_c = closes[i - 1]
            trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    return trs

def atr_series(highs, lows, closes, period=14):
    trs = true_ranges(highs, lows, closes)
    if len(trs) < period:
        return [0.0] * len(trs)
    atrs = [0.0] * len(trs)
    first = sum(trs[:period]) / period
    atrs[period - 1] = first
    for i in range(period, len(trs)):
        atrs[i] = (atrs[i - 1] * (period - 1) + trs[i]) / period
    for i in range(period - 1):
        atrs[i] = trs[i]
    return atrs
# ----------------------------------

# ---------- HTTP / Binance ----------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "EMA-Scanner/1.2-ATR-Interval (+https://example.local)",
    "Accept": "application/json",
    "Connection": "keep-alive",
})

def _request_json(url, params=None):
    backoff = BASE_BACKOFF
    tries = 0
    while True:
        tries += 1
        try:
            r = SESSION.get(url, params=params, timeout=10)
            if r.status_code in (418, 429):
                log(f"{url} rate/ban {r.status_code}; try {tries}/{MAX_TRIES_PER_CALL}")
                if tries >= MAX_TRIES_PER_CALL:
                    return None
                sleep_for = min(MAX_BACKOFF, backoff) + random.uniform(0, 0.5)
                time.sleep(sleep_for)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log(f"HTTPError {url}: {e}")
            return None
        except Exception as e:
            log(f"İstek hatası {url}: {e}")
            if tries >= MAX_TRIES_PER_CALL:
                return None
            sleep_for = min(MAX_BACKOFF, backoff) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            backoff *= 2

def get_futures_symbols():
    data = _request_json("https://fapi.binance.com/fapi/v1/exchangeInfo")
    if not data or "symbols" not in data:
        log(f"exchangeInfo beklenmedik: {data}")
        return []
    return [
        s["symbol"]
        for s in data["symbols"]
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
    ]

def get_klines(symbol, interval, limit=LIMIT):
    data = _request_json(
        "https://fapi.binance.com/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not data:
        return []
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    last_close_time = int(data[-1][6])
    if last_close_time > now_ms:
        data = data[:-1]
    return data
# ----------------------------------

# ---------- Analiz ----------
def last_cross_info(ema_fast, ema_slow):
    last_cross = None
    direction = None
    for i in range(1, len(ema_fast)):
        prev_diff = ema_fast[i - 1] - ema_slow[i - 1]
        curr_diff = ema_fast[i] - ema_slow[i]
        if prev_diff < 0 and curr_diff > 0:
            last_cross = i; direction = "UP"
        elif prev_diff > 0 and curr_diff < 0:
            last_cross = i; direction = "DOWN"
    if last_cross is None:
        return None, None
    bars_ago = len(ema_fast) - last_cross - 1
    return direction, bars_ago

def recent_slope_reversal(ema7, max_bars=3, lookback=3):
    if len(ema7) < lookback + max_bars + 2:
        return None, None
    dirs = []
    for shift in range(max_bars + 1):
        idx_now = -1 - shift
        idx_past = idx_now - lookback
        if abs(idx_past) > len(ema7):
            break
        val = ema7[idx_now] - ema7[idx_past]
        if val > 0: dirs.append("UP")
        elif val < 0: dirs.append("DOWN")
        else: dirs.append("FLAT")
    base = dirs[0]
    for i in range(1, len(dirs)):
        d = dirs[i]
        if base == "UP" and d == "DOWN":
            return "DOWN", i
        if base == "DOWN" and d == "UP":
            return "UP", i
    return None, None
# ----------------------------------

# ---------- Eşik Yardımcıları ----------
def env_float(name, default):
    try:
        v = os.getenv(name)
        return float(v) if v is not None else default
    except Exception:
        return default

def thresholds_for_interval(interval):
    """
    Öncelik sırası:
    1) ATR_MIN_PCT_{INTERVAL} / ATR_SLOPE_MULT_{INTERVAL} (örn. 1H, 4H, 1D)
    2) Global ATR_MIN_PCT / ATR_SLOPE_MULT (ENV)
    3) DEFAULT_THRESHOLDS[interval]
    """
    key = interval.upper()
    # 1) Interval spesifik ENV
    pct_env = os.getenv(f"ATR_MIN_PCT_{key}")
    mult_env = os.getenv(f"ATR_SLOPE_MULT_{key}")

    if pct_env is not None or mult_env is not None:
        pct = env_float(f"ATR_MIN_PCT_{key}", DEFAULT_THRESHOLDS.get(interval, {}).get("ATR_MIN_PCT", GLOBAL_ATR_MIN_PCT))
        mult = env_float(f"ATR_SLOPE_MULT_{key}", DEFAULT_THRESHOLDS.get(interval, {}).get("ATR_SLOPE_MULT", GLOBAL_ATR_SLOPE_MULT))
        return pct, mult

    # 2) Global ENV tanımlı ise onları kullan
    if os.getenv("ATR_MIN_PCT") or os.getenv("ATR_SLOPE_MULT"):
        return GLOBAL_ATR_MIN_PCT, GLOBAL_ATR_SLOPE_MULT

    # 3) Varsayılan tablo
    base = DEFAULT_THRESHOLDS.get(interval, {"ATR_MIN_PCT": GLOBAL_ATR_MIN_PCT, "ATR_SLOPE_MULT": GLOBAL_ATR_SLOPE_MULT})
    return base["ATR_MIN_PCT"], base["ATR_SLOPE_MULT"]
# --------------------------------------

# ---------- İş Akışı ----------
def process_symbol(sym, intervals, state):
    alerts = []
    for interval in intervals:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < max(EMA_99, GLOBAL_ATR_PERIOD + 5):
            continue

        highs  = [float(k[2]) for k in kl]
        lows   = [float(k[3]) for k in kl]
        closes = [float(k[4]) for k in kl]
        last_price = closes[-1]
        last_bar_close_time = int(kl[-1][6])  # ms

        # EMA'lar
        ema7  = ema(closes, EMA_7)
        ema25 = ema(closes, EMA_25)

        # ATR ve eşiği
        atr = atr_series(highs, lows, closes, GLOBAL_ATR_PERIOD)
        atr_now = atr[-1]
        atr_pct = atr_now / last_price if last_price > 0 else 0.0

        # Interval bazlı eşikleri al
        ATR_MIN_PCT_I, ATR_SLOPE_MULT_I = thresholds_for_interval(interval)

        # Volatilite eşiği sağlanmazsa hiçbir sinyal üretme
        if atr_pct < ATR_MIN_PCT_I:
            time.sleep(SLEEP_BETWEEN)
            continue

        # CROSS kontrolü
        cross_dir, bars_ago = last_cross_info(ema7, ema25)
        if cross_dir and bars_ago == 0:
            slope_now = slope_value(ema7, lookback=3)
            if abs(slope_now) >= ATR_SLOPE_MULT_I * atr_now:
                alerts.append((
                    "CROSS", interval, cross_dir, slope_now, last_price,
                    last_bar_close_time, atr_now, atr_pct, ATR_MIN_PCT_I, ATR_SLOPE_MULT_I
                ))
        else:
            # SLOPE reversal
            rev_dir, rev_bars_ago = recent_slope_reversal(ema7, max_bars=3, lookback=3)
            if rev_dir is not None:
                slope_now = slope_value(ema7, lookback=3)
                if abs(slope_now) >= ATR_SLOPE_MULT_I * atr_now:
                    alerts.append((
                        "SLOPE", interval, rev_dir, slope_now, last_price,
                        last_bar_close_time, atr_now, atr_pct, ATR_MIN_PCT_I, ATR_SLOPE_MULT_I
                    ))

        time.sleep(SLEEP_BETWEEN)
    return alerts

def main():
    log("🚀 EMA bot (ATR + interval bazlı eşik) başlatıldı")
    state = safe_load_json(ALERTS_FILE)
    if not isinstance(state, dict):
        state = {}

    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Binance Futures coin listesi boş; 10 dk sonra tekrar denenecek.")
    else:
        log(f"{len(symbols)} coin taranacak...")

    while True:
        if not symbols:
            symbols = get_futures_symbols()
            time.sleep(5)

        for sym in symbols:
            alerts = process_symbol(sym, INTERVALS, state)
            for a in alerts:
                (a_type, interval, direction, slope_now, last_price,
                 bar_close_ms, atr_now, atr_pct, ATR_MIN_PCT_I, ATR_SLOPE_MULT_I) = a

                key = f"{sym}_{interval}"
                prev = state.get(key, {})

                prev_bar = prev.get("bar_close_ms")
                prev_type = prev.get("type")
                prev_dir  = prev.get("direction")

                if a_type == "CROSS":
                    if not (prev_type == "CROSS" and prev_dir == direction and prev_bar == bar_close_ms):
                        text = (
                            f"⚡ CROSS: {sym} ({interval})\n"
                            f"Direction: {direction}\n"
                            f"Slope(EMA7): {slope_now:.6f}\n"
                            f"ATR({GLOBAL_ATR_PERIOD}): {atr_now:.6f}  ({atr_pct*100:.3f}%)\n"
                            f"Eşikler[{interval}] → ATR%≥{ATR_MIN_PCT_I*100:.2f}%, |slope|≥{ATR_SLOPE_MULT_I:.2f}×ATR\n"
                            f"Price: {last_price}\n"
                            f"BarClose: {bar_close_ms}\n"
                            f"Time: {nowiso()}"
                        )
                        send_telegram(text)
                        state[key] = {
                            "type": "CROSS",
                            "direction": direction,
                            "time": nowiso(),
                            "last_slope_dir": "UP" if slope_now > 0 else ("DOWN" if slope_now < 0 else "FLAT"),
                            "slope": slope_now,
                            "last_price": last_price,
                            "bar_close_ms": bar_close_ms,
                            "atr": atr_now,
                            "atr_pct": atr_pct,
                            "atr_min_pct_i": ATR_MIN_PCT_I,
                            "atr_slope_mult_i": ATR_SLOPE_MULT_I,
                        }
                        safe_save_json(ALERTS_FILE, state)
                else:
                    last_slope_dir = prev.get("last_slope_dir")
                    same_bar = (prev_bar == bar_close_ms)
                    if not (prev_type == "SLOPE" and prev_dir == direction and same_bar) and (last_slope_dir != direction or not same_bar):
                        text = (
                            f"⚠️ SLOPE REVERSAL: {sym} ({interval})\n"
                            f"New Direction: {direction}\n"
                            f"Slope(EMA7): {slope_now:.6f}\n"
                            f"ATR({GLOBAL_ATR_PERIOD}): {atr_now:.6f}  ({atr_pct*100:.3f}%)\n"
                            f"Eşikler[{interval}] → ATR%≥{ATR_MIN_PCT_I*100:.2f}%, |slope|≥{ATR_SLOPE_MULT_I:.2f}×ATR\n"
                            f"Price: {last_price}\n"
                            f"BarClose: {bar_close_ms}\n"
                            f"Time: {nowiso()}"
                        )
                        send_telegram(text)
                        state[key] = {
                            "type": "SLOPE",
                            "direction": direction,
                            "time": nowiso(),
                            "last_slope_dir": direction,
                            "slope": slope_now,
                            "last_price": last_price,
                            "bar_close_ms": bar_close_ms,
                            "atr": atr_now,
                            "atr_pct": atr_pct,
                            "atr_min_pct_i": ATR_MIN_PCT_I,
                            "atr_slope_mult_i": ATR_SLOPE_MULT_I,
                        }
                        safe_save_json(ALERTS_FILE, state)

        log(f"⏳ {SCAN_INTERVAL//60} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
