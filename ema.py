import os
import json
import time
from datetime import datetime
import requests

# ========= AYARLAR =========
EMA_7, EMA_25, EMA_99 = 7, 25, 99
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]
SLEEP_BETWEEN = 0.5        # Binance rate-limit dostu
SCAN_INTERVAL = 600        # 10 dakika
RETRY_418_SLEEP = 2.0      # 418 gelirse kısa bekleme
MAX_418_RETRY = 2

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Kalıcı kayıt dosyası (Render Paid + Persistent Disk varsa /data altında kalıcıdır)
ALERTS_FILE = os.getenv("ALERTS_FILE", "/data/alerts.json")
if not os.path.isdir("/data"):
    # /data yoksa proje dizinine yaz (restartta silinebilir ama çalışır)
    ALERTS_FILE = "alerts.json"

LOG_FILE = os.getenv("LOG_FILE", "log.txt")
# ===========================


# ---------- Yardımcılar ----------
def nowiso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
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
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        log(f"alerts.json okunamadı: {e}")
    return {}

def safe_save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
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
    """Son LOOKBACK bar için kabaca eğim (EMA7[-1] - EMA7[-lookback])."""
    if len(series) < (lookback + 1):
        return 0.0
    return series[-1] - series[-lookback]

def slope_sign(series, lookback=3):
    sv = slope_value(series, lookback)
    if sv > 0:
        return "UP"
    if sv < 0:
        return "DOWN"
    return "FLAT"
# ----------------------------------


# ---------- Binance API ----------
def get_futures_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "symbols" not in data:
            log(f"Binance exchangeInfo beklenmedik yanıt: {data}")
            return []
        return [
            s["symbol"]
            for s in data["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        ]
    except Exception as e:
        log(f"get_futures_symbols hatası: {e}")
        return []

def get_klines(symbol, interval, limit=LIMIT):
    """418 ve diğer HTTP hatalarına dayanıklı, küçük retry ile."""
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    tries = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 418:
                tries += 1
                log(f"{symbol} {interval}: HTTP 418 (teapot). Retry {tries}/{MAX_418_RETRY}")
                if tries > MAX_418_RETRY:
                    return []
                time.sleep(RETRY_418_SLEEP)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log(f"get_klines HTTPError {symbol} {interval}: {e}")
            return []
        except Exception as e:
            log(f"get_klines hata {symbol} {interval}: {e}")
            return []
# ----------------------------------


# ---------- Analiz Mantığı ----------
def last_cross_info(ema_fast, ema_slow):
    """EMA7/EMA25 son kesişim yönü ve kaç bar önce olduğunu döndürür."""
    last_cross = None
    direction = None
    for i in range(1, len(ema_fast)):
        prev_diff = ema_fast[i - 1] - ema_slow[i - 1]
        curr_diff = ema_fast[i] - ema_slow[i]
        if prev_diff < 0 and curr_diff > 0:
            last_cross = i
            direction = "UP"
        elif prev_diff > 0 and curr_diff < 0:
            last_cross = i
            direction = "DOWN"
    if last_cross is None:
        return None, None
    bars_ago = len(ema_fast) - last_cross - 1
    return direction, bars_ago

def recent_slope_reversal(ema7, max_bars=3, lookback=3):
    """
    Son max_bars içinde EMA7 yön değişimi var mı?
    Slope yönünü (UP/DOWN/FLAT) bar bar kontrol eder.
    """
    if len(ema7) < lookback + max_bars + 2:
        return None, None  # veri az
    # Son (max_bars+1) adet slope yönünü çıkar (örn: son 4 bar için 3 değişim kontrolü)
    dirs = []
    for shift in range(max_bars + 1):  # 0: şimdi, 1: -1 bar, ...
        # slope at t-shift  ~  ema7[-1-shift] - ema7[-1-shift-lookback]
        idx_now = -1 - shift
        idx_past = idx_now - lookback
        if abs(idx_past) > len(ema7):
            break
        val = ema7[idx_now] - ema7[idx_past]
        if val > 0:
            dirs.append("UP")
        elif val < 0:
            dirs.append("DOWN")
        else:
            dirs.append("FLAT")
    # Bir yön değişimi (UP<->DOWN) var mı? İlk farklılığı bul.
    base = dirs[0]
    for i in range(1, len(dirs)):
        d = dirs[i]
        if base == "UP" and d == "DOWN":
            return "DOWN", i  # i bar önce değişti ve yeni yön DOWN
        if base == "DOWN" and d == "UP":
            return "UP", i    # i bar önce değişti ve yeni yön UP
    return None, None
# ------------------------------------


# ---------- İş Akışı ----------
def process_symbol(sym, intervals, state):
    """
    DÖNEN:
      alerts: [(type, interval, direction, slope, price)]
        type: "CROSS" veya "SLOPE"
    """
    alerts = []
    for interval in intervals:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < EMA_99:
            continue

        closes = [float(k[4]) for k in kl]
        last_price = closes[-1]
        ema7 = ema(closes, EMA_7)
        ema25 = ema(closes, EMA_25)

        # 1) CROSS — son bar içinde kesişim
        cross_dir, bars_ago = last_cross_info(ema7, ema25)
        if cross_dir and bars_ago == 0:
            slope_now = slope_value(ema7, lookback=3)
            alerts.append(("CROSS", interval, cross_dir, slope_now, last_price))
        else:
            # 2) SLOPE REVERSAL — kesişim yoksa, son 3 bar içinde EMA7 yönü değişti mi?
            rev_dir, rev_bars_ago = recent_slope_reversal(ema7, max_bars=3, lookback=3)
            if rev_dir is not None:
                # tekrar spam engelleme: en son slope yönü aynıysa bildirme
                key = f"{sym}_{interval}"
                last_slope_dir = state.get(key, {}).get("last_slope_dir")
                if last_slope_dir != rev_dir:
                    slope_now = slope_value(ema7, lookback=3)
                    alerts.append(("SLOPE", interval, rev_dir, slope_now, last_price))

        time.sleep(SLEEP_BETWEEN)
    return alerts

def main():
    log("🚀 EMA bot başlatıldı")
    # Kalıcı state yükle
    state = safe_load_json(ALERTS_FILE)  # { key: {type, direction, time, last_slope_dir, last_price, slope} }
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
            time.sleep(10)

        for sym in symbols:
            alerts = process_symbol(sym, INTERVALS, state)
            for a_type, interval, direction, slope_now, last_price in alerts:
                key = f"{sym}_{interval}"
                # CROSS için tekrarsız gönderim: aynı type+direction ise tekrar etme
                prev = state.get(key, {})
                prev_type = prev.get("type")
                prev_dir = prev.get("direction")

                # CROSS öncelikli: CROSS geldiyse SLOPE sinyalini gölgeler
                if a_type == "CROSS":
                    if not (prev_type == "CROSS" and prev_dir == direction):
                        text = (
                            f"⚡ CROSS: {sym} ({interval})\n"
                            f"Direction: {direction}\n"
                            f"Slope(EMA7): {slope_now:.6f}\n"
                            f"Price: {last_price}\n"
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
                        }
                        safe_save_json(ALERTS_FILE, state)
                else:  # SLOPE
                    # CROSS ile aynı anda tekrar etme; daha önce aynı SLOPE yönü bildirilmişse de etme
                    if not (prev_type == "SLOPE" and prev_dir == direction):
                        text = (
                            f"⚠️ SLOPE REVERSAL: {sym} ({interval})\n"
                            f"New Direction: {direction}\n"
                            f"Slope(EMA7): {slope_now:.6f}\n"
                            f"Price: {last_price}\n"
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
                        }
                        safe_save_json(ALERTS_FILE, state)

        log(f"⏳ {SCAN_INTERVAL//60} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
