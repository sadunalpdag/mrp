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
RETRY_418_SLEEP = 2.0
MAX_418_RETRY = 2

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

ALERTS_FILE = os.getenv("ALERTS_FILE", "/data/alerts.json")
if not os.path.isdir("/data"):
    ALERTS_FILE = "alerts.json"

LOG_FILE = os.getenv("LOG_FILE", "log.txt")
# ===========================

def nowiso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} - {msg}\n")

def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("Telegram bilgileri eksik.")
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
        log(f"JSON okunamadı: {e}")
    return {}

def safe_save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"JSON kaydedilemedi: {e}")

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

def get_futures_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "symbols" not in data:
            log(f"Binance exchangeInfo yanıtı hatalı: {data}")
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
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    tries = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 418:
                tries += 1
                log(f"{symbol} {interval}: HTTP 418, retry {tries}/{MAX_418_RETRY}")
                if tries > MAX_418_RETRY:
                    return []
                time.sleep(RETRY_418_SLEEP)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log(f"HTTPError {symbol} {interval}: {e}")
            return []
        except Exception as e:
            log(f"get_klines hata {symbol} {interval}: {e}")
            return []

def last_cross_info(ema_fast, ema_slow):
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

def process_symbol(sym, intervals):
    alerts = []
    for interval in intervals:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < EMA_99:
            continue

        closes = [float(k[4]) for k in kl]
        last_price = closes[-1]
        ema7 = ema(closes, EMA_7)
        ema25 = ema(closes, EMA_25)
        cross_dir, bars_ago = last_cross_info(ema7, ema25)

        if cross_dir and bars_ago == 0:  # sadece son bar
            slope_now = slope_value(ema7, lookback=3)
            alerts.append((interval, cross_dir, slope_now, last_price))

        time.sleep(SLEEP_BETWEEN)
    return alerts

def main():
    log("🚀 EMA bot başlatıldı")
    state = safe_load_json(ALERTS_FILE)  # {"BTCUSDT_1h": {"direction": "UP"}}
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
            alerts = process_symbol(sym, INTERVALS)
            for interval, direction, slope_now, last_price in alerts:
                key = f"{sym}_{interval}"
                prev = state.get(key, {})
                prev_dir = prev.get("direction")

                # Aynı yön tekrar edilmesin
                if prev_dir != direction:
                    msg = (
                        f"⚡ CROSS: {sym} ({interval})\n"
                        f"Direction: {direction}\n"
                        f"Slope(EMA7): {slope_now:.6f}\n"
                        f"Price: {last_price}\n"
                        f"Time: {nowiso()}"
                    )
                    send_telegram(msg)
                    state[key] = {
                        "direction": direction,
                        "time": nowiso(),
                        "slope": slope_now,
                        "price": last_price,
                    }
                    safe_save_json(ALERTS_FILE, state)

        log(f"⏳ {SCAN_INTERVAL//60} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
