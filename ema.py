import requests
import time
from datetime import datetime
import os

# ======= AYARLAR =======
EMA_7, EMA_25, EMA_99 = 7, 25, 99
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]
SLEEP_BETWEEN = 0.5  # rate-limit dostu
SCAN_INTERVAL = 600   # 10 dakika

# Slope eşikleri (güçlü trend için)
SLOPE_UP_THRESHOLD = 0.2
SLOPE_DOWN_THRESHOLD = -0.2

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
LOG_FILE = "log.txt"
# =======================

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} - {msg}\n")

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        log(f"Telegram mesajı gönderildi: {msg}")
    except Exception as e:
        log(f"Telegram hatası: {e}")

def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals

def get_futures_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "symbols" not in data:
            log(f"Binance API hatası: {data}")
            return []
        return [s["symbol"] for s in data["symbols"]
                if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"]
    except Exception as e:
        log(f"get_futures_symbols hatası: {e}")
        return []

def get_klines(symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={LIMIT}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 418:
            log(f"{symbol} {interval}: 418 I’m a teapot, atlanıyor.")
            return []
        else:
            log(f"get_klines HTTPError {symbol} {interval}: {e}")
            return []
    except Exception as e:
        log(f"get_klines hatası {symbol} {interval}: {e}")
        return []

def last_cross_info(ema_fast, ema_slow):
    """
    EMA7/EMA25 kesişimini bulur ve kaç bar önce olduğunu döndürür.
    Son 1 bar için kullanacağız.
    """
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

def process_symbol(sym):
    alerts = []
    for interval in INTERVALS:
        klines = get_klines(sym, interval)
        if not klines or len(klines) < EMA_99:
            continue
        closes = [float(k[4]) for k in klines]
        ema7 = ema(closes, EMA_7)
        ema25 = ema(closes, EMA_25)
        cross_dir, bars_ago = last_cross_info(ema7, ema25)
        if cross_dir and bars_ago == 0:  # son 1 bar
            slope = ema7[-1] - ema7[-3]
            alerts.append((interval, cross_dir, slope))
        time.sleep(SLEEP_BETWEEN)
    return alerts

def main():
    log("🚀 EMA bot başlatıldı")
    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Binance Futures coin listesi boş, tekrar denenecek 10 dk sonra.")
    else:
        log(f"{len(symbols)} coin taranıyor...")

    last_alerts = {}  # {"BTCUSDT_1h": "UP"} -> tekrar göndermeyi engeller

    while True:
        if not symbols:
            symbols = get_futures_symbols()
            time.sleep(10)

        for sym in symbols:
            alerts = process_symbol(sym)
            for interval, direction, slope in alerts:
                alert_id = f"{sym}_{interval}"
                # Sadece güçlü trendler
                if (direction == "UP" and slope >= 0.2) or \
                   (direction == "DOWN" and slope <= -0.2):
                    if last_alerts.get(alert_id) != direction:
                        msg = f"⚡ {sym} ({interval}) EMA7-EMA25 kesişimi: {direction}, slope={slope:.4f}"
                        send_telegram(msg)
                        last_alerts[alert_id] = direction

        log(f"⏳ {SCAN_INTERVAL/60:.0f} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
