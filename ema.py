import requests
import time
from datetime import datetime
import os

# ======= AYARLAR =======
EMA_7, EMA_25, EMA_99 = 7, 25, 99
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]
SLEEP_BETWEEN = 0.15
SCAN_INTERVAL = 600  # 10 dakika

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
    r = requests.get(url)
    data = r.json()
    return [s["symbol"] for s in data["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]

def get_klines(symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={LIMIT}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

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

def process_symbol(sym):
    alerts = []
    for interval in INTERVALS:
        try:
            klines = get_klines(sym, interval)
            closes = [float(k[4]) for k in klines]
            if len(closes) < EMA_99:
                continue
            ema7 = ema(closes, EMA_7)
            ema25 = ema(closes, EMA_25)
            cross_dir, bars_ago = last_cross_info(ema7, ema25)
            if cross_dir and bars_ago == 0:
                alerts.append((interval, cross_dir))
        except Exception as e:
            log(f"Hata {sym} {interval}: {e}")
            time.sleep(0.2)
    return alerts

def main():
    log("🚀 EMA bot başlatıldı")
    symbols = get_futures_symbols()
    log(f"{len(symbols)} coin taranıyor...")
    last_alerts = set()

    while True:
        for sym in symbols:
            alerts = process_symbol(sym)
            for interval, direction in alerts:
                alert_id = f"{sym}_{interval}_{direction}"
                if alert_id not in last_alerts:
                    msg = f"⚡ {sym} ({interval}) yeni EMA7-EMA25 kesişimi: {direction}"
                    send_telegram(msg)
                    last_alerts.add(alert_id)
            time.sleep(SLEEP_BETWEEN)
        log(f"⏳ {SCAN_INTERVAL/60:.0f} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
