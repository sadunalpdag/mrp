import requests
import time
from datetime import datetime

# ======= AYARLAR =======
EMA_7, EMA_25, EMA_99 = 7, 25, 99
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]
SLEEP_BETWEEN = 0.15
SCAN_INTERVAL = 1800  # 30 dakika

BOT_TOKEN = "8028257005:AAFdfRWvWFrEAIGM4gIt6JRW7ebbfoNqyEo"
CHAT_ID = "990932798"
# =======================

def send_telegram(message):
    """Telegram’a mesaj gönder"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("Telegram hatası:", e)

def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals

def get_futures_symbols():
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    r = requests.get(url)
    data = r.json()
    usdt_pairs = [d["symbol"] for d in data if d["symbol"].endswith("USDT")]
    return sorted(set(usdt_pairs))

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

def scan(last_sent):
    results = []
    symbols = get_futures_symbols()
    print(f"🔍 {len(symbols)} Futures paritesi taranıyor...")

    for sym in symbols:
        for interval in INTERVALS:
            try:
                klines = get_klines(sym, interval)
                closes = [float(k[4]) for k in klines]
                if len(closes) < EMA_99:
                    continue

                ema7 = ema(closes, EMA_7)
                ema25 = ema(closes, EMA_25)
                ema99 = ema(closes, EMA_99)
                cross_dir, bars_ago = last_cross_info(ema7, ema25)

                if cross_dir and bars_ago <= 1:
                    signal_id = f"{sym}_{interval}_{cross_dir}"
                    if signal_id not in last_sent:
                        direction = "BUY" if cross_dir == "UP" else "SELL"
                        msg = (
                            f"⚡ EMA CROSS ALERT ⚡\n"
                            f"Symbol: {sym}\n"
                            f"Timeframe: {interval}\n"
                            f"Direction: {direction}\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        print(msg)
                        send_telegram(msg)
                        last_sent.add(signal_id)
                        results.append(msg)

                time.sleep(SLEEP_BETWEEN)

            except Exception as e:
                print(f"Hata {sym} {interval}: {e}")
                time.sleep(0.3)
    return results, last_sent

if __name__ == "__main__":
    print("🚀 Futures EMA Scanner (Telegram) başlatıldı.")
    last_sent_signals = set()
    while True:
        try:
            new_signals, last_sent_signals = scan(last_sent_signals)
            print(f"✅ {len(new_signals)} yeni sinyal bulundu.")
        except Exception as e:
            print("Genel hata:", e)
        print("⏸ 30 dakika bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

