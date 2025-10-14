import requests
import petl as etl
import time
import os
from telegram import Bot
from datetime import datetime

INTERVALS = ["1h", "4h", "1d"]
LIMIT = 300
EMA_7 = 7
EMA_25 = 25
EMA_99 = 99
SLEEP_BETWEEN = 0.25
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = Bot(token=BOT_TOKEN)

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

def get_klines(symbol, interval, limit):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
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
    results = []
    for interval in INTERVALS:
        try:
            klines = get_klines(sym, interval, LIMIT)
            closes = [float(k[4]) for k in klines]
            if len(closes) < EMA_99:
                continue
            ema7 = ema(closes, EMA_7)
            ema25 = ema(closes, EMA_25)
            cross_dir, bars_ago = last_cross_info(ema7, ema25)
            if cross_dir and bars_ago == 0:
                results.append((interval, cross_dir))
        except Exception:
            continue
    return results

def main():
    print("🚀 EMA bot başlatıldı", datetime.now())
    symbols = get_futures_symbols()
    print(f"{len(symbols)} coin taranıyor...")
    while True:
        for sym in symbols:
            alerts = process_symbol(sym)
            for interval, direction in alerts:
                msg = f"⚡ {sym} ({interval}) yeni EMA7-EMA25 kesişimi: {direction}"
                print(msg)
                bot.send_message(chat_id=CHAT_ID, text=msg)
            time.sleep(SLEEP_BETWEEN)
        print("⏳ 30 dk bekleniyor...")
        time.sleep(600)  # 30 dakika

if __name__ == "__main__":
    main()
