import os, time, json, requests
from datetime import datetime, timezone

# ========= AYARLAR =========
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]

# Her interval için EMA setleri: (fast, slow, long)
EMA_SETS = {
    "1h": (7, 25, 99),
    "4h": (9, 26, 200),
    "1d": (20, 50, 200),
}

# ATR ayarları
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# Interval bazlı ATR minimum yüzde eşikleri (premium için)
# ENV override: ATR_MIN_PCT_1H / _4H / _1D  (örn: 0.003 = %0.3)
ATR_MIN_PCT_DEFAULTS = {
    "1h": float(os.getenv("ATR_MIN_PCT_1H", "0.0035")),  # %0.35
    "4h": float(os.getenv("ATR_MIN_PCT_4H", "0.0025")),  # %0.25
    "1d": float(os.getenv("ATR_MIN_PCT_1D", "0.0015")),  # %0.15
}

# Premium için ekstra güç doğrulaması:
# |EMA_fast slope(3)| >= ATR_SLOPE_MULT * ATR  (ENV override: ATR_SLOPE_MULT_1H/_4H/_1D)
ATR_SLOPE_MULT_DEFAULTS = {
    "1h": float(os.getenv("ATR_SLOPE_MULT_1H", "0.6")),
    "4h": float(os.getenv("ATR_SLOPE_MULT_4H", "0.5")),
    "1d": float(os.getenv("ATR_SLOPE_MULT_1D", "0.4")),
}

SLEEP_BETWEEN = 0.25
SCAN_INTERVAL = 600
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
STATE_FILE = os.getenv("ALERTS_FILE", "alerts.json")
LOG_FILE = os.getenv("LOG_FILE", "log.txt")
# ===========================

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
        log("Telegram env eksik (BOT_TOKEN/CHAT_ID). Mesaj gönderilmedi.")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
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
        log(f"state okunamadı: {e}")
    return {}

def safe_save_json(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"state yazılamadı: {e}")

def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals

def slope_value(series, lookback=3):
    if len(series) < lookback + 1:
        return 0.0
    return series[-1] - series[-lookback]

def true_ranges(highs, lows, closes):
    trs=[]
    for i in range(len(highs)):
        if i==0:
            trs.append(highs[i] - lows[i])
        else:
            pc = closes[i-1]
            trs.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    return trs

def atr_series(highs, lows, closes, period):
    trs = true_ranges(highs, lows, closes)
    if len(trs) < period:
        return [0.0] * len(trs)
    atr = [0.0] * len(trs)
    atr[period-1] = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr[i] = (atr[i-1] * (period - 1) + trs[i]) / period
    for i in range(period - 1):
        atr[i] = trs[i]  # baş kısım kaba tahmin
    return atr

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-PremiumBot/1.0", "Accept": "application/json"})

def get_klines(symbol, interval, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for _ in range(4):
        try:
            r = SESSION.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Kapanmamış barı at
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                if data and int(data[-1][6]) > now_ms:
                    data = data[:-1]
                return data
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return []

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except Exception:
        return []

def last_cross_info(ema_fast, ema_slow):
    last=None;direction=None
    for i in range(1, len(ema_fast)):
        prev = ema_fast[i-1] - ema_slow[i-1]
        curr = ema_fast[i]   - ema_slow[i]
        if prev < 0 and curr > 0:
            last = i; direction = "UP"
        elif prev > 0 and curr < 0:
            last = i; direction = "DOWN"
    if last is None:
        return None, None
    bars_ago = len(ema_fast) - last - 1
    return direction, bars_ago

def trend_direction(ema_fast, ema_slow):
    if ema_fast[-1] > ema_slow[-1]:
        return "UP"
    if ema_fast[-1] < ema_slow[-1]:
        return "DOWN"
    return "FLAT"

def higher_tf_of(interval):
    return {"1h": "4h", "4h": "1d"}.get(interval, None)

def thresholds(interval):
    return ATR_MIN_PCT_DEFAULTS.get(interval, 0.003), ATR_SLOPE_MULT_DEFAULTS.get(interval, 0.5)

def process_symbol(sym, state):
    alerts = []  # (sym, interval, dir, price, premium_flag, premium_info, bar_close_ms)

    # 1) Verileri önceden çek ve cache'le
    cache = {}
    for interval in INTERVALS:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < 220:
            time.sleep(SLEEP_BETWEEN); continue
        closes = [float(k[4]) for k in kl]
        highs  = [float(k[2]) for k in kl]
        lows   = [float(k[3]) for k in kl]
        bar_close_ms = int(kl[-1][6])
        cache[interval] = {"closes": closes, "highs": highs, "lows": lows, "bar_ms": bar_close_ms}
        time.sleep(SLEEP_BETWEEN)

    # 2) Her interval kendi CROSS sinyalini üretir
    for interval in INTERVALS:
        if interval not in cache: continue
        closes = cache[interval]["closes"]
        highs  = cache[interval]["highs"]
        lows   = cache[interval]["lows"]
        bar_ms = cache[interval]["bar_ms"]
        price  = closes[-1]

        ef, es, el = EMA_SETS[interval]
        ema_f = ema(closes, ef)
        ema_s = ema(closes, es)

        dirn, bars_ago = last_cross_info(ema_f, ema_s)
        if not dirn or bars_ago != 0:
            continue  # sadece son barda cross

        # --- PREMIUM adayı mı? (üst TF trend uyumu + ATR güçlü)
        prem = False
        prem_explain = []
        # ATR hesapla (kendi TF)
        atr = atr_series(highs, lows, closes, ATR_PERIOD)
        atr_now = atr[-1]
        atr_pct = (atr_now / price) if price > 0 else 0.0
        slope_now = slope_value(ema_f, lookback=3)
        min_pct, slope_mult = thresholds(interval)

        atr_ok = atr_pct >= min_pct and abs(slope_now) >= slope_mult * atr_now
        if atr_ok:
            prem_explain.append(f"ATR OK ({atr_pct*100:.2f}% ≥ {min_pct*100:.2f}%, |slope|≥{slope_mult:.2f}×ATR)")
        else:
            prem_explain.append(f"ATR yetersiz ({atr_pct*100:.2f}%/{min_pct*100:.2f}%, slope-th:{slope_mult:.2f})")

        htfi = higher_tf_of(interval)
        trend_ok = False
        if htfi and htfi in cache:
            # üst TF trendi
            ef_h, es_h, _ = EMA_SETS[htfi]
            ema_f_h = ema(cache[htfi]["closes"], ef_h)
            ema_s_h = ema(cache[htfi]["closes"], es_h)
            trend_h = trend_direction(ema_f_h, ema_s_h)
            trend_ok = (trend_h == dirn)
            prem_explain.append(f"Trend({htfi}): {trend_h} {'✓' if trend_ok else '✗'}")
        elif interval == "1d":
            # 1d için üst TF yok: sadece ATR güçlü ise premium
            prem_explain.append("Üst TF yok (1d)")

        # Premium koşulu:
        if interval == "1d":
            prem = atr_ok
        else:
            prem = atr_ok and trend_ok

        # dedup: aynı bar için aynı interval+sym tekrar gönderme
        key = f"{sym}_{interval}"
        prev_bar = state.get(key, {}).get("bar_ms")
        if prev_bar == bar_ms:
            continue  # aynı bar

        alerts.append((sym, interval, dirn, price, prem, atr_now, atr_pct, slope_now, min_pct, slope_mult, htfi, prem_explain, bar_ms))

    return alerts

def main():
    log("🚀 Premium (Cross + Trend + ATR) bot başlatıldı")
    state = safe_load_json(STATE_FILE)
    if not isinstance(state, dict): state = {}

    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Sembol listesi boş; Binance erişimi kontrol edin.")
        symbols = []

    while True:
        for sym in symbols:
            alerts = process_symbol(sym, state)
            for (s, interval, dirn, price, prem, atr_now, atr_pct, slope_now,
                 min_pct, slope_mult, htfi, prem_explain, bar_ms) in alerts:

                tag = "⚡🔥 PREMIUM SİNYAL" if prem else "⚡ CROSS"
                lines = [
                    f"{tag}: {s} ({interval})",
                    f"Direction: {dirn}",
                ]
                if htfi:
                    # prem_explain içinde trend detayı var
                    lines.append([pe for pe in prem_explain if pe.startswith("Trend") or "Üst TF" in pe][0] if prem_explain else f"Trend({htfi}): n/a")
                lines += [
                    f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)",
                    f"Slope(fast,3): {slope_now:.6f}",
                    f"Eşikler[{interval}]: ATR%≥{min_pct*100:.2f}%, |slope|≥{slope_mult:.2f}×ATR",
                    f"Price: {price}",
                    f"Time: {nowiso()}",
                ]
                send_telegram("\n".join(lines))

                state[f"{s}_{interval}"] = {"bar_ms": bar_ms, "last_dir": dirn, "time": nowiso()}
                safe_save_json(STATE_FILE, state)

        log(f"⏳ {SCAN_INTERVAL//60} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
