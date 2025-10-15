import os, time, json, requests
from datetime import datetime, timezone

# ========= AYARLAR =========
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]

# EMA setleri (fast, slow, long)
EMA_SETS = {
    "1h": (7, 25, 99),
    "4h": (9, 26, 200),
    "1d": (20, 50, 200),
}

# ATR
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MIN_PCT_DEFAULTS = {
    "1h": float(os.getenv("ATR_MIN_PCT_1H", "0.0035")),  # %0.35
    "4h": float(os.getenv("ATR_MIN_PCT_4H", "0.0025")),  # %0.25
    "1d": float(os.getenv("ATR_MIN_PCT_1D", "0.0015")),  # %0.15
}
ATR_SLOPE_MULT_DEFAULTS = {
    "1h": float(os.getenv("ATR_SLOPE_MULT_1H", "0.6")),
    "4h": float(os.getenv("ATR_SLOPE_MULT_4H", "0.5")),
    "1d": float(os.getenv("ATR_SLOPE_MULT_1D", "0.4")),
}

# SL/TP öneri katsayıları (ENV override)
SL_MULT   = float(os.getenv("SL_MULT", "1.5"))
TP1_MULT  = float(os.getenv("TP1_MULT", "1.0"))
TP2_MULT  = float(os.getenv("TP2_MULT", "2.0"))
TP3_MULT  = float(os.getenv("TP3_MULT", "3.0"))

SLEEP_BETWEEN = 0.25
SCAN_INTERVAL = 600

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STATE_FILE = os.getenv("ALERTS_FILE", "alerts.json")
LOG_FILE = os.getenv("LOG_FILE", "log.txt")
PREMIUM_LOG_FILE = os.getenv("PREMIUM_LOG_FILE", "premium.log")
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

def log_premium(msg):
    try:
        with open(PREMIUM_LOG_FILE, "a", encoding="utf-8") as f:
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
        atr[i] = trs[i]
    return atr

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-PremiumBot/1.4", "Accept": "application/json"})

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

def last_bar_cross_direction(ema_fast, ema_slow):
    """Sadece son kapanmış barda kesişim: 'UP' / 'DOWN' / None"""
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return None
    prev = ema_fast[-2] - ema_slow[-2]
    curr = ema_fast[-1] - ema_slow[-1]
    if prev < 0 and curr > 0:
        return "UP"
    if prev > 0 and curr < 0:
        return "DOWN"
    return None

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

def long_short_label(dirn: str) -> str:
    return "LONG" if dirn == "UP" else "SHORT"

def sl_tp_from_atr(entry, atr, dirn):
    """ATR tabanlı SL/TP seviyeleri ve R:R değerleri"""
    sl = entry - SL_MULT * atr if dirn == "UP" else entry + SL_MULT * atr
    tp1 = entry + TP1_MULT * atr if dirn == "UP" else entry - TP1_MULT * atr
    tp2 = entry + TP2_MULT * atr if dirn == "UP" else entry - TP2_MULT * atr
    tp3 = entry + TP3_MULT * atr if dirn == "UP" else entry - TP3_MULT * atr
    # R:R = (TP - Entry) / (Entry - SL)  (mutlak değerlerle)
    risk = abs(entry - sl)
    rr1 = abs(tp1 - entry) / risk if risk > 0 else 0.0
    rr2 = abs(tp2 - entry) / risk if risk > 0 else 0.0
    rr3 = abs(tp3 - entry) / risk if risk > 0 else 0.0
    return sl, tp1, tp2, tp3, rr1, rr2, rr3

def process_symbol(sym, state):
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

    # 2) Her interval kendi CROSS sinyalini üretir (sadece SON BAR + erken dedup)
    for interval in INTERVALS:
        if interval not in cache:
            continue

        closes = cache[interval]["closes"]
        highs  = cache[interval]["highs"]
        lows   = cache[interval]["lows"]
        bar_ms = cache[interval]["bar_ms"]
        price  = closes[-1]

        ef, es, el = EMA_SETS[interval]
        ema_f = ema(closes, ef)
        ema_s = ema(closes, es)

        key = f"{sym}_{interval}"
        prev = state.get(key, {})

        # erken dedup — son kapanan bar değişmediyse bu interval için hiçbir şey yapma
        if prev.get("bar_ms") == bar_ms:
            continue

        # sadece son bar kesişimi
        dirn = last_bar_cross_direction(ema_f, ema_s)
        if not dirn:
            # bar değişti ama cross yok → barı işaretle
            state[key] = {"bar_ms": bar_ms, "last_dir": prev.get("last_dir"), "time": nowiso()}
            safe_save_json(STATE_FILE, state)
            continue

        # premium adayı mı? (üst TF trend uyumu + ATR güçlü)
        atr = atr_series(highs, lows, closes, ATR_PERIOD)
        atr_now = atr[-1]
        atr_pct = (atr_now / price) if price > 0 else 0.0
        slope_now = slope_value(ema_f, lookback=3)
        min_pct, slope_mult = thresholds(interval)

        atr_ok = (atr_pct >= min_pct) and (abs(slope_now) >= slope_mult * atr_now)
        atr_tag = "[ATR OK]" if atr_ok else "[ATR LOW]"

        htfi = higher_tf_of(interval)
        trend_ok = False
        trend_line = None
        if htfi and htfi in cache:
            ef_h, es_h, _ = EMA_SETS[htfi]
            ema_f_h = ema(cache[htfi]["closes"], ef_h)
            ema_s_h = ema(cache[htfi]["closes"], es_h)
            trend_h = trend_direction(ema_f_h, ema_s_h)
            trend_ok = (trend_h == dirn)
            trend_line = f"Trend({htfi}): {trend_h} {'✓' if trend_ok else '✗'}"
        elif interval == "1d":
            trend_line = "Üst TF yok (1d)"

        # premium koşulu:
        prem = (atr_ok if interval == "1d" else atr_ok and trend_ok)

        # SL/TP önerileri (ATR tabanlı)
        sl, tp1, tp2, tp3, rr1, rr2, rr3 = sl_tp_from_atr(price, atr_now, dirn)

        # mesaj
        tag = "⚡🔥 PREMIUM SİNYAL" if prem else "⚡ CROSS"
        direction_label = "LONG" if dirn == "UP" else "SHORT"
        lines = [
            f"{tag}: {sym} ({interval}) {atr_tag}",
            f"Direction: {dirn} ({direction_label})",
        ]
        if trend_line:
            lines.append(trend_line)
        lines += [
            f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)",
            f"Slope(fast,3): {slope_now:.6f}",
            f"Eşikler[{interval}]: ATR%≥{min_pct*100:.2f}%, |slope|≥{slope_mult:.2f}×ATR",
            f"Entry≈ {price}",
            f"SL≈ {sl}  |  TP1≈ {tp1} (R:R {rr1:.2f})  TP2≈ {tp2} (R:R {rr2:.2f})  TP3≈ {tp3} (R:R {rr3:.2f})",
            f"Time: {nowiso()}",
        ]
        msg = "\n".join(lines)
        send_telegram(msg)

        # premium olanları ayrı loga da yaz
        if prem:
            log_premium(msg)

        # bu barı işaretle (aynı bar ikinci kez gelmesin)
        state[key] = {"bar_ms": bar_ms, "last_dir": dirn, "time": nowiso()}
        safe_save_json(STATE_FILE, state)

def main():
    log("🚀 Premium (Cross + Trend + ATR) bot — Son bar kesişimi, ATR etiketi, LONG/SHORT ve SL/TP")
    state = safe_load_json(STATE_FILE)
    if not isinstance(state, dict):
        state = {}

    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Sembol listesi boş; Binance erişimi kontrol edin.")
        symbols = []

    while True:
        for sym in symbols:
            process_symbol(sym, state)
        log(f"⏳ {SCAN_INTERVAL//60} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
