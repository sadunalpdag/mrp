import os, time, json, requests
from datetime import datetime, timezone

# ========= AYARLAR =========
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]

# EMA setleri (fast, slow, long - long şu an trend çizgisi için kullanılmıyor)
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

# RSI
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_SWING_LOOKBACK = int(os.getenv("RSI_SWING_LOOKBACK", "12"))  # swing araması penceresi

SLEEP_BETWEEN = 0.25
SCAN_INTERVAL = 600

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

STATE_FILE       = os.getenv("ALERTS_FILE", "alerts.json")
LOG_FILE         = os.getenv("LOG_FILE", "log.txt")
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

# ---- RSI (Wilder, 14 default) ----
def rsi(values, period=14):
    n = len(values)
    if n < period + 1:
        return [None] * n
    deltas = [values[i] - values[i-1] for i in range(1, n)]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float('inf')
    first_rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100.0

    rsis = [first_rsi]
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))

    # baştaki None'ları doldur (uzunluk eşitleme)
    pad = n - len(rsis)
    return [None]*pad + rsis

def _local_extrema(series, start_idx, end_idx):
    """[start_idx, end_idx] dahil aralıkta lokal tepe/dip noktalarını (index, value) olarak döndürür."""
    peaks, troughs = []
    for i in range(max(start_idx,1), min(end_idx, len(series)-2)+1):
        a, b, c = series[i-1], series[i], series[i+1]
        if b is None or a is None or c is None:
            continue
        if b > a and b > c:
            peaks.append((i, b))
        if b < a and b < c:
            troughs.append((i, b))
    return peaks, troughs

def detect_rsi_divergence(closes, rsis, lookback=12):
    """
    Swing tabanlı RSI divergence:
      - Bearish: Price Higher High, RSI Lower High
      - Bullish: Price Lower Low, RSI Higher Low
    Dönüş: ("BEARISH"/"BULLISH"/None, position_index or None)
    """
    n = len(closes)
    if n < lookback + 3:
        return None, None

    start = max(0, n - lookback)
    end   = n - 1

    # Price swing'leri (None yok)
    closes_nn = [float(x) for x in closes]
    p_peaks, p_troughs = _local_extrema(closes_nn, start, end)

    # RSI swing'leri (None at)
    rsis_f = rsis[:]  # kopya
    r_peaks, r_troughs = _local_extrema(rsis_f, start, end)

    div_type, pos = None, None

    if len(p_peaks) >= 2 and len(r_peaks) >= 2:
        p1i, p1v = p_peaks[-2]
        p2i, p2v = p_peaks[-1]
        r1i, r1v = r_peaks[-2]
        r2i, r2v = r_peaks[-1]
        # Price HH & RSI LH
        if p2v > p1v and r2v < r1v:
            div_type, pos = "BEARISH", p2i

    if div_type is None and len(p_troughs) >= 2 and len(r_troughs) >= 2:
        t1i, t1v = p_troughs[-2]
        t2i, t2v = p_troughs[-1]
        rt1i, rt1v = r_troughs[-2]
        rt2i, rt2v = r_troughs[-1]
        # Price LL & RSI HL
        if t2v < t1v and rt2v > rt1v:
            div_type, pos = "BULLISH", t2i

    return div_type, pos

# ---- HTTP / Binance ----
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-PremiumBot/1.5", "Accept": "application/json"})

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

# ---- CROSS & TREND ----
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
    sl  = entry - SL_MULT * atr if dirn == "UP" else entry + SL_MULT * atr
    tp1 = entry + TP1_MULT * atr if dirn == "UP" else entry - TP1_MULT * atr
    tp2 = entry + TP2_MULT * atr if dirn == "UP" else entry - TP2_MULT * atr
    tp3 = entry + TP3_MULT * atr if dirn == "UP" else entry - TP3_MULT * atr
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
        dirn = last_bar_cross
        # sadece son bar kesişimi
        dirn = last_bar_cross_direction(ema_f, ema_s)
        if not dirn:
            # bar değişti ama cross yok → barı işaretle
            state[key] = {"bar_ms": bar_ms, "last_dir": prev.get("last_dir"), "time": nowiso()}
            safe_save_json(STATE_FILE, state)
            continue

        # ATR hesapla
        atr = atr_series(highs, lows, closes, ATR_PERIOD)
        atr_now = atr[-1]
        atr_pct = (atr_now / price) if price > 0 else 0.0
        slope_now = slope_value(ema_f, lookback=3)
        min_pct, slope_mult = thresholds(interval)
        atr_ok = (atr_pct >= min_pct) and (abs(slope_now) >= slope_mult * atr_now)
        atr_tag = "[ATR OK]" if atr_ok else "[ATR LOW]"

        # RSI + divergence
        rsis = rsi(closes, RSI_PERIOD)
        rsi_val = rsis[-1] if rsis[-1] is not None else 50
        div_type, _ = detect_rsi_divergence(closes, rsis, lookback=RSI_SWING_LOOKBACK)
        div_line = f"RSI: {rsi_val:.2f}"
        if div_type:
            div_line += f" → {div_type} DIVERGENCE"

        # Üst TF trend uyumu
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

        # PREMIUM koşulu
        prem = (atr_ok if interval == "1d" else atr_ok and trend_ok)
        if div_type:
            # Divergence trend yönüyle uyumluysa premium güçlenir
            if (dirn == "UP" and div_type == "BULLISH") or (dirn == "DOWN" and div_type == "BEARISH"):
                prem = True

        # SL/TP önerileri
        sl, tp1, tp2, tp3, rr1, rr2, rr3 = sl_tp_from_atr(price, atr_now, dirn)

        # Mesaj oluştur
        tag = "⚡🔥 PREMIUM SİNYAL" if prem else "⚡ CROSS"
        direction_label = "LONG" if dirn == "UP" else "SHORT"
        lines = [
            f"{tag}: {sym} ({interval}) {atr_tag}",
            f"Direction: {dirn} ({direction_label})",
            trend_line if trend_line else None,
            div_line,
            f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)",
            f"Slope(fast,3): {slope_now:.6f}",
            f"Eşikler[{interval}]: ATR%≥{min_pct*100:.2f}%, |slope|≥{slope_mult:.2f}×ATR",
            f"Entry≈ {price}",
            f"SL≈ {sl}  |  TP1≈ {tp1} (R:R {rr1:.2f})  TP2≈ {tp2} (R:R {rr2:.2f})  TP3≈ {tp3} (R:R {rr3:.2f})",
            f"Time: {nowiso()}",
        ]
        msg = "\n".join([l for l in lines if l])
        send_telegram(msg)
        if prem:
            log_premium(msg)

        # state kaydı
        state[key] = {"bar_ms": bar_ms, "last_dir": dirn, "time": nowiso()}
        safe_save_json(STATE_FILE, state)

def main():
    log("🚀 Binance EMA+ATR+Trend+RSI bot (premium sinyal sistemi) başlatıldı")
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
