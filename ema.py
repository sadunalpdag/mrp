import os, time, json, requests
from datetime import datetime, timezone

# ========= AYARLAR =========
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]

EMA_SETS = {
    "1h": (7, 25, 99),
    "4h": (9, 26, 200),
    "1d": (20, 50, 200),
}

ATR_PERIOD = 14
ATR_MIN_PCT_DEFAULTS = {"1h": 0.0035, "4h": 0.0025, "1d": 0.0015}
ATR_SLOPE_MULT_DEFAULTS = {"1h": 0.6, "4h": 0.5, "1d": 0.4}

SL_MULT, TP1_MULT, TP2_MULT, TP3_MULT = 1.5, 1.0, 2.0, 3.0
RSI_PERIOD = 14
RSI_SWING_LOOKBACK = 12
SLEEP_BETWEEN, SCAN_INTERVAL = 0.25, 300  # 5 dakika

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STATE_FILE = "alerts.json"
LOG_FILE, PREMIUM_LOG_FILE = "log.txt", "premium.log"
# ===========================


def nowiso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {msg}\n")
    except:
        pass


def log_premium(msg):
    try:
        with open(PREMIUM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {msg}\n")
    except:
        pass


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("Telegram env eksik (BOT_TOKEN/CHAT_ID). Mesaj gönderilmedi.")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        log(f"Telegram hatası: {e}")


def safe_load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return {}


def safe_save_json(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except:
        pass


# ---------- İNDİKATÖRLER ----------
def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals

def slope_value(series, lookback=3):
    return series[-1] - series[-lookback] if len(series) > lookback else 0.0

def atr_series(highs, lows, closes, period):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            pc = closes[i - 1]
            trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    if len(trs) < period:
        return [0.0]*len(trs)
    atr = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        atr.append((atr[-1] * (period - 1) + trs[i]) / period)
    return [0.0]*(len(trs)-len(atr)) + atr

def rsi(values, period=14):
    n = len(values)
    if n < period + 1:
        return [None] * n
    deltas = [values[i] - values[i-1] for i in range(1, n)]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - 100 / (1 + rs))
    return [None]*(n-len(rsis)) + rsis

def _local_extrema(series):
    peaks, troughs = [], []
    for i in range(1, len(series)-1):
        if series[i] > series[i-1] and series[i] > series[i+1]:
            peaks.append((i, series[i]))
        if series[i] < series[i-1] and series[i] < series[i+1]:
            troughs.append((i, series[i]))
    return peaks, troughs

def detect_rsi_divergence(closes, rsis, lookback=12):
    if len(closes) < lookback + 3:
        return None, None
    closes = closes[-lookback:]
    rsis   = rsis[-lookback:]
    peaks_p, troughs_p = _local_extrema(closes)
    peaks_r, troughs_r = _local_extrema([r for r in rsis if r is not None])
    if len(peaks_p) >= 2 and len(peaks_r) >= 2:
        if peaks_p[-1][1] > peaks_p[-2][1] and peaks_r[-1][1] < peaks_r[-2][1]:
            return "BEARISH", peaks_p[-1][0]
    if len(troughs_p) >= 2 and len(troughs_r) >= 2:
        if troughs_p[-1][1] < troughs_p[-2][1] and troughs_r[-1][1] > troughs_r[-2][1]:
            return "BULLISH", troughs_p[-1][0]
    return None, None

def just_crossed_now(ema_fast, ema_slow):
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return None
    prev_diff = ema_fast[-2] - ema_slow[-2]
    curr_diff = ema_fast[-1] - ema_slow[-1]
    if prev_diff < 0 and curr_diff > 0:
        return "UP"
    if prev_diff > 0 and curr_diff < 0:
        return "DOWN"
    return None

# ---------- EŞİKLER ----------
def thresholds(interval):
    return ATR_MIN_PCT_DEFAULTS.get(interval, 0.003), ATR_SLOPE_MULT_DEFAULTS.get(interval, 0.5)

def sl_tp_from_atr(entry, atr, dirn):
    sl  = entry - SL_MULT * atr if dirn=="UP" else entry + SL_MULT * atr
    tp1 = entry + TP1_MULT * atr if dirn=="UP" else entry - TP1_MULT * atr
    tp2 = entry + TP2_MULT * atr if dirn=="UP" else entry - TP2_MULT * atr
    tp3 = entry + TP3_MULT * atr if dirn=="UP" else entry - TP3_MULT * atr
    risk = abs(entry - sl)
    def rr(tp): return (abs(tp - entry) / risk) if risk > 0 else 0.0
    return sl, tp1, tp2, tp3, rr(tp1), rr(tp2), rr(tp3)

# ---------- Binance ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA/1.0", "Accept": "application/json"})

def get_klines(symbol, interval, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for _ in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(0.5)
    return []

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except:
        return []

# ---------- ANA İŞ AKIŞI ----------
def process_symbol(sym, state):
    for interval in INTERVALS:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < 220:
            continue

        closes = [float(k[4]) for k in kl]
        highs  = [float(k[2]) for k in kl]
        lows   = [float(k[3]) for k in kl]
        bar_ms = int(kl[-1][6])
        price  = closes[-1]

        ef, es, _ = EMA_SETS[interval]
        ema_f = ema(closes, ef)
        ema_s = ema(closes, es)

        key  = f"{sym}_{interval}"
        prev = state.get(key, {})

        dirn = just_crossed_now(ema_f, ema_s)
        if not dirn:
            continue
        if prev.get("last_signal_bar_ms") == bar_ms and prev.get("last_dir") == dirn:
            continue

        atr = atr_series(highs, lows, closes, ATR_PERIOD)
        atr_now = atr[-1]
        atr_pct = (atr_now / price) if price > 0 else 0.0
        slope_now = slope_value(ema_f, 3)
        min_pct, slope_mult = thresholds(interval)
        atr_ok = (atr_pct >= min_pct) and (abs(slope_now) >= slope_mult * atr_now)

        rsis = rsi(closes, RSI_PERIOD)
        rsi_val = rsis[-1] if rsis[-1] is not None else 50.0
        div_type, _ = detect_rsi_divergence(closes, rsis, RSI_SWING_LOOKBACK)
        rsi_status = f"{div_type} DIVERGENCE" if div_type else "NÖTR"

        level = "CROSS"
        if atr_ok:
            level = "PREMIUM"
        if atr_ok and div_type and ((dirn == "UP" and div_type == "BULLISH") or (dirn == "DOWN" and div_type == "BEARISH")):
            level = "ULTRA"

        sl, tp1, tp2, tp3, rr1, rr2, rr3 = sl_tp_from_atr(price, atr_now, dirn)

        tag = "⚡ CROSS"
        if level == "PREMIUM":
            tag = "⚡🔥 PREMIUM SİNYAL"
        elif level == "ULTRA":
            tag = "⚡🚀 ULTRA PREMIUM SİNYAL"
        atr_tag = "[ATR OK]" if atr_ok else "[ATR LOW]"

        lines = [
            f"{tag}: {sym} ({interval}) {atr_tag}",
            f"Direction: {dirn} ({'LONG' if dirn=='UP' else 'SHORT'})",
            f"Kesişim: EMA{ef} {'↗' if dirn=='UP' else '↘'} EMA{es}",
            f"RSI: {rsi_val:.2f} → {rsi_status}",
            f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)",
            f"Slope(fast,3): {slope_now:.6f}",
            f"Eşik[{interval}]: ATR%≥{min_pct*100:.2f} | |slope|≥{slope_mult:.2f}×ATR",
            f"Entry≈ {price}",
            f"SL≈ {sl} | TP1≈ {tp1:.4f} (R:R {rr1:.2f})  TP2≈ {tp2:.4f} (R:R {rr2:.2f})  TP3≈ {tp3:.4f} (R:R {rr3:.2f})",
            f"Time: {nowiso()}",
        ]
        msg = "\n".join(lines)
        send_telegram(msg)
        if level in ("PREMIUM", "ULTRA"):
            log_premium(msg)

        state[key] = {"last_signal_bar_ms": bar_ms, "last_dir": dirn}
        safe_save_json(STATE_FILE, state)

        time.sleep(SLEEP_BETWEEN)


def main():
    log("🚀 Binance EMA/ATR/RSI — Canlı bar + 5dk tarama + ULTRA PREMIUM + EMA kesişim bilgisi")
    state = safe_load_json(STATE_FILE)
    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Sembol listesi boş; Binance erişimini kontrol edin.")
        return
    while True:
        for sym in symbols:
            process_symbol(sym, state)
        log("⏳ 5 dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
