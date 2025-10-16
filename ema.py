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


def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals


def slope_value(series, lookback=3):
    return series[-1] - series[-lookback] if len(series) > lookback else 0


def atr_series(highs, lows, closes, period):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            pc = closes[i - 1]
            trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    atr = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        atr.append((atr[-1] * (period - 1) + trs[i]) / period)
    return [0] * (len(trs) - len(atr)) + atr


def rsi(values, period=14):
    if len(values) < period + 1:
        return [None] * len(values)
    deltas = [values[i] - values[i-1] for i in range(1, len(values))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain, avg_loss = sum(gains[:period]) / period, sum(losses[:period]) / period
    rsis = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss else 0
        rsis.append(100 - 100 / (1 + rs) if avg_loss else 100)
    return [None]*(len(values)-len(rsis)) + rsis


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
    closes, rsis = closes[-lookback:], rsis[-lookback:]
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
    if len(ema_fast) < 3:
        return None
    prev_diff = ema_fast[-2] - ema_slow[-2]
    curr_diff = ema_fast[-1] - ema_slow[-1]
    if prev_diff < 0 and curr_diff > 0:
        return "UP"
    if prev_diff > 0 and curr_diff < 0:
        return "DOWN"
    return None


def trend_direction(ema_fast, ema_slow):
    if ema_fast[-1] > ema_slow[-1]: return "UP"
    if ema_fast[-1] < ema_slow[-1]: return "DOWN"
    return "FLAT"


def higher_tf_of(interval):
    return {"1h": "4h", "4h": "1d"}.get(interval)


def thresholds(interval):
    return ATR_MIN_PCT_DEFAULTS.get(interval, 0.003), ATR_SLOPE_MULT_DEFAULTS.get(interval, 0.5)


def sl_tp_from_atr(entry, atr, dirn):
    sl  = entry - SL_MULT * atr if dirn=="UP" else entry + SL_MULT * atr
    tp1 = entry + TP1_MULT * atr if dirn=="UP" else entry - TP1_MULT * atr
    tp2 = entry + TP2_MULT * atr if dirn=="UP" else entry - TP2_MULT * atr
    tp3 = entry + TP3_MULT * atr if dirn=="UP" else entry - TP3_MULT * atr
    risk = abs(entry-sl)
    r = lambda tp: abs(tp-entry)/risk if risk>0 else 0
    return sl,tp1,tp2,tp3,r(tp1),r(tp2),r(tp3)
# ---------------------------------------------


def get_klines(symbol, interval, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()  # canlı bar dahil
        except:
            time.sleep(1)
    return []


def get_futures_symbols():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        return [s["symbol"] for s in r.json()["symbols"] if s["quoteAsset"]=="USDT"]
    except:
        return []


def process_symbol(sym, state):
    for interval in INTERVALS:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < 220: continue
        closes = [float(k[4]) for k in kl]
        highs  = [float(k[2]) for k in kl]
        lows   = [float(k[3]) for k in kl]
        bar_ms = int(kl[-1][6])
        price  = closes[-1]
        ef, es, _ = EMA_SETS[interval]
        ema_f = ema(closes, ef)
        ema_s = ema(closes, es)

        key = f"{sym}_{interval}"
        prev = state.get(key, {})
        last_dir = prev.get("last_dir")
        dirn = just_crossed_now(ema_f, ema_s)
        if not dirn or dirn == last_dir:
            continue

        atr = atr_series(highs, lows, closes, ATR_PERIOD)
        atr_now = atr[-1]
        atr_pct = atr_now / price if price > 0 else 0
        slope_now = slope_value(ema_f, 3)
        min_pct, slope_mult = thresholds(interval)
        atr_ok = (atr_pct >= min_pct and abs(slope_now) >= slope_mult * atr_now)

        rsis = rsi(closes, RSI_PERIOD)
        rsi_val = rsis[-1] if rsis[-1] else 50
        div_type, _ = detect_rsi_divergence(closes, rsis, RSI_SWING_LOOKBACK)
        rsi_status = f"{div_type} DIVERGENCE" if div_type else "NÖTR"

        prem = atr_ok
        if div_type and ((dirn == "UP" and div_type == "BULLISH") or (dirn == "DOWN" and div_type == "BEARISH")):
            prem = True

        sl,tp1,tp2,tp3,rr1,rr2,rr3 = sl_tp_from_atr(price, atr_now, dirn)
        tag = "⚡🔥 PREMIUM SİNYAL" if prem else "⚡ CROSS"
        atr_tag = "[ATR OK]" if atr_ok else "[ATR LOW]"
        msg = (
            f"{tag}: {sym} ({interval}) {atr_tag}\n"
            f"Direction: {dirn} ({'LONG' if dirn=='UP' else 'SHORT'})\n"
            f"RSI: {rsi_val:.2f} → {rsi_status}\n"
            f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)\n"
            f"Slope: {slope_now:.6f}\n"
            f"Eşik: ATR%≥{min_pct*100:.2f} | slope≥{slope_mult:.2f}×ATR\n"
            f"Entry≈ {price}\n"
            f"SL≈ {sl} | TP1≈ {tp1} (R:R {rr1:.2f}) TP2≈ {tp2} (R:R {rr2:.2f}) TP3≈ {tp3} (R:R {rr3:.2f})\n"
            f"Time: {nowiso()}"
        )
        send_telegram(msg)
        if prem: log_premium(msg)
        state[key] = {"bar_ms": bar_ms, "last_dir": dirn}
        safe_save_json(STATE_FILE, state)
        time.sleep(SLEEP_BETWEEN)


def main():
    log("🚀 Binance EMA+ATR+RSI bot (canlı bar, 5dk kontrol) başlatıldı")
    state = safe_load_json(STATE_FILE)
    symbols = get_futures_symbols()
    while True:
        for sym in symbols:
            process_symbol(sym, state)
        log(f"⏳ 5 dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
