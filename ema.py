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

# ATR & eşikler
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MIN_PCT_DEFAULTS = {"1h": 0.0035, "4h": 0.0025, "1d": 0.0015}
ATR_SLOPE_MULT_DEFAULTS = {"1h": 0.6, "4h": 0.5, "1d": 0.4}

# SL/TP (TP'ler 4 ondalık)
SL_MULT   = float(os.getenv("SL_MULT", "1.5"))
TP1_MULT  = float(os.getenv("TP1_MULT", "1.0"))
TP2_MULT  = float(os.getenv("TP2_MULT", "2.0"))
TP3_MULT  = float(os.getenv("TP3_MULT", "3.0"))

# RSI & Divergence
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_SWING_LOOKBACK = int(os.getenv("RSI_SWING_LOOKBACK", "12"))

# Destek/Direnç lookback
SR_LOOKBACK = int(os.getenv("SR_LOOKBACK", "100"))

# Erken onay: bar kapanışına ≤30dk kala canlı barda yön korunuyorsa sinyal ver
EARLY_CONFIRM_MS = int(os.getenv("EARLY_CONFIRM_MS", str(30*60*1000)))  # 30dk

# SCALP ayarları (EMA7 slope reversal)
SCALP_TF_CONFIRM = os.getenv("SCALP_TF_CONFIRM", "4h")  # 1h için üst TF onayı ("4h" önerilir)
SCALP_TP_MULT = float(os.getenv("SCALP_TP_MULT", "0.5"))   # ATR * 0.5
SCALP_SL_MULT = float(os.getenv("SCALP_SL_MULT", "0.25"))  # ATR * 0.25
SCALP_MIN_ATR_FACTOR = float(os.getenv("SCALP_MIN_ATR_FACTOR", "1.0"))  # atr_pct ≥ min_pct * factor

SLEEP_BETWEEN = 0.2
SCAN_INTERVAL = 300  # 5 dakika

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "alerts.json")
LOG_FILE   = os.getenv("LOG_FILE",   "log.txt")
PREMIUM_LOG_FILE = os.getenv("PREMIUM_LOG_FILE", "premium.log")
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
    # Bearish: Price HH & RSI LH
    if len(peaks_p) >= 2 and len(peaks_r) >= 2:
        if peaks_p[-1][1] > peaks_p[-2][1] and peaks_r[-1][1] < peaks_r[-2][1]:
            return "BEARISH", peaks_p[-1][0]
    # Bullish: Price LL & RSI HL
    if len(troughs_p) >= 2 and len(troughs_r) >= 2:
        if troughs_p[-1][1] < troughs_p[-2][1] and troughs_r[-1][1] > troughs_r[-2][1]:
            return "BULLISH", troughs_p[-1][0]
    return None, None

# --- Stabilizasyon (kapalı bar) + Erken onay (canlı bar) ---
def stabilized_or_early(ema_f_closed, ema_s_closed, ema_f_full, ema_s_full, bar_close_ms, now_ms, early_ms):
    """
    1) Kapanmış barlarla onay:
       prev_diff(−3), cross_diff(−2), after_closed(−1) aynı yönde → 'UP'/'DOWN', 'CLOSED'
    2) Erken onay:
       bar kapanışına <= early_ms kaldıysa ve (prev, cross, curr_live) aynı yönde → 'UP'/'DOWN', 'EARLY'
    """
    if len(ema_f_closed) < 3:
        return None, None

    prev_diff   = ema_f_closed[-3] - ema_s_closed[-3]
    cross_diff  = ema_f_closed[-2] - ema_s_closed[-2]
    after_diff  = ema_f_closed[-1] - ema_s_closed[-1]

    # Kapalı barla stabilizasyon
    if prev_diff < 0 and cross_diff > 0 and after_diff > 0:
        return "UP", "CLOSED"
    if prev_diff > 0 and cross_diff < 0 and after_diff < 0:
        return "DOWN", "CLOSED"

    # Erken onay (canlı bar yönüyle)
    if (bar_close_ms - now_ms) <= early_ms and len(ema_f_full) >= 2 and len(ema_s_full) >= 2:
        curr_live = ema_f_full[-1] - ema_s_full[-1]
        if prev_diff < 0 and cross_diff > 0 and curr_live > 0:
            return "UP", "EARLY"
        if prev_diff > 0 and cross_diff < 0 and curr_live < 0:
            return "DOWN", "EARLY"

    return None, None

# Destek / Direnç (son tepe/diplerden, yatay seviye)
def trend_lines_from_extrema(closes, lookback=100):
    clip = closes[-min(lookback, len(closes)):]
    peaks, troughs = _local_extrema(clip)
    if len(peaks) >= 2:
        resistance = (peaks[-1][1] + peaks[-2][1]) / 2.0
    else:
        resistance = max(clip) if clip else None
    if len(troughs) >= 2:
        support = (troughs[-1][1] + troughs[-2][1]) / 2.0
    else:
        support = min(clip) if clip else None
    return support, resistance


# ---------- Binance Kaynak ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA/1.8", "Accept": "application/json"})

def get_klines(symbol, interval, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for _ in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(0.4)
    return []

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except:
        return []


# ---------- ÜST TF YÖN HESABI (1h için 4h & 1d) ----------
def ema_trend_direction_from_closes(closes, fast_len, slow_len):
    ef = ema(closes, fast_len)
    es = ema(closes, slow_len)
    if ef[-1] > es[-1]: return "UP"
    if ef[-1] < es[-1]: return "DOWN"
    return "FLAT"

def get_higher_tf_dirs(sym):
    higher_dirs = {}
    for tf in ["4h", "1d"]:
        kl = get_klines(sym, tf, limit=150)
        if not kl:
            higher_dirs[tf] = "FLAT"
            continue
        closes = [float(k[4]) for k in kl]
        ef, es, _ = EMA_SETS[tf]
        higher_dirs[tf] = ema_trend_direction_from_closes(closes, ef, es)
        time.sleep(0.05)
    return higher_dirs

def alignment_score(dirn, higher_dirs):
    score = 0
    matches = sum(1 for d in higher_dirs.values() if d == dirn)
    opposes = sum(1 for d in higher_dirs.values() if (d != dirn and d != "FLAT"))
    if matches == 2: score += 10
    elif matches == 1: score += 5
    if opposes == 2: score -= 10
    return score

def arrow_for(d): return "↑" if d == "UP" else "↓" if d == "DOWN" else "-"


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


# ---------- Smart Scalp Trigger ----------
def detect_slope_reversal(ema_series):
    """
    EMA7 eğiminin yön değiştirdiği anı yakalar.
    slope_prev = EMA[-2] - EMA[-5]
    slope_now  = EMA[-1] - EMA[-4]
    """
    if len(ema_series) < 6:
        return None, (0.0, 0.0)
    slope_now  = ema_series[-1] - ema_series[-4]
    slope_prev = ema_series[-2] - ema_series[-5]
    if slope_prev < 0 and slope_now > 0:
        return "UP", (slope_prev, slope_now)
    if slope_prev > 0 and slope_now < 0:
        return "DOWN", (slope_prev, slope_now)
    return None, (slope_prev, slope_now)


# ---------- Trend Retest Confirm ----------
def detect_trend_retest(closes, support, resistance, tol=0.003):
    """
    Kırılım + retest algısı:
    - Direnç üzeri kapanış sonrası fiyatın direnci ~tol içinde test etmesi → 'UP'
    - Destek altı kapanış sonrası fiyatın desteği ~tol içinde test etmesi → 'DOWN'
    """
    if support is None or resistance is None or len(closes) < 3:
        return None
    prev, curr = closes[-2], closes[-1]
    broke_res = prev < resistance and curr > resistance
    broke_sup = prev > support and curr < support
    retest_res = broke_res and abs(curr - resistance) / max(resistance, 1e-9) < tol
    retest_sup = broke_sup and abs(curr - support)    / max(support,    1e-9) < tol
    if retest_res: return "UP"
    if retest_sup: return "DOWN"
    return None


# ---------- ANA İŞ AKIŞI ----------
def process_symbol(sym, state):
    for interval in INTERVALS:
        kl = get_klines(sym, interval)
        if not kl or len(kl) < 220:
            continue

        closes = [float(k[4]) for k in kl]  # canlı dahil
        highs  = [float(k[2]) for k in kl]
        lows   = [float(k[3]) for k in kl]
        bar_close_ms = int(kl[-1][6])   # canlı barın planlanan kapanışı
        prev_bar_ms  = int(kl[-2][6])   # son kapanmış barın kapanışı
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        price  = closes[-1]

        ef, es, _ = EMA_SETS[interval]

        # KAPANMIŞ ve TAM seri EMA'ları
        closes_closed = closes[:-1]
        if len(closes_closed) < 3:
            continue
        ema_f_closed = ema(closes_closed, ef)
        ema_s_closed = ema(closes_closed, es)
        ema_f_full   = ema(closes, ef)
        ema_s_full   = ema(closes, es)

        # Stabilizasyon + Erken onay kontrolü (Sadece CROSS için)
        dirn, confirm_mode = stabilized_or_early(
            ema_f_closed, ema_s_closed,
            ema_f_full,   ema_s_full,
            bar_close_ms, now_ms,
            EARLY_CONFIRM_MS
        )

        key  = f"{sym}_{interval}"
        prev = state.get(key, {})

        # ==== 1) ANA CROSS / POWER SİNYALİ ====
        if dirn:
            # Aynı PREV bar için tekrar etme (erken/kapalı fark etmeksizin prev_bar_ms'ye pinliyoruz)
            if prev.get("last_signal_bar_ms") != prev_bar_ms or prev.get("last_dir") != dirn:
                # ATR / RSI
                atr = atr_series(highs, lows, closes, ATR_PERIOD)
                atr_now = atr[-1]
                atr_pct = (atr_now / price) if price > 0 else 0.0
                slope_now = slope_value(ema_f_closed, 3)  # kapalı seriden eğim
                min_pct, slope_mult = thresholds(interval)
                rsis = rsi(closes, RSI_PERIOD)
                rsi_val = rsis[-1] if rsis[-1] is not None else 50.0
                div_type, _ = detect_rsi_divergence(closes, rsis, RSI_SWING_LOOKBACK)
                rsi_status = f"{div_type} DIVERGENCE" if div_type else "NÖTR"
                # S/R
                support, resistance = trend_lines_from_extrema(closes, lookback=SR_LOOKBACK)

                # ---- Dynamic Momentum Power (slope/ATR)
                atr_ok = (atr_pct >= min_pct) and (atr_now > 0)
                slope_factor = (abs(slope_now) / (atr_now * slope_mult)) if (atr_now > 0 and slope_mult > 0) else 0.0
                atr_factor = (atr_pct / min_pct) if (min_pct > 0) else 1.0
                momentum_boost = min(25.0, (slope_factor * 10.0) + (atr_factor * 8.0))
                if slope_factor >= 2.0: momentum_tag = "🔥 Aşırı Güçlü"
                elif slope_factor >= 1.0: momentum_tag = "💪 Sağlam"
                elif slope_factor >= 0.5: momentum_tag = "⚠️ Zayıf"
                else: momentum_tag = "🧊 Çok Zayıf"

                power = 40.0 + momentum_boost
                if div_type:
                    power += 10
                    if (dirn == "UP" and div_type == "BULLISH") or (dirn == "DOWN" and div_type == "BEARISH"):
                        power += 15
                if not atr_ok:
                    power -= 3

                trend_line = None
                if interval == "1h":
                    higher_dirs = get_higher_tf_dirs(sym)
                    add = alignment_score(dirn, higher_dirs)
                    power += add
                    indicator = "✅" if add > 0 else "❌" if add < 0 else "➖"
                    trend_line = f"Trend Uyumu: {indicator} 4h{arrow_for(higher_dirs.get('4h','-'))} | 1d{arrow_for(higher_dirs.get('1d','-'))}"

                power = max(0.0, min(power, 100.0))

                # Power etiketi
                if power >= 86:
                    power_tag = "🟦 Ultra Power"; level = "ULTRA"
                elif power >= 70:
                    power_tag = "🟩 Strong";      level = "PREMIUM"
                elif power >= 50:
                    power_tag = "🟨 Moderate";    level = "CROSS"
                else:
                    power_tag = "🟥 Weak";        level = "CROSS"

                # SL/TP
                sl, tp1, tp2, tp3, rr1, rr2, rr3 = sl_tp_from_atr(price, atr_now, dirn)
                rr = lambda x: f"{x:.2f}"
                atr_tag = "[ATR OK]" if atr_ok else "[ATR LOW]"
                early_tag = " ⏳ Early Confirm (30dk)" if confirm_mode == "EARLY" else ""
                tag = "⚡ CROSS"
                if level == "PREMIUM": tag = "⚡🔥 PREMIUM SİNYAL"
                elif level == "ULTRA": tag = "⚡🚀 ULTRA PREMIUM SİNYAL"

                lines = [
                    f"{tag}: {sym} ({interval}) {atr_tag}{early_tag}",
                    f"Power: {power_tag} ({power:.0f}/100)",
                    f"Momentum: {momentum_tag} (Slope/ATR={slope_factor:.2f}, ATR%={atr_pct*100:.2f}%)",
                    f"Direction: {dirn} ({'LONG' if dirn=='UP' else 'SHORT'})",
                    f"Kesişim: EMA{ef} {'↗' if dirn=='UP' else '↘'} EMA{es}",
                    trend_line,
                    f"RSI: {rsi_val:.2f} → {rsi_status}",
                    f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)",
                    f"Slope(fast,3): {slope_now:.6f}",
                    f"Support≈ {support:.4f} | Resistance≈ {resistance:.4f}" if (support is not None and resistance is not None) else None,
                    f"Eşik[{interval}]: ATR%≥{min_pct*100:.2f} | slope_mult={slope_mult:.2f}",
                    f"Entry≈ {price}",
                    f"SL≈ {sl} | TP1≈ {tp1:.4f} (R:R {rr(rr1)})  TP2≈ {tp2:.4f} (R:R {rr(rr2)})  TP3≈ {tp3:.4f} (R:R {rr(rr3)})",
                    f"Time: {nowiso()}",
                ]
                msg = "\n".join([l for l in lines if l])
                send_telegram(msg)
                if level in ("PREMIUM", "ULTRA"):
                    log_premium(msg)

                # Sinyali stabilizasyon barına pin'le → kapanışta tekrar etmez
                state[key] = {**prev, "last_signal_bar_ms": prev_bar_ms, "last_dir": dirn}
                safe_save_json(STATE_FILE, state)
                time.sleep(SLEEP_BETWEEN)

        # ==== 2) SMART SCALP TRIGGER (Power'dan bağımsız) ====
        if interval == "1h":
            slope_flip, (s_prev, s_now) = detect_slope_reversal(ema_f_closed)
            if slope_flip:
                # Anti-spam: aynı bar & yön için tekrar yollama
                if prev.get("last_scalp_bar_ms") != prev_bar_ms or prev.get("last_scalp_dir") != slope_flip:
                    # Üst TF onayı
                    higher_dirs = get_higher_tf_dirs(sym)
                    tf_dir = higher_dirs.get(SCALP_TF_CONFIRM, "FLAT")

                    # ATR yeterliliği
                    atr = atr_series(highs, lows, closes, ATR_PERIOD)
                    atr_now = atr[-1]
                    atr_pct = (atr_now / price) if price > 0 else 0.0
                    min_pct_1h = ATR_MIN_PCT_DEFAULTS["1h"]
                    atr_ok_for_scalp = atr_pct >= (min_pct_1h * SCALP_MIN_ATR_FACTOR)

                    if slope_flip == tf_dir and atr_ok_for_scalp:
                        # Mini TP/SL
                        tp = price + (SCALP_TP_MULT * atr_now if slope_flip == "UP" else -SCALP_TP_MULT * atr_now)
                        sl = price - (SCALP_SL_MULT * atr_now if slope_flip == "UP" else -SCALP_SL_MULT * atr_now)

                        # Scalp momentum skoru (görsel)
                        denom = (atr_now * ATR_SLOPE_MULT_DEFAULTS["1h"]) if atr_now > 0 else 1.0
                        scalp_power = max(0.0, min(100.0, 60.0 + (abs(s_now - s_prev) / denom) * 20.0))

                        scalp_text = (
                            f"💥 SCALP {('LONG' if slope_flip=='UP' else 'SHORT')} TRIGGER: {sym} (1h)\n"
                            f"{SCALP_TF_CONFIRM.upper()} Trend Onayı: {tf_dir} ✅\n"
                            f"Slope Change: {s_prev:+.6f} → {s_now:+.6f}\n"
                            f"ATR({ATR_PERIOD}): {atr_now:.6f} ({atr_pct*100:.2f}%)\n"
                            f"TP≈ {tp:.4f} | SL≈ {sl:.4f}  (TP {SCALP_TP_MULT}×ATR, SL {SCALP_SL_MULT}×ATR)\n"
                            f"Power: ⚡ Momentum {scalp_power:.0f}\n"
                            f"Time: {nowiso()}"
                        )
                        send_telegram(scalp_text)

                        # scalp spam engeli
                        state[key] = {**state.get(key, {}), "last_scalp_bar_ms": prev_bar_ms, "last_scalp_dir": slope_flip}
                        safe_save_json(STATE_FILE, state)
                        time.sleep(SLEEP_BETWEEN)

        # ==== 3) TREND RETEST CONFIRM (Power'dan bağımsız) ====
        # S/R'yi tekrar hesaplamaya gerek yok; CROSS kısmında yoksa burada hesaplayalım:
        support, resistance = trend_lines_from_extrema(closes, lookback=SR_LOOKBACK)
        retest_dir = detect_trend_retest(closes, support, resistance)
        if retest_dir:
            # EMA7 eğimi ile aynı yönde onay
            slope_confirm = slope_value(ema_f_closed, 3)
            if (retest_dir == "UP" and slope_confirm > 0) or (retest_dir == "DOWN" and slope_confirm < 0):
                msg = (
                    f"✅ Trend Retest Confirm: {sym} ({interval})\n"
                    f"{'Direnç kırıldı → destek testi' if retest_dir=='UP' else 'Destek kırıldı → direnç testi'}\n"
                    f"Slope: {slope_confirm:+.6f}\n"
                    f"Support≈ {support:.4f} | Resistance≈ {resistance:.4f}\n"
                    f"Time: {nowiso()}"
                )
                send_telegram(msg)
                time.sleep(SLEEP_BETWEEN)


def main():
    log("🚀 v7 | Binance only | EMA/ATR/RSI — Stabilizasyon + 30dk Early Confirm + Dynamic Momentum Power + 1h→(4h&1d) + SCALP + RETEST (Power bağımsız)")
    state = safe_load_json(STATE_FILE)

    binance_symbols = get_futures_symbols()
    if not binance_symbols:
        log("❌ Binance sembol listesi boş.")
        return

    while True:
        for sym in binance_symbols:
            process_symbol(sym, state)
        log("⏳ 5 dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
