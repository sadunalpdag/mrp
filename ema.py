# === EMA ULTRA FINAL v11.6 — Entry Integrated Edition ===
# Confirmed Cross (1-bar) + Multi-Tier Premium Colors 🟨🟦🟩
# 4h + 1D Trend Alignment, RSI Divergence, ATR Boost
# ADX (trend gücü) + Volume Spike → Dinamik Power Ayarı
# Telegram sinyalleri, Günlük rapor, No-Repeat (CROSS & SCALP)
# ✅ v11.6: CROSS & SCALP için Entry/TP/SL hesapla, TG'de göster, CSV'ye yaz
#          Açık pozisyonları (her iki tip için) TP/SL ile kapat ve raporla

import os, json, csv, time, requests
from datetime import datetime, timedelta, timezone

# ========= AYARLAR =========
INTERVAL_1H, INTERVAL_4H, INTERVAL_1D = "1h", "4h", "1d"
LIMIT_1H, LIMIT_4H, LIMIT_1D = 500, 300, 250

ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14
VOL_MA_PERIOD = 20         # hacim ortalaması penceresi
VOL_SPIKE_MULT = 1.8       # spike eşiği: son hacim / vol_SMA >= 1.8

# Cross onayı (fake cross filtresi)
CROSS_CONFIRM_BARS = 1  # 0: anlık | 1: 1-bar onaylı (önerilen)

# Power katmanları
POWER_NORMAL_MIN  = 60.0
POWER_PREMIUM_MIN = 68.0
POWER_ULTRA_MIN   = 75.0

# ATR Boost
ATR_BOOST_PCT = 0.004        # ATR/Fiyat ≥ %0.4 → +5 power
ATR_BOOST_ADD = 5.0

# ADX/DVOL katkısı (dinamik)
ADX_BASE    = 25.0           # üzeri güçlü trend sayılır
ADX_MAX_ADD = 10.0           # power'a maks katkı
VOL_MAX_ADD = 7.0            # power'a maks katkı

# SCALP TP/SL
SCALP_TP_PCT = 0.006         # %0.6
SCALP_SL_PCT = 0.10          # %10

# CROSS TP/SL (v11.6: giriş eklendi)
CROSS_TP_PCT = 0.010         # %1.0
CROSS_SL_PCT = 0.030         # %3.0

# Tarama döngüsü / beklemeler
SCAN_INTERVAL = 300          # 5 dk
SLEEP_BETWEEN = 0.12

# Günlük özet (İstanbul, UTC+3)
DAILY_SUMMARY_ENABLED = True
DAILY_SUMMARY_TIME = "23:59"

# Duplicate önleme
SCALP_COOLDOWN_BARS = 3

# Dosyalar
STATE_FILE   = "alerts.json"
OPEN_CSV     = "open_positions.csv"
CLOSED_CSV   = "closed_trades.csv"
LOG_FILE     = "log.txt"

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ========= ZAMAN / LOG / TG =========
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)

def now_ist():
    return now_ist_dt().isoformat()

def today_ist_date():
    return now_ist_dt().strftime("%Y-%m-%d")

def now_ist_hhmm():
    return now_ist_dt().strftime("%H:%M")

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_ist()} - {msg}\n")
    except:
        pass

def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[!] Telegram env eksik")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=12)
        log(f"[TG] {text.splitlines()[0]} ...")
    except Exception as e:
        log(f"Telegram error: {e}")

# ========= STATE / CSV =========
def ensure_dir(p):
    try: os.makedirs(p, exist_ok=True)
    except: pass

def ensure_csv(path, headers):
    ensure_dir(os.path.dirname(path) or ".")
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def safe_load_json(path):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
    except: pass
    return {}

def safe_save_json(path, data):
    try:
        tmp = path + ".tmp"
        json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except: pass

# CSV şemaları (v11.6: type sütunu eklendi)
ensure_csv(OPEN_CSV,   ["symbol","type","direction","entry","tp","sl","power","rsi","divergence","time_open"])
ensure_csv(CLOSED_CSV, ["symbol","type","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close"])

def ensure_state(st):
    st.setdefault("last_slope_dir", {})        # {sym: "UP"/"DOWN"} (bilgilendirme)
    st.setdefault("open_positions", [])        # [{symbol,type,dir,entry,tp,sl,power,rsi,div,open,bar}]
    st.setdefault("last_daily_summary_date", "")
    today = today_ist_date()
    st.setdefault("daily_counters", {
        "date": today,
        "cross_normal": 0, "cross_premium": 0, "cross_ultra": 0,
        "scalp_premium": 0, "scalp_ultra": 0,
        "cross_align_total": 0, "cross_align_match": 0
    })
    st.setdefault("last_cross_seen", {})   # { "BTCUSDT_UP":  bar_close_ms_confirmed }
    st.setdefault("last_scalp_seen", {})   # { "BTCUSDT_DOWN": last_bar_index }
    return st

def roll_daily_counters_if_needed(state):
    t = today_ist_date()
    if state["daily_counters"].get("date") != t:
        state["daily_counters"] = {
            "date": t,
            "cross_normal": 0, "cross_premium": 0, "cross_ultra": 0,
            "scalp_premium": 0, "scalp_ultra": 0,
            "cross_align_total": 0, "cross_align_match": 0
        }

# ========= İNDİKATÖRLER =========
def ema(vals, length):
    k = 2 / (length + 1)
    e = [vals[0]]
    for i in range(1, len(vals)):
        e.append(vals[i] * k + e[-1] * (1 - k))
    return e

def sma(vals, period):
    if len(vals) < period: return [sum(vals)/len(vals)]*len(vals)
    out=[]
    s=sum(vals[:period])
    out.extend([s/period]*(period-1))
    out.append(s/period)
    for i in range(period, len(vals)):
        s += vals[i] - vals[i-period]
        out.append(s/period)
    return out

def atr_series(highs, lows, closes, period=14):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            pc = closes[i-1]
            trs.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    if len(trs) < period: return [0]*len(trs)
    a = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        a.append((a[-1]*(period-1) + trs[i]) / period)
    return [0]*(len(trs)-len(a)) + a

def rsi(vals, period=14):
    if len(vals) < period + 1: return [50]*len(vals)
    deltas = [vals[i] - vals[i-1] for i in range(1, len(vals))]
    gains  = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50]*period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else 0
        rsis.append(100 - 100/(1 + rs))
    return [50]*(len(vals)-len(rsis)) + rsis

def adx_series(highs, lows, closes, period=14):
    if len(highs) < period + 2: return [0]*len(highs), [0]*len(highs), [0]*len(highs)
    tr, plus_dm, minus_dm = [], [], []
    for i in range(len(highs)):
        if i == 0:
            tr.append(highs[i]-lows[i]); plus_dm.append(0); minus_dm.append(0)
        else:
            up = highs[i] - highs[i-1]
            dn = lows[i-1] - lows[i]
            plus_dm.append(up if (up>dn and up>0) else 0)
            minus_dm.append(dn if (dn>up and dn>0) else 0)
            pc = closes[i-1]
            tr.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    tr_s   = [sum(tr[:period])]
    plus_s = [sum(plus_dm[:period])]
    minus_s= [sum(minus_dm[:period])]
    for i in range(period, len(highs)):
        tr_s.append(tr_s[-1] - tr_s[-1]/period + tr[i])
        plus_s.append(plus_s[-1] - plus_s[-1]/period + plus_dm[i])
        minus_s.append(minus_s[-1] - minus_s[-1]/period + minus_dm[i])
    di_plus = [0]*(period) + [ (plus_s[i-period]/tr_s[i-period])*100 if tr_s[i-period]!=0 else 0 for i in range(period, len(tr_s)) ]
    di_minus= [0]*(period) + [ (minus_s[i-period]/tr_s[i-period])*100 if tr_s[i-period]!=0 else 0 for i in range(period, len(tr_s)) ]
    dx = []
    for i in range(len(di_plus)):
        dplus = di_plus[i]; dminus = di_minus[i]
        den = (dplus + dminus)
        dx.append( (abs(dplus - dminus)/den*100) if den != 0 else 0 )
    adx = []
    first = sum(dx[period:period*2]) / period if len(dx) >= period*2 else 0
    adx = [0]*(period*2) + [first] if len(dx) >= period*2 else [0]*len(dx)
    for i in range(period*2+1, len(dx)):
        adx.append((adx[-1]*(period-1) + dx[i]) / period)
    need = len(highs) - len(adx)
    adx = [0]*need + adx if need>0 else adx[:len(highs)]
    di_plus = di_plus[:len(highs)]
    di_minus= di_minus[:len(highs)]
    return adx, di_plus, di_minus

# ========= POWER / DIVERGENCE / BOOST =========
def rsi_divergence(last_close, prev_close, rsi_now, rsi_prev):
    if last_close < prev_close and rsi_now > rsi_prev:  return "Bullish"
    if last_close > prev_close and rsi_now < rsi_prev:  return "Bearish"
    return "Neutral"

def atr_boost(atr_now, price):
    atr_pct = (atr_now / price) if price > 0 else 0.0
    return (ATR_BOOST_ADD if atr_pct >= ATR_BOOST_PCT else 0.0), atr_pct

def tier_color(power):
    if power >= POWER_ULTRA_MIN:   return "ULTRA", "🟩"
    if power >= POWER_PREMIUM_MIN: return "PREMIUM", "🟦"
    if power >= POWER_NORMAL_MIN:  return "NORMAL", "🟨"
    return "NONE", ""

def power_base(s_prev, s_now, atr_now, price, rsi_now):
    slope_comp = abs(s_now - s_prev) / (atr_now * 0.6) if atr_now > 0 else 0.0
    rsi_comp   = (rsi_now - 50) / 50.0
    atr_comp   = (atr_now / price) * 100.0 if price > 0 else 0.0
    base = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    return max(0.0, min(100.0, base))

def power_with_adx_vol(base_power, adx_now, vol_now, vol_ma):
    add = 0.0
    if adx_now > ADX_BASE:
        add += min(ADX_MAX_ADD, (adx_now - ADX_BASE) / 15.0 * ADX_MAX_ADD)
    mult = (vol_now / vol_ma) if (vol_ma and vol_ma>0) else 1.0
    if mult >= 1.0:
        add += min(VOL_MAX_ADD, (mult - 1.0) / (VOL_SPIKE_MULT - 1.0) * VOL_MAX_ADD) if mult >= VOL_SPIKE_MULT else (mult - 1.0) * (VOL_MAX_ADD / (VOL_SPIKE_MULT - 1.0))
    return max(0.0, min(100.0, base_power + add)), mult

# ========= BINANCE =========
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v11.6", "Accept": "application/json"})

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=12)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except: return []

def get_klines(symbol, interval, limit=500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
        if r.status_code == 200: return r.json()
    except: pass
    return []

def get_last_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        r = SESSION.get(url, timeout=6).json()
        return float(r["price"])
    except: return None

# ========= TREND HELPERS =========
def ema_trend_dir(closes):
    e7, e25 = ema(closes,7), ema(closes,25)
    return "UP" if e7[-1] > e25[-1] else "DOWN"

def trend_alignment(signal_dir, closes_4h, closes_1d):
    d4 = ema_trend_dir(closes_4h)
    d1 = ema_trend_dir(closes_1d)
    match = (signal_dir == d4) and (signal_dir == d1)
    return match, d4, d1

# ========= ENGINES =========
def run_cross_engine_confirmed(sym, kl1, kl4, kl1d):
    closes = [float(k[4]) for k in kl1]
    ema7, ema25 = ema(closes,7), ema(closes,25)

    cross_dir = None
    if CROSS_CONFIRM_BARS == 1:
        prev_diff    = ema7[-3] - ema25[-3]
        cross_diff   = ema7[-2] - ema25[-2]
        confirm_diff = ema7[-1] - ema25[-1]
        if prev_diff < 0 and cross_diff > 0 and confirm_diff > 0: cross_dir = "UP"
        if prev_diff > 0 and cross_diff < 0 and confirm_diff < 0: cross_dir = "DOWN"
        bar_close_ms = int(kl1[-1][6])  # confirm bar
    else:
        prev_diff = ema7[-2] - ema25[-2]
        curr_diff = ema7[-1] - ema25[-1]
        if prev_diff < 0 and curr_diff > 0: cross_dir = "UP"
        if prev_diff > 0 and curr_diff < 0: cross_dir = "DOWN"
        bar_close_ms = int(kl1[-1][6])

    if not cross_dir: return None

    highs = [float(k[2]) for k in kl1]
    lows  = [float(k[3]) for k in kl1]
    vols  = [float(k[5]) for k in kl1]
    atr_now = atr_series(highs, lows, closes, ATR_PERIOD)[-1]
    rsi_all = rsi(closes, RSI_PERIOD)
    rsi_now, rsi_prev = rsi_all[-1], rsi_all[-2]
    s_now  = ema7[-1] - ema7[-4]
    s_prev = ema7[-2] - ema7[-5]
    price  = closes[-1]

    adx_all, _, _ = adx_series(highs, lows, closes, ADX_PERIOD)
    adx_now = adx_all[-1]
    vol_ma = sma(vols, VOL_MA_PERIOD)[-1]

    base = power_base(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr_after_boost = base + boost
    pwr_final, vol_mult = power_with_adx_vol(pwr_after_boost, adx_now, vols[-1], vol_ma)

    div = rsi_divergence(closes[-1], closes[-2], rsi_now, rsi_prev)

    closes4  = [float(k[4]) for k in kl4]
    closes1d = [float(k[4]) for k in kl1d]
    aligned, d4, d1 = trend_alignment(cross_dir, closes4, closes1d)

    # v11.6: Entry/TP/SL
    entry = price
    tp = entry * (1 + CROSS_TP_PCT if cross_dir == "UP" else 1 - CROSS_TP_PCT)
    sl = entry * (1 - CROSS_SL_PCT if cross_dir == "UP" else 1 + CROSS_SL_PCT)

    return {
        "symbol": sym, "type": "CROSS", "dir": cross_dir, "price": price,
        "entry": entry, "tp": tp, "sl": sl,
        "power": pwr_final, "rsi": rsi_now, "div": div,
        "atr": atr_now, "atr_pct": atr_pct, "bar_close_ms": bar_close_ms,
        "aligned": aligned, "trend4h": d4, "trend1d": d1, "confirmed": CROSS_CONFIRM_BARS==1,
        "adx": adx_now, "vol_mult": vol_mult
    }

def run_scalp_engine(sym, kl1, kl4, kl1d):
    closes1 = [float(k[4]) for k in kl1]
    ema7_1  = ema(closes1, 7)
    if len(ema7_1) < 6: return None
    s_now  = ema7_1[-1] - ema7_1[-4]
    s_prev = ema7_1[-2] - ema7_1[-5]
    slope_dir = "UP" if (s_prev < 0 and s_now > 0) else ("DOWN" if (s_prev > 0 and s_now < 0) else None)
    if not slope_dir: return None

    closes4  = [float(k[4]) for k in kl4]
    closes1d = [float(k[4]) for k in kl1d]
    d4 = ema_trend_dir(closes4)
    if slope_dir != d4: return None   # scalp için 4h şart
    d1 = ema_trend_dir(closes1d)
    aligned = (slope_dir == d4) and (slope_dir == d1)

    highs1 = [float(k[2]) for k in kl1]
    lows1  = [float(k[3]) for k in kl1]
    vols1  = [float(k[5]) for k in kl1]
    atr_now = atr_series(highs1, lows1, closes1, ATR_PERIOD)[-1]
    rsi_now = rsi(closes1, RSI_PERIOD)[-1]
    price   = closes1[-1]

    adx_all, _, _ = adx_series(highs1, lows1, closes1, ADX_PERIOD)
    adx_now = adx_all[-1]
    vol_ma  = sma(vols1, VOL_MA_PERIOD)[-1]

    base = power_base(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr_after_boost = base + boost
    pwr_final, vol_mult = power_with_adx_vol(pwr_after_boost, adx_now, vols1[-1], vol_ma)

    rsi_prev = rsi(closes1, RSI_PERIOD)[-2]
    div = rsi_divergence(closes1[-1], closes1[-2], rsi_now, rsi_prev)

    if pwr_final < POWER_PREMIUM_MIN: return None

    entry = price
    tp = entry * (1 + SCALP_TP_PCT if slope_dir == "UP" else 1 - SCALP_TP_PCT)
    sl = entry * (1 - SCALP_SL_PCT if slope_dir == "UP" else 1 + SCALP_SL_PCT)

    return {
        "symbol": sym, "type": "SCALP", "dir": slope_dir, "price": price,
        "entry": entry, "tp": tp, "sl": sl,
        "power": pwr_final, "rsi": rsi_now, "div": div,
        "atr": atr_now, "atr_pct": atr_pct,
        "aligned": aligned, "trend4h": d4, "trend1d": d1,
        "adx": adx_now, "vol_mult": vol_mult
    }

# ========= GÜNLÜK ÖZET =========
def build_daily_summary_for(date_str, state):
    total = wins = loses = 0
    pnl_sum = 0.0
    try:
        with open(CLOSED_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row.get("time_close","").startswith(date_str): continue
                total += 1
                res = row.get("result","")
                pnl_sum += float(row.get("pnl","0") or 0)
                if res == "TP": wins += 1
                elif res == "SL": loses += 1
    except FileNotFoundError:
        pass
    winrate = (wins/total*100.0) if total else 0.0
    avg_pnl = (pnl_sum/total) if total else 0.0
    dc = state.get("daily_counters", {})
    align_rate = (dc["cross_align_match"]/dc["cross_align_total"]*100.0) if dc.get("cross_align_total",0)>0 else 0.0
    return {
        "date": date_str,
        "winrate": winrate, "avg_pnl": avg_pnl,
        "open_cnt": len(state.get("open_positions", [])),
        "dc": dc, "align_rate": align_rate
    }

def maybe_send_daily_summary(state):
    if not DAILY_SUMMARY_ENABLED: return
    if now_ist_hhmm() >= DAILY_SUMMARY_TIME and state.get("last_daily_summary_date") != today_ist_date():
        s = build_daily_summary_for(today_ist_date(), state)
        dc = s["dc"]
        send_tg(
            "📊 GÜNLÜK RAPOR (İstanbul)\n"
            f"📅 {s['date']}\n"
            f"Signals → CROSS: 🟨{dc.get('cross_normal',0)} | 🟦{dc.get('cross_premium',0)} | 🟩{dc.get('cross_ultra',0)}\n"
            f"           SCALP: 🟦{dc.get('scalp_premium',0)} | 🟩{dc.get('scalp_ultra',0)}\n"
            f"Alignment (CROSS) → {s['align_rate']:.1f}%\n"
            f"Winrate: {s['winrate']:.1f}% | AvgPnL: {s['avg_pnl']:.2f}%\n"
            f"Açık Pozisyon: {s['open_cnt']}"
        )
        state["last_daily_summary_date"] = today_ist_date()

# ========= BINANCE HELPERS =========
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v11.6", "Accept": "application/json"})

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=12)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except: return []

def get_klines(symbol, interval, limit=500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
        if r.status_code == 200: return r.json()
    except: pass
    return []

def get_last_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        r = SESSION.get(url, timeout=6).json()
        return float(r["price"])
    except: return None

# ========= ANA DÖNGÜ =========
def main():
    log("🚀 v11.6 başladı (Entry/TP/SL dahil)")
    state = ensure_state(safe_load_json(STATE_FILE))

    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Sembol yok"); return

    bar = 0
    while True:
        bar += 1
        for sym in symbols:
            kl1  = get_klines(sym, INTERVAL_1H, limit=LIMIT_1H)
            kl4  = get_klines(sym, INTERVAL_4H, limit=LIMIT_4H)
            kl1d = get_klines(sym, INTERVAL_1D, limit=LIMIT_1D)
            if not kl1 or len(kl1)<120 or not kl4 or len(kl4)<50 or not kl1d or len(kl1d)<50:
                continue

            # === CROSS (Confirmed) ===
            cross = run_cross_engine_confirmed(sym, kl1, kl4, kl1d)
            if cross:
                cross_key = f"{sym}_{cross['dir']}"
                last_seen_bar = state["last_cross_seen"].get(cross_key)
                if last_seen_bar != cross["bar_close_ms"]:
                    tier, color = tier_color(cross["power"])
                    if tier != "NONE":
                        # alignment sayaçları
                        state["daily_counters"]["cross_align_total"] += 1
                        if cross["aligned"]:
                            state["daily_counters"]["cross_align_match"] += 1
                        if tier=="ULTRA":   state["daily_counters"]["cross_ultra"] += 1
                        elif tier=="PREMIUM": state["daily_counters"]["cross_premium"] += 1
                        elif tier=="NORMAL":  state["daily_counters"]["cross_normal"] += 1

                        align_tag = "✅ 4h/1D Trend Match" if cross["aligned"] else f"⚠️ Counter-Trend (4h {cross['trend4h']}, 1D {cross['trend1d']})"
                        boost_tag = " | ATR Boosted" if cross["atr_pct"] >= ATR_BOOST_PCT else ""
                        div_tag   = {"Bullish":"⚡ Strong Up Momentum","Bearish":"⚠️ Weak Momentum"}.get(cross["div"], "Neutral Divergence")
                        confirm_tag = "Confirmed: ✅ (1-bar)" if cross["confirmed"] else "Confirmed: —"
                        adx_tag = f"ADX: {cross['adx']:.1f}"
                        vol_tag = f"VOL Spike x{cross['vol_mult']:.2f}" if cross["vol_mult"] >= VOL_SPIKE_MULT else f"VOL x{cross['vol_mult']:.2f}"

                        header_map = {"ULTRA":"ULTRA CROSS","PREMIUM":"PREMIUM CROSS","NORMAL":"CROSS SIGNAL"}
                        header = f"{color} {header_map[tier]}: {sym} ({INTERVAL_1H})"

                        send_tg(
                            f"{header}\n"
                            f"Entry: {cross['entry']:.4f} | TP: {cross['tp']:.4f} | SL: {cross['sl']:.4f}\n"
                            f"Direction: {cross['dir']}\n"
                            f"{align_tag}\n"
                            f"RSI: {cross['rsi']:.1f} | Power: {cross['power']:.1f}{boost_tag}\n"
                            f"Divergence: {cross['div']} ({div_tag})\n"
                            f"{confirm_tag} | {adx_tag} | {vol_tag}\n"
                            f"ATR({ATR_PERIOD}): {cross['atr']:.6f} ({cross['atr_pct']*100:.2f}%)\n"
                            f"Time: {now_ist()}"
                        )
                        # v11.6: Açık pozisyon (CROSS) kayıt
                        with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([sym, "CROSS", cross["dir"], cross["entry"], cross["tp"], cross["sl"],
                                                    cross["power"], cross["rsi"], cross["div"], now_ist()])
                        state["open_positions"].append({
                            "symbol": sym, "type": "CROSS", "dir": cross["dir"],
                            "entry": cross["entry"], "tp": cross["tp"], "sl": cross["sl"],
                            "power": cross["power"], "rsi": cross["rsi"], "div": cross["div"],
                            "open": now_ist(), "bar": bar
                        })
                    state["last_cross_seen"][cross_key] = cross["bar_close_ms"]

            # === SCALP (4h trend uyumlu) ===
            scalp = run_scalp_engine(sym, kl1, kl4, kl1d)
            if scalp:
                scalp_key = f"{sym}_{scalp['dir']}"
                last_idx = state["last_scalp_seen"].get(scalp_key)
                if last_idx is None or (bar - last_idx) > SCALP_COOLDOWN_BARS:
                    tier, color = tier_color(scalp["power"])  # ≥68 buraya zaten girdi
                    align_tag = "✅ 4h/1D Trend Match" if scalp["aligned"] else f"ℹ️ 4h OK, 1D {scalp['trend1d']}"
                    boost_tag = " ⚡ ATR Boost" if scalp["atr_pct"] >= ATR_BOOST_PCT else ""
                    adx_tag   = f"ADX: {scalp['adx']:.1f}"
                    vol_tag   = f"VOL Spike x{scalp['vol_mult']:.2f}" if scalp["vol_mult"] >= VOL_SPIKE_MULT else f"VOL x{scalp['vol_mult']:.2f}"
                    header = f"{color} {'ULTRA' if tier=='ULTRA' else 'PREMIUM'} SCALP: {sym} ({INTERVAL_1H})"

                    send_tg(
                        f"{header}\n"
                        f"Entry: {scalp['entry']:.4f} | TP: {scalp['tp']:.4f} | SL: {scalp['sl']:.4f}\n"
                        f"Direction: {scalp['dir']} (4h trend uyumlu)\n"
                        f"{align_tag}\n"
                        f"RSI: {scalp['rsi']:.1f} | Power: {scalp['power']:.1f}{boost_tag}\n"
                        f"{adx_tag} | {vol_tag}\n"
                        f"Time: {now_ist()}"
                    )
                    if tier=="ULTRA": state["daily_counters"]["scalp_ultra"] += 1
                    else:             state["daily_counters"]["scalp_premium"] += 1

                    state["last_scalp_seen"][scalp_key] = bar
                    with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([sym, "SCALP", scalp["dir"], scalp["entry"], scalp["tp"], scalp["sl"],
                                                scalp["power"], scalp["rsi"], scalp["div"], now_ist()])
                    state["open_positions"].append({
                        "symbol": sym, "type": "SCALP", "dir": scalp["dir"],
                        "entry": scalp["entry"], "tp": scalp["tp"], "sl": scalp["sl"],
                        "power": scalp["power"], "rsi": scalp["rsi"], "div": scalp["div"],
                        "open": now_ist(), "bar": bar
                    })

            time.sleep(SLEEP_BETWEEN)

        # === Açık pozisyonları TP/SL ile yönet (CROSS + SCALP) ===
        still_open=[]
        for t in state["open_positions"]:
            lp = get_last_price(t["symbol"])
            if lp is None:
                still_open.append(t); continue
            # TP/SL kontrol
            hit_tp = (lp >= t["tp"]) if t["dir"]=="UP" else (lp <= t["tp"])
            hit_sl = (lp <= t["sl"]) if t["dir"]=="UP" else (lp >= t["sl"])
            if not (hit_tp or hit_sl):
                still_open.append(t); continue

            res = "TP" if hit_tp else "SL"
            pnl = (lp - t["entry"])/t["entry"]*100 if t["dir"]=="UP" else (t["entry"] - lp)/t["entry"]*100
            bars_open = bar - t["bar"]

            send_tg(
                f"📘 {res} | {t['symbol']} {t['type']} {t['dir']}\n"
                f"Entry: {t['entry']:.4f}  Exit: {lp:.4f}\n"
                f"PnL: {pnl:.2f}%  Bars: {bars_open}\n"
                f"From: {t['type']}"
            )
            with open(CLOSED_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([t["symbol"], t["type"], t["dir"], t["entry"], lp, res, pnl, bars_open,
                                        t["power"], t["rsi"], t["div"], t["open"], now_ist()])
        state["open_positions"] = still_open

        # günlük idare
        roll_daily_counters_if_needed(state)
        maybe_send_daily_summary(state)

        safe_save_json(STATE_FILE, state)
        log(f"Scan done | Open: {len(state['open_positions'])}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()