# === EMA ULTRA FINAL v9.7 ===
# Hybrid: EMA Cross Engine (v9.2 tarzı, Power ≥ 60) + Scalp Slope Engine (4H trend uyumlu, Power ≥ 68, TP/SL)
# RSI Divergence + ATR Boost (+5 @ ATR% ≥ 0.4) | Günlük Özet (23:59 UTC+3) | Telegram | CSV Kayıtları

import os, json, csv, time, requests
from datetime import datetime, timedelta, timezone

# ========= SABİTLER / AYARLAR =========
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
LIMIT_1H = 500
LIMIT_4H = 300

ATR_PERIOD = 14
RSI_PERIOD = 14

# Güç eşikleri
POWER_NORMAL_MIN  = 60.0   # Cross için min power
POWER_PREMIUM_MIN = 68.0   # Scalp için min power

# ATR Boost (volatilite)
ATR_BOOST_PCT = 0.004      # ATR/Fiyat ≥ %0.4 -> +5 power
ATR_BOOST_ADD = 5.0

# Scalp TP/SL
SCALP_TP_PCT = 0.006       # %0.6
SCALP_SL_PCT = 0.10        # %10

# Taramalar
SCAN_INTERVAL = 300        # 5 dk
SLEEP_BETWEEN = 0.12

# Günlük rapor (İstanbul saati)
DAILY_SUMMARY_ENABLED = True
DAILY_SUMMARY_TIME = "23:59"  # HH:MM (UTC+3)

# Dosyalar
STATE_FILE = "alerts.json"
OPEN_CSV   = "open_positions.csv"
CLOSED_CSV = "closed_trades.csv"
LOG_FILE   = "log.txt"

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ========= ZAMAN / LOG =========
def now_ist():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0).isoformat()

def today_ist_date():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%d")

def now_ist_hhmm():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%H:%M")

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_ist()} - {msg}\n")
    except:
        pass

def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[!] Telegram bilgileri eksik")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=12)
        log(f"[TG] {text.splitlines()[0]} ...")
    except Exception as e:
        log(f"Telegram hatası: {e}")

def send_doc(bytes_data, filename, caption=""):
    if not BOT_TOKEN or not CHAT_ID:
        log("[!] Telegram bilgileri eksik (send_doc)")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        files = {"document": (filename, bytes_data)}
        data = {"chat_id": CHAT_ID, "caption": caption}
        requests.post(url, files=files, data=data, timeout=20)
        log(f"[TG] Document sent: {filename}")
    except Exception as e:
        log(f"send_doc hatası: {e}")

# ========= DOSYA / STATE =========
def safe_load_json(path):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
    except:
        pass
    return {}

def safe_save_json(path, data):
    try:
        tmp = path + ".tmp"
        json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except:
        pass

def ensure_csv(path, headers):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

ensure_csv(OPEN_CSV,   ["symbol","direction","entry","tp","sl","power","rsi","divergence","time_open"])
ensure_csv(CLOSED_CSV, ["symbol","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close"])

def ensure_state(st):
    st.setdefault("last_slope_dir", {})       # scalp duplicate engel
    st.setdefault("last_cross", {})           # cross duplicate engel: {sym: {"dir":UP/DOWN, "bar_close":ms}}
    st.setdefault("open_positions", [])       # scalp açık işlemler
    st.setdefault("last_daily_summary_date", "")
    today = today_ist_date()
    st.setdefault("daily_counters", {"date": today, "cross": 0, "scalp": 0})
    return st

def roll_daily_counters_if_needed(state):
    t = today_ist_date()
    if state.get("daily_counters", {}).get("date") != t:
        state["daily_counters"] = {"date": t, "cross": 0, "scalp": 0}

def build_daily_summary_for(date_str, state):
    total = wins = loses = 0
    pnl_sum = 0.0
    try:
        with open(CLOSED_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                tc = row.get("time_close", "")
                if not tc.startswith(date_str):
                    continue
                total += 1
                res = row.get("result", "")
                pnl = float(row.get("pnl", "0") or 0)
                pnl_sum += pnl
                if res == "TP":
                    wins += 1
                elif res == "SL":
                    loses += 1
    except FileNotFoundError:
        pass

    winrate = (wins / total * 100.0) if total else 0.0
    avg_pnl = (pnl_sum / total) if total else 0.0
    open_cnt = len(state.get("open_positions", []))
    counters = state.get("daily_counters", {"cross": 0, "scalp": 0})
    return {
        "date": date_str,
        "trades": total,
        "wins": wins,
        "loses": loses,
        "winrate": winrate,
        "avg_pnl": avg_pnl,
        "open_positions": open_cnt,
        "cross_signals": counters.get("cross", 0),
        "scalp_signals": counters.get("scalp", 0),
    }

def maybe_send_daily_summary(state):
    if not DAILY_SUMMARY_ENABLED:
        return
    hhmm = now_ist_hhmm()
    today = today_ist_date()
    if hhmm >= DAILY_SUMMARY_TIME and state.get("last_daily_summary_date") != today:
        s = build_daily_summary_for(today, state)
        msg = (
            "📊 GÜNLÜK RAPOR (İstanbul)\n"
            f"📅 {s['date']}\n"
            f"Signals → CROSS: {s['cross_signals']} | SCALP: {s['scalp_signals']}\n"
            f"Trades  → Total: {s['trades']} | TP: {s['wins']} | SL: {s['loses']}\n"
            f"Winrate → {s['winrate']:.1f}%  | AvgPnL: {s['avg_pnl']:.2f}%\n"
            f"Açık Pozisyon: {s['open_positions']}"
        )
        send_tg(msg)
        state["last_daily_summary_date"] = today

# ========= İNDİKATÖRLER =========
def ema(vals, length):
    k = 2 / (length + 1)
    e = [vals[0]]
    for i in range(1, len(vals)):
        e.append(vals[i] * k + e[-1] * (1 - k))
    return e

def atr_series(highs, lows, closes, period=14):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            pc = closes[i - 1]
            trs.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    if len(trs) < period:
        return [0]*len(trs)
    a = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        a.append((a[-1]*(period-1) + trs[i]) / period)
    return [0]*(len(trs)-len(a)) + a

def rsi(vals, period=14):
    if len(vals) < period + 1:
        return [50]*len(vals)
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

# ========= POWER v2 + Divergence + Boost =========
def power_v2(s_prev, s_now, atr_now, price, rsi_now):
    slope_comp = abs(s_now - s_prev) / (atr_now * 0.6) if atr_now > 0 else 0.0
    rsi_comp   = (rsi_now - 50) / 50.0
    atr_comp   = (atr_now / price) * 100.0 if price > 0 else 0.0
    base = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    return max(0.0, min(100.0, base))

def rsi_divergence(last_close, prev_close, rsi_now, rsi_prev):
    if last_close < prev_close and rsi_now > rsi_prev:
        return "Bullish"
    if last_close > prev_close and rsi_now < rsi_prev:
        return "Bearish"
    return "Neutral"

def atr_boost(atr_now, price):
    atr_pct = (atr_now / price) if price > 0 else 0.0
    return ATR_BOOST_ADD if atr_pct >= ATR_BOOST_PCT else 0.0, atr_pct

# ========= BINANCE =========
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v9.7", "Accept": "application/json"})

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=12)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except:
        return []

def get_klines(symbol, interval, limit=500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def get_last_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        r = SESSION.get(url, timeout=6).json()
        return float(r["price"])
    except:
        return None

# ========= MOTOR: EMA CROSS (v9.2 tarzı, Power ≥ 60) =========
def run_cross_engine(sym, kl1):
    closes = [float(k[4]) for k in kl1]
    ema7   = ema(closes, 7)
    ema25  = ema(closes, 25)
    ema99  = ema(closes, 99)  # bilgilendirme için

    # Sadece son barda kesişim
    prev_diff = ema7[-2] - ema25[-2]
    curr_diff = ema7[-1] - ema25[-1]
    cross_dir = None
    if prev_diff < 0 and curr_diff > 0: cross_dir = "UP"
    if prev_diff > 0 and curr_diff < 0: cross_dir = "DOWN"
    if not cross_dir: 
        return None

    # Güç, divergence, ATR
    highs = [float(k[2]) for k in kl1]
    lows  = [float(k[3]) for k in kl1]
    atr_now = atr_series(highs, lows, closes, ATR_PERIOD)[-1]
    rsi_all = rsi(closes, RSI_PERIOD)
    rsi_now, rsi_prev = rsi_all[-1], rsi_all[-2]
    s_now  = ema7[-1] - ema7[-4]
    s_prev = ema7[-2] - ema7[-5]
    price  = closes[-1]

    pwr = power_v2(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr += boost
    div = rsi_divergence(closes[-1], closes[-2], rsi_now, rsi_prev)

    return {
        "symbol": sym,
        "dir": cross_dir,
        "price": price,
        "ema99": ema99[-1],
        "power": pwr,
        "rsi": rsi_now,
        "div": div,
        "atr": atr_now,
        "atr_pct": atr_pct,
        "bar_close_ms": int(kl1[-1][6])
    }

# ========= MOTOR: SCALP SLOPE (trend uyumlu, Power ≥ 68, TP/SL) =========
def run_scalp_engine(sym, kl1, kl4):
    closes1 = [float(k[4]) for k in kl1]
    highs1  = [float(k[2]) for k in kl1]
    lows1   = [float(k[3]) for k in kl1]
    ema7_1  = ema(closes1, 7)
    if len(ema7_1) < 6: 
        return None

    # Slope reversal
    s_now  = ema7_1[-1] - ema7_1[-4]
    s_prev = ema7_1[-2] - ema7_1[-5]
    slope_flip = None
    if s_prev < 0 and s_now > 0: slope_flip = "UP"
    if s_prev > 0 and s_now < 0: slope_flip = "DOWN"
    if not slope_flip:
        return None

    # 4H trend
    closes4 = [float(k[4]) for k in kl4]
    ema7_4  = ema(closes4, 7)
    ema25_4 = ema(closes4, 25)
    trend_4h = "UP" if ema7_4[-1] > ema25_4[-1] else "DOWN"
    if slope_flip != trend_4h:
        return None  # trend uyumsuz ise scalp gösterme

    price   = closes1[-1]
    atr_now = atr_series(highs1, lows1, closes1, ATR_PERIOD)[-1]
    rsi_now = rsi(closes1, RSI_PERIOD)[-1]
    pwr     = power_v2(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr += boost

    # Divergence bilgi
    rsi_prev = rsi(closes1, RSI_PERIOD)[-2]
    div = rsi_divergence(closes1[-1], closes1[-2], rsi_now, rsi_prev)

    if pwr < POWER_PREMIUM_MIN:
        return None

    tp = price * (1 + SCALP_TP_PCT if slope_flip == "UP" else 1 - SCALP_TP_PCT)
    sl = price * (1 - SCALP_SL_PCT if slope_flip == "UP" else 1 + SCALP_SL_PCT)

    return {
        "symbol": sym,
        "dir": slope_flip,
        "price": price,
        "tp": tp,
        "sl": sl,
        "power": pwr,
        "rsi": rsi_now,
        "div": div,
        "atr": atr_now,
        "atr_pct": atr_pct,
    }

# ========= ANA DÖNGÜ =========
def main():
    log("🚀 v9.7 Başladı (Cross≥60 + Scalp≥68 | RSI Divergence + ATR Boost | Günlük Rapor)")
    state = ensure_state(safe_load_json(STATE_FILE))

    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Sembol alınamadı.")
        return

    bar = 0
    while True:
        bar += 1
        for sym in symbols:
            kl1 = get_klines(sym, INTERVAL_1H, limit=LIMIT_1H)
            kl4 = get_klines(sym, INTERVAL_4H, limit=LIMIT_4H)
            if not kl1 or len(kl1) < 120 or not kl4 or len(kl4) < 50:
                continue

            # === EMA CROSS ENGINE ===
            cross = run_cross_engine(sym, kl1)
            if cross:
                prev = state["last_cross"].get(sym, {})
                if not (prev.get("dir") == cross["dir"] and prev.get("bar_close_ms") == cross["bar_close_ms"]):
                    if cross["power"] >= POWER_NORMAL_MIN:
                        boost_tag = f" | Boost: +{ATR_BOOST_ADD:.0f}" if cross["atr_pct"] >= ATR_BOOST_PCT else ""
                        send_tg(
                            f"⚡ CROSS SIGNAL: {sym} ({INTERVAL_1H})\n"
                            f"Direction: {cross['dir']}\n"
                            f"RSI Divergence: {cross['div']}\n"
                            f"ATR({ATR_PERIOD}): {cross['atr']:.6f} ({cross['atr_pct']*100:.2f}%)" + boost_tag + "\n"
                            f"Power: {cross['power']:.1f}\n"
                            f"Time: {now_ist()}"
                        )
                        # günlük sayaç ↑
                        state["daily_counters"]["cross"] = state["daily_counters"].get("cross", 0) + 1
                    state["last_cross"][sym] = {"dir": cross["dir"], "bar_close_ms": cross["bar_close_ms"]}

            # === SCALP SLOPE ENGINE (trend uyumlu) ===
            scalp = run_scalp_engine(sym, kl1, kl4)
            if scalp:
                if state["last_slope_dir"].get(sym) != scalp["dir"]:
                    boost_tag = f" | ATR Boost +{ATR_BOOST_ADD:.0f}" if scalp["atr_pct"] >= ATR_BOOST_PCT else ""
                    send_tg(
                        f"🔥 SCALP SLOPE SIGNAL: {sym} ({INTERVAL_1H})\n"
                        f"Direction: {scalp['dir']} (4H trend uyumlu)\n"
                        f"RSI(14): {scalp['rsi']:.1f} | Power: {scalp['power']:.1f}{boost_tag}\n"
                        f"TP≈{scalp['tp']:.6f} | SL≈{scalp['sl']:.6f}\n"
                        f"Time: {now_ist()}"
                    )
                    # Açık pozisyona ekle (CSV + state)
                    with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([sym, scalp["dir"], scalp["price"], scalp["tp"], scalp["sl"],
                                                scalp["power"], scalp["rsi"], scalp["div"], now_ist()])
                    state["open_positions"].append({
                        "symbol": sym, "dir": scalp["dir"], "entry": scalp["price"],
                        "tp": scalp["tp"], "sl": scalp["sl"], "power": scalp["power"],
                        "rsi": scalp["rsi"], "div": scalp["div"], "open": now_ist(), "bar": bar
                    })
                    state["last_slope_dir"][sym] = scalp["dir"]
                    # günlük sayaç ↑
                    state["daily_counters"]["scalp"] = state["daily_counters"].get("scalp", 0) + 1

            time.sleep(SLEEP_BETWEEN)

        # === Açık scalp pozisyonlarını TP/SL için takip et ===
        still_open = []
        for t in state["open_positions"]:
            lp = get_last_price(t["symbol"])
            if lp is None:
                still_open.append(t); continue
            pnl = (lp - t["entry"])/t["entry"]*100 if t["dir"]=="UP" else (t["entry"] - lp)/t["entry"]*100
            bars_open = bar - t["bar"]

            hit_tp = (lp >= t["tp"]) if t["dir"]=="UP" else (lp <= t["tp"])
            hit_sl = (lp <= t["sl"]) if t["dir"]=="UP" else (lp >= t["sl"])
            if not (hit_tp or hit_sl):
                still_open.append(t); continue

            res = "TP" if hit_tp else "SL"
            send_tg(
                f"📘 {res} | {t['symbol']} {t['dir']}\n"
                f"Entry: {t['entry']:.6f}  Exit: {lp:.6f}\n"
                f"PnL: {pnl:.2f}%  Bars: {bars_open}\n"
                f"From: SCALP"
            )
            with open(CLOSED_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    t["symbol"], t["dir"], t["entry"], lp, res, pnl, bars_open,
                    t["power"], t["rsi"], t["div"], t["open"], now_ist()
                ])

        state["open_positions"] = still_open

        # Günlük sayaç/rapor yönetimi
        roll_daily_counters_if_needed(state)
        maybe_send_daily_summary(state)

        safe_save_json(STATE_FILE, state)
        log(f"Tarama bitti | Açık scalp pozisyon: {len(still_open)}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()