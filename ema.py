# === EMA ULTRA FINAL v11 ===
# Cross (v9.2 tarzı, Power ≥ 60) + Scalp Slope (4H trend uyumlu, Power ≥ 68, TP/SL)
# RSI Divergence + ATR Boost | Günlük Özet + Auto-Optimize (v10)
# 30 Günlük Öğrenme (Saatlik & Coin Analitiği) + Aylık Rapor (v11)
# Zaman: UTC+3 (Istanbul)

import os, json, csv, time, requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict

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

# ATR Boost
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

# Otomatik Optimizasyon (v10)
OPTIMIZE_APPLY_ON_START = True
OPTIMIZE_MIN_WINRATE = 60.0
OPTIMIZE_MIN_TRADES = 5
NEXT_PARAMS_FILE = "next_day_params.json"

# Rapor dosyaları
STATE_FILE = "alerts.json"
OPEN_CSV   = "open_positions.csv"
CLOSED_CSV = "closed_trades.csv"
LOG_FILE   = "log.txt"
REPORTS_DIR = "reports"
MONTHLY_TRADES = os.path.join(REPORTS_DIR, "monthly_trades.csv")
LEARNED_HOURS  = os.path.join(REPORTS_DIR, "learned_hours.json")
BEST_COINS_JSON= os.path.join(REPORTS_DIR, "best_coins_monthly.json")

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ========= ZAMAN / LOG =========
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)

def now_ist():
    return now_ist_dt().isoformat()

def today_ist_date():
    return now_ist_dt().strftime("%Y-%m-%d")

def now_ist_hhmm():
    return now_ist_dt().strftime("%H:%M")

def month_key_ist(dt=None):
    d = now_ist_dt() if dt is None else dt
    return d.strftime("%Y-%m")  # "2025-10"

def is_month_end():
    # İstanbul'a göre ay sonu: yarın ay değişiyorsa bugün son gün
    today = now_ist_dt().date()
    tomorrow = today + timedelta(days=1)
    return today.month != tomorrow.month

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
def ensure_dir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except:
        pass

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

def load_json(path, default=None):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
    except:
        pass
    return {} if default is None else default

def save_json(path, data):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def ensure_csv(path, headers):
    ensure_dir(os.path.dirname(path) or ".")
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

ensure_csv(OPEN_CSV,   ["symbol","direction","entry","tp","sl","power","rsi","divergence","time_open"])
ensure_csv(CLOSED_CSV, ["symbol","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close"])
ensure_csv(MONTHLY_TRADES, ["symbol","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close","month_key"])

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
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v11", "Accept": "application/json"})

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
    ema99  = ema(closes, 99)  # bilgi amaçlı

    # Sadece son barda kesişim
    prev_diff = ema7[-2] - ema25[-2]
    curr_diff = ema7[-1] - ema25[-1]
    cross_dir = None
    if prev_diff < 0 and curr_diff > 0: cross_dir = "UP"
    if prev_diff > 0 and curr_diff < 0: cross_dir = "DOWN"
    if not cross_dir: 
        return None

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

    s_now  = ema7_1[-1] - ema7_1[-4]
    s_prev = ema7_1[-2] - ema7_1[-5]
    slope_flip = None
    if s_prev < 0 and s_now > 0: slope_flip = "UP"
    if s_prev > 0 and s_now < 0: slope_flip = "DOWN"
    if not slope_flip:
        return None

    closes4 = [float(k[4]) for k in kl4]
    ema7_4  = ema(closes4, 7)
    ema25_4 = ema(closes4, 25)
    trend_4h = "UP" if ema7_4[-1] > ema25_4[-1] else "DOWN"
    if slope_flip != trend_4h:
        return None  # trend uyumsuzsa gösterme

    price   = closes1[-1]
    atr_now = atr_series(highs1, lows1, closes1, ATR_PERIOD)[-1]
    rsi_now = rsi(closes1, RSI_PERIOD)[-1]
    pwr     = power_v2(s_prev, s_now, atr_now, price, rsi_now)
    boost, atr_pct = atr_boost(atr_now, price)
    pwr += boost

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

# ========= GÜNLÜK ÖZET / OPTİMİZASYON / ANALİTİK =========
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

def read_closed_trades_for(date_str):
    rows = []
    try:
        with open(CLOSED_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("time_close","").startswith(date_str):
                    rows.append(row)
    except FileNotFoundError:
        pass
    return rows

def compute_day_stats(rows):
    out = {"total":0,"wins":0,"loses":0,"avg_pnl":0.0,"winrate":0.0,
           "tp_bars_avg":None,"sl_bars_avg":None,"hourly":{},"fastest_tp_bars":None}
    if not rows: return out
    pnl_sum = 0.0
    tp_b, sl_b = [], []
    for row in rows:
        out["total"] += 1
        res = row.get("result")
        pnl = float(row.get("pnl","0") or 0.0)
        bars = int(row.get("bars","0") or 0)
        pnl_sum += pnl
        if res == "TP":
            out["wins"] += 1
            tp_b.append(bars)
            if out["fastest_tp_bars"] is None or bars < out["fastest_tp_bars"]:
                out["fastest_tp_bars"] = bars
        elif res == "SL":
            out["loses"] += 1
            sl_b.append(bars)
        hh = (row.get("time_open","")[11:13] or "??")
        d = out["hourly"].setdefault(hh, {"trades":0,"wins":0,"loses":0})
        d["trades"] += 1
        if res == "TP": d["wins"] += 1
        elif res == "SL": d["loses"] += 1
    out["avg_pnl"] = pnl_sum / out["total"]
    out["winrate"] = (out["wins"] / out["total"] * 100.0) if out["total"] else 0.0
    if tp_b: out["tp_bars_avg"] = sum(tp_b)/len(tp_b)
    if sl_b: out["sl_bars_avg"] = sum(sl_b)/len(sl_b)
    return out

def recommend_params(rows, min_winrate=OPTIMIZE_MIN_WINRATE, min_trades=OPTIMIZE_MIN_TRADES):
    total = len(rows)
    if total < min_trades:
        return {}
    stats = compute_day_stats(rows)
    best_hour, best_hour_win = None, -1
    for hh, d in stats["hourly"].items():
        if d["trades"] >= 3:
            wr = (d["wins"]/d["trades"]*100.0) if d["trades"] else 0.0
            if wr > best_hour_win:
                best_hour_win, best_hour = wr, hh
    tp_opts   = [0.004, 0.006, 0.008]
    boost_opts= [0.003, 0.004, 0.005]
    pow_opts  = [66, 68, 70]
    def score(winrate, avg_tp_bars, tp_pct, pow_min, boost_pct):
        if winrate < min_winrate or avg_tp_bars is None: return -1e9
        return (winrate*2.0) + (100.0/max(1.0, avg_tp_bars)) - (tp_pct*1000*0.3) + (max(0,(70-pow_min))*0.5) - (boost_pct*1000*0.2)
    best = None
    for tpv in tp_opts:
        for bv in boost_opts:
            for pv in pow_opts:
                sc = score(stats["winrate"], stats["tp_bars_avg"], tpv, pv, bv)
                cand = {"SCALP_TP_PCT": tpv, "ATR_BOOST_PCT": bv, "POWER_PREMIUM_MIN": pv, "score": sc}
                if best is None or sc > best["score"]:
                    best = cand
    rec = {
        "SCALP_TP_PCT": best["SCALP_TP_PCT"] if best else 0.006,
        "SCALP_SL_PCT": SCALP_SL_PCT,
        "POWER_PREMIUM_MIN": best["POWER_PREMIUM_MIN"] if best else POWER_PREMIUM_MIN,
        "POWER_NORMAL_MIN": POWER_NORMAL_MIN,
        "ATR_BOOST_PCT": best["ATR_BOOST_PCT"] if best else ATR_BOOST_PCT,
        "best_hour": f"{best_hour}:00-{(int(best_hour)+1)%24:02d}:00" if best_hour is not None else None,
        "winrate": stats["winrate"],
        "avg_tp_bars": stats["tp_bars_avg"],
    }
    return rec

def make_and_send_strategy_report(date_str, rows):
    ensure_dir(REPORTS_DIR)
    csv_path = os.path.join(REPORTS_DIR, f"strategy_stats_{date_str}.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        try:
            with open(csv_path, "rb") as f:
                send_doc(f.read(), os.path.basename(csv_path), f"📎 Günlük strateji verisi ({date_str})")
        except:
            send_tg(f"📎 Günlük strateji verisi hazır: {csv_path}")

def append_monthly_trade(row):
    # row: dict (closed_trades ile aynı alanlar + month_key)
    ensure_csv(MONTHLY_TRADES, ["symbol","direction","entry","exit","result","pnl","bars","power","rsi","divergence","time_open","time_close","month_key"])
    with open(MONTHLY_TRADES, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row["symbol"], row["direction"], row["entry"], row["exit"], row["result"], row["pnl"],
            row["bars"], row["power"], row["rsi"], row["divergence"], row["time_open"], row["time_close"],
            month_key_ist()
        ])

def load_last_30_days_rows():
    cutoff = now_ist_dt() - timedelta(days=30)
    rows = []
    try:
        with open(MONTHLY_TRADES, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # time_close ISO+03:00
                tc = row.get("time_close","")
                if not tc:
                    continue
                try:
                    dt = datetime.fromisoformat(tc)
                except:
                    continue
                # dt is naive? assume already +03:00 iso; best-effort
                if dt >= cutoff:
                    rows.append(row)
    except FileNotFoundError:
        pass
    return rows

def compute_hourly_learning(last30_rows):
    # saat → trades, wins, loses, avg_pnl, avg_tp_bars
    hourly = {}
    for row in last30_rows:
        hh = (row.get("time_open","")[11:13] or "??")
        d = hourly.setdefault(hh, {"trades":0,"wins":0,"loses":0,"pnl_sum":0.0,"tp_bars_sum":0,"tp_bars_cnt":0})
        d["trades"] += 1
        res = row.get("result")
        pnl = float(row.get("pnl","0") or 0.0)
        bars = int(row.get("bars","0") or 0)
        d["pnl_sum"] += pnl
        if res == "TP":
            d["wins"] += 1
            d["tp_bars_sum"] += bars
            d["tp_bars_cnt"] += 1
        elif res == "SL":
            d["loses"] += 1
    # finalize
    out = {}
    for hh, d in hourly.items():
        avg_pnl = d["pnl_sum"]/d["trades"] if d["trades"] else 0.0
        winrate = (d["wins"]/d["trades"]*100.0) if d["trades"] else 0.0
        tp_bars_avg = (d["tp_bars_sum"]/d["tp_bars_cnt"]) if d["tp_bars_cnt"] else None
        out[hh] = {"trades":d["trades"], "winrate":round(winrate,1), "avg_pnl":round(avg_pnl,3),
                   "tp_bars_avg": (round(tp_bars_avg,2) if tp_bars_avg is not None else None)}
    # en iyi/zayıf saatleri seç
    best_hours = []
    weak_hours = []
    for hh, d in out.items():
        if d["trades"] >= 5 and d["winrate"] >= 70 and (d["tp_bars_avg"] is None or d["tp_bars_avg"] <= 5):
            best_hours.append(f"{hh}:00-{(int(hh)+1)%24:02d}:00")
        if d["trades"] >= 5 and d["winrate"] <= 55:
            weak_hours.append(f"{hh}:00-{(int(hh)+1)%24:02d}:00")
    payload = {"updated_at": now_ist(), "hours": out, "best_hours": best_hours, "weak_hours": weak_hours}
    save_json(LEARNED_HOURS, payload)
    return payload

def compute_best_coins(last30_rows):
    # coin → stats
    coins = {}
    for row in last30_rows:
        sym = row.get("symbol","?")
        d = coins.setdefault(sym, {"trades":0,"wins":0,"loses":0,"pnl_sum":0.0})
        d["trades"] += 1
        if row.get("result") == "TP": d["wins"] += 1
        elif row.get("result") == "SL": d["loses"] += 1
        d["pnl_sum"] += float(row.get("pnl","0") or 0.0)
    ranked = []
    for sym, d in coins.items():
        wr = (d["wins"]/d["trades"]*100.0) if d["trades"] else 0.0
        avg = d["pnl_sum"]/d["trades"] if d["trades"] else 0.0
        ranked.append((sym, d["trades"], wr, avg))
    ranked.sort(key=lambda x: (x[2], x[3], x[1]), reverse=True)
    best = [sym for sym, n, wr, avg in ranked if n>=10 and wr>=70 and avg>=0.4][:10]  # kriter
    payload = {"updated_at": now_ist(), "best_coins": best, "ranking": [
        {"symbol":sym,"trades":n,"winrate":round(wr,1),"avg_pnl":round(avg,3)} for sym,n,wr,avg in ranked
    ]}
    save_json(BEST_COINS_JSON, payload)
    return payload

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

        # Günlük raw veriyi TG'ye belge olarak da gönder
        rows = read_closed_trades_for(today)
        make_and_send_strategy_report(today, rows)

        # Optimize öneri üret ve kaydet
        rec = recommend_params(rows, OPTIMIZE_MIN_WINRATE, OPTIMIZE_MIN_TRADES)
        if rec:
            payload = {
                "date": today,
                "recommended": {
                    "SCALP_TP_PCT": rec["SCALP_TP_PCT"],
                    "SCALP_SL_PCT": rec["SCALP_SL_PCT"],
                    "POWER_PREMIUM_MIN": rec["POWER_PREMIUM_MIN"],
                    "POWER_NORMAL_MIN": rec["POWER_NORMAL_MIN"],
                    "ATR_BOOST_PCT": rec["ATR_BOOST_PCT"]
                },
                "insights": {
                    "winrate": rec["winrate"],
                    "avg_tp_bars": rec["avg_tp_bars"],
                    "best_hour_window": rec["best_hour"]
                }
            }
            save_json(NEXT_PARAMS_FILE, payload["recommended"])
            pretty = "\n".join([
                f"• SCALP_TP_PCT = {rec['SCALP_TP_PCT']}",
                f"• SCALP_SL_PCT = {rec['SCALP_SL_PCT']}",
                f"• POWER_PREMIUM_MIN = {rec['POWER_PREMIUM_MIN']}",
                f"• POWER_NORMAL_MIN = {rec['POWER_NORMAL_MIN']}",
                f"• ATR_BOOST_PCT = {rec['ATR_BOOST_PCT']}",
                f"• Best Hour = {rec['best_hour'] or '-'}",
                f"• Winrate = {rec['winrate']:.1f}% | Avg TP Bars = {rec['avg_tp_bars'] or '-'}",
            ])
            send_tg("🧪 Optimize Öneri (yarın için):\n" + pretty)
        else:
            send_tg("ℹ️ Optimize öneri için bugün yeterli veri yok.")

        # v11: 30 günlük öğrenme & aylık rapor
        last30 = load_last_30_days_rows()
        hrs = compute_hourly_learning(last30)
        coins = compute_best_coins(last30)
        send_tg(
            "🧠 30 Günlük Öğrenme Güncellendi\n"
            f"En iyi saatler: {', '.join(hrs.get('best_hours', [])) or '-'}\n"
            f"En iyi coinler: {', '.join(coins.get('best_coins', [])) or '-'}"
        )
        if is_month_end():
            # Aylık özet
            mk = month_key_ist()
            total = len([r for r in last30 if r.get('time_close','').startswith(mk)])
            send_tg(f"📅 Aylık Özet ({mk}) hazırlandı.\nToplam kapanan işlem: {total}")

        state["last_daily_summary_date"] = today

# ========= ANA DÖNGÜ (CROSS + SCALP + TP/SL) =========
def main():
    log("🚀 v11 Başladı (Cross≥60 + Scalp≥68 | Optimize + 30g Öğrenme)")
    state = ensure_state(safe_load_json(STATE_FILE))

    # v10: açılışta optimize parametreleri uygula
    if OPTIMIZE_APPLY_ON_START:
        params = load_json(NEXT_PARAMS_FILE, {})
        applied = []
        def setf(name, val):
            globals()[name] = float(val)
            applied.append(f"{name}={val}")
        if "SCALP_TP_PCT" in params: setf("SCALP_TP_PCT", params["SCALP_TP_PCT"])
        if "SCALP_SL_PCT" in params: setf("SCALP_SL_PCT", params["SCALP_SL_PCT"])
        if "POWER_PREMIUM_MIN" in params: setf("POWER_PREMIUM_MIN", params["POWER_PREMIUM_MIN"])
        if "POWER_NORMAL_MIN" in params: setf("POWER_NORMAL_MIN", params["POWER_NORMAL_MIN"])
        if "ATR_BOOST_PCT" in params: setf("ATR_BOOST_PCT", params["ATR_BOOST_PCT"])
        if applied:
            send_tg("🧠 Optimize parametreler yüklendi:\n" + "\n".join("• " + x for x in applied))
            log("[OPT] applied at start: " + ", ".join(applied))

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
                    with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([sym, scalp["dir"], scalp["price"], scalp["tp"], scalp["sl"],
                                                scalp["power"], scalp["rsi"], scalp["div"], now_ist()])
                    state["open_positions"].append({
                        "symbol": sym, "dir": scalp["dir"], "entry": scalp["price"],
                        "tp": scalp["tp"], "sl": scalp["sl"], "power": scalp["power"],
                        "rsi": scalp["rsi"], "div": scalp["div"], "open": now_ist(), "bar": bar
                    })
                    state["last_slope_dir"][sym] = scalp["dir"]
                    state["daily_counters"]["scalp"] = state["daily_counters"].get("scalp", 0) + 1

            time.sleep(SLEEP_BETWEEN)

        # === Açık scalp pozisyonlarını TP/SL için takip et & kapananları kaydet ===
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
            row = [t["symbol"], t["dir"], t["entry"], lp, res, pnl, bars_open,
                   t["power"], t["rsi"], t["div"], t["open"], now_ist()]
            with open(CLOSED_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            # monthly dataset'e de pushla
            append_monthly_trade({
                "symbol": t["symbol"], "direction": t["dir"], "entry": t["entry"], "exit": lp,
                "result": res, "pnl": pnl, "bars": bars_open, "power": t["power"], "rsi": t["rsi"],
                "divergence": t["div"], "time_open": t["open"], "time_close": now_ist()
            })

        state["open_positions"] = still_open

        # Günlük sayaç/rapor yönetimi + öğrenme
        roll_daily_counters_if_needed(state)
        maybe_send_daily_summary(state)

        safe_save_json(STATE_FILE, state)
        log(f"Tarama bitti | Açık scalp pozisyon: {len(still_open)}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()