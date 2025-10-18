# === EMA ULTRA FINAL v9.2 ===
# Binance only | Smart Scalp (Power≥68, TP 0.6%) | Anti-Duplicate | Bars since last reversal

import os, time, json, io, requests, csv
from datetime import datetime, timezone

# ========= AYARLAR =========
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]
ATR_PERIOD = 14
SCAN_INTERVAL = 300      # 5 dk
SLEEP_BETWEEN = 0.2

# --- SIMULASYON ---
SIM_ENABLE = True
SCALP_MIN_POWER = 68
SCALP_TP_PCT = 0.006     # %0.6
SIM_SL_PCT = 0.10        # %10
REPORT_INTERVAL_MIN = 60

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
STATE_FILE = "alerts.json"
LOG_FILE = "log.txt"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA/2.1"})


# ========= UTILS =========
def nowiso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {msg}\n")


def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except:
        pass


def send_doc(b, fname, cap=""):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        requests.post(url, files={'document': (fname, b)}, data={'chat_id': CHAT_ID, 'caption': cap}, timeout=20)
    except:
        pass


def safe_load(p):
    try:
        if os.path.exists(p):
            return json.load(open(p, "r", encoding="utf-8"))
    except:
        pass
    return {}


def safe_save(p, d):
    tmp = p + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, p)


# ========= INDIKATORLER =========
def ema(v, l):
    k = 2 / (l + 1)
    e = [v[0]]
    for i in range(1, len(v)):
        e.append(v[i] * k + e[-1] * (1 - k))
    return e


def atr_series(h, lw, c, p=14):
    trs = []
    for i in range(len(h)):
        if i == 0:
            trs.append(h[i] - lw[i])
        else:
            pc = c[i - 1]
            trs.append(max(h[i] - lw[i], abs(h[i] - pc), abs(lw[i] - pc)))
    if len(trs) < p:
        return [0] * len(trs)
    a = [sum(trs[:p]) / p]
    for i in range(p, len(trs)):
        a.append((a[-1] * (p - 1) + trs[i]) / p)
    return [0] * (len(trs) - len(a)) + a


# ========= BINANCE =========
def get_klines(sym, intv, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": sym, "interval": intv, "limit": limit}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except:
        return []


def get_syms():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        d = r.json()
        return [s["symbol"] for s in d["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    except:
        return []


# ========= STATE / SIM =========
def ensure_state(st):
    st.setdefault("positions", {})
    st.setdefault("history", [])
    st.setdefault("last_report_ts", 0)
    st.setdefault("last_slope_dir", {})  # 🔸 yeni ekleme
    st.setdefault("last_reversal_bar", {})  # 🔸 bar zamanı kaydı
    return st


def open_pos(st, sym, side, price, src, power, s_prev, s_now):
    st["positions"][sym] = {
        "is_open": True,
        "side": side,
        "entry": price,
        "tp_pct": SCALP_TP_PCT,
        "sl_pct": SIM_SL_PCT,
        "source": src,
        "power": power,
        "opened_at": nowiso(),
        "bars": 0,
        "s_prev": s_prev,
        "s_now": s_now
    }


def check_close(st, sym, price):
    p = st["positions"].get(sym)
    if not p or not p.get("is_open"):
        return

    side = p["side"]
    e = p["entry"]
    tp = e * (1 + (p["tp_pct"] if side == "LONG" else -p["tp_pct"]))
    sl = e * (1 + (-p["sl_pct"] if side == "LONG" else p["sl_pct"]))
    hit_tp = (price >= tp if side == "LONG" else price <= tp)
    hit_sl = (price <= sl if side == "LONG" else price >= sl)

    if hit_tp or hit_sl:
        res = "TP" if hit_tp else "SL"
        pnl = p["tp_pct"] if hit_tp else -p["sl_pct"]
        st["positions"][sym]["is_open"] = False
        st["history"].append({
            "symbol": sym,
            "side": side,
            "entry": round(e, 6),
            "exit": round(price, 6),
            "pnl_pct": round(pnl * 100, 2),
            "outcome": res,
            "bars": p["bars"],
            "source": p["source"],
            "power": p["power"],
            "slope_prev": p["s_prev"],
            "slope_now": p["s_now"],
            "slope_change": p["s_now"] - p["s_prev"],
            "closed_at": nowiso()
        })
        send_tg(f"📘 SIM | {res} | {sym} {side}\nPnL: {pnl*100:.2f}% Bars: {p['bars']} From: {p['source']} Power={p['power']}")
        safe_save(STATE_FILE, st)


def maybe_report(st):
    if not st["history"]:
        return
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if now_ts - st["last_report_ts"] < REPORT_INTERVAL_MIN * 60:
        return
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(st["history"][0].keys()))
    w.writeheader()
    w.writerows(st["history"])
    send_doc(buf.getvalue().encode(), "scalp_report.csv", f"📊 SIM Raporu ({len(st['history'])} işlem)")
    st["last_report_ts"] = now_ts
    safe_save(STATE_FILE, st)


# ========= SCALP DETECT =========
def detect_slope_rev(ema7):
    if len(ema7) < 6:
        return None, (0, 0)
    s_now = ema7[-1] - ema7[-4]
    s_prev = ema7[-2] - ema7[-5]
    if s_prev < 0 and s_now > 0:
        return "UP", (s_prev, s_now)
    if s_prev > 0 and s_now < 0:
        return "DOWN", (s_prev, s_now)
    return None, (s_prev, s_now)


# ========= MAIN =========
def process(sym, st, bar_idx):
    for intv in INTERVALS:
        kl = get_klines(sym, intv)
        if not kl or len(kl) < 100:
            continue
        closes = [float(k[4]) for k in kl]
        highs = [float(k[2]) for k in kl]
        lows = [float(k[3]) for k in kl]

        ema7 = ema(closes, 7)
        slope_flip, (s_prev, s_now) = detect_slope_rev(ema7)
        price = closes[-1]

        if intv == "1h" and slope_flip:
            # 🔸 anti-duplicate
            if st["last_slope_dir"].get(sym) == slope_flip:
                continue

            atr = atr_series(highs, lows, closes, ATR_PERIOD)
            atr_now = atr[-1]
            scalp_power = max(0, min(100, 60 + abs(s_now - s_prev) / (atr_now * 0.6) * 20))
            if scalp_power < SCALP_MIN_POWER:
                continue

            # 🔸 kaç bar sonra geldi?
            last_rev_bar = st["last_reversal_bar"].get(sym, bar_idx)
            bars_since = bar_idx - last_rev_bar if bar_idx > last_rev_bar else 0
            st["last_reversal_bar"][sym] = bar_idx

            # sinyal
            tp = price * (1 + 0.006 if slope_flip == "UP" else 1 - 0.006)
            sl = price * (1 - 0.10 if slope_flip == "UP" else 1 + 0.10)
            send_tg(
                f"💥 SCALP {('LONG' if slope_flip == 'UP' else 'SHORT')} TRIGGER: {sym}\n"
                f"Slope: {s_prev:+.6f} → {s_now:+.6f}\n"
                f"Bars since last reversal: {bars_since}\n"
                f"TP≈{tp:.6f} | SL≈{sl:.6f}\n"
                f"Power={scalp_power:.1f}\nTime: {nowiso()}"
            )
            open_pos(st, sym, "LONG" if slope_flip == "UP" else "SHORT", price, "SCALP", scalp_power, s_prev, s_now)
            st["last_slope_dir"][sym] = slope_flip
            safe_save(STATE_FILE, st)

        check_close(st, sym, price)
        if sym in st["positions"] and st["positions"][sym].get("is_open"):
            st["positions"][sym]["bars"] += 1


def main():
    log("🚀 v9.2 başlatıldı (Smart Scalp Anti-Duplicate + Bars since reversal)")
    st = ensure_state(safe_load(STATE_FILE))
    syms = get_syms()
    bar_idx = 0
    while True:
        bar_idx += 1
        for s in syms:
            process(s, st, bar_idx)
        maybe_report(st)
        log(f"⏳ 5dk bekleniyor... (bar {bar_idx})")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()