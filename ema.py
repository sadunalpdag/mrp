# === EMA ULTRA FINAL v9.4 ===
# Dual Signal Engine (Normal + Premium Scalp)
# Binance Futures | EMA + ATR + RSI + Trend Filter | TP/SL + CSV Logging | Istanbul Time

import os, json, csv, time, requests
from datetime import datetime, timedelta, timezone

# ========= AYARLAR =========
LIMIT = 300
ATR_PERIOD = 14
RSI_PERIOD = 14
SCAN_INTERVAL = 300        # 5 dakika
SLEEP_BETWEEN = 0.15
SCALP_TP_PCT = 0.006
SCALP_SL_PCT = 0.10
POWER_NORMAL_MIN = 60
POWER_PREMIUM_MIN = 68
STATE_FILE = "alerts.json"
OPEN_CSV = "open_positions.csv"
CLOSED_CSV = "closed_trades.csv"
LOG_FILE = "log.txt"

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ========= YARDIMCI =========
def now_ist():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0).isoformat()

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{now_ist()} - {msg}\n")

def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[!] Telegram bilgileri eksik")
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
            return json.load(open(path, "r", encoding="utf-8"))
    except:
        pass
    return {}

def safe_save_json(path, data):
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)

def ensure_csv(path, headers):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

# ========= INDIKATOR =========
def ema(vals, length):
    k = 2 / (length + 1)
    e = [vals[0]]
    for i in range(1, len(vals)):
        e.append(vals[i] * k + e[-1] * (1 - k))
    return e

def atr_series(h, l, c, p=14):
    trs = []
    for i in range(len(h)):
        if i == 0:
            trs.append(h[i] - l[i])
        else:
            prev_c = c[i - 1]
            trs.append(max(h[i] - l[i], abs(h[i] - prev_c), abs(l[i] - prev_c)))
    if len(trs) < p:
        return [0]*len(trs)
    a = [sum(trs[:p]) / p]
    for i in range(p, len(trs)):
        a.append((a[-1]*(p-1)+trs[i]) / p)
    return [0]*(len(trs)-len(a)) + a

def rsi(vals, p=14):
    if len(vals) < p+1:
        return [50]*len(vals)
    deltas = [vals[i]-vals[i-1] for i in range(1, len(vals))]
    gains = [max(d,0) for d in deltas]
    losses = [abs(min(d,0)) for d in deltas]
    avg_gain = sum(gains[:p])/p
    avg_loss = sum(losses[:p])/p
    rsis = [50]*p
    for i in range(p, len(deltas)):
        avg_gain = (avg_gain*(p-1)+gains[i])/p
        avg_loss = (avg_loss*(p-1)+losses[i])/p
        rs = avg_gain/avg_loss if avg_loss != 0 else 0
        r = 100 - 100/(1+rs)
        rsis.append(r)
    return [50]*(len(vals)-len(rsis)) + rsis

# ========= BINANCE =========
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v9.4"})

def get_klines(sym, interval, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        r = SESSION.get(url, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except:
        return []

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
        data = r.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except:
        return []

def get_last_price(sym):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}"
        r = SESSION.get(url, timeout=5).json()
        return float(r["price"])
    except:
        return None

# ========= POWER HESABI =========
def power_v2(s_prev, s_now, atr_now, price, rsi_now):
    slope_comp = abs(s_now - s_prev) / (atr_now * 0.6) if atr_now > 0 else 0
    rsi_comp = (rsi_now - 50) / 50
    atr_comp = atr_now / price * 100
    return max(0, min(100, 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2))

# ========= CSV =========
ensure_csv(OPEN_CSV, ["symbol","direction","entry","tp","sl","power","rsi","time_open"])
ensure_csv(CLOSED_CSV, ["symbol","direction","entry","exit","result","pnl","bars","power","rsi","time_open","time_close"])

# ========= ANA =========
def main():
    log("🚀 v9.4 Başlatıldı (Dual Signal + TP/SL CSV Takip)")
    state = safe_load_json(STATE_FILE)
    state.setdefault("last_slope_dir", {})
    state.setdefault("open_positions", [])
    symbols = get_futures_symbols()
    bar = 0

    while True:
        bar += 1
        for sym in symbols:
            kl1 = get_klines(sym, "1h")
            kl4 = get_klines(sym, "4h")
            if not kl1 or not kl4: continue

            closes = [float(k[4]) for k in kl1]
            highs = [float(k[2]) for k in kl1]
            lows = [float(k[3]) for k in kl1]
            price = closes[-1]

            ema7, ema25 = ema(closes,7), ema(closes,25)
            s_now = ema7[-1]-ema7[-4]
            s_prev = ema7[-2]-ema7[-5]
            atr_now = atr_series(highs,lows,closes,ATR_PERIOD)[-1]
            rsi_now = rsi(closes,RSI_PERIOD)[-1]
            pwr = power_v2(s_prev,s_now,atr_now,price,rsi_now)

            # 4H trend
            ema7_4h = ema([float(k[4]) for k in kl4],7)
            ema25_4h = ema([float(k[4]) for k in kl4],25)
            trend_4h = "UP" if ema7_4h[-1]>ema25_4h[-1] else "DOWN"

            # slope reversal check
            slope_flip = None
            if s_prev < 0 and s_now > 0: slope_flip="UP"
            if s_prev > 0 and s_now < 0: slope_flip="DOWN"
            if not slope_flip: continue
            if state["last_slope_dir"].get(sym)==slope_flip: continue
            state["last_slope_dir"][sym]=slope_flip

            # TP/SL seviyeleri
            tp = price*(1+SCALP_TP_PCT if slope_flip=="UP" else 1-SCALP_TP_PCT)
            sl = price*(1-SCALP_SL_PCT if slope_flip=="UP" else 1+SCALP_SL_PCT)

            # === NORMAL SINYAL ===
            if pwr >= POWER_NORMAL_MIN:
                send_tg(
                    f"⚡ NORMAL SIGNAL: {sym}\n"
                    f"Direction: {slope_flip}\n"
                    f"RSI(14): {rsi_now:.1f}\n"
                    f"Power: {pwr:.1f}\n"
                    f"Slope: {s_prev:+.6f}→{s_now:+.6f}\n"
                    f"ATR: {atr_now:.6f}\n"
                    f"Time: {now_ist()}"
                )

            # === PREMIUM SCALP ===
            if pwr >= POWER_PREMIUM_MIN and slope_flip==trend_4h:
                send_tg(
                    f"🔥 PREMIUM SCALP {('LONG' if slope_flip=='UP' else 'SHORT')}: {sym}\n"
                    f"Trend(4H): {trend_4h} ✅\n"
                    f"Power(v2): {pwr:.1f}\n"
                    f"RSI(14): {rsi_now:.1f}\n"
                    f"TP≈{tp:.6f} | SL≈{sl:.6f}\n"
                    f"Time: {now_ist()}"
                )
                with open(OPEN_CSV,"a",newline="",encoding="utf-8") as f:
                    csv.writer(f).writerow([sym,slope_flip,price,tp,sl,pwr,rsi_now,now_ist()])
                state["open_positions"].append({
                    "symbol":sym,"dir":slope_flip,"entry":price,
                    "tp":tp,"sl":sl,"power":pwr,"rsi":rsi_now,"open":now_ist(),"bars":bar
                })

            time.sleep(SLEEP_BETWEEN)

        # === AÇIK İŞLEMLERİ KONTROL ET ===
        new_open=[]
        for t in state["open_positions"]:
            lp = get_last_price(t["symbol"])
            if not lp: continue
            pnl=(lp-t["entry"])/t["entry"]*100 if t["dir"]=="UP" else (t["entry"]-lp)/t["entry"]*100
            bars_open=bar-t["bars"]
            if (t["dir"]=="UP" and lp>=t["tp"]) or (t["dir"]=="DOWN" and lp<=t["tp"]):
                res="TP"
            elif (t["dir"]=="UP" and lp<=t["sl"]) or (t["dir"]=="DOWN" and lp>=t["sl"]):
                res="SL"
            else:
                new_open.append(t); continue

            send_tg(f"📘 {res} | {t['symbol']} {t['dir']}\nEntry: {t['entry']:.6f} Exit: {lp:.6f}\nPnL: {pnl:.2f}% | Bars: {bars_open}\nFrom: SCALP (Power={t['power']:.1f})")
            with open(CLOSED_CSV,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([t["symbol"],t["dir"],t["entry"],lp,res,pnl,bars_open,t["power"],t["rsi"],t["open"],now_ist()])
        state["open_positions"]=new_open
        safe_save_json(STATE_FILE,state)

        log(f"⏳ Tarama tamamlandı. Açık pozisyon: {len(new_open)} | Bar={bar}")
        time.sleep(SCAN_INTERVAL)

if __name__=="__main__":
    main()