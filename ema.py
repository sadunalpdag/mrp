# === EMA ULTRA FINAL v9.5 ===
# Dual Mode: LIVE + SIM
# Binance Futures | EMA+ATR+RSI + 4H Trend Filter | Premium Scalp (TP 0.6% / SL 10%) | CSV Rapor
# Zaman damgaları: UTC+3 (Istanbul)

import os, json, csv, time, requests
from datetime import datetime, timedelta, timezone

# ========= MOD & GENEL AYARLAR =========
MODE = os.getenv("MODE", "LIVE").upper()  # "LIVE" veya "SIM"
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
LIMIT = 1000  # max klines çekimi
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# LIVE ayarları
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))  # saniye (5dk)
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.15"))
SCALP_TP_PCT = float(os.getenv("SCALP_TP_PCT", "0.006"))
SCALP_SL_PCT = float(os.getenv("SCALP_SL_PCT", "0.10"))
POWER_NORMAL_MIN = float(os.getenv("POWER_NORMAL_MIN", "60"))
POWER_PREMIUM_MIN = float(os.getenv("POWER_PREMIUM_MIN", "68"))

# SIM ayarları (tarih aralığı UTC)
SIM_START_DATE = os.getenv("SIM_START_DATE", "2025-07-01")
SIM_END_DATE   = os.getenv("SIM_END_DATE",   "2025-10-18")
# Eğer boşsa popüler birkaç sembolle çalışır; ENV ile list verilebilir: "BTCUSDT,ETHUSDT,SOLUSDT"
SIM_SYMBOLS = [s.strip() for s in os.getenv("SIM_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT").split(",")]

# Dosyalar
STATE_FILE = os.getenv("STATE_FILE", "alerts.json")
OPEN_CSV   = os.getenv("OPEN_CSV",   "open_positions.csv")
CLOSED_CSV = os.getenv("CLOSED_CSV", "closed_trades.csv")
SIM_CSV    = os.getenv("SIM_CSV",    "simulation_report.csv")
LOG_FILE   = os.getenv("LOG_FILE",   "log.txt")

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ========= ZAMAN & LOG =========
def now_ist():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0).isoformat()

def to_ist_iso(ts_ms: int):
    return (datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc) + timedelta(hours=3)).replace(microsecond=0).isoformat()

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_ist()} - {msg}\n")
    except:
        pass

def send_tg(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        log("[!] Telegram bilgileri eksik")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=12)
        log(f"[TG] {text.splitlines()[0]} ...")
    except Exception as e:
        log(f"Telegram hatası: {e}")

def send_doc(bytes_data: bytes, filename: str, caption: str = ""):
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

# ========= DOSYA YARDIMCILAR =========
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
            writer = csv.writer(f)
            writer.writerow(headers)

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

# ========= POWER (v2) =========
def power_v2(s_prev, s_now, atr_now, price, rsi_now):
    slope_comp = abs(s_now - s_prev) / (atr_now * 0.6) if atr_now > 0 else 0
    rsi_comp   = (rsi_now - 50) / 50
    atr_comp   = (atr_now / price) * 100 if price > 0 else 0
    # 55 taban + bileşen ağırlıkları
    return max(0, min(100, 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2))

# ========= BINANCE =========
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA-v9.5", "Accept": "application/json"})

def get_klines(symbol, interval, limit=LIMIT, start_ms=None, end_ms=None):
    """
    Tek çağrıda limit kadar çeker. startTime/endTime verilirse Binance filtre uygular.
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None: params["startTime"] = start_ms
    if end_ms   is not None: params["endTime"]   = end_ms
    for _ in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=12)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(0.3)
    return []

def get_klines_range(symbol, interval, start_ms, end_ms, chunk_limit=1000):
    """
    Belirli tarih aralığı için ardışık çağrılarla tüm kline'ları çeker.
    """
    out = []
    cursor = start_ms
    while True:
        batch = get_klines(symbol, interval, limit=chunk_limit, start_ms=cursor, end_ms=end_ms)
        if not batch:
            break
        out.extend(batch)
        # Son barın closeTime + 1 ms ile devam
        last_close = int(batch[-1][6])
        nxt = last_close + 1
        if nxt >= end_ms or len(batch) < chunk_limit:
            break
        cursor = nxt
        time.sleep(0.08)
    return out

def get_futures_symbols():
    try:
        r = SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=12)
        data = r.json()
        syms = [s["symbol"] for s in data["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
        return syms
    except:
        return []

def get_last_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        r = SESSION.get(url, timeout=6).json()
        return float(r["price"])
    except:
        return None

# ========= CSV Şemaları =========
ensure_csv(OPEN_CSV,   ["symbol","direction","entry","tp","sl","power","rsi","time_open"])
ensure_csv(CLOSED_CSV, ["symbol","direction","entry","exit","result","pnl","bars","power","rsi","time_open","time_close"])

# ========= LIVE MOD — NORMAL + PREMIUM =========
def live_loop():
    log("🚀 LIVE başlatıldı (Normal≥60 + Premium Scalp≥68, TP 0.6% / SL 10%)")
    state = safe_load_json(STATE_FILE)
    state.setdefault("last_slope_dir", {})
    state.setdefault("open_positions", [])

    symbols = get_futures_symbols()
    if not symbols:
        log("❌ Sembol listesi alınamadı.")
        return

    bar = 0
    while True:
        bar += 1
        for sym in symbols:
            kl1 = get_klines(sym, INTERVAL_1H, limit=500)
            kl4 = get_klines(sym, INTERVAL_4H, limit=300)
            if not kl1 or not kl4: 
                continue

            closes1 = [float(k[4]) for k in kl1]
            highs1  = [float(k[2]) for k in kl1]
            lows1   = [float(k[3]) for k in kl1]
            price   = closes1[-1]

            ema7_1  = ema(closes1, 7)
            ema25_1 = ema(closes1, 25)
            s_now = ema7_1[-1] - ema7_1[-4]
            s_prev= ema7_1[-2] - ema7_1[-5]
            atr1  = atr_series(highs1, lows1, closes1, ATR_PERIOD)[-1]
            rsi1  = rsi(closes1, RSI_PERIOD)[-1]

            # 4H trend yönü
            c4   = [float(k[4]) for k in kl4]
            ema7_4 = ema(c4, 7)
            ema25_4= ema(c4, 25)
            trend_4h = "UP" if ema7_4[-1] > ema25_4[-1] else "DOWN"

            slope_flip = None
            if s_prev < 0 and s_now > 0: slope_flip = "UP"
            if s_prev > 0 and s_now < 0: slope_flip = "DOWN"
            if not slope_flip:
                continue

            # duplicate engelle (aynı yönde üst üste)
            if state["last_slope_dir"].get(sym) == slope_flip:
                continue
            state["last_slope_dir"][sym] = slope_flip

            pwr = power_v2(s_prev, s_now, atr1, price, rsi1)
            tp = price * (1 + SCALP_TP_PCT if slope_flip == "UP" else 1 - SCALP_TP_PCT)
            sl = price * (1 - SCALP_SL_PCT if slope_flip == "UP" else 1 + SCALP_SL_PCT)

            # === NORMAL SİNYAL (Power ≥ 60) ===
            if pwr >= POWER_NORMAL_MIN:
                send_tg(
                    f"⚡ NORMAL SIGNAL: {sym}\n"
                    f"Direction: {slope_flip}\n"
                    f"Power: {pwr:.1f}  |  RSI(14): {rsi1:.1f}\n"
                    f"Slope: {s_prev:+.6f} → {s_now:+.6f}\n"
                    f"ATR({ATR_PERIOD}): {atr1:.6f}\n"
                    f"Time: {now_ist()}"
                )

            # === PREMIUM SCALP (Power ≥ 68 & 4H trend ile uyumlu) ===
            if pwr >= POWER_PREMIUM_MIN and slope_flip == trend_4h:
                send_tg(
                    f"🔥 PREMIUM SCALP {('LONG' if slope_flip=='UP' else 'SHORT')}: {sym}\n"
                    f"Trend(4H): {trend_4h} ✅\n"
                    f"Power(v2): {pwr:.1f} | RSI(14): {rsi1:.1f}\n"
                    f"TP≈{tp:.6f} | SL≈{sl:.6f}\n"
                    f"Time: {now_ist()}"
                )
                # Açık pozisyona ekle + CSV
                with open(OPEN_CSV, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([sym, slope_flip, price, tp, sl, pwr, rsi1, now_ist()])
                state["open_positions"].append({
                    "symbol": sym, "dir": slope_flip, "entry": price,
                    "tp": tp, "sl": sl, "power": pwr, "rsi": rsi1, "open": now_ist(), "bar": bar
                })

            time.sleep(SLEEP_BETWEEN)

        # === Açık pozisyonları TP/SL için kontrol et ===
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
                f"From: PREMIUM SCALP (Power={t['power']:.1f})"
            )
            with open(CLOSED_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    t["symbol"], t["dir"], t["entry"], lp, res, pnl, bars_open, t["power"], t["rsi"], t["open"], now_ist()
                ])

        state["open_positions"] = still_open
        safe_save_json(STATE_FILE, state)
        log(f"Tarama bitti. Açık pozisyon: {len(still_open)}")
        time.sleep(SCAN_INTERVAL)

# ========= SIM MOD — BACKTEST (PREMIUM SCALP) =========
def bar_hits_tp_sl(direction, entry, tp, sl, high, low):
    """
    1H bar içinde tetik sırası:
    LONG: önce high TP'ye dokunursa TP; önce low SL'ye dokunursa SL
    SHORT: önce low TP'ye dokunursa TP; önce high SL'ye dokunursa SL
    Intra-bar sıralamayı bilmediğimiz için konservatif sıra kullanılır:
    LONG: SL önce, sonra TP kontrol etmek daha temkinli olurdu; ama biz fırsat odaklı TP'yi öne alabiliriz.
    Burada standard: TP öncelikli (piyasa lehine spike'ı kaçırmamak için)
    """
    if direction == "UP":
        if high >= tp: return "TP"
        if low  <= sl: return "SL"
    else:
        if low  <= tp: return "TP"
        if high >= sl: return "SL"
    return None

def sim_power_v2(s_prev, s_now, atr_now, price, rsi_now):
    return power_v2(s_prev, s_now, atr_now, price, rsi_now)

def run_sim_for_symbol(symbol, start_ms, end_ms):
    """
    Premium scalp stratejisini 1H üzerinde; 4H trend filtresiyle simüle eder.
    Sinyal: EMA7 slope reversal + Power(v2) ≥ 68 + 4H trend ile uyum
    Çıkış: TP %0.6, SL %10 (bar high/low bazlı)
    """
    # 1H & 4H verileri al
    kl1 = get_klines_range(symbol, INTERVAL_1H, start_ms, end_ms, chunk_limit=1000)
    kl4 = get_klines_range(symbol, INTERVAL_4H, start_ms - 7*24*3600*1000, end_ms, chunk_limit=1000)  # 4H için biraz öncesinden
    if len(kl1) < 80 or len(kl4) < 50:
        return []

    closes1 = [float(k[4]) for k in kl1]
    highs1  = [float(k[2]) for k in kl1]
    lows1   = [float(k[3]) for k in kl1]
    times1  = [int(k[6]) for k in kl1]  # closeTime ms

    closes4 = [float(k[4]) for k in kl4]
    times4  = [int(k[6]) for k in kl4]

    ema7_1  = ema(closes1, 7)
    ema25_1 = ema(closes1, 25)
    atr1_all= atr_series(highs1, lows1, closes1, ATR_PERIOD)
    rsi1_all= rsi(closes1, RSI_PERIOD)

    ema7_4  = ema(closes4, 7)
    ema25_4 = ema(closes4, 25)

    # 4H trendi zamana göre bulmak için index pointer
    def trend_4h_dir_at(ts_ms):
        # ts_ms'den küçük/ eşit son 4h barını bul
        # binary search yerine linear geri kaydırma yeterli
        idx = None
        for i in range(len(times4)-1, -1, -1):
            if times4[i] <= ts_ms:
                idx = i; break
        if idx is None or idx < 25:  # yeterince geçmiş yoksa FLAT
            return "FLAT"
        return "UP" if ema7_4[idx] > ema25_4[idx] else "DOWN"

    results = []
    last_dir = None  # duplicate önleme
    i = 6  # slope hesapları için en az 6 bar
    while i < len(closes1):
        s_now  = ema7_1[i]   - ema7_1[i-3]  # 4 bar fark (i, i-3)
        s_prev = ema7_1[i-1] - ema7_1[i-4]
        slope_flip = None
        if s_prev < 0 and s_now > 0: slope_flip = "UP"
        if s_prev > 0 and s_now < 0: slope_flip = "DOWN"

        if slope_flip and slope_flip != last_dir:
            price = closes1[i]
            atr_now = atr1_all[i]
            rsi_now = rsi1_all[i] if rsi1_all[i] is not None else 50
            pwr = sim_power_v2(s_prev, s_now, atr_now, price, rsi_now)

            # 4H trend uyumu
            tdir = trend_4h_dir_at(times1[i])
            if pwr >= POWER_PREMIUM_MIN and slope_flip == tdir and atr_now > 0 and price > 0:
                # giriş & hedefler
                entry = price
                tp = entry * (1 + SCALP_TP_PCT if slope_flip=="UP" else 1 - SCALP_TP_PCT)
                sl = entry * (1 - SCALP_SL_PCT if slope_flip=="UP" else 1 + SCALP_SL_PCT)

                # ileri barlarda TP/SL testi
                outcome = None
                exit_price = entry
                bars_held = 0
                j = i+1
                while j < len(closes1):
                    high = float(kl1[j][2]); low = float(kl1[j][3])
                    outcome = bar_hits_tp_sl(slope_flip, entry, tp, sl, high, low)
                    bars_held += 1
                    if outcome is not None:
                        exit_price = tp if outcome=="TP" else sl
                        break
                    j += 1

                if outcome:
                    pnl = (exit_price - entry)/entry*100 if slope_flip=="UP" else (entry - exit_price)/entry*100
                    results.append({
                        "symbol": symbol,
                        "direction": slope_flip,
                        "entry": round(entry, 6),
                        "exit": round(exit_price, 6),
                        "result": outcome,
                        "pnl": round(pnl, 3),
                        "bars": bars_held,
                        "power": round(pwr, 1),
                        "rsi": round(rsi_now, 1),
                        "time_open": to_ist_iso(times1[i]),
                        "time_close": to_ist_iso(times1[i+bars_held]) if (i+bars_held) < len(times1) else to_ist_iso(times1[-1])
                    })
                    last_dir = slope_flip
                    # sinyal sonrası tekrar aynı yöne spam gelmesin diye i'yi outcome anına atabiliriz
                    i = j
                    continue
        i += 1

    return results

def sim_loop():
    log(f"🧪 SIM başlatıldı [{SIM_START_DATE} → {SIM_END_DATE}] Premium Scalp stratejisi")
    start_ms = int(datetime.fromisoformat(SIM_START_DATE + "T00:00:00+00:00").timestamp()*1000)
    end_ms   = int(datetime.fromisoformat(SIM_END_DATE   + "T23:59:59+00:00").timestamp()*1000)

    all_rows = []
    for sym in SIM_SYMBOLS:
        log(f"⏳ Simülasyon: {sym}")
        try:
            rows = run_sim_for_symbol(sym, start_ms, end_ms)
            all_rows.extend(rows)
            log(f"→ {sym}: {len(rows)} işlem")
        except Exception as e:
            log(f"{sym} simülasyon hatası: {e}")
        time.sleep(0.2)

    # CSV yaz
    if all_rows:
        with open(SIM_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["symbol","direction","entry","exit","result","pnl","bars","power","rsi","time_open","time_close"])
            w.writeheader()
            w.writerows(all_rows)

        # Özet
        wins = sum(1 for r in all_rows if r["result"]=="TP")
        loses= sum(1 for r in all_rows if r["result"]=="SL")
        avg_pnl = sum(r["pnl"] for r in all_rows) / len(all_rows)
        winrate = (wins / len(all_rows) * 100) if all_rows else 0.0
        caption = (
            f"📈 SIMULATION SUMMARY\n"
            f"Range: {SIM_START_DATE} → {SIM_END_DATE}\n"
            f"Symbols: {', '.join(SIM_SYMBOLS)}\n"
            f"Trades: {len(all_rows)} | Win: {wins} | Lose: {loses}\n"
            f"Winrate: {winrate:.1f}% | AvgPnL: {avg_pnl:.2f}%"
        )
        # Telegram'a gönder
        try:
            with open(SIM_CSV, "rb") as f:
                send_doc(f.read(), os.path.basename(SIM_CSV), caption)
        except:
            send_tg(caption)
        log("✅ Simülasyon tamamlandı ve rapor gönderildi.")
    else:
        send_tg(f"⚠️ SIMULATION: Sonuç bulunamadı. Tarih/sembol/koşulları gözden geçiriniz.")
        log("⚠️ Simülasyon verisi boş.")

# ========= MAIN =========
def main():
    if MODE == "SIM":
        sim_loop()
    else:
        live_loop()

if __name__ == "__main__":
    main()