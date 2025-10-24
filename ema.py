# ==============================================================
# 📘 EMA ULTRA v13.5 — Data Hybrid + Real Trade
#  - AutoTrade ON  => gerçek Binance Futures emirleri (MARKET)
#  - Simulate ON   => sadece veri toplar (limitsiz), emir açmaz
#  - MAX_BUY/MAX_SELL sadece AutoTrade modunda uygulanır
#  - open_positions.json her sinyali kaydeder (sim dahil)
#  - closed_trades.json TP/SL kapananları toplar
#  - Telegram komutlarıyla runtime param kontrolü (/set vb)
#  - Telegram offline olursa tg_queue.json'a bufferlar
#  - Günlük rapor /report ile veya otomatik
# ==============================================================

import os, json, time, math, requests, hmac, hashlib
from datetime import datetime, timezone, timedelta
import numpy as np

# ================= PATHS =================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE         = os.path.join(DATA_DIR, "state.json")
PARAM_FILE         = os.path.join(DATA_DIR, "params.json")
OPEN_POS_FILE      = os.path.join(DATA_DIR, "open_positions.json")
CLOSED_TRADES_FILE = os.path.join(DATA_DIR, "closed_trades.json")
TG_QUEUE_FILE      = os.path.join(DATA_DIR, "tg_queue.json")
LOG_FILE           = os.path.join(DATA_DIR, "log.txt")

# ================= ENV VARS =================
BOT_TOKEN  = os.getenv("BOT_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")

BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")

BINANCE_FAPI = "https://fapi.binance.com"

# ================= BASIC HELPERS =================
def safe_load(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return default

def safe_save(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[SAVE ERR] {e}", flush=True)

def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)

def now_iso():
    return now_ist_dt().isoformat()

def now_ts_ms():
    # Binance timestamp for signed req
    return int(datetime.now(timezone.utc).timestamp() * 1000)

# ================= TELEGRAM QUEUE SYSTEM =================
def _queue_append(entry):
    q = safe_load(TG_QUEUE_FILE, [])
    q.append(entry)
    safe_save(TG_QUEUE_FILE, q)

def tg_send(text: str):
    """Telegram'a direkt gönder, hata olursa kuyruğa koy."""
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] token/chat yok")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=10
        )
        log(f"[TG OK] {text[:80]}")
    except Exception as e:
        log(f"[TG ERR] {e}")
        _queue_append({"type":"text","text":text})

def tg_send_file(name: str, raw_bytes: bytes):
    """Telegram'a dosya gönder. Hata olursa kuyruğa yaz (latin1 encode)."""
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG FILE] token/chat yok")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
            data={"chat_id": CHAT_ID},
            files={"document": (name, raw_bytes)},
            timeout=20
        )
        log(f"[TG FILE OK] {name}")
    except Exception as e:
        log(f"[TG FILE ERR] {e}")
        _queue_append({
            "type":"file",
            "name":name,
            "data":raw_bytes.decode("latin1")
        })

def tg_flush_queue():
    """Bekleyen mesaj/dosyaları tekrar dene."""
    q = safe_load(TG_QUEUE_FILE, [])
    if not q:
        return
    new_q = []
    for item in q:
        t = item.get("type")
        try:
            if t == "text":
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    data={"chat_id": CHAT_ID, "text": item.get("text","")},
                    timeout=10
                )
            elif t == "file":
                nm  = item.get("name","file.bin")
                dat = item.get("data","").encode("latin1")
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                    data={"chat_id": CHAT_ID},
                    files={"document": (nm, dat)},
                    timeout=20
                )
            time.sleep(0.5)
        except Exception as e:
            log(f"[TG FLUSH KEEP] {e}")
            new_q.append(item)
    safe_save(TG_QUEUE_FILE, new_q)

# ================= BINANCE HELPERS =================
def _signed_request(method, path, payload):
    """Signed futures request."""
    if not BINANCE_KEY or not BINANCE_SECRET:
        raise RuntimeError("Binance API keys not set")

    query = "&".join([f"{k}={payload[k]}" for k in payload])
    sig = hmac.new(
        BINANCE_SECRET.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-MBX-APIKEY": BINANCE_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    url = BINANCE_FAPI + path + "?" + query + "&signature=" + sig
    if method == "POST":
        r = requests.post(url, headers=headers, timeout=10)
    elif method == "GET":
        r = requests.get(url, headers=headers, timeout=10)
    else:
        raise RuntimeError("Unsupported method for Binance signed req")

    if r.status_code != 200:
        raise RuntimeError(f"Binance HTTP {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(symbol):
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=5
        ).json()
        return float(r["price"])
    except:
        return None

def futures_exchange_info():
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/exchangeInfo",
            timeout=10
        ).json()
        return r.get("symbols", [])
    except:
        return []

def futures_get_klines(symbol, interval, limit):
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/klines",
            params={"symbol":symbol, "interval":interval, "limit":limit},
            timeout=10
        ).json()
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        # ileride kapanmamış future bar varsa çıkar
        if r and int(r[-1][6])>now_ms:
            r = r[:-1]
        return r
    except:
        return []

def futures_market_order(symbol, side, qty, positionSide):
    """
    Gerçek emir gönderir (AutoTrade ON durumunda).
    side: BUY / SELL
    positionSide: LONG / SHORT
    qty: coin miktarı (örnek: 0.1234 BTC)
    """
    payload = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "positionSide": positionSide,
        "timestamp": now_ts_ms()
    }
    return _signed_request("POST", "/fapi/v1/order", payload)

def calc_order_quantity(symbol, usdt_size):
    """
    USDT büyüklüğünden kabaca contract qty hesapla.
    qty = USDT / price
    round(.,4) yapıyoruz (basit). Ret gelirse Telegram'a error döner.
    """
    price = futures_get_price(symbol)
    if not price or price <= 0:
        return None
    qty = usdt_size / price
    return round(qty,4)

# ================= STATE / PARAM INIT =================
STATE = safe_load(STATE_FILE, {
    "open_positions": [],
    "last_cross_seen": {},
    "last_scalp_seen": {},
    "bar_index": 0,
    "auto_trade": False,   # gerçek emir kapalı default
    "simulate": True,      # sim açık default (data topluyor)
    "last_daily_sent_date": ""
})

DEFAULT_PARAM = {
    "POWER_NORMAL_MIN": 60.0,
    "POWER_PREMIUM_MIN": 68.0,
    "POWER_ULTRA_MIN": 75.0,

    "ATR_BOOST_PCT": 0.004,
    "ADX_BASE": 25.0,

    "SCALP_TP_PCT": 0.006,
    "SCALP_SL_PCT": 0.10,
    "CROSS_TP_PCT": 0.010,
    "CROSS_SL_PCT": 0.030,

    "SCALP_COOLDOWN_BARS": 3,

    "TRADE_SIZE_USDT": 250.0,
    "MAX_BUY": 15,
    "MAX_SELL": 15,

    # basit AI eşikleri (şu an kullanılmıyor ama yerleri sabit)
    "AI_PNL_THRESHOLD": 0.0,
    "AI_MIN_CONF": 0.0
}

PARAM = safe_load(PARAM_FILE, DEFAULT_PARAM)
# merge defaults (yeni alan eklendiyse doldur)
for k,v in DEFAULT_PARAM.items():
    PARAM.setdefault(k,v)
safe_save(PARAM_FILE, PARAM)

# eski sürümlerden kalan "params" snapshot varsa state->param merge et
if "params" in STATE and isinstance(STATE["params"], dict):
    for k,v in STATE["params"].items():
        if k in PARAM:
            PARAM[k] = v
safe_save(PARAM_FILE, PARAM)

def count_open_directions(open_positions, real_only=False):
    """
    real_only=True => sadece gerçek açılmış trade'leri say.
    sim/gerçek farkını anlamak için kaydederken "mode": "real"/"sim" tutacağız.
    """
    long_cnt = 0
    short_cnt= 0
    for p in open_positions:
        if real_only and p.get("mode")!="real":
            continue
        if p.get("dir")=="UP":
            long_cnt += 1
        elif p.get("dir")=="DOWN":
            short_cnt+= 1
    return long_cnt, short_cnt
# ================== INDICATORS ==================
def ema(vals, n):
    k = 2.0/(n+1.0)
    e = [vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+1:
        return [50]*len(vals)
    deltas = np.diff(vals)
    gains  = np.maximum(deltas,0)
    losses = -np.minimum(deltas,0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    out=[50]*period
    for i in range(period, len(deltas)):
        avg_gain=(avg_gain*(period-1)+gains[i])/period
        avg_loss=(avg_loss*(period-1)+losses[i])/period
        rs=(avg_gain/avg_loss) if avg_loss>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(highs,lows,closes,period=14):
    if len(highs)<2:
        return [0]*len(highs)
    trs=[]
    for i in range(len(highs)):
        if i==0:
            trs.append(highs[i]-lows[i])
        else:
            pc=closes[i-1]
            trs.append(max(highs[i]-lows[i], abs(highs[i]-pc), abs(lows[i]-pc)))
    if len(trs)<period:
        base=sum(trs)/len(trs)
        return [0]*(len(highs)-1)+[base]
    out=[sum(trs[:period])/period]
    for i in range(period,len(trs)):
        out.append((out[-1]*(period-1)+trs[i])/period)
    need=len(highs)-len(out)
    if need>0:
        out=[0]*need+out
    return out

def slope_angle_deg(slope, atr_val):
    if atr_val<=0: return 0.0
    return math.degrees(math.atan(slope/atr_val))

def angle_between_deg(s1,s2,atr_val):
    if atr_val<=0: return 0.0
    m1=s1/atr_val
    m2=s2/atr_val
    denom=1+m1*m2
    if abs(denom)<1e-9:
        return 90.0
    return math.degrees(math.atan(abs(m2-m1)/denom))

# POWER HELPERS
def calc_power(e7_now,e7_prev,e7_prev2, atr_now, price_now, rsi_now):
    slope_now  = e7_now  - e7_prev2
    slope_prev = e7_prev - e7_prev2
    slope_comp = abs(slope_now - slope_prev)/(atr_now*0.6) if atr_now>0 else 0
    rsi_comp   = (rsi_now-50)/50.0
    atr_comp   = (atr_now/price_now)*100 if price_now>0 else 0
    base = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    base = max(0.0, min(100.0, base))
    return base, slope_prev, slope_now

def tier_from_power(power):
    if power >= PARAM["POWER_ULTRA_MIN"]:   return "ULTRA","🟩"
    if power >= PARAM["POWER_PREMIUM_MIN"]: return "PREMIUM","🟦"
    if power >= PARAM["POWER_NORMAL_MIN"]:  return "NORMAL","🟨"
    return None,""

# ================== SIGNAL ENGINES ==================
def build_cross_signal(sym, kl1):
    closes=[float(k[4]) for k in kl1]
    ema7  = ema(closes,7)
    ema25 = ema(closes,25)
    if len(ema7)<6 or len(ema25)<6:
        return None

    prev_diff    = ema7[-3]-ema25[-3]
    cross_diff   = ema7[-2]-ema25[-2]
    confirm_diff = ema7[-1]-ema25[-1]

    direction=None
    if prev_diff<0 and cross_diff>0 and confirm_diff>0:
        direction="UP"
    elif prev_diff>0 and cross_diff<0 and confirm_diff<0:
        direction="DOWN"
    if not direction:
        return None

    highs=[float(k[2]) for k in kl1]
    lows =[float(k[3]) for k in kl1]
    atr_arr=atr_like(highs,lows,closes,14)
    atr_now=atr_arr[-1] if atr_arr else 0.0

    rsi_arr=rsi(closes,14)
    rsi_now=rsi_arr[-1]

    pwr, slope_prev, slope_now = calc_power(
        ema7[-1], ema7[-2], ema7[-5],
        atr_now, closes[-1], rsi_now
    )

    tier, color = tier_from_power(pwr)
    if tier is None:
        return None

    ang_now  = slope_angle_deg(slope_now, atr_now)
    ang_dif  = angle_between_deg(slope_prev, slope_now, atr_now)

    entry = closes[-1]
    if direction=="UP":
        tp = entry*(1+PARAM["CROSS_TP_PCT"])
        sl = entry*(1-PARAM["CROSS_SL_PCT"])
    else:
        tp = entry*(1-PARAM["CROSS_TP_PCT"])
        sl = entry*(1+PARAM["CROSS_SL_PCT"])

    return {
        "symbol": sym,
        "type": "CROSS",
        "dir": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "power": pwr,
        "rsi": rsi_now,
        "ang_now": ang_now,
        "ang_change": ang_dif,
        "tier": tier,
        "color": color,
        "time": now_iso()
    }

def build_scalp_signal(sym, kl1, last_scalp_seen, bar_index):
    closes=[float(k[4]) for k in kl1]
    ema7_1=ema(closes,7)
    if len(ema7_1)<6:
        return None

    slope_now  = ema7_1[-1]-ema7_1[-4]
    slope_prev = ema7_1[-2]-ema7_1[-5]

    if slope_prev<0 and slope_now>0:
        direction="UP"
    elif slope_prev>0 and slope_now<0:
        direction="DOWN"
    else:
        return None

    # cooldown kontrolü (sadece SCALP için)
    key=f"{sym}_{direction}"
    last_idx=last_scalp_seen.get(key)
    if last_idx is not None:
        if (bar_index - last_idx) <= PARAM["SCALP_COOLDOWN_BARS"]:
            return None

    highs=[float(k[2]) for k in kl1]
    lows =[float(k[3]) for k in kl1]
    atr_arr=atr_like(highs,lows,closes,14)
    atr_now=atr_arr[-1] if atr_arr else 0.0

    rsi_arr=rsi(closes,14)
    rsi_now=rsi_arr[-1]

    pwr, slope_prev2, slope_now2 = calc_power(
        ema7_1[-1], ema7_1[-2], ema7_1[-5],
        atr_now, closes[-1], rsi_now
    )

    tier, color = tier_from_power(pwr)
    if tier not in ("PREMIUM","ULTRA"):
        return None

    ang_now  = slope_angle_deg(slope_now2, atr_now)
    ang_dif  = angle_between_deg(slope_prev2, slope_now2, atr_now)

    entry = closes[-1]
    if direction=="UP":
        tp = entry*(1+PARAM["SCALP_TP_PCT"])
        sl = entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp = entry*(1-PARAM["SCALP_TP_PCT"])
        sl = entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol": sym,
        "type": "SCALP",
        "dir": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "power": pwr,
        "rsi": rsi_now,
        "ang_now": ang_now,
        "ang_change": ang_dif,
        "tier": tier,
        "color": color,
        "time": now_iso(),
        "cooldown_key": key
    }

# ================== POSITION MGMT ==================
def record_open(sig, mode_flag):
    """
    mode_flag: "real" ya da "sim"
    open_positions.json içine pushlar.
    """
    open_positions = safe_load(OPEN_POS_FILE, [])
    open_positions.append({
        "symbol": sig["symbol"],
        "type": sig["type"],
        "dir": sig["dir"],
        "entry": sig["entry"],
        "tp": sig["tp"],
        "sl": sig["sl"],
        "time_open": sig["time"],
        "power": sig["power"],
        "rsi": sig["rsi"],
        "ang_now": sig["ang_now"],
        "ang_change": sig["ang_change"],
        "mode": mode_flag  # "real" | "sim"
    })
    safe_save(OPEN_POS_FILE, open_positions)

def try_close_positions():
    """
    TP/SL tetiklenenleri kapatır.
    Kapatınca closed_trades.json'a taşır.
    Return True => en az 1 pozisyon kapandı (buna göre AutoTrade tekrar açabiliriz)
    """
    open_positions = safe_load(OPEN_POS_FILE, [])
    if not open_positions:
        return False

    closed_any = False
    still=[]
    closed_list=safe_load(CLOSED_TRADES_FILE, [])

    for pos in open_positions:
        sym    = pos["symbol"]
        cur_px = futures_get_price(sym)
        if cur_px is None:
            still.append(pos)
            continue

        hit_tp = (cur_px >= pos["tp"]) if pos["dir"]=="UP" else (cur_px <= pos["tp"])
        hit_sl = (cur_px <= pos["sl"]) if pos["dir"]=="UP" else (cur_px >= pos["sl"])

        if not (hit_tp or hit_sl):
            still.append(pos)
            continue

        result = "TP" if hit_tp else "SL"
        pnl_pct = (
            (cur_px-pos["entry"])/pos["entry"]*100
            if pos["dir"]=="UP" else
            (pos["entry"]-cur_px)/pos["entry"]*100
        )

        closed_list.append({
            "symbol": sym,
            "type": pos["type"],
            "dir": pos["dir"],
            "entry": pos["entry"],
            "exit": cur_px,
            "result": result,
            "pnl_pct": pnl_pct,
            "time_open": pos["time_open"],
            "time_close": now_iso(),
            "mode": pos.get("mode","?"),
            "power": pos.get("power"),
            "rsi": pos.get("rsi"),
            "ang_now": pos.get("ang_now"),
            "ang_change": pos.get("ang_change")
        })

        tg_send(
            f"📘 CLOSE {sym} {pos['type']} {pos['dir']} [{pos.get('mode','?')}]\n"
            f"Exit:{cur_px:.4f} {result} {pnl_pct:.2f}%"
        )
        closed_any = True

    if closed_any:
        safe_save(OPEN_POS_FILE, still)
        safe_save(CLOSED_TRADES_FILE, closed_list)

    return closed_any

# ================== AUTOTRADE / EXECUTION ==================
def enforce_limits_autotrade():
    """
    AutoTrade (gerçek emir) için limit kontrol:
    - sadece REAL pozisyonları sayıyoruz
    - eğer long_cnt >= MAX_BUY ya da short_cnt >= MAX_SELL ise
      artık yeni gerçek emir AÇMA.
    Ancak simülasyon her zaman devam eder.
    return True  => limit MÜSAİT (açılabilir)
    return False => limit DOLU (gerçek emir açma)
    """
    real_longs, real_shorts = count_open_directions(
        safe_load(OPEN_POS_FILE, []),
        real_only=True
    )
    if real_longs >= PARAM["MAX_BUY"] or real_shorts >= PARAM["MAX_SELL"]:
        # Sadece autotrade'i kapatıyoruz ama simulate açık kalabilir
        if STATE["auto_trade"]:
            STATE["auto_trade"] = False
            tg_send("⛔ Limit doldu → AutoTrade OFF. Sim devam ediyor.")
            safe_save(STATE_FILE, STATE)
        return False
    return True

def execute_signal(sig):
    """
    Sinyal geldiğinde ne yapıyoruz:
      - STATE["simulate"] True ise => sadece kayda al, sim trade mesajı
      - STATE["simulate"] False ve STATE["auto_trade"] True ise =>
            limit uygunsa gerçek emir dene
            gerçek emir başarılı/başarısız olsa da kayda al (mode="real")
      - STATE["simulate"] False ve STATE["auto_trade"] False ise =>
            hiçbir gerçek emir atma ama kayda al (mode="sim")
    """
    symbol = sig["symbol"]
    direction = sig["dir"]

    # 1) eğer simulate modu açık ise => her zaman sim trade
    if STATE["simulate"]:
        record_open(sig, "sim")
        tg_send(
            f"📒 SIM TRADE {symbol} {sig['type']} {direction}\n"
            f"entry={sig['entry']:.4f} tp={sig['tp']:.4f} sl={sig['sl']:.4f}"
        )
        return

    # 2) simulate kapalıysa ama auto_trade de kapalıysa => yine sadece sim kaydı
    if not STATE["auto_trade"]:
        record_open(sig, "sim")
        tg_send(
            f"📒 SIM TRADE {symbol} {sig['type']} {direction} [autotrade OFF]\n"
            f"entry={sig['entry']:.4f} tp={sig['tp']:.4f} sl={sig['sl']:.4f}"
        )
        return

    # 3) simulate kapalı, auto_trade açık => gerçek emir atmayı dene
    if not enforce_limits_autotrade():
        # limit dolu => fallback sim
        record_open(sig, "sim")
        tg_send(
            f"📒 SIM TRADE {symbol} {sig['type']} {direction} (limit dolu)\n"
            f"entry={sig['entry']:.4f} tp={sig['tp']:.4f} sl={sig['sl']:.4f}"
        )
        return

    # limit uygun → gerçek emir dene
    qty = calc_order_quantity(symbol, PARAM["TRADE_SIZE_USDT"])
    if qty is None or qty<=0:
        record_open(sig, "sim")
        tg_send(f"❌ qty hesaplanamadı. SIM kaydedildi {symbol}")
        return

    side = "BUY" if direction=="UP" else "SELL"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    try:
        resp = futures_market_order(symbol, side, qty, pos_side)
        tg_send(
            f"💸 REAL TRADE {symbol} {direction}\n"
            f"qty={qty} side={side} {pos_side}\n"
            f"entry≈{sig['entry']:.4f} tp={sig['tp']:.4f} sl={sig['sl']:.4f}"
        )
        log(f"[REAL TRADE OK] {resp}")
        record_open(sig, "real")
    except Exception as e:
        # emir atılamadı → sim olarak sakla
        record_open(sig, "sim")
        tg_send(f"❌ REAL TRADE ERR {symbol}\n{e}\nSim olarak kaydedildi.")
        log(f"[REAL TRADE ERR] {e}")

# ================== DAILY REPORT ==================
def build_daily_summary_payload():
    closed_list = safe_load(CLOSED_TRADES_FILE, [])
    total = len(closed_list)
    wins = [c for c in closed_list if c.get("result")=="TP"]
    winrate = (len(wins)/total*100.0) if total>0 else 0.0

    avg_pnl = 0.0
    if total>0:
        try:
            avg_pnl = sum(c.get("pnl_pct",0.0) for c in closed_list)/total
        except:
            avg_pnl = 0.0

    return {
        "total": total,
        "winrate": winrate,
        "avg_pnl": avg_pnl,
        "closed_list": closed_list
    }

def maybe_daily_summary():
    """
    Günlük otomatik rapor (her loop çağrılır ama günde 1 kere yollar)
    """
    today_str = now_ist_dt().strftime("%Y-%m-%d")
    if STATE.get("last_daily_sent_date","") == today_str:
        return

    summary = build_daily_summary_payload()

    tg_send(
        "📊 Günlük Rapor\n"
        f"Tarih: {today_str}\n"
        f"Toplam Trade: {summary['total']}\n"
        f"Winrate: {summary['winrate']:.1f}%\n"
        f"AvgPnL: {summary['avg_pnl']:.2f}%"
    )

    # closed trades dosyasını gönder
    raw_closed = json.dumps(summary["closed_list"], ensure_ascii=False, indent=2).encode("utf-8")
    tg_send_file(f"closed_trades_{today_str}.json", raw_closed)

    # state snapshot gönder
    raw_state = json.dumps(STATE, ensure_ascii=False, indent=2).encode("utf-8")
    tg_send_file(f"state_snapshot_{today_str}.json", raw_state)

    STATE["last_daily_sent_date"] = today_str
    safe_save(STATE_FILE, STATE)
# ================== TELEGRAM COMMANDS ==================
def tg_poll_commands(last_update_id):
    """
    Telegram'dan komut oku + cevapla.
    Komutlar:
      /status
      /params
      /set KEY VALUE    veya  /set KEY=VALUE
      /autotrade on/off
      /simulate on/off
      /queue
      /forceflush
      /report
    """
    if not BOT_TOKEN:
        return last_update_id

    try:
        r = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"timeout":1, "limit":20, "offset": last_update_id+1},
            timeout=5
        ).json()
    except Exception as e:
        log(f"[TG POLL ERR]{e}")
        return last_update_id

    if not r.get("ok"):
        return last_update_id

    for upd in r.get("result", []):
        uid = upd["update_id"]
        msg = upd.get("message", {})
        text = msg.get("text","").strip()
        if not text:
            last_update_id = uid
            continue

        lower = text.lower()

        # /status
        if lower == "/status":
            open_pos = safe_load(OPEN_POS_FILE, [])
            long_real, short_real = count_open_directions(open_pos, real_only=True)
            tg_send(
                "🤖 STATUS\n"
                f"AutoTrade: {'✅' if STATE['auto_trade'] else '❌'}\n"
                f"Simulate:  {'✅' if STATE['simulate'] else '❌'}\n"
                f"REAL Long: {long_real} / {PARAM['MAX_BUY']}\n"
                f"REAL Short:{short_real} / {PARAM['MAX_SELL']}\n"
                f"TradeSize: {PARAM['TRADE_SIZE_USDT']} USDT\n"
                f"OpenPos Total: {len(open_pos)} (sim+real)\n"
                f"Time: {now_iso()}"
            )

        # /params
        elif lower == "/params":
            pretty = []
            for k,v in PARAM.items():
                pretty.append(f"{k} = {v}")
            pretty_text = "\n".join(pretty)
            tg_send(
                "🔧 Aktif Parametreler:\n"
                f"{pretty_text}\n"
                f"\nAutoTrade: {'✅' if STATE['auto_trade'] else '❌'} | "
                f"Simulate: {'✅' if STATE['simulate'] else '❌'}"
            )

        # /autotrade on|off
        elif lower.startswith("/autotrade"):
            parts = lower.split()
            if len(parts)==2 and parts[1] in ("on","off"):
                if parts[1]=="on":
                    STATE["auto_trade"]=True
                    tg_send("🔓 AutoTrade ON (gerçek emir denenir)")
                else:
                    STATE["auto_trade"]=False
                    tg_send("🔒 AutoTrade OFF (gerçek emir yok)")
                safe_save(STATE_FILE, STATE)
            else:
                tg_send("kullanım: /autotrade on | /autotrade off")

        # /simulate on|off
        elif lower.startswith("/simulate"):
            parts = lower.split()
            if len(parts)==2 and parts[1] in ("on","off"):
                if parts[1]=="on":
                    STATE["simulate"]=True
                    tg_send("📝 Simulate ON (sadece veri topluyoruz)")
                else:
                    STATE["simulate"]=False
                    tg_send("📝 Simulate OFF (sim kapalı)")
                safe_save(STATE_FILE, STATE)
            else:
                tg_send("kullanım: /simulate on | /simulate off")

        # /queue
        elif lower == "/queue":
            q = safe_load(TG_QUEUE_FILE, [])
            tg_send(f"📨 Queue length: {len(q)}")

        # /forceflush
        elif lower == "/forceflush":
            tg_flush_queue()
            tg_send("📤 Queue flush denendi.")

        # /report (manuel günlük rapor)
        elif lower == "/report":
            summary = build_daily_summary_payload()
            tg_send(
                "📊 Rapor (manual)\n"
                f"Toplam Trade: {summary['total']}\n"
                f"Winrate: {summary['winrate']:.1f}%\n"
                f"AvgPnL: {summary['avg_pnl']:.2f}%"
            )
            raw_closed = json.dumps(summary["closed_list"], ensure_ascii=False, indent=2).encode("utf-8")
            tg_send_file("closed_trades_manual.json", raw_closed)

        # /set KEY VALUE  veya  /set KEY=VALUE
        elif lower.startswith("/set"):
            raw = text.replace("\n"," ").strip()

            # format 1: "/set KEY VALUE"
            # format 2: "/set KEY=VALUE"
            if " " in raw and "=" not in raw:
                parts = raw.split()
                if len(parts) != 3:
                    tg_send("kullanım:\n/set KEY VALUE\nya da\n/set KEY=VALUE")
                    last_update_id = uid
                    continue
                key_txt = parts[1]
                val_txt = parts[2]
            else:
                # dene "/set KEY=VALUE"
                try:
                    after = raw.split(" ",1)[1]  # "/set " sonrası
                except:
                    tg_send("kullanım:\n/set KEY VALUE\nya da\n/set KEY=VALUE")
                    last_update_id = uid
                    continue
                if "=" not in after:
                    tg_send("kullanım:\n/set KEY VALUE\nya da\n/set KEY=VALUE")
                    last_update_id = uid
                    continue
                key_txt, val_txt = after.split("=",1)

            key_txt = key_txt.strip().upper()
            val_txt = val_txt.strip()

            # alias
            if key_txt in ("SIZE","TRADE_SIZE","TRADE_SIZE_USDT"):
                key_txt = "TRADE_SIZE_USDT"
            if key_txt in ("MAXBUY","MAX_BUY"):
                key_txt = "MAX_BUY"
            if key_txt in ("MAXSELL","MAX_SELL"):
                key_txt = "MAX_SELL"

            if key_txt not in PARAM:
                tg_send(f"❌ bilinmeyen param: {key_txt}")
                last_update_id = uid
                continue

            # parse number
            try:
                if key_txt in ("MAX_BUY","MAX_SELL"):
                    new_val = int(val_txt)
                else:
                    new_val = float(val_txt)
            except:
                tg_send(f"❌ {key_txt} değeri sayı olmalı (gönderdiğin: {val_txt})")
                last_update_id = uid
                continue

            PARAM[key_txt] = new_val
            safe_save(PARAM_FILE, PARAM)
            STATE["params"] = PARAM
            safe_save(STATE_FILE, STATE)

            tg_send(f"⚙️ {key_txt} = {PARAM[key_txt]} olarak ayarlandı")

            # Eğer MAX_BUY/MAX_SELL düşürüldüyse ve şu an zaten üzerindeysek,
            # sadece AutoTrade'i kapat (sim devam eder).
            if key_txt in ("MAX_BUY","MAX_SELL"):
                real_l, real_s = count_open_directions(
                    safe_load(OPEN_POS_FILE, []),
                    real_only=True
                )
                if real_l >= PARAM["MAX_BUY"] or real_s >= PARAM["MAX_SELL"]:
                    STATE["auto_trade"] = False
                    safe_save(STATE_FILE, STATE)
                    tg_send("⛔ Yeni limit aşıldı → AutoTrade OFF (Sim açık kalır)")

        last_update_id = uid

    return last_update_id

# ================== MAIN LOOP ==================
def main():
    tg_send("🚀 EMA ULTRA v13.5 başladı (Hybrid Sim/Real + Runtime Params)")
    last_update_id = 0

    # sembolleri topla
    exinfo = futures_exchange_info()
    symbols = [s["symbol"] for s in exinfo if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    symbols.sort()

    while True:
        # flush bekleyen telegram mesaj/dosyaları
        tg_flush_queue()

        # telegram komutlarını işle
        last_update_id = tg_poll_commands(last_update_id)

        # açık pozisyonlarda TP/SL oldu mu? olduysa kapat
        closed_any = try_close_positions()
        if closed_any:
            # pozisyonlar kapandıysa tekrar auto_trade açmak ister miyiz?
            # hayır, burada otomatik açmıyoruz.
            # kullanıcı isterse /autotrade on der.
            tg_send("ℹ️ Pozisyon kapandı (en az biri). /autotrade on ile gerçek modu tekrar açabilirsin.")

        # sinyal tara
        STATE["bar_index"] += 1
        bar_i = STATE["bar_index"]

        for sym in symbols:
            kl1 = futures_get_klines(sym, "1h", 200)
            if len(kl1) < 120:
                continue

            # CROSS
            cross_sig = build_cross_signal(sym, kl1)
            if cross_sig:
                cross_key = f"{sym}_{cross_sig['dir']}"
                # tekrar aynı/time spam engeli
                if STATE["last_cross_seen"].get(cross_key) != cross_sig["time"]:
                    tg_send(
                        f"{cross_sig['color']} {cross_sig['tier']} CROSS {sym} {cross_sig['dir']}\n"
                        f"Pow:{cross_sig['power']:.1f} RSI:{cross_sig['rsi']:.1f}\n"
                        f"Açı:{cross_sig['ang_now']:+.1f}° Δ:{cross_sig['ang_change']:.1f}°"
                    )
                    execute_signal(cross_sig)
                    STATE["last_cross_seen"][cross_key] = cross_sig["time"]

            # SCALP
            scalp_sig = build_scalp_signal(sym, kl1, STATE["last_scalp_seen"], bar_i)
            if scalp_sig:
                scalp_key = scalp_sig["cooldown_key"]
                if STATE["last_scalp_seen"].get(scalp_key) != bar_i:
                    tg_send(
                        f"{scalp_sig['color']} {scalp_sig['tier']} SCALP {sym} {scalp_sig['dir']}\n"
                        f"Pow:{scalp_sig['power']:.1f} RSI:{scalp_sig['rsi']:.1f}\n"
                        f"Açı:{scalp_sig['ang_now']:+.1f}° Δ:{scalp_sig['ang_change']:.1f}°"
                    )
                    execute_signal(scalp_sig)
                    STATE["last_scalp_seen"][scalp_key] = bar_i

            time.sleep(0.08)

        # günlük rapor (günde 1 kere)
        maybe_daily_summary()

        # state ve param kalıcı yaz
        STATE["params"] = PARAM
        safe_save(STATE_FILE, STATE)
        safe_save(PARAM_FILE, PARAM)

        # ana döngü bekleme
        time.sleep(120)

# ================== ENTRY ==================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        tg_send(f"❗FATAL: {e}")
        _queue_append({"type":"text","text":f"FATAL: {e}"})
