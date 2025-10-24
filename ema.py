# ==============================================================
# 📘 EMA ULTRA v13.4 — Real Orders + Smart AutoSwitch + Queue Backup
#  - Gerçek Binance Futures emirleri (MARKET)
#  - AutoTrade <-> Sim mode otomatik geçiş
#  - Pozisyon limiti: MAX_BUY / MAX_SELL
#  - Trade size USDT bazlı
#  - Sinyaller: CROSS / SCALP (SCALP cooldown)
#  - open_positions.json & closed_trades.json kalıcı tutulur
#  - Telegram offline olsa bile tg_queue.json içinde saklanır
# ==============================================================

import os, json, time, math, requests, hmac, hashlib
from datetime import datetime, timezone, timedelta
import numpy as np

# ================= PATHS =================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE        = os.path.join(DATA_DIR, "state.json")
PARAM_FILE        = os.path.join(DATA_DIR, "params.json")
OPEN_POS_FILE     = os.path.join(DATA_DIR, "open_positions.json")
CLOSED_TRADES_FILE= os.path.join(DATA_DIR, "closed_trades.json")
TG_QUEUE_FILE     = os.path.join(DATA_DIR, "tg_queue.json")
LOG_FILE          = os.path.join(DATA_DIR, "log.txt")

# ================= ENV VARS =================
BOT_TOKEN  = os.getenv("BOT_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")

BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")

BINANCE_FAPI = "https://fapi.binance.com"

# ================= SAFE IO HELPERS =================
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

# ================= TELEGRAM QUEUE SYSTEM =================
def _queue_append(entry):
    q = safe_load(TG_QUEUE_FILE, [])
    q.append(entry)
    safe_save(TG_QUEUE_FILE, q)

def tg_send(text: str):
    """Gönderemezse kuyruğa yazar."""
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] token/chat eksik")
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
    """Gönderemezse kuyruğa yazar (latin1 encode ile saklıyoruz)."""
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG FILE] token/chat eksik")
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
    """Telegram bağlantısı gelir gelmez sıradaki kuyruğu boşaltmayı dener."""
    q = safe_load(TG_QUEUE_FILE, [])
    if not q:
        return
    new_q = []
    for item in q:
        t = item.get("type")
        try:
            if t == "text":
                tg_send(item.get("text",""))
            elif t == "file":
                nm  = item.get("name","file.bin")
                dat = item.get("data","").encode("latin1")
                tg_send_file(nm, dat)
            time.sleep(0.5)
        except Exception as e:
            log(f"[TG FLUSH RETAIN] {e}")
            new_q.append(item)
    safe_save(TG_QUEUE_FILE, new_q)

# ================= TIME HELPERS =================
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)

def now_iso():
    return now_ist_dt().isoformat()

def now_ts_ms():
    # Binance signature timestamp
    return int(datetime.now(timezone.utc).timestamp() * 1000)

# ================= STATE / PARAM INIT =================
STATE = safe_load(STATE_FILE, {
    "open_positions": [],
    "last_cross_seen": {},
    "last_scalp_seen": {},
    "auto_trade": False,
    "simulate": True,
    "bar_index": 0,
    "last_daily_sent_date": "",
    # eğer trade limiti dolunca autotrade kapattıysak burada hatırlar
})

# varsayılan paramlar
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

    # gelişmiş sinyal filtreleri (ileride kullanılabilir)
    "AI_PNL_THRESHOLD": 0.0,
    "AI_MIN_CONF": 0.0
}

PARAM = safe_load(PARAM_FILE, DEFAULT_PARAM)
# merge defaults in case new fields added
for k,v in DEFAULT_PARAM.items():
    PARAM.setdefault(k,v)
safe_save(PARAM_FILE, PARAM)

# STATE içindeki eski param snapshot varsa birleştir (geriye dönük uyum)
if "params" in STATE and isinstance(STATE["params"], dict):
    for k,v in STATE["params"].items():
        if k in PARAM:
            PARAM[k] = v
safe_save(PARAM_FILE, PARAM)

# =============== BINANCE FUTURES AUTH HELPERS ===============
def _signed_request(method, path, payload):
    """Low-level Binance futures private endpoint call."""
    if not BINANCE_KEY or not BINANCE_SECRET:
        raise RuntimeError("No Binance API keys in environment.")

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
        # gelecek barı at
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        if r and int(r[-1][6])>now_ms:
            r = r[:-1]
        return r
    except:
        return []

def futures_market_order(symbol, side, qty, positionSide):
    """
    Gerçek emir gönder. side: BUY/SELL, positionSide: LONG/SHORT
    qty: contract quantity (örn BTC miktarı).
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
    usdt_size'lik pozisyon açmak için yaklaşık kontrat miktarı hesapla.
    Basit versiyon: qty = usdt_size / mark_price
    Bu quantity Binance min step'ine uymuyorsa Binance reject edebilir,
    reject ederse yakalayıp raporlayacağız.
    """
    price = futures_get_price(symbol)
    if not price or price<=0:
        return None
    qty = usdt_size / price
    # çok kaba round, daha sonra lotSize filtresiyle iyileştirebiliriz
    return round(qty, 4)

def count_open_directions(open_positions):
    """
    Kaç aktif long / short var? (UP => long, DOWN => short)
    """
    long_cnt = sum(1 for p in open_positions if p.get("dir")=="UP")
    short_cnt= sum(1 for p in open_positions if p.get("dir")=="DOWN")
    return long_cnt, short_cnt
# ================= INDICATORS =================
def ema(vals, n):
    k = 2/(n+1)
    e = [vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+1: return [50]*len(vals)
    deltas = np.diff(vals)
    gains  = np.maximum(deltas,0)
    losses = -np.minimum(deltas,0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    out=[50]*period
    for i in range(period,len(deltas)):
        avg_gain=(avg_gain*(period-1)+gains[i])/period
        avg_loss=(avg_loss*(period-1)+losses[i])/period
        rs = (avg_gain/avg_loss) if avg_loss>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def adx_like_atr(highs,lows,closes,period=14):
    """
    Bizim power hesaplarımızda ATR benzeri volatilite metriği gibi kullandığımız adx_series
    benzeri çıktı istiyoruz (kısaltılmış).
    Döndürdüğümüz sadece "atr-ish" değeri olacak.
    """
    if len(highs)<2: return [0]*len(highs)
    trs=[]
    for i in range(len(highs)):
        if i==0:
            trs.append(highs[i]-lows[i])
        else:
            pc=closes[i-1]
            trs.append(max(highs[i]-lows[i],abs(highs[i]-pc),abs(lows[i]-pc)))
    if len(trs)<period:
        base = sum(trs)/len(trs)
        return [0]*(len(highs)-1)+[base]
    atr=[sum(trs[:period])/period]
    for i in range(period,len(trs)):
        atr.append((atr[-1]*(period-1)+trs[i])/period)
    need=len(highs)-len(atr)
    if need>0:
        atr = [0]*need+atr
    return atr

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

# ================= POWER / FILTER HELPERS =================
def power_score(e7_now,e7_prev,e7_prev2, atr_now, price_now, rsi_now):
    # momentum farkı
    slope_now  = e7_now  - e7_prev2
    slope_prev = e7_prev - e7_prev2
    slope_comp = abs(slope_now - slope_prev) / (atr_now*0.6) if atr_now>0 else 0
    rsi_comp   = (rsi_now-50)/50.0
    atr_comp   = (atr_now/price_now)*100 if price_now>0 else 0
    base = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    return max(0.0, min(100.0, base)), slope_prev, slope_now

def tier_from_power(power, p=PARAM):
    if power >= p["POWER_ULTRA_MIN"]:   return "ULTRA","🟩"
    if power >= p["POWER_PREMIUM_MIN"]: return "PREMIUM","🟦"
    if power >= p["POWER_NORMAL_MIN"]:  return "NORMAL","🟨"
    return None,""

# ================== SIGNAL ENGINES ==================
def build_cross_signal(sym, kl1):
    closes=[float(k[4]) for k in kl1]
    ema7  = ema(closes,7)
    ema25 = ema(closes,25)
    if len(ema7)<6 or len(ema25)<6:
        return None

    # 1-bar confirm cross
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
    atr_arr=adx_like_atr(highs,lows,closes,14)
    atr_now=atr_arr[-1] if atr_arr else 0.0

    rsi_arr=rsi(closes,14)
    rsi_now=rsi_arr[-1]

    pwr, slope_prev, slope_now = power_score(
        ema7[-1], ema7[-2], ema7[-5],
        atr_now,
        closes[-1],
        rsi_now
    )

    # açı bilgisi (rapor için)
    ang_now   = slope_angle_deg(slope_now, atr_now)
    ang_chng  = angle_between_deg(slope_prev, slope_now, atr_now)

    tier, color = tier_from_power(pwr)
    if tier is None:
        return None

    entry=closes[-1]
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
        "ang_change": ang_chng,
        "tier": tier,
        "color": color,
        "time": now_iso()
    }

def build_scalp_signal(sym, kl1, last_scalp_seen, bar_index):
    """
    SCALP sinyali: slope reversal + trend devamlılığı gibi.
    cooldown: aynı yönde SCALP sinyali için SCALP_COOLDOWN_BARS bekleniyor.
    """
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

    # cooldown kontrol
    key=f"{sym}_{direction}"
    last_idx=last_scalp_seen.get(key)
    if last_idx is not None:
        if (bar_index - last_idx) <= PARAM["SCALP_COOLDOWN_BARS"]:
            return None

    highs=[float(k[2]) for k in kl1]
    lows =[float(k[3]) for k in kl1]
    atr_arr=adx_like_atr(highs,lows,closes,14)
    atr_now=atr_arr[-1] if atr_arr else 0.0

    rsi_arr=rsi(closes,14)
    rsi_now=rsi_arr[-1]

    pwr, slope_prev2, slope_now2 = power_score(
        ema7_1[-1], ema7_1[-2], ema7_1[-5],
        atr_now, closes[-1], rsi_now
    )

    # scalp için premium altı sinyal istemiyoruz
    tier, color = tier_from_power(pwr)
    if tier not in ("PREMIUM","ULTRA"):
        return None

    ang_now   = slope_angle_deg(slope_now2, atr_now)
    ang_chng  = angle_between_deg(slope_prev2, slope_now2, atr_now)

    entry=closes[-1]
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
        "ang_change": ang_chng,
        "tier": tier,
        "color": color,
        "time": now_iso(),
        "cooldown_key": key
    }

# ================= TRADE LIFECYCLE =================
def record_open_position(sig):
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
    })
    safe_save(OPEN_POS_FILE, open_positions)

def try_close_positions():
    """
    TP/SL vurulmuş pozisyonları kapatır.
    AutoTrade açıksa gerçek kapatma emri göndermek isteyebilirsin,
    burada basit versiyonda 'piyasa kapandı' varsayımı yapıyoruz.
    """
    open_positions = safe_load(OPEN_POS_FILE, [])
    if not open_positions:
        return False  # kimse kapanmadı

    changed=False
    still=[]
    closed_list=safe_load(CLOSED_TRADES_FILE, [])

    for pos in open_positions:
        sym     = pos["symbol"]
        cur_px  = futures_get_price(sym)
        if cur_px is None:
            still.append(pos)
            continue

        hit_tp = (cur_px >= pos["tp"]) if pos["dir"]=="UP" else (cur_px <= pos["tp"])
        hit_sl = (cur_px <= pos["sl"]) if pos["dir"]=="UP" else (cur_px >= pos["sl"])

        if not (hit_tp or hit_sl):
            still.append(pos)
            continue

        # kapandı
        result = "TP" if hit_tp else "SL"
        pnl_pct = (cur_px-pos["entry"])/pos["entry"]*100 if pos["dir"]=="UP" else (pos["entry"]-cur_px)/pos["entry"]*100

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
            "power": pos.get("power"),
            "rsi": pos.get("rsi"),
            "ang_now": pos.get("ang_now"),
            "ang_change": pos.get("ang_change"),
        })

        tg_send(
            f"📘 CLOSE {sym} {pos['type']} {pos['dir']}\n"
            f"Exit:{cur_px:.4f} {result} {pnl_pct:.2f}%"
        )
        changed=True

    if changed:
        safe_save(OPEN_POS_FILE, still)
        safe_save(CLOSED_TRADES_FILE, closed_list)

    return changed

# ================= AUTOTRADE / LIMIT LOGIC =================
def enforce_mode_after_close():
    """
    Eğer trade kapandıysa, AutoTrade geri açılabilir.
    Kural:
    - bir şey kapandıysa → AutoTrade True, Sim False
    """
    STATE["auto_trade"] = True
    STATE["simulate"]   = False
    tg_send("✅ Pozisyon kapandı -> AutoTrade tekrar AKTIF, Sim OFF")
    safe_save(STATE_FILE, STATE)

def enforce_limits_before_open():
    """
    trade açmadan ÖNCE limitleri kontrol et.
    eğer limit doluysa:
    - AutoTrade False
    - Sim True
    - telegram bildir
    return True = limit OK, işlem açılabilir
    return False = limit dolu (trade açma)
    """
    open_positions = safe_load(OPEN_POS_FILE, [])
    long_cnt, short_cnt = count_open_directions(open_positions)

    # eşiğe ulaştıysak AutoTrade kapat, sim aç
    if long_cnt >= PARAM["MAX_BUY"] or short_cnt >= PARAM["MAX_SELL"]:
        if STATE["auto_trade"]:
            STATE["auto_trade"] = False
            STATE["simulate"]   = True
            tg_send("⛔ Limit doldu -> AutoTrade OFF, Sim ON. Yeni gerçek emir açılmayacak.")
            safe_save(STATE_FILE, STATE)
        return False

    return True

def execute_trade_if_allowed(sig):
    """
    Bu fonksiyon:
    1) Limitleri kontrol eder
    2) AutoTrade/Sim durumuna göre
       - gerçek emir yollar
       - ya da sadece sim log / kayıt yapar
    3) open_positions.json içine yazar
    """
    # pozisyon yönünü kontrol edelim
    dirn = sig["dir"]  # "UP" -> long, "DOWN" -> short

    # önce limit uygunsa devam
    if not enforce_limits_before_open():
        # limit dolu, gerçek emir açmıyoruz ama sim modunda kayıt tutabiliriz
        if STATE["simulate"]:
            record_open_position(sig)
            tg_send(f"📒 SIM ONLY | {sig['symbol']} {sig['type']} {dirn} (limit dolu)")
        return

    # buraya geldiysek limit OK.
    if STATE["auto_trade"] and not STATE["simulate"]:
        # gerçek emir modundayız
        qty = calc_order_quantity(sig["symbol"], PARAM["TRADE_SIZE_USDT"])
        if qty is None or qty<=0:
            tg_send(f"❌ Trade SKIP {sig['symbol']} qty hesaplanamadı.")
        else:
            side = "BUY" if dirn=="UP" else "SELL"
            positionSide = "LONG" if dirn=="UP" else "SHORT"
            try:
                resp = futures_market_order(sig["symbol"], side, qty, positionSide)
                tg_send(
                    f"💸 REAL TRADE {sig['symbol']} {dirn}\n"
                    f"qty={qty} side={side} pos={positionSide}\n"
                    f"entry≈{sig['entry']:.4f} tp={sig['tp']:.4f} sl={sig['sl']:.4f}"
                )
                log(f"[REAL TRADE OK] {resp}")
            except Exception as e:
                tg_send(f"❌ REAL TRADE ERR {sig['symbol']} {e}")
                log(f"[REAL TRADE ERR] {e}")
                # hata olsa bile yine de kaydı sim olarak tutabiliriz
        # kaydı open_positions'a yaz
        record_open_position(sig)
    else:
        # sim mode
        record_open_position(sig)
        tg_send(
            f"📒 SIM TRADE {sig['symbol']} {sig['type']} {dirn}\n"
            f"entry={sig['entry']:.4f} tp={sig['tp']:.4f} sl={sig['sl']:.4f}"
        )
# ================= TELEGRAM COMMAND POLLING =================
def tg_poll_commands(last_update_id):
    """
    Telegram'dan komutları çeker.
    last_update_id: en son işlenen update_id (geri dönüyoruz)
    return: new_last_update_id
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

        if lower == "/status":
            long_cnt, short_cnt = count_open_directions(safe_load(OPEN_POS_FILE, []))
            tg_send(
                "🤖 STATUS:\n"
                f"AutoTrade: {'✅' if STATE['auto_trade'] else '❌'}\n"
                f"SimMode: {'✅' if STATE['simulate'] else '❌'}\n"
                f"Open Long: {long_cnt} / {PARAM['MAX_BUY']}\n"
                f"Open Short:{short_cnt} / {PARAM['MAX_SELL']}\n"
                f"TradeSize: {PARAM['TRADE_SIZE_USDT']} USDT\n"
                f"Time: {now_iso()}"
            )

        elif lower.startswith("/autotrade"):
            # /autotrade on  | off
            parts = lower.split()
            if len(parts)==2 and parts[1] in ("on","off"):
                if parts[1]=="on":
                    STATE["auto_trade"]=True
                    # auto trade açıyorsak sim'i kapatıyoruz
                    STATE["simulate"]=False
                    tg_send("🔓 AutoTrade ON / Sim OFF")
                else:
                    STATE["auto_trade"]=False
                    STATE["simulate"]=True
                    tg_send("🔒 AutoTrade OFF / Sim ON")
                safe_save(STATE_FILE, STATE)
            else:
                tg_send("kullanım: /autotrade on | /autotrade off")

        elif lower.startswith("/simulate"):
            # /simulate on | off
            parts = lower.split()
            if len(parts)==2 and parts[1] in ("on","off"):
                if parts[1]=="on":
                    STATE["simulate"]=True
                    tg_send("📝 Sim Mode ON (gerçek emir yok)")
                else:
                    STATE["simulate"]=False
                    tg_send("📝 Sim Mode OFF")
                safe_save(STATE_FILE, STATE)
            else:
                tg_send("kullanım: /simulate on | /simulate off")

        elif lower.startswith("/set"):
            # /set PARAM VALUE
            parts = text.split()
            if len(parts)==3:
                key = parts[1].upper()
                val = parts[2]
                # özel alias'lar
                if key in ("SIZE","TRADE_SIZE","TRADE_SIZE_USDT"):
                    key="TRADE_SIZE_USDT"
                if key in ("MAXBUY","MAX_BUY"):
                    key="MAX_BUY"
                if key in ("MAXSELL","MAX_SELL"):
                    key="MAX_SELL"
                if key in PARAM:
                    try:
                        # int/float ayarı
                        if key in ("MAX_BUY","MAX_SELL"):
                            PARAM[key] = int(val)
                        else:
                            PARAM[key] = float(val)
                        safe_save(PARAM_FILE, PARAM)
                        # param değiştiyse state'e de yansıt
                        STATE["params"] = PARAM
                        safe_save(STATE_FILE, STATE)
                        tg_send(f"⚙️ {key} = {PARAM[key]} olarak ayarlandı")
                    except:
                        tg_send(f"❌ {key} değeri sayı olmalı")
                else:
                    tg_send(f"❌ bilinmeyen param: {key}")
            else:
                tg_send("kullanım: /set PARAM VALUE")

        elif lower == "/queue":
            q = safe_load(TG_QUEUE_FILE, [])
            tg_send(f"📨 Queue length: {len(q)}")

        elif lower == "/forceflush":
            tg_flush_queue()
            tg_send("📤 Queue flush denendi.")

        # ileride /report eklenebilir (closed_trades analizi vs.)
        last_update_id = uid

    return last_update_id

# ================= DAILY SUMMARY (BASİT) =================
def maybe_daily_summary():
    """
    Günlük rapor + dosya push.
    Burada sadece kapalı işlemleri json olarak atıyoruz.
    """
    today_str = now_ist_dt().strftime("%Y-%m-%d")
    if STATE.get("last_daily_sent_date","") == today_str:
        return

    closed_list = safe_load(CLOSED_TRADES_FILE, [])
    if closed_list:
        # basit özet
        wins = [t for t in closed_list if t.get("result")=="TP"]
        winrate = (len(wins)/len(closed_list)*100.0) if closed_list else 0.0
        tg_send(
            "📊 Günlük Rapor\n"
            f"Tarih: {today_str}\n"
            f"Toplam Trade: {len(closed_list)}\n"
            f"Winrate: {winrate:.1f}%"
        )
        # tüm closed_trades.json'ı yolla
        raw = json.dumps(closed_list, ensure_ascii=False, indent=2).encode("utf-8")
        tg_send_file(f"closed_trades_{today_str}.json", raw)
    # push state de
    raw_state = json.dumps(STATE, ensure_ascii=False, indent=2).encode("utf-8")
    tg_send_file(f"state_snapshot_{today_str}.json", raw_state)

    STATE["last_daily_sent_date"]=today_str
    safe_save(STATE_FILE, STATE)

# ================== MAIN LOOP ==================
def main():
    tg_send("🚀 EMA ULTRA v13.4 başladı (Real Orders + Smart AutoSwitch + Queue)")
    last_update_id = 0

    # sembolleri bir kere çekelim
    exinfo = futures_exchange_info()
    symbols = [s["symbol"] for s in exinfo if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    symbols.sort()

    while True:
        tg_flush_queue()

        # Telegram komutlarını işle
        last_update_id = tg_poll_commands(last_update_id)

        # open pozisyonlarda TP/SL kapanan oldu mu diye bak
        closed_happened = try_close_positions()
        if closed_happened:
            # kapanış olduysa autotrade tekrar açılabilir (kuralımıza göre)
            enforce_mode_after_close()

        # sinyal tara
        STATE["bar_index"] += 1
        bar_i = STATE["bar_index"]

        for sym in symbols:
            # klines çekiyoruz
            kl1 = futures_get_klines(sym, "1h", 200)
            if len(kl1) < 120:
                continue

            # CROSS sinyali
            cross_sig = build_cross_signal(sym, kl1)
            if cross_sig:
                cross_key = f"{sym}_{cross_sig['dir']}"
                # spam engelle
                if STATE["last_cross_seen"].get(cross_key) != cross_sig["time"]:
                    # telegram sinyal
                    tg_send(
                        f"{cross_sig['color']} {cross_sig['tier']} CROSS {sym} {cross_sig['dir']}\n"
                        f"Pow:{cross_sig['power']:.1f} RSI:{cross_sig['rsi']:.1f}\n"
                        f"Açı:{cross_sig['ang_now']:+.1f}° Δ:{cross_sig['ang_change']:.1f}°"
                    )
                    # trade dene
                    execute_trade_if_allowed(cross_sig)
                    # seen kaydet
                    STATE["last_cross_seen"][cross_key] = cross_sig["time"]

            # SCALP sinyali
            scalp_sig = build_scalp_signal(sym, kl1, STATE["last_scalp_seen"], bar_i)
            if scalp_sig:
                scalp_key = scalp_sig["cooldown_key"]
                if STATE["last_scalp_seen"].get(scalp_key) != bar_i:
                    tg_send(
                        f"{scalp_sig['color']} {scalp_sig['tier']} SCALP {sym} {scalp_sig['dir']}\n"
                        f"Pow:{scalp_sig['power']:.1f} RSI:{scalp_sig['rsi']:.1f}\n"
                        f"Açı:{scalp_sig['ang_now']:+.1f}° Δ:{scalp_sig['ang_change']:.1f}°"
                    )
                    execute_trade_if_allowed(scalp_sig)
                    STATE["last_scalp_seen"][scalp_key] = bar_i

            # loop nefes
            time.sleep(0.08)

        # günlük rapor kontrol
        maybe_daily_summary()

        # state ve paramları kalıcı yaz
        STATE["params"] = PARAM
        safe_save(STATE_FILE, STATE)
        safe_save(PARAM_FILE, PARAM)

        # ana döngü sleep
        time.sleep(120)

# ================== ENTRY ==================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        tg_send(f"❗FATAL: {e}")
        _queue_append({"type":"text","text":f"FATAL: {e}"})
