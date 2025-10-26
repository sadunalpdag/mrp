import os, json, time, requests, hmac, hashlib, threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================================
# EMA ULTRA v15.9.1 - ConfirmedBar / SilentSim / ULTRA MsgLock / TP-SL Fix
#
# ÖZET:
#   ✔ Sadece son kapanmış 1h mumdan gelen ters-slope onayına göre sinyal üretir
#     (confirmed bar logic).
#   ✔ ULTRA sinyaller:
#       - Telegram'a gider
#       - log.txt'ye yazılır
#       - gerçek MARKET emir açar
#       - TP/SL (closePosition=true) ekler (Binance param fix)
#       - TrendLock ile aynı yönde tekrar mesaj/işlem yok
#   ✔ PREMIUM / NORMAL sinyaller:
#       - Telegram'a gitmez
#       - log'a yazılmaz
#       - sadece simülasyona yazılır (sessiz)
#         -> 30/60/90/120 dk gecikmeli varyantlar
#         -> TP/SL tetiklenince sim_closed.json'a outcome (WIN/LOSS, TP/SL, gain_pct, vs)
#   ✔ Günlük volatilite |chg24h| >= %10 ise sinyal yok
#   ✔ ANGLE_MIN filtresi (min slope impulse)
#   ✔ MAX_BUY / MAX_SELL sınırı:
#       - aşıldıysa auto_trade_active=False (gerçek emir açmaz)
#       - normale dönünce tekrar True
#   ✔ TrendLock:
#       - Örn BTCUSDT "UP" kilitlenmişse, aynı yönde gelen ULTRA sinyali tekrar
#         Telegram'a yazılmaz ve yeni işlem açılmaz.
#       - Ancak slope gerçekten DOWN'a dönerse kilit çözülür ve yeni ULTRA tekrar aktif olur.
#   ✔ 4 saatte bir auto-report Telegram'a şu dosyaları yollar:
#       - ai_signals.json
#       - ai_analysis.json
#       - ai_rl_log.json
#       - sim_positions.json
#       - sim_closed.json
#
# Files in DATA_DIR:
#   state.json
#   params.json
#   ai_signals.json
#   ai_analysis.json
#   ai_rl_log.json
#   sim_positions.json
#   sim_closed.json
#   log.txt
# ==============================================================================

# ================= PATHS / FILES =================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE        = os.path.join(DATA_DIR,"state.json")
PARAM_FILE        = os.path.join(DATA_DIR,"params.json")
AI_SIGNALS_FILE   = os.path.join(DATA_DIR,"ai_signals.json")
AI_ANALYSIS_FILE  = os.path.join(DATA_DIR,"ai_analysis.json")
AI_RL_FILE        = os.path.join(DATA_DIR,"ai_rl_log.json")
SIM_POS_FILE      = os.path.join(DATA_DIR,"sim_positions.json")
SIM_CLOSED_FILE   = os.path.join(DATA_DIR,"sim_closed.json")
LOG_FILE          = os.path.join(DATA_DIR,"log.txt")

# ================= ENV VARS =================
BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")

BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()

# TrendLock: canlı tarafta yön kilidi (runtime only)
# Ör: TREND_LOCK["BTCUSDT"] = "UP"
# -> BTCUSDT için UP yönlü pozisyon açıldı. Aynı yönde ULTRA sinyal gelse bile
#    Telegram mesajı da yok, yeni trade de yok.
# Slope tersine dönüp DOWN ULTRA confirmed olunca unlock edilir.
TREND_LOCK = {}

# SIM_QUEUE: gelecekte açılacak sim girişleri (henüz aktif değil)
# Her sinyal için 30/60/90/120 dk gecikmeli varyant push ediyoruz.
SIM_QUEUE = []

# ================= BASIC HELPERS =================
def safe_load(p,d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return d

def safe_save(p,d):
    try:
        with SAVE_LOCK:
            tmp=p+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:
        print(f"[SAVE ERR] {e}", flush=True)

def log(msg):
    """
    log() sadece ULTRA gerçek trade akışıyla ve kritik eventlerle çağrılacak.
    PREMIUM / NORMAL sinyaller sessiz; onlar log() çağırmaz.
    """
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms():
    return int(datetime.now(timezone.utc).timestamp()*1000)

def now_ts_s():
    return int(datetime.now(timezone.utc).timestamp())

def now_local_iso():
    # UTC+3 readable timestamp
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def enforce_max_file_size(path, max_mb=10):
    """
    Dosya çok büyürse son %20'sini sakla.
    """
    try:
        if os.path.exists(path):
            sz = os.path.getsize(path)
            if sz > max_mb*1024*1024:
                with open(path,"r",encoding="utf-8") as f:
                    raw=f.read()
                tail = raw[-int(len(raw)*0.2):]
                with open(path,"w",encoding="utf-8") as f:
                    f.write(tail)
    except:
        pass

# ================= TELEGRAM =================
def tg_send(text):
    """
    Sadece ULTRA (real trade) ile ilgili kritik bilgileri göndereceğiz.
    PREMIUM / NORMAL sinyalleri asla göndermiyoruz.
    """
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":text},
            timeout=10
        )
    except:
        pass

def tg_send_file(path, caption):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(path):
        return
    try:
        with open(path,"rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":caption},
                files={"document":(os.path.basename(path),f)},
                timeout=20
            )
    except:
        pass

# ================= BINANCE HELPERS =================
def _signed_request(method, path, payload):
    """
    Binance Futures signed request helper.
    """
    query = "&".join([f"{k}={payload[k]}" for k in payload])
    sig = hmac.new(
        BINANCE_SECRET.encode(),
        query.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + query + "&signature=" + sig

    if method=="POST":
        r = requests.post(url, headers=headers, timeout=10)
    else:
        r = requests.get(url, headers=headers, timeout=10)

    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(symbol):
    """
    Son fiyat.
    """
    try:
        r = requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/price",
            params={"symbol":symbol},
            timeout=5
        ).json()
        return float(r["price"])
    except:
        return None

def futures_24h_change(symbol):
    """
    24h yüzde değişimi.
    """
    try:
        r = requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/24hr",
            params={"symbol":symbol},
            timeout=5
        ).json()
        return float(r["priceChangePercent"])
    except:
        return 0.0

def futures_get_klines(symbol, interval, limit):
    """
    Kapanmış mum verilerini al.
    Son henüz kapanmamış (gelecek) mum gibi görünen varsa at.
    """
    try:
        r = requests.get(
            BINANCE_FAPI+"/fapi/v1/klines",
            params={"symbol":symbol,"interval":interval,"limit":limit},
            timeout=10
        ).json()
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        # eğer son mum future candle'sa (kapanmamışsa) drop et
        if r and int(r[-1][6])>now_ms:
            r = r[:-1]
        return r
    except:
        return []

def get_symbol_filters(symbol):
    """
    LOT_SIZE / PRICE_FILTER -> stepSize, tickSize
    bunlar miktar ve fiyat precision için lazım.
    """
    try:
        info = requests.get(
            BINANCE_FAPI+"/fapi/v1/exchangeInfo",
            timeout=10
        ).json()
        s = next((x for x in info["symbols"] if x["symbol"]==symbol),None)
        lot = next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef = next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        stepSize = float(lot.get("stepSize","1"))
        tickSize = float(pricef.get("tickSize","0.01"))
        return {"stepSize":stepSize,"tickSize":tickSize}
    except:
        return {"stepSize":1.0,"tickSize":0.01}

def round_nearest(x, step):
    if step == 0:
        return x
    return round(round(x/step)*step, 12)

def adjust_precision(symbol, value, mode="price"):
    """
    Binance tickSize / stepSize yuvarlama + 0'dan büyük olmasını sağla.
    mode="price" -> PRICE_FILTER.tickSize
    mode="qty"   -> LOT_SIZE.stepSize
    """
    f = get_symbol_filters(symbol)
    step = f["tickSize"] if mode=="price" else f["stepSize"]
    adj  = round_nearest(value, step)
    adj  = max(adj, step)
    return float(f"{adj:.12f}")

def calc_order_qty(symbol, entry_price, notional_usdt):
    """
    TRADE_SIZE_USDT büyüklüğünde kaç kontrat?
    """
    raw = notional_usdt / max(entry_price,1e-12)
    return adjust_precision(symbol, raw, "qty")

def classify_symbol(symbol):
    """
    Basit sınıf: Major vs Alt.
    Bu analiz için (sim kayıtlarına yazıyoruz).
    """
    majors = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT"]
    return "Major" if symbol in majors else "Alt"

def daily_volatility_rank(chg24h_abs):
    """
    Günlük volatilite seviyesini etiketle.
    """
    if chg24h_abs < 2.0:
        return "low"
    elif chg24h_abs < 5.0:
        return "medium"
    else:
        return "high"

def open_market_position(symbol, direction, qty):
    """
    Gerçek market emri aç.
    direction: "UP"   -> BUY LONG
               "DOWN" -> SELL SHORT
    """
    side          = "BUY"  if direction=="UP" else "SELL"
    position_side = "LONG" if direction=="UP" else "SHORT"

    payload = {
        "symbol":symbol,
        "side":side,
        "type":"MARKET",
        "quantity":f"{qty}",
        "positionSide":position_side,  # hedge mode varsayımı
        "timestamp":now_ts_ms()
    }

    res = _signed_request("POST","/fapi/v1/order",payload)

    fills = res.get("avgPrice") or res.get("price") or None
    try:
        entry_exec = float(fills) if fills else futures_get_price(symbol)
    except:
        entry_exec = futures_get_price(symbol)

    entry_exec = adjust_precision(symbol, entry_exec, "price")

    return {
        "symbol":symbol,
        "dir":direction,
        "positionSide":position_side,
        "qty":qty,
        "entry":entry_exec
    }

def futures_set_tp_sl(symbol, direction, qty, entry_price, tp_pct, sl_pct):
    """
    TP/SL emirlerini kurar.
    IMPORTANT (v15.9.1 fix):
    - TAKE_PROFIT_MARKET ve STOP_MARKET emirlerinde Binance 'price' alanını istemiyor,
      stopPrice yeterli. Eskiden 'price' gönderiyorduk, -1106 hatası veriyordu.
    - reduceOnly yok, onun yerine closePosition=true kullanıyoruz ki
      mevcut pozisyonu kapatsın.
    """
    position_side = "LONG" if direction == "UP" else "SHORT"
    close_side    = "SELL" if direction == "UP" else "BUY"

    tp_raw = entry_price * (1 + tp_pct) if direction == "UP" else entry_price * (1 - tp_pct)
    sl_raw = entry_price * (1 - sl_pct) if direction == "UP" else entry_price * (1 + sl_pct)
    tp_s   = adjust_precision(symbol, tp_raw, "price")
    sl_s   = adjust_precision(symbol, sl_raw, "price")

    for ttype, pr in [("TAKE_PROFIT_MARKET", tp_s), ("STOP_MARKET", sl_s)]:
        payload = {
            "symbol": symbol,
            "side": close_side,
            "type": ttype,
            "stopPrice": f"{pr:.12f}",   # ✅ sadece stopPrice
            "quantity": f"{qty}",
            "workingType": "MARK_PRICE",
            "closePosition": "true",     # pozisyonu kapat
            "positionSide": position_side,
            "timestamp": now_ts_ms()
        }
        try:
            _signed_request("POST", "/fapi/v1/order", payload)
        except Exception as e:
            tg_send(f"⚠️ TP/SL ERR {symbol} {e}")
            log(f"[TP/SL ERR] {symbol} {e}")

def fetch_open_positions_real():
    """
    Binance üzerindeki gerçek aktif pozisyonları çekiyoruz:
      - MAX_BUY / MAX_SELL için
      - Duplicate guard için
    """
    result = {"long":{}, "short":{},"long_count":0,"short_count":0}
    try:
        payload={"timestamp":now_ts_ms()}
        acc = _signed_request("GET","/fapi/v2/positionRisk",payload)
        for p in acc:
            sym = p["symbol"]
            pos_amt = float(p["positionAmt"])
            if pos_amt>0:
                result["long"][sym]=abs(pos_amt)
            elif pos_amt<0:
                result["short"][sym]=abs(pos_amt)
        result["long_count"]=len(result["long"])
        result["short_count"]=len(result["short"])
    except Exception as e:
        log(f"[FETCH POS ERR] {e}")
    return result

# ================= PARAM / STATE / MEMORY =================
PARAM_DEFAULT = {
    "SCALP_TP_PCT":    0.006,   # %0.6 TP
    "SCALP_SL_PCT":    0.20,    # %20 SL
    "TRADE_SIZE_USDT": 250.0,
    "MAX_BUY":         30,
    "MAX_SELL":        30,
    "ANGLE_MIN":       0.0001   # min slope impulse
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index":         0,
    "last_report":       0,
    "auto_trade_active": True
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
if "auto_trade_active" not in STATE:
    STATE["auto_trade_active"]=True

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE,   [])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE,  [])
AI_RL         = safe_load(AI_RL_FILE,        [])
SIM_POSITIONS = safe_load(SIM_POS_FILE,      [])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE,   [])

# ================= INDICATORS =================
def ema(vals,n):
    k=2/(n+1)
    e=[vals[0]]
    for v in vals[1:]:
        e.append(v*k+e[-1]*(1-k))
    return e

def rsi(vals,period=14):
    """
    Classic RSI. Kapanmış mumlar üzerinde.
    """
    if len(vals)<period+2:
        return [50]*len(vals)
    d=np.diff(vals)
    g=np.maximum(d,0)
    l=-np.minimum(d,0)
    ag=np.mean(g[:period])
    al=np.mean(l[:period])
    out=[50]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period
        al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0:
            tr.append(h[i]-l[i])
        else:
            tr.append(max(
                h[i]-l[i],
                abs(h[i]-c[i-1]),
                abs(l[i]-c[i-1])
            ))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def calc_power(e7_now,e7_prev,e7_prev2,atr_v,price,rsi_val):
    """
    "power" metriği:
    slope farkı, ATR/price ve RSI katkısı ile 0-100 arası skor.
    """
    diff=abs(e7_now-e7_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55+diff*20+((rsi_val-50)/50)*15+(atr_v/price)*200
    score=min(100,max(0,base))
    return score

def tier_from_power(p):
    """
    Güç seviyesine göre tier ayırımı.
    """
    if p>=75:   return "ULTRA","🟩"
    elif p>=68: return "PREMIUM","🟦"
    elif p>=60: return "NORMAL","🟨"
    return None,""
# ================= SIM ENGINE =================
def queue_sim_variants(sig):
    """
    Her sinyal (ULTRA / PREMIUM / NORMAL) için 4 gecikmeli plan (30/60/90/120dk).
    Telegram YOK, log YOK.
    Bu kayıtlar ileride sim_positions.json / sim_closed.json datasını besleyecek.
    """
    delays_min = [30, 60, 90, 120]
    now_s = now_ts_s()

    slope_now  = sig["_s_now"]
    slope_prev = sig["_s_prev"]
    angle_diff_val = abs(slope_now - slope_prev)
    price_volatility_ratio = (sig["atr"]/sig["entry"]) if sig["entry"]>0 else 0.0
    trend_strength = angle_diff_val * (sig["rsi"]/50.0)
    vol_rank = daily_volatility_rank(abs(sig["chg24h"]))
    sym_class = classify_symbol(sig["symbol"])

    for dmin in delays_min:
        SIM_QUEUE.append({
            "symbol": sig["symbol"],
            "symbol_class": sym_class,
            "dir": sig["dir"],
            "tier": sig["tier"],
            "chg24h": sig["chg24h"],
            "daily_volatility_rank": vol_rank,

            "entry": sig["entry"],
            "tp": sig["tp"],
            "sl": sig["sl"],

            "rsi": sig["rsi"],
            "atr": sig["atr"],
            "power": sig["power"],
            "ema_slope_now": slope_now,
            "ema_slope_prev": slope_prev,
            "angle_diff": angle_diff_val,
            "price_volatility_ratio": price_volatility_ratio,
            "trend_strength": trend_strength,

            "delay_min": dmin,
            "planned_ts": now_s + dmin*60,
            "born_bar": sig["born_bar"],
            "signal_age_bars": 0,
            "timestamp_added": now_ts_s()
        })


def process_sim_queue_and_open_due():
    """
    planned_ts zamanı dolan queued sim girişlerini aktif sim pozisyonuna çevirir.
    Bunlar sim_positions.json'da tutulur.
    Telegram YOK, log YOK.
    """
    now_s = now_ts_s()
    still=[]
    opened_any=False

    for q in SIM_QUEUE:
        if now_s >= q["planned_ts"]:
            sim_pos = {
                "symbol": q["symbol"],
                "symbol_class": q["symbol_class"],
                "dir": q["dir"],
                "tier": q["tier"],
                "chg24h": q["chg24h"],
                "daily_volatility_rank": q["daily_volatility_rank"],

                "entry": q["entry"],
                "tp": q["tp"],
                "sl": q["sl"],

                "rsi": q["rsi"],
                "atr": q["atr"],
                "power": q["power"],
                "ema_slope_now": q["ema_slope_now"],
                "ema_slope_prev": q["ema_slope_prev"],
                "angle_diff": q["angle_diff"],
                "price_volatility_ratio": q["price_volatility_ratio"],
                "trend_strength": q["trend_strength"],

                "delay_min": q["delay_min"],
                "opened_at": now_local_iso(),
                "opened_ts": now_s,
                "born_bar": q["born_bar"],
                "signal_age_bars": q["signal_age_bars"],
                "timestamp_added": q["timestamp_added"],

                "closed": False
            }
            SIM_POSITIONS.append(sim_pos)
            opened_any=True
        else:
            still.append(q)

    SIM_QUEUE[:] = still

    if opened_any:
        safe_save(SIM_POS_FILE, SIM_POSITIONS)


def process_sim_closes():
    """
    SIM_POSITIONS içindeki açık sim işlemler TP/SL'e ulaştı mı?
    Ulaştıysa kapatıp SIM_CLOSED'e outcome olarak yaz.
    Telegram YOK, log YOK.
    """
    global SIM_POSITIONS
    changed=False
    price_cache={}

    still=[]
    now_s = now_ts_s()
    now_iso = now_local_iso()

    for p in SIM_POSITIONS:
        if p.get("closed"):
            continue

        sym = p["symbol"]
        if sym not in price_cache:
            price_cache[sym]=futures_get_price(sym)
        last_price = price_cache[sym]
        if last_price is None:
            still.append(p)
            continue

        hit=None
        outcome_type=None

        if p["dir"]=="UP":
            if last_price>=p["tp"]:
                hit="WIN"; outcome_type="TP"
            elif last_price<=p["sl"]:
                hit="LOSS"; outcome_type="SL"
        else:  # DOWN
            if last_price<=p["tp"]:
                hit="WIN"; outcome_type="TP"
            elif last_price>=p["sl"]:
                hit="LOSS"; outcome_type="SL"

        if hit is None:
            still.append(p)
            continue

        hold_time_min = (now_s - p["opened_ts"]) / 60.0
        if p["dir"]=="UP":
            gain_pct = (last_price / p["entry"] - 1.0) * 100.0
        else:
            gain_pct = (p["entry"] / last_price - 1.0) * 100.0

        SIM_CLOSED.append({
            "symbol": p["symbol"],
            "symbol_class": p.get("symbol_class","?"),

            "dir": p["dir"],
            "tier": p["tier"],
            "chg24h": p["chg24h"],
            "daily_volatility_rank": p.get("daily_volatility_rank"),

            "entry": p["entry"],
            "tp": p["tp"],
            "sl": p["sl"],
            "exit_price": last_price,

            "rsi": p["rsi"],
            "atr": p["atr"],
            "power": p["power"],
            "ema_slope_now": p["ema_slope_now"],
            "ema_slope_prev": p["ema_slope_prev"],
            "angle_diff": p["angle_diff"],
            "price_volatility_ratio": p["price_volatility_ratio"],
            "trend_strength": p["trend_strength"],

            "delay_min": p["delay_min"],
            "opened_at": p["opened_at"],
            "closed_at": now_iso,

            "born_bar": p.get("born_bar"),
            "signal_age_bars": p.get("signal_age_bars",0),

            "hold_time_min": hold_time_min,
            "gain_pct": gain_pct,
            "outcome_type": outcome_type,
            "result": hit,

            "timestamp_opened": p.get("opened_ts"),
            "timestamp_closed": now_s
        })
        changed=True

    SIM_POSITIONS = still

    if changed:
        safe_save(SIM_POS_FILE, SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE, SIM_CLOSED)


# ================= AI LOGGING / ANALYSIS =================
def ai_log_signal(sig):
    """
    Her sinyali (ULTRA / PREMIUM / NORMAL) kaydederiz.
    Telegram yok, sessiz.
    """
    AI_SIGNALS.append({
        "time":now_local_iso(),
        "symbol":sig["symbol"],
        "dir":sig["dir"],
        "tier":sig["tier"],
        "chg24h":sig["chg24h"],
        "power":sig["power"],
        "rsi":sig["rsi"],
        "atr":sig["atr"],
        "tp":sig["tp"],
        "sl":sig["sl"],
        "entry":sig["entry"]
    })
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)


def ai_update_analysis():
    """
    Küçük özet snapshot: kaç sinyal, kaç açık sim, kaç kapalı sim.
    Bu da AI_ANALYSIS dosyasına apendleniyor.
    """
    ultra_count = sum(1 for x in AI_SIGNALS if x.get("tier")=="ULTRA")
    prem_count  = sum(1 for x in AI_SIGNALS if x.get("tier")=="PREMIUM")
    norm_count  = sum(1 for x in AI_SIGNALS if x.get("tier")=="NORMAL")
    snapshot = {
        "time":now_local_iso(),
        "ultra_signals_total": ultra_count,
        "premium_signals_total": prem_count,
        "normal_signals_total": norm_count,
        "sim_open_count": len(SIM_POSITIONS),
        "sim_closed_count": len(SIM_CLOSED)
    }
    AI_ANALYSIS.append(snapshot)
    safe_save(AI_ANALYSIS_FILE, AI_ANALYSIS)


def auto_report_if_due():
    """
    Her 4 saatte bir Telegram'a kritik dosyaları gönder:
      - ai_signals.json
      - ai_analysis.json
      - ai_rl_log.json
      - sim_positions.json
      - sim_closed.json
    Ayrıca küçük bir "yedek gönderildi" mesajı at.
    """
    now_now = time.time()
    if now_now - STATE.get("last_report",0) < 14400:
        return

    ai_update_analysis()

    for fpath in [
        AI_SIGNALS_FILE,
        AI_ANALYSIS_FILE,
        AI_RL_FILE,
        SIM_POS_FILE,
        SIM_CLOSED_FILE
    ]:
        enforce_max_file_size(fpath)
        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")

    tg_send("🕐 4 saatlik yedek gönderildi.")

    STATE["last_report"] = now_now
    safe_save(STATE_FILE, STATE)


# ================= SIGNAL BUILD (CONFIRMED 1H BAR) =================
def build_scalp_signal(sym, kl, bar_i):
    """
    EMA7 slope reversal sinyali.
    *** CONFIRMED BAR LOGIC ***
    - Aktif (kapanmamış) barı kullanmıyoruz.
    - Yalnızca son TAM kapanmış 1h mumun eğimine bakıyoruz.
    - günlük |chg24h| >= 10% ise sinyal yok.
    - slope_impulse ANGLE_MIN altındaysa sinyal yok.
    - power -> tier (ULTRA / PREMIUM / NORMAL)
    - precision (tickSize / stepSize) TP/SL için uygulanıyor.
    """

    if len(kl) < 60:
        return None

    # Kapanmış mum verileri:
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    # 24h değişim filtresi
    chg = futures_24h_change(sym)
    if abs(chg) >= 10.0:
        return None

    # EMA7
    e7 = ema(closes,7)
    if len(e7) < 7:
        return None

    # --- CONFIRMED BAR SLOPE LOGIC ---
    # s_now   = son kapanmış barın eğimi
    # s_prev  = ondan bir önceki kapanmış barın eğimi
    #
    # e7[-2]   = son TAM kapanmış barın EMA7
    # e7[-5]   = bu barın 3 bar önceki EMA7
    # e7[-3]   = bir önceki TAM kapanmış barın EMA7
    # e7[-6]   = onun da 3 bar önceki EMA7
    s_now  = e7[-2] - e7[-5]
    s_prev = e7[-3] - e7[-6]

    # slope yön değişimi?
    if s_prev < 0 and s_now > 0:
        direction="UP"
    elif s_prev > 0 and s_now < 0:
        direction="DOWN"
    else:
        return None

    # slope impulse filtresi
    slope_impulse = abs(s_now - s_prev)
    if slope_impulse < PARAM["ANGLE_MIN"]:
        return None

    # TrendLock unlock:
    # Eğer kilitli yön != yeni sinyal yönü ise kilidi çözüyoruz.
    prev_locked = TREND_LOCK.get(sym)
    if prev_locked and direction != prev_locked:
        # Bu ULTRA tarafında önemli state değişimi -> log serbest
        del TREND_LOCK[sym]
        log(f"[UNLOCK] {sym} {prev_locked}->{direction}")

    # ATR / RSI hesapla (son kapanmış barlar)
    atr_v = atr_like(highs,lows,closes)[-1]
    r_val = rsi(closes)[-1]

    # power hesapla
    pwr = calc_power(
        e7[-2],   # confirmed son bar EMA7
        e7[-3],   # prev bar EMA7
        e7[-6],   # prev2 EMA7
        atr_v,
        closes[-1],
        r_val
    )

    tier, emoji = tier_from_power(pwr)
    if not tier:
        return None

    # Fiyat çek (anlık son fiyat)
    entry_raw = futures_get_price(sym)
    if entry_raw is None:
        return None

    # TP/SL hesapla ve precision uygula
    if direction=="UP":
        tp_raw = entry_raw*(1+PARAM["SCALP_TP_PCT"])
        sl_raw = entry_raw*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_raw = entry_raw*(1-PARAM["SCALP_TP_PCT"])
        sl_raw = entry_raw*(1+PARAM["SCALP_SL_PCT"])

    entry_adj = adjust_precision(sym, entry_raw, "price")
    tp_adj    = adjust_precision(sym, tp_raw,   "price")
    sl_adj    = adjust_precision(sym, sl_raw,   "price")

    # sim log enrichment
    price_vol_ratio = (atr_v/entry_adj) if entry_adj>0 else 0.0
    trend_strength = slope_impulse * (r_val/50.0)

    sig = {
        "symbol":sym,
        "dir":direction,
        "tier":tier,          # ULTRA / PREMIUM / NORMAL
        "emoji":emoji,

        "entry":entry_adj,
        "tp":tp_adj,
        "sl":sl_adj,

        "power":pwr,
        "rsi":r_val,
        "atr":atr_v,
        "chg24h":chg,

        "born_bar":bar_i,

        # confirmed bar slope data
        "_s_now": s_now,
        "_s_prev": s_prev,
        "angle_diff": slope_impulse,
        "price_volatility_ratio": price_vol_ratio,
        "trend_strength": trend_strength
    }
    return sig


def scan_symbol(sym, bar_i):
    kl = futures_get_klines(sym,"1h",200)
    if len(kl) < 60:
        return None
    return build_scalp_signal(sym,kl,bar_i)


def run_parallel(symbols, bar_i):
    """
    Sembolleri paralelde tara.
    """
    found=[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs=[ex.submit(scan_symbol,s,bar_i) for s in symbols]
        for f in as_completed(futs):
            try:
                sig=f.result()
            except:
                sig=None
            if sig:
                found.append(sig)
    return found


# ================= REAL TRADE CONTROL (ULTRA ONLY) =================
def dynamic_autotrade_state():
    """
    MAX_BUY / MAX_SELL koruması:
      limit dolarsa auto_trade_active=False
      normale dönünce tekrar True
    Bu sadece gerçek trade katmanını etkiler.
    """
    live = fetch_open_positions_real()

    if STATE.get("auto_trade_active",True):
        if (live["long_count"] >= PARAM["MAX_BUY"]) or (live["short_count"] >= PARAM["MAX_SELL"]):
            STATE["auto_trade_active"] = False
            tg_send(
                f"🚫 AutoTrade durduruldu — limit aşıldı "
                f"(long:{live['long_count']}/{PARAM['MAX_BUY']} "
                f"short:{live['short_count']}/{PARAM['MAX_SELL']})"
            )
            log("[AUTOTRADE] stopped by limits")
    else:
        if (live["long_count"] < PARAM["MAX_BUY"]) and (live["short_count"] < PARAM["MAX_SELL"]):
            STATE["auto_trade_active"] = True
            tg_send(
                f"✅ AutoTrade yeniden aktif "
                f"(long:{live['long_count']}/{PARAM['MAX_BUY']} "
                f"short:{live['short_count']}/{PARAM['MAX_SELL']})"
            )
            log("[AUTOTRADE] re-enabled")

    safe_save(STATE_FILE, STATE)


def should_skip_real_due_to_trendlock(sig):
    """
    Eğer TREND_LOCK[symbol] == aynı yöndeki sinyal yönü ise:
      - Telegram'a da sinyal mesajı gitmeyecek
      - trade de açılmayacak
    """
    sym = sig["symbol"]
    d   = sig["dir"]
    if TREND_LOCK.get(sym) == d:
        log(f"[LOCK] {sym} {d} locked -> skip real open/msg")
        return True
    return False


def should_skip_real_due_to_existing_position(sig):
    """
    Aynı yönde zaten açık gerçek pozisyon varsa tekrar açma.
    """
    sym = sig["symbol"]
    d   = sig["dir"]
    live = fetch_open_positions_real()
    if d=="UP" and sym in live["long"]:
        log(f"[DUP] {sym} already LONG real, skip")
        return True
    if d=="DOWN" and sym in live["short"]:
        log(f"[DUP] {sym} already SHORT real, skip")
        return True
    return False


def execute_real_trade(sig):
    """
    Sadece ULTRA sinyaller gerçek emir açabilir.
    Bu fonksiyon çağrılmadan önce:
      - TREND_LOCK aynı yöne kilitliyse zaten çağırmıyoruz
      - auto_trade_active kontrol ediliyor
    """
    if sig["tier"] != "ULTRA":
        return
    if not STATE.get("auto_trade_active",True):
        return
    if should_skip_real_due_to_trendlock(sig):
        return
    if should_skip_real_due_to_existing_position(sig):
        return

    sym = sig["symbol"]
    direc = sig["dir"]

    qty = calc_order_qty(sym, sig["entry"], PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        tg_send(f"❗ {sym} qty hesaplanamadı.")
        log(f"[QTY ERR] {sym} qty calc failed")
        return

    try:
        opened = open_market_position(sym, direc, qty)
        entry_exec = opened["entry"]

        futures_set_tp_sl(
            sym,
            direc,
            qty,
            entry_exec,
            PARAM["SCALP_TP_PCT"],
            PARAM["SCALP_SL_PCT"]
        )

        tg_send(
            f"✅ REAL {sym} {direc} {sig['tier']} qty:{qty}\n"
            f"Entry:{entry_exec:.12f}\n"
            f"TP%:{PARAM['SCALP_TP_PCT']*100:.3f} "
            f"SL%:{PARAM['SCALP_SL_PCT']*100:.1f}\n"
            f"time:{now_local_iso()}"
        )

        TREND_LOCK[sym] = direc
        log(f"[LOCK SET] {sym} -> {direc}")

        AI_RL.append({
            "time":now_local_iso(),
            "symbol":sym,
            "dir":direc,
            "entry":entry_exec,
            "power":sig["power"],
            "born_bar":sig["born_bar"]
        })
        safe_save(AI_RL_FILE, AI_RL)

    except Exception as e:
        tg_send(f"❌ OPEN ERR {sym} {direc} {e}")
        log(f"[OPEN ERR] {sym} {e}")
# ================= MAIN LOOP =================
def main():
    tg_send("🚀 EMA ULTRA v15.9.1 başladı (ConfirmedBar + ULTRA MsgLock + SilentSim + TP/SL Fix)")
    log("[START] EMA ULTRA v15.9.1 started")

    # Binance USDT sembollerini al
    info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo").json()
    symbols = [
        s["symbol"]
        for s in info["symbols"]
        if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"
    ]
    symbols.sort()

    while True:
        try:
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1️⃣ confirmed bar sinyalleri tara
            sigs = run_parallel(symbols, bar_i)

            # 2️⃣ sim queue + real trade yönet
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)

                # PREMIUM / NORMAL sinyaller sessiz
                if sig["tier"] != "ULTRA":
                    continue

                # ULTRA için trendlock kontrolü
                if TREND_LOCK.get(sig["symbol"]) == sig["dir"]:
                    continue

                # Telegram bildirimi (yalnızca yeni yön için)
                tg_send(
                    f"{sig['emoji']} {sig['tier']} {sig['symbol']} {sig['dir']}\n"
                    f"Pow:{sig['power']:.1f} RSI:{sig['rsi']:.1f} "
                    f"ATR:{sig['atr']:.4f} 24hΔ:{sig['chg24h']:.2f}%\n"
                    f"Entry:{sig['entry']:.12f}\nTP:{sig['tp']:.12f}\nSL:{sig['sl']:.12f}\n"
                    f"born_bar:{sig['born_bar']}"
                )
                log(f"[ULTRA SIG] {sig['symbol']} {sig['dir']} pwr={sig['power']:.1f} chg24h={sig['chg24h']:.2f}%")

                # Gerçek trade aç
                dynamic_autotrade_state()
                execute_real_trade(sig)

            # 3️⃣ sim queue işle
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4️⃣ auto report kontrol
            auto_report_if_due()

            # 5️⃣ state kaydet
            safe_save(STATE_FILE, STATE)

            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR] {e}")
            time.sleep(10)


# ================= ENTRYPOINT =================
if __name__ == "__main__":
    main()