# ==============================================================================
# 📘 EMA ULTRA v13.6.2 — SCALP ONLY (1H) + Volatility Filter + Soft Limit + True Market Entry
#
# Tek dosya deploy için hazır.
#
# 🔧 Nasıl Çalıştırırsın
# ----------------------
# 1) Sisteme environment değişkenlerini gir:
#    export BINANCE_API_KEY="API_KEYIN"
#    export BINANCE_SECRET_KEY="SECRET_KEYIN"
#    export BOT_TOKEN="TELEGRAM_BOT_TOKENIN"
#    export CHAT_ID="SANA_MESAJ_GELECEK_CHAT_ID"
#
# 2) Bağımlılıkları kur:
#    pip install requests numpy
#
# 3) Çalıştır:
#    python3 ema.py
#
# 📂 Runtime sırasında oluşan dosyalar (otomatik):
#    data/state.json
#    data/params.json
#    data/open_positions.json
#    data/closed_trades.json
#    data/log.txt
#
# 🧠 Strateji Özeti
# -----------------
# - Sadece SCALP sinyali açılır. CROSS tamamen kaldırıldı.
# - SCALP sinyali: EMA7 eğiminin yön değiştirmesi
#     slope_prev < 0 ve slope_now > 0  ⇒ LONG (UP)
#     slope_prev > 0 ve slope_now < 0  ⇒ SHORT (DOWN)
#
# - Zaman dilimi: 1h kapanmış mumlar (henüz kapanmamış bar yok).
#
# - True Market Entry:
#     Pozisyon açarken entry = anlık Binance futures fiyatı
#     (mum kapanış fiyatı değil)
#
# - Volatility Filter (VOLATILITY_LIMIT):
#     /fapi/v1/ticker/24hr 'priceChangePercent' çekiliyor.
#     abs(change) >= VOLATILITY_LIMIT ise bu coinde işlem açmıyoruz.
#
#     Varsayılan VOLATILITY_LIMIT = 7.0 (%7)
#     Telegram'dan değiştirilebilir: /set VOLATILITY_LIMIT 5
#
# - Soft Limit (hedge aware):
#     • Binance hedge pozisyonlarını sayar.
#     • MAX_BUY = 30 (LONG), MAX_SELL = 30 (SHORT)
#     • Eğer LONG tarafı limiti geçtiyse auto_trade_long=False olur,
#       SHORT tarafı limiti geçtiyse auto_trade_short=False olur.
#     • Limit altına inince ilgili yön geri açılır.
#
#     • İki yön de kapalıysa bot otomatik simulate=True yapar
#       (sadece kayıt tutar, gerçek emir atmaz).
#     • En az bir yön açıksa simulate=False (gerçek emir atabilir).
#
# - TP/SL emirleri:
#     • TP: TAKE_PROFIT_MARKET
#     • SL: STOP_MARKET
#     • reduceOnly parametresi YOK (Binance -1106 fix)
#     • Fiyatlar tickSize'e snap edilir.
#     • Miktar (qty) LOT_SIZE.stepSize'e snap edilir.
#
# - Telegram komutları:
#     /status
#     /params
#     /set KEY VALUE      veya /set KEY=VALUE
#     /simulate on|off
#     /autotrade on|off
#     /report
#     /export closed      -> closed_trades.json dosyasını Telegram'a yollar
#
#   Desteklenen /set anahtarları:
#     VOLATILITY_LIMIT
#     TRADE_SIZE_USDT
#     MAX_BUY
#     MAX_SELL
#     SCALP_TP_PCT
#     SCALP_SL_PCT
#     SCALP_COOLDOWN_BARS
#     ONLY_ULTRA_TRADES
#
# - simulate davranışı:
#     • STATE["simulate"] True ise hiçbir gerçek emir açılmaz.
#     • FALSE ise gerçek emir açılabilir (limit ve yön izinleri uygunsa).
#
# - auto_trade davranışı:
#     • /autotrade off -> global kapatır, simulate=True zorlanır.
#     • /autotrade on  -> global açar, simulate=False (ama soft limit tekrar override edebilir).
#
# ==============================================================================

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
LOG_FILE           = os.path.join(DATA_DIR, "log.txt")

# ================= ENV VARS =================
BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")

BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

# ================= BASIC HELPERS =================
def safe_load(path, default):
    try:
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return default

def safe_save(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[SAVE ERR] {e}", flush=True)

def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms():
    return int(datetime.now(timezone.utc).timestamp()*1000)

def now_iso():
    # Türkiye UTC+3 varsayımıyla
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

# ================= TELEGRAM HELPERS =================
def tg_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] token/chat yok")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=10
        )
        log(f"[TG OK] {text[:200]}")
    except Exception as e:
        log(f"[TG ERR] {e}")

def tg_send_document(path, caption="file"):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG FILE] token/chat yok")
        return
    if not os.path.exists(path):
        tg_send(f"📂 Dosya yok: {path}")
        return
    try:
        with open(path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"document": (os.path.basename(path), f)},
                timeout=20
            )
        log(f"[TG FILE OK] {path}")
    except Exception as e:
        log(f"[TG FILE ERR] {e}")
        tg_send(f"⚠️ Dosya gönderilemedi: {e}")

# ================= BINANCE CORE =================
def _signed_request(method, path, payload):
    """
    Signed Binance Futures request (Hedge mode assumed active).
    """
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
        raise RuntimeError("Unsupported method")

    if r.status_code != 200:
        raise RuntimeError(f"Binance HTTP {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(symbol):
    """
    Live ticker price (gerçek market fiyatı).
    Entry olarak bunu kullanıyoruz.
    """
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/ticker/price",
            params={"symbol":symbol},
            timeout=5
        ).json()
        return float(r["price"])
    except:
        return None

def futures_24h_change(symbol):
    """
    24 saatlik fiyat değişim yüzdesi.
    priceChangePercent döner. Örn "8.52"
    """
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/ticker/24hr",
            params={"symbol":symbol},
            timeout=5
        ).json()
        return float(r.get("priceChangePercent",0.0))
    except:
        return 0.0

def futures_get_klines(symbol, interval, limit):
    """
    Kline çek. Son kapanmamış barı at.
    """
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/klines",
            params={"symbol":symbol,"interval":interval,"limit":limit},
            timeout=10
        ).json()
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        # son bar kapanmadıysa at
        if r and int(r[-1][6]) > now_ms:
            r = r[:-1]
        return r
    except:
        return []

def futures_exchange_info():
    """
    Precision bilgileri almak için.
    """
    try:
        r = requests.get(
            BINANCE_FAPI + "/fapi/v1/exchangeInfo",
            timeout=10
        ).json()
        return r.get("symbols", [])
    except:
        return []

def get_symbol_filters(symbol, cache=None):
    """
    LOT_SIZE.stepSize ve PRICE_FILTER.tickSize'i al.
    """
    if cache is None:
        cache = futures_exchange_info()
    sym_info = next((s for s in cache if s["symbol"]==symbol), None)
    if not sym_info:
        return {"stepSize":1.0,"tickSize":0.01}
    lot = next((f for f in sym_info["filters"] if f["filterType"]=="LOT_SIZE"), {})
    pricef = next((f for f in sym_info["filters"] if f["filterType"]=="PRICE_FILTER"), {})
    step = float(lot.get("stepSize","1")) if lot else 1.0
    tick = float(pricef.get("tickSize","0.01")) if pricef else 0.01
    return {
        "stepSize": step,
        "tickSize": tick
    }

def snap_qty(qty, step):
    if step<=0:
        return round(qty,3)
    return math.floor(qty/step)*step

def snap_price(px, tick):
    if tick<=0:
        return round(px,6)
    return round(math.floor(px/tick)*tick, 8)

def calc_order_quantity(symbol, usdt_size, filters_cache=None):
    """
    TRADE_SIZE_USDT büyüklüğüyle qty hesapla, stepSize'e snap et.
    """
    live_px = futures_get_price(symbol)
    f = get_symbol_filters(symbol, filters_cache)
    if not live_px or live_px<=0:
        return None, f, live_px
    raw_qty = usdt_size / live_px
    qty_snapped = snap_qty(raw_qty, f["stepSize"])
    if qty_snapped <= 0:
        return None, f, live_px
    return qty_snapped, f, live_px

def futures_market_order(symbol, side, qty, positionSide):
    """
    Market giriş. reduceOnly göndermiyoruz (Binance -1106 fix).
    """
    payload = {
        "symbol": symbol,
        "side": side,  # BUY / SELL
        "type": "MARKET",
        "quantity": qty,
        "positionSide": positionSide,  # LONG / SHORT
        "timestamp": now_ts_ms()
    }
    return _signed_request("POST", "/fapi/v1/order", payload)

def futures_set_tp_sl(symbol, side, positionSide, qty, tp, sl, filters_cache=None):
    """
    TP/SL emirleri. reduceOnly parametresi yok.
    workingType=CONTRACT_PRICE
    """
    f = get_symbol_filters(symbol, filters_cache)
    tick = f["tickSize"]

    tp_s = snap_price(tp, tick)
    sl_s = snap_price(sl, tick)

    close_side = "SELL" if side=="BUY" else "BUY"

    # TAKE_PROFIT_MARKET
    try:
        tp_payload = {
            "symbol": symbol,
            "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{tp_s:.8f}",
            "quantity": f"{qty}",
            "positionSide": positionSide,
            "workingType": "CONTRACT_PRICE",
            "timestamp": now_ts_ms()
        }
        _signed_request("POST", "/fapi/v1/order", tp_payload)
    except Exception as e:
        log(f"[TP ERR] {symbol} {positionSide}: {e}")
        tg_send(f"⚠️ TP emir hatası {symbol} {positionSide}: {e}")

    # STOP_MARKET (SL)
    try:
        sl_payload = {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "stopPrice": f"{sl_s:.8f}",
            "quantity": f"{qty}",
            "positionSide": positionSide,
            "workingType": "CONTRACT_PRICE",
            "timestamp": now_ts_ms()
        }
        _signed_request("POST", "/fapi/v1/order", sl_payload)
    except Exception as e:
        log(f"[SL ERR] {symbol} {positionSide}: {e}")
        tg_send(f"⚠️ SL emir hatası {symbol} {positionSide}: {e}")

def futures_fetch_positions():
    """
    Hedge mod pozisyonlarını oku.
    LONG -> positionAmt > 0
    SHORT -> positionAmt < 0
    """
    payload = {
        "timestamp": now_ts_ms()
    }
    data = _signed_request("GET", "/fapi/v2/positionRisk", payload)
    out = []
    for p in data:
        amt = float(p.get("positionAmt","0"))
        side = p.get("positionSide","BOTH")
        entry_px = float(p.get("entryPrice","0"))
        sym = p.get("symbol","?")
        if side=="LONG" and amt>0:
            out.append({"symbol":sym,"side":"LONG","qty":amt,"entry":entry_px})
        elif side=="SHORT" and amt<0:
            out.append({"symbol":sym,"side":"SHORT","qty":abs(amt),"entry":entry_px})
    return out

# ================= INDICATORS =================
def ema(vals, n):
    if not vals:
        return []
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

def calc_power(e7_now,e7_prev,e7_prev2, atr_now, price_now, rsi_now):
    """
    Momentum kalitesi puanı.
    """
    slope_now  = e7_now  - e7_prev2
    slope_prev = e7_prev - e7_prev2
    slope_comp = abs(slope_now - slope_prev)/(atr_now*0.6) if atr_now>0 else 0
    rsi_comp   = (rsi_now-50)/50.0
    atr_comp   = (atr_now/price_now)*100 if price_now>0 else 0
    base = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    base = max(0.0, min(100.0, base))
    return base, slope_prev, slope_now

def tier_from_power(power, params):
    """
    ULTRA / PREMIUM derecelendirme.
    NORMAL yok çünkü scalp'te zayıf sinyal istemiyoruz.
    """
    if power >= params["POWER_ULTRA_MIN"]:   return "ULTRA","🟩"
    if power >= params["POWER_PREMIUM_MIN"]: return "PREMIUM","🟦"
    return None,""

def filter_ultra(tier, params):
    """
    ONLY_ULTRA_TRADES:
    1.0 => sadece ULTRA kabul
    0.0 => PREMIUM de kabul
    """
    if tier == "ULTRA":
        return True
    if tier == "PREMIUM" and params.get("ONLY_ULTRA_TRADES",0.0) < 1.0:
        return True
    return False

# ================= STATE / PARAM =================
STATE_DEFAULT = {
    "auto_trade": True,          # global izin
    "simulate": True,            # sadece veri toplama modu
    "auto_trade_long": True,     # long yön aktif mi
    "auto_trade_short": True,    # short yön aktif mi

    "last_status_sent": 0,
    "bar_index": 0,

    "last_scalp_seen": {},       # scalp cooldown için
    "last_daily_sent_date": ""
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
for k,v in STATE_DEFAULT.items():
    STATE.setdefault(k,v)

PARAM_DEFAULT = {
    "MAX_BUY":  30.0,          # LONG limit
    "MAX_SELL": 30.0,          # SHORT limit

    "TRADE_SIZE_USDT": 250.0,

    "SCALP_TP_PCT": 0.006,     # %0.6 TP
    "SCALP_SL_PCT": 0.10,      # %10 SL

    "SCALP_COOLDOWN_BARS": 3,  # scalp spam engeli (bar bazlı, 1h)

    "POWER_PREMIUM_MIN": 68.0,
    "POWER_ULTRA_MIN":   75.0,

    "ONLY_ULTRA_TRADES": 1.0,  # 1.0-> sadece ULTRA, 0.0-> PREMIUM da kabul

    "VOLATILITY_LIMIT": 7.0    # |24h change| >= 7% ise SKIP
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
for k,v in PARAM_DEFAULT.items():
    PARAM.setdefault(k,v)
safe_save(PARAM_FILE, PARAM)

# ================= POSITION TRACKING =================
def load_open_positions():
    return safe_load(OPEN_POS_FILE, [])

def save_open_positions(lst):
    safe_save(OPEN_POS_FILE, lst)

def load_closed_trades():
    return safe_load(CLOSED_TRADES_FILE, [])

def save_closed_trades(lst):
    safe_save(CLOSED_TRADES_FILE, lst)

def already_open_same_direction(symbol, direction):
    """
    direction: "UP"(LONG) veya "DOWN"(SHORT)
    Aynı coin aynı yön zaten açık mı?
    """
    for o in load_open_positions():
        if o.get("symbol")==symbol and o.get("dir")==direction:
            return True
    return False

def record_open(sig, mode_flag):
    """
    mode_flag: "real" ya da "sim"
    """
    opens = load_open_positions()
    opens.append({
        "symbol": sig["symbol"],
        "dir": sig["dir"],
        "type": "SCALP",
        "entry": sig["entry"],
        "tp": sig["tp"],
        "sl": sig["sl"],
        "time_open": sig["time"],
        "mode": mode_flag,
        "power": sig.get("power"),
        "tier": sig.get("tier"),
        "rsi": sig.get("rsi"),
        "ang_now": sig.get("ang_now"),
        "ang_change": sig.get("ang_change")
    })
    save_open_positions(opens)

def close_local_position(row, reason, exit_px):
    """
    Local pozisyonu kapatır, closed_trades'e taşır.
    reason: "TP", "SL", "SYNC_CLOSE"
    """
    opens = load_open_positions()
    closed = load_closed_trades()
    new_opens = []

    for o in opens:
        if o is row:
            # PnL yüzde
            if row["dir"]=="UP":
                pnl_pct = (exit_px-row["entry"])/row["entry"]*100 if row["entry"] else 0.0
            else:
                pnl_pct = (row["entry"]-exit_px)/row["entry"]*100 if row["entry"] else 0.0

            closed.append({
                "symbol": row["symbol"],
                "dir": row["dir"],
                "type": "SCALP",
                "entry": row.get("entry"),
                "exit": exit_px,
                "result": reason,
                "pnl_pct": pnl_pct,
                "time_open": row.get("time_open"),
                "time_close": now_iso(),
                "mode": row.get("mode","?"),
                "power": row.get("power"),
                "tier": row.get("tier"),
                "rsi": row.get("rsi"),
                "ang_now": row.get("ang_now"),
                "ang_change": row.get("ang_change")
            })

            tg_send(
                f"📘 CLOSE {row['symbol']} SCALP {row['dir']} [{row.get('mode','?')}]\n"
                f"Exit:{exit_px:.6f} {reason} {pnl_pct:.2f}%"
            )
        else:
            new_opens.append(o)

    save_open_positions(new_opens)
    save_closed_trades(closed)

def sync_real_positions():
    """
    Binance hedge pozisyonlarını oku ve open_positions.json içindeki
    "real" kayıtlarıyla sync et:
      - Varsa güncelle (entry fiyatını yenile)
      - Binance'te var ama local yoksa ekle
      - Binance'te yoksa ama local'de varsa kapat (SYNC_CLOSE)
    """
    live = futures_fetch_positions()
    live_keyset = set()
    for p in live:
        direction = "UP" if p["side"]=="LONG" else "DOWN"
        live_keyset.add((p["symbol"], direction))

    opens = load_open_positions()

    # Güncelle / ekle
    for p in live:
        direction = "UP" if p["side"]=="LONG" else "DOWN"
        key = (p["symbol"], direction)
        exists = False
        for o in opens:
            if o.get("mode")=="real" and (o["symbol"],o["dir"])==key:
                exists = True
                o["entry"] = p["entry"]
        if not exists:
            opens.append({
                "symbol": p["symbol"],
                "dir": direction,
                "type": "SCALP",
                "entry": p["entry"],
                "tp": None,
                "sl": None,
                "time_open": now_iso(),
                "mode": "real",
                "power": None,
                "tier": None,
                "rsi": None,
                "ang_now": None,
                "ang_change": None
            })

    # Binance'te yoksa local kapat
    for o in list(opens):
        if o.get("mode")=="real":
            k=(o["symbol"],o["dir"])
            if k not in live_keyset:
                cur_px = futures_get_price(o["symbol"])
                if cur_px is None:
                    cur_px = o.get("entry",0)
                close_local_position(o, "SYNC_CLOSE", cur_px)

    save_open_positions(load_open_positions())  # refresh

def try_local_tp_sl_hits():
    """
    Failsafe:
    Fiyat TP/SL'e vurduysa ve biz hala local açık listede tutuyorsak
    local olarak kapat (Binance'e ekstra emir atmaz).
    """
    opens = load_open_positions()
    if not opens:
        return
    for row in list(opens):
        tp = row.get("tp")
        sl = row.get("sl")
        if tp is None or sl is None:
            continue
        px = futures_get_price(row["symbol"])
        if px is None:
            continue

        hit_tp = False
        hit_sl = False
        if row["dir"]=="UP":
            hit_tp = (px >= tp)
            hit_sl = (px <= sl)
        else:
            hit_tp = (px <= tp)
            hit_sl = (px >= sl)

        if hit_tp:
            close_local_position(row, "TP", px)
        elif hit_sl:
            close_local_position(row, "SL", px)

# ================= SOFT LIMIT & SIMULATE SYNC =================
def count_real_long_short():
    """
    Binance hedge pozisyonlarına göre kaç LONG / SHORT var?
    """
    pos = futures_fetch_positions()
    long_cnt = sum(1 for p in pos if p["side"]=="LONG")
    short_cnt= sum(1 for p in pos if p["side"]=="SHORT")
    return long_cnt, short_cnt, pos

def enforce_limits_autotrade_soft():
    """
    MAX_BUY / MAX_SELL kontrolü.
    - LONG yön limit üstüne çıktıysa auto_trade_long=False
    - SHORT yön limit üstüne çıktıysa auto_trade_short=False
    - Limit altına inince yön yeniden açılır.

    Ek davranış:
    - Her iki yön de kapalıysa simulate=True
    - En az bir yön açıksa simulate=False
    """
    long_cnt, short_cnt, livepos = count_real_long_short()
    changed = False

    # LONG taraf
    if long_cnt >= PARAM["MAX_BUY"]:
        if STATE.get("auto_trade_long", True):
            STATE["auto_trade_long"] = False
            tg_send("⛔ BUY limit doldu → LONG yönü durduruldu.")
            changed = True
    else:
        if not STATE.get("auto_trade_long", True):
            STATE["auto_trade_long"] = True
            tg_send("✅ BUY limiti altında → LONG yönü yeniden aktif.")
            changed = True

    # SHORT taraf
    if short_cnt >= PARAM["MAX_SELL"]:
        if STATE.get("auto_trade_short", True):
            STATE["auto_trade_short"] = False
            tg_send("⛔ SELL limit doldu → SHORT yönü durduruldu.")
            changed = True
    else:
        if not STATE.get("auto_trade_short", True):
            STATE["auto_trade_short"] = True
            tg_send("✅ SELL limiti altında → SHORT yönü yeniden aktif.")
            changed = True

    both_off = (not STATE["auto_trade_long"]) and (not STATE["auto_trade_short"])
    if both_off and not STATE["simulate"]:
        STATE["simulate"] = True
        tg_send("🧠 Her iki yön kapalı → Simulate ON (sadece veri topluyoruz).")
        changed = True
    elif (not both_off) and STATE["simulate"]:
        STATE["simulate"] = False
        tg_send("💸 En az bir yön aktif → Simulate OFF (gerçek emir mümkün).")
        changed = True

    if changed:
        safe_save(STATE_FILE, STATE)

    return (STATE["auto_trade_long"] or STATE["auto_trade_short"])

# ================= SCALP SIGNAL ENGINE =================
def build_scalp_signal(sym, kl1, last_scalp_seen, bar_index, params):
    """
    SCALP sinyali:
    EMA7 eğim yön değiştirince tetiklenir.
    slope_prev<0 & slope_now>0 => UP
    slope_prev>0 & slope_now<0 => DOWN

    cooldown:
      aynı yönde aynı coinde tekrar spam atmasın diye SCALP_COOLDOWN_BARS

    volatility filter:
      24h change % |x| >= VOLATILITY_LIMIT => SKIP
    """
    # volatility filter (günlük değişim çok büyükse atla)
    chg_24h = futures_24h_change(sym)
    if abs(chg_24h) >= params["VOLATILITY_LIMIT"]:
        log(f"[SKIP VOLATILE] {sym} {chg_24h:.2f}% >= {params['VOLATILITY_LIMIT']}%")
        return None, None, None

    closes=[float(k[4]) for k in kl1]
    ema7_1=ema(closes,7)
    if len(ema7_1)<6:
        return None, None, None

    slope_now  = ema7_1[-1]-ema7_1[-4]
    slope_prev = ema7_1[-2]-ema7_1[-5]

    if slope_prev<0 and slope_now>0:
        direction="UP"
    elif slope_prev>0 and slope_now<0:
        direction="DOWN"
    else:
        return None, None, None

    scalp_key = f"{sym}_{direction}"

    # cooldown bar
    last_idx=last_scalp_seen.get(scalp_key)
    if last_idx is not None:
        if (bar_index - last_idx) <= params["SCALP_COOLDOWN_BARS"]:
            return None, None, None

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

    tier, color = tier_from_power(pwr, params)
    # ULTRA her zaman kabul,
    # PREMIUM sadece ONLY_ULTRA_TRADES <1.0 ise kabul
    if not filter_ultra(tier, params):
        return None, None, None

    ang_now  = slope_angle_deg(slope_now2, atr_now)
    ang_dif  = angle_between_deg(slope_prev2, slope_now2, atr_now)

    entry = futures_get_price(sym)
    if not entry:
        return None, None, None

    if direction=="UP":
        tp = entry*(1+params["SCALP_TP_PCT"])
        sl = entry*(1-params["SCALP_SL_PCT"])
    else:
        tp = entry*(1-params["SCALP_TP_PCT"])
        sl = entry*(1+params["SCALP_SL_PCT"])

    sig = {
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
        "time": now_iso()
    }

    return sig, scalp_key, bar_index

# ================= EXECUTION LOGIC =================
def execute_signal(sig, filters_cache=None):
    """
    Sinyal objesi:
    {
        "symbol":...,
        "dir":"UP"/"DOWN",
        "type":"SCALP",
        "entry":...,
        "tp":...,
        "sl":...,
        "time":...,
        ...
    }
    """

    symbol    = sig["symbol"]
    direction = sig["dir"]      # "UP" => LONG/BUY, "DOWN" => SHORT/SELL

    # Limitleri ve simulate durumunu güncelle
    enforce_limits_autotrade_soft()

    # Duplicate guard
    if already_open_same_direction(symbol, direction):
        log(f"[SKIP DUP] {symbol} {direction} zaten açık")
        return

    # Global autotrade OFF ise -> simulate zorla
    if not STATE.get("auto_trade", True):
        STATE["simulate"] = True
        safe_save(STATE_FILE, STATE)

    # Simulate modundaysak asla gerçek emir atma
    if STATE["simulate"]:
        record_open(sig, "sim")
        tg_send(
            f"📒 SIM TRADE {symbol} SCALP {direction}\n"
            f"entry={sig['entry']:.6f} tp={sig['tp']:.6f} sl={sig['sl']:.6f}\n"
            f"Pow:{sig.get('power','?'):.1f} Tier:{sig.get('tier','?')}"
        )
        return

    # Yön bazlı kontrol
    if direction=="UP" and not STATE.get("auto_trade_long", True):
        tg_send(f"🚫 LONG yönü kapalı ({symbol} skip).")
        return
    if direction=="DOWN" and not STATE.get("auto_trade_short", True):
        tg_send(f"🚫 SHORT yönü kapalı ({symbol} skip).")
        return

    # Miktar hesapla
    qty, f, live_px_for_qty = calc_order_quantity(symbol, PARAM["TRADE_SIZE_USDT"], filters_cache)
    if not qty or qty <= 0:
        record_open(sig, "sim")
        tg_send(f"❌ qty hesaplanamadı → {symbol} SIM olarak kaydedildi.")
        return

    side     = "BUY"  if direction=="UP"   else "SELL"
    pos_side = "LONG" if direction=="UP"   else "SHORT"

    try:
        futures_market_order(symbol, side, qty, pos_side)
        futures_set_tp_sl(symbol, side, pos_side, qty, sig["tp"], sig["sl"], filters_cache)

        tg_send(
            f"💸 REAL TRADE {symbol} {direction} SCALP\n"
            f"qty={qty} {side} {pos_side}\n"
            f"entry≈{sig['entry']:.6f} tp={sig['tp']:.6f} sl={sig['sl']:.6f}\n"
            f"Pow:{sig.get('power','?'):.1f} Tier:{sig.get('tier','?')}"
        )
        record_open(sig, "real")

    except Exception as e:
        record_open(sig, "sim")
        tg_send(f"❌ REAL TRADE ERR {symbol}\n{e}\nSim olarak kaydedildi.")
        log(f"[REAL TRADE ERR] {e}")

# ================= STATUS / REPORT =================
def maybe_status_report():
    """
    Her 10 dakikada bir durum raporu.
    """
    now_sec = int(time.time())
    last_sent = STATE.get("last_status_sent", 0)
    if now_sec - last_sent < 600:  # 10 dakika
        return

    long_cnt, short_cnt, livepos = count_real_long_short()

    opens_local  = load_open_positions()
    closed_local = load_closed_trades()

    wins_local = [c for c in closed_local if c.get("result")=="TP"]
    winrate_local = (len(wins_local)/len(closed_local)*100.0) if closed_local else 0.0

    tg_send(
        "📊 STATUS RAPORU\n"
        f"Simulate: {'✅' if STATE['simulate'] else '❌'}\n"
        f"LONG aktif : {'✅' if STATE.get('auto_trade_long',True) else '❌'}  ({long_cnt}/{int(PARAM['MAX_BUY'])})\n"
        f"SHORT aktif: {'✅' if STATE.get('auto_trade_short',True) else '❌'} ({short_cnt}/{int(PARAM['MAX_SELL'])})\n"
        f"Open(local): {len(opens_local)} | Closed(local): {len(closed_local)} | Winrate(local): {winrate_local:.1f}%\n"
        f"auto_trade global: {'✅' if STATE.get('auto_trade',True) else '❌'}\n"
        f"TradeSize: {PARAM['TRADE_SIZE_USDT']} USDT\n"
        f"VOLATILITY_LIMIT: {PARAM.get('VOLATILITY_LIMIT')}\n"
        f"Time: {now_iso()}"
    )

    STATE["last_status_sent"] = now_sec
    safe_save(STATE_FILE, STATE)

def build_daily_summary_payload():
    closed_list = load_closed_trades()
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
    today_str = (datetime.now(timezone.utc)+timedelta(hours=3)).strftime("%Y-%m-%d")
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

    STATE["last_daily_sent_date"] = today_str
    safe_save(STATE_FILE, STATE)

# ================= TELEGRAM COMMANDS =================
def tg_poll_commands(last_update_id):
    """
    Desteklenen komutlar:
      /status
      /params
      /set KEY VALUE  veya  /set KEY=VALUE
      /autotrade on|off
      /simulate on|off
      /report
      /export closed
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
            long_cnt, short_cnt, livepos = count_real_long_short()
            opens_local  = load_open_positions()
            closed_local = load_closed_trades()
            wins_local   = [c for c in closed_local if c.get("result")=="TP"]
            winrate_local= (len(wins_local)/len(closed_local)*100.0) if closed_local else 0.0
            tg_send(
                "🤖 STATUS\n"
                f"Simulate:  {'✅' if STATE['simulate'] else '❌'}\n"
                f"LONG aktif : {'✅' if STATE.get('auto_trade_long',True) else '❌'}  ({long_cnt}/{int(PARAM['MAX_BUY'])})\n"
                f"SHORT aktif: {'✅' if STATE.get('auto_trade_short',True) else '❌'} ({short_cnt}/{int(PARAM['MAX_SELL'])})\n"
                f"auto_trade global: {'✅' if STATE.get('auto_trade',True) else '❌'}\n"
                f"TradeSize: {PARAM['TRADE_SIZE_USDT']} USDT\n"
                f"VOLATILITY_LIMIT: {PARAM.get('VOLATILITY_LIMIT')}\n"
                f"Open(local): {len(opens_local)} | Closed(local): {len(closed_local)} | Winrate(local): {winrate_local:.1f}%\n"
                f"Time: {now_iso()}"
            )

        elif lower == "/params":
            pretty = []
            for k,v in PARAM.items():
                pretty.append(f"{k} = {v}")
            pretty_text = "\n".join(pretty)
            tg_send(
                "🔧 Aktif Parametreler:\n"
                f"{pretty_text}\n"
                f"\nSimulate: {'✅' if STATE['simulate'] else '❌'}\n"
                f"LONG: {'✅' if STATE.get('auto_trade_long',True) else '❌'} / "
                f"SHORT: {'✅' if STATE.get('auto_trade_short',True) else '❌'}\n"
                f"auto_trade global: {'✅' if STATE.get('auto_trade',True) else '❌'}"
            )

        elif lower.startswith("/autotrade"):
            parts = lower.split()
            if len(parts)==2 and parts[1] in ("on","off"):
                if parts[1]=="on":
                    STATE["auto_trade"]=True
                    STATE["simulate"]=False
                    tg_send("🔓 AutoTrade ON → Simulate OFF (gerçek emir izni)")
                else:
                    STATE["auto_trade"]=False
                    STATE["simulate"]=True
                    tg_send("🔒 AutoTrade OFF → Simulate ON (sadece kayıt)")
                safe_save(STATE_FILE, STATE)
            else:
                tg_send("kullanım: /autotrade on | /autotrade off")

        elif lower.startswith("/simulate"):
            parts = lower.split()
            if len(parts)==2 and parts[1] in ("on","off"):
                if parts[1]=="on":
                    STATE["simulate"]=True
                    tg_send("📝 Simulate ON (sadece veri topluyoruz)")
                else:
                    STATE["simulate"]=False
                    tg_send("📝 Simulate OFF (gerçek mod, limitlere tabi)")
                safe_save(STATE_FILE, STATE)
            else:
                tg_send("kullanım: /simulate on | /simulate off")

        elif lower == "/report":
            summary = build_daily_summary_payload()
            tg_send(
                "📊 Rapor (manual)\n"
                f"Toplam Trade: {summary['total']}\n"
                f"Winrate: {summary['winrate']:.1f}%\n"
                f"AvgPnL: {summary['avg_pnl']:.2f}%"
            )

        elif lower.startswith("/export"):
            parts = lower.split()
            if len(parts)==2 and parts[1]=="closed":
                tg_send_document(CLOSED_TRADES_FILE, caption="📊 closed_trades.json")
            else:
                tg_send("Kullanım: /export closed")

        elif lower.startswith("/set"):
            # /set KEY VALUE  veya  /set KEY=VALUE
            raw = text.replace("\n"," ").strip()

            # parse
            if " " in raw and "=" not in raw:
                parts = raw.split()
                if len(parts) != 3:
                    tg_send("kullanım:\n/set KEY VALUE\nya da\n/set KEY=VALUE")
                    last_update_id = uid
                    continue
                key_txt = parts[1]
                val_txt = parts[2]
            else:
                try:
                    after = raw.split(" ",1)[1]
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

            # aliaslar
            if key_txt in ("SIZE","TRADE_SIZE","TRADE_SIZE_USDT"):
                key_txt = "TRADE_SIZE_USDT"
            if key_txt in ("MAXBUY","MAX_BUY"):
                key_txt = "MAX_BUY"
            if key_txt in ("MAXSELL","MAX_SELL"):
                key_txt = "MAX_SELL"
            if key_txt in ("ULTRA","ONLY_ULTRA","ONLY_ULTRA_TRADES"):
                key_txt = "ONLY_ULTRA_TRADES"
            if key_txt in ("SCALPTP","SCALP_TP","SCALP_TP_PCT"):
                key_txt = "SCALP_TP_PCT"
            if key_txt in ("SCALPSL","SCALP_SL","SCALP_SL_PCT"):
                key_txt = "SCALP_SL_PCT"
            if key_txt in ("COOLDOWN","SCALP_COOLDOWN","SCALP_COOLDOWN_BARS"):
                key_txt = "SCALP_COOLDOWN_BARS"
            if key_txt in ("VOL","VOLATILITY","VOLATILITY_LIMIT"):
                key_txt = "VOLATILITY_LIMIT"

            if key_txt not in PARAM:
                tg_send(f"❌ bilinmeyen param: {key_txt}")
                last_update_id = uid
                continue

            # sayı parse
            try:
                if key_txt in ("MAX_BUY","MAX_SELL","SCALP_COOLDOWN_BARS"):
                    new_val = int(val_txt)
                else:
                    new_val = float(val_txt)
            except:
                tg_send(f"❌ {key_txt} değeri sayı olmalı (gönderdiğin: {val_txt})")
                last_update_id = uid
                continue

            PARAM[key_txt] = new_val
            safe_save(PARAM_FILE, PARAM)
            tg_send(f"⚙️ {key_txt} = {PARAM[key_txt]} olarak ayarlandı")

            # limit güncellendiyse, hemen tekrar değerlendir
            if key_txt in ("MAX_BUY","MAX_SELL"):
                enforce_limits_autotrade_soft()

        last_update_id = uid

    return last_update_id

# ================= MAIN LOOP =================
def main():
    tg_send("🚀 EMA ULTRA v13.6.2 SCALP-ONLY başladı (1H + Volatility Filter + SoftLimit + TrueMarket)")
    # sembolleri al
    exinfo = futures_exchange_info()
    symbols = [s["symbol"] for s in exinfo
               if s.get("quoteAsset")=="USDT"
               and s.get("status")=="TRADING"]
    symbols.sort()

    last_update_id = 0

    while True:
        # 1) Telegram komutlarını dinle
        last_update_id = tg_poll_commands(last_update_id)

        # 2) Binance pozisyonlarını local'e senkronize et
        sync_real_positions()

        # 3) Local pozisyonların TP/SL tetiklenip tetiklenmediğini kontrol et (failsafe)
        try_local_tp_sl_hits()

        # 4) Soft limit + simulate restore mantığını uygula
        enforce_limits_autotrade_soft()

        # 5) Periyodik status raporu
        maybe_status_report()

        # 6) Bar sayacı
        STATE["bar_index"] += 1
        bar_i = STATE["bar_index"]

        # 7) Sinyalleri tara (SCALP ONLY, 1h verisi)
        for sym in symbols:
            kl1 = futures_get_klines(sym, "1h", 200)
            if len(kl1) < 60:
                continue

            scalp_sig, scalp_key, scalp_bar_idx = build_scalp_signal(
                sym,
                kl1,
                STATE["last_scalp_seen"],
                bar_i,
                PARAM
            )

            if scalp_sig:
                tg_send(
                    f"{scalp_sig.get('color','')} {scalp_sig.get('tier','')} "
                    f"SCALP {sym} {scalp_sig['dir']}\n"
                    f"Pow:{scalp_sig.get('power','?'):.1f}\n"
                    f"Entry:{scalp_sig['entry']:.6f} TP:{scalp_sig['tp']:.6f} SL:{scalp_sig['sl']:.6f}"
                )

                execute_signal(scalp_sig)

                if scalp_key is not None:
                    STATE["last_scalp_seen"][scalp_key] = scalp_bar_idx

            time.sleep(0.08)

        # 8) Günlük rapor
        maybe_daily_summary()

        # 9) Persist state/params
        safe_save(STATE_FILE, STATE)
        safe_save(PARAM_FILE, PARAM)

        # 10) Loop bekle
        time.sleep(120)

# ================= ENTRY =================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        tg_send(f"❗FATAL ema.py: {e}")