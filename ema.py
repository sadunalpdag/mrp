# ==============================================================================
# 📘 EMA ULTRA v15.5 — Dynamic AutoTrade Recovery
# ==============================================================================
# Bu sürüm:
#  - EMA7 slope reversal sinyali + ANGLE filter (slope impulse >= ANGLE_MIN)
#  - 24h volatilite filtresi: |chg24h| >= 10% ise sinyal yok
#  - APPROVE_BARS bekleme (pending -> approve)
#  - Dinamik AutoTrade:
#       * ULTRA sinyal approve olduğunda:
#           - Eğer auto_trade_active == True ve limitler aşılmadıysa
#             => GERÇEK emir açılır (market + TP/SL)
#           - Aksi halde => sadece simülasyona yazılır
#       * PREMIUM / NORMAL => her zaman simülasyon
#  - auto_trade_active runtime'da açılıp kapanır:
#       * Eğer long_count >= MAX_BUY veya short_count >= MAX_SELL => kapat
#       * Sonra pozisyon sayısı limit altına inince => otomatik tekrar aç
#  - Sim işlemler Telegram'a gönderilmiyor (sessiz)
#  - Sim kapanışları da Telegram'a gönderilmiyor (sessiz)
#  - TP/SL precision fix hem real emirlerde hem sim tarafında var
#  - RL log (ai_rl_log.json) sadece GERÇEK açılan ULTRA emirlerde yazılıyor
#  - ai_signals.json, ai_analysis.json, closed_trades.json, open_positions.json korunuyor
#  - 4 saatte bir AutoReport 4 dosyayı Telegram'a atar
#  - status_report açılışta atılır
#  - Hiçbir önceki özellik kaldırılmadı, sadece logic düzenlendi.
#
# PARAMS (params.json override edebilir):
#   {
#     "SCALP_TP_PCT": 0.006,
#     "SCALP_SL_PCT": 0.20,
#     "TRADE_SIZE_USDT": 250.0,
#     "MAX_BUY": 30,
#     "MAX_SELL": 30,
#     "APPROVE_BARS": 1,
#     "ULTRA_ONLY_TRADE": true,
#     "ANGLE_MIN": 0.0001
#   }
# ==============================================================================

import os, json, time, requests, hmac, hashlib, threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ================= PATHS =================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE          = os.path.join(DATA_DIR, "state.json")
PARAM_FILE          = os.path.join(DATA_DIR, "params.json")
OPEN_POS_FILE       = os.path.join(DATA_DIR, "open_positions.json")
CLOSED_TRADES_FILE  = os.path.join(DATA_DIR, "closed_trades.json")
AI_SIGNALS_FILE     = os.path.join(DATA_DIR, "ai_signals.json")
AI_ANALYSIS_FILE    = os.path.join(DATA_DIR, "ai_analysis.json")
AI_RL_FILE          = os.path.join(DATA_DIR, "ai_rl_log.json")
LOG_FILE            = os.path.join(DATA_DIR, "log.txt")

# ================= ENV VARS =================
BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")

BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()

# ================= HELPERS =================
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
        with SAVE_LOCK:
            tmp = path + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
    except Exception as e:
        print(f"[SAVE ERR] {e}", flush=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms():
    return int(datetime.now(timezone.utc).timestamp()*1000)

def now_local_iso():
    # UTC+3 readable
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def enforce_max_file_size(path, max_mb=10):
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
    Get klines and drop last candle if it's still forming in the future.
    """
    try:
        r = requests.get(
            BINANCE_FAPI+"/fapi/v1/klines",
            params={"symbol":symbol,"interval":interval,"limit":limit},
            timeout=10
        ).json()
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        if r and int(r[-1][6])>now_ms:
            r = r[:-1]
        return r
    except:
        return []

def get_symbol_filters(symbol):
    """
    LOT_SIZE / PRICE_FILTER info.
    fallback tickSize=1e-8, stepSize=1.0 minimum.
    """
    try:
        info = requests.get(
            BINANCE_FAPI+"/fapi/v1/exchangeInfo",
            timeout=10
        ).json()
        s = next((x for x in info["symbols"] if x["symbol"]==symbol),None)
        lot = next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef = next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        tickSize=float(pricef.get("tickSize","1e-8"))
        stepSize=float(lot.get("stepSize","1"))
        return {
            "stepSize": stepSize,
            "tickSize": tickSize
        }
    except:
        # fallback güvenli
        return {"stepSize":1.0,"tickSize":1e-8}

def round_nearest(x, step):
    if step == 0:
        return x
    return round(round(x/step)*step, 12)

def adjust_precision(symbol, value, mode="price"):
    """
    mode="price": tickSize
    mode="qty":   stepSize

    ayrıca 0 veya - altına düşmeyi engelliyoruz.
    """
    f = get_symbol_filters(symbol)
    step = f["tickSize"] if mode=="price" else f["stepSize"]
    adj  = round_nearest(value, step)
    adj  = max(adj, step)  # asla 0 veya negatif dönme
    return float(f"{adj:.12f}")

def calc_order_qty(symbol, entry_price, notional_usdt):
    raw = notional_usdt / max(entry_price,1e-12)
    return adjust_precision(symbol, raw, "qty")

def open_market_position(symbol, direction, qty):
    """
    Gerçek hedge pozisyon açar. ULTRA sinyal için kullanılır.
    """
    side          = "BUY"  if direction=="UP" else "SELL"
    position_side = "LONG" if direction=="UP" else "SHORT"

    payload = {
        "symbol":symbol,
        "side":side,
        "type":"MARKET",
        "quantity":f"{qty}",
        "positionSide":position_side,
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
        "positionSide":position_side,
        "qty":qty,
        "entry_price":entry_exec
    }

def futures_set_tp_sl(symbol, direction, qty, entry_price, tp_pct, sl_pct):
    """
    TP/SL reduce-only orderları koyar. 'Stop price less than zero' fix'li.
    """
    position_side = "LONG" if direction=="UP" else "SHORT"
    close_side    = "SELL" if direction=="UP" else "BUY"

    if direction=="UP":
        tp_price = entry_price*(1+tp_pct)
        sl_price = entry_price*(1-sl_pct)
    else:
        tp_price = entry_price*(1-tp_pct)
        sl_price = entry_price*(1+sl_pct)

    if tp_price <= 0 or sl_price <= 0:
        tg_send(f"⚠ SKIP TP/SL {symbol}: invalid raw tp={tp_price} sl={sl_price}")
        return

    tp_s = adjust_precision(symbol, tp_price, "price")
    sl_s = adjust_precision(symbol, sl_price, "price")

    if tp_s <= 0 or sl_s <= 0:
        tg_send(f"⚠ SKIP TP/SL {symbol}: rounded to zero ({tp_s}/{sl_s})")
        return

    for ttype,price in [
        ("TAKE_PROFIT_MARKET",tp_s),
        ("STOP_MARKET",sl_s)
    ]:
        try:
            payload = {
                "symbol":symbol,
                "side":close_side,
                "type":ttype,
                "stopPrice":f"{price:.12f}",
                "quantity":f"{qty}",
                "positionSide":position_side,
                "workingType":"MARK_PRICE",
                "reduceOnly":"true",
                "timestamp":now_ts_ms()
            }
            _signed_request("POST","/fapi/v1/order",payload)
        except Exception as e:
            tg_send(f"⚠ TP/SL ERR {symbol} {e}")
            log(f"[TP/SL ERR] {symbol} {e}")

def fetch_open_positions_real():
    """
    Gerçek hedge pozisyonlarını çeker.
    Kullanım:
      - status_report
      - dinamik AutoTrade ON/OFF
      - MAX_BUY / MAX_SELL guard
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

# ================= PARAMS / STATE / STORAGE =================
PARAM_DEFAULT = {
    "SCALP_TP_PCT":     0.006,   # 0.6% TP
    "SCALP_SL_PCT":     0.20,    # 20% SL
    "TRADE_SIZE_USDT":  250.0,
    "MAX_BUY":          30,
    "MAX_SELL":         30,
    "APPROVE_BARS":     1,
    "ULTRA_ONLY_TRADE": True,
    "ANGLE_MIN":        0.0001   # EMA slope impulse eşiği
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index":        0,
    "last_report":      0,
    "last_scalp_seen":  {},
    "pending":          [],
    "auto_trade_active": True  # Dinamik olarak açılıp kapanır
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
if "last_scalp_seen" not in STATE:
    STATE["last_scalp_seen"]={}
if "pending" not in STATE:
    STATE["pending"]=[]
if "auto_trade_active" not in STATE:
    STATE["auto_trade_active"]=True

OPEN_POS   = safe_load(OPEN_POS_FILE, [])
CLOSED_LOG = safe_load(CLOSED_TRADES_FILE, [])
AI_SIGNALS = safe_load(AI_SIGNALS_FILE, [])
AI_ANALYSIS= safe_load(AI_ANALYSIS_FILE, [])
AI_RL      = safe_load(AI_RL_FILE, [])

# ================= STATS / REPORT HELPERS =================
def compute_local_stats():
    open_count  = len(OPEN_POS)
    closed_count= len(CLOSED_LOG)
    wins        = sum(1 for c in CLOSED_LOG if c.get("result")=="WIN")
    winrate     = (wins/closed_count*100.0) if closed_count>0 else 0.0
    return open_count, closed_count, winrate

def status_report(live_real=None):
    if live_real is None:
        live_real = fetch_open_positions_real()

    o,c,w = compute_local_stats()

    txt = []
    txt.append(f"AutoTradeActive(State): {'✅' if STATE.get('auto_trade_active',True) else '⛔'}")
    txt.append(f"REAL Long: {live_real['long_count']} / {PARAM['MAX_BUY']}")
    txt.append(f"REAL Short:{live_real['short_count']} / {PARAM['MAX_SELL']}")
    txt.append(f"TradeSize: {PARAM['TRADE_SIZE_USDT']} USDT")
    txt.append(f"UltraOnly: {1.0 if PARAM.get('ULTRA_ONLY_TRADE',True) else 0.0}")
    txt.append(f"Open(local): {o} | Closed(local): {c} | Winrate(local): {w:.1f}%")
    txt.append(f"Time: {now_local_iso()}")

    tg_send("\n".join(txt))

# ================= INDICATORS =================
def ema(vals,n):
    k=2/(n+1)
    e=[vals[0]]
    for v in vals[1:]:
        e.append(v*k+e[-1]*(1-k))
    return e

def rsi(vals,period=14):
    if len(vals)<period+1:
        return [50]*len(vals)
    d=np.diff(vals)
    g=np.maximum(d,0)
    l=-np.minimum(d,0)
    ag=np.mean(g[:period])
    al=np.mean(l[:period])
    rs=ag/al if al>0 else 0
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

def calc_power(e7,e7p,e7p2,atr,price,rsi_val):
    diff=abs(e7-e7p)/(atr*0.6) if atr>0 else 0
    base=55+diff*20+((rsi_val-50)/50)*15+(atr/price)*200
    score=min(100,max(0,base))
    return score

def tier_from_power(p):
    if p>=75:   return "ULTRA","🟩"
    elif p>=68: return "PREMIUM","🟦"
    elif p>=60: return "NORMAL","🟨"
    return None,""

# ================= SIGNAL BUILD =================
def build_scalp_signal(sym, kl, bar_i):
    """
    EMA7 slope reversal + ANGLE filter + daily %10 filter.
    """

    closes=[float(k[4]) for k in kl]
    if len(closes) < 10:
        return None

    chg = futures_24h_change(sym)
    if abs(chg) >= 10.0:
        return None  # aşırı volatil coin = yasak

    e7=ema(closes,7)
    if len(e7)<6:
        return None

    s_now  = e7[-1]-e7[-4]
    s_prev = e7[-2]-e7[-5]

    # yön flip kontrolü
    if s_prev<0 and s_now>0:
        direction="UP"
    elif s_prev>0 and s_now<0:
        direction="DOWN"
    else:
        return None

    # angle / momentum filtresi
    slope_impulse = abs(s_now - s_prev)
    if slope_impulse < PARAM.get("ANGLE_MIN", 0.0001):
        return None

    highs=[float(k[2])for k in kl]
    lows =[float(k[3])for k in kl]
    atr_v = atr_like(highs,lows,closes)[-1]
    r_val = rsi(closes)[-1]

    pwr = calc_power(
        e7[-1],
        e7[-2],
        e7[-5],
        atr_v,
        closes[-1],
        r_val
    )

    tier, emoji = tier_from_power(pwr)
    if not tier:
        return None

    entry_raw = futures_get_price(sym)
    if entry_raw is None:
        return None

    # TP / SL bruto
    if direction=="UP":
        tp_raw = entry_raw*(1+PARAM["SCALP_TP_PCT"])
        sl_raw = entry_raw*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_raw = entry_raw*(1-PARAM["SCALP_TP_PCT"])
        sl_raw = entry_raw*(1+PARAM["SCALP_SL_PCT"])

    # Precision normalize (sim ve real aynı hedef görsün)
    entry = adjust_precision(sym, entry_raw, "price")
    tp    = adjust_precision(sym, tp_raw,    "price")
    sl    = adjust_precision(sym, sl_raw,    "price")

    sig = {
        "symbol":sym,
        "dir":direction,
        "tier":tier,
        "emoji":emoji,
        "entry":entry,
        "tp":tp,
        "sl":sl,
        "power":pwr,
        "rsi":r_val,
        "atr":atr_v,
        "chg24h":chg,
        "time":now_local_iso(),
        "born_bar":bar_i
    }
    return sig

# ================= PARALLEL SCAN =================
def scan_symbol(sym, bar_i):
    kl = futures_get_klines(sym,"1h",200)
    if len(kl)<60:
        return None
    return build_scalp_signal(sym,kl,bar_i)

def run_parallel(symbols, bar_i):
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

# ================= AI LOGGING HELPERS =================
def ai_log_signal(sig):
    AI_SIGNALS.append({
        "time":sig["time"],
        "symbol":sig["symbol"],
        "dir":sig["dir"],
        "tier":sig["tier"],
        "power":sig["power"],
        "rsi":sig["rsi"],
        "atr":sig["atr"],
        "tp":sig["tp"],
        "sl":sig["sl"],
        "entry":sig["entry"]
    })
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)
    enforce_max_file_size(AI_SIGNALS_FILE)

def ai_update_analysis():
    stats={}
    for t in CLOSED_LOG:
        tier=t.get("tier","?")
        res=t.get("result")
        if tier not in stats:
            stats[tier]={"WIN":0,"LOSS":0,"TOTAL":0}
        stats[tier]["TOTAL"]+=1
        if res=="WIN": stats[tier]["WIN"]+=1
        elif res=="LOSS": stats[tier]["LOSS"]+=1

    snapshot={
        "time":now_local_iso(),
        "by_tier":[
            {
                "tier":tier,
                "total":v["TOTAL"],
                "win":v["WIN"],
                "loss":v["LOSS"],
                "winrate_pct": (v["WIN"]/v["TOTAL"]*100.0) if v["TOTAL"]>0 else 0.0
            }
            for tier,v in stats.items()
        ]
    }
    AI_ANALYSIS.append(snapshot)
    safe_save(AI_ANALYSIS_FILE, AI_ANALYSIS)
    enforce_max_file_size(AI_ANALYSIS_FILE)

# ================= SIM ENGINE =================
def record_sim_open(sig, bar_i):
    """
    Simülasyon pozisyonu aç.
    Premium/Normal her zaman buraya gider.
    ULTRA da buraya gider eğer auto_trade_active False ise
    veya limitlerden dolayı gerçek emir atlanıyorsa.
    Bu fonksiyon Telegram'a mesaj ATMAYACAK.
    """
    pos = {
        "symbol":sig["symbol"],
        "dir":sig["dir"],
        "tier":sig["tier"],
        "entry":sig["entry"],
        "tp":sig["tp"],
        "sl":sig["sl"],
        "time_open":now_local_iso(),
        "bar_open":bar_i,
        "closed":False
    }
    OPEN_POS.append(pos)
    safe_save(OPEN_POS_FILE, OPEN_POS)
    enforce_max_file_size(OPEN_POS_FILE)

def sim_has_open(symbol, direction):
    for p in OPEN_POS:
        if (not p.get("closed")) and p["symbol"]==symbol and p["dir"]==direction:
            return True
    return False

def check_sim_closes():
    """
    Sim pozisyonları TP/SL vurdu mu?
    Sessiz (Telegram yok).
    """
    global OPEN_POS
    changed=False
    price_cache={}
    still=[]

    for p in OPEN_POS:
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
        if p["dir"]=="UP":
            if last_price>=p["tp"]:
                hit="WIN"
            elif last_price<=p["sl"]:
                hit="LOSS"
        else:
            if last_price<=p["tp"]:
                hit="WIN"
            elif last_price>=p["sl"]:
                hit="LOSS"

        if hit:
            CLOSED_LOG.append({
                "symbol":sym,
                "dir":p["dir"],
                "tier":p["tier"],
                "entry":p["entry"],
                "exit_price":last_price,
                "tp":p["tp"],
                "sl":p["sl"],
                "time_open":p["time_open"],
                "time_close":now_local_iso(),
                "result":hit
            })
            changed=True
        else:
            still.append(p)

    OPEN_POS=still
    safe_save(OPEN_POS_FILE, OPEN_POS)
    safe_save(CLOSED_TRADES_FILE, CLOSED_LOG)
    enforce_max_file_size(OPEN_POS_FILE)
    enforce_max_file_size(CLOSED_TRADES_FILE)

    if changed:
        ai_update_analysis()

# ================= PENDING / APPROVAL =================
def add_pending(sig, approve_bars):
    """
    Sinyali pending'e koy.
    DuplicateGuard: aynı symbol+dir zaten pending ise ekleme.
    Telegram'a PENDING uyarısı atılır (bunu koruyoruz).
    """
    for p in STATE["pending"]:
        if p["symbol"]==sig["symbol"] and p["dir"]==sig["dir"]:
            return

    new_item = dict(sig)
    new_item["approve_at_bar"] = sig["born_bar"] + approve_bars
    STATE["pending"].append(new_item)

    tg_send(
        f"⏳ PENDING {sig['emoji']} {sig['tier']} {sig['symbol']} {sig['dir']}\n"
        f"Pow:{sig['power']:.1f} RSI:{sig.get('rsi',0):.1f} ATR:{sig.get('atr',0):.4f} 24hΔ:{sig.get('chg24h',0):.2f}%\n"
        f"Entry:{sig['entry']:.12f}\nTP:{sig['tp']:.12f}\nSL:{sig['sl']:.12f}\n"
        f"ApproveAtBar:{new_item['approve_at_bar']} born:{sig['born_bar']}"
    )

def approve_and_execute(current_bar):
    """
    Dinamik AutoTrade yönetimi + pozisyon açma.
    - Önce hedge pozisyonları oku
    - AutoTrade'i kapat/aç (STATE["auto_trade_active"])
    - Pending sinyalleri sırayla işle
    """

    live = fetch_open_positions_real()

    # --- Dynamic AutoTrade state machine ---
    # Eğer limitteysek kapat
    if STATE.get("auto_trade_active", True):
        if (live["long_count"] >= PARAM["MAX_BUY"]) or (live["short_count"] >= PARAM["MAX_SELL"]):
            STATE["auto_trade_active"] = False
            tg_send(
                f"🚫 AutoTrade durduruldu — limit aşıldı "
                f"(long:{live['long_count']}/{PARAM['MAX_BUY']} "
                f"short:{live['short_count']}/{PARAM['MAX_SELL']})"
            )
    else:
        # Kapalıyken tekrar limit altına düştüyse yeniden aç
        if (live["long_count"] < PARAM["MAX_BUY"]) and (live["short_count"] < PARAM["MAX_SELL"]):
            STATE["auto_trade_active"] = True
            tg_send(
                f"✅ AutoTrade yeniden aktif "
                f"(long:{live['long_count']}/{PARAM['MAX_BUY']} "
                f"short:{live['short_count']}/{PARAM['MAX_SELL']})"
            )

    still_pending=[]

    for p in STATE["pending"]:
        if current_bar < p["approve_at_bar"]:
            still_pending.append(p)
            continue

        symbol = p["symbol"]
        d      = p["dir"]
        tier   = p["tier"]

        # DuplicateGuard sim tarafı
        if sim_has_open(symbol, d):
            # zaten açık aynı yönde sim var => atla
            continue

        # ULTRA ise, ve AutoTrade aktifse, ve limitler uygunsa => gerçek emir
        can_real = (
            tier == "ULTRA"
            and STATE.get("auto_trade_active", True)
        )

        if can_real:
            # Yine de ekstra güvenlik: aynı yönde gerçek pozisyon zaten var mı?
            if d=="UP":
                if (symbol in live["long"]):
                    can_real = False
            else:
                if (symbol in live["short"]):
                    can_real = False

        if can_real:
            # Gerçek emir aç
            qty = calc_order_qty(symbol, p["entry"], PARAM["TRADE_SIZE_USDT"])
            if not qty or qty <= 0:
                tg_send(f"❗ {symbol} qty hesaplanamadı.")
                continue

            try:
                opened = open_market_position(symbol, d, qty)
                entry_exec = opened["entry_price"]

                futures_set_tp_sl(
                    symbol,
                    d,
                    qty,
                    entry_exec,
                    PARAM["SCALP_TP_PCT"],
                    PARAM["SCALP_SL_PCT"]
                )

                tg_send(
                    f"✅ REAL TRADE {symbol} {d} ({tier}) qty:{qty}\n"
                    f"EntryFill:{entry_exec:.12f}\n"
                    f"TP%:{PARAM['SCALP_TP_PCT']*100:.3f} "
                    f"SL%:{PARAM['SCALP_SL_PCT']*100:.1f}\n"
                    f"bar:{current_bar}"
                )

                # RL log'a yaz
                AI_RL.append({
                    "time":now_local_iso(),
                    "symbol":symbol,
                    "dir":d,
                    "entry":entry_exec,
                    "power":p["power"],
                    "bar_opened":current_bar
                })
                safe_save(AI_RL_FILE, AI_RL)
                enforce_max_file_size(AI_RL_FILE)

            except Exception as e:
                tg_send(f"❌ OPEN ERR {symbol} {d} {e}")
                log(f"[OPEN ERR] {symbol} {e}")
                # hata olduysa, yedek olarak sim'e yazalım:
                record_sim_open(p, current_bar)

        else:
            # sim aç (sessiz)
            record_sim_open(p, current_bar)

    STATE["pending"] = still_pending

# ================= REPORTING / BACKUP =================
def ai_update_and_report_if_due():
    """
    Her 4 saatte bir:
    - ai_analysis.json güncelle
    - dört dosyayı Telegram'a upload
    - STATE["last_report"] güncelle
    """
    now_ts = time.time()
    if now_ts - STATE.get("last_report",0) < 14400:
        return

    ai_update_analysis()

    for fpath in [AI_SIGNALS_FILE, AI_ANALYSIS_FILE, AI_RL_FILE, CLOSED_TRADES_FILE]:
        enforce_max_file_size(fpath)

    tg_send_file(AI_SIGNALS_FILE,   "📊 AutoBackup ai_signals.json")
    tg_send_file(AI_ANALYSIS_FILE,  "📊 AutoBackup ai_analysis.json")
    tg_send_file(AI_RL_FILE,        "📊 AutoBackup ai_rl_log.json")
    tg_send_file(CLOSED_TRADES_FILE,"📊 AutoBackup closed_trades.json")

    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"] = now_ts
    safe_save(STATE_FILE, STATE)

# ================= MAIN LOOP =================
def main():
    tg_send("🚀 EMA ULTRA v15.5 başladı (Dynamic AutoTrade Recovery)")
    status_report()

    # Binance'ten USDT paritelerini çek
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols = [
            s["symbol"]
            for s in info["symbols"]
            if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"
        ]
        symbols.sort()
    except Exception as e:
        log(f"[SYMBOL FETCH ERR] {e}")
        symbols=[]

    while True:
        try:
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1) sinyalleri tara
            sigs = run_parallel(symbols, bar_i)

            for s in sigs:
                # DuplicateGuard: aynı bar aynı yönde spamlama
                key = f"{s['symbol']}_{s['dir']}"
                if STATE["last_scalp_seen"].get(key)==bar_i:
                    continue
                STATE["last_scalp_seen"][key]=bar_i

                # sinyali Telegram'a bildiriyoruz (bu kalsın)
                tg_send(
                    f"{s['emoji']} {s['tier']} {s['symbol']} {s['dir']}\n"
                    f"Pow:{s['power']:.1f} RSI:{s['rsi']:.1f} ATR:{s.get('atr',0):.4f} 24hΔ:{s.get('chg24h',0):.2f}%\n"
                    f"Entry:{s['entry']:.12f}\nTP:{s['tp']:.12f}\nSL:{s['sl']:.12f}\n"
                    f"born_bar:{s['born_bar']}"
                )

                # sinyali AI loguna yaz
                ai_log_signal(s)

                # pending'e ekle (onay bekle)
                add_pending(s, PARAM["APPROVE_BARS"])

            # 2) onay zamanı gelenleri işle (gerçek veya sim)
            approve_and_execute(bar_i)

            # 3) sim pozisyonlar TP/SL vurdu mu? (sessiz)
            check_sim_closes()

            # 4) 4h oto rapor
            ai_update_and_report_if_due()

            # 5) state kaydet
            safe_save(STATE_FILE, STATE)

            # 6) diski koru
            enforce_max_file_size(LOG_FILE)

            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR] {e}")
            time.sleep(10)

# run
if __name__=="__main__":
    main()
