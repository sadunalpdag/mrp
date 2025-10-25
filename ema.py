# ==============================================================================
# 📘 EMA ULTRA v15.0 — SCALP ONLY + APPROVE + AutoTrade + RL Hooks
# ==============================================================================
# - SCALP ONLY (EMA7 slope reversal)
# - APPROVE_BARS bekleme sistemi:
#     * Sinyal doğar => pending
#     * APPROVE_BARS kadar bar bekle
#     * Hala uygun ise pozisyon aç
# - DuplicateGuard++:
#     * Aynı symbol+dir pending varken yeni pending yaratma
#     * Zaten açık pozisyon varsa tekrar açma
# - AutoTrade:
#     * Binance Futures (hedge mode)
#     * TRADE_SIZE_USDT büyüklüğünde market order aç
#     * TP (%0.6) & SL (%20) otomatik gir
# - MAX_BUY / MAX_SELL limitleri gerçek açık işlemlere göre kontrol edilir
# - 4h AutoReport: kapalı işlemler ve AI log dosyaları Telegram'a atılır
# - SafeSave (atomic write .tmp)
#
# ÇALIŞMASI İÇİN ENV:
#   BOT_TOKEN, CHAT_ID
#   BINANCE_API_KEY, BINANCE_SECRET_KEY
#
# PARAMLAR (params.json ile override edilebilir):
#   {
#     "SCALP_TP_PCT": 0.006,
#     "SCALP_SL_PCT": 0.20,
#     "TRADE_SIZE_USDT": 250.0,
#     "MAX_BUY": 30,
#     "MAX_SELL": 30,
#     "APPROVE_BARS": 1
#   }
# ==============================================================================

import os, json, time, math, requests, hmac, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
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
    """atomic save, render / termux friendly"""
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
    except: pass

def now_ts_ms(): 
    return int(datetime.now(timezone.utc).timestamp()*1000)

def now_local_iso():
    # UTC+3 human readable
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

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
    """
    Low-level signed request (GET or POST)
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
    LOT_SIZE / PRICE_FILTER info
    """
    try:
        info = requests.get(
            BINANCE_FAPI+"/fapi/v1/exchangeInfo",
            timeout=10
        ).json()
        s = next((x for x in info["symbols"] if x["symbol"]==symbol),None)
        lot = next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef = next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        return {
            "stepSize":float(lot.get("stepSize","1")),
            "tickSize":float(pricef.get("tickSize","0.01"))
        }
    except:
        return {"stepSize":1.0,"tickSize":0.01}

def round_nearest(x, step):
    # round to nearest multiple of step
    return round(round(x/step)*step, 8)

def calc_order_qty(symbol, entry_price, notional_usdt):
    """
    TRADE_SIZE_USDT / entry_price -> qty
    Snap to LOT_SIZE step.
    """
    f = get_symbol_filters(symbol)
    raw = notional_usdt / max(entry_price,1e-9)
    qty = round_nearest(raw, f["stepSize"])
    return qty

def open_market_position(symbol, direction, qty):
    """
    direction: "UP"  -> BUY  positionSide=LONG
               "DOWN"-> SELL positionSide=SHORT
    returns executed price if success
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

    # try to read fill price
    fills = res.get("avgPrice") or res.get("price") or None
    try:
        price_filled = float(fills) if fills else futures_get_price(symbol)
    except:
        price_filled = futures_get_price(symbol)

    return {
        "symbol":symbol,
        "positionSide":position_side,
        "qty":qty,
        "entry_price":price_filled
    }

def futures_set_tp_sl(symbol, direction, qty, entry_price, tp_pct, sl_pct):
    """
    Attach TP/SL reduce-only orders for this side.
    direction UP -> long, close side SELL
    direction DOWN -> short, close side BUY
    """
    position_side = "LONG" if direction=="UP" else "SHORT"
    close_side    = "SELL" if direction=="UP" else "BUY"

    if direction=="UP":
        tp_price = entry_price*(1+tp_pct)
        sl_price = entry_price*(1-sl_pct)
    else:
        tp_price = entry_price*(1-tp_pct)
        sl_price = entry_price*(1+sl_pct)

    f = get_symbol_filters(symbol)
    tick = f["tickSize"]

    tp_s = round_nearest(tp_price, tick)
    sl_s = round_nearest(sl_price, tick)

    for ttype,price in [
        ("TAKE_PROFIT_MARKET",tp_s),
        ("STOP_MARKET",sl_s)
    ]:
        try:
            payload = {
                "symbol":symbol,
                "side":close_side,
                "type":ttype,
                "stopPrice":f"{price:.8f}",
                "quantity":f"{qty}",
                "positionSide":position_side,
                "workingType":"MARK_PRICE",
                "timestamp":now_ts_ms()
            }
            _signed_request("POST","/fapi/v1/order",payload)
        except Exception as e:
            log(f"[TP/SL ERR] {e}")

def fetch_open_positions_real():
    """
    Reads real positions from Binance to enforce MAX_BUY / MAX_SELL.
    Returns dict:
      {
        "long": { "SYMBOL": qty_float, ... },
        "short":{ "SYMBOL": qty_float, ... },
        "long_count": total nonzero longs,
        "short_count": total nonzero shorts
      }
    """
    result = {"long":{}, "short":{},"long_count":0,"short_count":0}
    try:
        payload={"timestamp":now_ts_ms()}
        acc = _signed_request("GET","/fapi/v2/positionRisk",payload)
        # Only count non-zero positions
        for p in acc:
            sym = p["symbol"]
            pos_amt = float(p["positionAmt"])
            side = "long" if pos_amt>0 else "short" if pos_amt<0 else None
            if side=="long":
                result["long"][sym]=abs(pos_amt)
            elif side=="short":
                result["short"][sym]=abs(pos_amt)
        result["long_count"]=len([1 for v in result["long"].values() if v>0])
        result["short_count"]=len([1 for v in result["short"].values() if v>0])
    except Exception as e:
        log(f"[FETCH POS ERR] {e}")
    return result

# ================= PARAMS / STATE =================
PARAM_DEFAULT = {
    "SCALP_TP_PCT":     0.006,   # 0.6% TP
    "SCALP_SL_PCT":     0.20,    # 20% SL
    "TRADE_SIZE_USDT":  250.0,
    "MAX_BUY":          30,
    "MAX_SELL":         30,
    "APPROVE_BARS":     1
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index":        0,
    "last_report":      0,
    "last_scalp_seen":  {},   # "SYMBOL_DIR" -> born_bar last seen
    "pending":          []    # list of {symbol,dir,born_bar,approve_at_bar,entry,tp,sl,power,...}
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
if "last_scalp_seen" not in STATE:
    STATE["last_scalp_seen"]={}
if "pending" not in STATE:
    STATE["pending"]=[]

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
    # same logic as v14.x
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
    EMA7 slope reversal:
    s_prev < 0 and s_now > 0  => UP (long bias)
    s_prev > 0 and s_now < 0  => DOWN (short bias)
    """
    chg = futures_24h_change(sym)
    closes=[float(k[4]) for k in kl]
    e7=ema(closes,7)

    s_now  = e7[-1]-e7[-4]
    s_prev = e7[-2]-e7[-5]

    if s_prev<0 and s_now>0:
        direction="UP"
    elif s_prev>0 and s_now<0:
        direction="DOWN"
    else:
        return None

    highs=[float(k[2])for k in kl]
    lows =[float(k[3])for k in kl]
    atr_v = atr_like(highs,lows,closes)[-1]
    r_val = rsi(closes)[-1]

    pwr = calc_power(
        e7[-1],   # e7
        e7[-2],   # e7p
        e7[-5],   # e7p2
        atr_v,
        closes[-1],
        r_val
    )

    tier, emoji = tier_from_power(pwr)
    if not tier:
        return None

    entry = futures_get_price(sym)
    if entry is None:
        return None

    if direction=="UP":
        tp = entry*(1+PARAM["SCALP_TP_PCT"])
        sl = entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp = entry*(1-PARAM["SCALP_TP_PCT"])
        sl = entry*(1+PARAM["SCALP_SL_PCT"])

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
                continue
            if sig:
                found.append(sig)
    return found

# ================= PENDING / APPROVAL =================
def add_pending(sig, approve_bars):
    """
    pending'e ekle ama DuplicateGuard:
    - aynı symbol+dir zaten pending varsa yenisini ekleme
    """
    for p in STATE["pending"]:
        if p["symbol"]==sig["symbol"] and p["dir"]==sig["dir"]:
            # already waiting approval
            return

    new_item = dict(sig)
    new_item["approve_at_bar"] = sig["born_bar"] + approve_bars
    STATE["pending"].append(new_item)

    tg_send(
        f"⏳ PENDING {sig['emoji']} {sig['tier']} {sig['symbol']} {sig['dir']}\n"
        f"Pow:{sig['power']:.1f} RSI:{sig['rsi']:.1f} ATR:{sig['atr']:.4f} 24hΔ:{sig['chg24h']:.2f}%\n"
        f"Entry:{sig['entry']:.6f}\nTP:{sig['tp']:.6f}\nSL:{sig['sl']:.6f}\n"
        f"ApproveAtBar:{new_item['approve_at_bar']} born:{sig['born_bar']}"
    )

def approve_and_trade(current_bar):
    """
    PENDING listesini gez:
    - approve_at_bar <= current_bar ise (bekleme süresi dolduysa)
      trade açmayı dene
    - MAX_BUY / MAX_SELL kontrolü yap
    - aynı yönde zaten açık pozisyon varsa pas geç
    - başarılı açılanları pending'den düş
    """
    if not STATE["pending"]:
        return

    live = fetch_open_positions_real()

    still_pending=[]
    for p in STATE["pending"]:
        # beklemesi bitmemişse tekrar pending'te bırak
        if current_bar < p["approve_at_bar"]:
            still_pending.append(p)
            continue

        # limit kontrol:
        if p["dir"]=="UP":
            if live["long_count"] >= PARAM["MAX_BUY"]:
                tg_send(f"🚫 SKIP {p['symbol']} UP - MAX_BUY limit")
                still_pending.append(p)
                continue
            if p["symbol"] in live["long"]:
                tg_send(f"🚫 SKIP {p['symbol']} UP - already LONG")
                continue
        else:
            if live["short_count"] >= PARAM["MAX_SELL"]:
                tg_send(f"🚫 SKIP {p['symbol']} DOWN - MAX_SELL limit")
                still_pending.append(p)
                continue
            if p["symbol"] in live["short"]:
                tg_send(f"🚫 SKIP {p['symbol']} DOWN - already SHORT")
                continue

        # qty hesapla
        qty = calc_order_qty(p["symbol"], p["entry"], PARAM["TRADE_SIZE_USDT"])
        if not qty or qty<=0:
            tg_send(f"❗ {p['symbol']} qty hesaplanamadı.")
            continue

        # market order aç
        try:
            opened = open_market_position(p["symbol"], p["dir"], qty)
        except Exception as e:
            tg_send(f"❌ OPEN ERR {p['symbol']} {p['dir']} {e}")
            log(f"[OPEN ERR] {e}")
            continue

        entry_exec = opened["entry_price"]

        # tp/sl emirlerini gir
        try:
            futures_set_tp_sl(
                p["symbol"],
                p["dir"],
                qty,
                entry_exec,
                PARAM["SCALP_TP_PCT"],
                PARAM["SCALP_SL_PCT"]
            )
        except Exception as e:
            tg_send(f"⚠ TP/SL ERR {p['symbol']} {e}")
            log(f"[TP/SL ERR] {e}")

        tg_send(
            f"✅ OPENED {p['symbol']} {p['dir']} qty:{qty}\n"
            f"EntryFill:{entry_exec:.6f}\n"
            f"TP%:{PARAM['SCALP_TP_PCT']*100:.3f} "
            f"SL%:{PARAM['SCALP_SL_PCT']*100:.1f}\n"
            f"bar:{current_bar}"
        )

        # RL hook log
        rl_log = safe_load(AI_RL_FILE, [])
        rl_log.append({
            "time":now_local_iso(),
            "symbol":p["symbol"],
            "dir":p["dir"],
            "entry":entry_exec,
            "tp_pct":PARAM["SCALP_TP_PCT"],
            "sl_pct":PARAM["SCALP_SL_PCT"],
            "power":p["power"],
            "bar_opened":current_bar
        })
        safe_save(AI_RL_FILE, rl_log)

    STATE["pending"] = still_pending
# ================= REPORTING / BACKUP =================
def maybe_auto_report():
    """
    Her 4 saatte bir kritik dosyaları Telegram'a yollar
    """
    now_ts = time.time()
    if now_ts - STATE.get("last_report",0) < 14400:
        return

    for fpath in [AI_SIGNALS_FILE, AI_ANALYSIS_FILE, AI_RL_FILE, CLOSED_TRADES_FILE]:
        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")

    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"] = now_ts
    safe_save(STATE_FILE, STATE)

# ================= MAIN LOOP =================
def main():
    tg_send("🚀 EMA ULTRA v15.0 başladı (SCALP ONLY + APPROVE + AutoTrade + 4h Report)")

    # Binance'ten USDT çiftleri çek
    info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo").json()
    symbols = [
        s["symbol"]
        for s in info["symbols"]
        if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"
    ]
    symbols.sort()

    while True:
        try:
            # her döngü bar index +1
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1️⃣ yeni scalp sinyallerini tara
            sigs = run_parallel(symbols, bar_i)

            for s in sigs:
                key = f"{s['symbol']}_{s['dir']}"
                # DuplicateGuard: aynı bar'da aynı yönde sinyal yollama
                if STATE["last_scalp_seen"].get(key)==bar_i:
                    continue
                STATE["last_scalp_seen"][key]=bar_i

                tg_send(
                    f"{s['emoji']} {s['tier']} {s['symbol']} {s['dir']}\n"
                    f"Pow:{s['power']:.1f} RSI:{s['rsi']:.1f} ATR:{s['atr']:.4f} 24hΔ:{s['chg24h']:.2f}%\n"
                    f"Entry:{s['entry']:.6f}\nTP:{s['tp']:.6f}\nSL:{s['sl']:.6f}\n"
                    f"born_bar:{s['born_bar']}"
                )

                # Pending listesine ekle
                add_pending(s, PARAM["APPROVE_BARS"])

            # 2️⃣ beklemesi dolan pending'leri trade'e çevir
            approve_and_trade(bar_i)

            # 3️⃣ 4 saatlik auto backup kontrolü
            maybe_auto_report()

            # 4️⃣ state kaydet
            safe_save(STATE_FILE, STATE)

            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR] {e}")
            time.sleep(10)

# run
if __name__=="__main__":
    main()
