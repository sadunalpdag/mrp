# ======================== EMA ULTRA v15.9.65 — FULL MODE =======================
# trade_rules: STABLE TP/SL v15.9.51 (Clean Mode) — ❌ reduceOnly YOK (asla değişmeyecek)
# KC: Kıvanç Confirm (no TP), 4 LONG + 4 SHORT iç limit + Global 30/30
# Stratejiler: EARLY, SCALP, UT-STC, MACD, FVG (Power 65–75, Smart TP/SL Clean)
# MR: exit-only | Hedge | TrendLock 6h | RL/AI logs | Telegram komutları | CSV rapor
# ==============================================================================

import os, json, time, requests, hmac, hashlib, threading, math, random, csv
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Files
STATE_FILE        = os.path.join(DATA_DIR, "state.json")
PARAM_FILE        = os.path.join(DATA_DIR, "params.json")
LOG_FILE          = os.path.join(DATA_DIR, "log.txt")
KIVANC_FILE       = os.path.join(DATA_DIR, "kivanc_open.json")
AI_SIGNALS_FILE   = os.path.join(DATA_DIR, "ai_signals.json")
AI_ANALYSIS_FILE  = os.path.join(DATA_DIR, "ai_analysis.json")
AI_RL_FILE        = os.path.join(DATA_DIR, "ai_rl_log.json")
SIM_POS_FILE      = os.path.join(DATA_DIR, "sim_positions.json")
SIM_CLOSED_FILE   = os.path.join(DATA_DIR, "sim_closed.json")
CLOSED_CSV_FILE   = os.path.join(REPORT_DIR, "closed_trades.csv")

# ENV
BOT_TOKEN      = os.getenv("BOT_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"
TIME_OFFSET_H  = int(os.getenv("TIME_OFFSET_H", "3"))

SAVE_LOCK = threading.Lock()
getcontext().prec = 28
random.seed(42)

# Defaults
STATE_DEFAULT = {
    "bar_index": 0,
    "last_report_day": None,
    "closed_count": 0
}
PARAM_DEFAULT = {
    "TRADE_SIZE_USDT": 250.0,
    "POWER_MIN": 65.0, "POWER_MAX": 75.0,
    "TP_USD_MIN": 1.6, "TP_USD_MAX": 2.0,
    "FALLBACK_TP_PCT": 0.0020, "FALLBACK_SL_PCT": 0.0200,
    "GLOBAL_LONG_CAP": 30, "GLOBAL_SHORT_CAP": 30,
    "KC_LONG_CAP": 4, "KC_SHORT_CAP": 4,
    "SIM_MODE": False,                  # ✅ simülasyon anahtarı
    "CSV_EXPORT": True,
    "TELEMETRY": True
}

# Guards
TREND_LOCK, TREND_LOCK_TIME = {}, {}
TRENDLOCK_EXPIRY_SEC = 6 * 3600

# ------------------------------- Utils/IO -------------------------------------
def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

def safe_load(p, dflt):
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return dflt

def safe_save(p, obj):
    try:
        with SAVE_LOCK:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p)
    except Exception as e:
        log(f"[SAVE ERR] {e}")

def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=TIME_OFFSET_H)).replace(microsecond=0).isoformat()

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": t}, timeout=10)
    except: pass

def ensure_csv_header():
    if not os.path.exists(CLOSED_CSV_FILE):
        with open(CLOSED_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time","symbol","dir","qty","entry","exit","pnl","kind"])

# ----------------------------- Indicators -------------------------------------
def ema(vals, n):
    if not vals: return []
    k = 2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+2: return [50]*len(vals)
    d=np.diff(vals); g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:period]); al=np.mean(l[:period]); out=[50]*period
    for i in range(period, len(d)):
        ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0; out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out)) + out

def macd(vals, fast=12, slow=26, signal=9):
    ef=ema(vals,fast); es=ema(vals,slow)
    macd_line=np.array(ef)-np.array(es)
    signal_ln=ema(macd_line.tolist(),signal)
    hist=macd_line-np.array(signal_ln)
    return macd_line.tolist(), signal_ln, hist.tolist()

def atr_like(highs, lows, closes, period=14):
    tr=[]
    for i in range(len(highs)):
        if i==0: tr.append(highs[i]-lows[i])
        else:
            tr.append(max(highs[i]-lows[i],
                          abs(highs[i]-closes[i-1]),
                          abs(lows[i]-closes[i-1])))
    if len(tr)<period: return [0]*len(highs)
    a=[sum(tr[:period])/period]
    for i in range(period, len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(highs)-len(a)) + a

def supertrend_last(highs,lows,closes,period=10,mult=3.0):
    atr=atr_like(highs,lows,closes,period)
    mid=(np.array(highs)+np.array(lows))/2.0
    up=mid+mult*np.array(atr); lo=mid-mult*np.array(atr); dir_up=True
    for i in range(1, len(closes)):
        if closes[i] > up[i-1]: dir_up=True
        elif closes[i] < lo[i-1]: dir_up=False
        if dir_up: up[i]=max(up[i], closes[i])
        else:      lo[i]=min(lo[i], closes[i])
    st_val = up[-1] if dir_up else lo[-1]
    st_dir = "UP" if dir_up else "DOWN"
    return st_val, st_dir

# ---------------------------- Precision/Exchange ------------------------------
PRECISION_CACHE = {}

def _decimals_from_tick(tick_str):
    try:
        d = Decimal(str(tick_str))
        return max(0, -d.as_tuple().exponent)
    except:
        s = str(tick_str)
        if "." in s: return len(s.split(".")[1])
        return 0

def format_price_by_tick(sym, price_float):
    f = get_symbol_filters(sym)
    dec = _decimals_from_tick(f["tickSize"])
    p_dec = Decimal(str(price_float)).quantize(Decimal(f"1e-{dec}"), rounding=ROUND_HALF_UP)
    if p_dec == Decimal("-0"): p_dec = Decimal("0")
    return f"{float(p_dec):.{dec}f}"

def round_to_tick(sym, price_float):
    f = get_symbol_filters(sym)
    t = Decimal(str(f["tickSize"]))
    p = Decimal(str(price_float))
    if t <= 0: return float(p)
    q = (p/t).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(q*t)

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE: return PRECISION_CACHE[sym]
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo", timeout=10).json()
        s = next((x for x in info["symbols"] if x["symbol"]==sym), None)
        lot = next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"), {})
        pricef = next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"), {})
        PRECISION_CACHE[sym] = {
            "stepSize": float(lot.get("stepSize","1")),
            "tickSize": float(pricef.get("tickSize","0.01")),
            "minPrice": float(pricef.get("minPrice","0.00000001")),
            "maxPrice": float(pricef.get("maxPrice","100000000"))
        }
    except Exception as e:
        log(f"[PREC WARN]{sym} {e}")
        PRECISION_CACHE[sym] = {"stepSize":0.0001,"tickSize":0.0001,"minPrice":1e-8,"maxPrice":1e8}
    return PRECISION_CACHE[sym]

def calc_order_qty(sym, entry_price, usd):
    f = get_symbol_filters(sym)
    step = Decimal(str(f["stepSize"]))
    raw = Decimal(str(usd)) / Decimal(max(entry_price,1e-12))
    q = (raw/step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step
    return float(q)

def _signed_request(method, path, params):
    q = "&".join([f"{k}={params[k]}" for k in params])
    sig = hmac.new(BINANCE_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + q + "&signature=" + sig
    r = requests.post(url, headers=headers, timeout=10) if method == "POST" else requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def futures_get_price(sym):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                         params={"symbol":sym}, timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym, interval, limit):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                         params={"symbol":sym,"interval":interval,"limit":limit},
                         timeout=10).json()
        if r and int(r[-1][6]) > now_ts_ms(): r = r[:-1]   # son bar kapanmamışsa
        return r
    except:
        return []
def auto_init_symbols():
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo", timeout=10).json()
        symbols = [s["symbol"] for s in info["symbols"]
                   if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR] {e}")
        symbols = []
    symbols.sort()
    return symbols
# ------------------------------ Load state/params ------------------------------
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict): PARAM = PARAM_DEFAULT
for k,v in PARAM_DEFAULT.items(): PARAM.setdefault(k,v)

STATE = safe_load(STATE_FILE, STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

# ------------------------------ RL / AI logging -------------------------------
def ai_log_signal(sig):
    arr = safe_load(AI_SIGNALS_FILE, [])
    arr.append({**sig, "ts": now_ts_ms()})
    safe_save(AI_SIGNALS_FILE, arr)

def ai_log_analysis(entry):
    arr = safe_load(AI_ANALYSIS_FILE, [])
    entry["ts"] = now_ts_ms()
    arr.append(entry)
    safe_save(AI_ANALYSIS_FILE, arr)

def ai_log_rl(event, payload):
    arr = safe_load(AI_RL_FILE, [])
    arr.append({"t": now_local_iso(), "e": event, "p": payload})
    safe_save(AI_RL_FILE, arr)

def csv_append_closed(time_iso, symbol, direction, qty, entry, exitp, pnl, kind):
    if not PARAM.get("CSV_EXPORT", True): return
    ensure_csv_header()
    with open(CLOSED_CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([time_iso, symbol, direction, qty, f"{entry:.12f}", f"{exitp:.12f}", f"{pnl:.6f}", kind])

# ------------------------------ Telegram bot ----------------------------------
def tg_parse(text):
    parts = text.strip().split()
    return (parts[0].lower(), parts[1:]) if parts else ("", [])

def tg_handle_cmd(cmd, args):
    if cmd == "/status":
        L,S = _count_positions_all()
        msg = (f"📊 STATUS\n"
               f"Power: {PARAM['POWER_MIN']}-{PARAM['POWER_MAX']}\n"
               f"KC cap: {PARAM['KC_LONG_CAP']}/{PARAM['KC_SHORT_CAP']}  Global: {PARAM['GLOBAL_LONG_CAP']}/{PARAM['GLOBAL_SHORT_CAP']}\n"
               f"SIM_MODE: {PARAM['SIM_MODE']} CSV: {PARAM['CSV_EXPORT']}\n"
               f"Open L/S: {L}/{S}")
        tg_send(msg)

    elif cmd == "/param":
        # /param KEY VALUE
        if len(args) >= 2:
            key = args[0].upper()
            val = " ".join(args[1:])
            try:
                if key in PARAM:
                    if val.lower() in ("true","false"):
                        PARAM[key] = (val.lower()=="true")
                    else:
                        PARAM[key] = float(val) if val.replace(".","",1).isdigit() else val
                    safe_save(PARAM_FILE, PARAM)
                    tg_send(f"✅ PARAM updated: {key} = {PARAM[key]}")
                else:
                    tg_send(f"❌ Unknown key: {key}")
            except Exception as e:
                tg_send(f"❌ PARAM error: {e}")
        else:
            tg_send("ℹ️ Usage: /param KEY VALUE")

    elif cmd == "/export":
        # gönderim özetini bildir
        cnt = safe_load(STATE_FILE, STATE).get("closed_count", 0)
        tg_send(f"📤 Export ready. Closed trades: {cnt}\nPath: {CLOSED_CSV_FILE}")

    elif cmd == "/open" and len(args) >= 2:
        # /open SYMBOL UP|DOWN [usd]
        sym = args[0].upper(); d = args[1].upper()
        usd = float(args[2]) if len(args)>=3 else PARAM["TRADE_SIZE_USDT"]
        entry = futures_get_price(sym)
        if not entry: tg_send("❌ price err"); return
        qty = calc_order_qty(sym, entry, usd)
        try:
            fill = open_market(sym, d, qty)
            tp, sl = smart_tp_sl_prices(sym, fill, d)
            place_exit_orders(sym, d, qty, tp, sl)
            tg_send(f"🟦 MANUAL OPEN {sym} {d} qty:{qty}\nEntry:{format_price_by_tick(sym,fill)}\nTP:{format_price_by_tick(sym,tp)} SL:{format_price_by_tick(sym,sl)}")
        except Exception as e:
            tg_send(f"❌ open err: {e}")

    elif cmd == "/close" and len(args) >= 2:
        # /close SYMBOL LONG|SHORT qty
        sym = args[0].upper(); ps = args[1].upper()
        qty = float(args[2]) if len(args)>=3 else None
        try:
            if qty is None:
                pr = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
                rec = next((p for p in pr if p.get("symbol")==sym), None)
                if not rec: tg_send("❌ position not found"); return
                amt = float(rec.get("positionAmt",0))
                if ps=="LONG": qty = max(0.0, amt)
                else:          qty = abs(min(0.0, amt))
            # close by market
            side = "SELL" if ps=="LONG" else "BUY"
            _signed_request("POST","/fapi/v1/order",{
                "symbol": sym, "side": side, "type": "MARKET",
                "quantity": f"{qty}", "positionSide": ps, "timestamp": now_ts_ms()
            })
            tg_send(f"✅ MANUAL CLOSE {sym} {ps} qty:{qty}")
        except Exception as e:
            tg_send(f"❌ close err: {e}")

    elif cmd == "/positions":
        try:
            pr = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
            lines = ["📗 POSITIONS"]
            for p in pr:
                amt = float(p.get("positionAmt", 0))
                if abs(amt) < 1e-12:
                    continue
                sym = p.get("symbol")
                ep = float(p.get("entryPrice", 0))
                mp = futures_get_price(sym) or 0.0
                    
                pnl = (mp-ep)*amt
                lines.append(f"{sym} amt:{amt:.6f} EP:{ep:.6f} MP:{mp:.6f} PnL:{pnl:.4f}")
            tg_send("\n".join(lines))
        except Exception as e:
            tg_send(f"❌ positions err: {e}")

# (basit polling — webhook kullanmıyoruz)
def tg_poll_loop():
    if not BOT_TOKEN or not CHAT_ID: return
    offset = None
    while True:
        try:
            r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                             params={"offset": offset or 0, "timeout": 20}, timeout=25).json()
            for up in r.get("result", []):
                offset = up["update_id"] + 1
                msg = up.get("message") or up.get("edited_message") or {}
                text = (msg.get("text") or "").strip()
                if not text: continue
                cmd, args = tg_parse(text)
                tg_handle_cmd(cmd, args)
        except Exception as e:
            log(f"[TG POLL ERR] {e}")
            time.sleep(5)

# ------------------------- Kıvanç open-set (4/4 gerçek) -----------------------
def _kiv_load(): return safe_load(KIVANC_FILE, [])
def _kiv_save(lst): safe_save(KIVANC_FILE, lst)

def _kiv_refresh_from_positions():
    lst = _kiv_load()
    try:
        acc = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[KIV REF ERR] {e}"); return lst
    alive = []
    for it in lst:
        sym, direction = it["symbol"], it["dir"]
        pr = next((p for p in acc if p.get("symbol")==sym), None)
        if not pr: continue
        amt = float(pr.get("positionAmt",0))
        if direction=="UP" and amt>0: alive.append(it)
        if direction=="DOWN" and amt<0: alive.append(it)
    _kiv_save(alive)
    return alive
# ------------------------------ Power & Helpers --------------------------------
def compute_power(entry, st_val, r_val):
    dist_pct = abs(entry - st_val) / max(entry, 1e-9) * 100.0
    return 60.0 + dist_pct + (r_val - 50.0) / 2.0

def in_power_band(p): return PARAM["POWER_MIN"] <= p <= PARAM["POWER_MAX"]

def ema_cross_dir(closes, fast=9, slow=30):
    if len(closes) < slow+3: return None
    ef = ema(closes, fast); es = ema(closes, slow)
    up = (ef[-1] > es[-1]) and (ef[-2] <= es[-2])
    dn = (ef[-1] < es[-1]) and (ef[-2] >= es[-2])
    if up: return "UP"
    if dn: return "DOWN"
    return None

# ------------------------------ Kıvanç Confirm --------------------------------
def build_kivanc_confirm_signal(sym, kl, bar_i):
    if len(kl) < 120: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    st, sd = supertrend_last(h,l,c,10,3.0)
    cd = ema_cross_dir(c, 9, 30)
    if not cd or not sd or cd != sd: return None
    r = rsi(c,14)[-1]; entry = c[-1]
    pwr = compute_power(entry, st, r)
    sig = {"symbol":sym,"dir":cd,"entry":entry,"kind":"KIVANC_CONFIRM",
           "power":pwr,"time":now_local_iso(),"born_bar":bar_i}
    ai_log_signal(sig)
    return sig

# ------------------------------ EARLY -----------------------------------------
def build_early_signal(sym, kl, bar_i):
    if len(kl) < 50: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    e3=ema(c,3); e7=ema(c,7); atrv=atr_like(h,l,c,14)[-1]
    if atrv<=0: return None
    body=abs(c[-1]-c[-2]); spike=body/max(atrv,1e-9)
    if e3[-1]>e7[-1] and e3[-2]<=e7[-2] and spike>=0.03:
        st,_=supertrend_last(h,l,c); pwr=compute_power(c[-1],st,rsi(c,14)[-1])
        sig={"symbol":sym,"dir":"UP","entry":c[-1],"kind":"EARLY","power":pwr,"born_bar":bar_i}
        ai_log_signal(sig); return sig
    if e3[-1]<e7[-1] and e3[-2]>=e7[-2] and spike>=0.03:
        st,_=supertrend_last(h,l,c); pwr=compute_power(c[-1],st,rsi(c,14)[-1])
        sig={"symbol":sym,"dir":"DOWN","entry":c[-1],"kind":"EARLY","power":pwr,"born_bar":bar_i}
        ai_log_signal(sig); return sig
    return None

# ------------------------------ SCALP -----------------------------------------
def build_scalp_signal(sym, kl, bar_i):
    if len(kl) < 40: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    e7=ema(c,7); atrv=atr_like(h,l,c,14)[-1]
    if atrv<=0: return None
    body=abs(c[-1]-c[-2]); cond = body >= 0.10*atrv
    if c[-2]<=e7[-2] and c[-1]>e7[-1] and cond:
        st,_=supertrend_last(h,l,c); pwr=compute_power(c[-1],st,rsi(c,14)[-1])
        sig={"symbol":sym,"dir":"UP","entry":c[-1],"kind":"SCALP","power":pwr,"born_bar":bar_i}
        ai_log_signal(sig); return sig
    if c[-2]>=e7[-2] and c[-1]<e7[-1] and cond:
        st,_=supertrend_last(h,l,c); pwr=compute_power(c[-1],st,rsi(c,14)[-1])
        sig={"symbol":sym,"dir":"DOWN","entry":c[-1],"kind":"SCALP","power":pwr,"born_bar":bar_i}
        ai_log_signal(sig); return sig
    return None

# ------------------------------ UT-STC ----------------------------------------
def build_ut_stc_signal(sym, kl, bar_i):
    if len(kl) < 120: return None
    c=[float(k[4]) for k in kl]; h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]
    st,sd=supertrend_last(h,l,c,10,3.0)
    ml,ms,_=macd(c,12,26,9); r=rsi(c,14)[-1]
    if sd=="UP" and ml[-1]>ms[-1] and r>52:
        pwr=compute_power(c[-1],st,r); sig={"symbol":sym,"dir":"UP","entry":c[-1],"kind":"UT_STC","power":pwr,"born_bar":bar_i}
        ai_log_signal(sig); return sig
    if sd=="DOWN" and ml[-1]<ms[-1] and r<48:
        pwr=compute_power(c[-1],st,r); sig={"symbol":sym,"dir":"DOWN","entry":c[-1],"kind":"UT_STC","power":pwr,"born_bar":bar_i}
        ai_log_signal(sig); return sig
    return None

# ------------------------------ MACD Trend ------------------------------------
def build_macd_trend_signal(sym, kl, bar_i):
    if len(kl) < 40: return None
    c=[float(k[4]) for k in kl]; line,sig,_=macd(c,12,26,9)
    up=line[-1]>sig[-1] and line[-2]<=sig[-2]; dn=line[-1]<sig[-1] and line[-2]>=sig[-2]
    if up:
        st,_=supertrend_last([float(k[2]) for k in kl],[float(k[3]) for k in kl],c)
        pwr=compute_power(c[-1],st,rsi(c,14)[-1])
        s={"symbol":sym,"dir":"UP","entry":c[-1],"kind":"MACD","power":pwr,"born_bar":bar_i}; ai_log_signal(s); return s
    if dn:
        st,_=supertrend_last([float(k[2]) for k in kl],[float(k[3]) for k in kl],c)
        pwr=compute_power(c[-1],st,rsi(c,14)[-1])
        s={"symbol":sym,"dir":"DOWN","entry":c[-1],"kind":"MACD","power":pwr,"born_bar":bar_i}; ai_log_signal(s); return s
    return None

# ------------------------------ FVG Break -------------------------------------
def _find_last_fvg(highs, lows):
    for i in range(len(highs)-1, 0, -1):
        if lows[i] > highs[i-1]:  return ("UPGAP", highs[i-1], lows[i])
        if highs[i] < lows[i-1]: return ("DOWNGAP", highs[i], lows[i-1])
    return (None, None, None)

def build_fvg_break_signal(sym, kl, bar_i):
    if len(kl) < 30: return None
    h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]; c=[float(k[4]) for k in kl]
    kind,a,b = _find_last_fvg(h[-10:], l[-10:])
    if not kind: return None
    px=c[-1]
    if kind=="UPGAP" and px>b:
        st,_=supertrend_last(h,l,c); pwr=compute_power(px,st,rsi(c,14)[-1])
        s={"symbol":sym,"dir":"UP","entry":px,"kind":"FVG","power":pwr,"born_bar":bar_i}; ai_log_signal(s); return s
    if kind=="DOWNGAP" and px<a:
        st,_=supertrend_last(h,l,c); pwr=compute_power(px,st,rsi(c,14)[-1])
        s={"symbol":sym,"dir":"DOWN","entry":px,"kind":"FVG","power":pwr,"born_bar":bar_i}; ai_log_signal(s); return s
    return None

# ------------------------------ Mean-Reversion EXIT ---------------------------
MEAN_REV_FILE      = os.path.join(DATA_DIR,"mean_reversion_positions.json")
MEAN_REV_POS       = safe_load(MEAN_REV_FILE, [])
MEAN_REV_EXIT_DIST = 15.0; MEAN_REV_EXIT_MAX = 30.0; MEAN_REV_CONFIRM = 2; MEAN_REV_INTERVAL = 120

def _mr_save(): safe_save(MEAN_REV_FILE, MEAN_REV_POS)

def mean_reversion_distance_pct(sym):
    kl=futures_get_klines(sym,"1h",120)
    if not kl: return 0.0,None,None,None
    h=[float(k[2]) for k in kl]; l=[float(k[3]) for k in kl]; c=[float(k[4]) for k in kl]
    now=c[-1]; ma99=ema(c,99)[-1] if len(c)>=99 else np.mean(c[-50:])
    stv=(np.mean(h[-10:])+np.mean(l[-10:]))/2.0
    d_ma=abs(now-ma99)/max(ma99,1e-9)*100.0
    d_st=abs(now-stv)/max(stv,1e-9)*100.0
    return max(d_ma,d_st), now, ma99, stv

def close_market(sym, direction, qty):
    side = "SELL" if direction=="UP" else "BUY"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    _signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })

def close_mean_reversion(sym, direction, reason):
    try:
        pos = next((p for p in MEAN_REV_POS if p["symbol"]==sym and p["dir"]==direction), None)
        if not pos: return
        qty = pos["qty"]
        close_market(sym, direction, qty)
        tg_send(f"⚠️ {sym} Mean-Reversion Exit — {reason}")
        log(f"[MEAN-REV CLOSE] {sym} {direction} {reason}")
    except Exception as e:
        tg_send(f"❌ MEAN-REV CLOSE ERR {sym}\n{e}")
        log(f"[MEAN-REV CLOSE ERR] {sym} {e}")
    MEAN_REV_POS[:] = [p for p in MEAN_REV_POS if not (p["symbol"]==sym and p["dir"]==direction)]
    _mr_save()
    TREND_LOCK.pop(sym, None); TREND_LOCK_TIME.pop(sym, None)
    log(f"[MR TRENDLOCK CLEAR] {sym}")

def mean_reversion_watcher():
    while True:
        try:
            if not MEAN_REV_POS:
                time.sleep(MEAN_REV_INTERVAL); continue
            for pos in list(MEAN_REV_POS):
                sym, direction = pos["symbol"], pos["dir"]
                dist, _, _, _ = mean_reversion_distance_pct(sym)
                if dist >= MEAN_REV_EXIT_MAX:
                    close_mean_reversion(sym, direction, f"ortalamadan % {dist:.2f} [HARD]")
                    continue
                if dist >= MEAN_REV_EXIT_DIST:
                    pos["confirm"] = pos.get("confirm",0) + 1
                else:
                    pos["confirm"] = 0
                if pos.get("confirm",0) >= MEAN_REV_CONFIRM:
                    close_mean_reversion(sym, direction, f"ortalamadan % {dist:.2f}")
            _mr_save()
        except Exception as e:
            log(f"[MEAN-REV WATCH ERR] {e}")
        time.sleep(MEAN_REV_INTERVAL)

# ------------------------------ TP/SL Clean Mode ------------------------------
def place_exit_orders(sym, direction, qty, tp_price, sl_price):
    pos_side = "LONG" if direction=="UP" else "SHORT"
    base = {
        "symbol": sym, "side": "SELL" if direction=="UP" else "BUY",
        "timeInForce": "GTC", "positionSide": pos_side,
        "workingType": "CONTRACT_PRICE", "priceProtect": "true",
        "timestamp": now_ts_ms()
    }
    tp = base.copy(); tp.update({"type":"TAKE_PROFIT_MARKET","stopPrice": format_price_by_tick(sym, tp_price)})
    _signed_request("POST","/fapi/v1/order", tp); log(f"[TP SET] {sym} {direction} {tp_price}")
    sl = base.copy(); sl.update({"type":"STOP_MARKET","stopPrice": format_price_by_tick(sym, sl_price)})
    _signed_request("POST","/fapi/v1/order", sl); log(f"[SL SET] {sym} {direction} {sl_price}")

def smart_tp_sl_prices(sym, entry, direction):
    tp_usd = random.uniform(PARAM["TP_USD_MIN"], PARAM["TP_USD_MAX"])
    usd = PARAM["TRADE_SIZE_USDT"]
    if entry <= 0: return entry, entry
    # küçük fiyatlı coinlerde yüzde fallback
    tp_pct, sl_pct = PARAM["FALLBACK_TP_PCT"], PARAM["FALLBACK_SL_PCT"]
    if direction=="UP":
        tp = entry * (1 + tp_pct); sl = entry * (1 - sl_pct)
    else:
        tp = entry * (1 - tp_pct); sl = entry * (1 + sl_pct)
    return round_to_tick(sym, tp), round_to_tick(sym, sl)
# ------------------------------ Executors -------------------------------------
def _count_positions_all():
    try:
        a = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        l = sum(1 for p in a if float(p.get("positionAmt",0)) > 0)
        s = sum(1 for p in a if float(p.get("positionAmt",0)) < 0)
        return l, s
    except Exception as e:
        log(f"[POSRISK ERR] {e}"); return 0,0

def open_market(sym, direction, qty):
    if PARAM.get("SIM_MODE", False):
        # sim fill
        px = futures_get_price(sym) or 0.0
        pos = safe_load(SIM_POS_FILE, [])
        pos.append({"symbol":sym,"dir":direction,"qty":qty,"entry":px,"time":now_local_iso(),"kind":"SIM"})
        safe_save(SIM_POS_FILE, pos)
        return px
    side = "BUY" if direction=="UP" else "SELL"
    pos_side = "LONG" if direction=="UP" else "SHORT"
    r = _signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })
    return float(r.get("avgPrice") or r.get("price") or futures_get_price(sym) or 0.0)

def execute_kivanc_trade(sig):
    sym, direction = sig["symbol"], sig["dir"]
    L,S = _count_positions_all()
    if direction=="UP" and L>=PARAM["GLOBAL_LONG_CAP"]: return
    if direction=="DOWN" and S>=PARAM["GLOBAL_SHORT_CAP"]: return
    kv = _kiv_refresh_from_positions()
    kc_long  = sum(1 for x in kv if x["dir"]=="UP")
    kc_short = sum(1 for x in kv if x["dir"]=="DOWN")
    if direction=="UP" and kc_long>=PARAM["KC_LONG_CAP"]:  return
    if direction=="DOWN" and kc_short>=PARAM["KC_SHORT_CAP"]: return
    if TREND_LOCK.get(sym) == direction: return
    entry_ref = sig.get("entry") or futures_get_price(sym)
    if not entry_ref: return
    qty = calc_order_qty(sym, entry_ref, PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0: return
    try:
        fill = open_market(sym, direction, qty)
        kv = _kiv_refresh_from_positions(); kv.append({"symbol":sym,"dir":direction}); _kiv_save(kv)
        TREND_LOCK[sym] = direction; TREND_LOCK_TIME[sym] = now_ts_s()
        tg_send(f"✅ KIVANC OPEN (no TP)\n{sym} {direction} qty:{qty}\nEntry:{format_price_by_tick(sym,fill)}")
        ai_log_rl("KC_OPEN", {"symbol":sym,"dir":direction,"qty":qty,"entry":fill})
    except Exception as e:
        tg_send(f"❌ KIVANC OPEN ERR {sym}\n{e}"); log(f"[KIVANC OPEN ERR] {sym} {e}")

def execute_generic_trade(sig):
    sym, direction, kind = sig["symbol"], sig["dir"], sig.get("kind","GEN")
    pwr = sig.get("power", 70.0)
    if not in_power_band(pwr): return
    L,S = _count_positions_all()
    if direction=="UP" and L>=PARAM["GLOBAL_LONG_CAP"]: return
    if direction=="DOWN" and S>=PARAM["GLOBAL_SHORT_CAP"]: return
    if TREND_LOCK.get(sym) == direction: return
    entry_ref = sig.get("entry") or futures_get_price(sym)
    if not entry_ref: return
    qty = calc_order_qty(sym, entry_ref, PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0: return
    try:
        fill = open_market(sym, direction, qty)
        tp, sl = smart_tp_sl_prices(sym, fill, direction)
        place_exit_orders(sym, direction, qty, tp, sl)
        TREND_LOCK[sym] = direction; TREND_LOCK_TIME[sym] = now_ts_s()
        tg_send(f"🟦 {kind} OPEN\n{sym} {direction} qty:{qty}\nEntry:{format_price_by_tick(sym,fill)}\nTP:{format_price_by_tick(sym,tp)} SL:{format_price_by_tick(sym,sl)}")
        ai_log_rl("GEN_OPEN", {"kind":kind,"symbol":sym,"dir":direction,"qty":qty,"entry":fill,"tp":tp,"sl":sl})
    except Exception as e:
        tg_send(f"❌ {kind} OPEN ERR {sym}\n{e}"); log(f"[{kind} OPEN ERR] {sym} {e}")

# ------------------------------ Scanner ---------------------------------------
def scan_symbol(sym, bar_i):
    kl = futures_get_klines(sym, "1h", 200)
    if len(kl) < 60: return []
    res = []
    r = build_kivanc_confirm_signal(sym, kl, bar_i)
    if r: res.append(r)
    e = build_early_signal(sym, kl, bar_i)
    if e: res.append(e)
    s = build_scalp_signal(sym, kl, bar_i)
    if s: res.append(s)
    u = build_ut_stc_signal(sym, kl, bar_i)
    if u: res.append(u)
    m = build_macd_trend_signal(sym, kl, bar_i)
    if m: res.append(m)
    f = build_fvg_break_signal(sym, kl, bar_i)
    if f: res.append(f)
    return res

# ------------------------------ Reports ---------------------------------------
def daily_report_if_needed():
    # her gün bir kere mini özet
    today = (datetime.now(timezone.utc)+timedelta(hours=TIME_OFFSET_H)).date().isoformat()
    if STATE.get("last_report_day") == today: return
    try:
        L,S = _count_positions_all()
        msg = f"🗓 Daily report ({today})\nOpen L/S: {L}/{S}\nClosed total: {STATE.get('closed_count',0)}"
        tg_send(msg)
        STATE["last_report_day"] = today
        safe_save(STATE_FILE, STATE)
    except Exception as e:
        log(f"[DAILY REPORT ERR] {e}")

def _cleanup_trend_lock_expired():
    now_s = now_ts_s()
    expired = [sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        TREND_LOCK.pop(sym, None); TREND_LOCK_TIME.pop(sym, None)
        log(f"[TRENDLOCK TIMEOUT] {sym}")

# ------------------------------ Main ------------------------------------------
def main():
    ensure_csv_header()
    tg_send("🚀 EMA ULTRA v15.9.65 FULL — KC(no TP,4/4)+30/30 | EARLY/SCALP/UT/MACD/FVG (Power+TP/SL Clean) | MR Exit | PEMA OFF | RL/AI/CSV | TG cmds")
    log("[START] EMA ULTRA v15.9.65 FULL")
    symbols = auto_init_symbols()

    # arka plan işler
    threading.Thread(target=mean_reversion_watcher, daemon=True).start()
    threading.Thread(target=tg_poll_loop, daemon=True).start()

    while True:
        try:
            STATE["bar_index"] = STATE.get("bar_index", 0) + 1
            bar_i = STATE["bar_index"]

            # Sinyal tarama
            sigs = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = [ex.submit(scan_symbol, s, bar_i) for s in symbols]
                for f in as_completed(futs):
                    try: r = f.result()
                    except Exception as e: log(f"[SCAN ERR] {e}"); r=[]
                    if r: sigs.extend(r)

            # Uygulama
            for sig in sigs:
                if sig.get("kind") == "KIVANC_CONFIRM":
                    execute_kivanc_trade(sig)
                else:
                    execute_generic_trade(sig)

            _cleanup_trend_lock_expired()
            daily_report_if_needed()
            safe_save(STATE_FILE, STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[MAIN LOOP ERR] {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
