# ==============================================
# 📘 EMA ULTRA v15.9.36 — Hard Limit Fix + TP Safe v3
# ==============================================
import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

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

BOT_TOKEN      = os.getenv("BOT_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
PRECISION_CACHE = {}
TREND_LOCK = {}        # {SYMBOL: "UP"/"DOWN"}
TREND_LOCK_TIME = {}   # {SYMBOL: last_set_ts}
TRENDLOCK_EXPIRY_SEC = 6*3600

# ----- globals (sim/limit batch) -----
SIM_QUEUE = []
CANDIDATE_SIGNALS = []     # batch hard-limit seçimi için

def safe_load(p, d):
    try:
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return d

def safe_save(p, d):
    try:
        with SAVE_LOCK:
            tmp = p + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p)
    except Exception as e:
        print("[SAVE ERR]", e, flush=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())
def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":t},
            timeout=10
        )
    except: pass

def tg_send_file(p, cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p): return
    try:
        with open(p,"rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":cap},
                files={"document":(os.path.basename(p), f)},
                timeout=30
            )
    except: pass

def _signed_request(m, path, payload):
    q = "&".join([f"{k}={payload[k]}" for k in payload])
    sig = hmac.new(BINANCE_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = BINANCE_FAPI + path + "?" + q + "&signature=" + sig
    r = (requests.post(url, headers=headers, timeout=10) if m=="POST"
         else requests.get(url, headers=headers, timeout=10))
    if r.status_code != 200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE:
        return PRECISION_CACHE[sym]
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
        log(f"[PREC WARN]{sym}{e}")
        PRECISION_CACHE[sym] = {"stepSize":0.0001,"tickSize":0.0001,"minPrice":1e-8,"maxPrice":9e8}
    return PRECISION_CACHE[sym]

def adjust_precision(sym, v, kind="qty"):
    f = get_symbol_filters(sym)
    step = f["stepSize"] if kind=="qty" else f["tickSize"]
    if step <= 0: return v
    # to nearest step
    return round(round(v/step)*step, 12)

def futures_get_price(sym):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price", params={"symbol":sym}, timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym, it, lim):
    try:
        r = requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                         params={"symbol":sym,"interval":it,"limit":lim},
                         timeout=10).json()
        if r and int(r[-1][6]) > now_ts_ms(): r = r[:-1]
        return r
    except:
        return []

# ----------------- Indicators -----------------
def ema(vals, n):
    if not vals: return []
    k = 2/(n+1)
    e = [vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+2: return [50]*len(vals)
    d = np.diff(vals)
    g = np.maximum(d,0); l = -np.minimum(d,0)
    ag = np.mean(g[:period]); al = np.mean(l[:period])
    out = [50]*period
    for i in range(period, len(d)):
        ag = (ag*(period-1)+g[i])/period
        al = (al*(period-1)+l[i])/period
        rs = ag/al if al>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else:
            tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period, len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def slope(seq, span=3):
    if len(seq)<span+1: return 0.0
    y = np.array(seq[-(span+1):], dtype=float)
    x = np.arange(len(y))
    m, _ = np.polyfit(x, y, 1)
    return float(m)

def calc_power(e_now, e_prev, e_prev2, atr_v, price, rsi_val):
    diff = abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base = 55 + diff*20 + ((rsi_val-50)/50)*15 + (atr_v/max(price,1e-12))*200
    return min(100, max(0, base))

def tier_from_power(p):
    if 65<=p<75: return "REAL","🟩"
    if p>=75:    return "ULTRA","🟦"
    if p>=60:    return "NORMAL","🟨"
    return None,""

# ----------------- Params / State -----------------
PARAM_DEFAULT = {
    "SCALP_TP_PCT": 0.006,         # referans
    "SCALP_SL_PCT": 0.20,          # canlıda SL yok; sim için referans
    "TRADE_SIZE_USDT": 250.0,
    "MAX_BUY": 30,
    "MAX_SELL": 30,
    # High Sensitivity
    "ANGLE_MIN": 0.00002,
    "FAST_EMA_PERIOD": 3,
    "SLOW_EMA_PERIOD": 7,
    "ATR_SPIKE_RATIO": 0.08,       # biraz düşürüldü (Early artar)
    "APPROVE_MINUTES": [30,60,90,120],
    # LIMIT TP fail → MARKET close fallback süresi (sn)
    "LIMIT_TP_TIMEOUT_SEC": 120
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict): PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index": 0,
    "last_report": 0,
    "auto_trade_active": True,
    "last_api_check": 0,
    "long_blocked": False,
    "short_blocked": False,
    # hard-limit local counters
    "live_long_count": 0,
    "live_short_count": 0
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
for k,v in STATE_DEFAULT.items():
    STATE.setdefault(k, v)

AI_SIGNALS  = safe_load(AI_SIGNALS_FILE, [])
AI_ANALYSIS = safe_load(AI_ANALYSIS_FILE, [])
AI_RL       = safe_load(AI_RL_FILE, [])
SIM_POSITIONS = safe_load(SIM_POS_FILE, [])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE, [])

# ----------------- Signal Builders -----------------
def _chg24(sym):
    try:
        return float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                                  params={"symbol":sym}, timeout=5).json()["priceChangePercent"])
    except:
        return 0.0

def build_early_signal(sym, kl, bar_i):
    """ EMA(FAST,SLOW) cross + ATR spike + |chg24h|<10 + power band """
    if len(kl)<60: return None
    chg = _chg24(sym)
    if abs(chg)>=10: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    fper = PARAM.get("FAST_EMA_PERIOD",3)
    sper = PARAM.get("SLOW_EMA_PERIOD",7)
    ef=ema(closes,fper); es=ema(closes,sper)

    up = (ef[-2]>es[-2]) and (ef[-3]<=es[-3])
    dn = (ef[-2]<es[-2]) and (ef[-3]>=es[-3])
    if not (up or dn): return None

    atrs=atr_like(highs,lows,closes)
    if len(atrs)<2: return None
    if not (atrs[-1] >= atrs[-2]*(1.0+PARAM.get("ATR_SPIKE_RATIO",0.1))): return None

    direction = "UP" if up else "DOWN"
    entry = closes[-1]
    r_val = rsi(closes)[-1]
    pwr = calc_power(es[-1], es[-2], es[-5] if len(es)>=6 else es[-2], atrs[-1], entry, r_val)
    tier, emoji = tier_from_power(pwr)
    if not tier: tier, emoji = "EARLY", "⚡️"

    # reference tp/sl for sim
    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":emoji,"entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":True,
        "kind":"EARLY"
    }

def build_e200s_signal(sym, kl, bar_i):
    """ E200S: Price vs EMA200 trend onay + small pullback (directional filter) """
    if len(kl)<200: return None
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    e200 = ema(closes,200)
    e20  = ema(closes,20)
    chg = _chg24(sym)
    if abs(chg)>=12: return None

    price = closes[-1]
    up_tr = price>e200[-1] and e20[-1]>e200[-1]
    dn_tr = price<e200[-1] and e20[-1]<e200[-1]
    if not (up_tr or dn_tr): return None

    atrs=atr_like(highs,lows,closes); r_val=rsi(closes)[-1]
    pwr=calc_power(e20[-1],e20[-2],e20[-5] if len(e20)>=6 else e20[-2], atrs[-1], price, r_val)
    tier, emoji = tier_from_power(pwr)
    if not tier: return None

    direction = "UP" if up_tr else "DOWN"
    if direction=="UP":
        tp_guess=price*(1+PARAM["SCALP_TP_PCT"]); sl_guess=price*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=price*(1-PARAM["SCALP_TP_PCT"]); sl_guess=price*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":"📈","entry":price,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":False,
        "kind":"E200S"
    }

def build_rsi_engulf_signal(sym, kl, bar_i):
    """ RSI > 50 (up) veya <50 (down) + engulfing body teyidi """
    if len(kl)<30: return None
    opens =[float(k[1]) for k in kl]
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    chg=_chg24(sym)
    if abs(chg)>=15: return None

    rs = rsi(closes)
    r_now = rs[-1]
    # engulfing kontrol (basit body engulf)
    o1,c1 = opens[-2], closes[-2]   # previous
    o2,c2 = opens[-1], closes[-1]   # current
    bull_eng = (c2>o2) and (o2<=c1) and ((c2-o2)>(o1-c1 if (o1>c1) else 0))
    bear_eng = (c2<o2) and (o2>=c1) and ((o2-c2)>(c1-o1 if (c1>o1) else 0))

    direction=None
    if r_now>50 and bull_eng: direction="UP"
    if r_now<50 and bear_eng: direction="DOWN"
    if not direction: return None

    atrs=atr_like(highs,lows,closes)
    e20=ema(closes,20)
    pwr=calc_power(e20[-1], e20[-2], e20[-5] if len(e20)>=6 else e20[-2], atrs[-1], closes[-1], r_now)
    tier,emoji=tier_from_power(pwr)
    if not tier: return None

    entry=closes[-1]
    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":"🟪","entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_now,"atr":atrs[-1],
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":False,
        "kind":"RSI-ENGULF"
    }

def build_tf20_200_signal(sym, kl, bar_i):
    """ Trend-follow: EMA20 slope yönünde bar kapanışı """
    if len(kl)<25: return None
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    e20=ema(closes,20)
    sl = slope(e20, span=5)
    if abs(sl) < PARAM["ANGLE_MIN"]: return None
    direction = "UP" if sl>0 else "DOWN"
    chg=_chg24(sym)
    if abs(chg)>=18: return None

    atrs=atr_like(highs,lows,closes); r_val=rsi(closes)[-1]
    pwr=calc_power(e20[-1], e20[-2], e20[-6] if len(e20)>=7 else e20[-2], atrs[-1], closes[-1], r_val)
    tier,emoji=tier_from_power(pwr)
    if not tier: return None

    entry=closes[-1]
    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":tier,"emoji":"🟦","entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":False,
        "kind":"TF20/200","ema20_slope":sl
    }

def scan_symbol(sym, bar_i):
    kl1 = futures_get_klines(sym,"1h",200)
    if len(kl1)<60: return []

    sigs=[]
    s1 = build_early_signal(sym, kl1, bar_i)
    if s1: sigs.append(s1)

    s2 = build_e200s_signal(sym, kl1, bar_i)
    if s2: sigs.append(s2)

    s3 = build_rsi_engulf_signal(sym, kl1, bar_i)
    if s3: sigs.append(s3)

    s4 = build_tf20_200_signal(sym, kl1, bar_i)
    if s4: sigs.append(s4)

    return sigs

def run_parallel(symbols, bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(scan_symbol, s, bar_i) for s in symbols]
        for f in as_completed(futs):
            try: sigs=f.result()
            except: sigs=[]
            if sigs: out.extend(sigs)
    return out
# ----------------- SIM / AI / TrendLock -----------------
def enrich_with_ai_context(pos):
    best=None
    for s in reversed(AI_SIGNALS):
        if s.get("symbol")!=pos.get("symbol"): continue
        e_sig=s.get("entry"); e_pos=pos.get("entry")
        if not e_sig or not e_pos: continue
        if abs(e_sig-e_pos)/max(e_sig,1e-12) < 0.002:
            best=s; break
    if best:
        for k in ("rsi","atr","chg24h","born_bar","tier","power","early","kind"):
            if k in best: pos[k]=best.get(k)
    return pos

def queue_sim_variants(sig):
    now_s=now_ts_s()
    for mins in PARAM.get("APPROVE_MINUTES",[30,60,90,120]):
        SIM_QUEUE.append({
            "symbol":sig["symbol"], "dir":sig["dir"], "tier":sig["tier"],
            "entry":sig["entry"], "tp":sig["tp"], "sl":sig["sl"], "power":sig["power"],
            "created_ts":now_s, "open_after_ts":now_s+mins*60,
            "approve_delay_min":mins, "approve_label":f"approve_{mins}m",
            "status":"PENDING", "early":bool(sig.get("early",False)),
            "kind":sig.get("kind","")
        })
    safe_save(SIM_POS_FILE, SIM_QUEUE)

def process_sim_queue_and_open_due():
    global SIM_POSITIONS
    now_s=now_ts_s()
    remain=[]; opened=False
    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            SIM_POSITIONS.append({**q,"status":"OPEN","open_ts":now_s,"open_time":now_local_iso()})
            opened=True
            log(f"[SIM OPEN] {q['symbol']} {q['dir']} approve={q['approve_delay_min']}m kind={q.get('kind')}")
        else:
            remain.append(q)
    SIM_QUEUE[:]=remain
    if opened: safe_save(SIM_POS_FILE, SIM_POSITIONS)

def _unlock_trend_for(sym):
    TREND_LOCK.pop(sym, None)
    TREND_LOCK_TIME.pop(sym, None)
    log(f"[TRENDLOCK CLEAR] {sym}")

def process_sim_closes():
    global SIM_POSITIONS
    if not SIM_POSITIONS: return
    still=[]; changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")!="OPEN": continue
        last=futures_get_price(pos["symbol"])
        if last is None: still.append(pos); continue
        hit=None
        if pos["dir"]=="UP":
            if last>=pos["tp"]: hit="TP"
            elif last<=pos["sl"]: hit="SL"
        else:
            if last<=pos["tp"]: hit="TP"
            elif last>=pos["sl"]: hit="SL"
        if hit:
            close_time=now_local_iso()
            gain_pct=((last/pos["entry"]-1.0)*100.0 if pos["dir"]=="UP"
                      else (pos["entry"]/last-1.0)*100.0)
            SIM_CLOSED.append({
                **enrich_with_ai_context(dict(pos)),
                "status":"CLOSED","close_time":close_time,
                "exit_price":last,"exit_reason":hit,"gain_pct":gain_pct
            })
            _unlock_trend_for(pos["symbol"])
            changed=True
            log(f"[SIM CLOSE] {pos['symbol']} {pos['dir']} {hit} {gain_pct:.3f}% approve={pos.get('approve_delay_min')}m kind={pos.get('kind')}")
        else:
            still.append(pos)
    SIM_POSITIONS=still
    if changed:
        safe_save(SIM_POS_FILE, SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE, SIM_CLOSED)

# ----------------- TP Safe v3 -----------------
def _fmt_by_tick(tick):
    s=str(tick)
    dec = len(s.split(".")[1].rstrip("0")) if "." in s else 0
    return f"{{:.{dec}f}}"

def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd, 1e-12)
    price = entry_exec*(1+tp_pct) if direction=="UP" else entry_exec*(1-tp_pct)
    return price, tp_pct

def _ensure_min_tick_move(sym, entry_exec, target_price, direction):
    f = get_symbol_filters(sym)
    tick = f["tickSize"]
    # entry ile aynı yuvarlanırsa en az 1 tick uzağa it
    entry_r = adjust_precision(sym, entry_exec, "price")
    targ_r  = adjust_precision(sym, target_price, "price")
    if targ_r == entry_r:
        if direction=="UP": targ_r += tick
        else:               targ_r -= tick
    return max(f["minPrice"], min(f["maxPrice"], targ_r))

def _place_limit_tp(sym, direction, qty, entry_exec, tp_price):
    """ LIMIT TP (reduceOnly=True) """
    side = "SELL" if direction=="UP" else "BUY"
    fmt = _fmt_by_tick(get_symbol_filters(sym)["tickSize"])
    payload = {
        "symbol":sym,"side":side,"type":"LIMIT",
        "timeInForce":"GTC","quantity":f"{qty}",
        "price":fmt.format(tp_price),
        "reduceOnly":"true",               # güvenli kapatma
        "positionSide": ("LONG" if direction=="UP" else "SHORT"),
        "timestamp": now_ts_ms()
    }
    _signed_request("POST","/fapi/v1/order", payload)
    log(f"[TP LIMIT OK] {sym} price={fmt.format(tp_price)} qty={qty}")
    return True

def _place_take_profit_market(sym, direction, qty, stop_price):
    """ TAKE_PROFIT_MARKET (sadece stopPrice, price YOK) """
    side = "SELL" if direction=="UP" else "BUY"
    fmt = _fmt_by_tick(get_symbol_filters(sym)["tickSize"])
    payload = {
        "symbol":sym,"side":side,"type":"TAKE_PROFIT_MARKET",
        "stopPrice": fmt.format(stop_price),
        "workingType":"MARK_PRICE",
        "closePosition":"true",
        "positionSide": ("LONG" if direction=="UP" else "SHORT"),
        "timestamp": now_ts_ms()
    }
    # price asla gönderme
    _signed_request("POST","/fapi/v1/order", payload)
    log(f"[TP TPMARKET OK] {sym} stop={fmt.format(stop_price)}")
    return True

def futures_set_tp_only(sym, direction, qty, entry_exec, tp_low_usd=1.6, tp_high_usd=2.0):
    """ v3: Önce LIMIT, sonra TPMARKET; ikisinde de tick/min/max uyumu ve entry'den min 1 tick uzaklık """
    try:
        f = get_symbol_filters(sym)
        tick = f["tickSize"]; minp=f["minPrice"]; maxp=f["maxPrice"]
        fmt = _fmt_by_tick(tick)
        trade_usd = float(PARAM.get("TRADE_SIZE_USDT",250.0))

        # 0.1 → LIMIT dene
        for tp_usd in [round(x,1) for x in np.arange(tp_low_usd, tp_high_usd+0.001, 0.1)]:
            raw, tp_pct = _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd)
            tp_price = _ensure_min_tick_move(sym, entry_exec, raw, direction)
            if not (minp <= tp_price <= maxp):
                log(f"[TP RANGE] {sym} skip ${tp_usd} price={tp_price}")
                continue
            try:
                ok = _place_limit_tp(sym, direction, qty, entry_exec, tp_price)
                if ok: return True, tp_usd, tp_pct
            except Exception as e:
                log(f"[TP LIMIT FAIL] {sym} ${tp_usd} err={e}")

        # 0.01 → TAKE_PROFIT_MARKET
        for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
            raw, tp_pct = _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd)
            stop_price = _ensure_min_tick_move(sym, entry_exec, raw, direction)
            if not (minp <= stop_price <= maxp): continue
            try:
                ok = _place_take_profit_market(sym, direction, qty, stop_price)
                if ok: return True, tp_usd, tp_pct
            except Exception as e:
                log(f"[TP TPMARKET FAIL] {sym} ${tp_usd} err={e}")

        log(f"[NO TP] {sym} 1.6–2.0$ aralığında geçerli TP bulunamadı.")
        return False, None, None
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False, None, None

# ----------------- Guards / HB / Report -----------------
def update_directional_limits():
    live={"long":{}, "short":{}, "long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            amt=float(p["positionAmt"]); sym=p["symbol"]
            if amt>0: live["long"][sym]=amt
            elif amt<0: live["short"][sym]=abs(amt)
        live["long_count"]=len(live["long"]); live["short_count"]=len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")

    # API sayımı ile lokal sayımı sync: en büyük olanı baz al
    STATE["live_long_count"]  = max(STATE.get("live_long_count",0),  live["long_count"])
    STATE["live_short_count"] = max(STATE.get("live_short_count",0), live["short_count"])

    STATE["long_blocked"]  = (STATE["live_long_count"]  >= PARAM["MAX_BUY"])
    STATE["short_blocked"] = (STATE["live_short_count"] >= PARAM["MAX_SELL"])
    STATE["auto_trade_active"] = not (STATE["long_blocked"] and STATE["short_blocked"])
    safe_save(STATE_FILE, STATE)
    return live

def _cleanup_trend_lock_expired():
    now_s=now_ts_s()
    expired=[sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        _unlock_trend_for(sym); log(f"[TRENDLOCK TIMEOUT] {sym} (6h)")

def heartbeat_and_status_check(live_positions_snapshot):
    now=time.time()
    if now-STATE.get("last_api_check",0)<600: return
    STATE["last_api_check"]=now; safe_save(STATE_FILE,STATE)
    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift=abs(now_ts_ms()-st)
        ping_ok=requests.get(BINANCE_FAPI+"/fapi/v1/ping",timeout=5).status_code==200
        key_ok=True
        try: _=_signed_request("GET","/fapi/v2/account",{"timestamp":now_ts_ms()})
        except: key_ok=False
        hb=(f"✅ HEARTBEAT drift={int(drift)}ms ping={ping_ok} key={key_ok}"
            if ping_ok and key_ok and drift<1500 else
            f"⚠️ HEARTBEAT ping={ping_ok} key={key_ok} drift={int(drift)}")
        tg_send(hb); log(hb)
    except Exception as e:
        tg_send(f"❌ HEARTBEAT {e}"); log(f"[HBERR]{e}")

    msg=(f"📊 STATUS bar:{STATE.get('bar_index',0)} "
         f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
         f"long_blocked:{STATE.get('long_blocked')} "
         f"short_blocked:{STATE.get('short_blocked')} "
         f"long:{STATE.get('live_long_count',0)} "
         f"short:{STATE.get('live_short_count',0)} "
         f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
         f"sim_closed:{len(SIM_CLOSED)}")
    tg_send(msg); log(msg)

def ai_log_signal(sig):
    AI_SIGNALS.append({
        "time":now_local_iso(),"symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
        "chg24h":sig["chg24h"],"power":sig["power"],"rsi":sig.get("rsi"),"atr":sig.get("atr"),
        "tp":sig["tp"],"sl":sig["sl"],"entry":sig["entry"],"born_bar":sig.get("born_bar"),
        "early":bool(sig.get("early",False)),"kind":sig.get("kind",""),
        "ema20_slope":sig.get("ema20_slope")
    })
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)

def ai_update_analysis_snapshot():
    snapshot={
        "time":now_local_iso(),
        "ultra_signals_total": sum(1 for x in AI_SIGNALS if x.get("tier")=="ULTRA"),
        "real_signals_total":  sum(1 for x in AI_SIGNALS if x.get("tier")=="REAL"),
        "normal_signals_total":sum(1 for x in AI_SIGNALS if x.get("tier")=="NORMAL"),
        "early_signals_total": sum(1 for x in AI_SIGNALS if x.get("early")),
        "sim_open_count":len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
        "sim_closed_count":len(SIM_CLOSED)
    }
    AI_ANALYSIS.append(snapshot); safe_save(AI_ANALYSIS_FILE,AI_ANALYSIS)

def auto_report_if_due():
    now_now=time.time()
    if now_now-STATE.get("last_report",0) < 14400: return
    ai_update_analysis_snapshot()
    for fpath in [AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,SIM_POS_FILE,SIM_CLOSED_FILE]:
        try:
            if os.path.exists(fpath) and os.path.getsize(fpath)>10*1024*1024:
                with open(fpath,"r",encoding="utf-8") as f: raw=f.read()
                tail=raw[-int(len(raw)*0.2):]
                with open(fpath,"w",encoding="utf-8") as f: f.write(tail)
        except: pass
        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")
    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"]=now_now; safe_save(STATE_FILE,STATE)

# ----------------- Execution / Hard-limit batch -----------------
def open_market_position(sym, direction, qty):
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })
    fill = res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    if direction=="UP": STATE["live_long_count"] = STATE.get("live_long_count",0) + 1
    else:              STATE["live_short_count"]= STATE.get("live_short_count",0)+ 1
    safe_save(STATE_FILE, STATE)
    return {"symbol":sym,"dir":direction,"qty":qty,"entry":float(fill),"pos_side":pos_side}

def _duplicate_or_locked(sym, direction):
    if TREND_LOCK.get(sym)==direction:
        log(f"[TRENDLOCK HIT] {sym} {direction}")
        return True
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[POSRISK ERR]{e}"); acc=[]
    if direction=="UP":
        if sym in [p["symbol"] for p in acc if float(p["positionAmt"])>0]:
            log(f"[DUP-LONG] {sym}"); return True
    else:
        if sym in [p["symbol"] for p in acc if float(p["positionAmt"])<0]:
            log(f"[DUP-SHORT] {sym}"); return True
    return False

def _can_direction(direction):
    if not STATE.get("auto_trade_active", True): return False
    if direction=="UP" and STATE.get("long_blocked",False): return False
    if direction=="DOWN" and STATE.get("short_blocked",False): return False
    return True

def _set_trend_lock(sym, direction):
    TREND_LOCK[sym]=direction; TREND_LOCK_TIME[sym]=now_ts_s()
    log(f"[TRENDLOCK SET] {sym} {direction}")

def calc_order_qty(sym, entry, usd):
    raw = usd/max(entry,1e-12)
    return adjust_precision(sym, raw, "qty")

def execute_real_trade(sig):
    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig["power"]; is_early=bool(sig.get("early",False))
    ok_early = is_early and (65 <= pwr < 75)
    ok_real  = (not is_early) and (sig.get("tier") in ("REAL","ULTRA")) and (65 <= pwr < 75)
    if not (ok_early or ok_real): return

    if not _can_direction(direction): return
    if _duplicate_or_locked(sym, direction): return

    qty=calc_order_qty(sym, sig["entry"], PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return

    try:
        opened=open_market_position(sym, direction, qty)
        entry_exec=opened.get("entry") or futures_get_price(sym)
        if not entry_exec or entry_exec<=0:
            log(f"[OPEN FAIL] {sym} entry yok"); return

        tp_ok, tp_usd_used, tp_pct_used = futures_set_tp_only(
            sym, direction, qty, entry_exec, tp_low_usd=1.6, tp_high_usd=2.0
        )

        # LIMIT TP timeout → MARKET close fallback (emir yönetimi borsada tutulduğu için burada sinyal)
        if not tp_ok:
            # yine de trendlock set, telegram at
            pass

        _set_trend_lock(sym, direction)

        prefix = ("⚡️ EARLY" if is_early else "✅ REAL")
        if tp_ok:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"Power:{pwr:.2f}\nEntry:{entry_exec:.12f}\n"
                    f"TP hedefi:{tp_usd_used:.2f}$ ({(tp_pct_used or 0)*100:.3f}%)\n"
                    f"time:{now_local_iso()}")
        else:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"Power:{pwr:.2f}\nEntry:{entry_exec:.12f}\n"
                    f"TP: YOK (1.6–2.0$ tarama başarısız)\n"
                    f"time:{now_local_iso()}")

        AI_RL.append({
            "time":now_local_iso(),"symbol":sym,"dir":direction,"entry":entry_exec,
            "tp_usd_used":tp_usd_used,"tp_pct_used":tp_pct_used,"tp_ok":tp_ok,
            "power":pwr,"born_bar":sig.get("born_bar"),"early":is_early,"kind":sig.get("kind","")
        }); safe_save(AI_RL_FILE, AI_RL)
    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")

def batch_hard_limit_and_execute(signals):
    """ Aynı bar içindeki tüm sinyaller: tek seferde hard-limit uygula """
    if not signals: return
    # Güç sırasına göre (yüksek önce) — istersen farklı öncelik verebiliriz
    srt = sorted(signals, key=lambda s: s.get("power",0), reverse=True)

    # son canlı snapshot (API + local)
    live = update_directional_limits()

    long_free  = max(0, PARAM["MAX_BUY"]  - STATE.get("live_long_count",0))
    short_free = max(0, PARAM["MAX_SELL"] - STATE.get("live_short_count",0))

    selected=[]
    for sig in srt:
        if sig["dir"]=="UP":
            if long_free<=0: continue
            selected.append(sig); long_free -= 1
        else:
            if short_free<=0: continue
            selected.append(sig); short_free -= 1

    # kayıt + sim + gerçek
    for sig in selected:
        ai_log_signal(sig)
        queue_sim_variants(sig)
        execute_real_trade(sig)
# ----------------- MAIN LOOP -----------------
def main():
    tg_send("🚀 EMA ULTRA v15.9.36 aktif (HardLimit + TP Safe v3)")
    log("[START] EMA ULTRA v15.9.36 FULL")

    # USDT semboller
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}"); symbols=[]
    symbols.sort()

    while True:
        try:
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]

            # 1) Tara
            sigs = run_parallel(symbols, bar_i)

            # 2) Batch hard-limit + execute
            batch_hard_limit_and_execute(sigs)

            # 3) SIM approve ve kapanışlar
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4) Auto backup/report
            auto_report_if_due()

            # 5) HB + durum
            live=update_directional_limits()
            heartbeat_and_status_check(live)

            # 6) TrendLock expiry
            _cleanup_trend_lock_expired()

            # 7) Persist & sleep
            safe_save(STATE_FILE, STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# -------------- ENTRYPOINT --------------
if __name__=="__main__":
    main()
