import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================
# EMA ULTRA v15.9.15b PowerRangeRealOnly + Independent Buy/Sell Guard
#
#  - power 65–75 arası => REAL (Binance MARKET + TP/SL)
#  - Telegram sadece gerçek işlemleri bildirir
#  - BUY / SELL guard sistemi bağımsız çalışır
#       * BUY limit dolarsa sadece long işlemler durur
#       * SELL limit dolarsa sadece short işlemler durur
#  - Heartbeat / AutoBackup / TP-SL precision fix / TrendLock mevcut
# ==============================================================

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
TREND_LOCK = {}
SIM_QUEUE = []

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
            tmp = p + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:
        print("[SAVE ERR]", e, flush=True)

def log(msg):
    print(msg,flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())

def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except: pass

def tg_send_file(p,cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p): return
    try:
        with open(p,"rb") as f:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":cap},
                files={"document":(os.path.basename(p),f)},timeout=30)
    except: pass

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r = (requests.post(url,headers=headers,timeout=10)
         if m=="POST" else requests.get(url,headers=headers,timeout=10))
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE: return PRECISION_CACHE[sym]
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        step=float(lot.get("stepSize","1"))
        tick=float(pricef.get("tickSize","0.01"))
        min_price=float(pricef.get("minPrice","0.00000001"))
        PRECISION_CACHE[sym]={"stepSize":step,"tickSize":tick,"minPrice":min_price}
    except Exception as e:
        log(f"[PRECISION WARN]{sym}{e}")
        PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001,"minPrice":0.0001}
    return PRECISION_CACHE[sym]

def adjust_precision(sym,v,kind="qty"):
    f=get_symbol_filters(sym)
    step=f["stepSize"] if kind=="qty" else f["tickSize"]
    if step<=0: return v
    return round((round(v/step)*step),12)
def calc_order_qty(sym,entry,usd):
    raw = usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def futures_get_price(sym):
    try:
        r=requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/price",
            params={"symbol":sym},timeout=5
        ).json()
        return float(r["price"])
    except:
        return None

def futures_24h_change(sym):
    try:
        r=requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/24hr",
            params={"symbol":sym},timeout=5
        ).json()
        return float(r["priceChangePercent"])
    except:
        return 0.0

def futures_get_klines(sym,it,lim):
    """
    confirmed bar only:
    eğer son mum future timestamp taşıyorsa onu at
    """
    try:
        r=requests.get(
            BINANCE_FAPI+"/fapi/v1/klines",
            params={"symbol":sym,"interval":it,"limit":lim},
            timeout=10
        ).json()
        nowms = now_ts_ms()
        if r and int(r[-1][6])>nowms:
            r=r[:-1]
        return r
    except:
        return []

def open_market_position(sym,dir,qty):
    """
    Gerçek market order aç (hedge mod).
    Dönen değer:
      - entry: Binance'in verdiği fill/avg fiyat (mümkünse)
    """
    side="BUY" if dir=="UP" else "SELL"
    pos_side="LONG" if dir=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,
        "side":side,
        "type":"MARKET",
        "quantity":f"{qty}",
        "positionSide":pos_side,
        "timestamp":now_ts_ms()
    })
    fill_price = res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    entry_final = float(fill_price)
    return {"symbol":sym,"dir":dir,"qty":qty,"entry":entry_final,"pos_side":pos_side}

def fetch_open_positions_real():
    """
    Binance üzerindeki gerçek pozisyonları çek.
    Hem BUY/SELL guard hem duplicate guard için.
    """
    out={"long":{}, "short":{},"long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            sym = p["symbol"]
            amt = float(p["positionAmt"])
            if amt>0:
                out["long"][sym]=amt
            elif amt<0:
                out["short"][sym]=abs(amt)
        out["long_count"]=len(out["long"])
        out["short_count"]=len(out["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")
    return out

# ====== TP/SL hesaplayıcı ======
def calc_tp_sl_prices(sym, entry_price, dir, tp_pct, sl_pct):
    """
    entry_price + yüzdeler -> TP/SL
    tickSize'e snap'lenir
    minPrice altına düşmez
    entry ile aşırı yakınsa küçük offset uygulanacak (futures_set_tp_sl içinde)
    """
    f          = get_symbol_filters(sym)
    tick       = float(f["tickSize"])
    min_price  = float(f["minPrice"])

    if dir=="UP":
        tp_raw = entry_price*(1+tp_pct)
        sl_raw = entry_price*(1-sl_pct)
        tp     = math.floor(tp_raw/tick)*tick
        sl     = math.ceil(sl_raw/tick)*tick
        if sl>=entry_price:
            sl = entry_price - tick
    else:
        tp_raw = entry_price*(1-tp_pct)
        sl_raw = entry_price*(1+sl_pct)
        tp     = math.ceil(tp_raw/tick)*tick
        sl     = math.floor(sl_raw/tick)*tick
        if sl<=entry_price:
            sl = entry_price + tick

    if sl < min_price:
        sl = min_price * 1.1
    if tp < min_price:
        tp = min_price * 1.2

    return tp, sl

def futures_set_tp_sl(sym, dir, qty, entry_exec, tp_pct, sl_pct):
    """
    Güvenli TP/SL:
    - entry_exec (gerçek fill fiyatı) baz alınır
    - tp/sl yüzdelerine +%0.2 buffer
    - tickSize + minPrice + positionSide + closePosition
    - reduceOnly gönderilmez (Binance -1106 fix)
    """
    try:
        buffer_extra = 0.002  # %0.2
        adj_tp_pct = tp_pct + buffer_extra
        adj_sl_pct = sl_pct + buffer_extra

        tp, sl = calc_tp_sl_prices(sym, entry_exec, dir, adj_tp_pct, adj_sl_pct)

        f          = get_symbol_filters(sym)
        tick       = float(f["tickSize"])
        pos_side   = "LONG" if dir=="UP" else "SHORT"
        side       = "SELL" if dir=="UP" else "BUY"

        decimals=0
        if "." in str(tick):
            decimals=len(str(tick).split(".")[1].rstrip("0"))
        fmt=f"{{:.{decimals}f}}"

        for t,p in [("TAKE_PROFIT_MARKET",tp),("STOP_MARKET",sl)]:
            payload={
                "symbol":sym,
                "side":side,
                "type":t,
                "stopPrice":fmt.format(p),
                "quantity":f"{qty}",
                "workingType":"MARK_PRICE",
                "closePosition":"true",
                "positionSide":pos_side,
                "timestamp":now_ts_ms()
            }
            _signed_request("POST","/fapi/v1/order",payload)

        msg=(f"✅ TP/SL SET {sym} {dir} "
             f"TP={fmt.format(tp)} SL={fmt.format(sl)} qty={qty}")
        tg_send(msg)
        log(msg)

    except Exception as e:
        err=f"[TP/SL ERR]{sym} {e}"
        tg_send(f"⚠️ {err}")
        log(err)

# ====== PARAM / STATE INIT ======
PARAM_DEFAULT={
    "SCALP_TP_PCT":0.006,
    "SCALP_SL_PCT":0.20,
    "TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30,
    "MAX_SELL":30,
    "ANGLE_MIN":0.0001
}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
if not isinstance(PARAM,dict):
    PARAM=PARAM_DEFAULT

STATE_DEFAULT={
    "bar_index":0,
    "last_report":0,
    "auto_trade_active":True,     # legacy global flag (koruyoruz)
    "last_api_check":0,
    "buy_active":True,            # ✅ yeni: long tarafı açık mı
    "sell_active":True            # ✅ yeni: short tarafı açık mı
}
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
if "auto_trade_active" not in STATE:
    STATE["auto_trade_active"]=True
if "buy_active" not in STATE:
    STATE["buy_active"]=True
if "sell_active" not in STATE:
    STATE["sell_active"]=True

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE,[])
AI_RL         = safe_load(AI_RL_FILE,[])
SIM_POSITIONS = safe_load(SIM_POS_FILE,[])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE,[])
# SIM_QUEUE global

# ================== INDICATORS ==================
def ema(vals,n):
    k=2/(n+1)
    e=[vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals,period=14):
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

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_val):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55 + diff*20 + ((rsi_val-50)/50)*15 + (atr_v/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    """
    - 65 <= power <= 75  => REAL trade sinyali
    - diğer her şey      => SIM (sadece simülasyona gider)
    """
    if p>=65 and p<=75:
        return "REAL","🟩"
    return "SIM",""

# ================== SIGNAL BUILDER ==================
def build_scalp_signal(sym,kl,bar_i):
    """
    1h confirmed bar reversal scalp sinyali:
    - EMA7 slope reversal (down->up => UP, up->down => DOWN)
    - slope_impulse ANGLE_MIN altındaysa sinyal yok
    - 24h |chg| >=10% ise sinyal yok
    - power -> tier_from_power()
    """
    if len(kl)<60:
        return None

    chg=futures_24h_change(sym)
    if abs(chg)>=10.0:
        return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    e7=ema(closes,7)

    s_now  = e7[-2]-e7[-5]
    s_prev = e7[-3]-e7[-6]
    slope_impulse=abs(s_now-s_prev)
    if slope_impulse < PARAM["ANGLE_MIN"]:
        return None

    if   s_prev<0 and s_now>0: direction="UP"
    elif s_prev>0 and s_now<0: direction="DOWN"
    else: return None

    atr_v=atr_like(highs,lows,closes)[-1]
    r_val=rsi(closes)[-1]
    pwr=calc_power(
        e7[-1],
        e7[-2],
        e7[-5],
        atr_v,
        closes[-1],
        r_val
    )

    tier,emoji=tier_from_power(pwr)
    if not tier:
        return None

    entry=closes[-1]
    if direction=="UP":
        tp=entry*(1+PARAM["SCALP_TP_PCT"])
        sl=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp=entry*(1-PARAM["SCALP_TP_PCT"])
        sl=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,
        "dir":direction,
        "tier":tier,        # "REAL" ya da "SIM"
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

def scan_symbol(sym,bar_i):
    kl=futures_get_klines(sym,"1h",200)
    if len(kl)<60:
        return None
    return build_scalp_signal(sym,kl,bar_i)

def run_parallel(symbols,bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(scan_symbol,s,bar_i) for s in symbols]
        for f in as_completed(futs):
            try:
                sig=f.result()
            except:
                sig=None
            if sig:
                out.append(sig)
    return out

# ================== SIM ENGINE ==================
def queue_sim_variants(sig):
    """
    Sadece SIM sinyaller sim kuyruğuna sokulur.
    REAL sinyaller gerçek trade açacağı için buraya girmez.
    """
    if sig["tier"]=="REAL":
        return
    delays=[30*60,60*60,90*60,120*60]  # 0.5h / 1h / 1.5h / 2h
    now_s=now_ts_s()
    for d in delays:
        SIM_QUEUE.append({
            "symbol":sig["symbol"],
            "dir":sig["dir"],
            "tier":sig["tier"],
            "entry":sig["entry"],
            "tp":sig["tp"],
            "sl":sig["sl"],
            "power":sig["power"],
            "created_ts":now_s,
            "open_after_ts":now_s+d
        })
    safe_save(SIM_POS_FILE,SIM_QUEUE)

def process_sim_queue_and_open_due():
    global SIM_POSITIONS
    now_s=now_ts_s()
    remain=[]
    opened_any=False
    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            SIM_POSITIONS.append({
                **q,
                "status":"OPEN",
                "open_ts":now_s,
                "open_time":now_local_iso()
            })
            opened_any=True
        else:
            remain.append(q)
    SIM_QUEUE[:]=remain
    if opened_any:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
    safe_save(SIM_POS_FILE,SIM_QUEUE)
def process_sim_closes():
    """
    Sim pozisyonlarda TP/SL vurdu mu?
    Vurduysa kapat, gain_pct hesapla, SIM_CLOSED'e taşı.
    """
    global SIM_POSITIONS
    if not SIM_POSITIONS:
        return
    still=[]
    changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")!="OPEN":
            continue
        last_price=futures_get_price(pos["symbol"])
        if last_price is None:
            still.append(pos)
            continue
        hit=None
        if pos["dir"]=="UP":
            if last_price>=pos["tp"]:
                hit="TP"
            elif last_price<=pos["sl"]:
                hit="SL"
        else:
            if last_price<=pos["tp"]:
                hit="TP"
            elif last_price>=pos["sl"]:
                hit="SL"
        if hit:
            close_time=now_local_iso()
            gain_pct=(
                (last_price/pos["entry"]-1.0)*100.0
                if pos["dir"]=="UP"
                else (pos["entry"]/last_price-1.0)*100.0
            )
            SIM_CLOSED.append({
                **pos,
                "status":"CLOSED",
                "close_time":close_time,
                "exit_price":last_price,
                "exit_reason":hit,
                "gain_pct":gain_pct
            })
            changed=True
        else:
            still.append(pos)
    SIM_POSITIONS=still
    if changed:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE,SIM_CLOSED)

# ================== GUARDS ==================
def dynamic_autotrade_state():
    """
    MAX_BUY / MAX_SELL guard (bağımsız kontrol)
    Long (BUY) ve Short (SELL) ayrı ayrı durur veya aktif olur.
    """
    live = fetch_open_positions_real()

    # BUY tarafı (long pozisyonlar)
    if live["long_count"] >= PARAM["MAX_BUY"]:
        if STATE.get("buy_active", True):
            STATE["buy_active"] = False
            tg_send("🟥 BUY işlemleri durduruldu — MAX_BUY limit doldu.")
    else:
        if not STATE.get("buy_active", True):
            STATE["buy_active"] = True
            tg_send("🟩 BUY işlemleri yeniden aktif.")

    # SELL tarafı (short pozisyonlar)
    if live["short_count"] >= PARAM["MAX_SELL"]:
        if STATE.get("sell_active", True):
            STATE["sell_active"] = False
            tg_send("🟥 SELL işlemleri durduruldu — MAX_SELL limit doldu.")
    else:
        if not STATE.get("sell_active", True):
            STATE["sell_active"] = True
            tg_send("🟩 SELL işlemleri yeniden aktif.")

    # Eski global flag'i de güncel tutuyoruz (bilgi amaçlı)
    STATE["auto_trade_active"] = STATE.get("buy_active",True) or STATE.get("sell_active",True)

    safe_save(STATE_FILE,STATE)

# ================== HEARTBEAT + STATUS (10 dk) ==================
def heartbeat_and_status_check():
    """
    Her 10 dakikada bir:
     - Binance API health
     - Bot durumu
    Telegram + log.
    """
    now=time.time()
    if now-STATE.get("last_api_check",0)<600:
        return
    STATE["last_api_check"]=now
    safe_save(STATE_FILE,STATE)

    # Binance health
    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift=abs(now_ts_ms()-st)
        ping_ok=requests.get(BINANCE_FAPI+"/fapi/v1/ping",timeout=5).status_code==200
        key_ok=True
        try:
            _=_signed_request("GET","/fapi/v2/account",{"timestamp":now_ts_ms()})
        except:
            key_ok=False

        hb_msg = (
            f"✅ HEARTBEAT drift={int(drift)}ms ping={ping_ok} key={key_ok}"
            if all([ping_ok,key_ok,drift<1500])
            else f"⚠️ HEARTBEAT ping={ping_ok} key={key_ok} drift={int(drift)}"
        )
        tg_send(hb_msg)
        log(hb_msg)

    except Exception as e:
        hb_err=f"❌ HEARTBEAT {e}"
        tg_send(hb_err)
        log(f"[HBERR]{e}")

    # Status snapshot
    live=fetch_open_positions_real()
    msg=(f"📊 STATUS bar:{STATE.get('bar_index',0)} "
         f"buy_active:{'✅' if STATE.get('buy_active',True) else '🟥'} "
         f"sell_active:{'✅' if STATE.get('sell_active',True) else '🟥'} "
         f"long:{live['long_count']} short:{live['short_count']} "
         f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
         f"sim_closed:{len(SIM_CLOSED)}")
    tg_send(msg)
    log(msg)

# ================== REAL TRADE EXECUTION ==================
def execute_real_trade(sig):
    """
    Sadece tier == "REAL" sinyaller gerçek pozisyon açar.
    TrendLock: aynı sembol aynı yönden tekrar tekrar açmayı engeller.
    DuplicateGuard: zaten aynı yönde açık poz varsa bir daha açmaz.
    BUY/SELL bağımsız guard'a bakarız.
    TP/SL: gerçek fill fiyatını baz alır + buffer.
    Telegram bildirimi burada yapılır.
    """
    if sig["tier"]!="REAL":
        return

    sym=sig["symbol"]
    direction=sig["dir"]  # "UP" (long) ya da "DOWN" (short)

    # BUY/SELL bağımsız guard kontrolü
    if direction == "UP" and not STATE.get("buy_active", True):
        return
    if direction == "DOWN" and not STATE.get("sell_active", True):
        return

    # Duplicate guard (gerçek pozlar)
    live=fetch_open_positions_real()
    if direction=="UP" and sym in live["long"]:
        return
    if direction=="DOWN" and sym in live["short"]:
        return

    # Miktar hesapla
    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        tg_send(f"❗ {sym} qty hesaplanamadı.")
        return

    # TrendLock anti-spam (aynı sembol/aynı yön kilidi)
    if TREND_LOCK.get(sym)==direction:
        return

    try:
        # 1) MARKET emri aç
        opened=open_market_position(sym,direction,qty)

        # 2) Gerçek fill price
        entry_exec = opened.get("entry")
        if not entry_exec or entry_exec <= 0:
            entry_exec = futures_get_price(sym)

        # 3) TP/SL emirlerini fill fiyatına göre kur
        futures_set_tp_sl(
            sym,
            direction,
            qty,
            entry_exec,
            PARAM["SCALP_TP_PCT"],
            PARAM["SCALP_SL_PCT"]
        )

        # 4) TrendLock set et (aynı yönden tekrar açmayı kilitle)
        TREND_LOCK[sym]=direction

        # 5) Telegram trade bildirimi (sadece REAL trade burada duyurulacak)
        tg_send(
            f"✅ REAL {sym} {direction} qty:{qty}\n"
            f"Entry:{entry_exec:.12f}\n"
            f"TP%:{PARAM['SCALP_TP_PCT']*100:.3f} "
            f"SL%:{PARAM['SCALP_SL_PCT']*100:.1f}\n"
            f"Pow:{sig['power']:.1f} born_bar:{sig['born_bar']}\n"
            f"time:{now_local_iso()}"
        )
        log(f"[REAL] {sym} {direction} {qty} entry={entry_exec}")

        # 6) RL log kaydı
        AI_RL.append({
            "time":now_local_iso(),
            "symbol":sym,
            "dir":direction,
            "entry":entry_exec,
            "tp_pct":PARAM["SCALP_TP_PCT"],
            "sl_pct":PARAM["SCALP_SL_PCT"],
            "power":sig["power"],
            "born_bar":sig["born_bar"]
        })
        safe_save(AI_RL_FILE,AI_RL)

    except Exception as e:
        tg_send(f"❌ OPEN ERR {sym} {e}")
        log(f"[OPEN ERR]{sym}{e}")

# ================== AI LOGGING / BACKUP ==================
def ai_log_signal(sig):
    """
    Her sinyali (REAL / SIM) kaydet.
    Telegram spam yok; dosyaya yazıyoruz.
    """
    AI_SIGNALS.append({
        "time":now_local_iso(),
        "symbol":sig["symbol"],
        "dir":sig["dir"],
        "tier":sig["tier"],
        "chg24h":sig["chg24h"],
        "power":sig["power"],
        "rsi":sig.get("rsi"),
        "atr":sig.get("atr"),
        "tp":sig["tp"],
        "sl":sig["sl"],
        "entry":sig["entry"]
    })
    safe_save(AI_SIGNALS_FILE,AI_SIGNALS)

def ai_update_analysis_snapshot():
    """
    küçük dashboard snapshot -> AI_ANALYSIS'e append
    """
    real_count = sum(1 for x in AI_SIGNALS if x.get("tier")=="REAL")
    sim_count  = sum(1 for x in AI_SIGNALS if x.get("tier")=="SIM")

    snapshot = {
        "time":now_local_iso(),
        "real_signals_total": real_count,
        "sim_signals_total": sim_count,
        "sim_open_count": len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
        "sim_closed_count": len(SIM_CLOSED)
    }

    AI_ANALYSIS.append(snapshot)
    safe_save(AI_ANALYSIS_FILE, AI_ANALYSIS)

def auto_report_if_due():
    """
    Her 4 saatte bir:
      - snapshot ekle
      - büyük jsonları küçült (son %20)
      - Telegram'a dump at
    """
    now_now = time.time()
    if now_now - STATE.get("last_report",0) < 14400:
        return

    ai_update_analysis_snapshot()

    files_to_push = [
        AI_SIGNALS_FILE,
        AI_ANALYSIS_FILE,
        AI_RL_FILE,
        SIM_POS_FILE,
        SIM_CLOSED_FILE
    ]

    for fpath in files_to_push:
        try:
            if os.path.exists(fpath):
                sz = os.path.getsize(fpath)
                if sz > 10*1024*1024:
                    with open(fpath,"r",encoding="utf-8") as f:
                        raw=f.read()
                    tail = raw[-int(len(raw)*0.2):]
                    with open(fpath,"w",encoding="utf-8") as f:
                        f.write(tail)
        except:
            pass

        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")

    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"] = now_now
    safe_save(STATE_FILE,STATE)

# ================== MAIN LOOP ==================
def main():
    tg_send("🚀 EMA ULTRA v15.9.15b PowerRangeRealOnly + Split Buy/Sell Guard başladı")
    log("[START] EMA ULTRA v15.9.15b PowerRangeRealOnly + Split Buy/Sell Guard")

    # Binance'ten USDT pair listesini çek
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols = [
            s["symbol"]
            for s in info["symbols"]
            if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"
        ]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}")
        symbols = []
    symbols.sort()

    while True:
        try:
            # bar sayaç
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1) sinyal tara
            sigs = run_parallel(symbols, bar_i)

            # 2) sinyalleri işle
            for sig in sigs:
                # 2a) sinyali AI_SIGNALS'e kaydet
                ai_log_signal(sig)

                # 2b) SIM sinyal ise sessiz sim kuyruğuna sok
                queue_sim_variants(sig)

                # 2c) guard update (buy_active / sell_active güncelle)
                dynamic_autotrade_state()

                # 2d) REAL sinyalse gerçek trade aç ve TP/SL kur
                execute_real_trade(sig)

            # 3) sim kuyruğunda zamanı gelenleri OPEN yap
            process_sim_queue_and_open_due()

            # 4) açık sim pozisyonlarında TP veya SL tetiklendiyse kapat
            process_sim_closes()

            # 5) 4 saatlik backup gerekiyorsa gönder
            auto_report_if_due()

            # 6) heartbeat + status (10 dk'da bir)
            heartbeat_and_status_check()

            # 7) state kaydet
            safe_save(STATE_FILE, STATE)

            # 8) bekle
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ================== ENTRYPOINT ==================
if __name__=="__main__":
    main()
