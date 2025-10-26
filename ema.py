import os, json, time, requests, hmac, hashlib, threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.5 — ConfirmedBar + MsgLock + SilentSim + TP/SL Buffer
# + Heartbeat (10dk Binance API health check)
# ==============================================================================

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

BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
PRECISION_CACHE = {}
TREND_LOCK = {}
SIM_QUEUE = []

# ================= SAFE IO =================
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
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp,p)
    except Exception as e:
        print("[SAVE ERR]",e,flush=True)

def log(msg):
    print(msg,flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except:
        pass

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())
def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

# ================= TELEGRAM =================
def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID,"text":t},
            timeout=10
        )
    except:
        pass

def tg_send_file(p,cap):
    if not BOT_TOKEN or not CHAT_ID or not os.path.exists(p):
        return
    try:
        with open(p,"rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id":CHAT_ID,"caption":cap},
                files={"document":(os.path.basename(p),f)},
                timeout=30
            )
    except:
        pass

# ================= BINANCE CORE HELPERS =================
def _signed_request(m,path,payload):
    """
    m: "GET" or "POST"
    path: "/fapi/v1/..."
    payload: dict (timestamp dahil)
    """
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    h={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    if m=="POST":
        r=requests.post(url, headers=h, timeout=10)
    else:
        r=requests.get(url, headers=h, timeout=10)
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    """
    Precision cache:
    - LOT_SIZE.stepSize -> quantity precision
    - PRICE_FILTER.tickSize -> price precision
    """
    if sym in PRECISION_CACHE:
        return PRECISION_CACHE[sym]
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        step=float(lot.get("stepSize","1"))
        tick=float(pricef.get("tickSize","0.01"))
        if tick<1e-10: tick=0.00000001
        if step<1e-10: step=0.00000001
        PRECISION_CACHE[sym]={"stepSize":step,"tickSize":tick}
    except Exception as e:
        log(f"[PRECISION WARN]{sym}{e}")
        PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001}
    return PRECISION_CACHE[sym]

def round_nearest(x,s):
    return round(round(x/s)*s,12) if s else x

def adjust_precision(sym,v,mode="price"):
    f=get_symbol_filters(sym)
    step=f["tickSize"] if mode=="price" else f["stepSize"]
    adj=round_nearest(v,step)
    if adj==0:
        adj=v
    return float(f"{adj:.12f}")

def futures_get_price(sym):
    try:
        r=requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/price",
            params={"symbol":sym},
            timeout=5
        ).json()
        return float(r["price"])
    except:
        return None

def futures_24h_change(sym):
    try:
        r=requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/24hr",
            params={"symbol":sym},
            timeout=5
        ).json()
        return float(r["priceChangePercent"])
    except:
        return 0.0

def futures_get_klines(sym,it,lim):
    """
    Kapanmamış son mumu atıyoruz (confirmed bar mantığı).
    """
    try:
        r=requests.get(
            BINANCE_FAPI+"/fapi/v1/klines",
            params={"symbol":sym,"interval":it,"limit":lim},
            timeout=10
        ).json()
        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        if r and int(r[-1][6])>now_ms:
            r = r[:-1]
        return r
    except:
        return []

def calc_order_qty(sym,entry,usd):
    raw=usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def open_market_position(sym,dir,qty):
    """
    MARKET order açar ve ortalama fiyatı döner.
    Hedge mode varsayımı:
      dir UP   -> LONG / BUY
      dir DOWN -> SHORT / SELL
    """
    side="BUY" if dir=="UP" else "SELL"
    pos="LONG" if dir=="UP" else "SHORT"

    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,
        "side":side,
        "type":"MARKET",
        "quantity":f"{qty}",
        "positionSide":pos,
        "timestamp":now_ts_ms()
    })

    p=res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    return {
        "symbol":sym,
        "dir":dir,
        "positionSide":pos,
        "qty":qty,
        "entry":adjust_precision(sym,float(p),"price")
    }

def futures_set_tp_sl(sym,dir,qty,entry,tp_pct,sl_pct):
    """
    TAKE_PROFIT_MARKET / STOP_MARKET kuruyoruz.
    stop_buffer = %0.1 güvenlik payı -> 'Order would immediately trigger' engeli.
    """
    pos="LONG" if dir=="UP" else "SHORT"
    side="SELL" if dir=="UP" else "BUY"
    stop_buffer=0.001  # %0.1

    if dir=="UP":
        tp_raw=entry*(1+tp_pct+stop_buffer)
        sl_raw=entry*(1-sl_pct-stop_buffer)
    else:
        tp_raw=entry*(1-tp_pct-stop_buffer)
        sl_raw=entry*(1+sl_pct+stop_buffer)

    tp=adjust_precision(sym,tp_raw,"price")
    sl=adjust_precision(sym,sl_raw,"price")

    for order_type, stop_p in [("TAKE_PROFIT_MARKET",tp),("STOP_MARKET",sl)]:
        pay={
            "symbol":sym,
            "side":side,
            "type":order_type,
            "stopPrice":f"{stop_p:.12f}",
            "quantity":f"{qty}",
            "workingType":"MARK_PRICE",
            "closePosition":"true",
            "positionSide":pos,
            "timestamp":now_ts_ms()
        }
        try:
            _signed_request("POST","/fapi/v1/order",pay)
        except Exception as e:
            tg_send(f"⚠️ TP/SL ERR {sym} {e}")
            log(f"[TP/SL ERR]{sym}{e}")

def fetch_open_positions_real():
    """
    Binance üstündeki gerçek pozisyonları çeker.
    """
    out={"long":{}, "short":{},"long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            sym=p["symbol"]; amt=float(p["positionAmt"])
            if amt>0:
                out["long"][sym]=amt
            elif amt<0:
                out["short"][sym]=abs(amt)
        out["long_count"]=len(out["long"])
        out["short_count"]=len(out["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")
    return out

# ================= PARAM / STATE / MEMORY =================
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
    "auto_trade_active":True,
    "last_api_check":0
}
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
if "auto_trade_active" not in STATE:
    STATE["auto_trade_active"]=True
if "last_api_check" not in STATE:
    STATE["last_api_check"]=0

AI_SIGNALS   = safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS  = safe_load(AI_ANALYSIS_FILE,[])
AI_RL        = safe_load(AI_RL_FILE,[])
SIM_POSITIONS= safe_load(SIM_POS_FILE,[])
SIM_CLOSED   = safe_load(SIM_CLOSED_FILE,[])

# ================== API CHECK / HEARTBEAT ==================
def binance_api_check():
    """
    Binance Futures API health & key validity check.
    Ping, drift, key test.
    """
    try:
        # server time drift
        server_time = requests.get(BINANCE_FAPI + "/fapi/v1/time", timeout=5).json()["serverTime"]
        local_time = now_ts_ms()
        drift_ms = abs(local_time - server_time)
        drift_ok = drift_ms < 1000  # <1s tolerans

        # ping
        ping_ok = requests.get(BINANCE_FAPI + "/fapi/v1/ping", timeout=5).status_code == 200

        # api key validity
        try:
            _ = _signed_request("GET", "/fapi/v2/account", {"timestamp": now_ts_ms()})
            key_ok = True
        except Exception as e:
            key_ok = False
            log(f"[API CHECK] key test failed: {e}")

        return {
            "ping_ok": ping_ok,
            "drift_ms": drift_ms,
            "drift_ok": drift_ok,
            "key_ok": key_ok
        }

    except Exception as e:
        log(f"[API CHECK ERR] {e}")
        return {"error": str(e)}

def heartbeat_api_check(state):
    """
    Her 10 dakikada bir Binance bağlantısını test eder.
    Sorun varsa Telegram uyarısı yollar.
    Normal ise sadece log'a HEARTBEAT yazar.
    """
    now_t = time.time()
    last_check = state.get("last_api_check", 0)

    # 600 sn = 10 dakika
    if now_t - last_check < 600:
        return

    # update last check time
    state["last_api_check"] = now_t
    safe_save(STATE_FILE, state)

    check = binance_api_check()

    if "error" in check:
        tg_send(f"❌ API Check Error: {check['error']}")
        return

    ping_ok  = check["ping_ok"]
    drift_ok = check["drift_ok"]
    key_ok   = check["key_ok"]
    drift_ms = check["drift_ms"]

    if not all([ping_ok, drift_ok, key_ok]):
        msg = (
            "⚠️ Binance API sorun tespit edildi:\n"
            f"Ping:{ping_ok} Key:{key_ok} Drift:{drift_ms} ms"
        )
        tg_send(msg)
        log(msg)
    else:
        log(f"[HEARTBEAT] Binance API OK — drift {drift_ms} ms")
# ================= INDICATORS =================
def ema(vals,n):
    k=2/(n+1)
    e=[vals[0]]
    for v in vals[1:]:
        e.append(v*k + e[-1]*(1-k))
    return e

def rsi(vals,period=14):
    """
    RSI hesaplama (kapanmış mum seti üstünde).
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

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_val):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55+diff*20+((rsi_val-50)/50)*15+(atr_v/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    if p>=75: return "ULTRA","🟩"
    if p>=68: return "PREMIUM","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""

# ================== SIGNAL BUILDER (Confirmed 1h Bar) ==================
def build_scalp_signal(sym, kl, bar_i):
    """
    Confirmed bar reversal:
    - sadece kapanmış barları kullanır
    - günlük |chg24h| >10% ise sinyal yok
    - angle (slope change) ANGLE_MIN altındaysa sinyal yok
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

    # slope impulse: son kapanmış barlar
    s_now  = e7[-2]-e7[-5]
    s_prev = e7[-3]-e7[-6]
    slope_impulse=abs(s_now-s_prev)
    if slope_impulse < PARAM["ANGLE_MIN"]:
        return None

    if s_prev<0 and s_now>0:
        direction="UP"
    elif s_prev>0 and s_now<0:
        direction="DOWN"
    else:
        return None

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
    if not entry:
        return None

    if direction=="UP":
        tp=entry*(1+PARAM["SCALP_TP_PCT"])
        sl=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp=entry*(1-PARAM["SCALP_TP_PCT"])
        sl=entry*(1+PARAM["SCALP_SL_PCT"])

    sig={
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
                if sig:
                    out.append(sig)
            except:
                pass
    return out

# ================== SIM ENGINE ==================
def queue_sim_variants(sig):
    """
    PREMIUM / NORMAL sinyaller sessiz simülasyona alınır.
    ULTRA simülasyona girmez.
    30/60/90/120 dk gecikmeli varyantlar.
    """
    if sig["tier"]=="ULTRA":
        return
    delays=[30*60,60*60,90*60,120*60]
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
    """
    Vakti gelen simülasyon pozisyonlarını aç (sessiz).
    """
    now_s=now_ts_s()
    remain=[]
    opened_any=False
    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            SIM_POSITIONS.append({
                **q,
                "status":"OPEN",
                "open_ts": now_s,
                "open_time": now_local_iso()
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
    Açık sim pozisyonlar TP/SL'e ulaştı mı?
    stop_buffer = 0.001 (%0.1 tolerans) gerçek emir davranışıyla uyumlu.
    """
    global SIM_POSITIONS
    if not SIM_POSITIONS:
        return

    stop_buffer = 0.001
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
            tp_trigger = pos["tp"] * (1 - stop_buffer)
            sl_trigger = pos["sl"] * (1 + stop_buffer)
            if last_price >= tp_trigger:
                hit="TP"
            elif last_price <= sl_trigger:
                hit="SL"
        else: # DOWN
            tp_trigger = pos["tp"] * (1 + stop_buffer)
            sl_trigger = pos["sl"] * (1 - stop_buffer)
            if last_price <= tp_trigger:
                hit="TP"
            elif last_price >= sl_trigger:
                hit="SL"

        if hit:
            close_time = now_local_iso()
            gain_pct = (
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

    SIM_POSITIONS = still
    if changed:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE,SIM_CLOSED)

# ================== REAL TRADE CONTROL ==================
def dynamic_autotrade_state():
    """
    MAX BUY / SELL limit kontrolü.
    Aşınca auto_trade_active=False + Telegram uyarısı.
    Limit düşerse yeniden True yapar ve haber verir.
    """
    live=fetch_open_positions_real()

    if STATE["auto_trade_active"]:
        if (live["long_count"]>=PARAM["MAX_BUY"] or
            live["short_count"]>=PARAM["MAX_SELL"]):
            STATE["auto_trade_active"]=False
            tg_send("🟥 AutoTrade durduruldu — limit doldu.")
    else:
        if (live["long_count"]<PARAM["MAX_BUY"] and
            live["short_count"]<PARAM["MAX_SELL"]):
            STATE["auto_trade_active"]=True
            tg_send("🟩 AutoTrade yeniden aktif.")

    safe_save(STATE_FILE,STATE)

def execute_real_trade(sig):
    """
    Sadece ULTRA sinyaller gerçek trade açar.
    TrendLock:
      aynı sembol aynı yönde kilitliyse tekrar açmaz.
    DuplicateGuard:
      o yönde zaten açık pozisyon varsa tekrar açmaz.
    """
    if sig["tier"]!="ULTRA":
        return
    if not STATE.get("auto_trade_active",True):
        return

    sym=sig["symbol"]
    direction=sig["dir"]

    # TrendLock check
    if TREND_LOCK.get(sym)==direction:
        return

    # DuplicateGuard (gerçekte zaten açık mı?)
    live=fetch_open_positions_real()
    if direction=="UP" and sym in live["long"]:
        return
    if direction=="DOWN" and sym in live["short"]:
        return

    # qty hesapla
    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        tg_send(f"❗ {sym} qty hesaplanamadı.")
        return

    # emir aç
    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened["entry"]

        # TP/SL koy
        futures_set_tp_sl(
            sym,
            direction,
            qty,
            entry_exec,
            PARAM["SCALP_TP_PCT"],
            PARAM["SCALP_SL_PCT"]
        )

        # TrendLock set
        TREND_LOCK[sym]=direction

        # Telegram bildirimi
        tg_send(
            f"✅ REAL {sym} {direction} ULTRA qty:{qty}\n"
            f"Entry:{entry_exec:.12f}\n"
            f"TP%:{PARAM['SCALP_TP_PCT']*100:.3f} "
            f"SL%:{PARAM['SCALP_SL_PCT']*100:.1f}\n"
            f"time:{now_local_iso()}"
        )
        log(f"[REAL] {sym} {direction} {qty} entry={entry_exec}")

        # RL öğrenme log'u
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
# ================== AI LOGGING / ANALYSIS / REPORT ==================
def ai_log_signal(sig):
    """
    Her sinyali kaydet (ULTRA / PREMIUM / NORMAL).
    Telegram yok, sessiz.
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
    safe_save(AI_SIGNALS_FILE, AI_SIGNALS)

def ai_update_analysis_snapshot():
    """
    Küçük snapshot: toplam sinyal sayıları,
    açık sim sayısı, kapalı sim sayısı.
    """
    ultra_count = sum(1 for x in AI_SIGNALS if x.get("tier")=="ULTRA")
    prem_count  = sum(1 for x in AI_SIGNALS if x.get("tier")=="PREMIUM")
    norm_count  = sum(1 for x in AI_SIGNALS if x.get("tier")=="NORMAL")

    snapshot = {
        "time":now_local_iso(),
        "ultra_signals_total": ultra_count,
        "premium_signals_total": prem_count,
        "normal_signals_total": norm_count,
        "sim_open_count": len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
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
    Sonra küçük status mesajı.
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
        # dosya şişmişse son %20'si kalsın
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
    safe_save(STATE_FILE, STATE)

# ================== MAIN LOOP ==================
def main():
    tg_send("🚀 EMA ULTRA v15.9.5 başladı (Heartbeat + ConfirmedBar + SilentSim + TP/SL Buffer)")
    log("[START] EMA ULTRA v15.9.5 started")

    # Binance USDT sembollerini çek
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo", timeout=10).json()
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
            # bar index ilerlet
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1) confirmed bar sinyalleri tara
            sigs = run_parallel(symbols, bar_i)

            # 2) sinyaller üzerinde çalış
            for sig in sigs:
                # 2a) tüm sinyalleri kaydet
                ai_log_signal(sig)

                # 2b) PREMIUM / NORMAL -> sessiz sim kuyruğu
                queue_sim_variants(sig)

                # 2c) ULTRA değilse devam etme
                if sig["tier"] != "ULTRA":
                    continue

                sym = sig["symbol"]
                direction = sig["dir"]

                # TrendLock check: aynı sembol aynı yön ise mesaj da yok trade de yok
                if TREND_LOCK.get(sym) == direction:
                    continue

                # Telegram bildirimi (yeni yönlü ULTRA sinyal)
                tg_send(
                    f"{sig['emoji']} {sig['tier']} {sym} {direction}\n"
                    f"Pow:{sig['power']:.1f} RSI:{sig.get('rsi',0):.1f} "
                    f"ATR:{sig.get('atr',0):.4f} 24hΔ:{sig['chg24h']:.2f}%\n"
                    f"Entry:{sig['entry']:.12f}\nTP:{sig['tp']:.12f}\nSL:{sig['sl']:.12f}\n"
                    f"born_bar:{sig['born_bar']}"
                )
                log(f"[ULTRA SIG] {sym} {direction} Pow:{sig['power']:.1f} 24hΔ:{sig['chg24h']:.2f}%")

                # AutoTrade guard update (MAX_BUY / MAX_SELL)
                dynamic_autotrade_state()

                # Gerçek trade açmayı dene
                execute_real_trade(sig)

            # 3) sim engine
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4) periyodik rapor (4 saatte bir Telegram backup)
            auto_report_if_due()

            # 5) heartbeat (10 dakikada bir Binance API check)
            heartbeat_api_check(STATE)

            # 6) state kaydet
            safe_save(STATE_FILE, STATE)

            # döngü bekleme
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ================== ENTRYPOINT ==================
if __name__ == "__main__":
    main()
