import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================
# 📘 EMA ULTRA v15.9.16 FULL+HB+ApproveSim+RL+PowerGuard+RetryTP+DirectionalLimit+SilentSignals
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
            tmp=p+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(d,f,ensure_ascii=False,indent=2)
                f.flush(); os.fsync(f.fileno())
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
def now_local_iso(): return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

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
    r=(requests.post(url,headers=headers,timeout=10)
       if m=="POST" else requests.get(url,headers=headers,timeout=10))
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE:
        return PRECISION_CACHE[sym]
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        PRECISION_CACHE[sym]={
            "stepSize":float(lot.get("stepSize","1")),
            "tickSize":float(pricef.get("tickSize","0.01")),
            "minPrice":float(pricef.get("minPrice","0.00000001"))
        }
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
    raw=usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except: return None

def futures_24h_change(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",params={"symbol":sym},timeout=5).json()
        return float(r["priceChangePercent"])
    except: return 0.0

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",params={"symbol":sym,"interval":it,"limit":lim},timeout=10).json()
        nowms=now_ts_ms()
        if r and int(r[-1][6])>nowms: r=r[:-1]
        return r
    except: return []

def open_market_position(sym,dir,qty):
    side="BUY" if dir=="UP" else "SELL"
    pos_side="LONG" if dir=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET",
        "quantity":f"{qty}","positionSide":pos_side,"timestamp":now_ts_ms()
    })
    fill_price=res.get("avgPrice") or res.get("price") or futures_get_price(sym)
    return {"symbol":sym,"dir":dir,"qty":qty,"entry":float(fill_price),"pos_side":pos_side}

# ==============================================================
# 🛠️ TP/SL RETRY FIX (5 tries if stop price < 0)
# ==============================================================

def futures_set_tp_sl(sym,dir,qty,entry_exec,tp_pct,sl_pct):
    try:
        buffer_extra=0.002
        adj_tp_pct=tp_pct+buffer_extra
        adj_sl_pct=sl_pct+buffer_extra
        f=get_symbol_filters(sym)
        tick=float(f["tickSize"])
        min_price=float(f["minPrice"])
        pos_side="LONG" if dir=="UP" else "SHORT"
        side="SELL" if dir=="UP" else "BUY"

        def calc_prices():
            if dir=="UP":
                tp_raw=entry_exec*(1+adj_tp_pct)
                sl_raw=entry_exec*(1-adj_sl_pct)
                tp=max(min_price,math.floor(tp_raw/tick)*tick)
                sl=max(min_price,math.ceil(sl_raw/tick)*tick)
                if sl>=entry_exec: sl=entry_exec-tick
            else:
                tp_raw=entry_exec*(1-adj_tp_pct)
                sl_raw=entry_exec*(1+adj_sl_pct)
                tp=max(min_price,math.ceil(tp_raw/tick)*tick)
                sl=max(min_price,math.floor(sl_raw/tick)*tick)
                if sl<=entry_exec: sl=entry_exec+tick
            return tp,sl

        tp,sl=calc_prices()
        decimals=0
        if "." in str(tick):
            decimals=len(str(tick).split(".")[1].rstrip("0"))
        fmt=f"{{:.{decimals}f}}"

        for ttype,price in [("TAKE_PROFIT_MARKET",tp),("STOP_MARKET",sl)]:
            success=False
            for attempt in range(5):
                try:
                    payload={"symbol":sym,"side":side,"type":ttype,
                             "stopPrice":fmt.format(price),"quantity":f"{qty}",
                             "workingType":"MARK_PRICE","closePosition":"true",
                             "positionSide":pos_side,"timestamp":now_ts_ms()}
                    _signed_request("POST","/fapi/v1/order",payload)
                    success=True; break
                except Exception as e:
                    if "Stop price less than zero" in str(e):
                        price=price+tick if dir=="UP" else price-tick
                        time.sleep(0.2)
                    else:
                        time.sleep(0.2)
                        continue
            if not success:
                log(f"[TP/SL WARN] {sym} {ttype} price={price} 5 denemede de hata verdi.")

        msg=f"✅ TP/SL SET {sym} {dir} TP={fmt.format(tp)} SL={fmt.format(sl)} qty={qty}"
        tg_send(msg); log(msg)

    except Exception as e:
        err=f"[TP/SL ERR]{sym}{e}"
        tg_send(f"⚠️ {err}"); log(err)
# ==============================================================
# 📊 PARAM / STATE INIT
# ==============================================================

PARAM_DEFAULT = {
    "SCALP_TP_PCT":0.006,
    "SCALP_SL_PCT":0.20,
    "TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30,
    "MAX_SELL":30,
    "ANGLE_MIN":0.0001
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index":0,
    "last_report":0,
    "auto_trade_active":True,
    "last_api_check":0,
    # yeni alanlar directional guard için, default False
    "long_blocked": False,
    "short_blocked": False
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
if "auto_trade_active" not in STATE:
    STATE["auto_trade_active"]=True
if "long_blocked" not in STATE:
    STATE["long_blocked"]=False
if "short_blocked" not in STATE:
    STATE["short_blocked"]=False

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE,[])
AI_RL         = safe_load(AI_RL_FILE,[])
SIM_POSITIONS = safe_load(SIM_POS_FILE,[])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE,[])

# ==============================================================
# 📈 INDICATORS
# ==============================================================

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
    # ULTRA tanımı hala duruyor ama artık ULTRA otomatik trade açmayacak
    if p>=75: return "ULTRA","🟩"
    if p>=68: return "PREMIUM","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""

# ==============================================================
# 📡 SIGNAL BUILDER
# ==============================================================

def build_scalp_signal(sym,kl,bar_i):
    """
    1h confirmed bar reversal scalp sinyali:
    - EMA7 slope reversal (down->up => UP, up->down => DOWN)
    - slope_impulse ANGLE_MIN altındaysa sinyal yok
    - 24h |chg| >=10% ise sinyal yok
    - güç (power) -> tier seçim
    """
    if len(kl)<60:
        return None

    chg = futures_24h_change(sym)
    if abs(chg)>=10.0:
        return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    e7=ema(closes,7)

    # slope değişimi (yalnız kapanmış mumlardan)
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

    tier,emoji = tier_from_power(pwr)
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

# ==============================================================
# 📝 AI LOGGING / SNAPSHOT
# ==============================================================

def ai_log_signal(sig):
    """
    Her sinyali (ULTRA / PREMIUM / NORMAL) kaydet.
    born_bar dahil -> RL için.
    """
    AI_SIGNALS.append({
        "time": now_local_iso(),
        "symbol": sig["symbol"],
        "dir": sig["dir"],
        "tier": sig["tier"],
        "chg24h": sig["chg24h"],
        "power": sig["power"],
        "rsi": sig.get("rsi"),
        "atr": sig.get("atr"),
        "tp": sig["tp"],
        "sl": sig["sl"],
        "entry": sig["entry"],
        "born_bar": sig.get("born_bar")
    })
    safe_save(AI_SIGNALS_FILE,AI_SIGNALS)

def ai_update_analysis_snapshot():
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

# ==============================================================
# 🤖 RL ENRICH
# ==============================================================

def enrich_position_with_ai_context(pos):
    """
    RL için gerekli eksik state bilgilerini AI_SIGNALS'dan ekler.
    - symbol ve entry yakınlığına göre eşleşme yapar
    - rsi, atr, chg24h, born_bar alanlarını CLOSED trade'e inject eder
    """
    best = None
    for s in reversed(AI_SIGNALS):
        if s.get("symbol") != pos.get("symbol"):
            continue
        e_sig = s.get("entry")
        e_pos = pos.get("entry")
        if not e_sig or not e_pos:
            continue
        # entry farkı küçükse eşleştir (örn %0.2 tolerans)
        if abs(e_sig - e_pos) / max(e_sig,1e-12) < 0.002:
            best = s
            break

    if best:
        pos["rsi"]      = best.get("rsi")
        pos["atr"]      = best.get("atr")
        pos["chg24h"]   = best.get("chg24h")
        pos["born_bar"] = best.get("born_bar")
    return pos

# ==============================================================
# 🧪 SIM ENGINE (approve delay buckets)
# ==============================================================

def queue_sim_variants(sig):
    """
    TÜM sinyaller (ULTRA dahil) için 4 approve varyantı:
      - 30dk, 1h, 1.5h, 2h
    RL analizi için 'approve_delay_min' ve 'approve_label' tutar.
    """
    delays=[
        (30*60,  "approve_30m"),
        (60*60,  "approve_1h"),
        (90*60,  "approve_1h30"),
        (120*60, "approve_2h")
    ]

    now_s=now_ts_s()
    for secs,label in delays:
        SIM_QUEUE.append({
            "symbol":sig["symbol"],
            "dir":sig["dir"],
            "tier":sig["tier"],
            "entry":sig["entry"],
            "tp":sig["tp"],
            "sl":sig["sl"],
            "power":sig["power"],
            "approve_delay_min":secs/60,
            "approve_label":label,
            "created_ts":now_s,
            "open_after_ts":now_s+secs,
            "status":"PENDING"
        })

    safe_save(SIM_POS_FILE,SIM_QUEUE)

def process_sim_queue_and_open_due():
    """
    Kuyruktaki bekleyen sim varyantları zamanı gelince OPEN yap.
    """
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
                "open_time":now_local_iso(),
                "approve_bucket": q.get("approve_label")
            })
            opened_any=True
        else:
            remain.append(q)

    SIM_QUEUE[:] = remain

    if opened_any:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)

    # kuyruğun update'i de yaz
    safe_save(SIM_POS_FILE,SIM_QUEUE)

def process_sim_closes():
    """
    Sim pozisyonlarda TP/SL vurdu mu?
    Vurduysa kapat, gain_pct hesapla, SIM_CLOSED'e taşı.
    RL enrichment ile ATR/RSI/CHG24H/BORN_BAR inject et.
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
            if last_price>=pos["tp"]: hit="TP"
            elif last_price<=pos["sl"]: hit="SL"
        else:
            if last_price<=pos["tp"]: hit="TP"
            elif last_price>=pos["sl"]: hit="SL"

        if hit:
            close_time=now_local_iso()
            gain_pct=(
                (last_price/pos["entry"]-1.0)*100.0
                if pos["dir"]=="UP"
                else (pos["entry"]/last_price-1.0)*100.0
            )

            enriched = enrich_position_with_ai_context(dict(pos))

            SIM_CLOSED.append({
                **enriched,
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

# ==============================================================
# 🚦 DIRECTIONAL LIMIT GUARD
# ==============================================================

def fetch_open_positions_real():
    """
    Binance üzerindeki gerçek pozisyonları çek.
    Hem directional guard hem duplicate guard için.
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

def dynamic_autotrade_state():
    """
    Yön bazlı limit kontrolü:
    - long_count MAX_BUY'ı geçtiyse yeni long açılmaz (STATE.long_blocked=True)
    - short_count MAX_SELL'i geçtiyse yeni short açılmaz (STATE.short_blocked=True)
    - her iki taraf da doluysa auto_trade_active False olur (tam kilit)
    - biri serbestse auto_trade_active True olur (kısmi açık)
    """
    live=fetch_open_positions_real()

    STATE["long_blocked"]  = live["long_count"]  >= PARAM["MAX_BUY"]
    STATE["short_blocked"] = live["short_count"] >= PARAM["MAX_SELL"]

    both_blocked = STATE["long_blocked"] and STATE["short_blocked"]

    # global auto_trade_active ayarı
    if both_blocked:
        if STATE.get("auto_trade_active",True):
            STATE["auto_trade_active"]=False
            tg_send("🟥 AutoTrade tamamen durdu — hem LONG hem SHORT limit dolu.")
    else:
        if not STATE.get("auto_trade_active",True):
            STATE["auto_trade_active"]=True
            tg_send("🟩 AutoTrade aktif — en az bir yön hâlâ serbest.")

    safe_save(STATE_FILE,STATE)

# ==============================================================
# ❤️ HEARTBEAT / STATUS
# ==============================================================

def heartbeat_and_status_check():
    """
    Her 10 dakikada bir Binance health ve durum raporu.
    (Bu Telegram'a gider çünkü kritik sağlık sinyali)
    """
    now=time.time()
    if now-STATE.get("last_api_check",0)<600:
        return
    STATE["last_api_check"]=now
    safe_save(STATE_FILE,STATE)

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

    live=fetch_open_positions_real()
    msg=(f"📊 STATUS bar:{STATE.get('bar_index',0)} "
         f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
         f"long_blocked:{STATE.get('long_blocked')} "
         f"short_blocked:{STATE.get('short_blocked')} "
         f"long:{live['long_count']} short:{live['short_count']} "
         f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
         f"sim_closed:{len(SIM_CLOSED)}")
    tg_send(msg)
    log(msg)

# ==============================================================
# ⏰ AUTO BACKUP (4 HOURS)
# ==============================================================

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
# ==============================================================
# 🚀 REAL TRADE EXECUTION (Power 65–74 only)
# ==============================================================

def execute_real_trade(sig):
    """
    Gerçek emir açma:
      - Artık sadece 65 <= power < 75 arası sinyaller açılır.
      - ULTRA sinyaller (p>=75) simülasyona gider, gerçek emir açmaz.
      - Directional guard (long_blocked / short_blocked) uygulanır.
      - TrendLock aynı yönde spam engeller.
    """
    pwr = sig["power"]
    if not (65 <= pwr < 75):
        return

    if not STATE.get("auto_trade_active",True):
        return

    sym = sig["symbol"]
    direction = sig["dir"]

    # yön bazlı limit guard
    if direction == "UP" and STATE.get("long_blocked"):
        log(f"[GUARD] LONG blocked for {sym} limit dolu.")
        return
    if direction == "DOWN" and STATE.get("short_blocked"):
        log(f"[GUARD] SHORT blocked for {sym} limit dolu.")
        return

    # TrendLock (aynı symbol aynı yön tekrar açmasın)
    if TREND_LOCK.get(sym) == direction:
        return

    # Duplicate guard
    live = fetch_open_positions_real()
    if direction == "UP" and sym in live["long"]:
        return
    if direction == "DOWN" and sym in live["short"]:
        return

    qty = calc_order_qty(sym, sig["entry"], PARAM["TRADE_SIZE_USDT"])
    if not qty or qty <= 0:
        tg_send(f"❗ {sym} qty hesaplanamadı.")
        return

    try:
        opened = open_market_position(sym, direction, qty)
        entry_exec = opened.get("entry") or futures_get_price(sym)

        # TP/SL emirleri fill fiyatına göre kur
        futures_set_tp_sl(
            sym,
            direction,
            qty,
            entry_exec,
            PARAM["SCALP_TP_PCT"],
            PARAM["SCALP_SL_PCT"]
        )

        # TrendLock set
        TREND_LOCK[sym] = direction

        # ✅ Telegram bildirimi sadece burada gönderilecek
        tg_send(
            f"✅ REAL {sym} {direction} qty:{qty}\n"
            f"Power:{pwr:.2f}\n"
            f"Entry:{entry_exec:.12f}\n"
            f"TP%:{PARAM['SCALP_TP_PCT']*100:.3f} "
            f"SL%:{PARAM['SCALP_SL_PCT']*100:.1f}\n"
            f"time:{now_local_iso()}"
        )

        log(f"[REAL] {sym} {direction} qty={qty} entry={entry_exec:.6f} pwr={pwr:.2f}")

        # RL kaydı
        AI_RL.append({
            "time": now_local_iso(),
            "symbol": sym,
            "dir": direction,
            "entry": entry_exec,
            "tp_pct": PARAM["SCALP_TP_PCT"],
            "sl_pct": PARAM["SCALP_SL_PCT"],
            "power": sig["power"],
            "born_bar": sig["born_bar"]
        })
        safe_save(AI_RL_FILE, AI_RL)

    except Exception as e:
        tg_send(f"❌ OPEN ERR {sym} {e}")
        log(f"[OPEN ERR]{sym}{e}")

# ==============================================================
# 🔁 MAIN LOOP
# ==============================================================

def main():
    tg_send("🚀 EMA ULTRA v15.9.16 FULL başladı")
    log("[START] EMA ULTRA v15.9.16 FULL")

    # USDT pair listesi çek
    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols = [s["symbol"] for s in info["symbols"] if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}")
        symbols = []
    symbols.sort()

    while True:
        try:
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1️⃣ sinyalleri tara
            sigs = run_parallel(symbols, bar_i)

            # 2️⃣ sinyalleri işle
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)     # her sinyal simülasyona gider

                # artık Telegram'a sinyal mesajı atılmaz (Silent mode)
                log(f"[SIG] {sig['symbol']} {sig['dir']} tier={sig['tier']} pwr={sig['power']:.1f}")

                # limit guard'ları güncelle
                dynamic_autotrade_state()

                # 65–74 power aralığında gerçek trade aç
                execute_real_trade(sig)

            # 3️⃣ simülasyon kuyrukları
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4️⃣ 4 saatlik backup
            auto_report_if_due()

            # 5️⃣ heartbeat (10 dk)
            heartbeat_and_status_check()

            # 6️⃣ state kaydet
            safe_save(STATE_FILE, STATE)

            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ==============================================================
# 🟢 ENTRYPOINT
# ==============================================================

if __name__ == "__main__":
    main()
