import os, json, time, requests, hmac, hashlib, threading, math, random
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================
# 📘 EMA ULTRA v15.9.20 FULL + TP-AutoScale + No SL + Retry(5)
#   + NO-TP Fix + Only-REAL Telegram
# ==============================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE       = os.path.join(DATA_DIR,"state.json")
PARAM_FILE       = os.path.join(DATA_DIR,"params.json")
AI_SIGNALS_FILE  = os.path.join(DATA_DIR,"ai_signals.json")
AI_ANALYSIS_FILE = os.path.join(DATA_DIR,"ai_analysis.json")
AI_RL_FILE       = os.path.join(DATA_DIR,"ai_rl_log.json")
SIM_POS_FILE     = os.path.join(DATA_DIR,"sim_positions.json")
SIM_CLOSED_FILE  = os.path.join(DATA_DIR,"sim_closed.json")
LOG_FILE         = os.path.join(DATA_DIR,"log.txt")

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
    except: pass
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
    r=requests.post(url,headers=headers,timeout=10) if m=="POST" else requests.get(url,headers=headers,timeout=10)
    if r.status_code!=200: raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def get_symbol_filters(sym):
    if sym in PRECISION_CACHE: return PRECISION_CACHE[sym]
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
        log(f"[PREC WARN]{sym}{e}")
        PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001,"minPrice":0.0001}
    return PRECISION_CACHE[sym]

def adjust_precision(sym,v,kind="qty"):
    f=get_symbol_filters(sym)
    step=f["stepSize"] if kind=="qty" else f["tickSize"]
    if step<=0: return v
    return round(round(v/step)*step,12)

def calc_order_qty(sym,entry,usd):
    return adjust_precision(sym,usd/max(entry,1e-12),"qty")

def futures_get_price(sym):
    try:
        return float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                                  params={"symbol":sym},timeout=5).json()["price"])
    except: return None

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
            params={"symbol":sym,"interval":it,"limit":lim},timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms(): r=r[:-1]
        return r
    except: return []

# === Indicators ===
def ema(vals,n):
    k=2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k+e[-1]*(1-k))
    return e

def rsi(vals,period=14):
    if len(vals)<period+2: return [50]*len(vals)
    d=np.diff(vals); g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:period]); al=np.mean(l[:period]); out=[50]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0; out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]) if i else h[i]-l[i],
                      abs(l[i]-c[i-1]) if i else h[i]-l[i]))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)):
        a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_val):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55+diff*20+((rsi_val-50)/50)*15+(atr_v/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    if 65<=p<75: return "REAL","🟩"
    if p>=75: return "ULTRA","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""
# ==============================================================
# PARAM / STATE INIT
# ==============================================================

PARAM_DEFAULT = {
    "SCALP_TP_PCT":0.006,   # ilk TP denemesi için yüzdesel referans
    "SCALP_SL_PCT":0.20,    # canlıda kullanılmıyor ama uyumluluk için duruyor
    "TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30,
    "MAX_SELL":30,
    "ANGLE_MIN":0.00005     # sinyal filtresi (gevşetildi ki sinyal gelsin)
}
PARAM = safe_load(PARAM_FILE, PARAM_DEFAULT)
if not isinstance(PARAM, dict):
    PARAM = PARAM_DEFAULT

STATE_DEFAULT = {
    "bar_index":0,
    "last_report":0,
    "auto_trade_active":True,
    "last_api_check":0,
    "long_blocked":False,
    "short_blocked":False
}
STATE = safe_load(STATE_FILE, STATE_DEFAULT)
if "auto_trade_active" not in STATE:   STATE["auto_trade_active"]=True
if "long_blocked" not in STATE:        STATE["long_blocked"]=False
if "short_blocked" not in STATE:       STATE["short_blocked"]=False

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE,[])
AI_RL         = safe_load(AI_RL_FILE,[])
SIM_POSITIONS = safe_load(SIM_POS_FILE,[])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE,[])

# ==============================================================
# SIGNAL BUILDER
# ==============================================================

def build_scalp_signal(sym, kl, bar_i):
    """
    1h confirmed bar reversal scalp sinyali:
    - EMA7 slope reversal (down->up => UP, up->down => DOWN)
    - slope_impulse < ANGLE_MIN ise sinyal yok (çok zayıf dönüş)
    - abs(24h change) >= 10% ise sinyal yok (aşırı volatil coin filtre)
    - power hesapla ve tier belirle ("REAL"/"ULTRA"/"NORMAL")
    """
    if len(kl) < 60:
        return None

    # 24h değişim
    try:
        r = requests.get(
            BINANCE_FAPI+"/fapi/v1/ticker/24hr",
            params={"symbol":sym}, timeout=5
        ).json()
        chg = float(r["priceChangePercent"])
    except:
        chg = 0.0

    if abs(chg) >= 10.0:
        return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    e7 = ema(closes,7)

    # slope reversal momentumu
    s_now  = e7[-2] - e7[-5]
    s_prev = e7[-3] - e7[-6]
    slope_impulse = abs(s_now - s_prev)
    if slope_impulse < PARAM["ANGLE_MIN"]:
        return None

    if   s_prev < 0 and s_now > 0: direction="UP"
    elif s_prev > 0 and s_now < 0: direction="DOWN"
    else: return None

    atr_v = atr_like(highs,lows,closes)[-1]
    r_val = rsi(closes)[-1]
    pwr   = calc_power(e7[-1], e7[-2], e7[-5], atr_v, closes[-1], r_val)

    tier,emoji = tier_from_power(pwr)
    if not tier:
        return None

    entry = closes[-1]

    # sadece bilgi amaçlı TP/SL tahmini (artık gerçek TP hesaplaması bu değil)
    if direction == "UP":
        tp_guess = entry*(1+PARAM["SCALP_TP_PCT"])
        sl_guess = entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess = entry*(1-PARAM["SCALP_TP_PCT"])
        sl_guess = entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,
        "dir":direction,
        "tier":tier,      # "REAL" / "ULTRA" / "NORMAL"
        "emoji":emoji,
        "entry":entry,
        "tp":tp_guess,
        "sl":sl_guess,
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
# RL ENRICH (sim kapandığında sinyale bağla)
# ==============================================================

def enrich_with_ai_context(pos):
    """
    CLOSED sim trade'i RL açısından zenginleştir.
    AI_SIGNALS içindeki uygun sinyalden:
      rsi, atr, chg24h, born_bar, tier, power
    bilgilerini merge et.
    Eşleştirme: aynı symbol ve entry %0.2 tolerans içinde yakınsa.
    """
    best=None
    for s in reversed(AI_SIGNALS):
        if s.get("symbol")!=pos.get("symbol"):
            continue
        e_sig=s.get("entry"); e_pos=pos.get("entry")
        if not e_sig or not e_pos:
            continue
        if abs(e_sig-e_pos)/max(e_sig,1e-12) < 0.002:
            best=s
            break
    if best:
        pos["rsi"]=best.get("rsi")
        pos["atr"]=best.get("atr")
        pos["chg24h"]=best.get("chg24h")
        pos["born_bar"]=best.get("born_bar")
        pos["tier"]=best.get("tier")
        pos["power"]=best.get("power")
    return pos

# ==============================================================
# SIM ENGINE (4 approve varyantı, approve_delay_min loglu)
# ==============================================================

def queue_sim_variants(sig):
    """
    Her sinyal için dört onay varyantı oluştur:
      30dk / 1h / 1.5h / 2h
    approve_delay_min ve approve_label ile kaydediyoruz.
    """
    delays = [
        (30*60,  "approve_30m",   30),
        (60*60,  "approve_1h",    60),
        (90*60,  "approve_1h30",  90),
        (120*60, "approve_2h",    120)
    ]
    now_s = now_ts_s()
    for secs,label,mins in delays:
        SIM_QUEUE.append({
            "symbol":sig["symbol"],
            "dir":sig["dir"],
            "tier":sig["tier"],
            "entry":sig["entry"],
            "tp":sig["tp"],
            "sl":sig["sl"],
            "power":sig["power"],
            "created_ts":now_s,
            "open_after_ts":now_s+secs,
            "approve_delay_min":mins,
            "approve_label":label,
            "status":"PENDING"
        })
    safe_save(SIM_POS_FILE,SIM_QUEUE)

def process_sim_queue_and_open_due():
    """
    open_after_ts dolmuş PENDING pozisyonları OPEN'e çevirir.
    Log: [SIM OPEN] ... approve=xxm
    """
    global SIM_POSITIONS
    now_s=now_ts_s()
    remain=[]
    opened_any=False

    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            newpos={**q,
                "status":"OPEN",
                "open_ts":now_s,
                "open_time":now_local_iso()
            }
            SIM_POSITIONS.append(newpos)
            opened_any=True
            log(f"[SIM OPEN] {q['symbol']} {q['dir']} approve={q['approve_delay_min']}m entry={q['entry']}")
        else:
            remain.append(q)

    SIM_QUEUE[:] = remain
    if opened_any:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
    # SIM_QUEUE aynı dosyada tutuluyor, reuse ediyoruz

def process_sim_closes():
    """
    Her OPEN sim pozisyonu için TP/SL vurdu mu kontrol et.
    (SL canlı trade'de yok ama sim verisinde tutuyoruz ki RL analizi yapılabilsin.)
    TP/SL tetiklenirse CLOSED olarak SIM_CLOSED'e pushluyoruz.
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
            gain_pct = (
                (last_price/pos["entry"]-1.0)*100.0
                if pos["dir"]=="UP"
                else (pos["entry"]/last_price-1.0)*100.0
            )

            enriched = enrich_with_ai_context(dict(pos))

            SIM_CLOSED.append({
                **enriched,
                "status":"CLOSED",
                "close_time":close_time,
                "exit_price":last_price,
                "exit_reason":hit,
                "gain_pct":gain_pct
            })
            changed=True
            log(f"[SIM CLOSE] {pos['symbol']} {pos['dir']} {hit} {gain_pct:.3f}% approve={pos.get('approve_delay_min')}m")
        else:
            still.append(pos)

    SIM_POSITIONS=still
    if changed:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE,SIM_CLOSED)

# ==============================================================
# TP ONLY PLACEMENT (5 retry + trade-size-scaled fallback + NO-TP fix)
# ==============================================================

def calc_fallback_tp_pct():
    """
    TP yüzdesi sabit (~0.64%-0.8%) ama dolar karşılığı,
    TRADE_SIZE_USDT ile orantılı büyür.
    250$ trade için 1.6-2.0 USD hedeflenir.
    """
    base_ref = 250.0
    trade_usd = PARAM.get("TRADE_SIZE_USDT", base_ref)

    # sabit yüzde bandı (≈0.64%-0.8%)
    tp_pct = random.uniform(1.6, 2.0) / base_ref

    # log bilgi amaçlı
    tp_dollar = tp_pct * trade_usd
    log(f"[TP-FALLBACK] trade={trade_usd}$ -> TP≈${tp_dollar:.2f} ({tp_pct*100:.3f}%)")
    return tp_pct

def futures_set_tp_sl(sym, dir, qty, entry, tp_pct_in, _sl_pct_unused=None):
    """
    Artık sadece TAKE_PROFIT_MARKET emir kuruluyor. SL yok.
    Akış:
      1) tp_pct_in (örn SCALP_TP_PCT) + küçük buffer ile TP dener -> 5 retry
      2) başarısızsa fallback (calc_fallback_tp_pct) ile tekrar dener
      3) hala olmazsa:
         - Telegram'a ⚠️ TP kurulamadı uyarısı
         - log'a [NO TP]
         - fonksiyondan return
      4) başarı varsa log "✅ TP SET ..."
    """
    try:
        buf = 0.002
        adj_tp = tp_pct_in + buf

        f = get_symbol_filters(sym)
        tick = f["tickSize"]
        pos_side = "LONG" if dir=="UP" else "SHORT"
        side     = "SELL" if dir=="UP" else "BUY"

        decimals = len(str(tick).split(".")[1].rstrip("0")) if "." in str(tick) else 0
        fmt      = f"{{:.{decimals}f}}"

        def try_place(tp_pct, label):
            """
            TAKE_PROFIT_MARKET oluştur, 5 denemeye kadar. True/False döner.
            """
            success=False
            for attempt in range(5):
                try:
                    tp_price = (
                        entry*(1+tp_pct) if dir=="UP"
                        else entry*(1-tp_pct)
                    )
                    payload={
                        "symbol":sym,
                        "side":side,
                        "type":"TAKE_PROFIT_MARKET",
                        "stopPrice":fmt.format(tp_price),
                        "quantity":f"{qty}",
                        "workingType":"MARK_PRICE",
                        "closePosition":"true",
                        "positionSide":pos_side,
                        "timestamp":now_ts_ms()
                    }
                    _signed_request("POST","/fapi/v1/order",payload)
                    success=True
                    break
                except Exception as e:
                    if "Stop price less than zero" in str(e):
                        time.sleep(0.2)
                        continue
                    time.sleep(0.2)
            if not success:
                log(f"[TP WARN] {sym} {label} could not place TP after retries")
            return success

        ok_primary = try_place(adj_tp,"primary")
        if not ok_primary:
            tp_fb = calc_fallback_tp_pct()
            ok_fb = try_place(tp_fb + buf,"fallback")
            if not ok_fb:
                # hala başaramadıysak telegram uyarısı at
                tg_send(f"⚠️ TP kurulamadı {sym} {dir} qty:{qty}")
                log(f"[NO TP] {sym} TP kurulamadı.")
                return

        log(f"✅ TP SET {sym} {dir} qty={qty}")
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")

# ==============================================================
# POSITION / LIMIT GUARDS / HEARTBEAT / REPORT
# ==============================================================

def update_directional_limits():
    """
    long_blocked / short_blocked günceller.
    MAX_BUY ve MAX_SELL sınırlarına göre.
    Ayrıca auto_trade_active flagini set eder.
    """
    live={"long":{}, "short":{}, "long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            amt=float(p["positionAmt"])
            sym=p["symbol"]
            if amt>0:
                live["long"][sym]=amt
            elif amt<0:
                live["short"][sym]=abs(amt)
        live["long_count"]=len(live["long"])
        live["short_count"]=len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")

    STATE["long_blocked"]  = (live["long_count"]  >= PARAM["MAX_BUY"])
    STATE["short_blocked"] = (live["short_count"] >= PARAM["MAX_SELL"])

    both_blocked = STATE["long_blocked"] and STATE["short_blocked"]
    STATE["auto_trade_active"] = (not both_blocked)

    safe_save(STATE_FILE,STATE)
    return live

def heartbeat_and_status_check(live_positions_snapshot):
    """
    10 dakikada bir Binance API health + durum snapshot.
    Telegram'a göndeririz çünkü operasyonel health bilgisidir.
    """
    now=time.time()
    if now-STATE.get("last_api_check",0) < 600:
        return
    STATE["last_api_check"]=now
    safe_save(STATE_FILE,STATE)

    try:
        st=requests.get(BINANCE_FAPI+"/fapi/v1/time",timeout=5).json()["serverTime"]
        drift=abs(now_ts_ms()-st)
        ping_ok=requests.get(BINANCE_FAPI+"/fapi/v1/ping",timeout=5).status_code==200
        try:
            _signed_request("GET","/fapi/v2/account",{"timestamp":now_ts_ms()})
            key_ok=True
        except:
            key_ok=False

        hb_msg = (
            f"✅ HEARTBEAT drift={int(drift)}ms ping={ping_ok} key={key_ok}"
            if ping_ok and key_ok and drift<1500
            else f"⚠️ HEARTBEAT ping={ping_ok} key={key_ok} drift={int(drift)}"
        )
        tg_send(hb_msg)
        log(hb_msg)
    except Exception as e:
        hb_err=f"❌ HEARTBEAT {e}"
        tg_send(hb_err)
        log(f"[HBERR]{e}")

    msg=(f"📊 STATUS bar:{STATE.get('bar_index',0)} "
         f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
         f"long_blocked:{STATE.get('long_blocked')} "
         f"short_blocked:{STATE.get('short_blocked')} "
         f"long:{live_positions_snapshot.get('long_count',0)} "
         f"short:{live_positions_snapshot.get('short_count',0)} "
         f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
         f"sim_closed:{len(SIM_CLOSED)}")
    tg_send(msg)
    log(msg)

def ai_log_signal(sig):
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
        "entry":sig["entry"],
        "born_bar":sig.get("born_bar")
    })
    safe_save(AI_SIGNALS_FILE,AI_SIGNALS)

def ai_update_analysis_snapshot():
    ultra_count=sum(1 for x in AI_SIGNALS if x.get("tier")=="ULTRA")
    real_count =sum(1 for x in AI_SIGNALS if x.get("tier")=="REAL")
    norm_count =sum(1 for x in AI_SIGNALS if x.get("tier")=="NORMAL")
    snapshot={
        "time":now_local_iso(),
        "ultra_signals_total": ultra_count,
        "real_signals_total":  real_count,
        "normal_signals_total":norm_count,
        "sim_open_count":len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
        "sim_closed_count":len(SIM_CLOSED)
    }
    AI_ANALYSIS.append(snapshot)
    safe_save(AI_ANALYSIS_FILE,AI_ANALYSIS)

def auto_report_if_due():
    now_now=time.time()
    if now_now-STATE.get("last_report",0) < 14400:
        return
    ai_update_analysis_snapshot()

    files_to_push=[
        AI_SIGNALS_FILE,
        AI_ANALYSIS_FILE,
        AI_RL_FILE,
        SIM_POS_FILE,
        SIM_CLOSED_FILE
    ]

    for fpath in files_to_push:
        try:
            if os.path.exists(fpath):
                sz=os.path.getsize(fpath)
                if sz>10*1024*1024:
                    with open(fpath,"r",encoding="utf-8") as f:
                        raw=f.read()
                    tail=raw[-int(len(raw)*0.2):]
                    with open(fpath,"w",encoding="utf-8") as f:
                        f.write(tail)
        except: pass
        tg_send_file(fpath,f"📊 AutoBackup {os.path.basename(fpath)}")

    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"]=now_now
    safe_save(STATE_FILE,STATE)
# ==============================================================
# REAL TRADE EXECUTION HELPERS
# ==============================================================

def open_market_position(sym, direction, qty):
    """
    Market order açar (hedge mode).
    Döner:
      symbol, dir, qty, entry (fill avgPrice/price/ticker fallback), pos_side
    """
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,
        "side":side,
        "type":"MARKET",
        "quantity":f"{qty}",
        "positionSide":pos_side,
        "timestamp":now_ts_ms()
    })
    fill_price = (
        res.get("avgPrice")
        or res.get("price")
        or futures_get_price(sym)
    )
    entry_final = float(fill_price)
    return {
        "symbol":sym,
        "dir":direction,
        "qty":qty,
        "entry":entry_final,
        "pos_side":pos_side
    }

# ==============================================================
# EXECUTE REAL TRADE
#  - Only power 65-74
#  - Directional guard
#  - TrendLock
#  - Duplicate guard
#  - TP only
#  - Telegram ONLY if actual order filled
# ==============================================================

def execute_real_trade(sig):
    """
    Gerçek trade açma koşulları:
      - power 65 <= pwr < 75
      - auto_trade_active True
      - yön bloklu değil (long_blocked / short_blocked)
      - TREND_LOCK aynı yönde değil
      - duplicate pozisyon yok
    Eğer açılırsa:
      - market order
      - TP kur (SL yok)
      - TP kurulamadıysa futures_set_tp_sl() zaten ⚠️ telegram ve [NO TP] log atar
      - başarılıysa Telegram'a ✅ REAL mesajı atılır
      - RL loglanır
    """
    pwr = sig["power"]
    if not (65 <= pwr < 75):
        # bu sinyal gerçek trade'e aday değil
        return

    if not STATE.get("auto_trade_active", True):
        log("[SKIP REAL] auto_trade_active False")
        return

    sym = sig["symbol"]
    direction = sig["dir"]

    # Directional guard
    if direction == "UP" and STATE.get("long_blocked", False):
        log(f"[GUARD] BUY blocked for {sym}, limit dolu. Trade açılmadı.")
        return
    if direction == "DOWN" and STATE.get("short_blocked", False):
        log(f"[GUARD] SELL blocked for {sym}, limit dolu. Trade açılmadı.")
        return

    # TrendLock -> aynı yönden spam engelle
    if TREND_LOCK.get(sym) == direction:
        log(f"[SKIP REAL] TrendLock {sym} {direction}")
        return

    # Duplicate guard: zaten aynı yönde açık poz var mı?
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[POSRISK ERR]{e}")
        acc=[]

    long_syms=[p["symbol"] for p in acc if float(p["positionAmt"])>0]
    short_syms=[p["symbol"] for p in acc if float(p["positionAmt"])<0]

    if direction=="UP" and sym in long_syms:
        log(f"[SKIP REAL] already long {sym}")
        return
    if direction=="DOWN" and sym in short_syms:
        log(f"[SKIP REAL] already short {sym}")
        return

    # Miktar hesapla
    qty = calc_order_qty(sym, sig["entry"], PARAM["TRADE_SIZE_USDT"])
    if not qty or qty <= 0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı.")
        return

    # Market order dene
    try:
        opened = open_market_position(sym, direction, qty)
        entry_exec = opened.get("entry") or futures_get_price(sym)
        if not entry_exec or entry_exec <= 0:
            log(f"[OPEN FAIL] {sym} entry alınamadı.")
            return

        # TP kur (SL yok). futures_set_tp_sl kendi içinde:
        #  - retry
        #  - fallback
        #  - NO TP uyarısı (⚠️) Telegram'a atıyor gerekirse
        futures_set_tp_sl(
            sym,
            direction,
            qty,
            entry_exec,
            PARAM["SCALP_TP_PCT"]
        )

        # TrendLock set et ki aynı yönde tekrar patlamasın
        TREND_LOCK[sym] = direction

        # Telegram -> SADECE gerçekten emir açıldıysa
        tg_send(
            f"✅ REAL {sym} {direction} qty:{qty}\n"
            f"Power:{pwr:.2f}\n"
            f"Entry:{entry_exec:.12f}"
        )

        log(f"[REAL] {sym} {direction} qty={qty} entry={entry_exec:.6f} pwr={pwr:.2f}")

        # RL kaydı
        AI_RL.append({
            "time":now_local_iso(),
            "symbol":sym,
            "dir":direction,
            "entry":entry_exec,
            "tp_ref_pct":PARAM["SCALP_TP_PCT"],
            "power":sig["power"],
            "born_bar":sig["born_bar"]
        })
        safe_save(AI_RL_FILE,AI_RL)

    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")
        # ÖNEMLİ: burada Telegram ATMİYORUZ, çünkü trade aslında açılmadı

# ==============================================================
# MAIN LOOP
# ==============================================================

def main():
    tg_send("🚀 EMA ULTRA v15.9.20 FULL aktif (Only-REAL Telegram + TP fallback)")
    log("[START] EMA ULTRA v15.9.20 FULL")

    # Binance'ten tradable USDT çiftlerini çek
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[
            s["symbol"]
            for s in info["symbols"]
            if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"
        ]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}")
        symbols=[]
    symbols.sort()

    while True:
        try:
            # bar sayacı (state persist)
            STATE["bar_index"] = STATE.get("bar_index",0) + 1
            bar_i = STATE["bar_index"]

            # 1) sinyal tara
            sigs = run_parallel(symbols, bar_i)

            # 2) sinyalleri işle
            for sig in sigs:
                # 2a) AI sinyal loguna yaz (RL için ham veri)
                ai_log_signal(sig)

                # 2b) sim varyant kuyruğuna koy (30m / 1h / 1.5h / 2h)
                queue_sim_variants(sig)

                # 2c) sinyali sadece LOG'a yaz (Telegram YOK artık burada)
                log(
                    f"[SIG] {sig['emoji']} {sig['tier']} {sig['symbol']} {sig['dir']} "
                    f"Pow:{sig['power']:.1f} RSI:{sig.get('rsi',0):.1f} "
                    f"ATR:{sig.get('atr',0):.4f} Δ24h:{sig['chg24h']:.2f}% "
                    f"Entry:{sig['entry']:.6f} born_bar:{sig['born_bar']}"
                )

                # 2d) directional limitleri güncelle (MAX_BUY / MAX_SELL mantığı)
                live_positions_snapshot = update_directional_limits()

                # 2e) gerçek trade dene (yalnızca power 65-74)
                execute_real_trade(sig)

            # 3) sim queue -> zamanı gelenleri OPEN yap
            process_sim_queue_and_open_due()

            # 4) açık sim pozisyonları TP/SL vurduysa CLOSED yap
            process_sim_closes()

            # 5) otomatik rapor / backup (4 saatlik json dump)
            auto_report_if_due()

            # 6) heartbeat + status (10 dk'da bir)
            live_positions_snapshot = update_directional_limits()
            heartbeat_and_status_check(live_positions_snapshot)

            # 7) state kaydet
            safe_save(STATE_FILE,STATE)

            # 8) bekle
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ==============================================================
# ENTRYPOINT
# ==============================================================

if __name__=="__main__":
    main()
