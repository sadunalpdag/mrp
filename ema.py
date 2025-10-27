import os, json, time, requests, hmac, hashlib, threading, math, random
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==============================================================
# 📘 EMA ULTRA v15.9.18 FULL
# HB + ApproveSimLog + PowerGuard + RetryTP + USD-Fallback
# + DirectionalFix + SilentSignals
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

# ==============================================================
# 🧩 UTILITIES
# ==============================================================

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

# ==============================================================
# 🔌 BINANCE HELPERS
# ==============================================================

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r=requests.post(url,headers=headers,timeout=10) if m=="POST" else requests.get(url,headers=headers,timeout=10)
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
    return round((round(v/step)*step),12) if step>0 else v

def calc_order_qty(sym,entry,usd):
    raw=usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except:
        return None

# ==============================================================
# 💰 USD BASED TP/SL HELPER
# ==============================================================

def calc_usd_based_tp_sl(entry,dir):
    base=250.0
    trade_size=PARAM.get("TRADE_SIZE_USDT",base)
    tp_min=1.8/trade_size; tp_max=2.0/trade_size
    sl_min=40.0/trade_size; sl_max=50.0/trade_size
    tp_pct=random.uniform(tp_min,tp_max)
    sl_pct=random.uniform(sl_min,sl_max)
    log(f"[USD-BASED] {dir} tp_pct={tp_pct:.5f} sl_pct={sl_pct:.5f}")
    return tp_pct,sl_pct

# ==============================================================
# 💣 FUTURES TP/SL SET WITH RETRY + USD FALLBACK
# ==============================================================

def futures_set_tp_sl(sym,dir,qty,entry,tp_pct_in,sl_pct_in):
    try:
        buf=0.002
        adj_tp=tp_pct_in+buf; adj_sl=sl_pct_in+buf
        f=get_symbol_filters(sym)
        tick=f["tickSize"]; pos="LONG" if dir=="UP" else "SHORT"
        side="SELL" if dir=="UP" else "BUY"
        dec=len(str(tick).split(".")[1].rstrip("0")) if "." in str(tick) else 0
        fmt=f"{{:.{dec}f}}"

        def place(tp_pct,sl_pct,label):
            ok=True
            for ttype,pct in [("TAKE_PROFIT_MARKET",tp_pct),("STOP_MARKET",sl_pct)]:
                success=False
                for _ in range(5):
                    try:
                        price=entry*(1+pct) if ttype=="TAKE_PROFIT_MARKET" else entry*(1-pct) if dir=="UP" else entry*(1-pct) if ttype=="TAKE_PROFIT_MARKET" else entry*(1+pct)
                        _signed_request("POST","/fapi/v1/order",{
                            "symbol":sym,"side":side,"type":ttype,
                            "stopPrice":fmt.format(price),"quantity":f"{qty}",
                            "workingType":"MARK_PRICE","closePosition":"true",
                            "positionSide":pos,"timestamp":now_ts_ms()
                        })
                        success=True; break
                    except Exception as e:
                        if "Stop price" in str(e): time.sleep(0.2); continue
                if not success: ok=False; log(f"[TP/SL WARN]{sym}{ttype} {label} failed")
            return ok

        if not place(adj_tp,adj_sl,"primary"):
            tp2,sl2=calc_usd_based_tp_sl(entry,dir)
            place(tp2+buf,sl2+buf,"usd_fallback")

        log(f"✅ TP/SL SET {sym} {dir} qty={qty}")
    except Exception as e:
        log(f"[TP/SL ERR]{sym}{e}")
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
    "long_blocked":False,
    "short_blocked":False
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
    # ULTRA sinyaller gerçek trade açmıyor ama tier ayrımı RL için hâlâ önemli
    if p>=75: return "ULTRA","🟩"
    if p>=68: return "PREMIUM","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""

# ==============================================================
# 🔍 SIGNAL BUILDER
# ==============================================================

def build_scalp_signal(sym,kl,bar_i):
    """
    1h confirmed bar reversal scalp sinyali:
    - EMA7 slope reversal (down->up => UP, up->down => DOWN)
    - slope_impulse ANGLE_MIN altındaysa sinyal yok
    - 24h |chg| >=10% ise sinyal yok
    - power -> tier seçimi
    """
    if len(kl)<60:
        return None

    # 24h hareket çok uçtuysa alma
    try:
        chg=futures_24h_change(sym)
    except:
        chg=0.0
    if abs(chg)>=10.0:
        return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    e7=ema(closes,7)

    # slope değişimi
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
# 🤖 RL ENRICH HELPER
# ==============================================================

def enrich_with_ai_context(pos):
    """
    Kapandıktan sonra RL analizi için:
    - aynı sembol ve benzer entry üzerinden AI_SIGNALS bilgilerini (rsi, atr, chg24h, born_bar) inject eder
    """
    best=None
    for s in reversed(AI_SIGNALS):
        if s.get("symbol")!=pos.get("symbol"):
            continue
        e_sig=s.get("entry")
        e_pos=pos.get("entry")
        if not e_sig or not e_pos:
            continue
        # entry %0.2 yaklaştırma toleransı
        if abs(e_sig - e_pos)/max(e_sig,1e-12) < 0.002:
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
# 🎮 SIM ENGINE (approve_delay_min LOGGING EKLENDİ)
# ==============================================================

def queue_sim_variants(sig):
    """
    Tüm sinyaller için 4 farklı approve gecikmesi oluştur.
    Artık her varyanta:
      - approve_delay_min
      - approve_label
    ekliyoruz ki RL tarafında hangi onay bucket'ı kazandı bilelim.
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
    open_after_ts geldiğinde PENDING -> OPEN'e dönüşür.
    approve_delay_min ve approve_label korunur.
    """
    global SIM_POSITIONS
    now_s=now_ts_s()
    remain=[]
    opened_any=False

    for q in SIM_QUEUE:
        if q["open_after_ts"]<=now_s:
            newpos={
                **q,
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
    safe_save(SIM_POS_FILE,SIM_QUEUE)

def process_sim_closes():
    """
    Açık sim pozisyonlarında TP veya SL vurdu mu?
    Vurduysa kapat -> SIM_CLOSED'e taşı.
    enrich_with_ai_context() ile RL bilgilerini inject et.
    approve_delay_min ve approve_label de taşınır.
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
# 🚦 DIRECTIONAL LIMIT / AUTO TRADE STATE
# ==============================================================

def update_directional_limits():
    """
    long_blocked / short_blocked durumlarını günceller.
    burada: long_count >= MAX_BUY -> long_blocked=True
            short_count >= MAX_SELL -> short_blocked=True
    ayrıca eğer iki taraf birden block ise auto_trade_active False yapılabilir
    (şu an tam kapatma istemiyoruz; sadece block flag kullanıyoruz).
    """
    live = {
        "long":{},
        "short":{},
        "long_count":0,
        "short_count":0
    }
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            sym=p["symbol"]
            amt=float(p["positionAmt"])
            if amt>0:
                live["long"][sym]=amt
            elif amt<0:
                live["short"][sym]=abs(amt)
        live["long_count"]=len(live["long"])
        live["short_count"]=len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")

    # yön bazlı bloklama
    STATE["long_blocked"]  = (live["long_count"]  >= PARAM["MAX_BUY"])
    STATE["short_blocked"] = (live["short_count"] >= PARAM["MAX_SELL"])

    # global aktif/pasif mantığı:
    both_blocked = STATE["long_blocked"] and STATE["short_blocked"]
    if both_blocked:
        STATE["auto_trade_active"]=False
    else:
        STATE["auto_trade_active"]=True

    safe_save(STATE_FILE,STATE)
    return live  # canlı pozisyon özetini de geri döndürüyoruz, main loop status için kullanacağız

# ==============================================================
# ❤️ HEARTBEAT / STATUS SNAPSHOT
# ==============================================================

def heartbeat_and_status_check(live_positions_snapshot):
    """
    Her ~10 dakikada bir Binance API sağlık kontrolü + durum özeti.
    Telegram'a gönderiyoruz çünkü bu operasyonel sağlık bilgisi.
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

# ==============================================================
# ⏰ AUTO BACKUP (4 saat)
# ==============================================================

def ai_log_signal(sig):
    """
    Sinyali AI_SIGNALS listesine pushla (RL için ham veri).
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
        "entry":sig["entry"],
        "born_bar":sig.get("born_bar")
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

def auto_report_if_due():
    now_now=time.time()
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
                sz=os.path.getsize(fpath)
                if sz > 10*1024*1024:
                    with open(fpath,"r",encoding="utf-8") as f:
                        raw=f.read()
                    tail=raw[-int(len(raw)*0.2):]
                    with open(fpath,"w",encoding="utf-8") as f:
                        f.write(tail)
        except:
            pass

        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")

    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"] = now_now
    safe_save(STATE_FILE,STATE)
# ==============================================================
# 🚀 REAL TRADE EXECUTION (Directional Fix + Power Guard)
# ==============================================================

def execute_real_trade(sig):
    """
    Sadece power 65–74 arası sinyallerde gerçek trade açar.
    Directional guard aktif: bloklu yönde işlem açmaz ve Telegram mesajı da atmaz.
    """
    pwr = sig["power"]
    if not (65 <= pwr < 75):
        return
    if not STATE.get("auto_trade_active", True):
        return

    sym = sig["symbol"]
    direction = sig["dir"]

    # ✅ Directional guard kesin kontrol
    if direction == "UP" and STATE.get("long_blocked", False):
        log(f"[GUARD] BUY blocked for {sym}, limit dolu.")
        return
    if direction == "DOWN" and STATE.get("short_blocked", False):
        log(f"[GUARD] SELL blocked for {sym}, limit dolu.")
        return

    if TREND_LOCK.get(sym) == direction:
        return

    # Duplicate guard
    live = _signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    long_syms = [p["symbol"] for p in live if float(p["positionAmt"]) > 0]
    short_syms = [p["symbol"] for p in live if float(p["positionAmt"]) < 0]
    if direction=="UP" and sym in long_syms:
        return
    if direction=="DOWN" and sym in short_syms:
        return

    qty = calc_order_qty(sym, sig["entry"], PARAM["TRADE_SIZE_USDT"])
    if not qty or qty <= 0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı.")
        return

    try:
        opened = open_market_position(sym, direction, qty)
        entry_exec = opened.get("entry") or futures_get_price(sym)
        if not entry_exec or entry_exec <= 0:
            log(f"[OPEN FAIL] {sym} entry alınamadı.")
            return

        # TP/SL kur
        futures_set_tp_sl(
            sym, direction, qty, entry_exec,
            PARAM["SCALP_TP_PCT"], PARAM["SCALP_SL_PCT"]
        )

        TREND_LOCK[sym] = direction

        # ✅ Telegram sadece gerçekten emir açıldıysa
        tg_send(
            f"✅ REAL {sym} {direction} qty:{qty}\n"
            f"Power:{pwr:.2f}\n"
            f"Entry:{entry_exec:.12f}"
        )
        log(f"[REAL] {sym} {direction} qty={qty} entry={entry_exec:.6f} pwr={pwr:.2f}")

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
        log(f"[OPEN ERR]{sym}{e}")

# ==============================================================
# 🧠 MAIN LOOP
# ==============================================================

def main():
    tg_send("🚀 EMA ULTRA v15.9.18 FULL başlatıldı")
    log("[START] EMA ULTRA v15.9.18 FULL")

    try:
        info = requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols = [
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
            STATE["bar_index"] += 1
            bar_i = STATE["bar_index"]

            # 1️⃣ Sinyalleri tara
            sigs = run_parallel(symbols, bar_i)

            # 2️⃣ Sinyalleri işle
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)

                # sadece ULTRA hariç (çünkü 65–74 power aralığı real)
                if sig["tier"] == "ULTRA":
                    continue

                sym=sig["symbol"]; direction=sig["dir"]
                tg_send(
                    f"{sig['emoji']} {sig['tier']} {sym} {direction}\n"
                    f"Pow:{sig['power']:.1f} RSI:{sig.get('rsi',0):.1f} "
                    f"ATR:{sig.get('atr',0):.4f} Δ24h:{sig['chg24h']:.2f}%\n"
                    f"Entry:{sig['entry']:.12f}\nTP:{sig['tp']:.12f}\nSL:{sig['sl']:.12f}\n"
                    f"born_bar:{sig['born_bar']}"
                )
                log(f"[SIG] {sym} {direction} tier={sig['tier']} Pow:{sig['power']:.1f}")

                # 3️⃣ Directional limits update
                live_positions_snapshot = update_directional_limits()

                # 4️⃣ Gerçek trade (65–74 power) — directional fix
                execute_real_trade(sig)

            # 5️⃣ Sim queue açık pozisyonlara dönüştür
            process_sim_queue_and_open_due()

            # 6️⃣ Sim pozisyonlarda TP/SL kapanışlarını kontrol et
            process_sim_closes()

            # 7️⃣ Backup gerekiyorsa gönder
            auto_report_if_due()

            # 8️⃣ Heartbeat & status
            live_positions_snapshot = update_directional_limits()
            heartbeat_and_status_check(live_positions_snapshot)

            # 9️⃣ State kaydet
            safe_save(STATE_FILE,STATE)

            # 10️⃣ 30 saniye bekle
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ==============================================================
# 🏁 ENTRYPOINT
# ==============================================================

if __name__=="__main__":
    main()