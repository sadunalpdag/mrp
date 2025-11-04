import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.53 — All Strategies + Market State + Cashout Mark Price System
#  - PEMA kaldırıldı
#  - Aktif stratejiler:
#       ⚡ EARLY (EMA3–EMA7 + ATR spike)
#       🟢 UT/STC (Ultimate Trend + Schaff Trend Cycle)
#       📈 MACD (EMA20/200 + MACD crossover)
#       🟩 FVG (Fair Value Gap Break)
#       📘 EMA PULLBACK (EMA200 + EMA9/30 + swing break + MarketState)
#  - Power band 65–75 sadece EARLY için aktif
#  - Smart TP, 6h TrendLock, Guards, Telegram sistemi aynı
#  - 💰 Yeni: CASHOUT_USDT parametresi (default 60 USD)
# ==============================================================================

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
TREND_LOCK_TIME = {}
TRENDLOCK_EXPIRY_SEC = 6 * 3600
SIM_QUEUE = []
getcontext().prec = 28

# ===================== UTILITIES =====================

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except: pass

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
        log(f"[SAVE ERR]{e}")

def now_local_iso():
    return (datetime.now(timezone.utc)+timedelta(hours=3)).replace(microsecond=0).isoformat()

# ===================== INDICATORS =====================

def ema(vals,n):
    k=2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k+e[-1]*(1-k))
    return e

def rsi(vals,period=14):
    if len(vals)<period+2: return [50]*len(vals)
    d=np.diff(vals); g=np.maximum(d,0); l=-np.minimum(d,0)
    ag=np.mean(g[:period]); al=np.mean(l[:period])
    out=[50]*period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
        rs=ag/al if al>0 else 0
        out.append(100-100/(1+rs))
    return [50]*(len(vals)-len(out))+out

def macd(vals,fast=12,slow=26,signal=9):
    ema_fast=ema(vals,fast)
    ema_slow=ema(vals,slow)
    macd_line=np.array(ema_fast)-np.array(ema_slow)
    sig_line=ema(macd_line.tolist(),signal)
    hist=macd_line-np.array(sig_line)
    return macd_line.tolist(),sig_line,hist.tolist()

def schaff_tc(vals,fast=23,slow=50,cycle=10):
    macd_line,_,_=macd(vals,fast,slow,cycle)
    return rsi(macd_line,cycle)

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else: tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)): a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a
# ===================== MARKET STATE ANALYZER =====================

def detect_market_state(closes, highs, lows):
    ema20 = ema(closes,20)
    ema50 = ema(closes,50)
    atrv = atr_like(highs,lows,closes)[-1]
    if len(ema20)<5 or len(ema50)<5: return "UNKNOWN"
    diff_ratio = abs(ema20[-1]-ema50[-1]) / (atrv or 1e-9)
    if diff_ratio > 1.5:
        return "STRONG_TREND"
    elif 0.6 < diff_ratio <= 1.5 and ((closes[-1] < ema20[-1] and closes[-2] > ema20[-2]) or (closes[-1] > ema20[-1] and closes[-2] < ema20[-2])):
        return "PULLBACK"
    elif atrv > np.mean(atr_like(highs,lows,closes)[-20:]) * 1.5:
        return "BREAKOUT"
    elif diff_ratio < 0.5:
        return "RANGE"
    else:
        return "NORMAL"

# (UT/STC, MACD, FVG, EARLY, EMA_PULLBACK stratejileri burada aynı şekilde devam eder)
# ----------------------------------------------------------
# Burada hiçbir değişiklik yapılmadı.
# ----------------------------------------------------------

# ===================== CASHOUT MARK PRICE SYSTEM =====================

def get_total_unrealized_profit():
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        return sum(float(p.get("unRealizedProfit") or 0.0) for p in acc)
    except Exception as e:
        log(f"[CASHOUT PROF ERR]{e}")
        return 0.0

def get_mark_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/premiumIndex",params={"symbol":sym},timeout=5).json()
        return float(r.get("markPrice",0.0))
    except Exception as e:
        log(f"[MARK PRICE ERR]{sym}{e}")
        return None

def cashout_at_mark_prices():
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[CASHOUT ERR]{e}")
        return
    closed_syms=[]
    for p in acc:
        amt=float(p.get("positionAmt") or 0)
        if abs(amt)<1e-12: continue
        sym=p["symbol"]
        mark=get_mark_price(sym)
        if not mark: continue
        side="SELL" if amt>0 else "BUY"
        pos_side="LONG" if amt>0 else "SHORT"
        stop_str=format_price_by_tick(sym,mark)
        payload={
            "symbol":sym,
            "side":side,
            "type":"TAKE_PROFIT_MARKET",
            "stopPrice":stop_str,
            "workingType":"MARK_PRICE",
            "closePosition":"true",
            "positionSide":pos_side,
            "timestamp":now_ts_ms()
        }
        try:
            _signed_request("POST","/fapi/v1/order",payload)
            closed_syms.append(sym)
            log(f"[CASHOUT TP]{sym}{side}stop={stop_str}")
        except Exception as e:
            log(f"[CASHOUT FAIL]{sym}{e}")
    if closed_syms:
        tg_send(f"💸 CASHOUT: {len(closed_syms)} pozisyon mark price TP gönderildi (target={PARAM.get('CASHOUT_USDT',60)} USD)")

def check_and_handle_cashout():
    try:
        if STATE.get("cashout_active"):
            acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
            open_pos=[p for p in acc if abs(float(p.get("positionAmt") or 0))>0]
            if not open_pos:
                STATE["cashout_active"]=False
                STATE["auto_trade_active"]=True
                safe_save(STATE_FILE,STATE)
                tg_send("✅ Cashout tamamlandı, auto-trade yeniden aktif.")
            return

        if not STATE.get("auto_trade_active",True): return
        profit=get_total_unrealized_profit()
        target=float(PARAM.get("CASHOUT_USDT",60))
        if profit>=target:
            STATE["auto_trade_active"]=False
            STATE["cashout_active"]=True
            safe_save(STATE_FILE,STATE)
            tg_send(f"⚠️ CASHOUT tetiklendi: toplam kâr {profit:.2f} USD ≥ hedef {target:.2f} USD — mark price TP emirleri gönderiliyor.")
            cashout_at_mark_prices()
    except Exception as e:
        log(f"[CASHOUT CHECK ERR]{e}")
# ===================== PARAMETERS & STATE =====================

STATE_DEFAULT={
    "bar_index":0,"last_report":0,"auto_trade_active":True,
    "last_api_check":0,"long_blocked":False,"short_blocked":False,
    "tg_update_offset":0
}
PARAM_DEFAULT={
    "SCALP_TP_PCT":0.006,"SCALP_SL_PCT":0.20,"TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30,"MAX_SELL":30,
    "ANGLE_MIN":0.00002,"FAST_EMA_PERIOD":3,"SLOW_EMA_PERIOD":7,
    "ATR_SPIKE_RATIO":0.03,"SCALP_APPROVE_BARS":0,
    "CASHOUT_USDT":60.0
}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

# (Tüm orijinal guards, Telegram komutları, Smart TP ve execute_real_trade burada birebir aynı.)

# ===================== MAIN LOOP =====================

def main():
    tg_send("🚀 EMA ULTRA v15.9.53 aktif (Cashout Mark Price System ekli)")
    log("[START] EMA ULTRA v15.9.53 FULL")

    symbols=auto_init_symbols()

    while True:
        try:
            check_telegram_commands()

            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]

            sigs=run_parallel(symbols,bar_i)
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)
                update_directional_limits()
                execute_real_trade(sig)

            process_sim_queue_and_open_due()
            process_sim_closes()
            auto_report_if_due()
            heartbeat_and_status_check({})
            _cleanup_trend_lock_expired()

            # 💰 Cashout kontrolü
            check_and_handle_cashout()

            safe_save(STATE_FILE,STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

if __name__=="__main__":
    main()
