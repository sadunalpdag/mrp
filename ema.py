import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.53 — Active Strategies (EARLY removed)
#  - PEMA ve EARLY tamamen kaldırıldı
#  - Aktif stratejiler:
#       🟢 UT/STC (Ultimate Trend + Schaff Trend Cycle)
#       📈 MACD (EMA20/200 + MACD crossover)
#       🟩 FVG (Fair Value Gap Break)
#       📘 EMA PULLBACK (EMA200 + EMA9/30 + swing break + MarketState)
#       🧩 KIVANC CONFIRM (SuperTrend + EMA9/30 crossover)
#       🧩 C.E.S.T. (50 MA Double Top/Bottom Strategy)
#       📊 EMA-STRUCTURE (123 Move + EMA50 + Confirmation Candles - No SL)
#  - Power filtresi kaldırıldı
#  - Smart TP, 6h TrendLock, Guards, Telegram sistemi aynı
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

# ===================== Kıvanç Confirm Settings =====================
# KIVANC_CONFIRM now uses global PARAM settings (MAX_BUY, MAX_SELL, TRADE_SIZE_USDT)

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

def supertrend(highs, lows, closes, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator
    Returns: (supertrend_values, supertrend_direction)
    direction: "UP" for bullish, "DOWN" for bearish
    """
    atr_vals = atr_like(highs, lows, closes, period)
    
    basic_ub = []
    basic_lb = []
    for i in range(len(closes)):
        hl_avg = (highs[i] + lows[i]) / 2.0
        basic_ub.append(hl_avg + multiplier * atr_vals[i])
        basic_lb.append(hl_avg - multiplier * atr_vals[i])
    
    final_ub = [basic_ub[0]]
    final_lb = [basic_lb[0]]
    
    for i in range(1, len(closes)):
        # Upper band
        if basic_ub[i] < final_ub[i-1] or closes[i-1] > final_ub[i-1]:
            final_ub.append(basic_ub[i])
        else:
            final_ub.append(final_ub[i-1])
        
        # Lower band
        if basic_lb[i] > final_lb[i-1] or closes[i-1] < final_lb[i-1]:
            final_lb.append(basic_lb[i])
        else:
            final_lb.append(final_lb[i-1])
    
    # Determine SuperTrend and direction
    st_values = []
    st_direction = []
    
    # Initial direction
    if closes[0] <= final_ub[0]:
        st_values.append(final_ub[0])
        st_direction.append("DOWN")
    else:
        st_values.append(final_lb[0])
        st_direction.append("UP")
    
    for i in range(1, len(closes)):
        if st_direction[i-1] == "UP":
            if closes[i] <= final_lb[i]:
                st_values.append(final_ub[i])
                st_direction.append("DOWN")
            else:
                st_values.append(final_lb[i])
                st_direction.append("UP")
        else:  # DOWN
            if closes[i] >= final_ub[i]:
                st_values.append(final_lb[i])
                st_direction.append("UP")
            else:
                st_values.append(final_ub[i])
                st_direction.append("DOWN")
    
    return st_values, st_direction

# ===================== MARKET STATE ANALYZER =====================

def detect_market_state(closes, highs, lows):
    ema20 = ema(closes,20)
    ema50 = ema(closes,50)
    atrv = atr_like(highs,lows,closes)[-1]
    if len(ema20)<5 or len(ema50)<5: return "UNKNOWN"
    diff_ratio = abs(ema20[-1]-ema50[-1]) / (atrv or 1e-9)
    # Strong trend: EMA'lar açık ve yön net
    if diff_ratio > 1.5:
        return "STRONG_TREND"
    # Pullback: trend sonrası EMA yakınlaşması
    elif 0.6 < diff_ratio <= 1.5 and ((closes[-1] < ema20[-1] and closes[-2] > ema20[-2]) or (closes[-1] > ema20[-1] and closes[-2] < ema20[-2])):
        return "PULLBACK"
    # Breakout: ATR spike
    elif atrv > np.mean(atr_like(highs,lows,closes)[-20:]) * 1.5:
        return "BREAKOUT"
    # Range: düşük ATR ve EMA sıkışması
    elif diff_ratio < 0.5:
        return "RANGE"
    else:
        return "NORMAL"

# ===================== EMA-STRUCTURE HELPERS =====================

# Configuration constants for EMA-Structure strategy
EMA_STRUCTURE_TOUCH_TOLERANCE = 0.005  # 0.5% - Distance to consider EMA "touched"
EMA_STRUCTURE_STRONG_TREND_THRESHOLD = 0.002  # 0.2% - Extra requirement without confirmation

def detect_higher_high_higher_low(highs, lows, lookback=5):
    """
    Detect Higher High (HH) and Higher Low (HL) pattern
    Returns: (has_hh, has_hl, last_hh_idx, last_hl_idx)
    """
    if len(highs) < lookback + 1:
        return False, False, None, None
    
    # Look for Higher High in recent bars
    # Scan backwards from most recent bar, comparing each bar to its lookback window
    hh_found = False
    hh_idx = None
    # Range: start from second-to-last bar, go back lookback bars, stop at lookback index
    for i in range(len(highs) - 1, max(len(highs) - lookback - 1, lookback), -1):
        if highs[i] > max(highs[max(0, i-lookback):i]):
            hh_found = True
            hh_idx = i
            break
    
    # Look for Higher Low in recent bars
    hl_found = False
    hl_idx = None
    for i in range(len(lows) - 1, max(len(lows) - lookback - 1, lookback), -1):
        if lows[i] > min(lows[max(0, i-lookback):i]):
            hl_found = True
            hl_idx = i
            break
    
    return hh_found, hl_found, hh_idx, hl_idx

def detect_lower_low_lower_high(highs, lows, lookback=5):
    """
    Detect Lower Low (LL) and Lower High (LH) pattern
    Returns: (has_ll, has_lh, last_ll_idx, last_lh_idx)
    """
    if len(highs) < lookback + 1:
        return False, False, None, None
    
    # Look for Lower Low in recent bars
    ll_found = False
    ll_idx = None
    for i in range(len(lows) - 1, max(len(lows) - lookback - 1, lookback), -1):
        if lows[i] < min(lows[max(0, i-lookback):i]):
            ll_found = True
            ll_idx = i
            break
    
    # Look for Lower High in recent bars
    lh_found = False
    lh_idx = None
    for i in range(len(highs) - 1, max(len(highs) - lookback - 1, lookback), -1):
        if highs[i] < max(highs[max(0, i-lookback):i]):
            lh_found = True
            lh_idx = i
            break
    
    return ll_found, lh_found, ll_idx, lh_idx

def detect_confirmation_candle(opens, highs, lows, closes, direction):
    """
    Detect confirmation candle patterns:
    - Bullish: hammer, bullish engulfing
    - Bearish: shooting star, bearish engulfing
    Returns: True if confirmation candle detected
    """
    if len(closes) < 2:
        return False
    
    curr_open = opens[-1]
    curr_high = highs[-1]
    curr_low = lows[-1]
    curr_close = closes[-1]
    prev_open = opens[-2]
    prev_close = closes[-2]
    
    body_size = abs(curr_close - curr_open)
    candle_size = curr_high - curr_low
    
    if candle_size == 0:
        return False
    
    if direction == "UP":
        # Hammer: small body at top, long lower wick
        lower_wick = min(curr_open, curr_close) - curr_low
        upper_wick = curr_high - max(curr_open, curr_close)
        is_hammer = (lower_wick > 2 * body_size and upper_wick < body_size and curr_close > curr_open)
        
        # Bullish Engulfing: current candle body engulfs previous
        is_engulfing = (curr_close > curr_open and 
                       prev_close < prev_open and 
                       curr_close > prev_open and 
                       curr_open < prev_close)
        
        return is_hammer or is_engulfing
    
    else:  # direction == "DOWN"
        # Shooting Star: small body at bottom, long upper wick
        lower_wick = min(curr_open, curr_close) - curr_low
        upper_wick = curr_high - max(curr_open, curr_close)
        is_shooting_star = (upper_wick > 2 * body_size and lower_wick < body_size and curr_close < curr_open)
        
        # Bearish Engulfing: current candle body engulfs previous
        is_engulfing = (curr_close < curr_open and 
                       prev_close > prev_open and 
                       curr_close < prev_open and 
                       curr_open > prev_close)
        
        return is_shooting_star or is_engulfing

# ===================== C.E.S.T. HELPERS =====================

def detect_double_bottom(highs, lows, closes, ma50_values, lookback=10, tolerance=0.015):
    """
    Detect Double Bottom formation with MA touch requirement
    
    Args:
        highs, lows, closes: price arrays
        ma50_values: 50 MA values
        lookback: how many bars to look back for pattern
        tolerance: price tolerance for considering two bottoms similar (1.5%)
    
    Returns:
        (found, bottom1_idx, bottom2_idx, touches_ma)
    """
    if len(lows) < lookback + 2:
        return False, None, None, False
    
    # Find local minima in recent bars (potential bottoms)
    bottoms = []
    for i in range(len(lows) - lookback, len(lows) - 1):
        # Check if this is a local low
        is_local_low = True
        for j in range(max(0, i-2), min(len(lows), i+3)):
            if j != i and lows[j] < lows[i]:
                is_local_low = False
                break
        if is_local_low:
            bottoms.append(i)
    
    # Need at least 2 bottoms
    if len(bottoms) < 2:
        return False, None, None, False
    
    # Check the last two bottoms
    bottom2_idx = bottoms[-1]
    bottom1_idx = bottoms[-2]
    
    bottom1_price = lows[bottom1_idx]
    bottom2_price = lows[bottom2_idx]
    
    # Check if bottoms are similar in price (within tolerance)
    price_diff = abs(bottom1_price - bottom2_price) / max(bottom1_price, 1e-12)
    if price_diff > tolerance:
        return False, None, None, False
    
    # Check if at least one bottom touches MA50
    # Touch means: low/high/close/open is within small distance of MA
    touch_tolerance = 0.005  # 0.5% distance to consider a "touch"
    
    touches_ma = False
    for idx in [bottom1_idx, bottom2_idx]:
        ma50 = ma50_values[idx]
        # Check if any part of the candle touched MA50
        if (abs(lows[idx] - ma50) / max(ma50, 1e-12) < touch_tolerance or
            abs(highs[idx] - ma50) / max(ma50, 1e-12) < touch_tolerance or
            abs(closes[idx] - ma50) / max(ma50, 1e-12) < touch_tolerance):
            touches_ma = True
            break
    
    return True, bottom1_idx, bottom2_idx, touches_ma

def detect_double_top(highs, lows, closes, ma50_values, lookback=10, tolerance=0.015):
    """
    Detect Double Top formation with MA touch requirement
    
    Args:
        highs, lows, closes: price arrays
        ma50_values: 50 MA values
        lookback: how many bars to look back for pattern
        tolerance: price tolerance for considering two tops similar (1.5%)
    
    Returns:
        (found, top1_idx, top2_idx, touches_ma)
    """
    if len(highs) < lookback + 2:
        return False, None, None, False
    
    # Find local maxima in recent bars (potential tops)
    tops = []
    for i in range(len(highs) - lookback, len(highs) - 1):
        # Check if this is a local high
        is_local_high = True
        for j in range(max(0, i-2), min(len(highs), i+3)):
            if j != i and highs[j] > highs[i]:
                is_local_high = False
                break
        if is_local_high:
            tops.append(i)
    
    # Need at least 2 tops
    if len(tops) < 2:
        return False, None, None, False
    
    # Check the last two tops
    top2_idx = tops[-1]
    top1_idx = tops[-2]
    
    top1_price = highs[top1_idx]
    top2_price = highs[top2_idx]
    
    # Check if tops are similar in price (within tolerance)
    price_diff = abs(top1_price - top2_price) / max(top1_price, 1e-12)
    if price_diff > tolerance:
        return False, None, None, False
    
    # Check if at least one top touches MA50
    touch_tolerance = 0.005  # 0.5% distance to consider a "touch"
    
    touches_ma = False
    for idx in [top1_idx, top2_idx]:
        ma50 = ma50_values[idx]
        # Check if any part of the candle touched MA50
        if (abs(lows[idx] - ma50) / max(ma50, 1e-12) < touch_tolerance or
            abs(highs[idx] - ma50) / max(ma50, 1e-12) < touch_tolerance or
            abs(closes[idx] - ma50) / max(ma50, 1e-12) < touch_tolerance):
            touches_ma = True
            break
    
    return True, top1_idx, top2_idx, touches_ma

# ===================== STRATEGIES =====================

def build_utstc_signal(sym, kl, bar_i):
    if len(kl)<60: return None
    closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
    e13=ema(closes,13); e50=ema(closes,50)
    stc_vals=schaff_tc(closes)
    if e13[-1]>e50[-1] and stc_vals[-1]>60 and stc_vals[-2]<=60:
        direction="UP"; tag="🟢 UT/STC BUY"
    elif e13[-1]<e50[-1] and stc_vals[-1]<40 and stc_vals[-2]>=40:
        direction="DOWN"; tag="🔴 UT/STC SELL"
    else: return None
    atr_v=atr_like(highs,lows,closes)[-1]; r_val=rsi(closes)[-1]
    pwr=55+abs(e13[-1]-e50[-1])*200+(r_val-50)/2
    entry=closes[-1]
    tp=entry*(1.006 if direction=="UP" else 0.994)
    sl=entry*(0.8 if direction=="UP" else 1.2)
    return {"symbol":sym,"dir":direction,"tier":"UTSTC","emoji":"🟢" if direction=="UP" else "🔴",
            "entry":entry,"tp":tp,"sl":sl,"power":pwr,"rsi":r_val,"atr":atr_v,
            "time":now_local_iso(),"born_bar":bar_i,"early":False,
            "kind":"UTSTC","tag":tag}

def build_macd_trend_signal(sym, kl, bar_i):
    if len(kl)<200: return None
    closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
    e20=ema(closes,20); e200=ema(closes,200)
    macd_line,sig_line,_=macd(closes)
    if e20[-1]>e200[-1] and macd_line[-1]>sig_line[-1] and macd_line[-2]<=sig_line[-2]:
        direction="UP"; tag="📈 EMA/MACD BUY"
    elif e20[-1]<e200[-1] and macd_line[-1]<sig_line[-1] and macd_line[-2]>=sig_line[-2]:
        direction="DOWN"; tag="📉 EMA/MACD SELL"
    else: return None
    atr_v=atr_like(highs,lows,closes)[-1]; r_val=rsi(closes)[-1]
    pwr=60+abs(e20[-1]-e200[-1])*100+(r_val-50)/2
    entry=closes[-1]
    tp=entry*(1.006 if direction=="UP" else 0.994)
    sl=entry*(0.8 if direction=="UP" else 1.2)
    return {"symbol":sym,"dir":direction,"tier":"MACD","emoji":"📈" if direction=="UP" else "📉",
            "entry":entry,"tp":tp,"sl":sl,"power":pwr,"rsi":r_val,"atr":atr_v,
            "time":now_local_iso(),"born_bar":bar_i,"early":False,
            "kind":"MACD","tag":tag}

def build_fvg_break_signal(sym, kl, bar_i):
    if len(kl)<5: return None
    closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
    h1,h2,h3=highs[-3:]; l1,l2,l3=lows[-3:]; c_now=closes[-1]
    up_gap = l2>h1 and c_now>l2
    dn_gap = h2<l1 and c_now< h2
    if up_gap: direction="UP"; tag="🟩 FVG BREAK BUY"
    elif dn_gap: direction="DOWN"; tag="🟥 FVG BREAK SELL"
    else: return None
    atr_v=atr_like(highs,lows,closes)[-1]; r_val=rsi(closes)[-1]
    pwr=58+(atr_v/(closes[-1] or 1))*150
    entry=closes[-1]
    tp=entry*(1.005 if direction=="UP" else 0.995)
    sl=entry*(0.82 if direction=="UP" else 1.18)
    return {"symbol":sym,"dir":direction,"tier":"FVG","emoji":"🟩" if direction=="UP" else "🟥",
            "entry":entry,"tp":tp,"sl":sl,"power":pwr,"rsi":r_val,"atr":atr_v,
            "time":now_local_iso(),"born_bar":bar_i,"early":False,
            "kind":"FVG","tag":tag}

def build_early_signal(sym, kl, bar_i):
    if len(kl)<60: return None
    try:
        chg=float(requests.get(BINANCE_FAPI+"/fapi/v1/ticker/24hr",
                               params={"symbol":sym},timeout=5).json()["priceChangePercent"])
    except: chg=0.0
    if abs(chg)>=10.0: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    fper=PARAM.get("FAST_EMA_PERIOD",3)
    sper=PARAM.get("SLOW_EMA_PERIOD",7)
    ema_fast=ema(closes,fper)
    ema_slow=ema(closes,sper)

    up_cross = (ema_fast[-2] > ema_slow[-2]) and (ema_fast[-3] <= ema_slow[-3])
    dn_cross = (ema_fast[-2] < ema_slow[-2]) and (ema_fast[-3] >= ema_slow[-3])
    if not (up_cross or dn_cross): return None

    atrs=atr_like(highs,lows,closes)
    if len(atrs)<2: return None
    if not (atrs[-1] >= atrs[-2]*(1.0 + PARAM.get("ATR_SPIKE_RATIO",0.03))):
        return None

    direction="UP" if up_cross else "DOWN"
    entry=closes[-1]
    r_val=rsi(closes)[-1]
    pwr=55 + (abs(ema_slow[-1]-ema_slow[-2])/(atrs[-1] or 1e-12))*20 + ((r_val-50)/50)*15 + (atrs[-1]/entry)*200

    if direction=="UP":
        tp_guess=entry*(1+PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1-PARAM["SCALP_SL_PCT"])
    else:
        tp_guess=entry*(1-PARAM["SCALP_TP_PCT"]); sl_guess=entry*(1+PARAM["SCALP_SL_PCT"])

    return {
        "symbol":sym,"dir":direction,"tier":"EARLY","emoji":"⚡️","entry":entry,
        "tp":tp_guess,"sl":sl_guess,"power":pwr,"rsi":r_val,"atr":atrs[-1],
        "chg24h":chg,"time":now_local_iso(),"born_bar":bar_i,"early":True,
        "kind":"EARLY","tag":"⚡️ EARLY"
    }

def _last_swing_high_low(highs, lows, lookback=5):
    if len(highs) < lookback+2 or len(lows) < lookback+2:
        return None, None
    h_win = highs[-(lookback+1):-1]
    l_win = lows [-(lookback+1):-1]
    return max(h_win), min(l_win)

def build_ema_pullback_signal(sym, kl, bar_i):
    # EMA200 için güvenli tampon
    if len(kl) < 210: return None

    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]

    e9   = ema(closes,9)
    e30  = ema(closes,30)
    e200 = ema(closes,200)
    c_now = closes[-1]

    uptrend   = c_now > e200[-1]
    downtrend = c_now < e200[-1]

    up_pullback_done = (e9[-3] <= e30[-3]) and (e9[-2] > e30[-2])
    dn_pullback_done = (e9[-3] >= e30[-3]) and (e9[-2] < e30[-2])

    swing_h, swing_l = _last_swing_high_low(highs, lows, lookback=5)
    if swing_h is None: 
        return None

    if uptrend and up_pullback_done and (c_now > swing_h):
        direction="UP"; tag="📘 EMA PULLBACK BUY"
    elif downtrend and dn_pullback_done and (c_now < swing_l):
        direction="DOWN"; tag="📘 EMA PULLBACK SELL"
    else:
        return None

    sl_ref = e30[-1]
    if direction=="UP":
        risk = max(1e-12, c_now - sl_ref)
        tp_est = c_now + 1.5 * risk
        sl_est = sl_ref
    else:
        risk = max(1e-12, sl_ref - c_now)
        tp_est = c_now - 1.5 * risk
        sl_est = sl_ref

    atr_v=atr_like(highs,lows,closes)[-1]; r_val=rsi(closes)[-1]
    pwr=60 + abs(e9[-1]-e30[-1])*120 + (r_val-50)/2.0

    sig = {
        "symbol":sym,"dir":direction,"tier":"PULLBACK","emoji":"📘","entry":c_now,
        "tp":tp_est,"sl":sl_est,"power":pwr,"rsi":r_val,"atr":atr_v,
        "time":now_local_iso(),"born_bar":bar_i,"early":False,
        "kind":"EMA_PULLBACK","tag":tag
    }
    # 🔹 Sadece EMA Pullback için Market State etiketi
    sig["market_state"] = detect_market_state(closes, highs, lows)
    return sig

def build_ema_structure_signal(sym, kl, bar_i):
    """
    EMA-Structure Trend Strategy (123 Move + EMA Confirmation)
    
    Strategy Rules:
    1. Trend Definition: Price above/below EMA50 + HH-HL or LL-LH pattern
    2. Entry Point: 123 Move Setup (HH -> HL -> HH or LL -> LH -> LL)
    3. Area of Value Filter: Price pullback to EMA50/EMA20 + Support/Resistance
    4. Confirmation Candle: Engulfing, Hammer, or Shooting Star
    
    Returns signal or None
    """
    if len(kl) < 60:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    opens = [float(k[1]) for k in kl]
    
    # Calculate EMAs
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    
    c_now = closes[-1]
    
    # ========== STEP 1: Trend Definition ==========
    # Uptrend: Price > EMA50 + HH and HL pattern
    # Downtrend: Price < EMA50 + LL and LH pattern
    
    uptrend = c_now > e50[-1]
    downtrend = c_now < e50[-1]
    
    hh_found, hl_found, _, _ = detect_higher_high_higher_low(highs, lows, lookback=5)
    ll_found, lh_found, _, _ = detect_lower_low_lower_high(highs, lows, lookback=5)
    
    # Check if trend structure is valid
    uptrend_valid = uptrend and hh_found and hl_found
    downtrend_valid = downtrend and ll_found and lh_found
    
    if not (uptrend_valid or downtrend_valid):
        return None
    
    # ========== STEP 2: 123 Move Setup ==========
    # For uptrend: HH(1) -> HL(2) -> HH(3) - break above previous HH
    # For downtrend: LL(1) -> LH(2) -> LL(3) - break below previous LL
    
    # Simplified check: look for recent breakout
    if uptrend_valid:
        # Check if current price broke above recent swing high
        if len(highs) >= 6:
            recent_swing_high = max(highs[-6:-1])
        elif len(highs) >= 2:
            recent_swing_high = highs[-2]
        else:
            return None  # Not enough data
        
        if c_now <= recent_swing_high:
            return None
        direction = "UP"
    else:  # downtrend_valid
        # Check if current price broke below recent swing low
        if len(lows) >= 6:
            recent_swing_low = min(lows[-6:-1])
        elif len(lows) >= 2:
            recent_swing_low = lows[-2]
        else:
            return None  # Not enough data
        
        if c_now >= recent_swing_low:
            return None
        direction = "DOWN"
    
    # ========== STEP 3: Area of Value Filter ==========
    # Check if price recently touched EMA50 or EMA20 (pullback)
    # Look back 3-5 bars for EMA touch
    
    touched_ema = False
    for i in range(1, min(6, len(closes))):
        ema50_dist = abs(closes[-i] - e50[-i]) / max(e50[-i], 1e-12)
        ema20_dist = abs(closes[-i] - e20[-i]) / max(e20[-i], 1e-12)
        
        # Consider "touched" if within configured tolerance
        if ema50_dist < EMA_STRUCTURE_TOUCH_TOLERANCE or ema20_dist < EMA_STRUCTURE_TOUCH_TOLERANCE:
            touched_ema = True
            break
    
    # Optional: relax this filter if strong breakout
    atr_v = atr_like(highs, lows, closes)[-1]
    if len(closes) >= 2:
        strong_move = abs(c_now - closes[-2]) / max(atr_v, 1e-12) > 0.5
    else:
        strong_move = False
    
    if not touched_ema and not strong_move:
        return None
    
    # ========== STEP 4: Confirmation Candle ==========
    # Check for confirmation candle pattern
    has_confirmation = detect_confirmation_candle(opens, highs, lows, closes, direction)
    
    # If no confirmation candle, require stronger structure to reduce false signals
    if not has_confirmation:
        # For uptrend: need both HH and HL patterns, plus strong price above EMA
        has_strong_uptrend = (hh_found and hl_found and 
                              c_now > e50[-1] * (1 + EMA_STRUCTURE_STRONG_TREND_THRESHOLD))
        
        # For downtrend: need both LL and LH patterns, plus strong price below EMA
        has_strong_downtrend = (ll_found and lh_found and 
                                c_now < e50[-1] * (1 - EMA_STRUCTURE_STRONG_TREND_THRESHOLD))
        
        # Reject signal if structure is not strong enough
        if direction == "UP" and not has_strong_uptrend:
            return None
        elif direction == "DOWN" and not has_strong_downtrend:
            return None
    
    # ========== Position Management ==========
    # No Stop Loss - Only Take Profit based on entry price
    # Take Profit: Fixed percentage based on entry
    
    if direction == "UP":
        # No stop loss calculation - removed per requirement
        # TP based on fixed percentage (0.6% like other strategies)
        tp_est = c_now * 1.006
        sl_est = c_now * 0.8  # Placeholder for system compatibility (not used)
        tag = "📊 EMA-STRUCTURE BUY"
    else:
        # No stop loss calculation - removed per requirement
        # TP based on fixed percentage (0.6% like other strategies)
        tp_est = c_now * 0.994
        sl_est = c_now * 1.2  # Placeholder for system compatibility (not used)
        tag = "📊 EMA-STRUCTURE SELL"
    
    # Calculate power
    r_val = rsi(closes)[-1]
    pwr = 60 + abs(e50[-1] - e50[-2]) * 150 + (r_val - 50) / 2.0
    if has_confirmation:
        pwr += 5  # Bonus for confirmation candle
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "EMA_STRUCTURE",
        "emoji": "📊",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "EMA_STRUCTURE",
        "tag": tag,
        "has_confirmation": has_confirmation,
        "touched_ema": touched_ema
    }

def build_kivanc_confirm_signal(sym, kl, bar_i):
    """
    Kıvanç Özbilgıç SuperTrend + EMA Cross Strategy
    Signal only on the 1st candle AFTER:
    1) EMA9 crosses EMA30 (crossover just happened)
    2) SuperTrend direction aligns with crossover
    3) Price is within 2% of SuperTrend line
    """
    if len(kl) < 60:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    
    # Calculate SuperTrend
    st_values, st_direction = supertrend(highs, lows, closes)
    
    # Calculate EMAs
    ema9 = ema(closes, 9)
    ema30 = ema(closes, 30)
    
    # Get current and previous values
    st_dir_now = st_direction[-1]
    st_value = st_values[-1]
    ema9_now = ema9[-1]
    ema30_now = ema30[-1]
    ema9_prev = ema9[-2]
    ema30_prev = ema30[-2]
    entry = closes[-1]
    
    # Check for EMA crossover on the CURRENT candle (1st candle after crossover)
    # Bullish crossover: EMA9 was below/equal EMA30, now EMA9 is above EMA30
    bullish_cross = (ema9_prev <= ema30_prev) and (ema9_now > ema30_now)
    
    # Bearish crossover: EMA9 was above/equal EMA30, now EMA9 is below EMA30
    bearish_cross = (ema9_prev >= ema30_prev) and (ema9_now < ema30_now)
    
    # Determine direction based on crossover + SuperTrend alignment
    direction = None
    if bullish_cross and st_dir_now == "UP":
        direction = "UP"
    elif bearish_cross and st_dir_now == "DOWN":
        direction = "DOWN"
    
    if direction is None:
        return None
    
    # Check distance from SuperTrend (price should be close to SuperTrend)
    st_distance_pct = abs(entry - st_value) / max(st_value, 1e-12) * 100
    
    # Maximum allowed distance from SuperTrend (default 2%)
    max_st_distance_pct = 2.0
    
    if st_distance_pct > max_st_distance_pct:
        # Price too far from SuperTrend, skip signal
        return None
    
    # Calculate additional metrics
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    pwr = 60 + abs(ema9_now - ema30_now) * 120 + (r_val - 50) / 2.0
    
    # Set TP and SL like other strategies (will use Smart TP in execute_real_trade)
    if direction == "UP":
        tp = entry * 1.006
        sl = entry * 0.994
    else:
        tp = entry * 0.994
        sl = entry * 1.006
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "KIVANC",
        "emoji": "🧩",
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "KIVANC_CONFIRM",
        "tag": f"🧩 KIVANC {'BUY' if direction == 'UP' else 'SELL'} CROSS",
        "supertrend_dir": st_dir_now,
        "supertrend_value": st_value,
        "st_distance_pct": st_distance_pct,
        "ema9": ema9_now,
        "ema30": ema30_now,
        "crossover": True
    }

def build_cest_signal(sym, kl, bar_i):
    """
    C.E.S.T. – 50 MA Double Top/Bottom Strategy
    
    Strategy Rules:
    📈 Long (Alış):
        - Fiyat 50 MA'nın üstünde olmalı
        - Double Bottom formasyonu oluşmalı
        - İki dipten en az biri 50 MA'ya temas etmeli (gövde veya fitil fark etmez)
        - Entry: Double Bottom sonrası yeşil mum, 50 MA'nın üzerinde kapanmalı
    
    📉 Short (Satış):
        - Fiyat 50 MA'nın altında olmalı
        - Double Top formasyonu oluşmalı
        - İki tepeden en az biri 50 MA'ya temas etmeli
        - Entry: Double Top sonrası kırmızı mum, 50 MA'nın altında kapanmalı
    
    🛑 Stop Loss: Swing Low/High ± 1 ATR
    🎯 Target: Risk:Reward = 1:1.4 (or 1:2)
    """
    if len(kl) < 60:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    opens = [float(k[1]) for k in kl]
    
    # Calculate 50 MA
    ma50 = ema(closes, 50)
    
    # Calculate ATR for stop loss
    atr_vals = atr_like(highs, lows, closes)
    atr_v = atr_vals[-1]
    
    c_now = closes[-1]
    ma50_now = ma50[-1]
    
    # ========== LONG SETUP ==========
    # Check if price is above 50 MA
    if c_now > ma50_now:
        # Detect Double Bottom
        found, bottom1_idx, bottom2_idx, touches_ma = detect_double_bottom(
            highs, lows, closes, ma50, lookback=10, tolerance=0.015
        )
        
        if found and touches_ma:
            # Check for confirmation candle: green candle closing above MA50
            # Current candle should be green (close > open)
            is_green = closes[-1] > opens[-1]
            
            # Previous candle should have been below or at MA50
            prev_below_ma = closes[-2] <= ma50[-2]
            
            if is_green and prev_below_ma:
                direction = "UP"
                
                # Calculate Stop Loss: Last swing low - 1 ATR
                swing_low = min(lows[bottom1_idx], lows[bottom2_idx])
                sl_est = swing_low - atr_v
                
                # Calculate Take Profit: Risk:Reward = 1:1.4
                risk = c_now - sl_est
                tp_est = c_now + (1.4 * risk)
                
                # Calculate power
                r_val = rsi(closes)[-1]
                pwr = 60 + abs(c_now - ma50_now) * 100 + (r_val - 50) / 2.0
                
                return {
                    "symbol": sym,
                    "dir": direction,
                    "tier": "CEST",
                    "emoji": "🧩",
                    "entry": c_now,
                    "tp": tp_est,
                    "sl": sl_est,
                    "power": pwr,
                    "rsi": r_val,
                    "atr": atr_v,
                    "time": now_local_iso(),
                    "born_bar": bar_i,
                    "early": False,
                    "kind": "CEST",
                    "tag": "🧩 C.E.S.T. BUY",
                    "ma50": ma50_now,
                    "swing_low": swing_low
                }
    
    # ========== SHORT SETUP ==========
    # Check if price is below 50 MA
    if c_now < ma50_now:
        # Detect Double Top
        found, top1_idx, top2_idx, touches_ma = detect_double_top(
            highs, lows, closes, ma50, lookback=10, tolerance=0.015
        )
        
        if found and touches_ma:
            # Check for confirmation candle: red candle closing below MA50
            # Current candle should be red (close < open)
            is_red = closes[-1] < opens[-1]
            
            # Previous candle should have been above or at MA50
            prev_above_ma = closes[-2] >= ma50[-2]
            
            if is_red and prev_above_ma:
                direction = "DOWN"
                
                # Calculate Stop Loss: Last swing high + 1 ATR
                swing_high = max(highs[top1_idx], highs[top2_idx])
                sl_est = swing_high + atr_v
                
                # Calculate Take Profit: Risk:Reward = 1:1.4
                risk = sl_est - c_now
                tp_est = c_now - (1.4 * risk)
                
                # Calculate power
                r_val = rsi(closes)[-1]
                pwr = 60 + abs(ma50_now - c_now) * 100 + (r_val - 50) / 2.0
                
                return {
                    "symbol": sym,
                    "dir": direction,
                    "tier": "CEST",
                    "emoji": "🧩",
                    "entry": c_now,
                    "tp": tp_est,
                    "sl": sl_est,
                    "power": pwr,
                    "rsi": r_val,
                    "atr": atr_v,
                    "time": now_local_iso(),
                    "born_bar": bar_i,
                    "early": False,
                    "kind": "CEST",
                    "tag": "🧩 C.E.S.T. SELL",
                    "ma50": ma50_now,
                    "swing_high": swing_high
                }
    
    return None




# ===================== SCANNER =====================

def scan_symbol(sym,bar_i):
    kl=futures_get_klines(sym,"1h",200)
    if len(kl)<60: return []
    res=[]

    # EARLY strategy removed per requirement
    s_utstc = build_utstc_signal(sym,kl,bar_i)
    s_macd  = build_macd_trend_signal(sym,kl,bar_i)
    s_fvg   = build_fvg_break_signal(sym,kl,bar_i)
    s_kivanc = build_kivanc_confirm_signal(sym,kl,bar_i)
    s_cest = build_cest_signal(sym,kl,bar_i)

    # EMA Pullback için 210 bar güvenliği
    kl2 = kl if len(kl)>=210 else futures_get_klines(sym,"1h",210)
    s_pull = build_ema_pullback_signal(sym, kl2, bar_i)
    
    # EMA Structure Strategy
    s_structure = build_ema_structure_signal(sym, kl, bar_i)

    for s in (s_utstc, s_macd, s_fvg, s_kivanc, s_cest, s_pull, s_structure):
        if s: res.append(s)
    
    return res

def run_parallel(symbols,bar_i):
    out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(scan_symbol,s,bar_i) for s in symbols]
        for f in as_completed(futs):
            try: sigs=f.result()
            except: sigs=[]
            if sigs: out.extend(sigs)
    return out

# ===================== RL ENRICH / SIM ENGINE =====================

AI_SIGNALS    = safe_load(AI_SIGNALS_FILE,[])
AI_ANALYSIS   = safe_load(AI_ANALYSIS_FILE,[])
AI_RL         = safe_load(AI_RL_FILE,[])
SIM_POSITIONS = safe_load(SIM_POS_FILE,[])
SIM_CLOSED    = safe_load(SIM_CLOSED_FILE,[])

def enrich_with_ai_context(pos):
    best=None
    for s in reversed(AI_SIGNALS):
        if s.get("symbol")!=pos.get("symbol"): continue
        e_sig=s.get("entry"); e_pos=pos.get("entry")
        if not e_sig or not e_pos: continue
        if abs(e_sig-e_pos)/max(e_sig,1e-12) < 0.002:
            best=s; break
    if best:
        for k in ("rsi","atr","chg24h","born_bar","tier","power","early","kind","tag","market_state"):
            if k in best: pos[k]=best.get(k)
    return pos

def queue_sim_variants(sig):
    delays=[(30*60,"approve_30m",30),(60*60,"approve_1h",60),(90*60,"approve_1h30",90),(120*60,"approve_2h",120)]
    now_s=now_ts_s()
    for secs,label,mins in delays:
        SIM_QUEUE.append({
            "symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
            "entry":sig["entry"],"tp":sig["tp"],"sl":sig["sl"],"power":sig["power"],
            "created_ts":now_s,"open_after_ts":now_s+secs,
            "approve_delay_min":mins,"approve_label":label,
            "status":"PENDING","early":bool(sig.get("early",False)),
            "kind":sig.get("kind",""),"tag":sig.get("tag",""),
            "market_state":sig.get("market_state","")
        })
    safe_save(SIM_POS_FILE,SIM_QUEUE)

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
    SIM_QUEUE[:] = remain
    if opened: safe_save(SIM_POS_FILE,SIM_POSITIONS)

def _unlock_trend_for(sym, delay_unlock=False):
    if delay_unlock:
        TREND_LOCK_TIME[sym]=now_ts_s()
        log(f"[TRENDLOCK DELAY CLEAR] {sym} (6h cooldown started)")
        return
    TREND_LOCK.pop(sym,None); TREND_LOCK_TIME.pop(sym,None)
    log(f"[TRENDLOCK CLEAR] {sym}")

def process_sim_closes():
    global SIM_POSITIONS
    if not SIM_POSITIONS: return
    still=[]; changed=False
    for pos in SIM_POSITIONS:
        if pos.get("status")!="OPEN": 
            still.append(pos); 
            continue
        last=futures_get_price(pos["symbol"])
        if last is None:
            still.append(pos); continue
        hit=None
        if pos["dir"]=="UP":
            if last>=pos["tp"]: hit="TP"
            elif last<=pos["sl"]: hit="SL"
        else:
            if last<=pos["tp"]: hit="TP"
            elif last>=pos["sl"]: hit="SL"
        if hit:
            close_time=now_local_iso()
            gain_pct=((last/pos["entry"]-1.0)*100.0 if pos["dir"]=="UP" else (pos["entry"]/last-1.0)*100.0)
            SIM_CLOSED.append({
                **enrich_with_ai_context(dict(pos)),
                "status":"CLOSED","close_time":close_time,
                "exit_price":last,"exit_reason":hit,"gain_pct":gain_pct
            })
            _unlock_trend_for(pos["symbol"], delay_unlock=True)
            changed=True
            log(f"[SIM CLOSE] {pos['symbol']} {pos['dir']} {hit} {gain_pct:.3f}% approve={pos.get('approve_delay_min')}m kind={pos.get('kind')}")
        else:
            still.append(pos)
    SIM_POSITIONS=still
    if changed:
        safe_save(SIM_POS_FILE,SIM_POSITIONS)
        safe_save(SIM_CLOSED_FILE,SIM_CLOSED)
# ===================== TELEGRAM HELPERS =====================

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
                files={"document":(os.path.basename(p),f)},
                timeout=30
            )
    except: pass

# ===================== BINANCE CORE & HELPERS =====================

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def now_ts_s():  return int(datetime.now(timezone.utc).timestamp())

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    r = (requests.post(url,headers=headers,timeout=10) if m=="POST" else requests.get(url,headers=headers,timeout=10))
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
            "minPrice":float(pricef.get("minPrice","0.00000001")),
            "maxPrice":float(pricef.get("maxPrice","100000000"))
        }
    except Exception as e:
        log(f"[PREC WARN]{sym}{e}")
        PRECISION_CACHE[sym]={"stepSize":0.0001,"tickSize":0.0001,"minPrice":0.00000001,"maxPrice":99999999}
    return PRECISION_CACHE[sym]

def _decimals_from_tick(tick_str):
    try:
        d=Decimal(str(tick_str))
        return max(0,-d.as_tuple().exponent)
    except:
        s=str(tick_str)
        if "." in s: return len(s.split(".")[1])
        return 0

def round_to_tick(sym, price_float):
    f=get_symbol_filters(sym)
    t=Decimal(str(f["tickSize"]))
    p=Decimal(str(price_float))
    if t<=0: return float(p)
    q=(p/t).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    out=(q*t)
    return float(out)

def format_price_by_tick(sym, price_float):
    f=get_symbol_filters(sym)
    dec=_decimals_from_tick(str(f["tickSize"]))
    p_dec=Decimal(str(price_float)).quantize(Decimal(f"1e-{dec}"), rounding=ROUND_HALF_UP)
    if p_dec==Decimal("-0"): p_dec=Decimal("0")
    return f"{float(p_dec):.{dec}f}"

def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                       params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except:
        return None

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                       params={"symbol":sym,"interval":it,"limit":lim},
                       timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms():
            r=r[:-1]
        return r
    except:
        return []

# ===================== POWER/TIER (Bilgi amaçlı) =====================

def calc_power(e_now,e_prev,e_prev2,atr_v,price,rsi_val):
    diff=abs(e_now-e_prev)/(atr_v*0.6) if atr_v>0 else 0
    base=55+diff*20+((rsi_val-50)/50)*15+(atr_v/price)*200
    return min(100,max(0,base))

def tier_from_power(p):
    if 65<=p<75: return "REAL","🟩"
    if p>=75: return "ULTRA","🟦"
    if p>=60: return "NORMAL","🟨"
    return None,""

# ===================== GUARDS / HEARTBEAT / REPORT =====================

STATE_DEFAULT={
    "bar_index":0, "last_report":0, "auto_trade_active":True,
    "last_api_check":0, "long_blocked":False, "short_blocked":False,
    "tg_update_offset":0
}
PARAM_DEFAULT={
    "SCALP_TP_PCT":0.006, "SCALP_SL_PCT":0.20, "TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30, "MAX_SELL":30,
    "ANGLE_MIN":0.00002, "FAST_EMA_PERIOD":3, "SLOW_EMA_PERIOD":7,
    "ATR_SPIKE_RATIO":0.03, "SCALP_APPROVE_BARS":0
}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
if not isinstance(PARAM,dict): PARAM=PARAM_DEFAULT
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

def update_directional_limits():
    live={"long":{}, "short":{},"long_count":0,"short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            amt=float(p["positionAmt"]); sym=p["symbol"]
            if amt>0: live["long"][sym]=amt
            elif amt<0: live["short"][sym]=abs(amt)
        live["long_count"]=len(live["long"])
        live["short_count"]=len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")

    STATE["long_blocked"]  = (live["long_count"]  >= PARAM["MAX_BUY"])
    STATE["short_blocked"] = (live["short_count"] >= PARAM["MAX_SELL"])
    STATE["auto_trade_active"] = not (STATE["long_blocked"] and STATE["short_blocked"])
    safe_save(STATE_FILE,STATE)
    return live

def heartbeat_and_status_check(_snapshot):
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
        try: _=_signed_request("GET","/fapi/v2/account",{"timestamp":now_ts_ms()})
        except: key_ok=False
        hb = (f"✅ HEARTBEAT drift={int(drift)}ms ping={ping_ok} key={key_ok}"
              if ping_ok and key_ok and drift<1500 else
              f"⚠️ HEARTBEAT ping={ping_ok} key={key_ok} drift={int(drift)}")
        tg_send(hb); log(hb)
    except Exception as e:
        tg_send(f"❌ HEARTBEAT {e}"); log(f"[HBERR]{e}")

    msg=(f"📊 STATUS bar:{STATE.get('bar_index',0)} "
         f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
         f"long_blocked:{STATE.get('long_blocked')} "
         f"short_blocked:{STATE.get('short_blocked')} "
         f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
         f"sim_closed:{len(SIM_CLOSED)}")
    tg_send(msg); log(msg)

def ai_log_signal(sig):
    AI_SIGNALS.append({
        "time":now_local_iso(),"symbol":sig["symbol"],"dir":sig["dir"],"tier":sig["tier"],
        "chg24h":sig.get("chg24h"),"power":sig["power"],"rsi":sig.get("rsi"),"atr":sig.get("atr"),
        "tp":sig["tp"],"sl":sig["sl"],"entry":sig["entry"],"born_bar":sig.get("born_bar"),
        "early":bool(sig.get("early",False)),"kind":sig.get("kind",""),"tag":sig.get("tag",""),
        "market_state":sig.get("market_state","")
    })
    safe_save(AI_SIGNALS_FILE,AI_SIGNALS)

def ai_update_analysis_snapshot():
    snapshot={
        "time":now_local_iso(),
        "ultra_signals_total": sum(1 for x in AI_SIGNALS if x.get("tier")=="ULTRA"),
        "real_signals_total":  sum(1 for x in AI_SIGNALS if x.get("tier")=="REAL"),
        "normal_signals_total":sum(1 for x in AI_SIGNALS if x.get("tier")=="NORMAL"),
        # EARLY strategy removed
        "utstc_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="UTSTC"),
        "macd_signals_total":  sum(1 for x in AI_SIGNALS if x.get("kind")=="MACD"),
        "fvg_signals_total":   sum(1 for x in AI_SIGNALS if x.get("kind")=="FVG"),
        "pullback_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="EMA_PULLBACK"),
        "kivanc_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="KIVANC_CONFIRM"),
        "cest_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="CEST"),
        "ema_structure_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="EMA_STRUCTURE"),
        "sim_open_count":len([p for p in SIM_POSITIONS if p.get("status")=="OPEN"]),
        "sim_closed_count":len(SIM_CLOSED)
    }
    AI_ANALYSIS.append(snapshot); safe_save(AI_ANALYSIS_FILE,AI_ANALYSIS)

def auto_report_if_due():
    now_now=time.time()
    if now_now-STATE.get("last_report",0) < 14400:
        return
    ai_update_analysis_snapshot()
    for fpath in [AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,SIM_POS_FILE,SIM_CLOSED_FILE,PARAM_FILE,STATE_FILE]:
        try:
            if os.path.exists(fpath) and os.path.getsize(fpath)>20*1024*1024:
                with open(fpath,"r",encoding="utf-8") as f: raw=f.read()
                tail=raw[-int(len(raw)*0.2):]
                with open(fpath,"w",encoding="utf-8") as f: f.write(tail)
        except: pass
        tg_send_file(fpath, f"📊 AutoBackup {os.path.basename(fpath)}")
    tg_send("🕐 4 saatlik yedek gönderildi.")
    STATE["last_report"]=now_now; safe_save(STATE_FILE,STATE)

# ===================== TELEGRAM KOMUTLARI =====================

def _tg_get_updates():
    if not BOT_TOKEN: return []
    try:
        url=f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params={"timeout":0,"offset":STATE.get("tg_update_offset",0)}
        r=requests.get(url,params=params,timeout=10).json()
        return r.get("result",[])
    except: return []

def _tg_set_offset(new_off):
    STATE["tg_update_offset"]=new_off
    safe_save(STATE_FILE,STATE)

def _cmd_status():
    live=update_directional_limits()
    tg_send(
        f"📊 /status bar:{STATE.get('bar_index')} "
        f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'} "
        f"long:{live.get('long_count',0)} short:{live.get('short_count',0)} "
        f"sim_open:{len([p for p in SIM_POSITIONS if p.get('status')=='OPEN'])} "
        f"sim_closed:{len(SIM_CLOSED)}"
    )

def _cmd_report():
    ai_update_analysis_snapshot()
    tg_send_file(AI_SIGNALS_FILE,"📄 ai_signals.json")
    tg_send_file(AI_ANALYSIS_FILE,"📄 ai_analysis.json")
    tg_send_file(AI_RL_FILE,"📄 ai_rl_log.json")
    tg_send_file(SIM_POS_FILE,"📄 sim_positions.json")
    tg_send_file(SIM_CLOSED_FILE,"📄 sim_closed.json")

def _cmd_set(args):
    try:
        key=args[0]; val=" ".join(args[1:])
        if val.lower() in ("true","false"):
            v = (val.lower()=="true")
        else:
            try:
                v=float(val)
                if v.is_integer(): v=int(v)
            except:
                v=val
        PARAM[key]=v
        safe_save(PARAM_FILE,PARAM)
        tg_send(f"✅ /set {key} = {v}")
    except Exception as e:
        tg_send(f"❌ /set hata: {e}")

def _cmd_export():
    for fpath in [PARAM_FILE,STATE_FILE,AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,SIM_POS_FILE,SIM_CLOSED_FILE,LOG_FILE]:
        tg_send_file(fpath, f"📦 {os.path.basename(fpath)}")

def check_telegram_commands():
    if not BOT_TOKEN or not CHAT_ID: return
    updates=_tg_get_updates()
    if not updates: return
    for up in updates:
        _tg_set_offset(up["update_id"]+1)
        msg=up.get("message") or up.get("edited_message")
        if not msg: continue
        chat_id = str(msg.get("chat",{}).get("id"))
        if chat_id != str(CHAT_ID):
            continue
        text=msg.get("text","").strip()
        if not text.startswith("/"): continue
        parts=text.split(); cmd=parts[0].lower(); args=parts[1:]
        if cmd=="/status": _cmd_status()
        elif cmd=="/report": _cmd_report()
        elif cmd=="/set" and args: _cmd_set(args)
        elif cmd=="/export": _cmd_export()
        else:
            tg_send("Komutlar: /status, /report, /set KEY VALUE, /export")

# ===================== SMART TP =====================

def adjust_precision(sym,v,kind="qty"):
    f=get_symbol_filters(sym)
    step=f["stepSize"] if kind=="qty" else f["tickSize"]
    if step<=0: return v
    return round(round(v/step)*step,12)

def calc_order_qty(sym,entry,usd):
    raw = usd/max(entry,1e-12)
    return adjust_precision(sym,raw,"qty")

def _tp_price_from_usd(direction, entry_exec, tp_usd, trade_usd):
    tp_pct = tp_usd / max(trade_usd,1e-12)
    return (entry_exec*(1+tp_pct) if direction=="UP" else entry_exec*(1-tp_pct)), tp_pct

def futures_set_tp_only(sym, direction, qty, entry_exec, tp_low_usd=1.6, tp_high_usd=2.0):
    try:
        f=get_symbol_filters(sym)
        minp=f["minPrice"]; maxp=f["maxPrice"]
        pos_side="LONG" if direction=="UP" else "SHORT"; side="SELL" if direction=="UP" else "BUY"
        trade_usd=PARAM.get("TRADE_SIZE_USDT",250.0)
        usd_based = entry_exec>0.2

        def try_once(tp_price_candidate, order_type, tp_usd_used=None, tp_pct_used=None):
            price=round_to_tick(sym,tp_price_candidate)
            if price<minp: price=round_to_tick(sym,minp)
            if price>maxp: price=round_to_tick(sym,maxp)
            stop_str=format_price_by_tick(sym,price)
            if float(stop_str)<=0:
                price=round_to_tick(sym,max(minp,1e-12))
                stop_str=format_price_by_tick(sym,price)
                if float(stop_str)<=0:
                    log(f"[TP GUARD] {sym} stop=0 minp jump failed")
                    return False,None,None
            payload={"symbol":sym,"side":side,"type":order_type,"stopPrice":stop_str,
                     "quantity":f"{qty}","workingType":"MARK_PRICE","closePosition":"true",
                     "positionSide":pos_side,"timestamp":now_ts_ms()}
            try:
                _signed_request("POST","/fapi/v1/order",payload)
                log(f"[TP OK] {sym} {order_type} stop={stop_str} qty={qty}")
                return True,tp_usd_used,tp_pct_used
            except Exception as e:
                log(f"[TP FAIL] {sym} {order_type} stop={stop_str} err={e}")
                return False,None,None

        if usd_based:
            for tp_usd in [round(x,1) for x in np.arange(tp_low_usd, tp_high_usd+0.001, 0.1)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,"TAKE_PROFIT_MARKET",tp_usd,tp_pct)
                if ok: return True,u,p
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,"TAKE_PROFIT_MARKET",tp_usd,tp_pct)
                if ok: return True,u,p
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,"STOP_MARKET",tp_usd,tp_pct)
                if ok: return True,u,p
        else:
            for tp_pct in [round(x,4) for x in np.arange(0.0050, 0.0100+0.0001, 0.0005)]:
                tp_price = entry_exec*(1+tp_pct if direction=="UP" else 1-tp_pct)
                ok,u,p=try_once(tp_price,"TAKE_PROFIT_MARKET",None,tp_pct)
                if ok: return True,u,p

        log(f"[NO TP] {sym} uygun TP bulunamadı.")
        return False,None,None
    except Exception as e:
        log(f"[TP ERR]{sym} {e}")
        return False,None,None

# ===================== REAL TRADE =====================

def open_market_position(sym, direction, qty):
    side="BUY" if direction=="UP" else "SELL"
    pos_side="LONG" if direction=="UP" else "SHORT"
    res=_signed_request("POST","/fapi/v1/order",{
        "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
        "positionSide":pos_side,"timestamp":now_ts_ms()
    })
    # Try to get fill price from response, handling zero/empty values properly
    fill = None
    if res.get("avgPrice") is not None:
        try:
            fill = float(res.get("avgPrice"))
            if fill <= 0:
                fill = None
        except (ValueError, TypeError):
            fill = None
    
    if fill is None and res.get("price") is not None:
        try:
            fill = float(res.get("price"))
            if fill <= 0:
                fill = None
        except (ValueError, TypeError):
            fill = None
    
    # Fallback to fetching current market price
    if fill is None or fill <= 0:
        fill = futures_get_price(sym)
        if fill is None or fill <= 0:
            log(f"[PRICE ERR] {sym} could not get valid entry price")
            fill = 0.0
    
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
    if direction=="UP" and STATE.get("long_blocked",False):  return False
    if direction=="DOWN" and STATE.get("short_blocked",False): return False
    return True

def execute_real_trade(sig):
    approve_bars = int(PARAM.get("SCALP_APPROVE_BARS",0))
    if approve_bars>0 and (STATE.get("bar_index",0) - sig.get("born_bar",0) < approve_bars):
        return

    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig["power"]
    kind=sig.get("kind","")

    # 🔒 Duplicate / Direction limits
    if not _can_direction(direction): return
    if _duplicate_or_locked(sym,direction): return

    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return

    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened.get("entry")
        if entry_exec is None or entry_exec <= 0:
            # Try fallback to current price
            entry_exec = futures_get_price(sym)
        if entry_exec is None or entry_exec<=0:
            log(f"[OPEN FAIL] {sym} entry alınamadı."); return

        tp_ok, tp_usd_used, tp_pct_used = futures_set_tp_only(
            sym,direction,qty,entry_exec,tp_low_usd=1.6,tp_high_usd=2.0
        )

        TREND_LOCK[sym]=direction; TREND_LOCK_TIME[sym]=now_ts_s()
        log(f"[TRENDLOCK SET] {sym} {direction}")

        prefix = sig.get("tag", f"🟩 {kind}")
        ms = sig.get("market_state","")
        ms_line = f"State:{ms} " if ms else ""
        if tp_ok:
            tp_line = (f"TP hedefi:{tp_usd_used:.2f}$" if tp_usd_used is not None
                       else f"TP hedefi:%{(tp_pct_used or 0)*100:.2f}")
            tp_pct_show = (tp_pct_used or (tp_usd_used or 0)/max(PARAM.get('TRADE_SIZE_USDT',250.0),1e-12))*100
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"{ms_line}Power:{pwr:.2f}\n"
                    f"Entry:{entry_exec:.12f}\n"
                    f"{tp_line} ({tp_pct_show:.3f}%)\n"
                    f"time:{now_local_iso()}")
        else:
            tg_send(f"{prefix} {sym} {direction} qty:{qty}\n"
                    f"{ms_line}Power:{pwr:.2f}\n"
                    f"Entry:{entry_exec:.12f}\n"
                    f"TP: YOK (USD/% tarama başarısız)\n"
                    f"time:{now_local_iso()}")

        AI_RL.append({
            "time":now_local_iso(),"symbol":sym,"dir":direction,"entry":entry_exec,
            "tp_usd_used":tp_usd_used,"tp_pct_used":tp_pct_used,"tp_ok":tp_ok,
            "power":pwr,"born_bar":sig.get("born_bar"),
            "early":bool(sig.get("early",False)),"kind":kind,"tag":sig.get("tag",""),
            "market_state":sig.get("market_state","")
        })
        safe_save(AI_RL_FILE,AI_RL)

    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")

# ===================== TRENDLOCK TEMİZLİK =====================

def _cleanup_trend_lock_expired():
    now_s=now_ts_s()
    expired=[sym for sym,t in TREND_LOCK_TIME.items() if now_s - t >= TRENDLOCK_EXPIRY_SEC]
    for sym in expired:
        TREND_LOCK.pop(sym,None); TREND_LOCK_TIME.pop(sym,None)
        log(f"[TRENDLOCK TIMEOUT] {sym} (6h cooldown bitti)")

# ===================== SİNYAL DÖNGÜSÜ / MAIN =====================

def auto_init_symbols():
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        symbols=[s["symbol"] for s in info["symbols"]
                 if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING"]
    except Exception as e:
        log(f"[INIT SYMBOLS ERR]{e}"); symbols=[]
    symbols.sort(); return symbols

def main():
    tg_send("🚀 EMA ULTRA v15.9.53 aktif (EARLY removed, EMA-Structure No SL) — Power filter removed")
    log("[START] EMA ULTRA v15.9.53 FULL")

    symbols=auto_init_symbols()

    while True:
        try:
            # Telegram komutları
            updates=_tg_get_updates()
            if updates:
                for up in updates:
                    _tg_set_offset(up["update_id"]+1)
                    msg=up.get("message") or up.get("edited_message")
                    if not msg: continue
                    chat_id = str(msg.get("chat",{}).get("id"))
                    if chat_id != str(CHAT_ID):  # tek chat filtre
                        continue
                    text=msg.get("text","").strip()
                    if not text.startswith("/"): continue
                    parts=text.split(); cmd=parts[0].lower(); args=parts[1:]
                    if cmd=="/status": _cmd_status()
                    elif cmd=="/report": _cmd_report()
                    elif cmd=="/set" and args: _cmd_set(args)
                    elif cmd=="/export": _cmd_export()
                    else:
                        tg_send("Komutlar: /status, /report, /set KEY VALUE, /export")

            # bar index
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]

            # 1) Sinyal tarama
            sigs=run_parallel(symbols,bar_i)

            # 2) Sinyal kayıt + SIM approve + Gerçek trade
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)
                update_directional_limits()
                
                # Execute real trade for all strategies (including KIVANC_CONFIRM)
                execute_real_trade(sig)

            # 3) SIM open/close
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4) 4 saatlik auto-backup
            auto_report_if_due()

            # 5) Heartbeat (10 dk)
            heartbeat_and_status_check({})

            # 6) TrendLock cooldown temizliği
            _cleanup_trend_lock_expired()

            # 7) state save & sleep
            safe_save(STATE_FILE,STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[LOOP ERR]{e}")
            time.sleep(10)

# ===================== ENTRY =====================

if __name__=="__main__":
    main()
