import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.57 — Active Strategies (EARLY removed + 6 New Strategies)
#  - PEMA ve EARLY tamamen kaldırıldı
#  - UT/STC devre dışı bırakıldı
#  - Aktif stratejiler:
#       📈 MACD (EMA20/200 + MACD crossover)
#       🟩 FVG (Fair Value Gap Break)
#       📘 EMA PULLBACK (EMA200 + EMA9/30 + swing break + MarketState)
#       🧩 KIVANC CONFIRM (SuperTrend + EMA9/30 crossover)
#       🧩 C.E.S.T. (50 MA Double Top/Bottom Strategy)
#       🔥 ORB + FVG CONFIRM (Opening Range Breakout + FVG - 09:45-12:00 EST)
#       🌍 LONDON BREAKOUT (LO Session ORB - 08:00-10:00 GMT)
#       🔄 NY REVERSAL (Liquidity Sweep + Reversal - 09:30-11:00 EST)
#       ⚡ ICT POWER OF 3 (Accumulation-Manipulation-Distribution - 08:30-12:00 EST)
#       🌏 ASIAN RANGE BREAKOUT (ARB - 03:00-08:00 GMT)
#       🧱 FVG + BREAKER BLOCK (FVG + Breaker Zone - Session Independent)
#  - Power filtresi kaldırıldı,margin wallet geldi 60 dolarla kar al seceneği eklendi. 
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
REAL_CLOSED_FILE = os.path.join(DATA_DIR,"real_closed.json")
LOG_FILE         = os.path.join(DATA_DIR,"log.txt")
BALANCE_HISTORY_FILE = os.path.join(DATA_DIR,"balance_history.json")

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
REAL_POSITIONS_TRACKER = {}  # Track open positions with strategy info
LAST_REAL_CLOSE_CHECK = 0  # Timestamp of last real close check
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

# ===================== TIME-BASED UTILITIES =====================

def get_current_utc_hour():
    """Get current UTC hour (0-23)"""
    return datetime.now(timezone.utc).hour

def is_in_time_window(start_hour, end_hour):
    """
    Check if current UTC time is within the specified hour window.
    Handles wrap-around for windows that cross midnight.
    
    Args:
        start_hour: Start hour in UTC (0-23)
        end_hour: End hour in UTC (0-23)
    
    Returns:
        bool: True if current time is within window
    """
    current_hour = get_current_utc_hour()
    
    if start_hour <= end_hour:
        # Normal window (e.g., 8-10)
        return start_hour <= current_hour < end_hour
    else:
        # Wrap-around window (e.g., 23-2)
        return current_hour >= start_hour or current_hour < end_hour

def gmt_to_utc(gmt_hour):
    """Convert GMT hour to UTC hour (they're the same, but kept for clarity)"""
    return gmt_hour

def est_to_utc(est_hour):
    """Convert EST hour to UTC hour (EST = UTC-5)"""
    utc_hour = est_hour + 5
    return utc_hour % 24


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
    price_diff = abs(bottom1_price - bottom2_price) / max(abs(bottom1_price), abs(bottom2_price), 1e-12)
    if price_diff > tolerance:
        return False, None, None, False
    
    # Check if at least one bottom touches MA50
    # Touch means: low/high/close/open is within small distance of MA
    touch_tolerance = 0.005  # 0.5% distance to consider a "touch"
    
    touches_ma = False
    for idx in [bottom1_idx, bottom2_idx]:
        ma50 = ma50_values[idx]
        # Check if any part of the candle touched MA50
        if (abs(lows[idx] - ma50) / max(abs(ma50), 1e-12) < touch_tolerance or
            abs(highs[idx] - ma50) / max(abs(ma50), 1e-12) < touch_tolerance or
            abs(closes[idx] - ma50) / max(abs(ma50), 1e-12) < touch_tolerance):
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
    price_diff = abs(top1_price - top2_price) / max(abs(top1_price), abs(top2_price), 1e-12)
    if price_diff > tolerance:
        return False, None, None, False
    
    # Check if at least one top touches MA50
    touch_tolerance = 0.005  # 0.5% distance to consider a "touch"
    
    touches_ma = False
    for idx in [top1_idx, top2_idx]:
        ma50 = ma50_values[idx]
        # Check if any part of the candle touched MA50
        if (abs(lows[idx] - ma50) / max(abs(ma50), 1e-12) < touch_tolerance or
            abs(highs[idx] - ma50) / max(abs(ma50), 1e-12) < touch_tolerance or
            abs(closes[idx] - ma50) / max(abs(ma50), 1e-12) < touch_tolerance):
            touches_ma = True
            break
    
    return True, top1_idx, top2_idx, touches_ma

# ===================== NEW STRATEGIES HELPERS =====================

def get_session_range(klines, start_hour_utc, end_hour_utc):
    """
    Get high and low of a specific session range from klines.
    
    Args:
        klines: List of kline data [[time, open, high, low, close, ...], ...]
        start_hour_utc: Session start hour in UTC
        end_hour_utc: Session end hour in UTC
    
    Returns:
        (range_high, range_low) or (None, None) if not enough data
    """
    if len(klines) < 2:
        return None, None
    
    # Get recent candles within the time window
    session_candles = []
    for k in klines[-30:]:  # Check last 30 candles (30 hours)
        candle_time = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc)
        candle_hour = candle_time.hour
        
        # Check if candle is in session
        if start_hour_utc <= end_hour_utc:
            in_session = start_hour_utc <= candle_hour < end_hour_utc
        else:
            in_session = candle_hour >= start_hour_utc or candle_hour < end_hour_utc
        
        if in_session:
            session_candles.append(k)
    
    if len(session_candles) < 1:
        return None, None
    
    # Get high and low of session
    highs = [float(k[2]) for k in session_candles]
    lows = [float(k[3]) for k in session_candles]
    
    return max(highs), min(lows)

def detect_liquidity_sweep(highs, lows, closes, lookback=10):
    """
    Detect liquidity sweep pattern:
    - Price briefly breaks above previous high or below previous low
    - Then reverses direction (fake breakout)
    
    Returns:
        ("UP", sweep_level) for bullish sweep (broke below then reversed up)
        ("DOWN", sweep_level) for bearish sweep (broke above then reversed down)
        (None, None) if no sweep detected
    """
    if len(closes) < lookback + 2:
        return None, None
    
    # Get recent swing high and low
    recent_high = max(highs[-(lookback+1):-1])
    recent_low = min(lows[-(lookback+1):-1])
    
    current_high = highs[-1]
    current_low = lows[-1]
    current_close = closes[-1]
    prev_close = closes[-2]
    
    # Bullish sweep: broke below recent low but closed back above
    if current_low < recent_low and current_close > recent_low:
        return "UP", recent_low
    
    # Bearish sweep: broke above recent high but closed back below
    if current_high > recent_high and current_close < recent_high:
        return "DOWN", recent_high
    
    return None, None

def detect_breaker_block(highs, lows, closes, direction, lookback=20):
    """
    Detect breaker block: a previous support that became resistance (or vice versa).
    
    Args:
        direction: "UP" or "DOWN" - the intended trade direction
        lookback: how many bars to look back
    
    Returns:
        (found, breaker_level) - True and price level if breaker block found
    """
    if len(closes) < lookback + 5:
        return False, None
    
    # For UP direction: look for old resistance that was broken and is now support
    if direction == "UP":
        # Find a previous high that was broken
        for i in range(len(highs) - lookback, len(highs) - 3):
            level = highs[i]
            
            # Check if this level was broken upward
            broken = False
            for j in range(i + 1, len(closes)):
                if closes[j] > level:
                    broken = True
                    break
            
            if broken:
                # Check if price is now near this level (within 1%)
                current_price = closes[-1]
                distance = abs(current_price - level) / max(level, 1e-12)
                if distance < 0.01 and current_price >= level * 0.995:
                    return True, level
    
    # For DOWN direction: look for old support that was broken and is now resistance
    else:
        # Find a previous low that was broken
        for i in range(len(lows) - lookback, len(lows) - 3):
            level = lows[i]
            
            # Check if this level was broken downward
            broken = False
            for j in range(i + 1, len(closes)):
                if closes[j] < level:
                    broken = True
                    break
            
            if broken:
                # Check if price is now near this level (within 1%)
                current_price = closes[-1]
                distance = abs(current_price - level) / max(level, 1e-12)
                if distance < 0.01 and current_price <= level * 1.005:
                    return True, level
    
    return False, None

def detect_ict_power_of_3(highs, lows, closes, opens):
    """
    Detect ICT Power of 3 pattern:
    1. Accumulation - price consolidates in narrow range
    2. Manipulation - fake breakout (liquidity grab)
    3. Distribution - real move in opposite direction
    
    Returns:
        ("UP", manipulation_level) for bullish setup
        ("DOWN", manipulation_level) for bearish setup
        (None, None) if no pattern
    """
    if len(closes) < 15:
        return None, None
    
    # Phase 1: Check for accumulation (narrow range in bars -10 to -5)
    accumulation_highs = highs[-10:-5]
    accumulation_lows = lows[-10:-5]
    accumulation_range = max(accumulation_highs) - min(accumulation_lows)
    avg_price = sum(closes[-10:-5]) / 5
    
    # Range should be tight (< 1% of price)
    if accumulation_range / max(avg_price, 1e-12) > 0.01:
        return None, None
    
    # Phase 2: Check for manipulation (spike in bars -5 to -2)
    manipulation_high = max(highs[-5:-1])
    manipulation_low = min(lows[-5:-1])
    
    # Phase 3: Check for distribution (current bar shows reversal)
    current_close = closes[-1]
    prev_close = closes[-2]
    
    # Bullish P3: fake breakdown followed by rally
    if manipulation_low < min(accumulation_lows):
        # Check if current price is back above accumulation range
        if current_close > max(accumulation_highs):
            return "UP", manipulation_low
    
    # Bearish P3: fake breakout followed by drop
    if manipulation_high > max(accumulation_highs):
        # Check if current price is back below accumulation range
        if current_close < min(accumulation_lows):
            return "DOWN", manipulation_high
    
    return None, None



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
                    "tag": "🧩 C.E.S.T. SELL",
                    "ma50": ma50_now,
                    "swing_high": swing_high
                }
    
    return None


def build_orb_fvg_confirm_signal(sym, kl, bar_i):
    """
    ORB + FVG Confirm Strategy
    
    Opening Range Breakout combined with Fair Value Gap confirmation.
    Active: 09:45-12:00 EST (14:45-17:00 UTC) - approximate with hourly candles
    Entry: FVG breakout after range breakout
    TP/SL: 2:1 Risk/Reward
    """
    # Time window: 09:45-12:00 EST ≈ 14:00-17:00 UTC (hour-level approximation)
    # Since we work with hourly candles, we use 14:00-17:00 UTC
    if not is_in_time_window(14, 17):
        return None
    
    if len(kl) < 10:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    
    # Get opening range (first 30-60 min of trading, approximate with recent session)
    # Use last 3-5 bars as "opening range"
    or_high = max(highs[-5:-1])
    or_low = min(lows[-5:-1])
    
    c_now = closes[-1]
    
    # Check for range breakout
    broke_high = c_now > or_high
    broke_low = c_now < or_low
    
    if not (broke_high or broke_low):
        return None
    
    # Check for FVG confirmation
    h1, h2, h3 = highs[-3:]
    l1, l2, l3 = lows[-3:]
    
    # FVG patterns
    up_gap = l2 > h1 and c_now > l2
    dn_gap = h2 < l1 and c_now < h2
    
    # Combine: range breakout + FVG
    if broke_high and up_gap:
        direction = "UP"
        tag = "🔥 ORB+FVG BUY"
    elif broke_low and dn_gap:
        direction = "DOWN"
        tag = "🔥 ORB+FVG SELL"
    else:
        return None
    
    # Calculate TP/SL with 2:1 RR
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    if direction == "UP":
        sl_est = or_low
        risk = c_now - sl_est
        tp_est = c_now + 2.0 * risk
    else:
        sl_est = or_high
        risk = sl_est - c_now
        tp_est = c_now - 2.0 * risk
    
    pwr = 62 + (atr_v / c_now) * 150 + (r_val - 50) / 2.0
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "ORB_FVG",
        "emoji": "🔥",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "ORB_FVG_CONFIRM",
        "tag": tag,
        "or_high": or_high,
        "or_low": or_low
    }


def build_london_breakout_signal(sym, kl, bar_i):
    """
    London Breakout (LO) Strategy
    
    London session opening range breakout (08:00-10:00 GMT).
    Entry: Breakout of 30-minute London open range
    TP/SL: 2:1 Risk/Reward
    """
    # Time window: 08:00-10:00 GMT = 08:00-10:00 UTC
    if not is_in_time_window(8, 10):
        return None
    
    if len(kl) < 10:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    
    # Get London opening range (approx first 30 min)
    # Use bars from 08:00-08:30 (first 1-2 bars)
    lo_range_high, lo_range_low = get_session_range(kl, 8, 9)
    
    if lo_range_high is None:
        # Fallback: use recent range
        lo_range_high = max(highs[-3:-1])
        lo_range_low = min(lows[-3:-1])
    
    c_now = closes[-1]
    
    # Check for breakout with EMA20 trend confirmation
    e20 = ema(closes, 20)
    
    # Bullish breakout: price breaks above range + above EMA20
    if c_now > lo_range_high and c_now > e20[-1]:
        direction = "UP"
        tag = "🌍 LONDON BO BUY"
        sl_est = lo_range_low
        risk = c_now - sl_est
        tp_est = c_now + 2.0 * risk
    # Bearish breakout: price breaks below range + below EMA20
    elif c_now < lo_range_low and c_now < e20[-1]:
        direction = "DOWN"
        tag = "🌍 LONDON BO SELL"
        sl_est = lo_range_high
        risk = sl_est - c_now
        tp_est = c_now - 2.0 * risk
    else:
        return None
    
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    pwr = 63 + (atr_v / c_now) * 140 + (r_val - 50) / 2.0
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "LONDON_BO",
        "emoji": "🌍",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "LONDON_BREAKOUT",
        "tag": tag,
        "lo_range_high": lo_range_high,
        "lo_range_low": lo_range_low
    }


def build_ny_reversal_signal(sym, kl, bar_i):
    """
    NY Reversal Strategy
    
    New York reversal with liquidity sweep (09:30-11:00 EST).
    Entry: Liquidity sweep followed by reversal
    TP/SL: 1.5:1 Risk/Reward
    """
    # Time window: 09:30-11:00 EST ≈ 14:00-16:00 UTC (hour-level approximation)
    # Since we work with hourly candles, we use 14:00-16:00 UTC
    if not is_in_time_window(14, 16):
        return None
    
    if len(kl) < 15:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    
    # Detect liquidity sweep
    sweep_dir, sweep_level = detect_liquidity_sweep(highs, lows, closes, lookback=10)
    
    if sweep_dir is None:
        return None
    
    c_now = closes[-1]
    direction = sweep_dir
    
    # Confirm with TrendLock-style logic (RSI)
    r_val = rsi(closes)[-1]
    
    # For UP reversal: RSI should show recovery
    if direction == "UP" and r_val < 40:
        return None
    
    # For DOWN reversal: RSI should show weakness
    if direction == "DOWN" and r_val > 60:
        return None
    
    # Calculate TP/SL with 1.5:1 RR
    atr_v = atr_like(highs, lows, closes)[-1]
    
    if direction == "UP":
        sl_est = sweep_level - atr_v
        risk = c_now - sl_est
        tp_est = c_now + 1.5 * risk
        tag = "🔄 NY REV BUY"
    else:
        sl_est = sweep_level + atr_v
        risk = sl_est - c_now
        tp_est = c_now - 1.5 * risk
        tag = "🔄 NY REV SELL"
    
    pwr = 61 + (atr_v / c_now) * 130 + abs(r_val - 50)
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "NY_REVERSAL",
        "emoji": "🔄",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "NY_REVERSAL",
        "tag": tag,
        "sweep_level": sweep_level
    }


def build_ict_power_of_3_signal(sym, kl, bar_i):
    """
    ICT Power of 3 Strategy
    
    Accumulation -> Manipulation -> Distribution pattern (08:30-12:00 EST).
    Entry: Distribution phase after manipulation
    TP/SL: 2:1 Risk/Reward
    """
    # Time window: 08:30-12:00 EST ≈ 13:00-17:00 UTC (hour-level approximation)
    # Since we work with hourly candles, we use 13:00-17:00 UTC
    if not is_in_time_window(13, 17):
        return None
    
    if len(kl) < 20:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    opens = [float(k[1]) for k in kl]
    
    # Detect P3 pattern
    p3_dir, manipulation_level = detect_ict_power_of_3(highs, lows, closes, opens)
    
    if p3_dir is None:
        return None
    
    c_now = closes[-1]
    direction = p3_dir
    
    # Check FVG for additional confirmation
    h1, h2, h3 = highs[-3:]
    l1, l2, l3 = lows[-3:]
    
    up_gap = l2 > h1 and c_now > l2
    dn_gap = h2 < l1 and c_now < h2
    
    # Require FVG alignment
    if direction == "UP" and not up_gap:
        return None
    if direction == "DOWN" and not dn_gap:
        return None
    
    # Calculate TP/SL with 2:1 RR
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    if direction == "UP":
        sl_est = manipulation_level
        risk = c_now - sl_est
        tp_est = c_now + 2.0 * risk
        tag = "⚡ ICT P3 BUY"
    else:
        sl_est = manipulation_level
        risk = sl_est - c_now
        tp_est = c_now - 2.0 * risk
        tag = "⚡ ICT P3 SELL"
    
    pwr = 64 + (atr_v / c_now) * 145 + (r_val - 50) / 2.0
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "ICT_P3",
        "emoji": "⚡",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "ICT_POWER_OF_3",
        "tag": tag,
        "manipulation_level": manipulation_level
    }


def build_asian_range_breakout_signal(sym, kl, bar_i):
    """
    Asian Range Breakout (ARB) Strategy
    
    Asian session range breakout (03:00-08:00 GMT).
    Entry: Breakout of Asian range during London/NY session
    TP/SL: 2:1 Risk/Reward
    """
    # Active during Asian session breakout time: 03:00-08:00 GMT = 03:00-08:00 UTC
    # But we also allow signals shortly after (08:00-09:00) for breakout confirmation
    if not is_in_time_window(3, 9):
        return None
    
    if len(kl) < 10:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    
    # Get Asian session range (03:00-08:00 GMT)
    asian_high, asian_low = get_session_range(kl, 3, 8)
    
    if asian_high is None:
        # Fallback: use recent tight range
        asian_high = max(highs[-6:-1])
        asian_low = min(lows[-6:-1])
    
    c_now = closes[-1]
    
    # Check for breakout
    broke_high = c_now > asian_high
    broke_low = c_now < asian_low
    
    if not (broke_high or broke_low):
        return None
    
    # Calculate TP/SL with 2:1 RR
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    if broke_high:
        direction = "UP"
        tag = "🌏 ASIA BO BUY"
        sl_est = asian_low
        risk = c_now - sl_est
        tp_est = c_now + 2.0 * risk
    else:
        direction = "DOWN"
        tag = "🌏 ASIA BO SELL"
        sl_est = asian_high
        risk = sl_est - c_now
        tp_est = c_now - 2.0 * risk
    
    pwr = 62 + (atr_v / c_now) * 135 + (r_val - 50) / 2.0
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "ASIAN_BO",
        "emoji": "🌏",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "ASIAN_RANGE_BREAKOUT",
        "tag": tag,
        "asian_high": asian_high,
        "asian_low": asian_low
    }


def build_fvg_breaker_block_signal(sym, kl, bar_i):
    """
    FVG + Breaker Block Strategy
    
    Fair Value Gap with Breaker Block confirmation (session independent).
    Entry: FVG breakout at breaker block level
    TP/SL: 2:1 Risk/Reward
    """
    # Session independent - no time filter
    
    if len(kl) < 25:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    
    # Check for FVG first
    h1, h2, h3 = highs[-3:]
    l1, l2, l3 = lows[-3:]
    c_now = closes[-1]
    
    up_gap = l2 > h1 and c_now > l2
    dn_gap = h2 < l1 and c_now < h2
    
    if not (up_gap or dn_gap):
        return None
    
    # Determine direction
    direction = "UP" if up_gap else "DOWN"
    
    # Check for breaker block confirmation
    has_breaker, breaker_level = detect_breaker_block(highs, lows, closes, direction, lookback=20)
    
    if not has_breaker:
        return None
    
    # Calculate TP/SL with 2:1 RR
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    if direction == "UP":
        sl_est = breaker_level - atr_v
        risk = c_now - sl_est
        tp_est = c_now + 2.0 * risk
        tag = "🧱 FVG+BREAKER BUY"
    else:
        sl_est = breaker_level + atr_v
        risk = sl_est - c_now
        tp_est = c_now - 2.0 * risk
        tag = "🧱 FVG+BREAKER SELL"
    
    pwr = 65 + (atr_v / c_now) * 140 + (r_val - 50) / 2.0
    
    return {
        "symbol": sym,
        "dir": direction,
        "tier": "FVG_BREAKER",
        "emoji": "🧱",
        "entry": c_now,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "FVG_BREAKER_BLOCK",
        "tag": tag,
        "breaker_level": breaker_level
    }




def scan_symbol(sym,bar_i):
    kl=futures_get_klines(sym,"1h",200)
    if len(kl)<60: return []
    res=[]

    # EARLY strategy removed per requirement
    # UT/STC strategy disabled per requirement
    s_utstc = None  # Disabled - was: build_utstc_signal(sym,kl,bar_i)
    s_macd  = build_macd_trend_signal(sym,kl,bar_i)
    s_fvg   = build_fvg_break_signal(sym,kl,bar_i)
    s_kivanc = build_kivanc_confirm_signal(sym,kl,bar_i)
    s_cest = build_cest_signal(sym,kl,bar_i)

    # EMA Pullback için 210 bar güvenliği
    kl2 = kl if len(kl)>=210 else futures_get_klines(sym,"1h",210)
    s_pull = build_ema_pullback_signal(sym, kl2, bar_i)
    
    # New strategies (6 new ones)
    s_orb_fvg = build_orb_fvg_confirm_signal(sym, kl, bar_i)
    s_london_bo = build_london_breakout_signal(sym, kl, bar_i)
    s_ny_rev = build_ny_reversal_signal(sym, kl, bar_i)
    s_ict_p3 = build_ict_power_of_3_signal(sym, kl, bar_i)
    s_asian_bo = build_asian_range_breakout_signal(sym, kl, bar_i)
    s_fvg_breaker = build_fvg_breaker_block_signal(sym, kl, bar_i)

    for s in (s_utstc, s_macd, s_fvg, s_kivanc, s_cest, s_pull,
              s_orb_fvg, s_london_bo, s_ny_rev, s_ict_p3, s_asian_bo, s_fvg_breaker):
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
REAL_CLOSED   = safe_load(REAL_CLOSED_FILE,[])
BALANCE_HISTORY = safe_load(BALANCE_HISTORY_FILE,[])

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



def _unlock_trend_for(sym, delay_unlock=False):
    if delay_unlock:
        TREND_LOCK_TIME[sym]=now_ts_s()
        log(f"[TRENDLOCK DELAY CLEAR] {sym} (6h cooldown started)")
        return
    TREND_LOCK.pop(sym,None); TREND_LOCK_TIME.pop(sym,None)
    log(f"[TRENDLOCK CLEAR] {sym}")



def check_and_log_real_closed_trades():
    """
    Check for closed real positions and log them with strategy information.
    This runs periodically to track which strategies resulted in closed trades.
    Throttled to run max once per minute to avoid excessive API calls.
    """
    global REAL_CLOSED, REAL_POSITIONS_TRACKER, LAST_REAL_CLOSE_CHECK
    
    # Throttle: only check once per minute
    now = now_ts_s()
    if now - LAST_REAL_CLOSE_CHECK < 60:
        return
    LAST_REAL_CLOSE_CHECK = now
    
    try:
        # Get current positions from Binance
        acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
        current_positions = {}
        
        for p in acc:
            amt = float(p["positionAmt"])
            if amt != 0:  # Position is still open
                sym = p["symbol"]
                current_positions[sym] = {
                    "symbol": sym,
                    "amount": amt,
                    "entry_price": float(p["entryPrice"]),
                    "unrealized_pnl": float(p["unRealizedProfit"])
                }
        
        # Check if any tracked positions have closed
        closed_symbols = []
        for sym, pos_info in REAL_POSITIONS_TRACKER.items():
            if sym not in current_positions:
                # Position has closed
                closed_symbols.append(sym)
                
                # Try to get the last trade to find exit price
                exit_price = None
                pnl = None
                try:
                    trades = _signed_request("GET", "/fapi/v3/userTrades", {
                        "symbol": sym,
                        "limit": 10,
                        "timestamp": now_ts_ms()
                    })
                    # Find the closing trade (most recent opposite direction trade)
                    for trade in reversed(trades):
                        if trade["symbol"] == sym:
                            exit_price = float(trade["price"])
                            break
                except:
                    pass
                
                # Calculate PnL percentage if we have exit price
                entry_price = pos_info.get("entry_price", 0)
                direction = pos_info.get("direction")
                if exit_price and entry_price > 0:
                    if direction == "UP":
                        pnl_pct = ((exit_price / entry_price) - 1) * 100
                    else:  # SHORT
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                else:
                    pnl_pct = None
                
                # Log the closed trade with strategy information
                closed_trade = {
                    "symbol": sym,
                    "direction": direction,
                    "strategy": pos_info.get("kind", "UNKNOWN"),
                    "tag": pos_info.get("tag", ""),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "power": pos_info.get("power"),
                    "open_time": pos_info.get("open_time"),
                    "close_time": now_local_iso(),
                    "tp_target": pos_info.get("tp_target"),
                    "market_state": pos_info.get("market_state", "")
                }
                
                REAL_CLOSED.append(closed_trade)
                safe_save(REAL_CLOSED_FILE, REAL_CLOSED)
                
                pnl_str = f"{pnl_pct:.2f}" if pnl_pct is not None else "N/A"
                exit_str = f"{exit_price}" if exit_price is not None else "N/A"
                log(f"[REAL CLOSED] {sym} {direction} Strategy:{pos_info.get('kind', 'UNKNOWN')} "
                    f"PnL:{pnl_str}% Exit:{exit_str}")
        
        # Remove closed positions from tracker
        for sym in closed_symbols:
            REAL_POSITIONS_TRACKER.pop(sym, None)
            
    except Exception as e:
        log(f"[CHECK REAL CLOSED ERR] {e}")

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

def futures_get_mark_price(sym):
    """Get mark price from Binance Futures premiumIndex endpoint"""
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/premiumIndex",
                       params={"symbol":sym},timeout=5).json()
        return float(r["markPrice"])
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
    "cest_long_blocked":False, "cest_short_blocked":False,
    "tg_update_offset":0,
    "initial_margin_balance":0.0, "last_profit_check_ts":0,
    "last_hourly_margin_log":0
}
PARAM_DEFAULT={
    "SCALP_TP_PCT":0.006, "SCALP_SL_PCT":0.20, "TRADE_SIZE_USDT":250.0,
    "MAX_BUY":30, "MAX_SELL":30,
    "MAX_CEST_BUY":15, "MAX_CEST_SELL":15,
    "ANGLE_MIN":0.00002, "FAST_EMA_PERIOD":3, "SLOW_EMA_PERIOD":7,
    "ATR_SPIKE_RATIO":0.03, "SCALP_APPROVE_BARS":0,
    "PROFIT_TARGET_USD":20.0
}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
if not isinstance(PARAM,dict): PARAM=PARAM_DEFAULT
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

def update_directional_limits():
    live={"long":{}, "short":{},"long_count":0,"short_count":0,"cest_long_count":0,"cest_short_count":0}
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
        for p in acc:
            amt=float(p["positionAmt"]); sym=p["symbol"]
            if amt>0: 
                live["long"][sym]=amt
                # Check if this is a CEST position
                if sym in REAL_POSITIONS_TRACKER and REAL_POSITIONS_TRACKER[sym].get("kind") == "CEST":
                    live["cest_long_count"] += 1
            elif amt<0: 
                live["short"][sym]=abs(amt)
                # Check if this is a CEST position
                if sym in REAL_POSITIONS_TRACKER and REAL_POSITIONS_TRACKER[sym].get("kind") == "CEST":
                    live["cest_short_count"] += 1
        live["long_count"]=len(live["long"])
        live["short_count"]=len(live["short"])
    except Exception as e:
        log(f"[FETCH POS ERR]{e}")

    STATE["long_blocked"]  = (live["long_count"]  >= PARAM["MAX_BUY"])
    STATE["short_blocked"] = (live["short_count"] >= PARAM["MAX_SELL"])
    STATE["cest_long_blocked"]  = (live["cest_long_count"]  >= PARAM.get("MAX_CEST_BUY", 15))
    STATE["cest_short_blocked"] = (live["cest_short_count"] >= PARAM.get("MAX_CEST_SELL", 15))
    STATE["auto_trade_active"] = not (STATE["long_blocked"] and STATE["short_blocked"])
    safe_save(STATE_FILE,STATE)
    return live

# ===================== CASH OUT / PROFIT TARGET =====================

def get_account_balance():
    """Fetch current futures account balance (margin balance)"""
    try:
        acc = _signed_request("GET", "/fapi/v2/account", {"timestamp": now_ts_ms()})
        # Get total margin balance (includes unrealized PnL)
        balance = float(acc.get("totalMarginBalance", 0))
        return balance
    except Exception as e:
        log(f"[GET BALANCE ERR] {e}")
        return None

def get_unrealized_pnl():
    """Get total unrealized PnL from all open positions"""
    try:
        acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
        total_pnl = sum(float(p.get("unRealizedProfit", 0)) for p in acc)
        return total_pnl
    except Exception as e:
        log(f"[GET UNREALIZED PNL ERR] {e}")
        return 0.0

def close_all_positions_at_market(exit_reason="PROFIT_TARGET"):
    """
    Close all open positions at market price.
    Args:
        exit_reason: Reason for closing ("PROFIT_TARGET" or "MANUAL_CLOSE")
    Returns list of closed position symbols.
    """
    closed_symbols = []
    try:
        # Get all open positions
        acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
        
        for p in acc:
            amt = float(p["positionAmt"])
            if amt == 0:  # Skip positions with no amount
                continue
            
            sym = p["symbol"]
            
            # Determine side and position side
            if amt > 0:  # Long position
                side = "SELL"
                pos_side = "LONG"
            else:  # Short position
                side = "BUY"
                pos_side = "SHORT"
                amt = abs(amt)
            
            # Place market close order
            try:
                payload = {
                    "symbol": sym,
                    "side": side,
                    "type": "MARKET",
                    "positionSide": pos_side,
                    "closePosition": "true",
                    "timestamp": now_ts_ms()
                }
                res = _signed_request("POST", "/fapi/v1/order", payload)
                closed_symbols.append(sym)
                log(f"[CLOSE ALL] {sym} {pos_side} closed at market")
                
                # Remove from trend lock
                TREND_LOCK.pop(sym, None)
                TREND_LOCK_TIME.pop(sym, None)
                
                # Log to closed trades with exit reason
                entry_price = float(p.get("entryPrice", 0))
                # Get mark price as exit price
                exit_price = futures_get_mark_price(sym)
                
                # Get position info from tracker if available
                pos_info = REAL_POSITIONS_TRACKER.get(sym, {})
                
                # Calculate PnL percentage
                direction = "UP" if pos_side == "LONG" else "DOWN"
                if exit_price and entry_price > 0:
                    if direction == "UP":
                        pnl_pct = ((exit_price / entry_price) - 1) * 100
                    else:
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                else:
                    pnl_pct = None
                
                closed_trade = {
                    "symbol": sym,
                    "direction": direction,
                    "strategy": pos_info.get("kind", "UNKNOWN"),
                    "tag": pos_info.get("tag", ""),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "power": pos_info.get("power"),
                    "open_time": pos_info.get("open_time"),
                    "close_time": now_local_iso(),
                    "exit_reason": exit_reason,
                    "market_state": pos_info.get("market_state", ""),
                    "closed_by_profit_target": (exit_reason == "PROFIT_TARGET")
                }
                
                REAL_CLOSED.append(closed_trade)
                
                # Remove from tracker
                REAL_POSITIONS_TRACKER.pop(sym, None)
                
            except Exception as e:
                log(f"[CLOSE ALL ERR] {sym} {e}")
        
        # Save closed trades
        if closed_symbols:
            safe_save(REAL_CLOSED_FILE, REAL_CLOSED)
        
        return closed_symbols
    except Exception as e:
        log(f"[CLOSE ALL POSITIONS ERR] {e}")
        return []

def check_profit_target():
    """
    Check if profit target has been reached.
    If yes, close all positions and reset initial balance.
    Throttled to run max once per 30 seconds.
    """
    global STATE
    
    # Throttle: only check every 30 seconds
    now = now_ts_s()
    if now - STATE.get("last_profit_check_ts", 0) < 30:
        return
    
    STATE["last_profit_check_ts"] = now
    
    # Get initial balance
    initial_balance = STATE.get("initial_margin_balance", 0)
    
    # If no initial balance is set, set it now
    if initial_balance == 0:
        current_balance = get_account_balance()
        if current_balance:
            STATE["initial_margin_balance"] = current_balance
            safe_save(STATE_FILE, STATE)
            log(f"[CASH OUT] Initial margin balance set: ${current_balance:.2f}")
        return
    
    # Get current balance
    current_balance = get_account_balance()
    if not current_balance:
        return
    
    # Calculate profit
    profit = current_balance - initial_balance
    
    # Get profit target
    profit_target = PARAM.get("PROFIT_TARGET_USD", 60.0)
    
    # Check if profit target reached
    if profit >= profit_target:
        log(f"[CASH OUT] Profit target reached! Profit: ${profit:.2f}, Target: ${profit_target:.2f}")
        tg_send(f"💰 CASH OUT - Profit target reached!\n"
                f"Initial Balance: ${initial_balance:.2f}\n"
                f"Current Balance: ${current_balance:.2f}\n"
                f"Profit: ${profit:.2f} (Target: ${profit_target:.2f})\n"
                f"Closing all positions at mark price...")
        
        # Close all positions
        closed_symbols = close_all_positions_at_market()
        
        if closed_symbols:
            tg_send(f"✅ Closed {len(closed_symbols)} positions: {', '.join(closed_symbols[:10])}")
            log(f"[CASH OUT] Closed {len(closed_symbols)} positions")
        else:
            tg_send(f"ℹ️ No open positions to close")
        
        # Get new balance after closing
        time.sleep(2)  # Wait for orders to settle
        new_balance = get_account_balance()
        if new_balance:
            STATE["initial_margin_balance"] = new_balance
            safe_save(STATE_FILE, STATE)
            final_profit = new_balance - initial_balance
            tg_send(f"✅ Cash out complete!\n"
                    f"New margin balance: ${new_balance:.2f}\n"
                    f"Realized profit: ${final_profit:.2f}")
            log(f"[CASH OUT] Complete. New balance: ${new_balance:.2f}, Realized: ${final_profit:.2f}")

def send_hourly_margin_log():
    """
    Send hourly Telegram log showing how much is left until margin cashout target.
    This runs once per hour to keep users informed of progress.
    Also tracks balance changes history and estimates time to target.
    """
    global STATE, BALANCE_HISTORY
    
    # Check if an hour has passed since last log
    now = now_ts_s()
    last_log = STATE.get("last_hourly_margin_log", 0)
    
    # Hourly check: 3600 seconds = 1 hour
    if now - last_log < 3600:
        return
    
    # Update last log time
    STATE["last_hourly_margin_log"] = now
    safe_save(STATE_FILE, STATE)
    
    try:
        # Get current balance
        current_balance = get_account_balance()
        if not current_balance:
            log("[HOURLY MARGIN LOG] Could not fetch balance")
            return
        
        # Get initial balance
        initial_balance = STATE.get("initial_margin_balance", 0)
        
        # If no initial balance is set, set it now and skip this log
        if initial_balance == 0:
            STATE["initial_margin_balance"] = current_balance
            safe_save(STATE_FILE, STATE)
            log(f"[HOURLY MARGIN LOG] Initial margin balance set: ${current_balance:.2f}")
            return
        
        # Get profit target
        profit_target = PARAM.get("PROFIT_TARGET_USD", 60.0)
        
        # Calculate current profit
        current_profit = current_balance - initial_balance
        
        # Calculate remaining to target
        remaining = profit_target - current_profit
        
        # Calculate progress percentage
        progress_pct = (current_profit / profit_target * 100) if profit_target > 0 else 0
        
        # Get unrealized PnL
        unrealized_pnl = get_unrealized_pnl()
        
        # Get open positions count and CEST positions
        try:
            acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
            open_positions = 0
            cest_long_count = 0
            cest_short_count = 0
            
            for p in acc:
                amt = float(p["positionAmt"])
                if amt != 0:
                    open_positions += 1
                    sym = p["symbol"]
                    
                    # Check if this is a CEST position
                    if sym in REAL_POSITIONS_TRACKER and REAL_POSITIONS_TRACKER[sym].get("kind") == "CEST":
                        if amt > 0:
                            cest_long_count += 1
                        else:
                            cest_short_count += 1
        except:
            open_positions = 0
            cest_long_count = 0
            cest_short_count = 0
        
        # Calculate estimated hours to target based on recent profit rate
        estimated_hours = None
        profit_per_hour = None
        
        if len(BALANCE_HISTORY) > 0 and remaining > 0:
            # Get the last balance record
            last_record = BALANCE_HISTORY[-1]
            last_balance = last_record.get("balance", initial_balance)
            last_timestamp = last_record.get("timestamp", now - 3600)
            
            # Calculate profit change since last record
            balance_change = current_balance - last_balance
            time_elapsed_hours = (now - last_timestamp) / 3600.0
            
            if time_elapsed_hours > 0 and balance_change > 0:
                # Calculate profit per hour
                profit_per_hour = balance_change / time_elapsed_hours
                # Estimate hours to reach target
                estimated_hours = remaining / profit_per_hour
        
        # Record this balance change in history
        balance_record = {
            "timestamp": now,
            "time": now_local_iso(),
            "balance": current_balance,
            "initial_balance": initial_balance,
            "current_profit": current_profit,
            "target": profit_target,
            "remaining": remaining,
            "progress_pct": progress_pct,
            "unrealized_pnl": unrealized_pnl,
            "open_positions": open_positions,
            "cest_long_count": cest_long_count,
            "cest_short_count": cest_short_count,
            "profit_per_hour": profit_per_hour,
            "estimated_hours_to_target": estimated_hours
        }
        
        BALANCE_HISTORY.append(balance_record)
        
        # Keep only last 1000 records to prevent file from growing too large
        if len(BALANCE_HISTORY) > 1000:
            BALANCE_HISTORY[:] = BALANCE_HISTORY[-1000:]
        
        safe_save(BALANCE_HISTORY_FILE, BALANCE_HISTORY)
        
        # Send the hourly log
        if remaining > 0:
            msg = (f"⏰ HOURLY MARGIN UPDATE\n"
                   f"━━━━━━━━━━━━━━━━\n"
                   f"💰 Current Profit: ${current_profit:.2f}\n"
                   f"🎯 Target: ${profit_target:.2f}\n"
                   f"📊 Remaining: ${remaining:.2f}\n"
                   f"📈 Progress: {progress_pct:.1f}%\n"
                   f"💵 Unrealized PnL: ${unrealized_pnl:.2f}\n"
                   f"📌 Open Positions: {open_positions}\n"
                   f"🧩 CEST Long: {cest_long_count}/{PARAM.get('MAX_CEST_BUY', 15)}\n"
                   f"🧩 CEST Short: {cest_short_count}/{PARAM.get('MAX_CEST_SELL', 15)}")
            
            # Add estimated time to target if available
            if estimated_hours is not None:
                if estimated_hours < 1:
                    minutes = int(estimated_hours * 60)
                    msg += f"\n⏱️ Est. Time to Target: ~{minutes} min"
                else:
                    msg += f"\n⏱️ Est. Time to Target: ~{estimated_hours:.1f} hrs"
            
            msg += f"\n🕐 {now_local_iso()}"
        else:
            # Target already reached (shouldn't normally happen as positions would be closed)
            msg = (f"⏰ HOURLY MARGIN UPDATE\n"
                   f"━━━━━━━━━━━━━━━━\n"
                   f"✅ TARGET REACHED!\n"
                   f"💰 Current Profit: ${current_profit:.2f}\n"
                   f"🎯 Target: ${profit_target:.2f}\n"
                   f"📊 Excess: ${-remaining:.2f}\n"
                   f"💵 Unrealized PnL: ${unrealized_pnl:.2f}\n"
                   f"📌 Open Positions: {open_positions}\n"
                   f"🧩 CEST Long: {cest_long_count}/{PARAM.get('MAX_CEST_BUY', 15)}\n"
                   f"🧩 CEST Short: {cest_short_count}/{PARAM.get('MAX_CEST_SELL', 15)}\n"
                   f"🕐 {now_local_iso()}")
        
        tg_send(msg)
        log(f"[HOURLY MARGIN LOG] Sent. Profit: ${current_profit:.2f}, Remaining: ${remaining:.2f}, Est: {estimated_hours:.1f}h" if estimated_hours else f"[HOURLY MARGIN LOG] Sent. Profit: ${current_profit:.2f}, Remaining: ${remaining:.2f}")
        
    except Exception as e:
        log(f"[HOURLY MARGIN LOG ERR] {e}")

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
         f"cest_long_blocked:{STATE.get('cest_long_blocked')} "
         f"cest_short_blocked:{STATE.get('cest_short_blocked')}")
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
        # New strategies tracking
        "orb_fvg_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="ORB_FVG_CONFIRM"),
        "london_bo_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="LONDON_BREAKOUT"),
        "ny_reversal_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="NY_REVERSAL"),
        "ict_p3_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="ICT_POWER_OF_3"),
        "asian_bo_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="ASIAN_RANGE_BREAKOUT"),
        "fvg_breaker_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="FVG_BREAKER_BLOCK")
    }
    AI_ANALYSIS.append(snapshot); safe_save(AI_ANALYSIS_FILE,AI_ANALYSIS)

def auto_report_if_due():
    now_now=time.time()
    if now_now-STATE.get("last_report",0) < 14400:
        return
    ai_update_analysis_snapshot()
    for fpath in [AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,REAL_CLOSED_FILE,PARAM_FILE,STATE_FILE]:
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
        f"long:{live.get('long_count',0)}/{PARAM.get('MAX_BUY',30)} short:{live.get('short_count',0)}/{PARAM.get('MAX_SELL',30)} "
        f"cest_long:{live.get('cest_long_count',0)}/{PARAM.get('MAX_CEST_BUY',15)} cest_short:{live.get('cest_short_count',0)}/{PARAM.get('MAX_CEST_SELL',15)} "
        f"real_closed:{len(REAL_CLOSED)}"
    )

def _cmd_report():
    ai_update_analysis_snapshot()
    tg_send_file(AI_SIGNALS_FILE,"📄 ai_signals.json")
    tg_send_file(AI_ANALYSIS_FILE,"📄 ai_analysis.json")
    tg_send_file(AI_RL_FILE,"📄 ai_rl_log.json")
    tg_send_file(REAL_CLOSED_FILE,"📄 real_closed.json")

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
    for fpath in [PARAM_FILE,STATE_FILE,AI_SIGNALS_FILE,AI_ANALYSIS_FILE,AI_RL_FILE,REAL_CLOSED_FILE,LOG_FILE]:
        tg_send_file(fpath, f"📦 {os.path.basename(fpath)}")

def _cmd_balance():
    """Show current balance and unrealized profit"""
    try:
        current_balance = get_account_balance()
        if not current_balance:
            tg_send("❌ Could not fetch balance")
            return
        
        initial_balance = STATE.get("initial_margin_balance", 0)
        if initial_balance == 0:
            STATE["initial_margin_balance"] = current_balance
            safe_save(STATE_FILE, STATE)
            initial_balance = current_balance
        
        unrealized_pnl = get_unrealized_pnl()
        profit = current_balance - initial_balance
        profit_target = PARAM.get("PROFIT_TARGET_USD", 60.0)
        
        # Get open positions count
        try:
            acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
            open_positions = sum(1 for p in acc if float(p["positionAmt"]) != 0)
        except:
            open_positions = 0
        
        msg = (f"💰 BALANCE STATUS\n"
               f"━━━━━━━━━━━━━━━━\n"
               f"Initial Balance: ${initial_balance:.2f}\n"
               f"Current Balance: ${current_balance:.2f}\n"
               f"Unrealized PnL: ${unrealized_pnl:.2f}\n"
               f"Profit: ${profit:.2f}\n"
               f"Target: ${profit_target:.2f}\n"
               f"Progress: {(profit/profit_target*100):.1f}%\n"
               f"Open Positions: {open_positions}")
        
        tg_send(msg)
    except Exception as e:
        tg_send(f"❌ /balance error: {e}")

def _cmd_settarget(args):
    """Set new profit target"""
    try:
        if not args:
            tg_send("❌ Usage: /settarget <amount>\nExample: /settarget 100")
            return
        
        new_target = float(args[0])
        if new_target <= 0:
            tg_send("❌ Target must be positive")
            return
        
        PARAM["PROFIT_TARGET_USD"] = new_target
        safe_save(PARAM_FILE, PARAM)
        tg_send(f"✅ Profit target set to ${new_target:.2f}")
        log(f"[SETTARGET] Profit target changed to ${new_target:.2f}")
    except Exception as e:
        tg_send(f"❌ /settarget error: {e}")

def _cmd_resettarget():
    """Reset margin balance to current value"""
    try:
        current_balance = get_account_balance()
        if not current_balance:
            tg_send("❌ Could not fetch balance")
            return
        
        old_balance = STATE.get("initial_margin_balance", 0)
        STATE["initial_margin_balance"] = current_balance
        safe_save(STATE_FILE, STATE)
        
        tg_send(f"✅ Margin balance reset\n"
                f"Old: ${old_balance:.2f}\n"
                f"New: ${current_balance:.2f}")
        log(f"[RESETTARGET] Margin balance reset from ${old_balance:.2f} to ${current_balance:.2f}")
    except Exception as e:
        tg_send(f"❌ /resettarget error: {e}")

def _cmd_closeall():
    """Manually close all open positions"""
    try:
        # Get open positions count first
        acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
        open_count = sum(1 for p in acc if float(p["positionAmt"]) != 0)
        
        if open_count == 0:
            tg_send("ℹ️ No open positions to close")
            return
        
        tg_send(f"🔄 Closing {open_count} open positions at market price...")
        
        closed_symbols = close_all_positions_at_market(exit_reason="MANUAL_CLOSE")
        
        if closed_symbols:
            tg_send(f"✅ Closed {len(closed_symbols)} positions: {', '.join(closed_symbols[:10])}")
            if len(closed_symbols) > 10:
                tg_send(f"... and {len(closed_symbols) - 10} more")
            log(f"[CLOSEALL] Manually closed {len(closed_symbols)} positions")
        else:
            tg_send("❌ Failed to close positions")
    except Exception as e:
        tg_send(f"❌ /closeall error: {e}")
        log(f"[CLOSEALL ERR] {e}")

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
        elif cmd=="/balance": _cmd_balance()
        elif cmd=="/settarget": _cmd_settarget(args)
        elif cmd=="/resettarget": _cmd_resettarget()
        elif cmd=="/closeall": _cmd_closeall()
        else:
            tg_send("Komutlar: /status, /report, /set KEY VALUE, /export\n"
                    "/balance - Show balance and profit\n"
                    "/settarget <amount> - Set profit target\n"
                    "/resettarget - Reset margin balance\n"
                    "/closeall - Close all positions")

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
    
    # Fallback to fetching mark price (instead of last traded price)
    if fill is None or fill <= 0:
        fill = futures_get_mark_price(sym)
        if fill is None or fill <= 0:
            log(f"[PRICE ERR] {sym} could not get valid entry price from mark price")
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

def _can_direction(direction, kind=""):
    if not STATE.get("auto_trade_active", True): return False
    if direction=="UP" and STATE.get("long_blocked",False):  return False
    if direction=="DOWN" and STATE.get("short_blocked",False): return False
    
    # Check CEST-specific limits
    if kind == "CEST":
        if direction=="UP" and STATE.get("cest_long_blocked",False):
            log(f"[CEST LIMIT] CEST long positions blocked (max: {PARAM.get('MAX_CEST_BUY', 15)})")
            return False
        if direction=="DOWN" and STATE.get("cest_short_blocked",False):
            log(f"[CEST LIMIT] CEST short positions blocked (max: {PARAM.get('MAX_CEST_SELL', 15)})")
            return False
    
    return True

def execute_real_trade(sig):
    approve_bars = int(PARAM.get("SCALP_APPROVE_BARS",0))
    if approve_bars>0 and (STATE.get("bar_index",0) - sig.get("born_bar",0) < approve_bars):
        return

    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig["power"]
    kind=sig.get("kind","")

    # 🔒 Duplicate / Direction limits
    if not _can_direction(direction, kind): return
    if _duplicate_or_locked(sym,direction): return

    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return

    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened.get("entry")
        if entry_exec is None or entry_exec <= 0:
            # Try fallback to mark price
            entry_exec = futures_get_mark_price(sym)
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
        
        # Track this position for later closure detection
        REAL_POSITIONS_TRACKER[sym] = {
            "symbol": sym,
            "direction": direction,
            "entry_price": entry_exec,
            "kind": kind,
            "tag": sig.get("tag", ""),
            "power": pwr,
            "open_time": now_local_iso(),
            "tp_target": tp_usd_used or tp_pct_used,
            "market_state": sig.get("market_state", "")
        }

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
    tg_send("🚀 EMA ULTRA v15.9.57 aktif (UT/STC devre dışı) — ORB+FVG, London BO, NY Rev, ICT P3, Asian BO, FVG+Breaker")
    log("[START] EMA ULTRA v15.9.57 FULL (UT/STC disabled)")

    symbols=auto_init_symbols()

    while True:
        try:
            # Telegram komutları
            check_telegram_commands()

            # bar index
            STATE["bar_index"]=STATE.get("bar_index",0)+1
            bar_i=STATE["bar_index"]

            # 1) Sinyal tarama
            sigs=run_parallel(symbols,bar_i)

            # 2) Sinyal kayıt + Gerçek trade
            for sig in sigs:
                ai_log_signal(sig)
                update_directional_limits()
                
                # Execute real trade for all strategies (including KIVANC_CONFIRM)
                execute_real_trade(sig)
            
            # 3.1) Check and log real closed trades
            check_and_log_real_closed_trades()
            
            # 3.2) Check profit target (cash out feature)
            check_profit_target()
            
            # 3.3) Send hourly margin progress log
            send_hourly_margin_log()

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
