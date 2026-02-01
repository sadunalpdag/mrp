import os, time, requests, hmac, hashlib, threading, math, json
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.70 — Active Strategies (Asian & London disabled)
#  - PEMA, EARLY, UT/STC, KIVANC CONFIRM tamamen kaldırıldı
#  - Aktif stratejiler (10 strateji - Asian & London disabled):
#       📈 MACD (EMA20/200 + MACD crossover)
#       🟩 FVG (Fair Value Gap Break)
#       📘 EMA PULLBACK (EMA200 + EMA9/30 + swing break + MarketState)
#       🧩 C.E.S.T. (50 MA Double Top/Bottom Strategy - IMPROVED)
#       🔥 ORB + FVG CONFIRM (Opening Range Breakout + FVG - 09:45-12:00 EST)
#       🔄 NY REVERSAL (Liquidity Sweep + Reversal - 09:30-11:00 EST)
#       ⚡ ICT POWER OF 3 (Accumulation-Manipulation-Distribution - 08:30-12:00 EST)
#       🧱 FVG + BREAKER BLOCK (FVG + Breaker Zone - Session Independent)
#       🔄 RE-ENTRY (4H reference + 5m entries - Kill Zone optimized)
#       ⭐ FVG + MSS (Highest Winrate - FVG + Market Structure Shift + OB)
#  - ASIAN SESSION & LONDON BREAKOUT disabled per user request
#  - Re-entry specific limits: 5 buy / 5 sell (adjustable via Telegram)
#  - Strategy enable/disable via Telegram commands
#  - CEST improvements: Multi-timeframe, RSI filter, body quality, session filter
#  - TAKE_PROFIT_MARKET order type uses Algo Order API endpoint (/fapi/v1/algoOrder) to fix API error -4120
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
HOURLY_STATS_FILE = os.path.join(DATA_DIR,"hourly_stats.json")

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
LAST_MAX_PROFIT_UPDATE = 0  # Timestamp of last max profit update
HOURLY_STATS = {}  # Hourly performance statistics
getcontext().prec = 28

# Trading Signal Quality Filter
DEFAULT_MIN_POWER_THRESHOLD = 69.0  # Minimum power score to execute trades (scale: ~50-100)

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

# ===================== VWAP & VOLUME HELPERS =====================

def calculate_vwap(klines):
    """
    Calculate VWAP (Volume Weighted Average Price)
    
    Args:
        klines: List of klines [[time, open, high, low, close, volume, ...], ...]
    
    Returns:
        float: VWAP value or None if insufficient data
    """
    if len(klines) < 2:
        return None
    
    try:
        total_volume = 0.0
        total_pv = 0.0  # price * volume
        
        for k in klines:
            high = float(k[2])
            low = float(k[3])
            close = float(k[4])
            volume = float(k[5])
            
            # Typical price: (high + low + close) / 3
            typical_price = (high + low + close) / 3.0
            pv = typical_price * volume
            
            total_pv += pv
            total_volume += volume
        
        if total_volume > 0:
            return total_pv / total_volume
        return None
    except:
        return None

def detect_volume_spike(klines, spike_threshold=1.5):
    """
    Detect if current volume is significantly higher than average (volume spike)
    
    Args:
        klines: List of klines
        spike_threshold: Multiplier to consider a spike (default 1.5x)
    
    Returns:
        (has_spike, volume_ratio): bool and float ratio
    """
    if len(klines) < 10:
        return False, 1.0
    
    try:
        volumes = [float(k[5]) for k in klines]
        
        # Current volume
        current_vol = volumes[-1]
        
        # Average volume of previous bars (exclude current)
        avg_vol = sum(volumes[-10:-1]) / 9.0
        
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            has_spike = vol_ratio >= spike_threshold
            return has_spike, vol_ratio
        
        return False, 1.0
    except:
        return False, 1.0

def check_ema_trend_alignment(closes, period1=50, period2=200):
    """
    Check if EMA50 and EMA200 are aligned in same direction (trend filter)
    
    Args:
        closes: List of close prices
        period1: First EMA period (default 50)
        period2: Second EMA period (default 200)
    
    Returns:
        ("UP", "DOWN", or None): Aligned trend direction or None if not aligned
    """
    if len(closes) < period2 + 5:
        return None
    
    try:
        ema_short = ema(closes, period1)
        ema_long = ema(closes, period2)
        
        # Both EMAs sloping up = uptrend
        if ema_short[-1] > ema_long[-1] and ema_short[-1] > ema_short[-3]:
            return "UP"
        
        # Both EMAs sloping down = downtrend
        if ema_short[-1] < ema_long[-1] and ema_short[-1] < ema_short[-3]:
            return "DOWN"
        
        # Not aligned or unclear
        return None
    except:
        return None

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
        candle_time = datetime.fromtimestamp(float(k[0]) / 1000, tz=timezone.utc)
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


# ===================== RE-ENTRY STRATEGY HELPERS =====================

def detect_4h_trend(closes_4h):
    """
    Detect 4H trend direction for re-entry strategy.
    
    Returns:
        "UP" for bullish, "DOWN" for bearish, None for unclear trend
    """
    if len(closes_4h) < 20:
        return None
    
    # Use EMA20 and price action
    ema20_4h = ema(closes_4h, 20)
    
    # Check if price is consistently above/below EMA20
    recent_closes = closes_4h[-5:]
    recent_ema = ema20_4h[-5:]
    
    above_count = sum(1 for i in range(len(recent_closes)) if recent_closes[i] > recent_ema[i])
    below_count = sum(1 for i in range(len(recent_closes)) if recent_closes[i] < recent_ema[i])
    
    # Clear uptrend: most recent closes above EMA20
    if above_count >= 4:
        return "UP"
    # Clear downtrend: most recent closes below EMA20
    elif below_count >= 4:
        return "DOWN"
    else:
        return None

def detect_5m_reentry(klines_5m, zone_high, zone_low, trend_direction):
    """
    Detect re-entry pattern on 5m timeframe.
    
    Args:
        klines_5m: 5-minute klines
        zone_high: 4H zone high (from last completed 4H candle)
        zone_low: 4H zone low (from last completed 4H candle)
        trend_direction: "UP" or "DOWN" from 4H analysis
    
    Returns:
        ("UP", entry_price, sl_price) or ("DOWN", entry_price, sl_price) or (None, None, None)
    """
    if len(klines_5m) < 10:
        return None, None, None
    
    closes = [float(k[4]) for k in klines_5m]
    highs = [float(k[2]) for k in klines_5m]
    lows = [float(k[3]) for k in klines_5m]
    opens = [float(k[1]) for k in klines_5m]
    
    # Check last 5 candles for the pattern
    for i in range(len(closes) - 5, len(closes) - 1):
        if i < 0:
            continue
        
        # Pattern: Break out of zone, then re-enter
        prev_close = closes[i - 1] if i > 0 else closes[i]
        curr_close = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_open = opens[i]
        
        # Check for body size (strong candle requirement)
        body_size = abs(curr_close - curr_open)
        candle_range = curr_high - curr_low
        
        # Body should be at least 60% of the range for "strong" candle
        if candle_range > 0 and body_size / candle_range < 0.6:
            continue
        
        # LONG setup (4H bullish, looking at lower zone)
        if trend_direction == "UP":
            # Check if we broke below zone_low with strong body close
            broke_below = curr_close < zone_low and curr_open > zone_low
            
            if broke_below:
                # Now check if next candle re-enters zone
                for j in range(i + 1, min(i + 4, len(closes))):
                    reenter_close = closes[j]
                    reenter_open = opens[j]
                    reenter_body = abs(reenter_close - reenter_open)
                    reenter_range = highs[j] - lows[j]
                    
                    # Strong bullish candle closing back inside zone
                    if (reenter_close > zone_low and reenter_close < zone_high and
                        reenter_close > reenter_open and
                        reenter_range > 0 and reenter_body / reenter_range >= 0.6):
                        
                        # Stop loss below the swing low
                        sl = lows[i] - (zone_high - zone_low) * 0.1  # 10% buffer
                        return "UP", reenter_close, sl
        
        # SHORT setup (4H bearish, looking at upper zone)
        elif trend_direction == "DOWN":
            # Check if we broke above zone_high with strong body close
            broke_above = curr_close > zone_high and curr_open < zone_high
            
            if broke_above:
                # Now check if next candle re-enters zone
                for j in range(i + 1, min(i + 4, len(closes))):
                    reenter_close = closes[j]
                    reenter_open = opens[j]
                    reenter_body = abs(reenter_close - reenter_open)
                    reenter_range = highs[j] - lows[j]
                    
                    # Strong bearish candle closing back inside zone
                    if (reenter_close < zone_high and reenter_close > zone_low and
                        reenter_close < reenter_open and
                        reenter_range > 0 and reenter_body / reenter_range >= 0.6):
                        
                        # Stop loss above the swing high
                        sl = highs[i] + (zone_high - zone_low) * 0.1  # 10% buffer
                        return "DOWN", reenter_close, sl
    
    return None, None, None


# ===================== MSS (Market Structure Shift) HELPERS =====================

def detect_mss(highs, lows, closes, lookback=20):
    """
    Detect Market Structure Shift (MSS) - a clear break of market structure.
    
    MSS is when price breaks the most recent swing high (for bullish) or swing low (for bearish)
    indicating a potential trend change or continuation.
    
    Returns:
        ("UP", break_level) for bullish MSS
        ("DOWN", break_level) for bearish MSS
        (None, None) if no MSS
    """
    if len(closes) < lookback + 5:
        return None, None
    
    # Find recent swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(len(highs) - lookback, len(highs) - 2):
        # Swing high: higher than surrounding bars
        if i > 1 and i < len(highs) - 2:
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1]:
                swing_highs.append((i, highs[i]))
            
            # Swing low: lower than surrounding bars
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1]:
                swing_lows.append((i, lows[i]))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None, None
    
    # Get most recent swing levels
    last_swing_high = swing_highs[-1][1]
    last_swing_low = swing_lows[-1][1]
    
    current_close = closes[-1]
    prev_close = closes[-2]
    
    # Bullish MSS: price breaks above the last swing high
    if prev_close <= last_swing_high and current_close > last_swing_high:
        return "UP", last_swing_high
    
    # Bearish MSS: price breaks below the last swing low
    if prev_close >= last_swing_low and current_close < last_swing_low:
        return "DOWN", last_swing_low
    
    return None, None

def detect_order_block(highs, lows, closes, opens, direction, lookback=20):
    """
    Detect Order Block (OB) - institutional supply/demand zone.
    
    An order block is the last opposing candle before a strong move.
    For bullish OB: last bearish candle before bullish move
    For bearish OB: last bullish candle before bearish move
    
    Returns:
        (found, ob_high, ob_low) - True and OB levels if found
    """
    if len(closes) < lookback + 5:
        return False, None, None
    
    # Look for strong moves (multiple consecutive candles in same direction)
    for i in range(len(closes) - lookback, len(closes) - 5):
        if direction == "UP":
            # Find a sequence of bullish candles
            bullish_count = 0
            for j in range(i, min(i + 5, len(closes))):
                if closes[j] > opens[j]:
                    bullish_count += 1
            
            # If we have 3+ bullish candles, look for last bearish candle before them
            if bullish_count >= 3:
                for k in range(i - 1, max(0, i - 5), -1):
                    if closes[k] < opens[k]:  # Bearish candle
                        # This is the order block
                        ob_high = highs[k]
                        ob_low = lows[k]
                        
                        # Check if current price is near this OB (retest)
                        current_price = closes[-1]
                        if ob_low <= current_price <= ob_high * 1.02:  # Within 2% above OB
                            return True, ob_high, ob_low
        
        elif direction == "DOWN":
            # Find a sequence of bearish candles
            bearish_count = 0
            for j in range(i, min(i + 5, len(closes))):
                if closes[j] < opens[j]:
                    bearish_count += 1
            
            # If we have 3+ bearish candles, look for last bullish candle before them
            if bearish_count >= 3:
                for k in range(i - 1, max(0, i - 5), -1):
                    if closes[k] > opens[k]:  # Bullish candle
                        # This is the order block
                        ob_high = highs[k]
                        ob_low = lows[k]
                        
                        # Check if current price is near this OB (retest)
                        current_price = closes[-1]
                        if ob_high >= current_price >= ob_low * 0.98:  # Within 2% below OB
                            return True, ob_high, ob_low
    
    return False, None, None



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
    sig = {"symbol":sym,"dir":direction,"tier":"MACD","emoji":"📈" if direction=="UP" else "📉",
            "entry":entry,"tp":tp,"sl":sl,"power":pwr,"rsi":r_val,"atr":atr_v,
            "time":now_local_iso(),"born_bar":bar_i,"early":False,
            "kind":"MACD","tag":tag}
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "ema20": e20[-1],
        "ema200": e200[-1],
        "macd_line": macd_line[-1],
        "macd_signal": sig_line[-1],
        "macd_prev": macd_line[-2],
        "signal_prev": sig_line[-2],
        "ema_spread": abs(e20[-1] - e200[-1])
    }
    
    return sig

def build_fvg_break_signal(sym, kl, bar_i):
    if len(kl)<5: return None
    closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
    h1,h2,h3=highs[-3:]; l1,l2,l3=lows[-3:]; c_now=closes[-1]
    up_gap = l2>h1 and c_now>l2
    dn_gap = h2<l1 and c_now< h2
    if up_gap: direction="UP"; tag="🟩 FVG BREAK BUY"; gap_size = l2 - h1
    elif dn_gap: direction="DOWN"; tag="🟥 FVG BREAK SELL"; gap_size = l1 - h2
    else: return None
    atr_v=atr_like(highs,lows,closes)[-1]; r_val=rsi(closes)[-1]
    pwr=58+(atr_v/(closes[-1] or 1))*150
    entry=closes[-1]
    tp=entry*(1.005 if direction=="UP" else 0.995)
    sl=entry*(0.82 if direction=="UP" else 1.18)
    sig = {"symbol":sym,"dir":direction,"tier":"FVG","emoji":"🟩" if direction=="UP" else "🟥",
            "entry":entry,"tp":tp,"sl":sl,"power":pwr,"rsi":r_val,"atr":atr_v,
            "time":now_local_iso(),"born_bar":bar_i,"early":False,
            "kind":"FVG","tag":tag}
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "gap_size": gap_size,
        "gap_size_pct": (gap_size / entry) * 100,
        "h1": h1, "h2": h2, "h3": h3,
        "l1": l1, "l2": l2, "l3": l3,
        "up_gap": up_gap,
        "dn_gap": dn_gap
    }
    
    return sig

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
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "ema9": e9[-1],
        "ema30": e30[-1],
        "ema200": e200[-1],
        "swing_high": swing_h,
        "swing_low": swing_l,
        "uptrend": uptrend,
        "downtrend": downtrend
    }
    
    return sig



def build_cest_signal(sym, kl, bar_i):
    """
    C.E.S.T. – 50 MA Double Top/Bottom Strategy (IMPROVED VERSION)
    
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
    
    ✨ IMPROVEMENTS:
    1. Multi-timeframe confirmation (4H trend check)
    2. Enhanced pattern quality (tighter tolerance)
    3. RSI filter for better entries
    4. Confirmation candle quality check (body size, volume)
    5. ATR-based SL/TP optimization
    6. Time/session filter (London/NY sessions)
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
    
    # Get parameters from PARAM
    tolerance = PARAM.get("CEST_TOLERANCE", 0.015)
    lookback = PARAM.get("CEST_LOOKBACK", 10)
    min_body_ratio = PARAM.get("CEST_MIN_BODY_RATIO", 0.6)
    rr_ratio = PARAM.get("CEST_RR_RATIO", 1.4)
    
    # IMPROVEMENT 1: Multi-timeframe confirmation (4H trend check)
    try:
        kl_4h = futures_get_klines(sym, "4h", 30)
        if len(kl_4h) >= 20:
            closes_4h = [float(k[4]) for k in kl_4h]
            trend_4h = detect_4h_trend(closes_4h)
        else:
            trend_4h = None
    except:
        trend_4h = None
    
    # IMPROVEMENT 3: RSI filter
    r_val = rsi(closes)[-1]
    
    # IMPROVEMENT 7: Time/session filter (London/NY sessions)
    current_hour = get_current_utc_hour()
    in_session = (8 <= current_hour < 12) or (13 <= current_hour < 17)  # London or NY
    
    # If session filter is enabled but we're not in session, skip
    if not in_session and PARAM.get("CEST_SESSION_FILTER", False):
        return None
    
    # ========== LONG SETUP ==========
    # Check if price is above 50 MA
    if c_now > ma50_now:
        # IMPROVEMENT 1: Check 4H trend alignment (should be bullish or neutral for long)
        if trend_4h == "DOWN":
            return None  # Don't trade against 4H trend
        
        # Detect Double Bottom with improved parameters
        found, bottom1_idx, bottom2_idx, touches_ma = detect_double_bottom(
            highs, lows, closes, ma50, lookback=lookback, tolerance=tolerance
        )
        
        if found and touches_ma:
            # IMPROVEMENT 4: Check for confirmation candle quality
            # Current candle should be green (close > open)
            is_green = closes[-1] > opens[-1]
            
            # Calculate body to range ratio for confirmation candle
            body_size = abs(closes[-1] - opens[-1])
            candle_range = highs[-1] - lows[-1]
            body_ratio = body_size / max(candle_range, 1e-12)
            
            # Body should be strong (>60% of range by default)
            if body_ratio < min_body_ratio:
                return None
            
            # Previous candle should have been below or at MA50
            prev_below_ma = closes[-2] <= ma50[-2]
            
            # IMPROVEMENT 3: RSI should not be overbought for long entry
            if r_val > 69:
                return None  # Skip overbought conditions
            
            if is_green and prev_below_ma:
                direction = "UP"
                
                # IMPROVEMENT 5: ATR-based SL optimization
                swing_low = min(lows[bottom1_idx], lows[bottom2_idx])
                sl_est = swing_low - atr_v
                
                # Calculate Take Profit with configurable RR ratio
                risk = c_now - sl_est
                tp_est = c_now + (rr_ratio * risk)
                
                # Calculate power with improved metrics
                pwr = 62 + abs(c_now - ma50_now) * 100 + (50 - r_val) / 2.0 + body_ratio * 10
                
                # IMPROVEMENT 6: Check for FVG/Liquidity Sweep near pattern
                # (simplified check - can be enhanced)
                has_fvg = False
                if len(highs) >= 3:
                    h1, h2, h3 = highs[-3:]
                    l1, l2, l3 = lows[-3:]
                    has_fvg = l2 > h1  # Bullish FVG
                
                sig = {
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
                    "tag": "🧩 C.E.S.T. BUY" + (" +FVG" if has_fvg else ""),
                    "ma50": ma50_now,
                    "swing_low": swing_low,
                    "trend_4h": trend_4h,
                    "body_ratio": body_ratio,
                    "has_fvg": has_fvg
                }
                
                # 📊 Strategy condition parameters for analysis
                sig["conditions"] = {
                    "ma50": ma50_now,
                    "swing_low": swing_low,
                    "body_ratio": body_ratio,
                    "has_fvg": has_fvg,
                    "trend_4h": trend_4h,
                    "bottom1_idx": bottom1_idx,
                    "bottom2_idx": bottom2_idx,
                    "touches_ma": touches_ma,
                    "tolerance": tolerance,
                    "rr_ratio": rr_ratio
                }
                
                return sig
    
    # ========== SHORT SETUP ==========
    # Check if price is below 50 MA
    if c_now < ma50_now:
        # IMPROVEMENT 1: Check 4H trend alignment (should be bearish or neutral for short)
        if trend_4h == "UP":
            return None  # Don't trade against 4H trend
        
        # Detect Double Top with improved parameters
        found, top1_idx, top2_idx, touches_ma = detect_double_top(
            highs, lows, closes, ma50, lookback=lookback, tolerance=tolerance
        )
        
        if found and touches_ma:
            # IMPROVEMENT 4: Check for confirmation candle quality
            # Current candle should be red (close < open)
            is_red = closes[-1] < opens[-1]
            
            # Calculate body to range ratio for confirmation candle
            body_size = abs(closes[-1] - opens[-1])
            candle_range = highs[-1] - lows[-1]
            body_ratio = body_size / max(candle_range, 1e-12)
            
            # Body should be strong (>60% of range by default)
            if body_ratio < min_body_ratio:
                return None
            
            # Previous candle should have been above or at MA50
            prev_above_ma = closes[-2] >= ma50[-2]
            
            # IMPROVEMENT 3: RSI should not be oversold for short entry
            if r_val < 30:
                return None  # Skip oversold conditions
            
            if is_red and prev_above_ma:
                direction = "DOWN"
                
                # IMPROVEMENT 5: ATR-based SL optimization
                swing_high = max(highs[top1_idx], highs[top2_idx])
                sl_est = swing_high + atr_v
                
                # Calculate Take Profit with configurable RR ratio
                risk = sl_est - c_now
                tp_est = c_now - (rr_ratio * risk)
                
                # Calculate power with improved metrics
                pwr = 62 + abs(c_now - ma50_now) * 100 + (r_val - 50) / 2.0 + body_ratio * 10
                
                # IMPROVEMENT 6: Check for FVG/Liquidity Sweep near pattern
                # (simplified check - can be enhanced)
                has_fvg = False
                if len(highs) >= 3:
                    h1, h2, h3 = highs[-3:]
                    l1, l2, l3 = lows[-3:]
                    has_fvg = h2 < l1  # Bearish FVG
                
                sig = {
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
                    "tag": "🧩 C.E.S.T. SELL" + (" +FVG" if has_fvg else ""),
                    "ma50": ma50_now,
                    "swing_high": swing_high,
                    "trend_4h": trend_4h,
                    "body_ratio": body_ratio,
                    "has_fvg": has_fvg
                }
                
                # 📊 Strategy condition parameters for analysis
                sig["conditions"] = {
                    "ma50": ma50_now,
                    "swing_high": swing_high,
                    "body_ratio": body_ratio,
                    "has_fvg": has_fvg,
                    "trend_4h": trend_4h,
                    "top1_idx": top1_idx,
                    "top2_idx": top2_idx,
                    "touches_ma": touches_ma,
                    "tolerance": tolerance,
                    "rr_ratio": rr_ratio
                }
                
                return sig
    
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
    
    sig = {
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
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_high - or_low,
        "or_range_pct": ((or_high - or_low) / c_now) * 100,
        "fvg_gap_size": (l2 - h1) if direction == "UP" else (l1 - h2),
        "broke_high": broke_high,
        "broke_low": broke_low,
        "up_gap": up_gap,
        "dn_gap": dn_gap,
        "risk": risk,
        "rr_ratio": 2.0
    }
    
    return sig


def build_london_breakout_signal(sym, kl, bar_i):
    """
    London Breakout (LO) Strategy - ENHANCED
    
    London session opening range breakout (08:00-10:00 GMT).
    Entry: Breakout of 30-minute London open range
    TP/SL: 2:1 Risk/Reward
    
    NEW FILTERS:
    - Volume Spike: Confirms breakout strength
    - VWAP Filter: Price above VWAP = Long safer, below = Short safer
    - EMA Trend Alignment (5M/15M): EMA50+200 same direction = stronger breakout
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
    
    # NEW: Calculate VWAP for session
    vwap = calculate_vwap(kl[-10:])  # Last 10 bars for VWAP
    
    # NEW: Detect volume spike
    has_volume_spike, volume_ratio = detect_volume_spike(kl)
    
    # NEW: Check EMA trend alignment (try to get 5M data if available, fallback to 1H)
    ema_trend = None
    try:
        # Try 5M data for better trend detection
        kl_5m = futures_get_klines(sym, "5m", 200)
        if kl_5m and len(kl_5m) >= 200:
            closes_5m = [float(k[4]) for k in kl_5m]
            ema_trend = check_ema_trend_alignment(closes_5m, period1=50, period2=200)
    except:
        pass
    
    # Fallback to 1H data if 5M not available
    if ema_trend is None and len(closes) >= 200:
        ema_trend = check_ema_trend_alignment(closes, period1=50, period2=200)
    
    # Check for breakout with EMA20 trend confirmation
    e20 = ema(closes, 20)
    
    # Bullish breakout: price breaks above range + above EMA20
    if c_now > lo_range_high and c_now > e20[-1]:
        direction = "UP"
        tag = "🌍 LONDON BO BUY"
        
        # NEW: VWAP Filter - Long is safer if price above VWAP
        if vwap and c_now < vwap:
            # Price below VWAP but trying to go long - less safe, skip or reduce confidence
            # We'll continue but mark it in conditions
            pass
        
        # NEW: EMA Trend Filter - Check alignment
        if ema_trend == "DOWN":
            # EMAs pointing down but trying to go long - fakeout risk, skip
            return None
        
        sl_est = lo_range_low
        risk = c_now - sl_est
        tp_est = c_now + 2.0 * risk
        
    # Bearish breakout: price breaks below range + below EMA20
    elif c_now < lo_range_low and c_now < e20[-1]:
        direction = "DOWN"
        tag = "🌍 LONDON BO SELL"
        
        # NEW: VWAP Filter - Short is safer if price below VWAP
        if vwap and c_now > vwap:
            # Price above VWAP but trying to go short - less safe, skip or reduce confidence
            pass
        
        # NEW: EMA Trend Filter - Check alignment
        if ema_trend == "UP":
            # EMAs pointing up but trying to go short - fakeout risk, skip
            return None
        
        sl_est = lo_range_high
        risk = sl_est - c_now
        tp_est = c_now - 2.0 * risk
    else:
        return None
    
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    # Power calculation with new filters
    pwr = 63 + (atr_v / c_now) * 140 + (r_val - 50) / 2.0
    
    # Boost power for volume spike
    if has_volume_spike:
        pwr += 5
    
    # Boost power for EMA trend alignment
    if ema_trend == direction:
        pwr += 5
    
    # Boost power for VWAP confirmation
    if vwap:
        if (direction == "UP" and c_now > vwap) or (direction == "DOWN" and c_now < vwap):
            pwr += 3
    
    sig = {
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
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "lo_range_high": lo_range_high,
        "lo_range_low": lo_range_low,
        "lo_range": lo_range_high - lo_range_low,
        "lo_range_pct": ((lo_range_high - lo_range_low) / c_now) * 100,
        "ema20": e20[-1],
        "above_ema20": c_now > e20[-1],
        "risk": risk,
        "rr_ratio": 2.0,
        # NEW condition parameters
        "vwap": vwap,
        "price_vs_vwap": "above" if (vwap and c_now > vwap) else "below" if vwap else "unknown",
        "has_volume_spike": has_volume_spike,
        "volume_ratio": volume_ratio,
        "ema_trend": ema_trend,
        "ema_aligned": (ema_trend == direction) if ema_trend else False
    }
    
    return sig


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
    
    sig = {
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
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "sweep_level": sweep_level,
        "sweep_direction": direction,
        "risk": risk,
        "rr_ratio": 1.5
    }
    
    return sig


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
    Asian Session Strategy (IMPROVED - Range-Bound/Mean Reversion)
    
    Asian session (03:00-08:00 GMT) is characterized by low volatility and ranging behavior.
    Instead of breakout-following (which often fails), this strategy uses:
    
    1. **Liquidity Sweep + Reversal** (Primary - Highest Winrate):
       - Detects when price sweeps Asian range high/low (stop hunt)
       - Waits for strong reversal candle closing back inside range
       - Enters on mean reversion back to range center
    
    2. **Range Scalping** (Secondary):
       - Fades extremes: sells at range top, buys at range bottom
       - Quick scalps to range mean (VWAP/midpoint)
    
    3. **Micro FVG Fill**:
       - Small FVG gaps within Asian range get filled quickly
       - Mean reversion to fair value
    
    Entry: Liquidity sweep detection + 5m strong reversal candle
    TP/SL: 1:1.5 Risk/Reward (Asian session has lower volatility)
    Time: Active 03:00-09:00 GMT (includes early London for sweep detection)
    
    NEW FILTERS (User Requested):
    - Volume Spike: Confirms breakout strength
    - VWAP Filter: Price above VWAP = Long safer, below = Short safer
    - Range Width Filter: 0.3-0.8% = take trade, >1% = skip (too wide)
    - EMA Trend Filter (5M/15M): EMA50+200 same direction = breakout stronger
    """
    # Active during and shortly after Asian session: 03:00-09:00 GMT
    if not is_in_time_window(3, 9):
        return None
    
    if len(kl) < 15:
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    opens = [float(k[1]) for k in kl]
    
    # Get Asian session range (03:00-08:00 GMT) using 1H candles
    asian_high, asian_low = get_session_range(kl, 3, 8)
    
    if asian_high is None:
        # Fallback: use recent tight range
        asian_high = max(highs[-6:-1])
        asian_low = min(lows[-6:-1])
    
    # Calculate range midpoint for mean reversion 
    range_mid = (asian_high + asian_low) / 2.0
    range_size = asian_high - asian_low
    range_size_pct = (range_size / range_mid) * 100
    
    # NEW: Range Width Filter
    # Rule: Asian range 0.3-0.8% = take trade, >1% = skip (too wide)
    if range_size_pct < 0.3:
        # Range too narrow, might not be enough movement
        return None
    if range_size_pct > 1.0:
        # Range too wide, breakout likely weak/fake
        return None
    
    # Original filter kept as backup (less strict than user's request)
    if range_size / range_mid > 0.02:  # More than 2% range
        return None
    
    c_now = closes[-1]
    h_now = highs[-1]
    l_now = lows[-1]
    o_now = opens[-1]
    
    # Get previous candle data for sweep detection
    if len(closes) < 2:
        return None
    c_prev = closes[-2]
    h_prev = highs[-2]
    l_prev = lows[-2]
    
    # ===== STRATEGY 1: LIQUIDITY SWEEP + REVERSAL (Primary) =====
    # Detect liquidity sweep: wick above/below range, but body closes back inside
    
    # Bullish Sweep: Swept below asian_low (stop hunt), then reversed up
    sweep_buffer = range_size * 0.002  # 0.2% buffer for sweep detection
    bullish_sweep = (
        l_now < (asian_low - sweep_buffer) and  # Wick swept below range
        c_now > asian_low and                    # But closed back inside range
        c_now > o_now and                        # Strong bullish candle (green)
        (c_now - o_now) / max(h_now - l_now, 1e-12) > 0.6  # Strong body (>60%)
    )
    
    # Bearish Sweep: Swept above asian_high (stop hunt), then reversed down
    bearish_sweep = (
        h_now > (asian_high + sweep_buffer) and  # Wick swept above range
        c_now < asian_high and                   # But closed back inside range
        c_now < o_now and                        # Strong bearish candle (red)
        (o_now - c_now) / max(h_now - l_now, 1e-12) > 0.6  # Strong body (>60%)
    )
    
    # ===== STRATEGY 2: RANGE FADE (Secondary) =====
    # Fade extremes: price is at range edge, fade back to mean
    
    # At range high, fade down (sell)
    at_range_high = c_now >= asian_high * 0.995  # Within 0.5% of range high
    fade_short = at_range_high and c_now < o_now  # Bearish candle at top
    
    # At range low, fade up (buy)
    at_range_low = c_now <= asian_low * 1.005  # Within 0.5% of range low
    fade_long = at_range_low and c_now > o_now  # Bullish candle at bottom
    
    # ===== STRATEGY 3: MICRO FVG FILL (Bonus) =====
    # Small FVG within range gets filled quickly
    has_micro_fvg_up = False
    has_micro_fvg_down = False
    
    if len(highs) >= 3:
        h1, h2, h3 = highs[-3:]
        l1, l2, l3 = lows[-3:]
        
        # Micro bullish FVG (small gap)
        if l2 > h1 and (l2 - h1) / range_mid < 0.005:  # Gap < 0.5%
            has_micro_fvg_up = c_now > l2  # Price above FVG, likely to fill down
        
        # Micro bearish FVG (small gap)
        if h2 < l1 and (l1 - h2) / range_mid < 0.005:  # Gap < 0.5%
            has_micro_fvg_down = c_now < h2  # Price below FVG, likely to fill up
    
    # NEW: Calculate VWAP for Asian session
    vwap = calculate_vwap(kl[-10:])  # Last 10 bars for VWAP
    
    # NEW: Detect volume spike
    has_volume_spike, volume_ratio = detect_volume_spike(kl)
    
    # NEW: Check EMA trend alignment (try to get 5M or 15M data if available, fallback to 1H)
    ema_trend = None
    try:
        # Try 5M data for better trend detection
        kl_5m = futures_get_klines(sym, "5m", 200)
        if kl_5m and len(kl_5m) >= 200:
            closes_5m = [float(k[4]) for k in kl_5m]
            ema_trend = check_ema_trend_alignment(closes_5m, period1=50, period2=200)
        
        # If 5M didn't give result, try 15M
        if ema_trend is None:
            kl_15m = futures_get_klines(sym, "15m", 200)
            if kl_15m and len(kl_15m) >= 200:
                closes_15m = [float(k[4]) for k in kl_15m]
                ema_trend = check_ema_trend_alignment(closes_15m, period1=50, period2=200)
    except:
        pass
    
    # Fallback to 1H data if 5M/15M not available
    if ema_trend is None and len(closes) >= 200:
        ema_trend = check_ema_trend_alignment(closes, period1=50, period2=200)
    
    # ===== DETERMINE ENTRY =====
    direction = None
    entry_type = None
    
    # Priority 1: Liquidity Sweep (highest winrate)
    if bullish_sweep:
        direction = "UP"
        entry_type = "SWEEP"
        tag = "🌏 ASIA SWEEP BUY"
        
        # NEW: VWAP Filter - Long is safer if price above VWAP
        if vwap and c_now < vwap:
            # Price below VWAP but trying to go long - less safe, reduce confidence
            pass  # Continue but mark in conditions
        
        # NEW: EMA Trend Filter - stronger breakout if aligned
        if ema_trend == "DOWN":
            # EMAs pointing down but trying to go long - fakeout risk
            # For sweeps, we're more lenient as they're reversal plays
            pass
        
    elif bearish_sweep:
        direction = "DOWN"
        entry_type = "SWEEP"
        tag = "🌏 ASIA SWEEP SELL"
        
        # NEW: VWAP Filter - Short is safer if price below VWAP
        if vwap and c_now > vwap:
            # Price above VWAP but trying to go short - less safe
            pass  # Continue but mark in conditions
        
        # NEW: EMA Trend Filter
        if ema_trend == "UP":
            # EMAs pointing up but trying to go short - fakeout risk
            pass
    
    # Priority 2: Range Fade (mean reversion)
    elif fade_long and not bullish_sweep:
        direction = "UP"
        entry_type = "FADE"
        tag = "🌏 ASIA FADE BUY"
        
        # VWAP filter for fades
        if vwap and c_now < vwap:
            pass
        
    elif fade_short and not bearish_sweep:
        direction = "DOWN"
        entry_type = "FADE"
        tag = "🌏 ASIA FADE SELL"
        
        # VWAP filter for fades
        if vwap and c_now > vwap:
            pass
    
    # Priority 3: Micro FVG Fill
    elif has_micro_fvg_down:
        direction = "UP"
        entry_type = "FVG"
        tag = "🌏 ASIA FVG BUY"
    elif has_micro_fvg_up:
        direction = "DOWN"
        entry_type = "FVG"
        tag = "🌏 ASIA FVG SELL"
    
    if direction is None:
        return None
    
    # ===== CALCULATE TP/SL (Asian session = lower volatility, tighter stops) =====
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    if direction == "UP":
        # Target: Range midpoint or slightly above
        if entry_type == "SWEEP":
            tp_est = range_mid + (range_size * 0.25)  # 25% above mid
            sl_est = l_now - (atr_v * 0.5)  # Tight stop below sweep wick
        elif entry_type == "FADE":
            tp_est = range_mid  # Target mean
            sl_est = asian_low - (atr_v * 0.3)  # Very tight stop
        else:  # FVG
            tp_est = range_mid
            sl_est = c_now - (atr_v * 0.5)
        
        # Ensure 1:1.5 minimum RR
        risk = c_now - sl_est
        if (tp_est - c_now) < (risk * 1.5):
            tp_est = c_now + (risk * 1.5)
    
    else:  # DOWN
        # Target: Range midpoint or slightly below
        if entry_type == "SWEEP":
            tp_est = range_mid - (range_size * 0.25)  # 25% below mid
            sl_est = h_now + (atr_v * 0.5)  # Tight stop above sweep wick
        elif entry_type == "FADE":
            tp_est = range_mid  # Target mean
            sl_est = asian_high + (atr_v * 0.3)  # Very tight stop
        else:  # FVG
            tp_est = range_mid
            sl_est = c_now + (atr_v * 0.5)
        
        # Ensure 1:1.5 minimum RR
        risk = sl_est - c_now
        if (c_now - tp_est) < (risk * 1.5):
            tp_est = c_now - (risk * 1.5)
    
    # Power calculation (lower for Asian session - less aggressive)
    pwr = 58 + (atr_v / c_now) * 100 + abs(r_val - 50) / 2.0
    if entry_type == "SWEEP":
        pwr += 5  # Sweep has higher winrate
    
    # NEW: Boost power for volume spike
    if has_volume_spike:
        pwr += 4
    
    # NEW: Boost power for EMA trend alignment
    if ema_trend == direction:
        pwr += 4
    
    # NEW: Boost power for VWAP confirmation
    if vwap:
        if (direction == "UP" and c_now > vwap) or (direction == "DOWN" and c_now < vwap):
            pwr += 3
    
    # NEW: Boost for optimal range width (0.3-0.8%)
    if 0.3 <= range_size_pct <= 0.8:
        pwr += 3
    
    sig = {
        "symbol": sym,
        "dir": direction,
        "tier": "ASIAN_SESSION",
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
        "kind": "ASIAN_RANGE_BREAKOUT",  # Keep same kind for compatibility
        "tag": tag,
        "asian_high": asian_high,
        "asian_low": asian_low,
        "range_mid": range_mid,
        "entry_type": entry_type,  # SWEEP, FADE, or FVG
        "is_sweep": entry_type == "SWEEP",
        "is_fade": entry_type == "FADE",
        "is_fvg": entry_type == "FVG"
    }
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "asian_high": asian_high,
        "asian_low": asian_low,
        "range_mid": range_mid,
        "range_size": range_size,
        "range_size_pct": range_size_pct,
        "entry_type": entry_type,
        # NEW condition parameters
        "vwap": vwap,
        "price_vs_vwap": "above" if (vwap and c_now > vwap) else "below" if vwap else "unknown",
        "has_volume_spike": has_volume_spike,
        "volume_ratio": volume_ratio,
        "ema_trend": ema_trend,
        "ema_aligned": (ema_trend == direction) if ema_trend else False,
        "in_optimal_range": 0.3 <= range_size_pct <= 0.8
    }
    
    return sig


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


def build_reentry_signal(sym, kl, bar_i):
    """
    Re-entry Strategy (4H reference + 5m entries)
    
    Strategy Rules:
    1. Get last completed 4H candle (reference candle)
    2. Mark 4H High and 4H Low as zone
    3. On 5m: price breaks out of zone, then re-enters
    4. Trend filter: 4H bullish → long only, 4H bearish → short only
    5. Fresh zone: use each zone only once
    6. Time filter: prefer London/NY sessions
    7. Entry/TP/SL: 1:2 risk/reward minimum
    """
    # Need both 4H and 5m data
    kl_4h = futures_get_klines(sym, "4h", 50)
    kl_5m = futures_get_klines(sym, "5m", 100)
    
    if len(kl_4h) < 10 or len(kl_5m) < 20:
        return None
    
    # Get last completed 4H candle (not the current one)
    last_4h = kl_4h[-2]  # -2 because -1 might be incomplete
    zone_high = float(last_4h[2])  # High
    zone_low = float(last_4h[3])   # Low
    
    # Detect 4H trend
    closes_4h = [float(k[4]) for k in kl_4h]
    trend_4h = detect_4h_trend(closes_4h)
    
    if trend_4h is None:
        return None  # No clear trend, skip
    
    # Check time window (Kill Zone filter)
    # London: 08:00-12:00 GMT, NY: 13:00-17:00 GMT (EST 08:00-12:00 = GMT 13:00-17:00)
    current_hour = get_current_utc_hour()
    in_kill_zone = (8 <= current_hour < 12) or (13 <= current_hour < 17)
    
    if not in_kill_zone:
        return None  # Outside kill zone
    
    # Detect 5m re-entry pattern
    direction, entry, sl = detect_5m_reentry(kl_5m, zone_high, zone_low, trend_4h)
    
    if direction is None:
        return None
    
    # Calculate TP with 1:2 risk/reward
    closes_5m = [float(k[4]) for k in kl_5m]
    highs_5m = [float(k[2]) for k in kl_5m]
    lows_5m = [float(k[3]) for k in kl_5m]
    
    atr_v = atr_like(highs_5m, lows_5m, closes_5m)[-1]
    r_val = rsi(closes_5m)[-1]
    
    risk = abs(entry - sl)
    if direction == "UP":
        tp = entry + 2.0 * risk
    else:
        tp = entry - 2.0 * risk
    
    pwr = 66 + (atr_v / entry) * 130 + abs(r_val - 50) / 2.0
    
    sig = {
        "symbol": sym,
        "dir": direction,
        "tier": "REENTRY",
        "emoji": "🔄",
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "REENTRY_4H_5M",
        "tag": f"🔄 REENTRY {'BUY' if direction == 'UP' else 'SELL'}",
        "zone_high": zone_high,
        "zone_low": zone_low,
        "trend_4h": trend_4h
    }
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "zone_high": zone_high,
        "zone_low": zone_low,
        "zone_range": zone_high - zone_low,
        "zone_range_pct": ((zone_high - zone_low) / entry) * 100,
        "trend_4h": trend_4h,
        "risk": risk,
        "rr_ratio": 2.0,
        "in_kill_zone": in_kill_zone,
        "current_hour": current_hour
    }
    
    return sig


def build_fvg_mss_signal(sym, kl, bar_i):
    """
    FVG Zone + MSS (Market Structure Shift) Entry Strategy
    
    This is the highest winrate strategy combining:
    - 4H/1H FVG (Fair Value Gap) as main zone
    - 5m BOS/MSS for clear direction change
    - OB (Order Block) retest for confirmation
    
    Only enters when: MSS + FVG + OB retest align
    TP/SL: 1:2 to 1:5 risk/reward
    """
    # Get both higher timeframe (1H) and entry timeframe (5m for MSS)
    kl_1h = futures_get_klines(sym, "1h", 100)
    
    if len(kl_1h) < 20 or len(kl) < 50:  # kl is already 1h from scan_symbol
        return None
    
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    opens = [float(k[1]) for k in kl]
    
    # 1. Detect FVG on 1H timeframe (imbalance zone)
    # FVG: gap between candles (already implemented in build_fvg_break_signal)
    h1, h2, h3 = highs[-3:]
    l1, l2, l3 = lows[-3:]
    c_now = closes[-1]
    
    # Bullish FVG: gap up (l2 > h1)
    has_bullish_fvg = l2 > h1 and c_now > l2
    # Bearish FVG: gap down (h2 < l1)
    has_bearish_fvg = h2 < l1 and c_now < h2
    
    if not (has_bullish_fvg or has_bearish_fvg):
        return None
    
    # 2. Detect MSS (Market Structure Shift)
    mss_direction, mss_level = detect_mss(highs, lows, closes, lookback=20)
    
    if mss_direction is None:
        return None
    
    # 3. Check for Order Block retest
    ob_found, ob_high, ob_low = detect_order_block(highs, lows, closes, opens, mss_direction, lookback=20)
    
    if not ob_found:
        return None
    
    # 4. Combine all three: FVG + MSS + OB
    direction = None
    if has_bullish_fvg and mss_direction == "UP":
        direction = "UP"
    elif has_bearish_fvg and mss_direction == "DOWN":
        direction = "DOWN"
    else:
        return None  # Signals don't align
    
    # Calculate entry, TP, SL
    atr_v = atr_like(highs, lows, closes)[-1]
    r_val = rsi(closes)[-1]
    
    entry = c_now
    
    if direction == "UP":
        # SL below OB low
        sl_est = ob_low - atr_v * 0.5
        risk = entry - sl_est
        # TP: 1:3 risk/reward (high confidence setup)
        tp_est = entry + 3.0 * risk
        tag = "⭐ FVG+MSS BUY"
    else:
        # SL above OB high
        sl_est = ob_high + atr_v * 0.5
        risk = sl_est - entry
        # TP: 1:3 risk/reward
        tp_est = entry - 3.0 * risk
        tag = "⭐ FVG+MSS SELL"
    
    # High power for this high-quality setup
    pwr = 70 + (atr_v / entry) * 150 + abs(r_val - 50) / 2.0
    
    sig = {
        "symbol": sym,
        "dir": direction,
        "tier": "FVG_MSS",
        "emoji": "⭐",
        "entry": entry,
        "tp": tp_est,
        "sl": sl_est,
        "power": pwr,
        "rsi": r_val,
        "atr": atr_v,
        "time": now_local_iso(),
        "born_bar": bar_i,
        "early": False,
        "kind": "FVG_MSS_ENTRY",
        "tag": tag,
        "mss_level": mss_level,
        "ob_high": ob_high,
        "ob_low": ob_low,
        "fvg_zone": "bullish" if has_bullish_fvg else "bearish"
    }
    
    # 📊 Strategy condition parameters for analysis
    sig["conditions"] = {
        "mss_level": mss_level,
        "mss_direction": mss_direction,
        "ob_high": ob_high,
        "ob_low": ob_low,
        "ob_range": ob_high - ob_low,
        "fvg_gap_size": (l2 - h1) if has_bullish_fvg else (l1 - h2),
        "fvg_zone": "bullish" if has_bullish_fvg else "bearish",
        "has_bullish_fvg": has_bullish_fvg,
        "has_bearish_fvg": has_bearish_fvg,
        "risk": risk,
        "rr_ratio": 3.0
    }
    
    return sig


def scan_symbol(sym,bar_i):
    kl=futures_get_klines(sym,"1h",200)
    if len(kl)<60: return []
    res=[]

    # Check strategy enable/disable flags from PARAM
    # EARLY strategy removed per requirement
    # UT/STC strategy disabled per requirement
    # KIVANC_CONFIRM removed per user request
    s_utstc = None  # Disabled - was: build_utstc_signal(sym,kl,bar_i)
    
    s_macd  = build_macd_trend_signal(sym,kl,bar_i) if PARAM.get("ENABLE_MACD", True) else None
    s_fvg   = build_fvg_break_signal(sym,kl,bar_i) if PARAM.get("ENABLE_FVG", True) else None
    s_cest = build_cest_signal(sym,kl,bar_i) if PARAM.get("ENABLE_CEST", True) else None

    # EMA Pullback için 210 bar güvenliği
    kl2 = kl if len(kl)>=210 else futures_get_klines(sym,"1h",210)
    s_pull = build_ema_pullback_signal(sym, kl2, bar_i) if PARAM.get("ENABLE_PULLBACK", True) else None
    
    # Session-based strategies
    s_orb_fvg = build_orb_fvg_confirm_signal(sym, kl, bar_i) if PARAM.get("ENABLE_ORB_FVG", True) else None
    s_london_bo = build_london_breakout_signal(sym, kl, bar_i) if PARAM.get("ENABLE_LONDON_BO", True) else None
    s_ny_rev = build_ny_reversal_signal(sym, kl, bar_i) if PARAM.get("ENABLE_NY_REV", True) else None
    s_ict_p3 = build_ict_power_of_3_signal(sym, kl, bar_i) if PARAM.get("ENABLE_ICT_P3", True) else None
    s_asian_bo = build_asian_range_breakout_signal(sym, kl, bar_i) if PARAM.get("ENABLE_ASIAN_BO", True) else None
    s_fvg_breaker = build_fvg_breaker_block_signal(sym, kl, bar_i) if PARAM.get("ENABLE_FVG_BREAKER", True) else None
    
    # New high-quality strategies
    s_reentry = build_reentry_signal(sym, kl, bar_i) if PARAM.get("ENABLE_REENTRY", True) else None
    s_fvg_mss = build_fvg_mss_signal(sym, kl, bar_i) if PARAM.get("ENABLE_FVG_MSS", True) else None

    for s in (s_utstc, s_macd, s_fvg, s_cest, s_pull,
              s_orb_fvg, s_london_bo, s_ny_rev, s_ict_p3, s_asian_bo, s_fvg_breaker,
              s_reentry, s_fvg_mss):
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
HOURLY_STATS  = safe_load(HOURLY_STATS_FILE,{})

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
                    "market_state": pos_info.get("market_state", ""),
                    "conditions": pos_info.get("conditions", {}),  # 📊 Include strategy condition parameters
                    "max_profit": pos_info.get("max_profit", 0.0)  # Include maximum profit reached
                }
                
                REAL_CLOSED.append(closed_trade)
                safe_save(REAL_CLOSED_FILE, REAL_CLOSED)
                
                # Update hourly performance statistics
                update_hourly_stats_from_closed_trade(closed_trade)
                
                pnl_str = f"{pnl_pct:.2f}" if pnl_pct is not None else "N/A"
                exit_str = f"{exit_price}" if exit_price is not None else "N/A"
                max_profit_str = f"{pos_info.get('max_profit', 0.0):.2f}"
                log(f"[REAL CLOSED] {sym} {direction} Strategy:{pos_info.get('kind', 'UNKNOWN')} "
                    f"PnL:{pnl_str}% Exit:{exit_str} MaxProfit:${max_profit_str}")
        
        # Remove closed positions from tracker
        for sym in closed_symbols:
            REAL_POSITIONS_TRACKER.pop(sym, None)
            
    except Exception as e:
        log(f"[CHECK REAL CLOSED ERR] {e}")

def update_max_profit_tracking():
    """
    Update max profit tracking for all open positions.
    Tracks the maximum profit value reached by each unclosed trade.
    Returns the average of all max profits.
    Throttled to run max once per 30 seconds to avoid rate limiting.
    """
    global REAL_POSITIONS_TRACKER, LAST_MAX_PROFIT_UPDATE
    
    # Throttle: only check every 30 seconds
    now = now_ts_s()
    if now - LAST_MAX_PROFIT_UPDATE < 30:
        # Return last known average from STATE if available
        return STATE.get("avg_max_profit", 0.0)
    
    LAST_MAX_PROFIT_UPDATE = now
    
    try:
        # Get current positions from Binance
        acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
        if not acc:
            # Return cached value on API error
            return STATE.get("avg_max_profit", 0.0)
        
        total_max_profit = 0.0
        position_count = 0
        
        for p in acc:
            amt = float(p["positionAmt"])
            if amt == 0:
                continue
            
            sym = p["symbol"]
            if sym not in REAL_POSITIONS_TRACKER:
                continue
            
            # Get position info
            pos_info = REAL_POSITIONS_TRACKER[sym]
            
            # Get current unrealized profit
            unrealized_pnl = float(p.get("unRealizedProfit", 0))
            
            # Update max profit if current profit is higher
            current_max_profit = pos_info.get("max_profit", 0.0)
            if unrealized_pnl > current_max_profit:
                REAL_POSITIONS_TRACKER[sym]["max_profit"] = unrealized_pnl
                current_max_profit = unrealized_pnl
            
            # Add to total for average calculation
            total_max_profit += current_max_profit
            position_count += 1
        
        # Calculate average max profit
        avg_max_profit = total_max_profit / position_count if position_count > 0 else 0.0
        
        # Log the average if there are open positions
        if position_count > 0:
            log(f"[MAX PROFIT] Open positions: {position_count}, Avg max profit: ${avg_max_profit:.2f}")
        
        return avg_max_profit
        
    except Exception as e:
        log(f"[UPDATE MAX PROFIT ERR] {e}")
        # Return cached value on exception
        return STATE.get("avg_max_profit", 0.0)

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

def parse_iso_to_timestamp(iso_str):
    """Parse ISO 8601 datetime string to Unix timestamp"""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except (ValueError, AttributeError, TypeError):
        return 0

def _signed_request(m,path,payload):
    q="&".join([f"{k}={payload[k]}" for k in payload])
    sig=hmac.new(BINANCE_SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    headers={"X-MBX-APIKEY":BINANCE_KEY}
    url=BINANCE_FAPI+path+"?"+q+"&signature="+sig
    if m=="POST":
        r = requests.post(url,headers=headers,timeout=10)
    elif m=="DELETE":
        r = requests.delete(url,headers=headers,timeout=10)
    else:
        r = requests.get(url,headers=headers,timeout=10)
    if r.status_code!=200:
        raise RuntimeError(f"Binance {r.status_code}: {r.text}")
    return r.json()

def cancel_all_algo_orders(sym):
    """
    Cancel all open algo orders (TAKE_PROFIT_MARKET, STOP_MARKET, etc.) for a symbol.
    This is necessary before closing positions to avoid error -4130.
    Returns True if successful, False otherwise.
    """
    try:
        # Get all open algo orders for the symbol
        payload = {
            "symbol": sym,
            "timestamp": now_ts_ms()
        }
        open_orders = _signed_request("GET", "/fapi/v1/openOrders", payload)
        
        # Cancel each algo order
        cancelled_count = 0
        for order in open_orders:
            order_type = order.get("type")
            # Cancel stop loss and take profit orders
            if order_type in ["TAKE_PROFIT_MARKET", "STOP_MARKET", "TAKE_PROFIT", "STOP_LOSS", "STOP", "TAKE_PROFIT_LIMIT", "STOP_LOSS_LIMIT"]:
                try:
                    cancel_payload = {
                        "symbol": sym,
                        "orderId": order["orderId"],
                        "timestamp": now_ts_ms()
                    }
                    _signed_request("DELETE", "/fapi/v1/order", cancel_payload)
                    cancelled_count += 1
                except Exception as cancel_err:
                    log(f"[CANCEL ORDER WARN] {sym} orderId={order['orderId']} {cancel_err}")
        
        if cancelled_count > 0:
            log(f"[CANCEL ORDERS] {sym} cancelled {cancelled_count} algo order(s)")
        
        return True
    except Exception as e:
        log(f"[CANCEL ORDERS ERR] {sym} {e}")
        return False

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
    "last_hourly_margin_log":0,
    "avg_max_profit":0.0  # Average of max profits from open positions
}
PARAM_DEFAULT={
    "SCALP_TP_PCT":0.006, "SCALP_SL_PCT":0.20, "TRADE_SIZE_USDT":350.0,
    "MAX_BUY":45, "MAX_SELL":45,  # Global limits for all strategies combined
    "MAX_CEST_BUY":15, "MAX_CEST_SELL":15,  # CEST-specific limits (within global limit)
    "ANGLE_MIN":0.00002, "FAST_EMA_PERIOD":3, "SLOW_EMA_PERIOD":7,
    "ATR_SPIKE_RATIO":0.03, "SCALP_APPROVE_BARS":0,
    "PROFIT_TARGET_USD":20.0,
    "MIN_POWER_THRESHOLD":DEFAULT_MIN_POWER_THRESHOLD,  # Minimum power score to execute trades (power scale: ~50-100, higher = stronger signal)
    # Strategy enable/disable flags (all enabled by default)
    "ENABLE_MACD": True,
    "ENABLE_FVG": True,
    "ENABLE_CEST": True,
    "ENABLE_PULLBACK": True,
    "ENABLE_ORB_FVG": True,
    "ENABLE_LONDON_BO": False,  # Disabled per user request
    "ENABLE_NY_REV": True,
    "ENABLE_ICT_P3": True,
    "ENABLE_ASIAN_BO": False,  # Disabled per user request
    "ENABLE_FVG_BREAKER": True,
    "ENABLE_REENTRY": True,
    "ENABLE_FVG_MSS": True,
    # CEST improvements
    "CEST_TOLERANCE": 0.015,  # Double top/bottom price tolerance (1.5%)
    "CEST_LOOKBACK": 10,  # Bars to look back for patterns
    "CEST_MIN_BODY_RATIO": 0.6,  # Minimum body to range ratio for confirmation
    "CEST_RR_RATIO": 1.4,  # Risk/Reward ratio (1:1.4)
    # Hourly performance analysis thresholds
    "HOURLY_MIN_TRADES": 20,  # Minimum trades to consider an hour for blocking
    "HOURLY_MIN_WIN_RATE": 40.0,  # Minimum win rate % (below this = block)
    "HOURLY_MIN_AVG_PNL": -0.5  # Minimum average PnL % (below this = block)
}
PARAM=safe_load(PARAM_FILE,PARAM_DEFAULT)
if not isinstance(PARAM,dict): PARAM=PARAM_DEFAULT
STATE=safe_load(STATE_FILE,STATE_DEFAULT)
for k,v in STATE_DEFAULT.items(): STATE.setdefault(k,v)

# ===================== HOURLY PERFORMANCE TRACKING =====================

def initialize_hourly_stats():
    """Initialize hourly statistics structure"""
    global HOURLY_STATS
    
    if not HOURLY_STATS:
        HOURLY_STATS = {
            "start_date": None,  # Start date for 2-week data collection
            "analysis_active": False,  # Whether to use hourly filtering
            "blocked_hours": [],  # List of hours where trading is blocked
            "hours": {}  # Statistics per hour (0-23)
        }
        
        # Initialize each hour with empty stats
        for hour in range(24):
            HOURLY_STATS["hours"][str(hour)] = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "win_rate": 0.0,
                "strategies": {}  # Per-strategy stats
            }
        
        safe_save(HOURLY_STATS_FILE, HOURLY_STATS)

def update_hourly_stats_from_closed_trade(closed_trade):
    """
    Update hourly statistics when a trade closes.
    
    Args:
        closed_trade: Dict with trade information including open_time, pnl_pct, strategy, etc.
    """
    global HOURLY_STATS
    
    try:
        # Parse open time to get hour (UTC)
        open_time_str = closed_trade.get("open_time")
        if not open_time_str:
            return
        
        # Parse ISO format timestamp
        from datetime import datetime
        open_time = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
        hour = str(open_time.hour)  # 0-23
        
        # Get trade metrics
        pnl_pct = closed_trade.get("pnl_pct", 0.0)
        strategy = closed_trade.get("strategy", "UNKNOWN")
        exit_reason = closed_trade.get("exit_reason", "UNKNOWN")
        
        # Determine if win or loss
        is_win = (exit_reason == "TP" or (pnl_pct and pnl_pct > 0))
        
        # Update hour stats
        hour_stats = HOURLY_STATS["hours"][hour]
        hour_stats["total_trades"] += 1
        
        if is_win:
            hour_stats["wins"] += 1
        else:
            hour_stats["losses"] += 1
        
        if pnl_pct:
            hour_stats["total_pnl_pct"] += pnl_pct
            hour_stats["avg_pnl_pct"] = float(hour_stats["total_pnl_pct"]) / float(hour_stats["total_trades"])
        
        # Calculate win rate
        if hour_stats["total_trades"] > 0:
            hour_stats["win_rate"] = (float(hour_stats["wins"]) / float(hour_stats["total_trades"])) * 100
        
        # Update strategy-specific stats for this hour
        if strategy not in hour_stats["strategies"]:
            hour_stats["strategies"][strategy] = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "win_rate": 0.0
            }
        
        strat_stats = hour_stats["strategies"][strategy]
        strat_stats["total_trades"] += 1
        if is_win:
            strat_stats["wins"] += 1
        else:
            strat_stats["losses"] += 1
        
        if pnl_pct:
            strat_stats["total_pnl_pct"] += pnl_pct
            strat_stats["avg_pnl_pct"] = float(strat_stats["total_pnl_pct"]) / float(strat_stats["total_trades"])
        
        if strat_stats["total_trades"] > 0:
            strat_stats["win_rate"] = (float(strat_stats["wins"]) / float(strat_stats["total_trades"])) * 100
        
        # Save updated stats
        safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
        
        log(f"[HOURLY STATS] Updated for hour {hour}: {hour_stats['total_trades']} trades, {hour_stats['win_rate']:.1f}% WR")
        
    except Exception as e:
        log(f"[HOURLY STATS ERR] {e}")

def check_and_activate_hourly_analysis():
    """
    Check if 2 weeks have passed since start_date.
    If yes, activate hourly analysis and block poor-performing hours.
    """
    global HOURLY_STATS
    
    # If analysis is already active, skip
    if HOURLY_STATS.get("analysis_active", False):
        return
    
    # If no start date, set it now
    if not HOURLY_STATS.get("start_date"):
        HOURLY_STATS["start_date"] = now_local_iso()
        safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
        log(f"[HOURLY ANALYSIS] Data collection started. Will activate analysis after 2 weeks.")
        tg_send(f"📊 Hourly performance tracking started!\n"
                f"Collecting data for 2 weeks before activating hour-based filtering.\n"
                f"Start: {HOURLY_STATS['start_date']}")
        return
    
    # Check if 2 weeks have passed
    try:
        from datetime import datetime, timedelta
        start_date = datetime.fromisoformat(HOURLY_STATS["start_date"].replace('Z', '+00:00'))
        now_date = datetime.now(timezone.utc)
        days_passed = (now_date - start_date).days
        
        # Activate after 14 days (2 weeks)
        if days_passed >= 14:
            HOURLY_STATS["analysis_active"] = True
            
            # Calculate which hours to block based on performance
            update_blocked_hours()
            
            safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
            
            log(f"[HOURLY ANALYSIS] Activated after {days_passed} days of data collection")
            tg_send(f"✅ Hourly analysis ACTIVATED!\n"
                    f"Data collection period: {days_passed} days\n"
                    f"Blocked hours: {HOURLY_STATS.get('blocked_hours', [])}\n"
                    f"System will now avoid trading in poor-performing hours.")
        else:
            # Log remaining days (once per day)
            remaining = 14 - days_passed
            log(f"[HOURLY ANALYSIS] Data collection: {days_passed}/14 days. {remaining} days remaining.")
    
    except Exception as e:
        log(f"[HOURLY ANALYSIS CHECK ERR] {e}")

def update_blocked_hours():
    """
    Analyze hourly performance and update blocked hours list.
    Blocks hours with poor performance based on configurable thresholds.
    """
    global HOURLY_STATS
    
    # Get thresholds from PARAM (configurable via Telegram)
    min_trades_threshold = PARAM.get("HOURLY_MIN_TRADES", 20)  # Minimum trades to consider
    min_win_rate_threshold = PARAM.get("HOURLY_MIN_WIN_RATE", 40.0)  # Minimum win rate %
    min_avg_pnl_threshold = PARAM.get("HOURLY_MIN_AVG_PNL", -0.5)  # Minimum average PnL %
    
    blocked_hours = []
    
    for hour in range(24):
        hour_str = str(hour)
        hour_stats = HOURLY_STATS["hours"][hour_str]
        
        total_trades = hour_stats.get("total_trades", 0)
        win_rate = hour_stats.get("win_rate", 0.0)
        avg_pnl = hour_stats.get("avg_pnl_pct", 0.0)
        
        # Only consider hours with enough data
        if total_trades < min_trades_threshold:
            continue
        
        # Block hour if performance is poor
        if win_rate < min_win_rate_threshold or avg_pnl < min_avg_pnl_threshold:
            blocked_hours.append(hour)
            log(f"[HOURLY BLOCK] Hour {hour}: WR={win_rate:.1f}%, AvgPnL={avg_pnl:.2f}% (blocked)")
    
    HOURLY_STATS["blocked_hours"] = blocked_hours
    safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
    
    return blocked_hours

def is_hour_blocked_for_trading():
    """
    Check if current hour is blocked for trading based on performance analysis.
    
    Returns:
        bool: True if current hour is blocked, False otherwise
    """
    # If analysis is not active, don't block any hours
    if not HOURLY_STATS.get("analysis_active", False):
        return False
    
    # Get current UTC hour
    current_hour = get_current_utc_hour()
    
    # Check if current hour is in blocked list
    blocked = current_hour in HOURLY_STATS.get("blocked_hours", [])
    
    if blocked:
        log(f"[HOUR BLOCKED] Trading blocked for hour {current_hour} due to poor performance")
    
    return blocked

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
            
            # Cancel any existing algo orders (TP/SL) for this symbol first
            # This prevents error -4130: "An open stop or take profit order with GTE and closePosition in the direction is existing."
            cancel_all_algo_orders(sym)
            
            # Determine side and position side
            if amt > 0:  # Long position
                side = "SELL"
                pos_side = "LONG"
            else:  # Short position
                side = "BUY"
                pos_side = "SHORT"
                amt = abs(amt)
            
            # Place take profit order at mark price
            try:
                # Get current mark price
                mark_price = futures_get_mark_price(sym)
                if not mark_price:
                    log(f"[CLOSE ALL SKIP] {sym} - unable to get mark price")
                    continue
                
                # Format stop price according to symbol's tick size
                stop_price_str = format_price_by_tick(sym, mark_price)
                
                # Use TAKE_PROFIT_MARKET with Algo Order API endpoint (required since Dec 2025)
                # This is a market order that triggers when triggerPrice is reached
                payload = {
                    "symbol": sym,
                    "side": side,
                    "type": "TAKE_PROFIT_MARKET",
                    "algoType": "CONDITIONAL",
                    "triggerPrice": stop_price_str,
                    "workingType": "MARK_PRICE",
                    "closePosition": "true",
                    "positionSide": pos_side,
                    "timestamp": now_ts_ms()
                }
                
                try:
                    res = _signed_request("POST", "/fapi/v1/algoOrder", payload)
                    closed_symbols.append(sym)
                    log(f"[CLOSE ALL] {sym} {pos_side} closed with TP at mark price {stop_price_str}")
                except Exception as tp_err:
                    # Check if error is -2021 "Order would immediately trigger" or 
                    # -4130 "An open stop or take profit order with GTE and closePosition in the direction is existing" or
                    # -4509 "Time in Force (TIF) GTE can only be used with open positions"
                    err_str = str(tp_err)
                    if ("-2021" in err_str or "would immediately trigger" in err_str.lower() or
                        "-4130" in err_str or "-4509" in err_str):
                        # Fallback: close position directly with MARKET order
                        log(f"[CLOSE ALL] {sym} TP/Algo order failed, using MARKET order. Error: {tp_err}")
                        market_payload = {
                            "symbol": sym,
                            "side": side,
                            "type": "MARKET",
                            "quantity": f"{amt}",
                            "positionSide": pos_side,
                            "timestamp": now_ts_ms()
                        }
                        res = _signed_request("POST", "/fapi/v1/order", market_payload)
                        closed_symbols.append(sym)
                        log(f"[CLOSE ALL] {sym} {pos_side} closed with MARKET order (direct close)")
                    else:
                        # Re-raise if it's a different error
                        raise
                
                # Note: TRENDLOCK is intentionally NOT removed during cashout
                # This prevents reopening positions for same symbols immediately after cashout
                
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
                    "closed_by_profit_target": (exit_reason == "PROFIT_TARGET"),
                    "conditions": pos_info.get("conditions", {}),  # 📊 Include strategy condition parameters
                    "max_profit": pos_info.get("max_profit", 0.0)  # Include maximum profit reached
                }
                
                REAL_CLOSED.append(closed_trade)
                
                # Update hourly performance statistics
                update_hourly_stats_from_closed_trade(closed_trade)
                
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

def get_recent_closed_position_stats(now):
    """
    Calculate statistics about recently closed positions (last hour).
    Returns a formatted string with stats, or empty string if no recent trades.
    """
    if len(REAL_CLOSED) == 0:
        return ""
    
    # Get recent closed trades (last hour)
    recent_closed = [t for t in REAL_CLOSED if t.get("close_time") and 
                    (now - parse_iso_to_timestamp(t["close_time"]) < 3600)]
    
    if len(recent_closed) == 0:
        return ""
    
    # Calculate stats about position targets
    targets_reached = 0
    valid_trades = 0  # Count trades with valid tp_target and pnl_pct
    avg_target_pct = 0
    avg_actual_pnl = 0
    
    for trade in recent_closed:
        tp_target = trade.get("tp_target")
        pnl_pct = trade.get("pnl_pct")
        
        if tp_target is not None and pnl_pct is not None:
            # Convert tp_target to percentage if needed
            try:
                if isinstance(tp_target, (int, float)):
                    target_value = float(tp_target)
                    if 0 < target_value < 1:
                        target_pct = target_value * 100
                    else:
                        target_pct = target_value
                elif isinstance(tp_target, str):
                    target_pct = float(tp_target)
                else:
                    continue  # Skip invalid types
            except (ValueError, TypeError):
                continue  # Skip if conversion fails
            
            valid_trades += 1
            avg_target_pct += target_pct
            avg_actual_pnl += pnl_pct
            
            # Check if target was reached (95% of target counts as reached)
            if pnl_pct >= target_pct * 0.95:
                targets_reached += 1
    
    if valid_trades == 0:
        return ""
    
    avg_target_pct /= valid_trades
    avg_actual_pnl /= valid_trades
    
    return (f"\n━━━━━━━━━━━━━━━━\n"
           f"📍 Recent Closed (1h):\n"
           f"   Total: {len(recent_closed)}\n"
           f"   Targets Reached: {targets_reached}/{valid_trades}\n"
           f"   Avg Target: {avg_target_pct:.2f}%\n"
           f"   Avg PnL: {avg_actual_pnl:.2f}%")

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
        
        # Get average max profit (updated by main loop's update_max_profit_tracking())
        avg_max_profit = STATE.get("avg_max_profit", 0.0)
        
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
            "estimated_hours_to_target": estimated_hours,
            "avg_max_profit": avg_max_profit
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
                   f"🧩 CEST Short: {cest_short_count}/{PARAM.get('MAX_CEST_SELL', 15)}\n"
                   f"🔝 Avg Max Profit: ${avg_max_profit:.2f}")
            
            # Add position target info when all positions are closed
            if open_positions == 0:
                stats = get_recent_closed_position_stats(now)
                if stats:
                    msg += stats
            
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
                   f"🔝 Avg Max Profit: ${avg_max_profit:.2f}\n"
                   f"🕐 {now_local_iso()}")
            
            # Add position target info when all positions are closed
            if open_positions == 0:
                stats = get_recent_closed_position_stats(now)
                if stats:
                    msg += stats
        
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
         f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'}\n"
         f"long_blocked:{STATE.get('long_blocked')} short_blocked:{STATE.get('short_blocked')}\n"
         f"cest_long_blocked:{STATE.get('cest_long_blocked')} cest_short_blocked:{STATE.get('cest_short_blocked')}")
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
        # KIVANC_CONFIRM removed per user request
        "utstc_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="UTSTC"),
        "macd_signals_total":  sum(1 for x in AI_SIGNALS if x.get("kind")=="MACD"),
        "fvg_signals_total":   sum(1 for x in AI_SIGNALS if x.get("kind")=="FVG"),
        "pullback_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="EMA_PULLBACK"),
        "cest_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="CEST"),
        # Session-based strategies tracking
        "orb_fvg_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="ORB_FVG_CONFIRM"),
        "london_bo_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="LONDON_BREAKOUT"),
        "ny_reversal_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="NY_REVERSAL"),
        "ict_p3_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="ICT_POWER_OF_3"),
        "asian_bo_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="ASIAN_RANGE_BREAKOUT"),
        "fvg_breaker_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="FVG_BREAKER_BLOCK"),
        # New high-quality strategies tracking
        "reentry_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="REENTRY_4H_5M"),
        "fvg_mss_signals_total": sum(1 for x in AI_SIGNALS if x.get("kind")=="FVG_MSS_ENTRY")
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
    
    # Calculate strategy-specific position counts and distances to TP
    try:
        acc = _signed_request("GET", "/fapi/v2/positionRisk", {"timestamp": now_ts_ms()})
        
        # Count positions by strategy
        strategy_counts = {}
        tp_distances = []
        total_unrealized_pnl = 0.0
        
        for p in acc:
            amt = float(p["positionAmt"])
            if amt == 0:
                continue
            
            sym = p["symbol"]
            entry_price = float(p.get("entryPrice", 0))
            mark_price = float(p.get("markPrice", 0))
            unrealized_pnl = float(p.get("unRealizedProfit", 0))
            total_unrealized_pnl += unrealized_pnl
            
            # Get strategy info from tracker
            pos_info = REAL_POSITIONS_TRACKER.get(sym, {})
            strategy = pos_info.get("kind", "UNKNOWN")
            direction = "LONG" if amt > 0 else "SHORT"
            
            # Count by strategy
            strategy_key = f"{strategy}_{direction}"
            strategy_counts[strategy_key] = strategy_counts.get(strategy_key, 0) + 1
            
            # Calculate distance to TP if we have target info
            tp_target = pos_info.get("tp_target")
            if tp_target and entry_price > 0 and mark_price > 0:
                # Calculate current PnL%
                if direction == "LONG":
                    current_pnl_pct = ((mark_price / entry_price) - 1) * 100
                else:
                    current_pnl_pct = ((entry_price - mark_price) / entry_price) * 100
                
                # If tp_target is a percentage
                if isinstance(tp_target, float) and 0 < tp_target < 1:
                    tp_pct = tp_target * 100
                    distance_pct = tp_pct - current_pnl_pct
                else:
                    # tp_target is in USD, estimate percentage
                    trade_size = PARAM.get("TRADE_SIZE_USDT", 250.0)
                    tp_pct = (tp_target / trade_size) * 100
                    distance_pct = tp_pct - current_pnl_pct
                
                tp_distances.append({
                    "symbol": sym,
                    "strategy": strategy,
                    "direction": direction,
                    "current_pnl": current_pnl_pct,
                    "tp_target": tp_pct,
                    "distance": distance_pct
                })
        
        # Build strategy breakdown message
        strategy_msg = "\n━━━━━ STRATEGY BREAKDOWN ━━━━━\n"
        for strat_key, count in sorted(strategy_counts.items()):
            strategy_msg += f"{strat_key}: {count}\n"
        
        # Add top 5 closest to TP
        if tp_distances:
            tp_distances.sort(key=lambda x: x["distance"])
            strategy_msg += "\n━━━ CLOSEST TO TP (Top 5) ━━━\n"
            for i, td in enumerate(tp_distances[:5]):
                strategy_msg += (f"{i+1}. {td['symbol']} ({td['strategy']} {td['direction']})\n"
                               f"   Current: {td['current_pnl']:.2f}%, Target: {td['tp_target']:.2f}%\n"
                               f"   Distance: {td['distance']:.2f}%\n")
        
        # Check hourly analysis status
        hourly_msg = "\n━━━━━ HOURLY ANALYSIS ━━━━━\n"
        if HOURLY_STATS.get("analysis_active", False):
            blocked_hours = HOURLY_STATS.get("blocked_hours", [])
            current_hour = get_current_utc_hour()
            hourly_msg += f"✅ Active (Current hour: {current_hour} UTC)\n"
            if blocked_hours:
                hourly_msg += f"🚫 Blocked hours: {blocked_hours}\n"
            else:
                hourly_msg += "✅ No hours blocked\n"
        else:
            start_date = HOURLY_STATS.get("start_date")
            if start_date:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                now_dt = datetime.now(timezone.utc)
                days_passed = (now_dt - start_dt).days
                remaining = max(0, 14 - days_passed)
                hourly_msg += f"📊 Collecting data: {days_passed}/14 days\n"
                hourly_msg += f"⏳ {remaining} days until activation\n"
            else:
                hourly_msg += "⏳ Not started\n"
        
    except Exception as e:
        strategy_msg = f"\n⚠️ Error getting strategy breakdown: {e}\n"
        hourly_msg = ""
    
    tg_send(
        f"📊 STATUS bar:{STATE.get('bar_index')} "
        f"auto:{'✅' if STATE.get('auto_trade_active',True) else '🟥'}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"General long:{live.get('long_count',0)}/{PARAM.get('MAX_BUY',45)} short:{live.get('short_count',0)}/{PARAM.get('MAX_SELL',45)}\n"
        f"CEST long:{live.get('cest_long_count',0)}/{PARAM.get('MAX_CEST_BUY',15)} short:{live.get('cest_short_count',0)}/{PARAM.get('MAX_CEST_SELL',15)}\n"
        f"Closed trades:{len(REAL_CLOSED)}\n"
        f"Unrealized PnL: ${total_unrealized_pnl:.2f}"
        f"{strategy_msg}"
        f"{hourly_msg}"
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

def _cmd_enable(args):
    """Enable a strategy"""
    try:
        if not args:
            tg_send("❌ Usage: /enable <strategy_name>\n"
                    "Available strategies:\n"
                    "MACD, FVG, CEST, PULLBACK,\n"
                    "ORB_FVG, LONDON_BO, NY_REV, ICT_P3,\n"
                    "ASIAN_BO, FVG_BREAKER, REENTRY, FVG_MSS")
            return
        
        strategy = args[0].upper()
        key = f"ENABLE_{strategy}"
        
        # Check if it's a valid strategy key
        valid_strategies = ["MACD", "FVG", "CEST", "PULLBACK", 
                          "ORB_FVG", "LONDON_BO", "NY_REV", "ICT_P3", 
                          "ASIAN_BO", "FVG_BREAKER", "REENTRY", "FVG_MSS"]
        
        if strategy not in valid_strategies:
            tg_send(f"❌ Unknown strategy: {strategy}\n"
                    f"Available: {', '.join(valid_strategies)}")
            return
        
        PARAM[key] = True
        safe_save(PARAM_FILE, PARAM)
        tg_send(f"✅ Strategy {strategy} enabled")
        log(f"[ENABLE] Strategy {strategy} enabled via Telegram")
    except Exception as e:
        tg_send(f"❌ /enable error: {e}")

def _cmd_disable(args):
    """Disable a strategy"""
    try:
        if not args:
            tg_send("❌ Usage: /disable <strategy_name>\n"
                    "Available strategies:\n"
                    "MACD, FVG, CEST, PULLBACK,\n"
                    "ORB_FVG, LONDON_BO, NY_REV, ICT_P3,\n"
                    "ASIAN_BO, FVG_BREAKER, REENTRY, FVG_MSS")
            return
        
        strategy = args[0].upper()
        key = f"ENABLE_{strategy}"
        
        # Check if it's a valid strategy key
        valid_strategies = ["MACD", "FVG", "CEST", "PULLBACK", 
                          "ORB_FVG", "LONDON_BO", "NY_REV", "ICT_P3", 
                          "ASIAN_BO", "FVG_BREAKER", "REENTRY", "FVG_MSS"]
        
        if strategy not in valid_strategies:
            tg_send(f"❌ Unknown strategy: {strategy}\n"
                    f"Available: {', '.join(valid_strategies)}")
            return
        
        PARAM[key] = False
        safe_save(PARAM_FILE, PARAM)
        tg_send(f"✅ Strategy {strategy} disabled")
        log(f"[DISABLE] Strategy {strategy} disabled via Telegram")
    except Exception as e:
        tg_send(f"❌ /disable error: {e}")

def _cmd_strategies():
    """List all strategies and their status"""
    try:
        strategies = [
            ("MACD", "📈 MACD Trend"),
            ("FVG", "🟩 FVG Break"),
            ("CEST", "🧩 C.E.S.T."),
            ("PULLBACK", "📘 EMA Pullback"),
            ("ORB_FVG", "🔥 ORB+FVG"),
            ("LONDON_BO", "🌍 London Breakout"),
            ("NY_REV", "🔄 NY Reversal"),
            ("ICT_P3", "⚡ ICT Power of 3"),
            ("ASIAN_BO", "🌏 Asian Breakout"),
            ("FVG_BREAKER", "🧱 FVG+Breaker"),
            ("REENTRY", "🔄 Re-entry 4H+5m"),
            ("FVG_MSS", "⭐ FVG+MSS (Highest WR)")
        ]
        
        msg = "📊 STRATEGY STATUS\n━━━━━━━━━━━━━━━━\n"
        for key, name in strategies:
            enabled = PARAM.get(f"ENABLE_{key}", True)
            status = "✅" if enabled else "❌"
            msg += f"{status} {name}\n"
        
        msg += f"\n📌 Global Limits (All Strategies):\n"
        msg += f"Long: {PARAM.get('MAX_BUY', 45)}\n"
        msg += f"Short: {PARAM.get('MAX_SELL', 45)}\n"
        msg += f"\n🧩 CEST Sub-Limits (within global):\n"
        msg += f"Long: {PARAM.get('MAX_CEST_BUY', 15)}\n"
        msg += f"Short: {PARAM.get('MAX_CEST_SELL', 15)}"
        
        tg_send(msg)
    except Exception as e:
        tg_send(f"❌ /strategies error: {e}")

def _cmd_setlimits(args):
    """Set trading limits"""
    try:
        if len(args) < 2:
            tg_send("❌ Usage: /setlimits <type> <value>\n"
                    "Types:\n"
                    "  buy, sell - Global limits (all strategies)\n"
                    "  cest_buy, cest_sell - CEST sub-limits\n"
                    "Example: /setlimits buy 50\n"
                    "Example: /setlimits cest_buy 20")
            return
        
        limit_type = args[0].lower()
        value = int(args[1])
        
        if value < 0 or value > 150:
            tg_send("❌ Value must be between 0 and 150")
            return
        
        if limit_type == "buy":
            PARAM["MAX_BUY"] = value
            safe_save(PARAM_FILE, PARAM)
            tg_send(f"✅ Global MAX_BUY limit set to {value}")
            log(f"[SETLIMITS] MAX_BUY = {value}")
        elif limit_type == "sell":
            PARAM["MAX_SELL"] = value
            safe_save(PARAM_FILE, PARAM)
            tg_send(f"✅ Global MAX_SELL limit set to {value}")
            log(f"[SETLIMITS] MAX_SELL = {value}")
        elif limit_type == "cest_buy":
            PARAM["MAX_CEST_BUY"] = value
            safe_save(PARAM_FILE, PARAM)
            tg_send(f"✅ CEST MAX_BUY limit set to {value} (within global limit)")
            log(f"[SETLIMITS] MAX_CEST_BUY = {value}")
        elif limit_type == "cest_sell":
            PARAM["MAX_CEST_SELL"] = value
            safe_save(PARAM_FILE, PARAM)
            tg_send(f"✅ CEST MAX_SELL limit set to {value} (within global limit)")
            log(f"[SETLIMITS] MAX_CEST_SELL = {value}")
        else:
            tg_send(f"❌ Unknown limit type: {limit_type}\n"
                    "Available: buy, sell, cest_buy, cest_sell")
            return
        
    except ValueError:
        tg_send("❌ Value must be a number")
    except Exception as e:
        tg_send(f"❌ /setlimits error: {e}")

def _cmd_hourlystats():
    """Show hourly performance statistics"""
    try:
        if not HOURLY_STATS.get("hours"):
            tg_send("📊 No hourly statistics available yet")
            return
        
        # Check if analysis is active
        analysis_status = "✅ Active" if HOURLY_STATS.get("analysis_active", False) else "⏳ Collecting data"
        start_date = HOURLY_STATS.get("start_date", "Not set")
        blocked_hours = HOURLY_STATS.get("blocked_hours", [])
        
        msg = f"📊 HOURLY PERFORMANCE STATS\n"
        msg += f"━━━━━━━━━━━━━━━━\n"
        msg += f"Status: {analysis_status}\n"
        msg += f"Start date: {start_date}\n"
        if blocked_hours:
            msg += f"🚫 Blocked hours: {blocked_hours}\n"
        msg += f"\n━━━━━━━━━━━━━━━━\n"
        
        # Show stats for hours with trades
        hours_with_trades = []
        for hour in range(24):
            hour_stats = HOURLY_STATS["hours"][str(hour)]
            if hour_stats.get("total_trades", 0) > 0:
                hours_with_trades.append((hour, hour_stats))
        
        if not hours_with_trades:
            msg += "No trades recorded yet\n"
        else:
            # Sort by total trades descending
            hours_with_trades.sort(key=lambda x: x[1]["total_trades"], reverse=True)
            
            msg += "Top 10 active hours:\n\n"
            for i, (hour, stats) in enumerate(hours_with_trades[:10]):
                total = stats["total_trades"]
                wins = stats["wins"]
                wr = stats["win_rate"]
                avg_pnl = stats["avg_pnl_pct"]
                blocked_mark = "🚫" if hour in blocked_hours else "✅"
                
                msg += f"{blocked_mark} Hour {hour:02d}: {total} trades, WR {wr:.1f}%, Avg {avg_pnl:.2f}%\n"
        
        tg_send(msg)
        
    except Exception as e:
        tg_send(f"❌ /hourlystats error: {e}")
        log(f"[HOURLYSTATS ERR] {e}")

def _cmd_blockhour(args):
    """Manually block or unblock an hour"""
    try:
        if not args:
            tg_send("❌ Usage: /blockhour <hour> [block|unblock]\n"
                   "Example: /blockhour 3 block\n"
                   "Example: /blockhour 14 unblock")
            return
        
        hour = int(args[0])
        if hour < 0 or hour > 23:
            tg_send("❌ Hour must be between 0 and 23")
            return
        
        action = args[1].lower() if len(args) > 1 else "block"
        
        if action not in ["block", "unblock"]:
            tg_send("❌ Action must be 'block' or 'unblock'")
            return
        
        blocked_hours = HOURLY_STATS.get("blocked_hours", [])
        
        if action == "block":
            if hour not in blocked_hours:
                blocked_hours.append(hour)
                blocked_hours.sort()
                HOURLY_STATS["blocked_hours"] = blocked_hours
                safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
                tg_send(f"✅ Hour {hour} blocked for trading")
                log(f"[BLOCKHOUR] Hour {hour} manually blocked")
            else:
                tg_send(f"ℹ️ Hour {hour} is already blocked")
        else:  # unblock
            if hour in blocked_hours:
                blocked_hours.remove(hour)
                HOURLY_STATS["blocked_hours"] = blocked_hours
                safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
                tg_send(f"✅ Hour {hour} unblocked for trading")
                log(f"[BLOCKHOUR] Hour {hour} manually unblocked")
            else:
                tg_send(f"ℹ️ Hour {hour} is not blocked")
        
    except ValueError:
        tg_send("❌ Invalid hour value")
    except Exception as e:
        tg_send(f"❌ /blockhour error: {e}")
        log(f"[BLOCKHOUR ERR] {e}")

def _cmd_resethourlystats():
    """Reset hourly statistics and restart data collection"""
    try:
        global HOURLY_STATS
        
        # Reset to empty state
        HOURLY_STATS = {
            "start_date": None,
            "analysis_active": False,
            "blocked_hours": [],
            "hours": {}
        }
        
        # Initialize each hour
        for hour in range(24):
            HOURLY_STATS["hours"][str(hour)] = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "win_rate": 0.0,
                "strategies": {}
            }
        
        safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
        
        tg_send("✅ Hourly statistics reset!\n"
               "Data collection will restart on next trade.\n"
               "Analysis will activate after 2 weeks.")
        log("[RESETHOURLYSTATS] Hourly statistics reset")
        
    except Exception as e:
        tg_send(f"❌ /resethourlystats error: {e}")
        log(f"[RESETHOURLYSTATS ERR] {e}")

def _cmd_forcehourlyanalysis():
    """Force activate hourly analysis (bypass 2-week wait)"""
    try:
        global HOURLY_STATS
        
        if HOURLY_STATS.get("analysis_active", False):
            tg_send("ℹ️ Hourly analysis is already active")
            return
        
        # Activate analysis
        HOURLY_STATS["analysis_active"] = True
        
        # Calculate blocked hours
        update_blocked_hours()
        
        safe_save(HOURLY_STATS_FILE, HOURLY_STATS)
        
        blocked_hours = HOURLY_STATS.get("blocked_hours", [])
        tg_send(f"✅ Hourly analysis FORCE ACTIVATED!\n"
               f"Blocked hours: {blocked_hours}\n"
               f"System will now avoid trading in poor-performing hours.")
        log("[FORCEHOURLYANALYSIS] Hourly analysis force activated")
        
    except Exception as e:
        tg_send(f"❌ /forcehourlyanalysis error: {e}")
        log(f"[FORCEHOURLYANALYSIS ERR] {e}")

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
        elif cmd=="/enable": _cmd_enable(args)
        elif cmd=="/disable": _cmd_disable(args)
        elif cmd=="/strategies": _cmd_strategies()
        elif cmd=="/setlimits": _cmd_setlimits(args)
        elif cmd=="/hourlystats": _cmd_hourlystats()
        elif cmd=="/blockhour": _cmd_blockhour(args)
        elif cmd=="/resethourlystats": _cmd_resethourlystats()
        elif cmd=="/forcehourlyanalysis": _cmd_forcehourlyanalysis()
        else:
            tg_send("📋 AVAILABLE COMMANDS:\n"
                    "━━━━━━━━━━━━━━━━\n"
                    "/status - Bot status\n"
                    "/balance - Balance and profit\n"
                    "/strategies - List all strategies\n"
                    "/enable <strategy> - Enable strategy\n"
                    "/disable <strategy> - Disable strategy\n"
                    "/setlimits <type> <value> - Set limits\n"
                    "/settarget <amount> - Set profit target\n"
                    "/resettarget - Reset margin balance\n"
                    "/closeall - Close all positions\n"
                    "/hourlystats - View hourly performance\n"
                    "/blockhour <hour> [block|unblock] - Block/unblock hour\n"
                    "/resethourlystats - Reset hourly data\n"
                    "/forcehourlyanalysis - Force activate analysis\n"
                    "/set KEY VALUE - Set parameter\n"
                    "/report - Generate report\n"
                    "/export - Export all data")

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

        def try_once(tp_price_candidate, tp_usd_used=None, tp_pct_used=None):
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
            
            # Use TAKE_PROFIT_MARKET with Algo Order API endpoint (required since Dec 2025)
            # This is a market order that triggers when triggerPrice is reached
            payload={"symbol":sym,"side":side,"type":"TAKE_PROFIT_MARKET","algoType":"CONDITIONAL",
                     "triggerPrice":stop_str,"workingType":"MARK_PRICE","closePosition":"true",
                     "positionSide":pos_side,"timestamp":now_ts_ms()}
            
            try:
                _signed_request("POST","/fapi/v1/algoOrder",payload)
                log(f"[TP OK] {sym} TAKE_PROFIT_MARKET triggerPrice={stop_str}")
                return True,tp_usd_used,tp_pct_used
            except Exception as e:
                log(f"[TP FAIL] {sym} TAKE_PROFIT_MARKET triggerPrice={stop_str} err={e}")
                return False,None,None

        if usd_based:
            # Try TAKE_PROFIT_MARKET with different TP targets
            for tp_usd in [round(x,1) for x in np.arange(tp_low_usd, tp_high_usd+0.001, 0.1)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,tp_usd,tp_pct)
                if ok: return True,u,p
            for tp_usd in [round(x,2) for x in np.arange(tp_low_usd, tp_high_usd+0.0001, 0.01)]:
                tp_price,tp_pct=_tp_price_from_usd(direction,entry_exec,tp_usd,trade_usd)
                ok,u,p=try_once(tp_price,tp_usd,tp_pct)
                if ok: return True,u,p
        else:
            # Percentage-based path (for low-priced assets)
            for tp_pct in [round(x,4) for x in np.arange(0.0050, 0.0100+0.0001, 0.0005)]:
                tp_price = entry_exec*(1+tp_pct if direction=="UP" else 1-tp_pct)
                ok,u,p=try_once(tp_price,None,tp_pct)
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
    
    # Fallback to fetching current market price (last traded price)
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

def _can_direction(direction, kind=""):
    if not STATE.get("auto_trade_active", True): return False
    
    # Check global limits first - these apply to ALL strategies including REENTRY and CEST
    if direction=="UP" and STATE.get("long_blocked",False):
        log(f"[GLOBAL LIMIT] Long positions blocked (max: {PARAM['MAX_BUY']})")
        return False
    if direction=="DOWN" and STATE.get("short_blocked",False):
        log(f"[GLOBAL LIMIT] Short positions blocked (max: {PARAM['MAX_SELL']})")
        return False
    
    # Check CEST-specific limits (in addition to global limits)
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
        return False

    sym=sig["symbol"]; direction=sig["dir"]; pwr=sig["power"]
    kind=sig.get("kind","")

    # 🔒 Check minimum power threshold
    min_power = PARAM.get("MIN_POWER_THRESHOLD", DEFAULT_MIN_POWER_THRESHOLD)
    if pwr < min_power:
        log(f"[LOW POWER] {sym} {kind} power={pwr:.2f} < {min_power:.2f}, skipping trade")
        return False

    # 🔒 Check if current hour is blocked for trading
    if is_hour_blocked_for_trading():
        return False

    # 🔒 Duplicate / Direction limits
    if not _can_direction(direction, kind): return False
    if _duplicate_or_locked(sym,direction): return False

    qty=calc_order_qty(sym,sig["entry"],PARAM["TRADE_SIZE_USDT"])
    if not qty or qty<=0:
        log(f"[QTY ERR] {sym} qty hesaplanamadı."); return False

    try:
        opened=open_market_position(sym,direction,qty)
        entry_exec=opened.get("entry")
        if entry_exec is None or entry_exec <= 0:
            # Try fallback to current price
            entry_exec = futures_get_price(sym)
        if entry_exec is None or entry_exec<=0:
            log(f"[OPEN FAIL] {sym} entry alınamadı."); return False

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
            "market_state": sig.get("market_state", ""),
            "conditions": sig.get("conditions", {}),  # 📊 Store strategy condition parameters
            "max_profit": 0.0  # Track maximum profit reached
        }
        
        return True  # Successfully opened position

    except Exception as e:
        log(f"[OPEN ERR]{sym}{e}")
        return False

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
    # Initialize hourly statistics tracking
    initialize_hourly_stats()
    
    tg_send("🚀 EMA ULTRA v15.9.70 aktif — KIVANC removed, Asian/London disabled\n"
            "📊 10 strategies active (Asian & London disabled) | Re-entry limits: 5 buy/5 sell\n"
            "🎛️ Use /strategies to see all | /enable, /disable to control\n"
            "⏱️ Hourly performance tracking enabled")
    log("[START] EMA ULTRA v15.9.70 - KIVANC removed, Asian & London disabled")

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
            # Update limits once before processing batch
            update_directional_limits()
            
            # Track positions opened in this batch to update local counts
            batch_opened = {"cest_long": 0, "cest_short": 0, "general_long": 0, "general_short": 0}
            
            for sig in sigs:
                ai_log_signal(sig)
                
                # Execute real trade for all strategies
                trade_opened = execute_real_trade(sig)
                
                # If trade was opened, update ALL counts immediately to prevent exceeding limits
                if trade_opened:
                    kind = sig.get("kind", "")
                    direction = sig["dir"]
                    
                    # Update GLOBAL counts first (counts ALL open positions regardless of strategy)
                    total_long_count = len([s for s in REAL_POSITIONS_TRACKER.values() if s.get("direction") == "UP"])
                    total_short_count = len([s for s in REAL_POSITIONS_TRACKER.values() if s.get("direction") == "DOWN"])
                    STATE["long_blocked"] = (total_long_count >= PARAM["MAX_BUY"])
                    STATE["short_blocked"] = (total_short_count >= PARAM["MAX_SELL"])
                    
                    # Update CEST-specific counts (only CEST has sub-limits)
                    if kind == "CEST":
                        if direction == "UP":
                            batch_opened["cest_long"] += 1
                            current_count = len([s for s in REAL_POSITIONS_TRACKER.values() if s.get("kind") == "CEST" and s.get("direction") == "UP"])
                            STATE["cest_long_blocked"] = (current_count >= PARAM.get("MAX_CEST_BUY", 15))
                        else:
                            batch_opened["cest_short"] += 1
                            current_count = len([s for s in REAL_POSITIONS_TRACKER.values() if s.get("kind") == "CEST" and s.get("direction") == "DOWN"])
                            STATE["cest_short_blocked"] = (current_count >= PARAM.get("MAX_CEST_SELL", 15))
                    
                    # Log the current counts for monitoring
                    log(f"[LIMIT CHECK] Total: L={total_long_count}/{PARAM['MAX_BUY']} S={total_short_count}/{PARAM['MAX_SELL']}")
            
            # Update limits once after batch to sync with exchange state
            if any(batch_opened.values()):
                update_directional_limits()
            
            # 3.1) Check and log real closed trades
            check_and_log_real_closed_trades()
            
            # 3.2) Update max profit tracking for open positions
            avg_max_profit = update_max_profit_tracking()
            # Use rounded comparison to avoid unnecessary saves due to floating point precision
            if round(avg_max_profit, 2) != round(STATE.get("avg_max_profit", 0.0), 2):
                STATE["avg_max_profit"] = avg_max_profit
                safe_save(STATE_FILE, STATE)
            
            # 3.3) Check profit target (cash out feature)
            check_profit_target()
            
            # 3.4) Send hourly margin progress log
            send_hourly_margin_log()
            
            # 3.5) Check and activate hourly analysis if 2 weeks have passed
            check_and_activate_hourly_analysis()

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
