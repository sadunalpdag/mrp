# ==============================================================
# 📘 EMA ULTRA v13.1 — Full Binance Futures + Persistent State
#   EMA + RSI + ADX + ATR + Angle  (Scalp + Cross)
#   AutoTrade (default OFF) / Simulate (default ON)
#   State never resets (no daily/24h purge)
# ==============================================================

import os, json, csv, time, math, requests
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE   = os.path.join(DATA_DIR, "state.json")
OPEN_JSON    = os.path.join(DATA_DIR, "open_positions.json")
CLOSED_JSON  = os.path.join(DATA_DIR, "closed_trades.json")
LOG_FILE     = os.path.join(DATA_DIR, "log.txt")
PARAM_FILE   = os.path.join(DATA_DIR, "params.json")

# ================= TELEGRAM =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} {msg}\n")
    except: pass

def tg_send(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] token eksik"); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text}, timeout=15)
    except Exception as e:
        log(f"[TG ERR] {e}")

def tg_file(name, data):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
            data={"chat_id": CHAT_ID}, files={"document": (name, data)}, timeout=30)
    except Exception as e:
        log(f"[TG FILE ERR] {e}")

# ================= STATE HELPERS =================
def safe_load(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return default

def safe_save(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log(f"[SAVE ERR] {e}")

STATE = safe_load(STATE_FILE, {
    "open_positions": [],
    "last_cross_seen": {},
    "last_scalp_seen": {},
    "auto_trade": False,
    "simulate": True,
    "params": {}
})

# ================= DEFAULT PARAMS =================
PARAM = {
    "POWER_NORMAL_MIN": 60.0,
    "POWER_PREMIUM_MIN": 68.0,
    "POWER_ULTRA_MIN": 75.0,
    "ATR_BOOST_PCT": 0.004,
    "ADX_BASE": 25.0,
    "SCALP_TP_PCT": 0.006,
    "SCALP_SL_PCT": 0.10,
    "CROSS_TP_PCT": 0.010,
    "CROSS_SL_PCT": 0.030,
    "SCALP_COOLDOWN_BARS": 3,
    "TRADE_SIZE_USDT": 200.0,
    "MAX_BUY": 10,
    "MAX_SELL": 10
}
PARAM.update(STATE.get("params", {}))
safe_save(PARAM_FILE, PARAM)

# ================= INDICATORS =================
def ema(vals, n):
    k = 2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k+e[-1]*(1-k))
    return e

def rsi(vals, period=14):
    if len(vals)<period+1: return [50]*len(vals)
    deltas=np.diff(vals); gain=np.maximum(deltas,0); loss=-np.minimum(deltas,0)
    avg_gain=np.mean(gain[:period]); avg_loss=np.mean(loss[:period])
    rsis=[50]*period
    for i in range(period,len(deltas)):
        avg_gain=(avg_gain*(period-1)+gain[i])/period
        avg_loss=(avg_loss*(period-1)+loss[i])/period
        rs=avg_gain/avg_loss if avg_loss>0 else 0
        rsis.append(100-100/(1+rs))
    return [50]*(len(vals)-len(rsis))+rsis

def adx_series(high,low,close,period=14):
    if len(high)<period+2: return [0]*len(high)
    tr,plus,minus=[],[],[]
    for i in range(len(high)):
        if i==0:
            tr.append(high[i]-low[i]); plus.append(0); minus.append(0)
        else:
            up=high[i]-high[i-1]; dn=low[i-1]-low[i]
            plus.append(up if up>dn and up>0 else 0)
            minus.append(dn if dn>up and dn>0 else 0)
            pc=close[i-1]
            tr.append(max(high[i]-low[i],abs(high[i]-pc),abs(low[i]-pc)))
    atr=[sum(tr[:period])/period]
    for i in range(period,len(tr)): atr.append((atr[-1]*(period-1)+tr[i])/period)
    di_plus=[0]*period; di_minus=[0]*period
    for i in range(period,len(atr)):
        p=plus[i]/atr[i]*100 if atr[i]!=0 else 0
        m=minus[i]/atr[i]*100 if atr[i]!=0 else 0
        di_plus.append(p); di_minus.append(m)
    dx=[abs(p-m)/(p+m)*100 if (p+m)!=0 else 0 for p,m in zip(di_plus,di_minus)]
    adx=[sum(dx[:period])/period]
    for i in range(period,len(dx)): adx.append((adx[-1]*(period-1)+dx[i])/period)
    return [0]*(len(high)-len(adx))+adx
# ================= ANGLE / POWER HELPERS =================
def slope_change(v_now, v_prev):
    return v_now - v_prev

def slope_angle_deg(slope, atr_now, eps=1e-9):
    # ATR normalize + arctan → derece
    if atr_now <= eps:
        return 0.0
    s_norm = slope / atr_now
    return math.degrees(math.atan(s_norm))

def angle_between_deg(s_prev, s_now, atr_now, eps=1e-9):
    # İki eğim arasındaki açı
    if atr_now <= eps:
        return 0.0
    m1 = s_prev / atr_now
    m2 = s_now  / atr_now
    denom = 1.0 + (m1*m2)
    if abs(denom) < eps:
        return 90.0
    return math.degrees(math.atan(abs(m2 - m1) / denom))

def atr_series(highs, lows, closes, period=14):
    trs=[]
    for i in range(len(highs)):
        if i==0:
            trs.append(highs[i]-lows[i])
        else:
            pc=closes[i-1]
            trs.append(max(highs[i]-lows[i],abs(highs[i]-pc),abs(lows[i]-pc)))
    if len(trs)<period:
        return [0]*len(trs)
    out=[sum(trs[:period])/period]
    for i in range(period,len(trs)):
        out.append((out[-1]*(period-1)+trs[i])/period)
    return [0]*(len(trs)-len(out))+out

def sma(vals, period):
    if len(vals)<period:
        return [sum(vals)/len(vals)]*len(vals)
    out=[]; s=sum(vals[:period])
    out.extend([s/period]*(period-1))
    out.append(s/period)
    for i in range(period,len(vals)):
        s += vals[i]-vals[i-period]
        out.append(s/period)
    return out

def base_power(ema_s_prev, ema_s_now, atr_now, price_now, rsi_now):
    # temel güç
    slope_comp = abs(ema_s_now - ema_s_prev)/(atr_now*0.6) if atr_now>0 else 0.0
    rsi_comp   = (rsi_now-50)/50.0
    atr_comp   = (atr_now/price_now)*100.0 if price_now>0 else 0.0
    raw = 55 + slope_comp*20 + rsi_comp*15 + atr_comp*2
    return max(0.0, min(100.0, raw))

def add_adx_vol_power(pwr, adx_now, vol_now, vol_ma):
    add=0.0
    if adx_now > PARAM["ADX_BASE"]:
        add += min(10.0, (adx_now-PARAM["ADX_BASE"])/15.0*10.0)
    if vol_ma>0:
        mult = vol_now/vol_ma
    else:
        mult = 1.0
    if mult >= 1.0:
        # daha yüksek hacim → momentum boost
        add += min(7.0, (mult-1.0)*7.0)
    return max(0.0, min(100.0, pwr+add)), mult

def add_angle_power(pwr, ang_now, ang_change):
    # yüksek açı => trend net → bonus
    bonus = min(6.0, max(0.0, (abs(ang_now)/75.0)*6.0))
    # aşırı ani dönüş => ceza
    penalty = min(5.0, (ang_change/45.0)*5.0)
    return max(0.0, min(100.0, pwr + bonus - penalty))

def rsi_div(last_close, prev_close, rsi_now, rsi_prev):
    if last_close < prev_close and rsi_now > rsi_prev:
        return "Bullish"
    if last_close > prev_close and rsi_now < rsi_prev:
        return "Bearish"
    return "Neutral"

def tier_from_power(p):
    if p >= PARAM["POWER_ULTRA_MIN"]:
        return "ULTRA","🟩"
    if p >= PARAM["POWER_PREMIUM_MIN"]:
        return "PREMIUM","🟦"
    if p >= PARAM["POWER_NORMAL_MIN"]:
        return "NORMAL","🟨"
    return "NONE",""

# ================= TIME HELPERS =================
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)
def now_ist_iso(): return now_ist_dt().isoformat()
def now_hhmm():    return now_ist_dt().strftime("%H:%M")
def today_date():  return now_ist_dt().strftime("%Y-%m-%d")

# ================= BINANCE PUBLIC =================
FAPI_BASE = "https://fapi.binance.com"
SESSION   = requests.Session()
SESSION.headers.update({"User-Agent":"EMA-ULTRA-v13.1","Accept":"application/json"})

def get_klines(symbol, interval, limit):
    try:
        r = SESSION.get(
            FAPI_BASE+"/fapi/v1/klines",
            params={"symbol":symbol,"interval":interval,"limit":limit},
            timeout=15
        )
        data=r.json()
        # ileriye açık bar varsa at
        now_ms=int(datetime.now(timezone.utc).timestamp()*1000)
        if data and int(data[-1][6])>now_ms:
            data=data[:-1]
        return data
    except Exception as e:
        log(f"[KLN_ERR]{symbol} {interval} {e}")
        return []

def get_last_price(symbol):
    try:
        r=SESSION.get(
            FAPI_BASE+"/fapi/v1/ticker/price",
            params={"symbol":symbol},
            timeout=10
        ).json()
        return float(r["price"])
    except Exception as e:
        log(f"[PRICE_ERR]{symbol} {e}")
        return None

def get_futures_symbols():
    try:
        r=SESSION.get(FAPI_BASE+"/fapi/v1/exchangeInfo",timeout=15).json()
        return [
            s["symbol"] for s in r["symbols"]
            if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"
        ]
    except Exception as e:
        log(f"[SYMBOLS_ERR]{e}")
        return []

# ================= SIGNAL ENGINES (v12.6 style) =================
# CROSS: EMA7/EMA25 kesişimi + confirm
def build_cross_signal(sym, kl1h, kl4h, kld1):
    closes = [float(x[4]) for x in kl1h]
    ema7   = ema(closes,7)
    ema25  = ema(closes,25)
    if len(ema7)<6 or len(ema25)<6:
        return None

    # 1-bar confirm logic
    prev_diff    = ema7[-3]-ema25[-3]
    cross_diff   = ema7[-2]-ema25[-2]
    confirm_diff = ema7[-1]-ema25[-1]

    direction=None
    if prev_diff < 0 and cross_diff > 0 and confirm_diff > 0:
        direction="UP"
    elif prev_diff > 0 and cross_diff < 0 and confirm_diff < 0:
        direction="DOWN"
    if direction is None: return None

    highs=[float(x[2]) for x in kl1h]
    lows =[float(x[3]) for x in kl1h]
    vols =[float(x[5]) for x in kl1h]
    atr_now = atr_series(highs,lows,closes,14)[-1]
    rsi_all = rsi(closes,14)
    rsi_now = rsi_all[-1]
    rsi_prev= rsi_all[-2]

    # ema slope (3-bar window tarzı)
    s_now  = ema7[-1]-ema7[-4]
    s_prev = ema7[-2]-ema7[-5]

    ang_now    = slope_angle_deg(s_now, atr_now)
    ang_change = angle_between_deg(s_prev, s_now, atr_now)

    adx_all = adx_series(highs,lows,closes,14)
    adx_now = adx_all[-1] if adx_all else 0

    vol_ma = sma(vols,20)[-1] if len(vols)>=20 else (sum(vols)/len(vols) if vols else 0)

    base   = base_power(s_prev, s_now, atr_now, closes[-1], rsi_now)
    pwr1,_ = add_adx_vol_power(base, adx_now, vols[-1] if vols else 0, vol_ma)
    pwr2   = add_angle_power(pwr1, ang_now, ang_change)
    divergence = rsi_div(closes[-1], closes[-2], rsi_now, rsi_prev)

    # Trend align check (4h ve 1d)
    closes4  = [float(x[4]) for x in kl4h]
    closes1d = [float(x[4]) for x in kld1]
    ema7_4   = ema(closes4,7); ema25_4=ema(closes4,25)
    ema7_1d  = ema(closes1d,7); ema25_1d=ema(closes1d,25)
    trend4h  = "UP" if ema7_4[-1]>ema25_4[-1] else "DOWN"
    trend1d  = "UP" if ema7_1d[-1]>ema25_1d[-1] else "DOWN"
    aligned  = (direction==trend4h==trend1d)

    entry = closes[-1]
    tp = entry*(1+PARAM["CROSS_TP_PCT"] if direction=="UP" else 1-PARAM["CROSS_TP_PCT"])
    sl = entry*(1-PARAM["CROSS_SL_PCT"] if direction=="UP" else 1+PARAM["CROSS_SL_PCT"])

    # which tier?
    tier, color = tier_from_power(pwr2)
    if tier=="NONE":
        return None

    sig = {
        "symbol": sym,
        "type": "CROSS",
        "dir": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "power": pwr2,
        "rsi": rsi_now,
        "atr_pct": (atr_now/entry if entry>0 else 0),
        "adx": adx_now,
        "vol": vols[-1] if vols else 0,
        "trend4h": trend4h,
        "trend1d": trend1d,
        "aligned": aligned,
        "ang_now": ang_now,
        "ang_change": ang_change,
        "div": divergence,
        "tier": tier,
        "color": color,
        "ts": now_ist_iso()
    }
    return sig

# SCALP: EMA7 eğim dönüşü son 3 bar içinde + 4h uyum
def build_scalp_signal(sym, kl1h, kl4h, kld1, cooldown_tracker, bar_index, cooldown_bars):
    closes = [float(x[4]) for x in kl1h]
    ema7_1 = ema(closes,7)
    if len(ema7_1)<6:
        return None

    # slope check son 3 bar (değiştirdik: artık son 3 bar'a bakıyoruz)
    s_curr = ema7_1[-1] - ema7_1[-4]
    s_prev = ema7_1[-2] - ema7_1[-5]
    if s_prev<0 and s_curr>0:
        direction="UP"
    elif s_prev>0 and s_curr<0:
        direction="DOWN"
    else:
        return None

    # 4h trend aynı yönde mi
    closes4=[float(x[4]) for x in kl4h]
    ema7_4=ema(closes4,7); ema25_4=ema(closes4,25)
    trend4h="UP" if ema7_4[-1]>ema25_4[-1] else "DOWN"
    if trend4h!=direction:
        return None

    # 1d sadece bilgi amaçlı
    closes1d=[float(x[4]) for x in kld1]
    ema7_1d=ema(closes1d,7); ema25_1d=ema(closes1d,25)
    trend1d="UP" if ema7_1d[-1]>ema25_1d[-1] else "DOWN"

    highs=[float(x[2]) for x in kl1h]
    lows =[float(x[3]) for x in kl1h]
    vols =[float(x[5]) for x in kl1h]
    atr_now = atr_series(highs,lows,closes,14)[-1]
    rsi_now = rsi(closes,14)[-1]
    adx_now = adx_series(highs,lows,closes,14)[-1]
    vol_ma  = sma(vols,20)[-1] if len(vols)>=20 else (sum(vols)/len(vols) if vols else 0)

    ang_now    = slope_angle_deg(s_curr, atr_now)
    ang_change = angle_between_deg(s_prev, s_curr, atr_now)

    basep      = base_power(s_prev, s_curr, atr_now, closes[-1], rsi_now)
    pwr1,vmult = add_adx_vol_power(basep, adx_now, vols[-1] if vols else 0, vol_ma)
    pwr2       = add_angle_power(pwr1, ang_now, ang_change)

    if pwr2 < PARAM["POWER_PREMIUM_MIN"]:
        return None

    entry = closes[-1]
    tp = entry*(1+PARAM["SCALP_TP_PCT"] if direction=="UP" else 1-PARAM["SCALP_TP_PCT"])
    sl = entry*(1-PARAM["SCALP_SL_PCT"] if direction=="UP" else 1+PARAM["SCALP_SL_PCT"])

    scalp_key = f"{sym}_{direction}"
    last_idx  = cooldown_tracker.get(scalp_key)
    if last_idx is not None and (bar_index - last_idx) <= cooldown_bars:
        return None

    sig = {
        "symbol": sym,
        "type": "SCALP",
        "dir": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "power": pwr2,
        "rsi": rsi_now,
        "atr_pct": (atr_now/entry if entry>0 else 0),
        "adx": adx_now,
        "vol_mult": vmult,
        "trend4h": trend4h,
        "trend1d": trend1d,
        "aligned": (direction==trend4h==trend1d),
        "ang_now": ang_now,
        "ang_change": ang_change,
        "tier": "ULTRA" if pwr2>=PARAM["POWER_ULTRA_MIN"] else "PREMIUM",
        "color": "🟩" if pwr2>=PARAM["POWER_ULTRA_MIN"] else "🟦",
        "ts": now_ist_iso(),
    }
    return sig

# ================= TRADE VALIDATION =================
def validate_signal(sig):
    # aynı kontrolleri order öncesi yapıyoruz
    try:
        entry=float(sig["entry"]); tp=float(sig["tp"]); sl=float(sig["sl"])
        if not (entry>0 and tp>0 and sl>0): return False
        if sig["dir"]=="UP" and not(tp>entry>sl): return False
        if sig["dir"]=="DOWN" and not(tp<entry<sl): return False
        if sig["atr_pct"]<0.0001 or sig["atr_pct"]>0.05: return False
        if abs(tp-entry)/entry<0.001: return False
        if abs(entry-sl)/entry<0.001: return False
        ang_now = abs(float(sig["ang_now"]))
        ang_ch  = abs(float(sig["ang_change"]))
        if ang_now>85 and ang_ch>60: return False
        return True
    except:
        return False

# ================= AUTOTRADE HANDLER (SIM / REAL PLACEHOLDER) =================
def auto_trade_handle(sig, STATE):
    # default'ta kapalı
    auto_on   = PARAM.get("AUTO_TRADE_ON", False)
    simulate  = PARAM.get("SIMULATE", True)

    # backward compat: eğer param dosyasında yoksa STATE içinden çekelim
    if "AUTO_TRADE_ON" not in PARAM:
        auto_on = STATE.get("auto_trade", False)
    if "SIMULATE" not in PARAM:
        simulate = STATE.get("simulate", True)

    if not auto_on:
        return  # işlem açma yok

    # trade limiti kontrol
    if sig["dir"]=="UP":
        active_buys = sum(1 for p in STATE["open_positions"] if p["dir"]=="UP")
        if active_buys >= PARAM["MAX_BUY"]:
            tg_send(f"⚠️ BUY limit dolu {sig['symbol']}")
            return
    else:
        active_sells = sum(1 for p in STATE["open_positions"] if p["dir"]=="DOWN")
        if active_sells >= PARAM["MAX_SELL"]:
            tg_send(f"⚠️ SELL limit dolu {sig['symbol']}")
            return

    # simulate ise sadece kayda geç
    if simulate:
        pos = {
            "symbol": sig["symbol"],
            "dir": sig["dir"],
            "type": sig["type"],
            "entry": sig["entry"],
            "tp": sig["tp"],
            "sl": sig["sl"],
            "time_open": now_ist_iso(),
            "power": sig["power"],
            "rsi": sig["rsi"],
            "ang_now": sig.get("ang_now",0.0),
            "ang_change": sig.get("ang_change",0.0),
            "atr_pct": sig.get("atr_pct",0.0),
            "bars_open": STATE.get("bar_index",0)
        }
        STATE["open_positions"].append(pos)
        safe_save(OPEN_JSON, STATE["open_positions"])
        tg_send(
            f"💹 SIM {sig['color']} {sig['type']} {sig['symbol']} {sig['dir']}\n"
            f"Entry:{sig['entry']:.4f} TP:{sig['tp']:.4f} SL:{sig['sl']:.4f}\n"
            f"RSI:{sig['rsi']:.1f} Pow:{sig['power']:.1f}\n"
            f"Açı:{sig['ang_now']:+.1f}° Δ:{sig['ang_change']:.1f}°"
        )
        return

    # gerçek emir (future enhancement)
    # burada gerçek Binance emir logic'i girecek
    tg_send(f"🚨 REAL TRADE PLACEHOLDER {sig['symbol']} {sig['dir']}")
    return
# ================= PNL CHECK / POSITION CLOSE =================
def check_positions_close(STATE):
    """TP veya SL vuruldu mu? vurulduysa kapat ve CLOSED_JSON'a yaz."""
    still=[]
    for p in STATE["open_positions"]:
        lp = get_last_price(p["symbol"])
        if lp is None:
            still.append(p)
            continue

        hit_tp = (lp >= p["tp"]) if p["dir"]=="UP" else (lp <= p["tp"])
        hit_sl = (lp <= p["sl"]) if p["dir"]=="UP" else (lp >= p["sl"])
        if not (hit_tp or hit_sl):
            still.append(p); continue

        res="TP" if hit_tp else "SL"
        pnl_pct = (
            (lp - p["entry"])/p["entry"]*100 if p["dir"]=="UP"
            else (p["entry"] - lp)/p["entry"]*100
        )
        bars_open = STATE["bar_index"] - p["bars_open"]

        closed_row = {
            "symbol": p["symbol"],
            "type": p["type"],
            "dir": p["dir"],
            "entry": p["entry"],
            "exit": lp,
            "result": res,
            "pnl": round(pnl_pct,2),
            "bars": bars_open,
            "power": p.get("power",0.0),
            "rsi": p.get("rsi",50.0),
            "ang_now": p.get("ang_now",0.0),
            "ang_change": p.get("ang_change",0.0),
            "atr_pct": p.get("atr_pct",0.0),
            "time_open": p["time_open"],
            "time_close": now_ist_iso()
        }

        # save closed
        past = safe_load(CLOSED_JSON, [])
        past.append(closed_row)
        safe_save(CLOSED_JSON, past)

        tg_send(
            f"📘 {res} | {p['symbol']} {p['type']} {p['dir']}\n"
            f"Entry:{p['entry']:.4f} Exit:{lp:.4f}\n"
            f"PnL:{pnl_pct:.2f}% Bars:{bars_open}"
        )

    STATE["open_positions"]=still
    safe_save(OPEN_JSON, still)

# ================= TELEGRAM COMMANDS =================
LAST_UPDATE_ID=None
def poll_telegram(STATE):
    global LAST_UPDATE_ID, PARAM
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        url=f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params={"timeout":1,"limit":20}
        if LAST_UPDATE_ID:
            params["offset"]=LAST_UPDATE_ID+1
        r=requests.get(url,params=params,timeout=5).json()

        for upd in r.get("result",[]):
            LAST_UPDATE_ID=max(LAST_UPDATE_ID or 0, upd["update_id"])
            msg=upd.get("message")
            if not msg: continue
            if str(msg["chat"]["id"])!=str(CHAT_ID): continue
            txt = msg.get("text","").strip()
            parts = txt.split()
            cmd = parts[0].lower() if parts else ""

            if cmd=="/params":
                tg_send("\n".join([f"{k}: {v}" for k,v in PARAM.items()]))

            elif cmd=="/set" and len(parts)>=3:
                key = parts[1]
                val = " ".join(parts[2:])
                if key not in PARAM:
                    tg_send("⚠️ Bilinmeyen param"); continue
                base = PARAM[key]
                try:
                    if isinstance(base,bool):
                        PARAM[key] = val.lower() in ("on","1","true","yes")
                    elif isinstance(base,int):
                        PARAM[key] = int(val)
                    elif isinstance(base,float):
                        PARAM[key] = float(val)
                    else:
                        PARAM[key] = val
                    # kalıcı kaydet
                    safe_save(PARAM_FILE, PARAM)
                    STATE["params"]=PARAM
                    safe_save(STATE_FILE, STATE)
                    tg_send(f"✅ {key}={PARAM[key]}")
                except:
                    tg_send("⚠️ Geçersiz değer")

            elif cmd=="/autotrade" and len(parts)>=2:
                val = parts[1].lower() in ("on","1","true","yes")
                PARAM["AUTO_TRADE_ON"]=val
                STATE["auto_trade"]=val
                safe_save(PARAM_FILE, PARAM)
                safe_save(STATE_FILE, STATE)
                tg_send(f"AUTO_TRADE_ON={val}")

            elif cmd=="/simulate" and len(parts)>=2:
                val = parts[1].lower() in ("on","1","true","yes")
                PARAM["SIMULATE"]=val
                STATE["simulate"]=val
                safe_save(PARAM_FILE, PARAM)
                safe_save(STATE_FILE, STATE)
                tg_send(f"SIMULATE={val}")

            elif cmd=="/status":
                tg_send(
                    f"OpenPos:{len(STATE['open_positions'])}\n"
                    f"AUTO:{STATE.get('auto_trade',False)} SIM:{STATE.get('simulate',True)}\n"
                    f"Time:{now_hhmm()}  Day:{today_date()}"
                )

            elif cmd=="/report":
                tg_send(build_report())

    except Exception as e:
        log(f"[TG POLL ERR]{e}")

# ================= REPORTING =================
def build_report(days_back=14):
    # kapatılmış işlemler analizi
    closed = safe_load(CLOSED_JSON, [])
    if not closed:
        return "⚠️ closed_trades boş."
    cutoff = now_ist_dt() - timedelta(days=days_back)

    filt=[]
    for row in closed:
        t=row.get("time_close")
        if not t: continue
        try:
            dt=datetime.fromisoformat(t)
        except:
            continue
        if dt>=cutoff:
            filt.append(row)
    if not filt:
        return f"⚠️ Son {days_back} günde işlem yok."

    total=len(filt)
    wins=sum(1 for r in filt if r.get("result")=="TP")
    winrate= (wins/total*100.0) if total else 0.0
    pnl_list=[float(x.get("pnl",0)) for x in filt]
    bars_list=[float(x.get("bars",0)) for x in filt]
    avg_pnl = np.mean(pnl_list) if pnl_list else 0.0
    avg_bars= np.mean(bars_list) if bars_list else 0.0

    hours={}
    for r in filt:
        tc=r.get("time_close")
        if not tc: continue
        try:
            hh=datetime.fromisoformat(tc).hour
            hours[hh]=hours.get(hh,0)+1
        except: pass
    hot_hours=", ".join([f"{h}:00({c})" for h,c in sorted(hours.items(), key=lambda x:x[1], reverse=True)[:5]])

    # hangi param en çok hızlı kapatıyor?
    bars_np=np.array(bars_list) if bars_list else np.array([0,0])
    keys=["power","rsi","ang_now","ang_change","atr_pct"]
    corrs={}
    for k in keys:
        try:
            arr=np.array([float(r.get(k,0)) for r in filt])
            if np.std(arr)>0 and np.std(bars_np)>0:
                corrs[k]=float(np.corrcoef(arr,bars_np)[0,1])
            else:
                corrs[k]=0.0
        except:
            corrs[k]=0.0
    if corrs:
        best_param, best_val = max(corrs.items(), key=lambda x:abs(x[1]))
    else:
        best_param, best_val = ("-",0.0)

    rep = (
        f"📊 Günlük Rapor ({days_back}g)\n"
        f"Toplam:{total} Win%:{winrate:.1f}\n"
        f"AvgPnL:{avg_pnl:.2f}% AvgBars:{avg_bars:.1f}\n"
        f"Sıcak Saatler:{hot_hours}\n"
        f"Hızlı Kapanış En Etkili:{best_param} corr={best_val:.2f}\n"
        f"AUTO:{STATE.get('auto_trade',False)} SIM:{STATE.get('simulate',True)}"
    )
    return rep

def send_daily_package():
    # günlük rapor + json dump
    rep = build_report()
    tg_send(rep)

    # dosyaları tek tek yolla
    for pth in [STATE_FILE, OPEN_JSON, CLOSED_JSON, PARAM_FILE]:
        if os.path.exists(pth):
            with open(pth,"rb") as f:
                tg_file(os.path.basename(pth), f.read())

# ================= MAIN LOOP =================
def main_loop():
    # ilk init
    tg_send("🚀 EMA ULTRA v13.1 başladı (Scalp+Cross, AutoTrade OFF, Sim ON).")

    # state hiç resetlenmez, sadece diskten alır ve üstüne yazarız
    STATE["params"]=PARAM
    STATE.setdefault("bar_index", 0)
    STATE.setdefault("last_daily_sent_date","")  # rapor gönderildi mi takibi
    STATE.setdefault("last_cross_seen", {})
    STATE.setdefault("last_scalp_seen", {})
    STATE.setdefault("open_positions", STATE.get("open_positions", []))
    STATE.setdefault("auto_trade", STATE.get("auto_trade", False))
    STATE.setdefault("simulate", STATE.get("simulate", True))

    symbols = get_futures_symbols()
    if not symbols:
        tg_send("⚠️ Sembol listesi boş geldi Binance'ten.")
        symbols=[]

    while True:
        STATE["bar_index"] += 1

        # TELEGRAM KOMUTLARI
        poll_telegram(STATE)

        # GÜN SONU RAPORU / BACKUP
        today_str = today_date()
        if STATE.get("last_daily_sent_date","") != today_str:
            # her loop'ta değil sadece ilk defa aynı günü görünce at
            # yani bot restart sonrası ilk turda da atılacak
            send_daily_package()
            STATE["last_daily_sent_date"] = today_str
            safe_save(STATE_FILE, STATE)

        # SİNYAL ÜRETİMİ
        # her tur tüm sembolleri dönmek agresif olur ama burada basit yapıyoruz:
        for sym in symbols[:60]:  # ilk 60 sembolü tarayalım ki aşırı yormayalım
            k1 = get_klines(sym, "1h", 200)
            k4 = get_klines(sym, "4h", 120)
            kD = get_klines(sym, "1d", 90)
            if len(k1)<120 or len(k4)<40 or len(kD)<40:
                continue

            # CROSS
            cross_sig = build_cross_signal(sym, k1, k4, kD)
            if cross_sig and validate_signal(cross_sig):
                ck = f"{sym}_{cross_sig['dir']}"
                seen_bar = STATE["last_cross_seen"].get(ck)
                # burada sinyali tekrar yollamayı engelliyoruz, bar time olarak son kapanış ms kullanıyoruz
                last_bar_close_ms = int(k1[-1][6])
                if seen_bar != last_bar_close_ms:
                    tg_send(
                        f"{cross_sig['color']} CROSS {sym} {cross_sig['dir']}\n"
                        f"E:{cross_sig['entry']:.4f} TP:{cross_sig['tp']:.4f} SL:{cross_sig['sl']:.4f}\n"
                        f"RSI:{cross_sig['rsi']:.1f} Pow:{cross_sig['power']:.1f}\n"
                        f"4h:{cross_sig['trend4h']} 1D:{cross_sig['trend1d']} Align:{cross_sig['aligned']}\n"
                        f"Açı:{cross_sig['ang_now']:+.1f}° Δ:{cross_sig['ang_change']:.1f}°\n"
                        f"{now_ist_iso()}"
                    )
                    # AutoTrade sim/real handler
                    auto_trade_handle(cross_sig, STATE)
                    STATE["last_cross_seen"][ck] = last_bar_close_ms

            # SCALP
            scalp_sig = build_scalp_signal(
                sym, k1, k4, kD,
                STATE["last_scalp_seen"],
                STATE["bar_index"],
                PARAM["SCALP_COOLDOWN_BARS"]
            )
            if scalp_sig and validate_signal(scalp_sig):
                sk = f"{sym}_{scalp_sig['dir']}"
                # cooldown mantığı: sinyali yollamadan önce last_scalp_seen bakıldı zaten
                tg_send(
                    f"{scalp_sig['color']} SCALP {sym} {scalp_sig['dir']}\n"
                    f"E:{scalp_sig['entry']:.4f} TP:{scalp_sig['tp']:.4f} SL:{scalp_sig['sl']:.4f}\n"
                    f"RSI:{scalp_sig['rsi']:.1f} Pow:{scalp_sig['power']:.1f}\n"
                    f"4h:{scalp_sig['trend4h']} 1D:{scalp_sig['trend1d']} Align:{scalp_sig['aligned']}\n"
                    f"Açı:{scalp_sig['ang_now']:+.1f}° Δ:{scalp_sig['ang_change']:.1f}°\n"
                    f"{now_ist_iso()}"
                )
                auto_trade_handle(scalp_sig, STATE)
                STATE["last_scalp_seen"][sk] = STATE["bar_index"]

        # POZİSYONLARI KONTROL ET (TP/SL)
        check_positions_close(STATE)

        # STATE & PARAM kaydet
        STATE["params"]=PARAM
        safe_save(STATE_FILE, STATE)
        safe_save(PARAM_FILE, PARAM)

        # bekleme
        time.sleep(20)

# ================= MAIN =================
if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        log(f"[FATAL]{e}")
        try: tg_send(f"❗ Bot hata verdi: {e}")
        except: pass
