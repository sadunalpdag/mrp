import os, json, time, random, requests
from datetime import datetime, timezone

# ========= AYARLAR =========
EMA_7, EMA_25, EMA_99 = 7, 25, 99
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]

# ATR ayarları (interval bazlı)
DEFAULT_THRESHOLDS = {
    "1h": {"ATR_MIN_PCT": 0.0035, "ATR_SLOPE_MULT": 0.6},
    "4h": {"ATR_MIN_PCT": 0.0025, "ATR_SLOPE_MULT": 0.5},
    "1d": {"ATR_MIN_PCT": 0.0015, "ATR_SLOPE_MULT": 0.4},
}

GLOBAL_ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

SLEEP_BETWEEN = 0.25
SCAN_INTERVAL = 600
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
ALERTS_FILE = "alerts.json"
LOG_FILE = "log.txt"
# ===========================

def nowiso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {msg}\n")

def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("Telegram bilgileri eksik")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        log(f"[TG] {text}")
    except Exception as e:
        log(f"Telegram hatası: {e}")

def ema(values, length):
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for i in range(1, len(values)):
        ema_vals.append(values[i]*k + ema_vals[-1]*(1-k))
    return ema_vals

def slope_value(series, lookback=3):
    if len(series) < lookback+1: return 0.0
    return series[-1]-series[-lookback]

def true_ranges(highs,lows,closes):
    trs=[]
    for i in range(len(highs)):
        if i==0: trs.append(highs[i]-lows[i])
        else:
            prev=closes[i-1]
            trs.append(max(highs[i]-lows[i], abs(highs[i]-prev), abs(lows[i]-prev)))
    return trs

def atr_series(highs,lows,closes,period=14):
    trs=true_ranges(highs,lows,closes)
    if len(trs)<period: return [0]*len(trs)
    atr=[0]*len(trs)
    atr[period-1]=sum(trs[:period])/period
    for i in range(period,len(trs)):
        atr[i]=(atr[i-1]*(period-1)+trs[i])/period
    for i in range(period-1): atr[i]=trs[i]
    return atr

SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"EMA-MultiBot/1.0"})

def get_klines(sym,interval,limit=LIMIT):
    url="https://fapi.binance.com/fapi/v1/klines"
    p={"symbol":sym,"interval":interval,"limit":limit}
    for _ in range(3):
        try:
            r=SESSION.get(url,params=p,timeout=10)
            if r.status_code==200: return r.json()
            time.sleep(1)
        except: time.sleep(1)
    return []

def get_futures_symbols():
    try:
        data=SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo",timeout=10).json()
        return [s["symbol"] for s in data["symbols"]
                if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except: return []

def last_cross_info(ema_fast,ema_slow):
    last=None;direction=None
    for i in range(1,len(ema_fast)):
        if ema_fast[i-1]<ema_slow[i-1] and ema_fast[i]>ema_slow[i]:
            last=i;direction="UP"
        elif ema_fast[i-1]>ema_slow[i-1] and ema_fast[i]<ema_slow[i]:
            last=i;direction="DOWN"
    if last is None: return None,None
    return direction,len(ema_fast)-last-1

def thresholds_for_interval(interval):
    d=DEFAULT_THRESHOLDS.get(interval,{"ATR_MIN_PCT":0.003,"ATR_SLOPE_MULT":0.5})
    return d["ATR_MIN_PCT"], d["ATR_SLOPE_MULT"]

def process_symbol(sym,intervals,state):
    alerts=[]
    for interval in intervals:
        kl=get_klines(sym,interval)
        if not kl or len(kl)<EMA_99: continue
        highs=[float(k[2]) for k in kl]
        lows=[float(k[3]) for k in kl]
        closes=[float(k[4]) for k in kl]
        last_price=closes[-1]
        ema7=ema(closes,EMA_7)
        ema25=ema(closes,EMA_25)
        atr=atr_series(highs,lows,closes,GLOBAL_ATR_PERIOD)
        atr_now=atr[-1]; atr_pct=atr_now/last_price if last_price>0 else 0
        ATR_MIN_PCT_I,ATR_SLOPE_MULT_I=thresholds_for_interval(interval)
        cross_dir,bars_ago=last_cross_info(ema7,ema25)
        if cross_dir and bars_ago==0:
            slope_now=slope_value(ema7,lookback=3)
            alerts.append(("CROSS",interval,cross_dir,slope_now,last_price))
            if atr_pct>=ATR_MIN_PCT_I and abs(slope_now)>=ATR_SLOPE_MULT_I*atr_now:
                alerts.append(("ATR_CROSS",interval,cross_dir,slope_now,last_price,atr_now,atr_pct,ATR_MIN_PCT_I,ATR_SLOPE_MULT_I))
        time.sleep(SLEEP_BETWEEN)
    return alerts

def main():
    log("🚀 EMA + ATR bot başlatıldı")
    state={}
    symbols=get_futures_symbols()
    log(f"{len(symbols)} coin bulundu.")
    while True:
        for sym in symbols:
            alerts=process_symbol(sym,INTERVALS,state)
            for a in alerts:
                if a[0]=="CROSS":
                    _,interval,dir,slope,price=a
                    msg=(f"⚡ CROSS: {sym} ({interval})\n"
                         f"Dir: {dir}\nSlope: {slope:.6f}\nPrice: {price}\nTime: {nowiso()}")
                    send_telegram(msg)
                elif a[0]=="ATR_CROSS":
                    _,interval,dir,slope,price,atr_now,atr_pct,thr_pct,thr_mult=a
                    msg=(f"⚡ ATR CROSS: {sym} ({interval})\n"
                         f"Dir: {dir}\nSlope: {slope:.6f}\nATR: {atr_now:.6f} ({atr_pct*100:.2f}%)\n"
                         f"Eşikler: ≥{thr_pct*100:.2f}% ATR, slope≥{thr_mult:.2f}×ATR\n"
                         f"Price: {price}\nTime: {nowiso()}")
                    send_telegram(msg)
        log(f"⏳ {SCAN_INTERVAL//60} dk bekleniyor...\n")
        time.sleep(SCAN_INTERVAL)

if __name__=="__main__":
    main()
