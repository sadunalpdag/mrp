# === EMA ULTRA FINAL v9.1 ===
# Binance only | Smart Scalp (Power≥68, TP 0.6%) | Power≥60 SIM | ATR/RSI/Divergence | Excel rapor

import os, time, json, io, requests, csv
from datetime import datetime, timezone

# ========= AYARLAR =========
LIMIT = 300
INTERVALS = ["1h", "4h", "1d"]
EMA_SETS = {"1h": (7,25,99), "4h": (9,26,200), "1d": (20,50,200)}
ATR_PERIOD = 14
RSI_PERIOD = 14
SR_LOOKBACK = 100
EARLY_CONFIRM_MS = 30*60*1000

# --- SIMULASYON ---
SIM_ENABLE = True
SIM_MIN_POWER = 60
SIM_TP_PCT = 0.01       # %1
SIM_SL_PCT = 0.10       # %10
SCALP_MIN_POWER = 68
SCALP_TP_PCT = 0.006    # %0.6

REPORT_INTERVAL_MIN = 60
SLEEP_BETWEEN = 0.2
SCAN_INTERVAL = 300

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
STATE_FILE = "alerts.json"
LOG_FILE   = "log.txt"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EMA-ULTRA/2.0"})

# ========= UTILS =========
def nowiso(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def log(msg): print(msg); open(LOG_FILE,"a").write(f"{datetime.now()} - {msg}\n")

def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        url=f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url,data={"chat_id":CHAT_ID,"text":text},timeout=10)
    except: pass

def send_doc(b,fname,cap=""):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        url=f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        requests.post(url,files={'document':(fname,b)},data={'chat_id':CHAT_ID,'caption':cap},timeout=20)
    except: pass

def safe_load(p): 
    try:
        if os.path.exists(p): return json.load(open(p,"r",encoding="utf-8"))
    except: pass
    return {}
def safe_save(p,d):
    tmp=p+".tmp"; json.dump(d,open(tmp,"w",encoding="utf-8"),indent=2); os.replace(tmp,p)

# ========= INDIKATORLER =========
def ema(v,l): k=2/(l+1); e=[v[0]]
 for i in range(1,len(v)): e.append(v[i]*k+e[-1]*(1-k))
 return e

def slope_value(s,l=3): return s[-1]-s[-l] if len(s)>l else 0

def atr_series(h,lw,c,p=14):
 trs=[]
 for i in range(len(h)):
  if i==0: trs.append(h[i]-lw[i])
  else: pc=c[i-1]; trs.append(max(h[i]-lw[i],abs(h[i]-pc),abs(lw[i]-pc)))
 if len(trs)<p: return [0]*len(trs)
 a=[sum(trs[:p])/p]
 for i in range(p,len(trs)): a.append((a[-1]*(p-1)+trs[i])/p)
 return [0]*(len(trs)-len(a))+a

def rsi(v,p=14):
 n=len(v)
 if n<p+1: return [None]*n
 d=[v[i]-v[i-1] for i in range(1,n)]
 g=[max(x,0) for x in d]; s=[abs(min(x,0)) for x in d]
 avg_g=sum(g[:p])/p; avg_s=sum(s[:p])/p; r=[]
 for i in range(p,len(d)):
  avg_g=(avg_g*(p-1)+g[i])/p; avg_s=(avg_s*(p-1)+s[i])/p
  if avg_s==0: r.append(100.0)
  else:
   rs=avg_g/avg_s; r.append(100-100/(1+rs))
 return [None]*(n-len(r))+r

# ========= BINANCE =========
def get_klines(sym,intv,limit=LIMIT):
 url="https://fapi.binance.com/fapi/v1/klines"
 try:
  r=SESSION.get(url,params={"symbol":sym,"interval":intv,"limit":limit},timeout=10)
  return r.json() if r.status_code==200 else []
 except: return []

def get_syms():
 try:
  r=SESSION.get("https://fapi.binance.com/fapi/v1/exchangeInfo",timeout=10)
  d=r.json()
  return [s["symbol"] for s in d["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
 except: return []

# ========= SIM =========
def ensure_sim(st):
 st.setdefault("positions",{}); st.setdefault("history",[]); st.setdefault("last_report_ts",0)
 return st

def open_pos(st,sym,side,price,src,power,s_prev=None,s_now=None):
 if sym in st["positions"] and st["positions"][sym].get("is_open"): return
 tp_pct = SCALP_TP_PCT if src=="SCALP" else SIM_TP_PCT
 st["positions"][sym]={"is_open":True,"side":side,"entry":price,"tp_pct":tp_pct,"sl_pct":SIM_SL_PCT,
  "source":src,"power":power,"opened_at":nowiso(),"bars":0,"s_prev":s_prev,"s_now":s_now}
def tick(st,sym): 
 if sym in st["positions"] and st["positions"][sym].get("is_open"):
  st["positions"][sym]["bars"]+=1

def check_close(st,sym,price):
 p=st["positions"].get(sym)
 if not p or not p.get("is_open"): return
 side=p["side"]; e=p["entry"]; tp=e*(1+(p["tp_pct"] if side=="LONG" else -p["tp_pct"]))
 sl=e*(1+(-p["sl_pct"] if side=="LONG" else p["sl_pct"]))
 hit_tp=(price>=tp if side=="LONG" else price<=tp)
 hit_sl=(price<=sl if side=="LONG" else price>=sl)
 if hit_tp or hit_sl:
  res="TP" if hit_tp else "SL"; pnl=p["tp_pct"] if hit_tp else -p["sl_pct"]
  h={"symbol":sym,"side":side,"entry":round(e,6),"exit":round(price,6),
     "pnl_pct":round(pnl*100,2),"outcome":res,"bars":p["bars"],
     "source":p["source"],"power":p["power"],
     "slope_prev":p.get("s_prev"),"slope_now":p.get("s_now"),
     "slope_change":(p.get("s_now")-p.get("s_prev")) if (p.get("s_prev") is not None) else None,
     "closed_at":nowiso()}
  st["history"].append(h); st["positions"][sym]={"is_open":False}
  send_tg(f"📘 SIM | {res} | {sym} {side}\nPnL: {pnl*100:.2f}% Bars: {p['bars']} From: {p['source']} Power={p['power']}")
  safe_save(STATE_FILE,st)

def make_report_bytes(hist):
 buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=["symbol","side","entry","exit","pnl_pct","outcome","bars","source","power","slope_prev","slope_now","slope_change","closed_at"])
 w.writeheader()
 for r in hist: w.writerow(r)
 return buf.getvalue().encode("utf-8")

def maybe_report(st):
 if not st["history"]: return
 now_ts=int(datetime.now(timezone.utc).timestamp())
 if now_ts-st["last_report_ts"]<REPORT_INTERVAL_MIN*60: return
 b=make_report_bytes(st["history"])
 send_doc(b,"sim_report.csv",f"📊 SIM Raporu | {len(st['history'])} işlem")
 st["last_report_ts"]=now_ts; safe_save(STATE_FILE,st)

# ========= SCALP TESPİT =========
def detect_slope_rev(ema7):
 if len(ema7)<6: return None,(0,0)
 s_now=ema7[-1]-ema7[-4]; s_prev=ema7[-2]-ema7[-5]
 if s_prev<0 and s_now>0: return "UP",(s_prev,s_now)
 if s_prev>0 and s_now<0: return "DOWN",(s_prev,s_now)
 return None,(s_prev,s_now)

# ========= ANA =========
def process(sym,st):
 for intv in INTERVALS:
  kl=get_klines(sym,intv)
  if not kl or len(kl)<100: continue
  closes=[float(k[4]) for k in kl]; highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]
  ema7=ema(closes,7)
  slope_flip,(s_prev,s_now)=detect_slope_rev(ema7)
  price=closes[-1]
  if intv=="1h" and slope_flip:
   atr=atr_series(highs,lows,closes,ATR_PERIOD); atr_now=atr[-1]; atr_pct=atr_now/price if price>0 else 0
   scalp_power=max(0,min(100,60+abs(s_now-s_prev)/(atr_now*0.6)*20))
   if scalp_power>=SCALP_MIN_POWER:
    tp=price*(1+0.006 if slope_flip=="UP" else 1-0.006)
    sl=price*(1-0.10 if slope_flip=="UP" else 1+0.10)
    send_tg(f"💥 SCALP {('LONG' if slope_flip=='UP' else 'SHORT')} TRIGGER {sym}\nSlope: {s_prev:+.6f}→{s_now:+.6f}\nTP≈{tp:.6f} SL≈{sl:.6f}\nPower={scalp_power:.1f}\nTime:{nowiso()}")
    open_pos(st,sym,"LONG" if slope_flip=="UP" else "SHORT",price,"SCALP",round(scalp_power,1),s_prev,s_now)
  # pozisyon takip
  tick(st,sym); check_close(st,sym,price)

def main():
 log("🚀 v9.1 başlatıldı (Scalp TP 0.6% SL 10% Power≥68)")
 st=ensure_sim(safe_load(STATE_FILE))
 syms=get_syms()
 while True:
  for s in syms: process(s,st)
  maybe_report(st)
  log("⏳ 5dk bekleniyor..."); time.sleep(SCAN_INTERVAL)

if __name__=="__main__": main()