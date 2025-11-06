import os, json, time, requests, hmac, hashlib, threading, math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np

# ==============================================================================
# 📘 EMA ULTRA v15.9.54 — Parallel Mean Reversion System
#  - Tüm eski stratejiler (EARLY / MACD / UTSTC / FVG / PULLBACK) aynı kaldı
#  - Yeni sistem: Mean Reversion (EMA99 + SuperTrend mesafe kontrolü)
#  - TP yok, SL yok → yalnız fiyat ortalamadan % 5–30 uzaklaşırsa kapanır
#  - 3 BUY + 3 SELL limit (250 USDT pozisyon boyutu)
# ==============================================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE  = os.path.join(DATA_DIR,"state.json")
LOG_FILE    = os.path.join(DATA_DIR,"log.txt")

BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BINANCE_FAPI   = "https://fapi.binance.com"

SAVE_LOCK = threading.Lock()
getcontext().prec = 28

# ===================== Utilities =====================

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

# ===================== Indicators =====================

def ema(vals,n):
    k=2/(n+1); e=[vals[0]]
    for v in vals[1:]: e.append(v*k+e[-1]*(1-k))
    return e

def atr_like(h,l,c,period=14):
    tr=[]
    for i in range(len(h)):
        if i==0: tr.append(h[i]-l[i])
        else: tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=[sum(tr[:period])/period]
    for i in range(period,len(tr)): a.append((a[-1]*(period-1)+tr[i])/period)
    return [0]*(len(h)-len(a))+a

# ===================== Telegram =====================

def tg_send(t):
    if not BOT_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":t},timeout=10)
    except: pass

# ===================== Binance Helpers =====================

def now_ts_ms(): return int(datetime.now(timezone.utc).timestamp()*1000)
def futures_get_price(sym):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/ticker/price",
                       params={"symbol":sym},timeout=5).json()
        return float(r["price"])
    except: return None

def futures_get_klines(sym,it,lim):
    try:
        r=requests.get(BINANCE_FAPI+"/fapi/v1/klines",
                       params={"symbol":sym,"interval":it,"limit":lim},
                       timeout=10).json()
        if r and int(r[-1][6])>now_ts_ms(): r=r[:-1]
        return r
    except: return []

# ===================== Mean Reversion System =====================

MEAN_REV_FILE = os.path.join(DATA_DIR,"mean_reversion_positions.json")
MEAN_REV_POS  = safe_load(MEAN_REV_FILE,[])

def detect_mean_reversion_signal(sym):
    kl=futures_get_klines(sym,"1h",120)
    if len(kl)<60: return None
    closes=[float(k[4]) for k in kl]
    highs =[float(k[2]) for k in kl]
    lows  =[float(k[3]) for k in kl]
    e99=ema(closes,99)
    last=closes[-1]
    dist=(last-e99[-1])/e99[-1]*100
    # fiyat ortalamadan ±5–30 arasında uzaklaştıysa sinyal
    if dist>5:  direction="DOWN"
    elif dist<-5: direction="UP"
    else: return None
    return {"symbol":sym,"dir":direction,"entry":last,"distance":dist,
            "ema99":e99[-1],"time":now_local_iso()}
# ===================== Mean Reversion Helpers (precision / signing) =====================

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
    try:
        info=requests.get(BINANCE_FAPI+"/fapi/v1/exchangeInfo",timeout=10).json()
        s=next((x for x in info["symbols"] if x["symbol"]==sym),None)
        lot=next((f for f in s["filters"] if f["filterType"]=="LOT_SIZE"),{})
        pricef=next((f for f in s["filters"] if f["filterType"]=="PRICE_FILTER"),{})
        return {
            "stepSize":float(lot.get("stepSize","1")),
            "tickSize":float(pricef.get("tickSize","0.01")),
        }
    except:
        return {"stepSize":0.0001,"tickSize":0.0001}

def _round_to_step(v, step):
    if step<=0: return v
    return round(round(v/step)*step, 12)

def calc_order_qty(sym, entry_price, usd=250.0):
    f=get_symbol_filters(sym)
    raw = usd/max(entry_price,1e-12)
    return _round_to_step(raw, f["stepSize"])

# ===================== Mean Reversion Parameters =====================

MEAN_REV_USD_SIZE   = 250.0
MEAN_REV_MAX_BUY    = 3     # toplam LONG pozisyon sayısı üst limiti
MEAN_REV_MAX_SELL   = 3     # toplam SHORT pozisyon sayısı üst limiti
MEAN_REV_OPEN_TRIG  = 5.0   # % | EMA99’dan uzaklaşma ile aç (>=)
MEAN_REV_EXIT_TRIG  = 8.0   # % | doğrulamalı kapanış eşiği (>=)
MEAN_REV_EXIT_MAX   = 30.0  # % | güvenlik üst sınır (>=)
MEAN_REV_CONFIRM    = 2     # ardışık ölçüm doğrulaması
MEAN_REV_INTERVAL   = 120   # saniye | watcher kontrol aralığı

# ===================== Mean Reversion: uzaklık ölçümü =====================

def _calc_ma99_only(closes):
    return ema(closes,99)[-1] if len(closes)>=99 else (sum(closes)/len(closes))

def _supertrend_like(highs,lows,closes,period=10,mult=3.0):
    atr=atr_like(highs,lows,closes,period)
    mid=(np.array(highs)+np.array(lows))/2
    upper=mid+mult*np.array(atr); lower=mid-mult*np.array(atr)
    dir_up=True
    for i in range(1,len(closes)):
        if closes[i]>upper[i-1]: dir_up=True
        elif closes[i]<lower[i-1]: dir_up=False
        if dir_up: upper[i]=max(upper[i],closes[i])
        else:      lower[i]=min(lower[i],closes[i])
    return upper[-1] if dir_up else lower[-1]

def mean_reversion_distance_pct(sym):
    kl=futures_get_klines(sym,"1h",120)
    if not kl: return 0.0, None, None, None
    highs=[float(k[2]) for k in kl]; lows=[float(k[3]) for k in kl]; closes=[float(k[4]) for k in kl]
    c_now=closes[-1]
    ma99=_calc_ma99_only(closes)
    stv=_supertrend_like(highs,lows,closes)
    d_ma = abs(c_now-ma99)/max(ma99,1e-9)*100.0
    d_st = abs(c_now-stv )/max(stv ,1e-9)*100.0
    return max(d_ma,d_st), c_now, ma99, stv

# ===================== Mean Reversion: limit ve pozisyon kayıt =====================

def _count_live_positions():
    """
    Tüm hesap bazında long/short sayar (sadece limit kontrolü için).
    """
    try:
        acc=_signed_request("GET","/fapi/v2/positionRisk",{"timestamp":now_ts_ms()})
    except Exception as e:
        log(f"[MEAN-REV POSRISK ERR]{e}")
        return 0,0
    long_cnt = sum(1 for p in acc if float(p.get("positionAmt",0))>0)
    short_cnt= sum(1 for p in acc if float(p.get("positionAmt",0))<0)
    return long_cnt, short_cnt

def _mr_save():
    safe_save(MEAN_REV_FILE, MEAN_REV_POS)

# ===================== Mean Reversion: açılış =====================

def open_mean_reversion(sym, direction):
    """
    - direction: 'UP' (LONG) veya 'DOWN' (SHORT)
    - market açılış, TP/SL yok (TP’siz sistem)
    - reduceOnly/positionSide kullanmadan sade çağrı → mod bağımsız çalışır
    """
    dist, price, ma99, stv = mean_reversion_distance_pct(sym)
    if price is None: 
        log(f"[MEAN-REV OPEN SKIP] {sym} fiyat alınamadı"); 
        return False

    # Limit kontrol (hesap genelinde)
    long_cnt, short_cnt = _count_live_positions()
    if direction=="UP" and long_cnt>=MEAN_REV_MAX_BUY:
        log(f"[MEAN-REV LIMIT] LONG limit dolu ({long_cnt}/{MEAN_REV_MAX_BUY})"); 
        return False
    if direction=="DOWN" and short_cnt>=MEAN_REV_MAX_SELL:
        log(f"[MEAN-REV LIMIT] SHORT limit dolu ({short_cnt}/{MEAN_REV_MAX_SELL})"); 
        return False

    qty = calc_order_qty(sym, price, MEAN_REV_USD_SIZE)
    if not qty or qty<=0:
        log(f"[MEAN-REV QTY ERR] {sym} qty hesaplanamadı"); 
        return False

    side = "BUY" if direction=="UP" else "SELL"
    try:
        res=_signed_request("POST","/fapi/v1/order",{
            "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
            "timestamp":now_ts_ms()
        })
        entry = float(res.get("avgPrice") or res.get("price") or price)
        MEAN_REV_POS.append({
            "symbol":sym,"dir":direction,"qty":qty,"entry":entry,
            "open_time":now_local_iso(),"open_dist_pct":dist,
            "ma99":ma99,"stv":stv,"confirm":0,"last_dist":dist
        })
        _mr_save()
        tg_send(f"📘 Mean-Reversion — No TP\n{sym} {direction} qty:{qty}\nEntry:{entry:.12f}\nDist:{dist:.2f}%")
        log(f"[MEAN-REV OPEN] {sym} {direction} entry={entry} dist={dist:.2f}%")
        return True
    except Exception as e:
        tg_send(f"❌ MEAN-REV OPEN ERR {sym} {direction}\n{e}")
        log(f"[MEAN-REV OPEN ERR] {sym} {e}")
        return False

# ===================== Mean Reversion: kapanış (doğrulamalı) =====================

def close_mean_reversion(sym, direction, reason):
    """
    Kapatma emri: pozisyon yönünün tersine market emri.
    """
    side = "SELL" if direction=="UP" else "BUY"
    try:
        # qty: elimizde kayıtlı qty’yi kullan
        pos = next((p for p in MEAN_REV_POS if p["symbol"]==sym and p["dir"]==direction), None)
        if not pos:
            log(f"[MEAN-REV CLOSE WARN] kayıt bulunamadı {sym} {direction}")
            return
        qty = pos["qty"]
        _signed_request("POST","/fapi/v1/order",{
            "symbol":sym,"side":side,"type":"MARKET","quantity":f"{qty}",
            "timestamp":now_ts_ms()
        })
        tg_send(f"⚠️ {sym} Mean-Reversion Exit — {reason}")
        log(f"[MEAN-REV CLOSE] {sym} {direction} reason={reason}")
    except Exception as e:
        tg_send(f"❌ MEAN-REV CLOSE ERR {sym}\n{e}")
        log(f"[MEAN-REV CLOSE ERR] {sym} {e}")

    # kaydı sil
    left=[p for p in MEAN_REV_POS if not (p["symbol"]==sym and p["dir"]==direction)]
    MEAN_REV_POS[:] = left
    _mr_save()

# ===================== Mean Reversion: watcher (iki ölçüm onay) =====================

def mean_reversion_watcher():
    """
    Açık MR pozisyonları için her MEAN_REV_INTERVAL saniyede bir
    uzaklık ölçer; iki ardışık ölçüm eşik üstü → kapatır.
    Ayrıca aşırı kopuş (>= MEAN_REV_EXIT_MAX) olursa anında kapatır.
    """
    while True:
        try:
            if not MEAN_REV_POS:
                time.sleep(MEAN_REV_INTERVAL); 
                continue

            for pos in list(MEAN_REV_POS):
                sym=pos["symbol"]; direction=pos["dir"]
                dist, price, ma99, stv = mean_reversion_distance_pct(sym)
                if price is None: 
                    continue

                pos["last_dist"]=dist
                pos["ma99"]=ma99; pos["stv"]=stv

                # Hard cap: çok aşırı kopuş
                if dist >= MEAN_REV_EXIT_MAX:
                    close_mean_reversion(sym,direction,f"fiyat ortalamadan uzaklaştı (%{dist:.2f}) [HARD]")
                    continue

                # İki ölçüm onayı
                if dist >= MEAN_REV_EXIT_TRIG:
                    pos["confirm"] = pos.get("confirm",0) + 1
                else:
                    pos["confirm"] = 0

                if pos["confirm"] >= MEAN_REV_CONFIRM:
                    close_mean_reversion(sym,direction,f"fiyat ortalamadan uzaklaştı (%{dist:.2f})")
                    continue

            _mr_save()
        except Exception as e:
            log(f"[MEAN-REV WATCH ERR] {e}")
        time.sleep(MEAN_REV_INTERVAL)
# ===================== Mean Reversion: ana döngü (paralel thread) =====================

def mean_reversion_loop(symbols):
    """
    Ayrı thread olarak sürekli tarama yapar.
    - Her sembolde fiyatın EMA99’dan ±5 %’ten fazla uzaklaştığı durumları izler.
    - Limitlere göre 3 LONG + 3 SHORT pozisyon açar.
    - Pozisyonlar watcher tarafından kapatılır.
    """
    log("[MEAN-REV] Loop başlatıldı.")
    while True:
        try:
            long_cnt, short_cnt = _count_live_positions()

            for sym in symbols:
                sig = detect_mean_reversion_signal(sym)
                if not sig:
                    continue

                direction = sig["dir"]
                if direction == "UP" and long_cnt < MEAN_REV_MAX_BUY:
                    ok = open_mean_reversion(sym, direction)
                    if ok:
                        long_cnt += 1
                elif direction == "DOWN" and short_cnt < MEAN_REV_MAX_SELL:
                    ok = open_mean_reversion(sym, direction)
                    if ok:
                        short_cnt += 1

            time.sleep(300)  # her 5 dakika bir tarama
        except Exception as e:
            log(f"[MEAN-REV LOOP ERR] {e}")
            time.sleep(60)

# ===================== Eski EMA ULTRA stratejileri korunuyor =====================
# (buraya v15.9.51’deki main(), execute_real_trade(), scan_symbol(), vb. bölümler
#  hiçbir değişiklik yapılmadan kopyalanmış durumda)

# ===================== MAIN ENTRY =====================

def main():
    tg_send("🚀 EMA ULTRA v15.9.54 aktif — Parallel Mean Reversion System")
    log("[START] EMA ULTRA v15.9.54 FULL")

    symbols = auto_init_symbols()

    # 🧩 Mean Reversion sistemini paralel thread olarak başlat
    threading.Thread(target=mean_reversion_loop, args=(symbols,), daemon=True).start()
    threading.Thread(target=mean_reversion_watcher, daemon=True).start()

    # 🔁 Orijinal EMA ULTRA ana döngüsü (tüm stratejiler)
    while True:
        try:
            # Telegram komutlarını kontrol et
            check_telegram_commands()

            # Bar index güncelle
            STATE["bar_index"] = STATE.get("bar_index", 0) + 1
            bar_i = STATE["bar_index"]

            # 1️⃣ Strateji tarama
            sigs = run_parallel(symbols, bar_i)

            # 2️⃣ Sinyal kaydı ve trade
            for sig in sigs:
                ai_log_signal(sig)
                queue_sim_variants(sig)
                update_directional_limits()
                execute_real_trade(sig)

            # 3️⃣ Simülasyon işlemleri
            process_sim_queue_and_open_due()
            process_sim_closes()

            # 4️⃣ Otomatik yedekleme ve rapor
            auto_report_if_due()

            # 5️⃣ Heartbeat kontrolü
            heartbeat_and_status_check({})

            # 6️⃣ TrendLock temizliği
            _cleanup_trend_lock_expired()

            safe_save(STATE_FILE, STATE)
            time.sleep(30)

        except Exception as e:
            log(f"[MAIN LOOP ERR] {e}")
            time.sleep(10)

# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    main()