"""Paper-only 4H falling-trend breakout research using paced, incremental REST.

Only one 4H kline request per 120 seconds. Reuses closed candles for four
hours; 15m confirmation uses existing websocket cache (no extra REST).
Never changes Crash Bounce V2 qualification, Telegram signals or orders.
"""
import csv
import io
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PERIOD = 4 * 3600
MIN_REST_GAP = 120
WATCH_COUNT = 30
HORIZONS = (1, 4, 12, 24)


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def ema(values, period):
    alpha = 2.0/(period+1)
    e = values[0]
    for x in values[1:]:
        e = alpha*x + (1-alpha)*e
    return e


def rsi(values):
    if len(values)<15:
        return None
    changes = [values[i]-values[i-1] for i in range(len(values)-14, len(values))]
    up = sum(max(0, d) for d in changes)/14
    down = sum(max(0, -d) for d in changes)/14
    return 100 if down==0 else 100-100/(1+up/down)


def candidates(tickers, market, limit=WATCH_COUNT):
    """24h buckets prioritize weak/recovering coins, then liquidity."""
    buckets = ([], [], [])
    for t in tickers:
        symbol = t.get("symbol")
        if (not isinstance(symbol,str) or not symbol.isascii()
                or not symbol.endswith("USDT") or symbol not in market):
            continue
        price, change, vol = (number(t.get(key)) for key in
                               ("lastPrice", "priceChangePercent", "quoteVolume"))
        if price is None or price<=0 or change is None or vol is None or vol<10_000_000:
            continue
        item = (symbol, price, change, vol)
        if -10 <= change <= -3:
            buckets[0].append(item)
        elif -30 <= change < -10:
            buckets[1].append(item)
        elif 5 <= change <= 15:
            buckets[2].append(item)
    for b in buckets:
        b.sort(key=lambda item:item[3],reverse=True)
    selected = []
    seen = set()
    for b, max_count in zip(buckets,(12,10,8)):
        for item in b[:max_count]:
            if item[0] not in seen:
                selected.append(item)
                seen.add(item[0])
    for item in sorted((item for b in buckets for item in b if item[0] not in seen),
                       key=lambda item:item[3],reverse=True):
        if len(selected)>=limit:
            break
        selected.append(item)
        seen.add(item[0])
    return selected[:limit]


def analyze(symbol, rows, now):
    """Trend line uses three descending local highs from prior CLOSED bars."""
    try:
        bars = sorted((r for r in rows if len(r)>7 and int(r[6])<now*1000),
                      key=lambda r:int(r[0]))[-65:]
        if len(bars)<45:
            return None
        highs = [float(r[2]) for r in bars]
        closes = [float(r[4]) for r in bars]
        vol = [float(r[7]) for r in bars]
        if any(not math.isfinite(x) or x<=0 for x in highs+closes):
            return None
        # Restrict pivots to previous bars: never use the current bar as a pivot.
        pivots = [i for i in range(max(1,len(highs)-42),len(highs)-2)
                  if highs[i]>highs[i-1] and highs[i]>highs[i+1]]
        if len(pivots)<3:
            return None
        a,b,c = pivots[-3:]
        if not highs[a]>highs[b]>highs[c]:
            return None
        slope = (highs[c]-highs[b])/(c-b)
        trend_now = highs[c]+slope*(len(closes)-1-c)
        trend_before = trend_now-slope
        if trend_now<=0 or trend_before<=0:
            return None
        volume_base = sum(vol[-21:-1])/20
        vr = vol[-1]/volume_base if volume_base>0 else None
        rr = rsi(closes)
        direction = ema(closes,25)-ema(closes[:-1],25)
        prior_direction = ema(closes[:-1],25)-ema(closes[:-2],25)
        slope_improving = direction>=prior_direction
        distance = 100*(closes[-1]/trend_now-1)
        crossed = closes[-2]<=trend_before and closes[-1]>trend_now
        if crossed:
            if distance>2.0:
                status = "OVEREXTENDED"
            elif vr is None or vr<1.5:
                status = "WEAK_VOLUME"
            elif rr is not None and 35<=rr<=70:
                status = "BREAKOUT"
            else:
                status = "MOMENTUM_UNCONFIRMED"
        elif -2 <= distance <= 0 and rr is not None and 35<=rr<=55 and slope_improving:
            status = "APPROACHING"
        else:
            status = "NONE"
        return {"symbol":symbol,"status":status,
                "close_at":(int(bars[-1][6])+1)/1000,
                "close_ms":int(bars[-1][6]),
                "close":closes[-1],"line":round(trend_now,10),
                "distance_pct":round(distance,3),
                "change_5d_pct":round(100*(closes[-1]/closes[-31]-1),3),
                "rsi14_4h":round(rr,2) if rr is not None else None,
                "volume_ratio_4h":round(vr,2) if vr is not None else None,
                "ema25_improving":slope_improving, "three_lower_highs":True}
    except (KeyError,ValueError,TypeError,OverflowError,IndexError,ZeroDivisionError):
        return None


def confirm_15m(cache, symbol, now):
    """Best effort from WebSocket cache only; no new network requests."""
    try:
        bars=sorted((r for r in cache.candles(symbol)
                     if len(r)>6 and int(r[6])<now*1000),
                    key=lambda r:int(r[0]))
        closes=[float(r[4]) for r in bars[-50:]]
        if len(closes)<26:
            return "NO_WS_CANDLES"
        rr=rsi(closes)
        return "CONFIRMED" if closes[-1]>ema(closes,25) and 45<=rr<=70 else "WAIT"
    except (AttributeError,ValueError,TypeError,IndexError):
        return "NO_WS_CANDLES"


def export_csv(events):
    fields=["event_id","symbol","status","close_at","close","line",
            "distance_pct","rsi14_4h","volume_ratio_4h","confirmation_15m",
            "observations","max_observed_gain_pct","gain_1h","gain_4h",
            "gain_12h","gain_24h"]
    out=io.StringIO()
    writer=csv.DictWriter(out,fieldnames=fields,extrasaction="ignore")
    writer.writeheader()
    for event in events:
        row=event.copy()
        for h in HORIZONS:
            row["gain_%dh"%h]=event.get("completed_horizons",{}).get(str(h))
        writer.writerow(row)
    return out.getvalue()


class BreakoutResearch:
    def __init__(self, directory, cache, fetch, telegram, market_symbols,
                 log=print, clock=time.time, env=None):
        self.env=os.environ if env is None else env
        self.enabled=self.env.get("CRASH_4H_BREAKOUT_ENABLED","1")=="1"
        self.root=Path(directory)
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/"breakout_4h_state.json"
        self.cache,self.fetch,self.telegram,self.market_symbols=cache,fetch,telegram,market_symbols
        self.log,self.clock=log,clock
        self.lock=threading.RLock()
        self.state={"cache":{},"events":[],"last_rest_attempt":0,"last_alert_slot":-1}
        if self.path.exists():
            loaded=json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded,dict) or not isinstance(loaded.get("events"),list):
                raise ValueError("BREAKOUT_4H_STATE_INVALID")
            self.state.update(loaded)

    def save(self):
        tmp=self.path.with_suffix(".tmp")
        with tmp.open("w",encoding="utf-8") as f:
            json.dump(self.state,f,ensure_ascii=False,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.path)

    def step(self):
        if not self.enabled:
            return
        now=self.clock()
        tickers=self.cache.snapshot(now)
        market=set(self.market_symbols() or ())
        if not tickers or not market:
            return
        prices={t.get("symbol"):number(t.get("lastPrice")) for t in tickers}
        selected=candidates(tickers,market)
        if not selected:
            return
        slot=int(now//PERIOD)
        latest_close_ms=slot*PERIOD*1000-1
        with self.lock:
            for item in self.state["events"]:
                if item.get("status")!="BREAKOUT" or now-item["close_at"]>24*3600:
                    continue
                p=prices.get(item["symbol"])
                if p is None or p<=0 or now<=item["close_at"]:
                    continue
                last=item.get("last_observed_at")
                if last is not None and now-last>180:
                    item["gap_count"]=item.get("gap_count",0)+1
                if last is not None and now<=last:
                    continue
                gain=100*(p/item["close"]-1)
                item["observations"]=item.get("observations",0)+1
                item["last_observed_at"]=now
                item["max_observed_gain_pct"]=round(max(
                    gain,item.get("max_observed_gain_pct") or gain),3)
                for h in HORIZONS:
                    if now<=item["close_at"]+h*3600:
                        partial=item.setdefault("partial_max",{})
                        partial[str(h)]=max(gain,partial.get(str(h),gain))
                    else:
                        pmax=item.get("partial_max",{}).get(str(h))
                        if pmax is not None:
                            item.setdefault("completed_horizons",{})[str(h)]=round(pmax,3)
            # Do not hit Binance on each loop. If a 4H close is missing, take
            # exactly one candidate, at least 120s after the previous request.
            missing=[]
            for symbol,price,change,vol in selected:
                entry=self.state["cache"].get(symbol,{})
                if entry.get("close_ms",0)<latest_close_ms:
                    missing.append(symbol)
            if missing and now-self.state["last_rest_attempt"]>=MIN_REST_GAP:
                symbol=missing[0]
                # Reserve the slot on disk before network call, including errors.
                self.state["last_rest_attempt"]=now
                self.save()
                try:
                    rows=self.fetch("/fapi/v1/klines",
                        {"symbol":symbol,"interval":"4h","limit":65})
                    result=analyze(symbol,rows,now)
                    last_closed=max((int(r[6]) for r in rows if len(r)>6
                                     and int(r[6])<now*1000),default=0)
                    if last_closed:
                        self.state["cache"][symbol]={"close_ms":last_closed,
                            "result":result,"fetched_at":now}
                    if result and result["status"]!="NONE":
                        eid=symbol+":"+str(result["close_ms"])+":"+result["status"]
                        if not any(r.get("event_id")==eid for r in self.state["events"]):
                            result.update(event_id=eid,observations=0,
                                          confirmation_15m=confirm_15m(self.cache,symbol,now),
                                          max_observed_gain_pct=None,gap_count=0)
                            self.state["events"].append(result)
                except Exception as error:
                    self.log("[BREAKOUT 4H REST DEFERRED] "+symbol+" "+type(error).__name__)
            self.state["events"]=self.state["events"][-1500:]
            self.state["cache"]={k:v for k,v in self.state["cache"].items()
                                  if k in {r[0] for r in selected}}
            self.save()
            # Report only after a full 60-minute slow intake period after close.
            # May report fewer candidates if rate limiting/WS coverage prevents fetching.
            if (now-slot*PERIOD>=3600 and self.state["last_alert_slot"]!=slot):
                self.state["last_alert_slot"]=slot
                current=[r for r in self.state["events"]
                         if r["close_ms"]==latest_close_ms]
                approaching=[r for r in current if r["status"]=="APPROACHING"]
                confirmed=[r for r in current if r["status"]=="BREAKOUT"]
                self.save()
                lines=["🔎 4H DÜŞEN TREND | SANAL ARAŞTIRMA",
                       f"{len(selected)} aday | 4H kırılımı {len(confirmed)} | yaklaşan {len(approaching)}",
                       "Trend: 3 alçalan tepe | hacim >=1.5x | sadece kapanmış 4H mum.",
                       "REST: en fazla 1 sorgu / 120 sn, önbellek kullanılır."]
                for r in (confirmed[:5]+approaching[:5]):
                    lines.append(f"{r['symbol']} {r['status']} | RSI {r['rsi14_4h']} "
                                 f"| hacim {r['volume_ratio_4h']}x | 15m {r['confirmation_15m']}")
                lines.append("Giriş emri açılmaz; V2 sinyal kuralları değişmedi.")
                try:
                    self.telegram("\n".join(lines))
                except Exception as error:
                    self.log("[BREAKOUT 4H TELEGRAM ERR] "+type(error).__name__)

    def start(self):
        if not self.enabled:
            self.log("[BREAKOUT 4H] OFF")
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as error:
                    self.log("[BREAKOUT 4H WORKER ERR] "+type(error).__name__)
                time.sleep(60)
        threading.Thread(target=run,daemon=True,name="crash-4h-breakout").start()
