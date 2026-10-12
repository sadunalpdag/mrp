"""Independent Fibonacci LONG research signals; no account/order endpoints.
Pivots require two closed candles on either side. Confirmation is the latest
closed 15m candle; entry is the fresh observed price after that close.
"""
import json
import math
import os
import threading
import time
import zipfile
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from datetime import datetime, timezone
from crash_history import HistoryCollector


def candles(rows, now, interval):
    result = sorted((r for r in rows if int(r[6]) < now*1000), key=lambda r:int(r[0]))
    if not result or now*1000-int(result[-1][6]) > interval:
        return []
    if any(int(b[0])-int(a[0]) != interval for a,b in zip(result,result[1:])):
        return []
    for r in result:
        if any(not math.isfinite(float(r[i])) or float(r[i])<=0 for i in (1,2,3,4)):
            return []
    return result


def ema(values, period):
    if len(values)<period+1:
        return None, None
    value=sum(values[:period])/period
    prior=value
    for v in values[period:]:
        prior=value
        value+=(v-value)*2/(period+1)
    return value,prior


def select(tickers, market):
    candidates=[]
    for t in tickers:
        try:
            change,volume=float(t['priceChangePercent']),float(t['quoteVolume'])
            if t['symbol'] in market and 5<=change<=10 and math.isfinite(volume) and volume>0:
                candidates.append(t)
        except (KeyError,TypeError,ValueError):
            continue
    return sorted(candidates,key=lambda t:(-float(t['quoteVolume']),t['symbol']))[:10]


def setup(rows, hourly, price, tick, now):
    rows=candles(rows,now,900000)
    hourly=candles(hourly,now,3600000)
    if len(rows)<32 or len(hourly)<201:
        return None
    hc=[float(r[4]) for r in hourly]
    e200,_=ema(hc,200);e50,p50=ema(hc,50)
    if hc[-1]<=e200 or e50<=p50:
        return None
    low=[float(r[3]) for r in rows];high=[float(r[2]) for r in rows]
    close=[float(r[4]) for r in rows]
    # Confirmed local trough followed by a confirmed local peak.
    troughs=[i for i in range(2,len(rows)-2) if low[i]<min(low[i-2:i]+low[i+1:i+3])]
    peaks=[i for i in range(2,len(rows)-2) if high[i]>max(high[i-2:i]+high[i+1:i+3])]
    if not peaks:
        return None
    b=peaks[-1]
    before=[i for i in troughs if i<b]
    if not before or b>=len(rows)-2:
        return None
    a=before[-1];bottom,top=low[a],high[b]
    if top<=bottom or min(low[b+1:])<=bottom:
        return None
    zone_low=top-.618*(top-bottom);zone_high=top-.5*(top-bottom)
    # Latest candle must test the zone and close through previous candle high.
    if not (low[-1]<=zone_high and high[-1]>=zone_low and close[-1]>high[-2]
            and close[-1]>float(rows[-1][1])):
        return None
    tr=[max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1])) for i in range(1,len(rows))]
    atr=sum(tr[-14:])/14
    step=Decimal(str(tick))
    entry=float((Decimal(str(price))/step).to_integral_value(rounding=ROUND_CEILING)*step)
    sl=float((Decimal(str(min(low[b+1:])-.2*atr))/step).to_integral_value(rounding=ROUND_FLOOR)*step)
    target=float((Decimal(str(top))/step).to_integral_value(rounding=ROUND_FLOOR)*step)
    if not 0<sl<entry<target or (target-entry)/(entry-sl)<2:
        return None
    return dict(entry=entry,sl=sl,tp=target,rr=(target-entry)/(entry-sl),
                swing_low=bottom,swing_high=top,zone_low=zone_low,zone_high=zone_high,
                a_time=int(rows[a][0]),b_time=int(rows[b][0]),confirmation_time=int(rows[-1][6]),
                ema200_1h=e200,ema50_1h=e50,atr14_15m=atr,
                strategy_version='fib_pullback_v1')


class FibonacciEngine:
    def __init__(self, directory, fetch, snapshot, market, send, send_file, blocked, log):
        self.directory=directory
        os.makedirs(directory,exist_ok=True)
        self.fetch,self.snapshot,self.market=fetch,snapshot,market
        self.send,self.send_file,self.blocked,self.log=send,send_file,blocked,log
        self.lock=threading.RLock()
        path=os.path.join(directory,'state.json')
        if os.path.exists(path):
            with open(path,encoding='utf-8') as f:self.state=json.load(f)
        else:self.state=dict(records=[],next_report=time.time()+86400)
        self.cache={}
        self.thread=None
        self.history=HistoryCollector(os.path.join(directory,'history_cache'),fetch,self.emit,blocked,log)

    def save(self):
        path=os.path.join(self.directory,'state.json')
        with open(path+'.tmp','w',encoding='utf-8') as f:
            json.dump(self.state,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(path+'.tmp',path)

    def emit(self,event,now):
        day=datetime.fromtimestamp(now,timezone.utc).strftime('%Y-%m-%d')
        with self.lock,open(os.path.join(self.directory,day+'.jsonl'),'a',encoding='utf-8') as f:
            f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

    def get_rows(self,symbol,interval,limit,now):
        span=900 if interval=='15m' else 3600
        key=(symbol,interval,int(now//span))
        if key not in self.cache:
            rows=self.fetch('/fapi/v1/klines',dict(symbol=symbol,interval=interval,limit=limit))
            self.cache[key]=rows
        return self.cache[key]

    def report(self,now):
        if now<self.state['next_report']:
            return
        path=os.path.join(self.directory,'fibonacci_data.zip')
        with self.lock,zipfile.ZipFile(path+'.tmp','w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('signals_and_outcomes.json',json.dumps(self.state['records']))
            archive.writestr('README.txt','Fibonacci simulated LONG signals; sampled prices, no actual fills or fees. history_5d events join via signal_id. as_of is original signal time. Incomplete history is explicitly marked. SL/TP checked every scan; gaps can miss both. Outcomes are not exchange executions.')
            for name in os.listdir(self.directory):
                if name.endswith('.jsonl') and os.path.getmtime(os.path.join(self.directory,name))>=now-8*86400:
                    archive.write(os.path.join(self.directory,name),'observations/'+name)
        os.replace(path+'.tmp',path)
        if os.path.getsize(path)<=49*1024*1024 and self.send_file(path,'📊 FIBONACCI — sinyaller, sanal TP/SL sonuçları ve 5 günlük 15m geçmiş'):
            self.state['next_report']=now+86400
        else:self.state['next_report']=now+1800
        self.save()

    def scan(self,now=None):
        now=time.time() if now is None else now
        tickers=self.snapshot(now);market=self.market()
        current={t['symbol']:t for t in tickers}
        # Outcomes and pending messages continue even when REST is paused.
        for r in self.state['records']:
            if r['status']=='OPEN' and r['symbol'] in current and now>r['created_at']:
                price=float(current[r['symbol']]['lastPrice'])
                if math.isfinite(price) and price>0:
                    previous=r.get('last_observed_at',r['created_at'])
                    r['coverage_gaps']=r.get('coverage_gaps',0)+int(now-previous>540)
                    r.update(last_observed_at=now,last_observed_price=price)
                    if price<=r['sl']:r.update(status='SL',finished_at=now)
                    elif price>=r['tp']:r.update(status='TP',finished_at=now)
                    elif now-r['created_at']>=72*3600:r.update(status='TIMEOUT',finished_at=now)
                    self.emit(dict(type='tracking',signal_id=r['signal_id'],observed_at=now,price=price,status=r['status']),now)
            if not r.get('sent'):
                if now-r['created_at']>900:r['sent']=True;r['delivery_status']='STALE'
                elif self.send(r['message']):r['sent']=True;r['delivery_status']='SENT'
            if not r.get('history_queued'):
                self.history.submit(r,'signal',r['created_at']);r['history_queued']=True
        self.save()
        self.history.start()
        self.report(now)
        if self.blocked():return
        for t in select(tickers,market):
            symbol=t['symbol']
            if any(r['symbol']==symbol and r['status']=='OPEN' for r in self.state['records']):continue
            try:
                rows=self.get_rows(symbol,'15m',64,now)
                hourly=self.get_rows(symbol,'1h',260,now)
                plan=setup(rows,hourly,float(t['lastPrice']),market[symbol],now)
                if not plan:continue
                key=f"FIB-{symbol}-{plan['a_time']}-{plan['b_time']}"
                if any(r['signal_id']==key for r in self.state['records']):continue
                message=(f"📐 FIBONACCI | {symbol} | LONG SİNYALİ\n"
                         f"24h değişim: %{float(t['priceChangePercent']):.2f}\n"
                         f"%50–%61,8 bölgesi: {plan['zone_low']:.8g}–{plan['zone_high']:.8g}\n"
                         f"Giriş referansı: {plan['entry']:.8g}\nSL: {plan['sl']:.8g}\n"
                         f"TP (önceki tepe): {plan['tp']:.8g}\nKazanç/risk: {plan['rr']:.2f}\n"
                         "1h trend uygun; kapalı 15m mumla yukarı dönüş teyidi.\n"
                         "Sanal takip; emir açılmadı. 5 günlük geçmiş ayrıca kaydedilir.")
                r=dict(plan,symbol=symbol,signal_id=key,created_at=now,status='OPEN',sent=False,message=message,
                       ticker=t,signal_15m=rows,trend_1h=hourly)
                self.state['records'].append(r);self.save()
                self.emit(dict(type='signal',record=r),now)
                if self.send(message):r['sent']=True;r['delivery_status']='SENT'
                self.history.submit(r,'signal',now);r['history_queued']=True;self.save()
            except Exception as error:
                self.log(f'[FIBONACCI ERR] {symbol}: {type(error).__name__}')
                if self.blocked():break
        self.cache={k:v for k,v in self.cache.items() if k[2]>=int(now//(900 if k[1]=='15m' else 3600))-1}

    def start(self):
        if self.thread and self.thread.is_alive():return
        def run():
            while True:
                try:self.scan()
                except Exception as error:self.log(f'[FIBONACCI LOOP ERR] {type(error).__name__}')
                time.sleep(180)
        self.thread=threading.Thread(target=run,name='fibonacci-signals',daemon=True)
        self.thread.start()
