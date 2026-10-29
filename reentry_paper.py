"""Isolated 4h/5m paper re-entry. Public candles only; no broker/order adapter."""
import json
import math
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

FIVE = 300_000
FOUR = 14_400_000
MAX_AGE = 150 * 3600


def closed_rows(raw, interval, now):
    rows = []
    for r in raw:
        if len(r) < 7 or int(r[6]) >= now*1000:
            continue
        t, end = int(r[0]), int(r[6])
        o, h, l, c, v = map(float, r[1:6])
        if (t % interval or end != t+interval-1 or not all(math.isfinite(x) for x in (o,h,l,c,v))
                or min(o,h,l,c) <= 0 or v < 0 or h < max(o,c) or l > min(o,c)):
            raise ValueError('INVALID_CANDLES')
        rows.append([t,o,h,l,c,v,end])
    if not rows or any(b[0]-a[0] != interval for a,b in zip(rows, rows[1:])):
        raise ValueError('MISSING_CANDLES')
    return rows


def setup(rows5, rows4):
    """Only newest closed 5m confirmation; freeze 4h zone BEFORE break candle."""
    if len(rows5) < 10:
        return None
    confirm = rows5[-1]
    for i in range(len(rows5)-2, max(-1,len(rows5)-5), -1):
        br = rows5[i]
        context = [r for r in rows4 if r[6] < br[0]]
        if len(context) < 25 or br[0]-context[-1][6] > FOUR:
            continue
        closes = [r[4] for r in context]
        e = closes[0]; emas = [e]
        for c in closes[1:]:
            e = c*(2/21)+e*(19/21); emas.append(e)
        up = sum(c>e for c,e in zip(closes[-5:],emas[-5:])) >= 4
        down = sum(c<e for c,e in zip(closes[-5:],emas[-5:])) >= 4
        lo, hi = context[-1][3], context[-1][2]
        def strong(r):
            return r[2]>r[3] and abs(r[4]-r[1])/(r[2]-r[3]) >= .6
        if hi <= lo or not strong(br) or not strong(confirm):
            continue
        if not lo < confirm[4] < hi:
            continue
        direction = None
        if up and br[1]>lo>br[4] and confirm[4]>confirm[1]:
            direction = 'LONG'
        elif down and br[1]<hi<br[4] and confirm[4]<confirm[1]:
            direction = 'SHORT'
        if direction is None:
            continue
        # Earlier confirming candles would already have triggered this break; do not re-signal it.
        if any(strong(r) and lo<r[4]<hi and ((direction=='LONG' and r[4]>r[1]) or
               (direction=='SHORT' and r[4]<r[1])) for r in rows5[i+1:-1]):
            continue
        entry = confirm[4]
        return dict(direction=direction, entry=entry,
                    tp=entry*(1.01 if direction=='LONG' else .99),
                    zone_low=lo, zone_high=hi, break_at=br[0], signal_at=confirm[6],
                    context_at=context[-1][6])
    return None


def advance(trade, rows, now):
    if trade['status'] != 'OPEN':
        return
    after = [r for r in rows if r[0] > trade['last_bar']]
    if after and after[0][0] != trade['last_bar']+FIVE:
        trade.update(status='DATA_GAP', closed_at=now, reason='Eksik mumlar nedeniyle sonuç belirlenemedi')
        return
    for r in after:
        # Timeout evaluated chronologically, never credit TP after tracking horizon.
        if r[0]/1000 >= trade['entered_at']+MAX_AGE:
            trade.update(status='TIMEOUT', closed_at=r[0]/1000)
            return
        long = trade['direction']=='LONG'
        tp = r[2]>=trade['tp'] if long else r[3]<=trade['tp']
        adverse = max(0, (trade['entry']-r[3])/trade['entry']*100) if long else max(0, (r[2]-trade['entry'])/trade['entry']*100)
        trade['max_adverse_pct'] = max(trade.get('max_adverse_pct',0), adverse)
        trade['last_bar'] = r[0]
        trade['last_price'] = r[4]
        if tp:
            trade.update(status='TP', closed_at=r[6]/1000)
            return


class ReentryPaper:
    def __init__(self, directory, fetch, universe, deliver, log, cooldown=lambda:0, clock=time.time):
        self.fetch, self.universe, self.deliver, self.log = fetch, universe, deliver, log
        self.cooldown, self.clock = cooldown, clock
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory,'reentry_paper_state.json')
        self.lock = threading.RLock()
        self.state = {'trades':{}, 'outbox':[], 'metrics':{}}
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                self.state = json.load(f)
            if not isinstance(self.state.get('trades'),dict) or not isinstance(self.state.get('outbox'),list):
                raise ValueError('INVALID_REENTRY_STATE')
        self.cache4 = {}
        self.next_scan = 0
        self.thread = None
        self.started = clock()

    def save(self):
        with open(self.path+'.tmp','w',encoding='utf-8') as f:
            json.dump(self.state,f,ensure_ascii=False,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(self.path+'.tmp',self.path)

    def notify(self, text):
        self.state['outbox'].append({'text':text,'retry_at':0})

    def summary(self):
        with self.lock:
            m = dict(self.state.get('metrics',{}))
            counts = {s:sum(t['status']==s for t in self.state['trades'].values())
                      for s in ('OPEN','TP','TIMEOUT','DATA_GAP')}
        at=m.get('completed_at',0)
        stamp=datetime.fromtimestamp(at,ZoneInfo('Europe/Paris')).strftime('%d/%m %H:%M') if at else 'henüz yok'
        fresh = at>=self.started and self.clock()-at<=660
        return (f"RE-ENTRY 4H/5m — SANAL: {m.get('status','başlangıç')} | tarama {'güncel' if fresh else 'doğrulanamadı'}\n"
                f"Son tarama: {stamp} | İncelenen: {m.get('checked',0)} | Veri bekleyen/hata: {m.get('errors',0)}\n"
                f"Açık {counts['OPEN']} | TP {counts['TP']} | SL yok | "
                f"Süre doldu {counts['TIMEOUT']} | Veri boşluğu {counts['DATA_GAP']}\n"
                f"Son sinyal: {m.get('last_signal','henüz yok')} | Telegram: {m.get('delivery','henüz yok')}\n"
                'Sanal TP %1; SL yok; takip 150 saat. Komisyon/kayma dahil değil; gerçek emir yok.')

    def _deliver(self, now):
        if not self.state['outbox'] or now<self.state['outbox'][0]['retry_at']:
            return
        item=self.state['outbox'][0]
        item['retry_at']=now+300;self.save()
        receipt=self.deliver(item['text'])
        if receipt.get('ok') or receipt.get('ambiguous'):
            self.state['outbox'].pop(0)
        self.state.setdefault('metrics',{})['delivery']='başarılı' if receipt.get('ok') else 'teslim doğrulanamadı'
        self.save()

    def step(self):
        now=self.clock()
        # Keep the reporting snapshot responsive while network requests are in progress.
        with self.lock:
            self._deliver(now)
        if now<self.next_scan:
            return
        self.next_scan=(int(now//300)+1)*300+5
        m=dict(self.state.get('metrics',{}),checked=0,errors=0,status='taranıyor')
        if now<self.cooldown():
            m['status']='API beklemesi'
        else:
            tickers=self.universe(now)
            ranked=[]
            for t in tickers:
                try:
                    v=float(t.get('quoteVolume',0))
                    if math.isfinite(v) and v>0:
                        ranked.append((v,t['symbol']))
                except (ValueError,KeyError,TypeError):
                    continue
            selected=[s for _,s in sorted(ranked,reverse=True)[:5]]
            opened={t['symbol'] for t in self.state['trades'].values() if t['status']=='OPEN'}
            watch=sorted(opened | set(selected))
            if not selected:
                m['status']='Güncel WS / piyasa verisi bekleniyor'
            else:
                m['status']='aktif'
            for symbol in watch:
                if self.clock()<self.cooldown():
                    m['status']='API beklemesi';break
                try:
                    raw=self.fetch('/fapi/v1/klines',{'symbol':symbol,'interval':'5m','limit':100})
                    rows=closed_rows(raw,FIVE,self.clock())
                    if len(rows)<10 or self.clock()*1000-rows[-1][6]>360000:
                        raise ValueError('STALE_5M')
                    with self.lock:
                        for t in self.state['trades'].values():
                            if t['symbol']!=symbol or t['status']!='OPEN':continue
                            advance(t,rows,self.clock())
                            if t['status']!='OPEN':
                                self.notify(f'📋 RE-ENTRY SANAL SONUÇ | {symbol} | {t["direction"]}\nSonuç: {t["status"]}\nGiriş {t["entry"]:.8g} | TP {t["tp"]:.8g} | En büyük ters hareket %{t.get("max_adverse_pct",0):.2f}\n5m mum takibi; SL ve gerçek emir yok. Ters hareket ölçümü TP mumunun tamamını kapsayabilir.')
                        self.save()
                    m['checked']+=1
                    if symbol not in selected or any(t['symbol']==symbol and t['status']=='OPEN' for t in self.state['trades'].values()):continue
                    if sum(t['status']=='OPEN' for t in self.state['trades'].values())>=5:continue
                    bucket=int(self.clock()*1000//FOUR)
                    if symbol not in self.cache4 or self.cache4[symbol][0]!=bucket:
                        raw4=self.fetch('/fapi/v1/klines',{'symbol':symbol,'interval':'4h','limit':60})
                        self.cache4[symbol]=(bucket,closed_rows(raw4,FOUR,self.clock()))
                    context=self.cache4[symbol][1]
                    if len(context)<25 or self.clock()*1000-context[-1][6]>FOUR:
                        raise ValueError('INSUFFICIENT_OR_STALE_4H')
                    signal=setup(rows,context)
                    if signal is None:continue
                    sid=f'{symbol}:{signal["direction"]}:{signal["break_at"]}'
                    if sid in self.state['trades']:continue
                    with self.lock:
                        self.state['trades'][sid]=dict(signal,symbol=symbol,status='OPEN',
                            entered_at=(signal['signal_at']+1)/1000,last_bar=rows[-1][0],
                            model='closed_5m_signal_price_paper_v1')
                        m['last_signal']=symbol+' '+signal['direction']
                        self.notify(f'🧪 RE-ENTRY 4H/5m | SANAL {signal["direction"]} | {symbol}\n'
                            f'Giriş varsayımı: {signal["entry"]:.8g}\nTP %1: {signal["tp"]:.8g} | SL yok\n'
                            f'4h bölge: {signal["zone_low"]:.8g}–{signal["zone_high"]:.8g}\n'
                            '5m kapalı mumla bölgeye dönüş. Gerçek emir gönderilmedi; fiyat ve sonuçlar sanaldır.')
                        self.save()
                except Exception as error:
                    m['errors']+=1
                    self.log(f'[REENTRY PAPER DATA] {symbol} {type(error).__name__}')
            if watch and m['checked']==len(watch) and not m['errors'] and m['status']=='aktif':
                m['completed_at']=self.clock()
        with self.lock:
            self.state['metrics']=m;self.save()

    def start(self):
        if self.thread and self.thread.is_alive():return
        def run():
            while True:
                try:self.step()
                except Exception as error:
                    with self.lock:
                        self.state.setdefault('metrics',{}).update(status='çalışma hatası',errors=1)
                    self.log(f'[REENTRY PAPER ERR] {type(error).__name__}')
                time.sleep(5)
        self.thread=threading.Thread(target=run,name='reentry-paper',daemon=True)
        self.thread.start()
