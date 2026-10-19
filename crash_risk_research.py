"""Independent paper-risk observer. Never places orders or gates live signals.
Fees/slippage are sensitivity assumptions, not account rates. Funding and full
order-book impact are unknown. Fixed 50 USDT per entry, 1000 USDT reference
capital, unlimited hypothetical slots; leverage and liquidation are not modeled.
"""
import copy
import itertools
import json
import math
import os
import queue
import re
import random
import threading
import time

CONFIG = dict(version='shadow_risk_v1',notional=50.,capital=1000.,fee_bps_per_side=5.,
              slippage_bps_per_side=2.,horizon_seconds=5*86400,retention_days=20, enrollment_days=15,
              max_spread_bps=5.,min_top_depth_usdt=1000.,min_volume_24h_usdt=10000000.,
              funding_mode='NOT_MODELED',leverage_mode='NOT_MODELED',quote_max_age_seconds=5)


def number(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None


def combinations():
    for tp,sl,score,liquidity in itertools.product([1.,1.2,1.5],[None,2.,3.,5.],[0,30,50,70],['all','spread','liquid']):
        key=f'tp{tp}_sl{sl}_score{score}_{liquidity}'
        yield key,dict(tp=tp,sl=sl,min_score=score,liquidity=liquidity)


class RiskResearch:
    def __init__(self,directory,websocket_lib=None,log=print):
        self.directory=directory;os.makedirs(directory,exist_ok=True)
        self.log=log;self.websocket_lib=websocket_lib
        self.lock=threading.RLock();self.events=queue.Queue(maxsize=50000)
        self.quotes={};self.last_ids={};self.last_sample={};self.sockets={};self.subscriptions={}
        self.thread=None;self.dropped=0;self.watches={};self.desired=frozenset()
        self.path=os.path.join(directory,'state.json')
        if os.path.exists(self.path):
            with open(self.path,encoding='utf8') as f:self.state=json.load(f)
            if self.state['config']!=CONFIG:raise ValueError('Research config changed; use a new experiment directory')
            self.state['quality']['restarts']+=1
            for sc in self.state['scenarios'].values():
                for pos in sc['positions'].values():
                    if pos['status']=='OPEN':pos['coverage_incomplete']=True
        else:
            self.state=dict(config=CONFIG,started_at=time.time(),entries={},scenarios={},
                            quality=dict(restarts=0,disconnects=0,dropped=0,stale_events=0,errors=0))
            for key,cfg in combinations():
                self.state['scenarios'][key]=dict(config=cfg,positions={},realized=0.,peak=CONFIG['capital'],
                    max_drawdown_usdt=0.,max_drawdown_pct=0.,max_open=0,skipped=0,missing_filter_data=0,unknown_cost_exits=0)
        self.positions_by_symbol={}
        for name,scenario in self.state['scenarios'].items():
            active=[p for p in scenario['positions'].values() if p['status']=='OPEN']
            scenario['unrealized']=sum(p['mark_net'] for p in active)
            scenario['open_count']=len(active)
            index=self.positions_by_symbol[name]={}
            for key,pos in scenario['positions'].items():
                index.setdefault(pos['symbol'],{})[key]=pos
        # Quotes are deliberately not restored: a restart creates unknown coverage.

    def submit(self,kind,payload):
        try:self.events.put_nowait((kind,copy.deepcopy(payload)))
        except queue.Full:self.dropped+=1

    def emit(self,event):
        path=os.path.join(self.directory,time.strftime('%Y-%m-%d',time.gmtime())+'.jsonl')
        with open(path,'a',encoding='utf8') as f:f.write(json.dumps(event,allow_nan=False)+'\n')

    def fresh_quote(self,symbol,at):
        q=self.quotes.get(symbol)
        return q if q and 0<=at-q['at']<=CONFIG['quote_max_age_seconds'] else None

    def on_quote(self,d):
        at=number(d.get('T',d.get('E')));bid=number(d.get('b'));ask=number(d.get('a'))
        bq=number(d.get('B'));aq=number(d.get('A'))
        if None in (at,bid,ask,bq,aq) or not 0<bid<=ask or min(bq,aq)<0:return
        q=dict(at=at/1000,bid=bid,ask=ask,bid_depth=bid*bq,ask_depth=ask*aq,
               spread_bps=(ask-bid)/((ask+bid)/2)*10000,
               received_at=d.get('_received_at'), latency_ms=(d.get('_received_at',at/1000)-at/1000)*1000)
        symbol=d['s'];old=self.quotes.get(symbol)
        if old and q['at']<=old['at']:return
        self.quotes[symbol]=q
        if q['at']-self.last_sample.get(symbol,0)>=10:
            self.last_sample[symbol]=q['at'];self.emit(dict(type='quote',symbol=symbol,**q))

    def enter(self,r):
        key=r['signal_id'];symbol=r['symbol'];at=number(r.get('entered_at'))
        price=number(r.get('observed_entry_price',r.get('virtual_fill_price')))
        if key in self.state['entries'] or at is None or price is None or price<=0:return
        if not self.state['started_at']<=at<self.state['started_at']+CONFIG['enrollment_days']*86400:return
        q=self.fresh_quote(symbol,at);score=number((r.get('strength_score') or {}).get('score'))
        volume=number(((r.get('features') or {}).get('ticker_24h') or {}).get('quoteVolume'))
        rec=dict(signal_id=key,symbol=symbol,at=at,reference_price=price,score=score,volume_24h=volume,
                 quote=q)
        self.state['entries'][key]=rec;self.emit(dict(type='entry',record=rec))
        for name,scenario in self.state['scenarios'].items():
            cfg=scenario['config'];ok=(cfg['min_score']==0 or score is not None and score>=cfg['min_score'])
            if cfg['liquidity']!='all':ok=ok and q is not None and q['spread_bps']<=CONFIG['max_spread_bps']
            if cfg['liquidity']=='liquid':ok=ok and volume is not None and volume>=CONFIG['min_volume_24h_usdt'] and min(q['bid_depth'],q['ask_depth'])>=CONFIG['min_top_depth_usdt']
            if not ok:
                scenario['skipped']+=1
                scenario['missing_filter_data']+=int((cfg['min_score']>0 and score is None) or
                    (cfg['liquidity']!='all' and q is None) or (cfg['liquidity']=='liquid' and volume is None))
                continue
            fill=(q['ask'] if q else price)*(1+CONFIG['slippage_bps_per_side']/10000)
            pos=dict(symbol=symbol,at=at,reference_price=price,fill=fill,qty=CONFIG['notional']/fill,
                     entry_quote_known=q is not None,status='OPEN',last_price=price,
                     coverage_incomplete=False,sl_then_target=False,mae_pct=0.,mfe_pct=0.,
                     mark_net=0.)
            liquidation=(q['bid'] if q else price)*(1-CONFIG['slippage_bps_per_side']/10000)
            pos['mark_net']=pos['qty']*(liquidation-fill)-(CONFIG['notional']+pos['qty']*liquidation)*CONFIG['fee_bps_per_side']/10000
            scenario['positions'][key]=pos
            self.positions_by_symbol[name].setdefault(symbol,{})[key]=pos
            scenario['unrealized']+=pos['mark_net'];scenario['open_count']+=1
        self.mark(at)

    def mark(self,at):
        for s in self.state['scenarios'].values():
            equity=CONFIG['capital']+s['realized']+s['unrealized']
            s['equity']=equity;s['max_open']=max(s['max_open'],s['open_count'])
            s['peak']=max(s['peak'],equity)
            s['max_drawdown_usdt']=max(s['max_drawdown_usdt'],s['peak']-equity)
            s['max_drawdown_pct']=max(s['max_drawdown_pct'],100*(s['peak']-equity)/s['peak'])
            s['required_peak_notional']=s['max_open']*CONFIG['notional']
            s['marked_at']=at

    def trade(self,d):
        symbol=d['s'];at=float(d['T'])/1000;price=number(d['p']);tid=int(d['a'])
        if price is None or price<=0 or tid<=self.last_ids.get(symbol,-1):return
        self.last_ids[symbol]=tid;q=self.fresh_quote(symbol,at)
        exit_price=(q['bid'] if q else price)*(1-CONFIG['slippage_bps_per_side']/10000)
        transitions=[]
        for name,s in self.state['scenarios'].items():
            cfg=s['config']
            for key,p in self.positions_by_symbol[name].get(symbol,{}).items():
                if (p['status']=='SL' and p['exit_at']<at<=p['at']+CONFIG['horizon_seconds']
                        and price>=p['reference_price']*(1+cfg['tp']/100)):
                    p['sl_then_target']=True
                if p['status']!='OPEN' or at<=p.get('last_at',p['at']):continue
                if at-p.get('last_at',p['at'])>30:p['coverage_incomplete']=True
                p['last_at']=at;p['last_price']=price
                gross=p['qty']*(exit_price-p['fill'])
                fees=(CONFIG['notional']+p['qty']*exit_price)*CONFIG['fee_bps_per_side']/10000
                net=gross-fees
                p.update(mark_gross=gross,mark_fees=fees,exit_quote=q,
                    price_proxy_gross=CONFIG['notional']*(price/p['reference_price']-1),
                    fill_drag_usdt=CONFIG['notional']*(price/p['reference_price']-1)-gross)
                s['unrealized']+=net-p['mark_net']
                p['mark_net']=net
                ret=(price/p['reference_price']-1)*100
                p['mae_pct']=min(p['mae_pct'],ret);p['mfe_pct']=max(p['mfe_pct'],ret)
                reason=('TIMEOUT' if at-p['at']>=CONFIG['horizon_seconds'] else
                        'SL' if cfg['sl'] is not None and price<=p['reference_price']*(1-cfg['sl']/100) else
                        'TP' if price>=p['reference_price']*(1+cfg['tp']/100) else None)
                if reason:
                    p.update(status=reason,exit_at=at,exit_price=exit_price,net_usdt=net,exit_quote_known=q is not None)
                    s['realized']+=net;s['unrealized']-=net;s['open_count']-=1
                    s['unknown_cost_exits']+=int(q is None or not p['entry_quote_known'])
                    transitions.append(dict(scenario=name,signal_id=key,status=reason,net_usdt=net))
        self.mark(at)
        if transitions:self.emit(dict(type='exits',at=at,price=price,transitions=transitions))

    def summary(self):
        now=time.time();end=self.state['started_at']+CONFIG['enrollment_days']*86400
        result=dict(config=CONFIG,quality=self.state['quality'],started_at=self.state['started_at'],
                    enrollment_ends_at=end,followup_ends_at=end+CONFIG['horizon_seconds'],
                    enrollment_complete=now>=end,followup_complete=now>=end+CONFIG['horizon_seconds'],
                    caveat='Observed paper results; missing quotes use trade-price proxy. Fees/slippage assumed; funding, liquidation and full depth impact excluded. Gaps can miss thresholds; open/stale marks are not final results.',scenarios={})
        for k,s in self.state['scenarios'].items():
            values=list(s['positions'].items());out={a:b for a,b in s.items() if a!='positions'}
            out['counts']={status:sum(p['status']==status for _,p in values) for status in ['OPEN','TP','SL','TIMEOUT']}
            out['sl_then_target']=sum(p.get('sl_then_target',False) for _,p in values)
            out['incomplete_coverage_positions']=sum(p.get('coverage_incomplete',False) for _,p in values)
            out['entry_quote_missing']=sum(not p['entry_quote_known'] for _,p in values)
            out['reference_capital_exceeded']=s.get('required_peak_notional',0)>CONFIG['capital']
            out['modeled_fees_usdt']=sum(p.get('mark_fees',0) for _,p in values)
            out['modeled_fill_drag_usdt']=sum(p.get('fill_drag_usdt',0) for _,p in values)
            out['closed_net_usdt']=s['realized']
            closed=[p for _,p in values if p['status']!='OPEN']
            out['mean_closed_net_usdt']=s['realized']/len(closed) if closed else None
            out['observed_tp_rate_closed_pct']=100*sum(p['status']=='TP' for p in closed)/len(closed) if closed else None
            out['worst_position_net_usdt']=min((p['mark_net'] for _,p in values),default=None)
            out['open_net_usdt']=sum(p['mark_net'] for _,p in values if p['status']=='OPEN')
            out['net_marked_usdt']=s.get('equity',CONFIG['capital'])-CONFIG['capital']
            out['net_return_reference_capital_pct']=out['net_marked_usdt']/CONFIG['capital']*100
            out['stale_open_marks']=sum(p['status']=='OPEN' and time.time()-p.get('last_at',p['at'])>60 for _,p in values)
            result['scenarios'][k]=out
        return result

    def checkpoint(self):
        if self.dropped:
            for sc in self.state['scenarios'].values():
                for pos in sc['positions'].values():
                    if pos['status']=='OPEN':pos['coverage_incomplete']=True
        self.state['quality']['dropped']=self.dropped+self.state['quality'].get('dropped',0);self.dropped=0
        self.state['quality']['queue_size']=self.events.qsize()
        self.state['quality']['last_checkpoint_at']=time.time()
        self.state['quality']['connected_channels']=sorted(self.sockets)
        self.state['quality']['fresh_quote_symbols']=sum(0<=time.time()-q['at']<=5 for q in self.quotes.values())
        with self.lock:
            for name,data in [('state.json',self.state),('summary.json',self.summary())]:
                path=os.path.join(self.directory,name)
                with open(path+'.tmp','w',encoding='utf8') as f:json.dump(data,f,allow_nan=False);f.flush();os.fsync(f.fileno())
                os.replace(path+'.tmp',path)
        for name in os.listdir(self.directory):
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}\.jsonl',name):
                path=os.path.join(self.directory,name)
                if time.time()-os.path.getmtime(path)>CONFIG['retention_days']*86400:os.remove(path)

    def wanted(self):
        now=time.time()
        self.watches={s:t for s,t in self.watches.items() if t>now}
        active={p['symbol'] for sc in self.state['scenarios'].values() for p in sc['positions'].values() if p['status']=='OPEN'}
        return active | set(self.watches) | {r['symbol'] for r in self.state['entries'].values() if now-r['at']<=CONFIG['horizon_seconds']}

    def sync(self):
        # Routed public quotes and market aggTrades require separate sockets.
        with self.lock:
            for channel, suffix in [('public','@bookTicker'),('market','@aggTrade')]:
                ws=self.sockets.get(channel)
                if ws is None:continue
                desired=self.desired
                previous=self.subscriptions.get(channel,set())
                for method,syms in [('SUBSCRIBE',desired-previous),('UNSUBSCRIBE',previous-desired)]:
                    if syms:ws.send(json.dumps(dict(method=method,
                        params=[symbol.lower()+suffix for symbol in sorted(syms)],id=int(time.time()*1000))))
                self.subscriptions[channel]=desired

    def start(self):
        if self.thread is not None:return
        def consume():
            next_save=0
            while True:
                try:
                    kind,payload=self.events.get(timeout=1)
                    if kind=='entry':self.enter(payload)
                    elif kind=='watch':
                        if time.time()<self.state['started_at']+CONFIG['enrollment_days']*86400:
                            self.watches[payload['symbol']]=payload['until']
                    elif kind=='stale':self.state['quality']['stale_events']+=1
                    elif kind=='quote':self.on_quote(payload)
                    elif kind=='trade':self.trade(payload)
                    elif kind=='connection':
                        self.quotes.clear()
                        for sc in self.state['scenarios'].values():
                            for pos in sc['positions'].values():
                                if pos['status']=='OPEN':pos['coverage_incomplete']=True
                        if payload['status']=='closed':self.state['quality']['disconnects']+=1
                        self.emit(dict(type='connection',**payload))
                except queue.Empty:pass
                except Exception as e:self.state['quality']['errors']+=1;self.log(f'[CRASH RESEARCH ERR] {type(e).__name__}: {str(e)[:160]}')
                if time.time()>=next_save:
                    try:
                        wanted=self.wanted()
                        if len(wanted)>1000:
                            raise ValueError('Research symbol capacity exceeded (1000); coverage incomplete')
                        self.desired=frozenset(wanted);self.checkpoint();self.sync()
                    except Exception as e:self.log(f'[CRASH RESEARCH SAVE ERR] {type(e).__name__}: {str(e)[:160]}')
                    next_save=time.time()+15
        self.thread=threading.Thread(target=consume,daemon=True,name='crash-risk-research');self.thread.start()
        def socket_loop(channel):
            delay=5
            while True:
                def message(ws,raw):
                    try:
                        d=json.loads(raw);d=d.get('data',d)
                        if 'code' in d:
                            self.log(f'[CRASH RESEARCH SUBSCRIBE ERR] channel={channel} code={d.get("code")} msg={str(d.get("msg"))[:160]}')
                        kind={'bookTicker':'quote','aggTrade':'trade'}.get(d.get('e'))
                        if kind:
                            at=number(d.get('T',d.get('E')))
                            if at is not None and 0<=time.time()-at/1000<=5:
                                d['_received_at']=time.time();self.submit(kind,d)
                            else:self.submit('stale',{})
                    except (ValueError,TypeError):pass
                def opened(ws):
                    with self.lock:
                        self.subscriptions[channel]=set();self.sockets[channel]=ws
                    self.submit('connection',dict(status='open',channel=channel,at=time.time()))
                def closed(ws,code,msg):
                    self.log(f'[CRASH RESEARCH WS CLOSED] channel={channel} code={code} reason={str(msg)[:160]}')
                try:
                    app=self.websocket_lib.WebSocketApp(f'wss://fstream.binance.com/{channel}/ws',on_message=message,on_open=opened,on_close=closed,
                        on_error=lambda ws,e:self.log(f'[CRASH RESEARCH WS ERR] {type(e).__name__}: {str(e)[:160]}'))
                    began=time.time();app.run_forever(ping_interval=60,ping_timeout=20)
                    if time.time()-began>120:delay=5
                except Exception as e:self.log(f'[CRASH RESEARCH WS ERR] {type(e).__name__}: {str(e)[:160]}')
                finally:
                    with self.lock:self.sockets.pop(channel,None)
                    self.submit('connection',dict(status='closed',channel=channel,at=time.time(),reason='reconnect',retry_seconds=delay))
                time.sleep(delay+random.random());delay=min(60,delay*2)
        if self.websocket_lib:
            for channel in ('public','market'):
                threading.Thread(target=socket_loop,args=(channel,),daemon=True,name='crash-risk-'+channel).start()
