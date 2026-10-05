"""Offline WS/cache/reconnect/bootstrap tests, no live connections."""
import ast
import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class WebSocketTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        functions={'_crash_write_json','_crash_get_candles','_crash_cached_candles_valid','_crash_bounce_worker'}
        subset=ast.Module(body=[n for n in tree.body if
            (isinstance(n,ast.ClassDef) and n.name=='CrashBounceWebSocketCache') or
            (isinstance(n,ast.FunctionDef) and n.name in functions)],type_ignores=[])
        self.now=2_000_000_000.
        self.g=dict(time=SimpleNamespace(time=lambda:self.now,time_ns=lambda:int(self.now*1e9)),
            json=json,math=math,os=os,threading=threading,SAVE_LOCK=threading.Lock(),log=Mock(),
            CRASH_BOUNCE_CANDLES_FILE=self.tmp.name+'/candles.json',
            CRASH_BOUNCE_MARKET_FILE=self.tmp.name+'/market.json',
            _crash_cooldown_until=0.,_crash_seed_retry={},_crash_seed_attempts=[],
            _crash_public_get=Mock(),BinanceRateLimiter=SimpleNamespace(is_banned=lambda:False))
        exec(compile(subset,'ema.py','exec'),self.g)
        self.cache=self.g['CrashBounceWebSocketCache']()
        self.g['_CRASH_WS_CACHE']=self.cache
        self.cache.connected=True; self.cache.opened_at=self.now-30
        self.rows=[[int((self.now-(16-i)*900)*1000),'1','1.1','.9','1','1',
                    int((self.now-(15-i)*900)*1000)-1] for i in range(16)]

    def event(self,symbol='AUSDT',change='-12',st=1,offset=0):
        return dict(s=symbol,c='1',P=change,E=int((self.now+offset)*1000),st=st)

    def test_delta_merge_um_filter_invalid_and_freshness(self):
        self.cache.on_message(None,json.dumps({'data':[self.event()]}))
        self.cache.on_message(None,json.dumps([self.event('BUSDT'),self.event('CMUSDT',st=2),
                                               self.event('BADUSDT',change='NaN')]))
        self.assertEqual({r['symbol'] for r in self.cache.snapshot(self.now)},{'AUSDT','BUSDT'})
        self.cache.on_message(None,json.dumps([self.event(change='-99',offset=-1)]))
        self.assertEqual(next(r for r in self.cache.snapshot(self.now) if r['symbol']=='AUSDT')['priceChangePercent'],'-12')
        self.now+=121
        self.assertEqual(self.cache.snapshot(self.now),[])

    def test_closed_kline_only_dedup_and_cache_persistence(self):
        r=self.rows[-1]
        k=dict(i='15m',x=False,t=r[0],o=r[1],h=r[2],l=r[3],c=r[4],v=r[5],T=r[6])
        self.cache.on_message(None,json.dumps({'e':'kline','s':'AUSDT','k':k}))
        self.assertEqual(self.cache.candles('AUSDT'),[])
        k['x']=True
        for _ in range(2): self.cache.on_message(None,json.dumps({'e':'kline','s':'AUSDT','k':k}))
        self.assertEqual(len(self.cache.candles('AUSDT')),1)
        self.cache.flush()
        second=self.g['CrashBounceWebSocketCache'](); second.load()
        self.assertEqual(second.candles('AUSDT'),self.cache.candles('AUSDT'))

    def test_warmup_disconnect_clear_and_restore_subscriptions(self):
        self.cache.reconcile({'AUSDT'})
        ws=Mock(); self.cache.on_open(ws)
        message=json.loads(ws.send.call_args.args[0])
        self.assertEqual(message['params'],['ausdt@kline_15m'])
        self.cache.on_message(None,json.dumps([self.event()]))
        self.assertEqual(self.cache.snapshot(self.now),[])
        self.now+=20
        self.assertEqual(len(self.cache.snapshot(self.now)),1)
        self.cache.on_close(ws,1006,None)
        self.assertEqual(self.cache.snapshot(self.now),[])
        self.cache.on_open(ws)
        self.now+=20
        self.assertEqual(self.cache.snapshot(self.now),[])
        self.assertIn('/market/',self.cache.URL)

    def test_subscription_diff_not_repeated_each_scan(self):
        self.cache.ws=Mock()
        self.cache.reconcile({'AUSDT','BUSDT'})
        self.cache.reconcile({'AUSDT','BUSDT'})
        self.assertEqual(self.cache.ws.send.call_count,1)
        self.cache.reconcile({'BUSDT','CUSDT'})
        messages=[json.loads(c.args[0]) for c in self.cache.ws.send.call_args_list]
        self.assertEqual(messages[-2]['method'],'UNSUBSCRIBE')
        self.assertEqual(messages[-1]['params'],['cusdt@kline_15m'])

    def test_seed_once_then_only_cache_no_ticker_rest(self):
        self.g['_crash_public_get'].return_value=self.rows
        one=self.g['_crash_get_candles']('AUSDT')
        two=self.g['_crash_get_candles']('AUSDT')
        self.assertEqual(one,two)
        self.g['_crash_public_get'].assert_called_once()
        self.assertEqual(self.g['_crash_public_get'].call_args.args[0],'/fapi/v1/klines')

    def test_rest_ban_does_not_block_cached_candles(self):
        self.cache.merge_candles('AUSDT',self.rows)
        self.g['_crash_cooldown_until']=self.now+3600
        self.assertEqual(self.g['_crash_get_candles']('AUSDT'),self.rows)
        self.assertEqual(self.g['_crash_get_candles']('BUSDT'),[])
        self.g['_crash_public_get'].assert_not_called()

    def test_failed_seed_backoff_budget_and_gap_reseed(self):
        self.g['_crash_public_get'].side_effect=RuntimeError('429')
        self.g['_crash_get_candles']('AUSDT'); self.g['_crash_get_candles']('AUSDT')
        self.assertEqual(self.g['_crash_public_get'].call_count,1)
        self.g['_crash_seed_attempts'][:]=[self.now]*24
        self.g['_crash_get_candles']('BUSDT')
        self.assertEqual(self.g['_crash_public_get'].call_count,1)
        damaged=[list(r) for r in self.rows]; damaged[5][0]+=1
        self.assertFalse(self.g['_crash_cached_candles_valid'](damaged,self.now))

    def test_worker_reports_and_scans_ws_during_rest_cooldown(self):
        self.g['time'].sleep=Mock(side_effect=StopIteration)
        self.g.update(_crash_load_market_cache=lambda:({'AUSDT':'.001'},self.now),
            _crash_load_state=lambda:{'history':[],'symbols':{}},_crash_save_state=Mock(),
            _crash_daily_report=Mock(),run_crash_bounce_alerts=Mock(),
            _crash_track_results=Mock(),CRASH_BOUNCE_SCAN_SECONDS=180,
            _crash_cooldown_until=self.now+86400)
        self.cache.start=Mock(); self.cache.flush=Mock()
        with self.assertRaises(StopIteration): self.g['_crash_bounce_worker']()
        self.g['_crash_daily_report'].assert_called_once()
        self.g['run_crash_bounce_alerts'].assert_called_once()
        self.g['_crash_public_get'].assert_not_called()


if __name__=='__main__': unittest.main()
