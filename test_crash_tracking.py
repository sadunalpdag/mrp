"""Offline tracking, research exports and delivery retry regression tests."""
import ast
import json
import math
import os
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        names={'_crash_initialize_tracking','_crash_track_results','_crash_summary',
               '_crash_features','_crash_daily_report','_crash_send_report_file'}
        subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        self.g=dict(math=math,os=os,json=json,datetime=datetime,timezone=timezone,
                    CRASH_BOUNCE_RESEARCH_DIR=self.tmp.name,_crash_save_state=Mock(),
                    _crash_append_data=Mock(),_crash_send_report_file=Mock(return_value=True),
                    log=Mock(),BOT_TOKEN='test',CHAT_ID='test',requests=Mock())
        exec(compile(subset,'ema.py','exec'),self.g)
        self.g['_crash_send_report_file']=Mock(return_value=True)
        self.now=2_000_000_000
        self.r=dict(symbol='AUSDT',created_at=self.now,entry='1',tp='1.01')
        self.state={'history':[self.r],'symbols':{}}

    def sample(self,offset,price):
        self.g['_crash_track_results'](self.state,[{'symbol':'AUSDT','lastPrice':str(price)}],self.now+offset)

    def test_entry_then_later_tp_and_no_same_snapshot_tp(self):
        self.sample(180,.99)
        self.assertEqual(self.r['tracking_status'],'OPEN')
        self.sample(360,1.02)
        self.assertEqual(self.r['tracking_status'],'TP')
        self.assertEqual(self.r['tp_hours'],.05)
        self.assertAlmostEqual(self.r['observed_return_pct'],1)
        self.assertEqual(self.g['_crash_summary']([self.r])['observed_tp_rate_resolved_pct'],100)

    def test_research_continues_after_tp_without_changing_outcome(self):
        self.sample(180,.99); self.sample(360,1.02)
        self.g['_crash_append_data'].reset_mock()
        self.sample(540,1.1)
        self.assertEqual(self.r['tracking_status'],'TP')
        samples=self.g['_crash_append_data'].call_args.args[0]['samples']
        self.assertTrue(samples[0]['after_result'])
        self.assertEqual(samples[0]['price'],1.1)

    def test_no_signal_time_or_above_entry_fill(self):
        self.sample(0,.9); self.sample(180,1.02)
        self.assertEqual(self.r['tracking_status'],'WAIT_ENTRY')
        self.sample(72*3600+1,.8)
        self.assertEqual(self.r['tracking_status'],'NO_ENTRY_OBSERVED')
        self.assertIsNone(self.g['_crash_summary']([self.r])['observed_tp_rate_resolved_pct'])

    def test_timeout_is_not_retroactive_tp_after_deadline(self):
        self.sample(180,.99)
        self.sample(180+72*3600+1,1.05)
        self.assertEqual(self.r['tracking_status'],'TIMEOUT')
        self.assertGreater(self.r['coverage_gaps'],0)
        self.assertEqual(self.r['observed_exit_price'],.99)

    def test_stale_missing_prices_and_gap_markers(self):
        self.g['_crash_track_results'](self.state,[{'symbol':'AUSDT','lastPrice':'.99',
            'closeTime':(self.now-1000)*1000}],self.now+180)
        self.assertEqual(self.r['samples'],0)
        self.sample(1000,.99)
        self.assertEqual(self.r['coverage_gaps'],1)

    def test_features_exclude_unclosed_candle(self):
        rows=[[int((self.now-(64-i)*900)*1000),'1','1.2','.9','1','100',
               int((self.now-(63-i)*900)*1000)-1] for i in range(64)]
        rows.append([int(self.now*1000),'1','99','.1','90','999999',int((self.now+900)*1000)])
        f=self.g['_crash_features'](rows,{},self.now)
        self.assertEqual(len(f['closed_15m_ohlcv']),64)
        self.assertEqual(f['ema7'],1)
        self.assertEqual(f['ema25'],1)
        self.assertEqual(f['volume_ratio_last_vs_previous20'],1)
        self.assertEqual(f['rsi14_wilder'],50)

    def test_report_first_due_in_24h_and_failed_send_reuses_export(self):
        self.g['_crash_daily_report'](self.state,self.now)
        self.g['_crash_send_report_file'].assert_not_called()
        self.g['_crash_send_report_file'].return_value=False
        due=self.now+86400
        self.g['_crash_daily_report'](self.state,due)
        pending=dict(self.state['report']['pending'])
        with zipfile.ZipFile(pending['path']) as z:
            self.assertIn('signals_and_outcomes.json',z.namelist())
            self.assertIn('report.json',z.namelist())
        self.assertEqual(self.state['report']['next_due_at'],due)
        self.g['_crash_daily_report'](self.state,due+60)
        self.assertEqual(self.g['_crash_send_report_file'].call_count,1)
        self.g['_crash_send_report_file'].return_value=True
        self.g['_crash_daily_report'](self.state,due+1800)
        self.assertNotIn('pending',self.state['report'])
        self.assertEqual(self.state['report']['next_due_at'],due+1800+86400)
        self.assertEqual(self.g['_crash_send_report_file'].call_args.args[0],pending['path'])

    def test_scan_tracking_integration_reuses_bulk_ticker_without_tracking_calls(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        names={'run_crash_bounce_alerts','crash_bounce_plan'}
        subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        self.g.update(Decimal=Decimal,time=SimpleNamespace(time=lambda:self.now),
            CRASH_BOUNCE_ENTRY_BUFFER=Decimal('.003'),CRASH_BOUNCE_TP_PCT=Decimal('.01'),
            CRASH_BOUNCE_TOP_LOSERS=20,CRASH_BOUNCE_DROP_PCT=12,
            CRASH_BOUNCE_COOLDOWN_SECONDS=86400,_crash_send_pending=Mock(),
            BinanceRateLimiter=SimpleNamespace(is_banned=lambda:False))
        exec(compile(subset,'ema.py','exec'),self.g)
        rows=[[int((self.now-(64-i)*900)*1000),'1','1.2','.9','1','100',
               int((self.now-(63-i)*900)*1000)-1] for i in range(64)]
        state={'symbols':{},'history':[]}; calls=[]
        price=['1']
        def get(path,params=None):
            calls.append(path)
            return rows if 'klines' in path else [dict(symbol='AUSDT',priceChangePercent='-20',lastPrice=price[0])]
        self.g['_CRASH_WS_CACHE']=SimpleNamespace(snapshot=lambda now:get('WS_SNAPSHOT'),reconcile=Mock())
        self.g['_crash_get_candles']=lambda symbol:get('/fapi/v1/klines')
        for elapsed,value in [(0,'1'),(180,'.9'),(360,'.92')]:
            price[0]=value
            self.g['run_crash_bounce_alerts'](state,{'AUSDT':'.001'},self.now+elapsed)
        self.assertEqual(len(state['history']),1)
        self.assertEqual(state['history'][0]['tracking_status'],'TP')
        self.assertEqual(len(state['history'][0]['features']['closed_15m_ohlcv']),64)
        self.assertEqual(calls.count('/fapi/v1/klines'),1)
        self.assertEqual(calls.count('WS_SNAPSHOT'),3)
        self.assertNotIn('/fapi/v1/ticker/24hr',calls)

    def test_document_ack_controls_success_and_no_token_no_send(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_send_report_file')
        exec(compile(ast.Module(body=[n],type_ignores=[]),'ema.py','exec'),self.g)
        p=self.tmp.name+'/data.zip'; Path(p).write_bytes(b'zip')
        self.g['requests'].post.return_value.json.return_value={'ok':False}
        self.assertFalse(self.g['_crash_send_report_file'](p,'test'))
        self.g['requests'].post.return_value.json.return_value={'ok':True}
        self.assertTrue(self.g['_crash_send_report_file'](p,'test'))
        self.g['BOT_TOKEN']=None
        self.g['requests'].post.reset_mock()
        self.assertFalse(self.g['_crash_send_report_file'](p,'test'))
        self.g['requests'].post.assert_not_called()


if __name__=='__main__': unittest.main()
