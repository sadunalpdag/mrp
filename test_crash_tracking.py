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
                    CRASH_BOUNCE_FILTER_VERSION='five_day_entry_v2_only',
                    CRASH_BOUNCE_RESEARCH_DIR=self.tmp.name,_crash_save_state=Mock(),
                    _crash_append_data=Mock(),_crash_entry_features=Mock(),
                    _crash_send_report_file=Mock(return_value=True),
                    log=Mock(),BOT_TOKEN='test',CHAT_ID='test',requests=Mock())
        exec(compile(subset,'ema.py','exec'),self.g)
        self.g['_crash_send_report_file']=Mock(return_value=True)
        self.now=2_000_000_000
        self.r=dict(symbol='AUSDT',created_at=self.now,entry='1',tp='1.01')
        self.state={'history':[self.r],'symbols':{}}

    def sample(self,offset,price):
        self.g['_crash_track_results'](self.state,[{'symbol':'AUSDT','lastPrice':str(price)}],self.now+offset)

    def upward_fixture(self):
        r=self.state['history'][0]
        r.update(tracking_model='sampled_upward_cross_v2',reference_low='.95',
                 last_price='.96',previous_watch_price=.96)
        return r

    def test_upward_cross_enters_then_later_tp(self):
        r=self.upward_fixture()
        self.sample(180,.97)
        self.assertEqual(r['tracking_status'],'WAIT_ENTRY')
        self.sample(360,1.02)
        self.assertEqual(r['tracking_status'],'OPEN')
        self.sample(540,1.02)
        self.assertEqual(r['tracking_status'],'TP')

    def test_new_low_invalidates_waiting_upward_entry(self):
        r=self.upward_fixture()
        self.sample(180,.94)
        self.assertEqual(r['tracking_status'],'INVALIDATED_BEFORE_ENTRY')
        self.sample(360,1.02)
        self.assertEqual(r['tracking_status'],'INVALIDATED_BEFORE_ENTRY')
        self.assertEqual(self.g['_crash_summary']([r])['INVALIDATED_BEFORE_ENTRY'],1)

    def test_already_above_entry_needs_observed_below_then_cross(self):
        r=self.upward_fixture();r['previous_watch_price']=1.02
        self.sample(180,1.03)
        self.assertEqual(r['tracking_status'],'WAIT_ENTRY')
        self.sample(360,.98);self.sample(540,1.01)
        self.assertEqual(r['tracking_status'],'OPEN')

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

    def test_new_ws_model_does_not_count_three_minute_snapshot_as_tp(self):
        r=self.upward_fixture()
        r.update(tracking_model='ws_aggtrade_upcross_v3', v2_status='STRONG_V2')
        self.sample(180,.97)
        self.sample(360,1.03)
        self.sample(540,1.04)
        self.assertEqual(r['tracking_status'],'WAIT_ENTRY')
        self.assertNotIn('entered_at',r)

    def test_scan_records_candidate_but_waits_for_ws_execution(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        names={'run_crash_bounce_alerts','crash_bounce_plan'}
        subset=ast.Module(body=[n for n in tree.body
            if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        self.g.update(Decimal=Decimal,time=SimpleNamespace(time=lambda:self.now),
            CRASH_BOUNCE_ENTRY_BUFFER=Decimal('.003'),CRASH_BOUNCE_TP_PCT=Decimal('.01'),
            CRASH_BOUNCE_TOP_LOSERS=20,CRASH_BOUNCE_DROP_PCT=12,
            CRASH_BOUNCE_COOLDOWN_SECONDS=86400,
            _crash_send_pending=Mock(),_crash_review_pending_v2=Mock(),
            _crash_apply_filter_migration=Mock(),_crash_queue_history=Mock(),
            _crash_review_candidate=Mock(),
            _crash_initialize_tracking=self.g['_crash_initialize_tracking'],
            _crash_features=Mock(return_value={'rsi14_wilder':26,'ema25_slope_pct':-.1}),
            _crash_signal_score=Mock(return_value={'score':50}),
            BinanceRateLimiter=SimpleNamespace(is_banned=lambda:False))
        exec(compile(subset,'ema.py','exec'),self.g)
        rows=[[int((self.now-(64-i)*900)*1000),'1','1.2','.9','1','100',
               int((self.now-(63-i)*900)*1000)-1] for i in range(64)]
        self.g['_CRASH_WS_CACHE']=SimpleNamespace(
            snapshot=lambda now:[{'symbol':'AUSDT','priceChangePercent':'-20',
                                  'lastPrice':'1','closeTime':int(now*1000)}],
            reconcile=Mock())
        self.g['_crash_get_candles']=lambda symbol:rows
        state={'symbols':{},'history':[]}
        self.g['run_crash_bounce_alerts'](state,{'AUSDT':'.001'},self.now)
        self.assertEqual(len(state['history']),1)
        r=state['history'][0]
        self.assertEqual(r['tracking_model'],'ws_aggtrade_upcross_v3')
        self.assertEqual(r['v2_status'],'PENDING_5D_HISTORY')
        self.assertNotIn('entered_at',r)

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
