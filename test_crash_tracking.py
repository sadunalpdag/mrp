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

    def confirmation_fixture(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_confirmation_step')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'ema.py','exec'),self.g)
        self.g.update(Decimal=Decimal,CRASH_BOUNCE_TP_PCT=Decimal('.01'),tg_send=Mock(return_value=True),log=Mock())
        r=self.state['history'][0]
        r.update(confirmation_enabled=True,tracking_status='WAIT_ENTRY',reference_low='.95',tick_size='.001',
                 signal_id='test',sent=True,delivery_status='SENT',daily_signal_number=1,signal_day='2033-05-18')
        self.state['symbols']={'AUSDT':dict(r)}
        # Opens after the candidate; closes at +900s; green with high 1.05.
        self.confirm_rows=[[int(self.now*1000),'.96','1.05','.95','1.02','1',int((self.now+900)*1000)-1]]
        return r

    def confirm_sample(self,offset,price,rows=None):
        self.g['_crash_confirmation_step'](self.state,[dict(symbol='AUSDT',lastPrice=str(price))],
            self.now+offset,lambda symbol:self.confirm_rows if rows is None else rows)

    def test_confirmation_only_post_close_upward_break_and_one_message(self):
        r=self.confirmation_fixture()
        self.confirm_sample(800,1.06)  # Unfinished candle cannot confirm.
        self.assertNotIn('reversal_confirmation',r)
        self.confirm_sample(1000,1.04);self.confirm_sample(1180,1.06)
        c=r['reversal_confirmation']
        self.assertEqual(c['entry'],'1.06')
        self.assertEqual(c['tp'],'1.071')
        self.assertEqual(c['tracking_status'],'OPEN')
        self.assertEqual(self.g['tg_send'].call_count,1)
        self.confirm_sample(1360,1.08)
        self.assertEqual(c['tracking_status'],'TP')
        self.assertEqual(self.g['tg_send'].call_count,1)

    def test_dip_break_invalidates_confirmation_even_if_candidate_entered(self):
        r=self.confirmation_fixture();r['tracking_status']='OPEN'
        self.confirm_sample(1000,.94);self.confirm_sample(1180,1.06)
        self.assertEqual(r['confirmation_status'],'INVALIDATED')
        self.g['tg_send'].assert_not_called()

    def test_closed_candle_low_break_invalidates_when_current_price_recovered(self):
        r=self.confirmation_fixture();self.confirm_rows[0][3]='.94'
        self.confirm_sample(1000,1.04)
        self.assertEqual(r['confirmation_status'],'INVALIDATED')

    def test_pre_candidate_and_red_candles_cannot_confirm(self):
        r=self.confirmation_fixture()
        old=list(self.confirm_rows[0]);old[0]=int((self.now-900)*1000)
        self.confirm_rows[0][4]='.95'
        self.confirm_sample(1000,1.04,rows=[old]+self.confirm_rows)
        self.confirm_sample(1180,1.06,rows=[old]+self.confirm_rows)
        self.assertNotIn('reversal_confirmation',r)

    def test_new_candidate_supersedes_unconfirmed_old_candidate(self):
        r=self.confirmation_fixture()
        self.state['symbols']['AUSDT']['signal_id']='new'
        self.confirm_sample(1000,1.04);self.confirm_sample(1180,1.06)
        self.assertEqual(r['confirmation_status'],'SUPERSEDED')
        self.g['tg_send'].assert_not_called()

    def test_confirmation_delivery_retry_and_restart_are_durable(self):
        r=self.confirmation_fixture();self.g['tg_send'].return_value=False
        self.confirm_sample(1000,1.04);self.confirm_sample(1180,1.06)
        c=r['reversal_confirmation'];self.assertFalse(c['sent'])
        self.state=json.loads(json.dumps(self.state))
        self.g['tg_send'].return_value=True
        self.confirm_sample(1360,1.06)
        self.assertTrue(self.state['history'][0]['reversal_confirmation']['sent'])
        self.confirm_sample(1540,1.06)
        self.assertEqual(self.g['tg_send'].call_count,2)

    def test_confirmation_persistence_failure_blocks_telegram(self):
        self.confirmation_fixture();self.confirm_sample(1000,1.04)
        self.g['_crash_save_state'].side_effect=OSError('disk full')
        with self.assertRaises(OSError):self.confirm_sample(1180,1.06)
        self.g['tg_send'].assert_not_called()

    def test_confirmation_timeout_and_gap_are_separate_from_candidate(self):
        r=self.confirmation_fixture();self.confirm_sample(1000,1.04);self.confirm_sample(1180,1.06)
        self.confirm_sample(1900,1.06)
        self.assertEqual(r['reversal_confirmation']['coverage_gaps'],1)
        self.confirm_sample(1180+72*3600+1,1.06)
        self.assertEqual(r['reversal_confirmation']['tracking_status'],'TIMEOUT')
        self.assertEqual(r['tracking_status'],'WAIT_ENTRY')

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
            CRASH_BOUNCE_COOLDOWN_SECONDS=86400,_crash_send_pending=Mock(),_crash_confirmation_step=Mock(),
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
        for elapsed,value in [(0,'1'),(180,'.9'),(360,'.905'),(540,'.92')]:
            price[0]=value
            self.g['run_crash_bounce_alerts'](state,{'AUSDT':'.001'},self.now+elapsed)
        self.assertEqual(len(state['history']),1)
        self.assertEqual(state['history'][0]['tracking_status'],'TP')
        self.assertEqual(len(state['history'][0]['features']['closed_15m_ohlcv']),64)
        self.assertEqual(calls.count('/fapi/v1/klines'),1)
        self.assertEqual(calls.count('WS_SNAPSHOT'),4)
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
