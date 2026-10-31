import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from crash_derivatives import *
from crash_report_schedule import prepare, slot, success
from crash_live_v2 import advance_trade, MODEL

END = 1800000000000 // STEP * STEP

def rows(buy=60, source='BINANCE_REST'):
    return [[t,'100','101','89','90','10',t+STEP-1,'100',10,'6',str(buy),'0',source]
            for t in range(END-16*STEP,END,STEP)]

class ShadowTests(unittest.TestCase):
    def record(self):
        return {'signal_id':'test','symbol':'BTCUSDT','created_at':END/1000+100,
                'v2_features':{'strength_score':{'score':68}}}

    def test_rules_unknown_priority_and_clipping(self):
        self.assertEqual(classify_derivatives_regime(-1,-2,1,-1),'DELEVERAGING')
        self.assertEqual(classify_derivatives_regime(-1,3,1,-1),'OI_RISING_SELL_PRESSURE')
        self.assertEqual(classify_derivatives_regime(-1,3,1,1),'POSSIBLE_ABSORPTION')
        self.assertEqual(classify_derivatives_regime(1,0,1,1),'SPOT_CONFIRMED_RECOVERY')
        for missing in (None,float('nan'),float('inf')):
            self.assertEqual(classify_derivatives_regime(-1,missing,1,1),'UNKNOWN')
        d = empty_snapshot(self.record())
        self.assertEqual(d['shadow_score'],68)
        d.update(price_change_4h=-1,oi_change_4h=3,spot_cvd_4h=-1,perp_cvd_4h=-1)
        self.assertEqual(finalize(d)['shadow_score'],53)
        d['base_score']=5
        self.assertEqual(finalize(d)['shadow_score'],0)

    def test_cvd_windows_missing_zero_and_bybit(self):
        self.assertEqual(cvd(rows(),END,16),320)
        self.assertEqual(cvd(rows(),END,4),80)
        self.assertEqual(cvd(rows(50),END,16),0)  # real zero is permitted
        self.assertIsNone(cvd(rows()[:-1],END,16))
        self.assertIsNone(cvd(rows(source='BYBIT_LINEAR'),END,16))
        bad=rows();bad[-1][10]=None
        self.assertIsNone(cvd(bad,END,16))
        self.assertEqual(cvd(rows()+[[END,'1','2','1','2','1',END+STEP-1,'1',1,'1','1']],END,16),320)

    def test_oi_exact_asof_endpoints_no_future_no_value_bias(self):
        oi=[{'timestamp':END-h*3600000,'sumOpenInterest':100 if h else 103,
             'sumOpenInterestValue':999999} for h in (0,1,4)]
        self.assertAlmostEqual(oi_changes(oi,END)['oi_change_4h'],3)
        self.assertIsNone(oi_changes(oi[1:],END)['oi_change_4h'])
        oi[0]['timestamp']+=STEP
        self.assertIsNone(oi_changes(oi,END)['oi_change_4h'])

    def test_collector_uses_cache_limits_rest_and_keeps_signal_unchanged(self):
        def fetch(path,params):
            if path=='/api/v3/klines': return rows(40)
            if 'openInterest' in path:
                return [{'timestamp':END-h*3600000,'sumOpenInterest':103 if not h else 100} for h in (0,1,4)]
            if 'fundingRate' in path:
                return [{'fundingTime':END,'fundingRate':'.001'}, {'fundingTime':END+999999,'fundingRate':'9'}]
            self.fail('unnecessary perp REST')
        with tempfile.TemporaryDirectory() as root:
            f=Mock(side_effect=fetch);log=Mock()
            c=ShadowCollector(root,f,lambda:False,lambda s:rows(40),log,lambda:END/1000+110)
            r=self.record();before=copy.deepcopy(r)
            c.submit(r);c.submit(r);c.step()
            d=c.results.get_nowait()
            self.assertEqual(r,before)
            self.assertEqual(d['derivatives_regime'],'OI_RISING_SELL_PRESSURE')
            self.assertEqual(d['funding_rate'],.001)
            self.assertEqual(f.call_count,3)
            self.assertEqual(len(Path(root,'snapshots.jsonl').read_text().splitlines()),1)
            self.assertEqual(d['window_end_ms'],END)

    def test_cooldown_unknown_never_fabricated_and_no_requests(self):
        with tempfile.TemporaryDirectory() as root:
            f=Mock();c=ShadowCollector(root,f,lambda:True,lambda s:[],Mock())
            d=c.collect(empty_snapshot(self.record()))
            self.assertEqual(d['derivatives_regime'],'UNKNOWN')
            self.assertEqual(d['base_score'],d['shadow_score'])
            self.assertIsNone(d['spot_cvd_4h']);f.assert_not_called()
            self.assertIn('bilinmiyor',format_shadow(d))

    def test_observational_fields_do_not_change_transitions(self):
        base={'tracking_model':MODEL,'tracking_status':'WAIT_ENTRY','created_at':100,
              'entry':'1','tp':'1.01','reference_low':'.95','previous_watch_price':.98}
        shadow=copy.deepcopy(base);shadow['derivatives_shadow']={}
        for i,(t,p) in enumerate([(101,1.005),(102,.94),(103,1.02)],1):
            self.assertEqual(advance_trade(base,p,t,i),advance_trade(shadow,p,t,i))
        self.assertEqual(base['tracking_status'],shadow['tracking_status'])
        self.assertAlmostEqual(shadow['shadow_mae_pct'],-6)
        self.assertTrue(shadow['shadow_new_low_observed'])

    def test_schedule_calendar_restart_and_dst(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        zone=ZoneInfo('Europe/Paris')
        t=datetime(2026,10,24,22,tzinfo=zone).timestamp()
        r=prepare({},t);self.assertLess(r['next_due_at'],t)
        success(r,t)
        self.assertEqual(datetime.fromtimestamp(r['next_due_at'],zone).hour,21)
        self.assertEqual(r['next_due_at']-slot(t),25*3600)
        saved=dict(r);prepare(r,t+60);self.assertEqual(r,saved)

    def test_export_does_not_treat_open_as_loss(self):
        r=self.record();r.update(v2_sent=True,entered_at=10,tracking_status='OPEN',derivatives_shadow=empty_snapshot(r))
        summary=summarize([r])['groups']['UNKNOWN']
        self.assertIsNone(summary['tp_rate_resolved_pct'])
        self.assertIn('oi_change_4h',export_csv([r]))

    def test_auto_off_and_v2_unchanged(self):
        tree=ast.parse(Path('ema.py').read_text())
        values={n.targets[0].id:n.value.value for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
        self.assertEqual(values['AUTO_TRADING_ENABLED'],0)

    def test_signal_send_never_calls_rest_and_queues_only_after_acceptance(self):
        from test_crash_v2 import CrashV2Tests
        fixture=CrashV2Tests()
        env, receipt = fixture._delivery_context()
        state, record = fixture._candidate()
        collector = Mock(); env['_CRASH_SHADOW'] = collector
        env['_crash_send_pending'](state,105)
        collector.submit.assert_called_once_with(record)
        collector.collect.assert_not_called()
        self.assertTrue(record['v2_sent'])
        self.assertEqual(record['derivatives_shadow']['derivatives_regime'],'UNKNOWN')
        self.assertIn('Gölge puan', receipt.call_args.args[0])
        state, record = fixture._candidate(); collector.reset_mock()
        receipt.return_value={'ok':False,'error':'FAILED'}
        env['_crash_send_pending'](state,105)
        collector.submit.assert_not_called()
