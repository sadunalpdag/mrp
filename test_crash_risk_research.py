"""Deterministic offline research scenarios; no network, Telegram or orders."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from crash_risk_research import RiskResearch, CONFIG
from crash_history import HistoryCollector


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r=RiskResearch(self.tmp.name)
        self.at=self.r.state['started_at']+1
        self.id=0

    def enter(self,key='one',score=60,symbol='AUSDT',at=None):
        self.r.enter(dict(signal_id=key,symbol=symbol,entered_at=at or self.at,
            observed_entry_price=100,strength_score={'score':score},
            features={'ticker_24h':{'quoteVolume':20000000}}))

    def trade(self,price,offset=1,symbol='AUSDT'):
        self.id+=1
        self.r.trade(dict(s=symbol,T=(self.at+offset)*1000,p=str(price),a=self.id))

    def quote(self,offset=0,bid=99.99,ask=100.01):
        self.r.on_quote(dict(s='AUSDT',T=(self.at+offset)*1000,b=str(bid),a=str(ask),B='100',A='100'))

    def scenario(self,tp=1.,sl=None,score=0,liquidity='all'):
        return self.r.state['scenarios'][f'tp{tp}_sl{sl}_score{score}_{liquidity}']

    def test_tp_levels_and_net_costs(self):
        self.quote();self.enter();self.trade(101)
        self.assertEqual(self.scenario()['positions']['one']['status'],'TP')
        self.assertEqual(self.scenario(1.2)['positions']['one']['status'],'OPEN')
        self.trade(101.2,2);self.trade(101.5,3)
        for tp in (1.,1.2,1.5):
            p=self.scenario(tp)['positions']['one']
            self.assertEqual(p['status'],'TP')
            self.assertLess(p['net_usdt'],50*tp/100)
            self.assertAlmostEqual(p['net_usdt'],p['mark_gross']-p['mark_fees'])

    def test_sl_then_tp_must_happen_after_sl(self):
        self.enter();self.trade(101.1,1)
        self.trade(97.9,2)
        p=self.scenario(1.5,2.)['positions']['one']
        self.assertEqual(p['status'],'SL');self.assertFalse(p['sl_then_target'])
        self.trade(101.1,3);self.assertFalse(p['sl_then_target'])
        self.trade(101.6,4);self.assertTrue(p['sl_then_target'])
        self.assertEqual(self.scenario(1.5)['positions']['one']['status'],'TP')

    def test_sl_levels(self):
        self.enter();self.trade(97.9)
        self.assertEqual(self.scenario(sl=2.)['positions']['one']['status'],'SL')
        self.assertEqual(self.scenario(sl=3.)['positions']['one']['status'],'OPEN')
        self.trade(96.9,2);self.trade(94.9,3)
        for sl in (2.,3.,5.):self.assertEqual(self.scenario(sl=sl)['positions']['one']['status'],'SL')
        self.assertEqual(self.scenario()['positions']['one']['status'],'OPEN')

    def test_filters_missing_and_future_quote(self):
        self.quote(offset=1);self.enter()
        self.assertEqual(self.scenario(liquidity='spread')['missing_filter_data'],1)
        self.assertFalse(self.scenario()['positions']['one']['entry_quote_known'])
        self.r.quotes.clear();self.quote();self.enter('two')
        self.assertIn('two',self.scenario(liquidity='liquid')['positions'])
        self.assertNotIn('two',self.scenario(score=70)['positions'])
        self.assertIn('two',self.scenario(score=50)['positions'])

    def test_open_losses_drawdown_concurrency(self):
        self.enter();self.enter('two',symbol='BUSDT');self.trade(90)
        s=self.scenario()
        self.assertEqual(s['max_open'],2)
        self.assertLess(s['equity'],995)
        self.assertGreater(s['max_drawdown_usdt'],5)
        self.assertEqual(s['required_peak_notional'],100)

    def test_restart_dedup_and_quality(self):
        self.enter();self.trade(99);self.r.checkpoint()
        restored=RiskResearch(self.tmp.name)
        restored.enter(dict(signal_id='one',symbol='AUSDT',entered_at=self.at,observed_entry_price=100))
        self.assertEqual(len(restored.state['entries']),1)
        self.assertTrue(restored.state['scenarios']['tp1.0_slNone_score0_all']['positions']['one']['coverage_incomplete'])
        self.assertEqual(restored.quotes,{})
        restored.checkpoint()
        self.assertTrue(Path(self.tmp.name,'summary.json').is_file())

    def test_enrollment_followup_and_timeout(self):
        self.enter();self.enter('late',at=self.at+15*86400)
        self.assertNotIn('late',self.r.state['entries'])
        self.trade(100,5*86400+1)
        p=self.scenario()['positions']['one']
        self.assertEqual(p['status'],'TIMEOUT');self.assertTrue(p['coverage_incomplete'])
        self.assertGreater(self.r.summary()['scenarios']['tp1.0_slNone_score0_all']['unknown_cost_exits'],0)

    def test_watch_before_entry_and_routed_subscriptions(self):
        self.r.watches['AUSDT']=self.at+60
        self.r.desired=frozenset(self.r.wanted())
        self.r.sockets={k:Mock() for k in ('public','market')}
        self.r.sync();self.r.sync()
        for k,suffix in [('public','@bookTicker'),('market','@aggTrade')]:
            ws=self.r.sockets[k];self.assertEqual(ws.send.call_count,1)
            self.assertEqual(json.loads(ws.send.call_args.args[0])['params'],['ausdt'+suffix])

    def test_duplicate_and_out_of_order_prints(self):
        self.enter();self.trade(99,3)
        self.r.trade(dict(s='AUSDT',T=(self.at+4)*1000,p='102',a=1))
        self.trade(102,2)
        self.assertEqual(self.scenario()['positions']['one']['status'],'OPEN')

    def test_bad_history_timestamp_does_not_poison_queue(self):
        h=HistoryCollector(self.tmp.name,Mock(),Mock(),lambda:False,Mock())
        with self.assertRaisesRegex(ValueError,'timestamp'):
            h.submit({'signal_id':'bad','symbol':'AUSDT'},'signal',float('nan'))
        self.assertEqual(h.jobs,{})
        h.submit({'signal_id':'good','symbol':'AUSDT'},'signal',self.at)
        self.assertIn('good:signal',h.jobs)

    def test_history_queue_log_context_and_retry(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_queue_history')
        collector=Mock();collector.submit.side_effect=ValueError('invalid timestamp')
        log=Mock();namespace=dict(time=Mock(time=lambda:1000),_CRASH_HISTORY_COLLECTOR=collector,
            _CRASH_HISTORY_ERRORS={},log=log)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'ema.py','exec'),namespace)
        for _ in range(2):namespace['_crash_queue_history']({'signal_id':'s1','symbol':'AUSDT'},'signal',float('nan'))
        self.assertEqual(log.call_count,1)
        message=log.call_args.args[0]
        for part in ('AUSDT','s1','signal','invalid timestamp','retry=300s'):self.assertIn(part,message)

if __name__=='__main__':unittest.main()
