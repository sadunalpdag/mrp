"""Offline manual-position repair and file-toggle tests. Never contacts Binance."""
import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from decimal import Decimal as D
from crash_night import NightExecutor, BinanceBroker, BrokerError, error_details, day
from crash_manual_tp import ManualTPGuard, tp_coverage
from test_crash_services import INFO, timestamp


class ManualBroker:
    def __init__(self):
        self.ps=[dict(symbol='AUSDT',positionAmt='5',entryPrice='10',positionSide='BOTH')]
        self.open=[];self.queries={};self.posts=[];self.failure=None;self.bid=D('10')
        self.dual=False
    def request(self,*args):return {'dualSidePosition':self.dual}
    def positions(self):return copy.deepcopy(self.ps)
    def closing_orders(self,symbol):return copy.deepcopy(self.open)
    def info(self,symbol):return copy.deepcopy(INFO)
    def best_bid(self,symbol):return self.bid
    def repair_tp(self,symbol,qty,price,cid):
        self.posts.append((symbol,qty,price,cid))
        if self.failure:raise self.failure
        self.queries[cid]=dict(symbol=symbol,side='SELL',type='LIMIT',status='NEW',positionSide='BOTH',reduceOnly=True,origQty=qty,executedQty='0',price=price)
        self.open.append(self.queries[cid])
    def order(self,symbol,cid):
        if cid not in self.queries:raise BrokerError('-2013')
        return copy.deepcopy(self.queries[cid])


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.now=timestamp('2026-10-28T12:00:00');self.b=ManualBroker()
        self.e=NightExecutor(self.tmp.name,self.b,Mock(return_value={'ok':True}),Mock(),lambda:self.now,manual_tp=True)
        self.addCleanup(self.e.guard.close);self.g=ManualTPGuard(self.e)

    def test_manual_daytime_long_missing_tp_repaired_verified(self):
        self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.assertEqual(tuple(D(v) for v in self.b.posts[0][1:3]),(D('5'),D('10.1')))
        self.assertEqual(self.g.state['AUSDT']['phase'],'PROTECTED')
        self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.assertFalse(self.e.enabled.is_set())

    def test_existing_conditional_close_all_tp_no_duplicate(self):
        self.b.open=[dict(symbol='AUSDT',side='SELL',orderType='TAKE_PROFIT_MARKET',algoStatus='NEW',positionSide='BOTH',closePosition=True,triggerPrice='10.2')]
        self.g.scan();self.assertFalse(self.b.posts)

    def test_partial_tp_and_sl_preserved_warn_without_second_order(self):
        self.b.open=[dict(symbol='AUSDT',side='SELL',type='LIMIT',status='NEW',reduceOnly=True,price='10.2',origQty='2',executedQty='0')]
        self.g.scan();self.assertFalse(self.b.posts)
        self.assertIn('PARTIAL_TP',self.e.state['outbox'][-1]['text'])
        self.b.open[0]['type']='STOP_MARKET'
        self.g.scan();self.assertIn('ORDER_CONFLICT',self.e.state['outbox'][-1]['text'])

    def test_api_query_error_never_treated_as_no_tp(self):
        self.b.closing_orders=Mock(side_effect=BrokerError('-2015'))
        self.g.scan();self.assertFalse(self.b.posts)
        msg=self.e.state['outbox'][-1]['text'];self.assertIn('-2015',msg);self.assertIn('Futures yetkisi',msg)

    def test_unknown_post_is_query_only_even_after_restart(self):
        self.b.failure=BrokerError('UNKNOWN',uncertain=True)
        self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.e.guard.close()
        self.e=NightExecutor(self.tmp.name,self.b,Mock(),Mock(),lambda:self.now,manual_tp=True)
        self.addCleanup(self.e.guard.close);self.g=ManualTPGuard(self.e)
        self.now+=3600;self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.assertEqual(self.g.state['AUSDT']['phase'],'UNKNOWN')

    def test_rejected_tp_two_attempts_only_with_specific_error(self):
        self.b.failure=BrokerError('-2022')
        self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.now+=61;self.g.scan();self.assertEqual(len(self.b.posts),2)
        self.now+=3600;self.g.scan();self.assertEqual(len(self.b.posts),2)
        self.assertIn('-2022',self.e.state['outbox'][0]['text'])
        self.assertIn('Reduce-only',self.e.state['outbox'][0]['text'])

    def test_accepted_post_timeout_reconciles_without_duplicate(self):
        original = self.b.repair_tp
        def timeout_after_accept(*args):
            original(*args)
            raise BrokerError('UNKNOWN', uncertain=True)
        self.b.repair_tp = timeout_after_accept
        self.g.scan(); self.g.scan()
        self.assertEqual(len(self.b.posts), 1)
        self.assertEqual(self.g.state['AUSDT']['phase'], 'PROTECTED')

    def test_open_algo_failure_does_not_claim_absent(self):
        b = BinanceBroker('fake','fake')
        b.request = Mock(side_effect=[[], BrokerError('-2015')])
        with self.assertRaises(BrokerError): b.closing_orders('AUSDT')

    def test_conditional_entries_count_toward_five_symbols(self):
        b = BinanceBroker('fake','fake')
        b.request = Mock(side_effect=[{'dualSidePosition':False}, [], [],
                                      [{'symbol':str(i)} for i in range(5)]])
        with self.assertRaisesRegex(ValueError,'MAX_5'): b.preflight('AUSDT')
        self.assertFalse(any(c.args[0]=='POST' for c in b.request.call_args_list))

    def test_target_passed_does_not_market_sell(self):
        self.b.bid=D('10.11');self.g.scan();self.assertFalse(self.b.posts)
        self.assertIn('TARGET_PASSED',self.e.state['outbox'][-1]['text'])

    def test_changed_position_during_preflight_does_not_post(self):
        self.b.positions=Mock(side_effect=[copy.deepcopy(self.b.ps),[dict(self.b.ps[0],positionAmt='3')]])
        self.g.scan();self.assertFalse(self.b.posts)
        self.assertIn('POSITION_CHANGED',self.e.state['outbox'][-1]['text'])

    def test_own_bot_positions_excluded_and_short_skipped(self):
        self.e.state['orders']['id']=dict(symbol='AUSDT',phase='TP_UNKNOWN',day=day(self.now))
        self.g.scan();self.assertFalse(self.b.posts)
        self.e.state['orders'].clear();self.b.ps[0]['positionAmt']='-5'
        self.g.scan();self.assertFalse(self.b.posts)

    def test_hedge_mode_warns_without_changing_mode(self):
        self.b.dual=True;self.g.scan();self.assertFalse(self.b.posts)
        self.assertIn('ONE_WAY_MODE_REQUIRED',self.e.state['outbox'][-1]['text'])

    def test_position_closed_before_submit_no_order(self):
        self.b.positions=Mock(side_effect=[copy.deepcopy(self.b.ps),[]])
        self.g.scan();self.assertFalse(self.b.posts)

    def test_cancelled_tp_warns_without_replacing_user_cancellation(self):
        self.g.scan();self.b.open=[]
        cid=self.b.posts[0][3];self.b.queries[cid]['status']='CANCELED'
        self.g.scan();self.assertEqual(len(self.b.posts),1)
        self.assertEqual(self.g.state['AUSDT']['phase'],'HOLD')

    def test_persistence_failure_prevents_post(self):
        self.e.save=Mock(side_effect=OSError('disk'))
        with self.assertRaises(OSError):self.g.scan()
        self.assertFalse(self.b.posts)

    def test_closed_then_new_same_size_position_has_new_id(self):
        self.g.scan();first=self.b.posts[0][3]
        self.b.ps=[];self.b.open=[];self.g.scan()
        self.b.ps=[dict(symbol='AUSDT',positionAmt='5',entryPrice='10')]
        self.g.scan();self.assertEqual(len(self.b.posts),2);self.assertNotEqual(first,self.b.posts[1][3])

    def test_default_daily_limit_is_five(self):
        self.assertEqual(self.e.max_daily,5)
        self.e.enabled.set();self.now=timestamp('2026-10-28T21:00:00')
        for i in range(5):self.e.state['orders'][str(i)]=dict(symbol=str(i),day=day(self.now),phase='DONE')
        self.b.info=Mock()
        self.e.enter(dict(symbol='AUSDT',signal_id='six',v2_sent=True,v2_status='STRONG_V2',tracking_status='OPEN'),self.now)
        self.b.info.assert_not_called()

    def test_repair_order_is_maker_reduce_only(self):
        b=BinanceBroker('fake','fake');b.request=Mock()
        b.repair_tp('AUSDT','5','10.1','cid')
        params=b.request.call_args.args[2]
        self.assertEqual((params['timeInForce'],params['reduceOnly']),('GTX','true'))

    def test_error_never_includes_raw_exception_or_secret(self):
        secret='https://host/?signature=VERY_SECRET'
        self.assertNotIn('VERY_SECRET',error_details(BrokerError(secret)))
        self.assertNotIn('VERY_SECRET',error_details(RuntimeError(secret)))


class FileFlagTests(unittest.TestCase):
    def test_ema_flags_zero_and_exact_env_names(self):
        source=Path(__file__).with_name('ema.py').read_text();tree=ast.parse(source)
        values={n.targets[0].id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
        self.assertIn(values['AUTO_TRADING_ENABLED'],(0,1))
        self.assertIn(values['MANUAL_TP_REPAIR_ENABLED'],(0,1))
        self.assertEqual(values['AUTO_MAX_DAILY_ENTRIES'],5)
        self.assertIn('os.getenv("BINANCE_API_KEY")',source)
        self.assertIn('os.getenv("BINANCE_SECRET_KEY")',source)

    def test_file_flags_control_executor_telegram_cannot_override(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_start_services'],type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CRASH_NIGHT_STATE_DIR':tmp,'CRASH_NIGHT_TRADING':'live'},clear=True),patch('crash_night.NightExecutor') as executor,patch('crash_night.BinanceBroker'),patch('crash_heartbeat.Heartbeat'),patch('crash_telegram_control.TelegramControl') as tg:
            g=dict(os=os,time=SimpleNamespace(time=lambda:1),DATA_DIR=tmp,log=Mock(),BINANCE_KEY='fake',BINANCE_SECRET='fake',_crash_telegram_receipt=Mock(),_CRASH_WS_CACHE=Mock(),AUTO_TRADING_ENABLED=0,MANUAL_TP_REPAIR_ENABLED=0,AUTO_MAX_DAILY_ENTRIES=5)
            exec(compile(subset,'ema.py','exec'),g);g['_crash_start_services']()
            executor.assert_not_called();tg.assert_not_called()
            g['MANUAL_TP_REPAIR_ENABLED']=1;g['_crash_start_services']()
            self.assertFalse(executor.call_args.kwargs['enabled'])
            self.assertTrue(executor.call_args.kwargs['manual_tp'])
            g['AUTO_TRADING_ENABLED']=1;g['_crash_start_services']()
            self.assertTrue(executor.call_args.kwargs['enabled']);tg.assert_not_called()

    def test_missing_disk_setting_uses_data_dir_and_guards_daily_quota(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_start_services'],type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True),patch('crash_night.NightExecutor') as executor,patch('crash_night.BinanceBroker'),patch('crash_heartbeat.Heartbeat'):
            executor.return_value.state={}
            g=dict(os=os,time=SimpleNamespace(time=lambda:1),DATA_DIR=tmp,log=Mock(),BINANCE_KEY='fake',BINANCE_SECRET='fake',_crash_telegram_receipt=Mock(),_CRASH_WS_CACHE=Mock(),AUTO_TRADING_ENABLED=1,MANUAL_TP_REPAIR_ENABLED=1,AUTO_MAX_DAILY_ENTRIES=5)
            exec(compile(subset,'ema.py','exec'),g);g['_crash_start_services']()
            self.assertEqual(executor.call_args.args[0],os.path.join(tmp,'night_execution'))
            self.assertEqual(executor.return_value.state['unverified_day'],day(1))
            executor.return_value.start.assert_called_once()
