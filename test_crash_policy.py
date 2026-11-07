"""Offline regressions for 24-hour tracking and unsupported symbols."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from crash_policy import TRACKING_SECONDS, POST_ENTRY_TRACKING_SECONDS, supported_symbol, prune_active_state
from crash_live_v2 import advance_trade, MODEL
from crash_history import HistoryCollector

class PolicyTests(unittest.TestCase):
    def record(self):
        return dict(symbol='AUSDT',tracking_model=MODEL,tracking_status='WAIT_ENTRY',
                    created_at=100.,entry='1',tp='1.01',reference_low='.95',
                    previous_watch_price=.99)

    def test_unresolved_open_trade_tracks_tp_after_24h(self):
        r=self.record(); t=100+23*3600
        r.update(ws_last_event_at=t-1,ws_last_trade_id=1)
        self.assertEqual(advance_trade(r,1,t,2),'OPEN')
        self.assertEqual(advance_trade(r,1.02,100+24*3600+2,3),'TP')
        self.assertEqual(r['tracking_status'],'TP')

    def test_open_trade_closes_virtual_window_after_150h_from_entry(self):
        r=self.record();t=100+23*3600
        r.update(ws_last_event_at=t-1,ws_last_trade_id=1)
        self.assertEqual(advance_trade(r,1,t,2),'OPEN')
        self.assertIsNone(advance_trade(r,1.02,t+POST_ENTRY_TRACKING_SECONDS,3))
        self.assertEqual(r['tracking_status'],'OPEN') # timer loop owns TIMEOUT

    def test_prune_after_24h_preserves_recent_signal(self):
        old=self.record()
        new=dict(old,signal_id='recent',created_at=101.,tracking_status='WAIT_ENTRY')
        old_open=dict(old,signal_id='open',tracking_status='OPEN',entered_at=100+3600)
        state={'history':[old,new,old_open], 'symbols':{
            'OLDUSDT':old,'NEWUSDT':new,'OPENUSDT':old_open}}
        removed=prune_active_state(state,100+TRACKING_SECONDS)
        self.assertEqual(removed,2)
        self.assertEqual(state['history'],[new,old_open])
        self.assertEqual(list(state['symbols']),['NEWUSDT','OPENUSDT'])

    def test_wait_and_open_deadlines(self):
        r=self.record()
        self.assertIsNone(advance_trade(r,1,101+TRACKING_SECONDS,1))
        self.assertNotIn('entered_at',r)
        self.assertEqual(advance_trade(r,1,101,1),'OPEN')
        self.assertIsNone(advance_trade(r,1.02,101+POST_ENTRY_TRACKING_SECONDS,2))
        self.assertEqual(r['tracking_status'],'OPEN') # timer loop owns TIMEOUT

    def test_old_terminal_result_not_reopened(self):
        r=self.record();r.update(tracking_status='TIMEOUT',finished_at=100+72*3600)
        self.assertIsNone(advance_trade(r,1.02,100+80*3600,1))
        self.assertEqual(r['tracking_status'],'TIMEOUT')

    def test_symbols_and_persisted_jobs(self):
        for symbol in ['BTCUSDT','1000PEPEUSDT','A_USDT']:
            self.assertTrue(supported_symbol(symbol))
        for symbol in ['牛来USDT','龙虾USDT','../BTCUSDT','BTC/USDT','btcUSDT',None,'A'*41+'USDT']:
            self.assertFalse(supported_symbol(symbol))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pending.json'
            good=dict(symbol='BTCUSDT',signal_id='ok',phase='signal',as_of=100.,retry_after=0)
            path.write_text(json.dumps({'ok':good,'bad':dict(good,symbol='牛来USDT')}))
            fetch=Mock();log=Mock()
            worker=HistoryCollector(directory,fetch,Mock(),lambda:False,log)
            self.assertEqual(list(worker.jobs),['ok'])
            self.assertEqual(list(json.loads(path.read_text())),['ok'])
            with self.assertRaises(ValueError):worker.collect(dict(good,symbol='../BTCUSDT'))
            fetch.assert_not_called()

    def test_history_queue_skip_does_not_retry_or_start_collector(self):
        tree=ast.parse(Path('ema.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_queue_history')
        g={'log':Mock(),'_CRASH_HISTORY_COLLECTOR':None}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'ema.py','exec'),g)
        r={'symbol':'牛来USDT','signal_id':'old'}
        g['_crash_queue_history'](r,'signal',100.)
        g['_crash_queue_history'](r,'signal',200.)
        self.assertIsNone(g['_CRASH_HISTORY_COLLECTOR'])
        self.assertEqual(r['history_excluded'],'NON_ASCII_OR_INVALID_SYMBOL')
        g['log'].assert_called_once()
