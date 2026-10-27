"""Offline regressions for 150-hour tracking and unsupported symbols."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from crash_policy import TRACKING_SECONDS, supported_symbol
from crash_live_v2 import advance_trade, MODEL
from crash_history import HistoryCollector

class PolicyTests(unittest.TestCase):
    def record(self):
        return dict(symbol='AUSDT',tracking_model=MODEL,tracking_status='WAIT_ENTRY',
                    created_at=100.,entry='1',tp='1.01',reference_low='.95',
                    previous_watch_price=.99)

    def test_late_entry_gets_own_full_150_hours(self):
        r=self.record(); t=100+149*3600
        r.update(ws_last_event_at=t-1,ws_last_trade_id=1)
        self.assertEqual(advance_trade(r,1,t,2),'OPEN')
        self.assertEqual(advance_trade(r,1.02,t+149*3600,3),'TP')

    def test_wait_and_open_deadlines(self):
        r=self.record()
        self.assertIsNone(advance_trade(r,1,101+TRACKING_SECONDS,1))
        self.assertNotIn('entered_at',r)
        self.assertEqual(advance_trade(r,1,101,1),'OPEN')
        self.assertIsNone(advance_trade(r,1.02,102+TRACKING_SECONDS,2))
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
