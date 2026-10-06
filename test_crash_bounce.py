"""Offline tests: do not import ema.py's legacy startup/network side effects."""
import ast
import json
import math
import threading
import time
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class CrashBounceTests(unittest.TestCase):
    def setUp(self):
        names = {"crash_bounce_plan", "run_crash_bounce_alerts", "_crash_send_pending", "tg_send", "tg_send_file"}
        tree = ast.parse(Path(__file__).with_name("ema.py").read_text())
        subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        self.now = 2_000_000_000
        self.g = dict(Decimal=Decimal, math=math, time=SimpleNamespace(time=lambda: self.now),
                      CRASH_BOUNCE_ENTRY_BUFFER=Decimal("0.003"), CRASH_BOUNCE_TP_PCT=Decimal("0.01"),
                      CRASH_BOUNCE_DROP_PCT=12., CRASH_BOUNCE_COOLDOWN_SECONDS=86400,
                      CRASH_BOUNCE_TOP_LOSERS=20,
                      CRASH_BOUNCE_ONLY_TELEGRAM=True, _TG_COMMAND_CONTEXT=threading.local(),
                      BOT_TOKEN="test", CHAT_ID="test", requests=Mock(), log=Mock(),
                      _crash_save_state=Mock(), BinanceRateLimiter=SimpleNamespace(is_banned=lambda: False),
                      _crash_track_results=Mock(), _crash_features=Mock(return_value={}),
                      _crash_initialize_tracking=Mock(), _crash_append_data=Mock())
        exec(compile(subset, "ema.py", "exec"), self.g)
        self.rows = [[int((self.now - (16-i)*900)*1000), "1", "1.1", "0.99", "1", "1", int((self.now-(15-i)*900)*1000)-1] for i in range(16)]
        self.g['_crash_public_get'] = lambda path, params=None: self.rows if 'klines' in path else self.tickers
        self.g['_CRASH_WS_CACHE'] = SimpleNamespace(snapshot=lambda now: self.tickers, reconcile=Mock())
        self.g['_crash_get_candles'] = lambda symbol: self.rows
        self.g['tg_send'] = Mock(return_value=True)
        self.tickers = []

    def test_only_top_20_futures_ranked_before_cooldown(self):
        self.tickers = [dict(symbol=f'C{i:02}USDT', priceChangePercent=str(-40+i), lastPrice='1') for i in range(25)]
        self.tickers += [dict(symbol='SPOTUSDT', priceChangePercent='-99', lastPrice='1'),
                         dict(symbol='BADUSDT', priceChangePercent='NaN', lastPrice='1')]
        market = {t['symbol']:'.001' for t in self.tickers if t['symbol']!='SPOTUSDT'}
        state={'symbols':{'C00USDT':{'created_at':self.now, 'sent':True}}, 'history':[]}
        self.g['run_crash_bounce_alerts'](state, market, self.now)
        self.assertEqual(len(state['history']),19)
        self.assertEqual(set(state['symbols']), {f'C{i:02}USDT' for i in range(20)})

    def test_entry_and_tp_are_tick_aligned(self):
        plan = self.g['crash_bounce_plan'](self.rows, "1", "0.001", self.now*1000)
        self.assertEqual(plan['entry'], '0.993')
        self.assertEqual(plan['tp'], '1.003')
        self.assertGreaterEqual(Decimal(plan['tp'])/Decimal(plan['entry']), Decimal('1.01'))

    def test_unfinished_stale_and_missing_candles(self):
        row = list(self.rows[-1]); row[3] = '0.1'; row[6] = (self.now+900)*1000
        plan = self.g['crash_bounce_plan'](self.rows+[row], '1', '.001', self.now*1000)
        self.assertEqual(plan['entry'], '0.993')
        self.assertIsNone(self.g['crash_bounce_plan'](self.rows[:15], '1', '.001', self.now*1000))
        self.assertIsNone(self.g['crash_bounce_plan'](self.rows, '1', '.001', (self.now+3600)*1000))

    def test_threshold_universe_and_restart_duplicate_floor(self):
        self.tickers = [dict(symbol=s, priceChangePercent=d, lastPrice='1') for s,d in [('AUSDT','-12'),('BUSDT','-11.99'),('SPOTUSDT','-40')]]
        state = {'symbols':{}, 'history':[]}
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001','BUSDT':'.001'}, self.now)
        self.assertEqual(set(state['symbols']), {'AUSDT'})
        self.assertEqual(self.g['tg_send'].call_count, 1)
        restarted = json.loads(json.dumps(state))
        self.g['run_crash_bounce_alerts'](restarted, {'AUSDT':'.001'}, self.now+86399)
        self.assertEqual(self.g['tg_send'].call_count, 1)
        self.g['run_crash_bounce_alerts'](restarted, {'AUSDT':'.001'}, self.now+86400)
        self.assertEqual(self.g['tg_send'].call_count, 1)

    def test_failed_delivery_retries_same_plan_not_new_signal(self):
        self.tickers = [dict(symbol='AUSDT', priceChangePercent='-15', lastPrice='1')]
        state = {'symbols':{},'history':[]}
        self.g['tg_send'].return_value = False
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now)
        self.assertFalse(state['symbols']['AUSDT']['sent'])
        self.g['tg_send'].return_value = True
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now+60)
        self.assertTrue(state['symbols']['AUSDT']['sent'])
        self.assertEqual(len(state['history']), 1)

    def revision_fixture(self):
        self.tickers = [dict(symbol='AUSDT', priceChangePercent='-15', lastPrice='1')]
        state = {'symbols':{}, 'history':[]}
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now)
        return state

    def force_level(self, entry):
        self.g['crash_bounce_plan'] = Mock(return_value=dict(entry=entry, tp=str(float(entry)*1.01),
            reference_low=str(float(entry)/1.003), last_price='1', confirmation='test'))

    def test_revision_floor_exact_two_percent_and_restart(self):
        state = self.revision_fixture()
        old = Decimal(state['symbols']['AUSDT']['entry'])
        self.force_level(str(old*Decimal('.981')))
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now+180)
        self.assertEqual(len(state['history']),1)
        self.force_level(str(old*Decimal('.98')))
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now+360)
        self.assertEqual(len(state['history']),2)
        self.assertEqual(state['history'][-1]['daily_signal_number'],2)
        self.assertIn('#2',self.g['tg_send'].call_args.args[0])
        restarted = json.loads(json.dumps(state))
        self.g['run_crash_bounce_alerts'](restarted, {'AUSDT':'.001'}, self.now+540)
        self.assertEqual(len(restarted['history']),2)
        self.assertEqual(self.g['tg_send'].call_count,2)

    def test_after_tp_outside_top20_and_recovered_threshold_still_watched(self):
        state = self.revision_fixture()
        state['symbols']['AUSDT']['tracking_status']='TP'
        self.force_level('.95')
        self.tickers = [dict(symbol=f'C{i}USDT',priceChangePercent='-20',lastPrice='1') for i in range(20)]
        self.tickers.append(dict(symbol='AUSDT',priceChangePercent='-5',lastPrice='.94'))
        market={t['symbol']:'.001' for t in self.tickers}
        self.g['run_crash_bounce_alerts'](state,market,self.now+180)
        a=[r for r in state['history'] if r['symbol']=='AUSDT']
        self.assertEqual(len(a),2)
        self.assertEqual(a[-1]['previous_entry'],a[0]['entry'])
        self.assertIsNone(a[-1]['loser_rank'])
        self.assertIn('AUSDT',self.g['_CRASH_WS_CACHE'].reconcile.call_args.args[0])

    def test_paris_daily_number_resets_without_duplicate_floor_reset(self):
        state=self.revision_fixture()
        self.force_level('.95')
        self.g['run_crash_bounce_alerts'](state,{'AUSDT':'.001'},self.now+86400)
        self.assertEqual(state['history'][-1]['daily_signal_number'],1)
        self.assertNotEqual(state['history'][0]['signal_day'],state['history'][-1]['signal_day'])
        self.g['run_crash_bounce_alerts'](state,{'AUSDT':'.001'},self.now+86580)
        self.assertEqual(len(state['history']),2)

    def test_pending_delivery_is_not_overwritten_by_lower_revision(self):
        self.g['tg_send'].return_value=False
        state=self.revision_fixture()
        self.force_level('.9')
        self.g['run_crash_bounce_alerts'](state,{'AUSDT':'.001'},self.now+180)
        self.assertEqual(len(state['history']),1)
        self.assertNotEqual(state['symbols']['AUSDT']['entry'],'.9')

    def test_only_crash_automatic_messages_but_command_replies_work(self):
        tree = ast.parse(Path(__file__).with_name('ema.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='tg_send')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'ema.py','exec'),self.g)
        response = self.g['requests'].post.return_value
        response.json.return_value = {'ok':True}
        self.assertFalse(self.g['tg_send']('old strategy'))
        self.g['requests'].post.assert_not_called()
        self.assertTrue(self.g['tg_send']('crash', source='CRASH_BOUNCE'))
        self.g['_TG_COMMAND_CONTEXT'].active = True
        self.assertTrue(self.g['tg_send']('explicit /status reply'))

    def test_persistence_failure_cannot_send_unrecorded_alert(self):
        self.tickers = [dict(symbol='AUSDT', priceChangePercent='-15', lastPrice='1')]
        state = {'symbols':{},'history':[]}
        self.g['_crash_save_state'].side_effect = OSError('disk unavailable')
        self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now)
        self.g['tg_send'].assert_not_called()
        with self.assertRaises(OSError):
            self.g['run_crash_bounce_alerts'](state, {'AUSDT':'.001'}, self.now+60)
        self.g['tg_send'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
