"""Offline failure-path tests. No network, credentials, or real orders."""
import ast
import copy
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from crash_heartbeat import Heartbeat, INTERVAL
from crash_night import NightExecutor, BinanceBroker, BrokerError, plan, night, day, ZONE


def timestamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=ZONE).timestamp()


INFO = dict(status='TRADING', contractType='PERPETUAL', filters=[
    dict(filterType='PRICE_FILTER', tickSize='0.001', minPrice='0.001', maxPrice='100000'),
    dict(filterType='LOT_SIZE', stepSize='0.1', minQty='0.1', maxQty='100000'),
    dict(filterType='MIN_NOTIONAL', notional='5')])


class HeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.now = timestamp('2026-10-27T21:00:00')
        self.deliver = Mock(return_value={'ok': True, 'message_id': 1})
        self.health = dict(connected=True, ticker_age_seconds=1, ticker_count=100)
        self.h = self.make()

    def make(self):
        return Heartbeat(self.tmp.name, self.deliver, lambda:self.health, Mock(), lambda:self.now)

    def test_two_hour_interval_survives_restart(self):
        self.h.step(); self.deliver.assert_not_called()
        self.now += 180
        self.h.update(scan_completed_at=self.now, candles_ok=3)
        self.h.step(); self.assertEqual(self.deliver.call_count, 1)
        self.h = self.make()
        self.now += INTERVAL-1
        self.h.step(); self.assertEqual(self.deliver.call_count, 1)
        self.now += 1
        self.h.step(); self.assertEqual(self.deliver.call_count, 2)

    def test_stale_scan_stream_and_empty_data_not_claimed_healthy(self):
        self.h.update(scan_completed_at=self.now, candles_ok=1)
        self.assertTrue(self.h.message(self.now).startswith('✅'))
        self.health['ticker_age_seconds'] = 121
        self.assertTrue(self.h.message(self.now).startswith('⚠️'))
        self.health['ticker_age_seconds'] = 1
        self.assertTrue(self.h.message(self.now+601).startswith('⚠️'))
        self.h.update(candles_ok=0, candles_wait=3)
        self.assertTrue(self.h.message(self.now).startswith('⚠️'))
        self.h.update(candles_wait=0)
        self.assertTrue(self.h.message(self.now).startswith('✅'))  # no eligible candidates is normal
        self.h.update(scan_errors=1)
        self.assertTrue(self.h.message(self.now).startswith('⚠️'))

    def test_signal_delivery_failure_marks_health_warning(self):
        self.h.update(scan_completed_at=self.now, candles_ok=10, delivery_error='HTTP_500')
        self.assertTrue(self.h.message(self.now).startswith('⚠️'))

    def test_failed_delivery_retries_without_flood(self):
        self.now += 180
        self.deliver.return_value = {'ok':False, 'error':'HTTP_500'}
        self.h.step(); self.h.step()
        self.assertEqual(self.deliver.call_count,1)
        self.now += 300
        self.h.step(); self.assertEqual(self.deliver.call_count,2)

    def test_ambiguous_delivery_does_not_duplicate_immediately(self):
        self.now += 180
        self.deliver.return_value = {'ok':False, 'ambiguous':True}
        self.h.step()
        self.now += 300
        self.h.step(); self.assertEqual(self.deliver.call_count,1)

    def test_restart_cannot_claim_previous_process_scan_is_current(self):
        self.h.update(scan_completed_at=self.now, candles_ok=10)
        self.h.save(); self.now += 1
        h = self.make()
        self.assertTrue(h.message(self.now).startswith('⚠️'))


class FakeBroker:
    def __init__(self):
        self.orders = {}; self.buys = []; self.tps = []
        self.fill = '5'; self.fail_tp = False; self.unknown_buy = False

    def info(self, symbol): return copy.deepcopy(INFO)
    def preflight(self, symbol): pass

    def buy(self, symbol, p, client_id):
        self.buys.append((symbol, p, client_id))
        self.orders[client_id] = dict(status='FILLED' if self.fill != '0' else 'EXPIRED', executedQty=self.fill)
        if self.unknown_buy: raise BrokerError('TIMEOUT')
        return self.orders[client_id]

    def tp(self, symbol, qty, price, client_id):
        self.tps.append((symbol, qty, price, client_id))
        if self.fail_tp: raise BrokerError('TP_REJECTED')
        self.orders[client_id] = dict(side='SELL', reduceOnly=True, origQty=qty, price=price, status='NEW')
        return self.orders[client_id]

    def order(self, symbol, client_id):
        if client_id not in self.orders: raise BrokerError('NOT_FOUND')
        return self.orders[client_id]


class NightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.now = timestamp('2026-10-27T21:00:00')
        self.b = FakeBroker(); self.deliver=Mock(return_value={'ok':True})
        self.e = self.make()
        self.r = dict(symbol='AUSDT', signal_id='signal-1', entry='10', tp='10.10',
                      v2_status='STRONG_V2', v2_sent=True, tracking_status='OPEN')

    def make(self):
        e = NightExecutor(self.tmp.name, self.b, self.deliver, Mock(), lambda:self.now)
        self.addCleanup(e.guard.close)
        return e

    def enter(self, **changes):
        self.e.enter(dict(self.r, **changes), self.now)

    def test_night_and_paris_day_boundaries(self):
        for value, allowed in [('07:59:59', True), ('08:00:00', False), ('19:59:59', False), ('20:00:00', True)]:
            self.assertEqual(night(timestamp('2026-10-27T'+value)), allowed)
        self.assertNotEqual(day(timestamp('2026-10-27T23:59:59')), day(timestamp('2026-10-28T00:00:00')))
        self.assertTrue(night(timestamp('2026-10-25T02:30:00')))  # DST night

    def test_valid_entry_protected_and_deduplicated(self):
        self.enter(); self.e.deliver_pending(); self.enter()
        self.assertEqual(len(self.b.buys),1)
        self.assertEqual(len(self.b.tps),1)
        self.assertEqual(self.e.state['orders']['signal-1']['phase'],'PROTECTED')

    def test_partial_fill_tp_only_filled_quantity(self):
        self.b.fill='2.3'; self.enter()
        self.assertEqual(self.b.tps[0][1],'2.3')

    def test_no_fill_no_tp_and_counts_attempt(self):
        self.b.fill='0'; self.enter()
        self.assertFalse(self.b.tps)
        self.assertIn('bugün 1/10',self.e.summary())
        self.assertEqual(self.e.state['orders']['signal-1']['phase'],'NO_FILL')

    def test_tp_failure_halts_and_alerts_without_closing_or_duplicate(self):
        self.b.fail_tp=True; self.enter()
        self.assertEqual(self.e.state['halt'],'TP DOĞRULANAMADI')
        self.e.deliver_pending(); self.enter(signal_id='signal-2',symbol='BUSDT')
        self.e.step()
        self.assertEqual(len(self.b.buys),1); self.assertEqual(len(self.b.tps),1)
        self.assertIn('otomatik kapatılmadı', self.deliver.call_args.args[0])

    def test_unknown_buy_response_reconciles_without_repeat(self):
        self.b.unknown_buy=True; self.enter()
        self.assertEqual(self.e.state['orders']['signal-1']['phase'],'PROTECTED')
        self.e.deliver_pending(); self.enter()
        self.assertEqual(len(self.b.buys),1)

    def test_restart_reserved_entry_is_query_only(self):
        self.e.state['orders']['signal-1'] = dict(day=day(self.now), symbol='AUSDT', phase='ENTRY_UNKNOWN', entry_id='missing', tp_id='tp', quantity='5', price='10', tp='10.1')
        self.e.save(); self.e.guard.close()
        self.e=self.make(); self.e.step()
        self.assertFalse(self.b.buys)
        self.assertEqual(self.e.state['halt'],'GİRİŞ SONUCU BELİRSİZ')

    def test_daily_limit_persisted_and_resets_at_midnight(self):
        for i in range(10):
            self.e.state['orders'][str(i)]=dict(day=day(self.now),phase='DONE',symbol='OLDUSDT')
        self.e.save(); self.e.guard.close(); self.e=self.make()
        self.enter(); self.assertFalse(self.b.buys)
        self.now=timestamp('2026-10-28T00:00:01')
        self.enter(); self.assertEqual(len(self.b.buys),1)

    def test_stale_unsent_nonascii_and_daytime_do_not_enter(self):
        self.e.enter(self.r,self.now-6)
        self.enter(v2_sent=False); self.enter(symbol='龙虾USDT')
        self.now=timestamp('2026-10-28T12:00:00'); self.enter()
        self.assertFalse(self.b.buys)

    def test_failed_alert_blocks_new_entry_then_retries(self):
        self.e.alert('test'); self.deliver.return_value={'ok':False}
        self.e.step(); self.enter(); self.assertFalse(self.b.buys)
        self.deliver.return_value={'ok':True}; self.now+=300
        self.e.step(); self.enter(); self.assertEqual(len(self.b.buys),1)

    def test_cancelled_tp_detected_outside_night(self):
        self.enter(); self.e.deliver_pending()
        o=self.e.state['orders']['signal-1']; self.b.orders[o['tp_id']]['status']='CANCELED'
        self.now=timestamp('2026-10-28T12:00:00'); self.e.step()
        self.assertEqual(self.e.state['halt'],'TP DOĞRULANAMADI')

    def test_bad_state_and_multiple_processes_fail_closed(self):
        with self.assertRaises(BlockingIOError): self.make()
        self.e.guard.close()
        Path(self.e.path).write_text('{bad')
        with self.assertRaises(ValueError): self.make()

    def test_missing_previously_initialized_state_fails_closed(self):
        self.e.guard.close(); os.unlink(self.e.path)
        with self.assertRaises(ValueError): self.make()

    def test_plan_rounding_and_minimum_notional(self):
        p=plan(dict(self.r,entry='10.0004',tp='10.10001'),INFO)
        self.assertEqual(p['price'],'10.000'); self.assertEqual(p['tp'],'10.101')
        info=copy.deepcopy(INFO); info['filters'][-1]['notional']='100'
        with self.assertRaisesRegex(ValueError,'MIN_NOTIONAL'): plan(self.r,info)

    def test_symbol_preflight_fail_sends_alert_no_entry(self):
        self.b.preflight=Mock(side_effect=ValueError('POSITION_EXISTS_OR_MAX_5'))
        self.enter(); self.assertFalse(self.b.buys)
        self.assertIn('GİRİŞ YAPILMADI',self.e.state['outbox'][0]['text'])


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.b=BinanceBroker('fake','fake')

    def test_hedge_mode_rejected_without_mutation(self):
        self.b.request=Mock(return_value={'dualSidePosition':True})
        with self.assertRaises(ValueError): self.b.preflight('AUSDT')
        self.assertEqual(self.b.request.call_count,1)

    def test_existing_positions_max_five_and_pending_orders(self):
        for positions, orders in [([{'symbol':'AUSDT','positionAmt':'1'}],[]),
                                  ([{'symbol':str(i),'positionAmt':'1'} for i in range(5)],[]),
                                  ([],[{'symbol':str(i)} for i in range(5)]),
                                  ([],[{'symbol':'AUSDT'}])]:
            self.b.request=Mock(side_effect=[{'dualSidePosition':False}, positions, orders])
            with self.assertRaises(ValueError): self.b.preflight('AUSDT')
            self.assertFalse(any(c.args[0]=='POST' for c in self.b.request.call_args_list))

    def test_leverage_ack_required(self):
        self.b.request=Mock(side_effect=[{'dualSidePosition':False},[],[],{'leverage':5}])
        with self.assertRaisesRegex(ValueError,'LEVERAGE_NOT_CONFIRMED'): self.b.preflight('AUSDT')

    def test_tp_is_reduce_only_limit_not_market(self):
        self.b.request=Mock()
        self.b.tp('AUSDT','5','10.1','id')
        p=self.b.request.call_args.args[2]
        self.assertEqual((p['type'],p['side'],p['reduceOnly']),('LIMIT','SELL','true'))


class StartupTests(unittest.TestCase):
    def test_default_off_does_not_construct_broker_or_executor(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_crash_start_services'],type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch('crash_heartbeat.Heartbeat') as heartbeat, patch('crash_night.BinanceBroker') as broker:
            g=dict(os=os,time=SimpleNamespace(time=lambda:1),DATA_DIR=tmp,log=Mock(),_crash_telegram_receipt=Mock(),_CRASH_WS_CACHE=Mock())
            exec(compile(subset,'ema.py','exec'),g)
            g['_crash_start_services']()
            broker.assert_not_called()
            self.assertIsNone(g['_CRASH_NIGHT'])
            heartbeat.return_value.start.assert_called_once()
