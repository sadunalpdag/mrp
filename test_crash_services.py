"""Offline failure-path tests. No network, credentials, or real orders."""
import ast
import copy
import json
import os
import tempfile
import unittest
from decimal import Decimal as D
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
        e = NightExecutor(self.tmp.name, self.b, self.deliver, Mock(), lambda:self.now, enabled=True, max_daily=10)
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

    def test_daytime_entry_allowed_with_same_notional(self):
        self.now=timestamp('2026-10-28T12:00:00')
        self.enter()
        self.assertEqual(len(self.b.buys),1)
        p=self.b.buys[0][1]
        self.assertLessEqual(D(p['quantity'])*D(p['price']),D('50'))

    def test_hedge_entry_and_tp_survive_reconcile(self):
        self.b.preflight=lambda symbol:'LONG'
        original=self.b.tp
        def hedge_tp(symbol,qty,price,cid,position_side='BOTH'):
            result=original(symbol,qty,price,cid)
            result.update(positionSide=position_side,reduceOnly=False)
            return result
        self.b.tp=hedge_tp
        self.enter()
        order=self.e.state['orders']['signal-1']
        self.assertEqual(order['position_side'],'LONG')
        self.assertEqual(order['phase'],'PROTECTED')
        self.e.reconcile(order)
        self.assertEqual(len(self.b.tps),1)
        self.assertEqual(order['phase'],'PROTECTED')

    def pending_entry(self):
        original=self.b.buy
        def pending(symbol,p,cid):
            result=original(symbol,p,cid)
            result.update(status='NEW',executedQty='0')
            return result
        self.b.buy=pending
        self.enter(tracking_status='WAIT_ENTRY')
        return self.e.state['orders']['signal-1']

    def test_gtc_pending_survives_restart_then_fill_protects_once(self):
        order=self.pending_entry()
        self.assertFalse(self.b.tps);self.assertIsNone(self.e.state['halt'])
        self.assertIn('ALIŞ LİMİTİ BEKLİYOR',self.e.state['outbox'][-1]['text'])
        self.e.save();self.e.guard.close();self.e=self.make()
        order=self.e.state['orders']['signal-1']
        self.e.reconcile(order);self.assertEqual(len(self.b.buys),1)
        self.b.orders[order['entry_id']].update(status='FILLED',executedQty='5')
        self.e.reconcile(order);self.e.reconcile(order)
        self.assertEqual(len(self.b.tps),1);self.assertEqual(order['phase'],'PROTECTED')

    def test_gtc_partial_cancels_remainder_and_protects_final_race_fill(self):
        order=self.pending_entry()
        self.b.orders[order['entry_id']].update(status='PARTIALLY_FILLED',executedQty='2')
        def cancel(symbol,cid):
            self.b.orders[cid].update(status='CANCELED',executedQty='3')
        self.b.cancel_entry=Mock(side_effect=cancel)
        self.e.reconcile(order)
        self.b.cancel_entry.assert_called_once()
        self.assertEqual(self.b.tps[0][1],'3')

    def test_gtc_cancel_uncertain_does_not_assume_final_quantity(self):
        order=self.pending_entry()
        self.b.orders[order['entry_id']].update(status='PARTIALLY_FILLED',executedQty='2')
        self.b.cancel_entry=Mock(side_effect=BrokerError('UNKNOWN'))
        self.e.reconcile(order)
        self.assertFalse(self.b.tps);self.assertTrue(self.e.state['halt'])
        self.assertEqual(order['phase'],'ENTRY_UNKNOWN')

    def test_candidate_immediate_fill_without_upward_cross(self):
        self.enter(tracking_status='WAIT_ENTRY')
        self.assertEqual(len(self.b.buys),1)
        self.assertEqual(self.e.state['orders']['signal-1']['phase'],'PROTECTED')

    def test_gtc_user_cancels_unfilled_order_no_tp(self):
        order=self.pending_entry()
        self.b.orders[order['entry_id']].update(status='CANCELED',executedQty='0')
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'NO_FILL');self.assertFalse(self.b.tps)

    def test_counter_override_persists_increments_and_resets_next_day(self):
        self.assertEqual(self.e.daily_count(),0)
        self.enter();self.assertEqual(self.e.daily_count(),1)
        self.e.apply_control('counter:0')
        self.assertEqual(self.e.daily_count(),0)
        self.assertEqual(len(self.e.state['orders']),1)
        self.e.save();self.e.guard.close();self.e=self.make()
        self.assertEqual(self.e.daily_count(),0)
        self.e.state['orders']['second']=dict(day=day(self.now),phase='DONE',symbol='BUSDT')
        self.assertEqual(self.e.daily_count(),1)
        self.e.apply_control('counter:10')
        self.assertEqual(self.e.daily_count(),10)
        self.now=timestamp('2026-10-28T00:00:00')
        self.assertEqual(self.e.daily_count(),0)
        self.assertEqual(len(self.e.state['orders']),2)

    def test_counter_invalid_values_do_not_change_quota(self):
        for value in ('-1','11','x','1.5'):
            self.e.apply_control('counter:'+value)
            self.assertEqual(self.e.daily_count(),0)

    def test_old_unknown_history_marker_no_longer_blocks_first_entry(self):
        self.e.state['unverified_day']=day(self.now)
        self.e.save();self.e.guard.close();self.e=self.make()
        self.assertNotIn('unverified_day',self.e.state)
        self.enter();self.assertEqual(len(self.b.buys),1)

    def test_pending_entry_expires_at_eight_hours_not_before(self):
        order=self.pending_entry()
        def cancel(symbol,cid):
            self.b.orders[cid].update(status='CANCELED',executedQty='0')
        self.b.cancel_entry=Mock(side_effect=cancel)
        self.now+=8*3600-1;self.e.reconcile(order)
        self.b.cancel_entry.assert_not_called()
        self.now+=1;self.e.reconcile(order)
        self.b.cancel_entry.assert_called_once()
        self.assertEqual(order['phase'],'NO_FILL');self.assertFalse(self.b.tps)
        self.assertIn('8 saat',self.e.state['outbox'][-1]['text'])

    def test_expiry_racing_fill_gets_tp_instead_of_no_fill(self):
        order=self.pending_entry();self.now+=8*3600
        def cancel(symbol,cid):
            self.b.orders[cid].update(status='FILLED',executedQty='5')
        self.b.cancel_entry=Mock(side_effect=cancel)
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'PROTECTED');self.assertEqual(self.b.tps[0][1],'5')

    def test_legacy_order_uses_exchange_creation_time(self):
        order=self.pending_entry();order.pop('submitted_at')
        self.b.orders[order['entry_id']]['time']=(self.now-8*3600)*1000
        def cancel(symbol,cid):self.b.orders[cid].update(status='CANCELED')
        self.b.cancel_entry=Mock(side_effect=cancel)
        self.e.reconcile(order);self.b.cancel_entry.assert_called_once()

    def test_orphan_sweep_only_expires_own_buy_not_tp_or_manual(self):
        cid='cbn'+'a'*24+'e'
        old=dict(symbol='AUSDT',clientOrderId=cid,side='BUY',type='LIMIT',positionSide='LONG',time=(self.now-8*3600)*1000)
        tp=dict(old,side='SELL',clientOrderId='cbn'+'a'*24+'t')
        manual=dict(old,clientOrderId='manual-order')
        fresh=dict(old,clientOrderId='cbn'+'b'*24+'e',time=self.now*1000)
        self.b.open_entry_orders=Mock(return_value=[old,tp,manual,fresh])
        self.b.cancel_entry=Mock()
        self.b.orders[cid]=dict(status='CANCELED',executedQty='0')
        self.e.expire_untracked_entries()
        self.b.cancel_entry.assert_called_once_with('AUSDT',cid)

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
        self.assertTrue(self.e.state['halt'].startswith('TP DOĞRULANAMADI'))
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
        self.assertTrue(self.e.state['halt'].startswith('GİRİŞ SONUCU BELİRSİZ'))

    def test_daily_limit_persisted_and_resets_at_midnight(self):
        for i in range(10):
            self.e.state['orders'][str(i)]=dict(day=day(self.now),phase='DONE',symbol='OLDUSDT')
        self.e.save(); self.e.guard.close(); self.e=self.make()
        self.enter(); self.assertFalse(self.b.buys)
        self.now=timestamp('2026-10-28T00:00:01')
        self.enter(); self.assertEqual(len(self.b.buys),1)

    def test_stale_unsent_nonascii_do_not_enter(self):
        self.e.enter(self.r,self.now-121)
        self.enter(v2_sent=False); self.enter(symbol='龙虾USDT')
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
        self.assertTrue(self.e.state['halt'].startswith('TP DOĞRULANAMADI'))

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

    def test_buy_uses_waiting_gtc_with_limit_cap(self):
        self.b.request=Mock()
        self.b.buy('AUSDT',dict(quantity='2',price='22.5',position_side='LONG'),'cid')
        p=self.b.request.call_args.args[2]
        self.assertEqual(p['timeInForce'],'GTC');self.assertEqual(p['price'],'22.5')
        self.assertEqual(p['type'],'LIMIT');self.assertEqual(p['side'],'BUY')

    def test_hedge_mode_supported_without_changing_account_mode(self):
        self.b.request=Mock(side_effect=[{'dualSidePosition':True},[],[],[],{'leverage':10}])
        self.assertEqual(self.b.preflight('AUSDT'),'LONG')
        self.assertFalse(any(c.args[0]=='POST' and c.args[1]=='/fapi/v1/positionSide/dual' for c in self.b.request.call_args_list))

    def test_existing_positions_max_five_and_pending_orders(self):
        for positions, orders in [([{'symbol':'AUSDT','positionAmt':'1'}],[]),
                                  ([{'symbol':str(i),'positionAmt':'1'} for i in range(5)],[]),
                                  ([],[{'symbol':str(i)} for i in range(5)]),
                                  ([],[{'symbol':'AUSDT'}])]:
            self.b.request=Mock(side_effect=[{'dualSidePosition':False}, positions, orders])
            with self.assertRaises(ValueError): self.b.preflight('AUSDT')
            self.assertFalse(any(c.args[0]=='POST' for c in self.b.request.call_args_list))

    def test_three_pending_two_closing_orders_only_count_three_slots(self):
        orders=[dict(symbol=f'P{i}',side='BUY',positionSide='LONG') for i in range(3)]
        orders += [dict(symbol='C1',side='SELL',positionSide='LONG')]
        algo=[dict(symbol='C2',side='SELL',positionSide='BOTH',closePosition='true')]
        self.b.request=Mock(side_effect=[{'dualSidePosition':True},[],orders,algo,{'leverage':10}])
        self.assertEqual(self.b.preflight('AUSDT'),'LONG')

    def test_three_pending_plus_two_positions_still_five_not_seven(self):
        positions=[dict(symbol='C1',positionAmt='1'),dict(symbol='C2',positionAmt='1')]
        orders=[dict(symbol=f'P{i}',side='BUY',positionSide='LONG') for i in range(3)]
        orders += [dict(symbol=s,side='SELL',positionSide='LONG') for s in ('C1','C2')]
        self.b.request=Mock(side_effect=[{'dualSidePosition':True},positions,orders])
        with self.assertRaisesRegex(ValueError,'açık=2 giriş=3 toplam=5 kapatma_hariç=2'):
            self.b.preflight('AUSDT')

    def test_closing_classifier_preserves_unknown_and_short_entry_slots(self):
        for order in (dict(reduceOnly=True),dict(closePosition='true'),
                      dict(positionSide='LONG',side='SELL'),dict(positionSide='SHORT',side='BUY')):
            self.assertTrue(self.b.is_closing_order(order))
        for order in ({},dict(positionSide='SHORT',side='SELL'),dict(positionSide='BOTH',side='SELL'),
                      dict(type='TAKE_PROFIT',side='SELL',reduceOnly=False)):
            self.assertFalse(self.b.is_closing_order(order))

    def test_leverage_ack_required(self):
        self.b.request=Mock(side_effect=[{'dualSidePosition':False},[],[],[],{'leverage':5}])
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
