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
                      v2_status='STRONG_V2', v2_sent=True, tracking_status='OPEN',
                      failure_shadow=dict(version='tp_nonclosure_entry_gate_v1',
                          flags=dict(weak_volume=False, falling_rsi_ema=False,
                                     repeat_lower_entry=False)))

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
        self.assertLessEqual(D(p['quantity'])*D(p['price']),D('500'))

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

    def test_manual_scan_runs_every_five_minutes(self):
        self.e.manual_tp_enabled=True;self.e.manual_guard=Mock()
        self.e.step()
        self.now+=5;self.e.step()
        self.assertEqual(self.e.manual_guard.scan.call_count,1)
        self.now+=295;self.e.step()
        self.assertEqual(self.e.manual_guard.scan.call_count,2)

    def test_order_and_expiry_poll_intervals(self):
        self.e.state['orders'] = {
            'a': dict(phase='ENTRY_UNKNOWN',entry_id='a'),
            'b': dict(phase='TP_UNKNOWN',entry_id='b'),
            'c': dict(phase='PROTECTED',entry_id='c')}
        self.e.reconcile = Mock()
        self.e.expire_untracked_entries = Mock()
        self.e.step()
        self.assertEqual(self.e.reconcile.call_count,3)
        self.now += 299; self.e.step()
        self.assertEqual(self.e.reconcile.call_count,3)
        self.now += 1; self.e.step()
        self.assertEqual(self.e.reconcile.call_count,6)
        self.assertFalse(self.e.reconcile.call_args.kwargs['check_expiry'])
        self.assertEqual(self.e.expire_untracked_entries.call_count,1)
        self.now += 600; self.e.step()
        self.assertTrue(self.e.reconcile.call_args.kwargs['check_expiry'])
        self.assertEqual(self.e.expire_untracked_entries.call_count,2)

    def test_rate_pause_keeps_control_and_telegram_without_exchange_poll(self):
        self.b.cooldown_until=self.now+300
        self.e.manual_tp_enabled=True;self.e.manual_guard=Mock()
        self.e.alert('test');self.e.step()
        self.e.manual_guard.scan.assert_not_called()
        self.assertFalse(self.e.state['outbox'])

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

    def make_halted_cancelled_tp(self):
        self.enter()
        self.e.deliver_pending()
        order = self.e.state['orders']['signal-1']
        self.b.orders[order['tp_id']]['status'] = 'CANCELED'
        self.e.reconcile(order)
        self.assertTrue(self.e.state['halt'].startswith('TP DOĞRULANAMADI'))
        self.assertTrue(order['warned'])
        return order

    def test_exchange_confirmed_manual_close_does_not_create_tp_halt(self):
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.assertEqual(order['phase'], 'PROTECTED')
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        # Another coin may stay open: only the affected symbol must be flat.
        self.b.positions = Mock(return_value=[
            dict(symbol='AUSDT', positionAmt='0', positionSide='BOTH'),
            dict(symbol='BUSDT', positionAmt='5', positionSide='BOTH')])
        self.b.closing_orders = Mock(return_value=[])
        with patch('crash_night.threading.Thread') as thread:
            self.e.reconcile(order)
            thread.assert_called_once()
        self.assertEqual(order['phase'], 'CLOSED_EXTERNALLY')
        self.assertEqual(order['external_close_reason'], 'TP_NOT_ACTIVE_POSITION_FLAT')
        self.assertEqual(order['close_verification'],
                         'LIVE_POSITION_ZERO_NO_OPEN_ORDERS')
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual((len(self.b.buys), len(self.b.tps)), (1, 1))
        self.assertEqual(self.b.positions.call_count, 2)
        self.b.closing_orders.assert_called_once_with('AUSDT')
        self.assertIn('CRASH EXTERNAL CLOSE CONFIRMED',
                      '\n'.join(c.args[0] for c in self.e.log.call_args_list))

    def test_missing_tp_with_live_position_keeps_safety_halt(self):
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        self.b.positions = Mock(return_value=[
            dict(symbol='AUSDT', positionAmt='5', positionSide='BOTH')])
        self.b.closing_orders = Mock(return_value=[])
        self.e.reconcile(order)
        self.assertEqual(order['phase'], 'PROTECTED')
        self.assertTrue(self.e.state['halt'].startswith('TP DOĞRULANAMADI'))
        self.b.closing_orders.assert_not_called()
        self.assertIn('OPEN_POSITION',
                      '\n'.join(c.args[0] for c in self.e.log.call_args_list))

    def test_manual_close_with_open_normal_or_algo_order_isolates_coin(self):
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        self.b.positions = Mock(return_value=[])
        opened = [dict(symbol='AUSDT', side='SELL', type='TAKE_PROFIT_MARKET',
                       algoStatus='NEW', closePosition=True)]
        self.b.closing_orders = Mock(return_value=opened)
        with patch('crash_night.threading.Thread') as thread:
            self.e.reconcile(order)
            thread.assert_called_once()
        self.assertEqual(order['phase'], 'CLOSED_EXTERNALLY')
        self.assertEqual(order['close_verification'], 'LIVE_POSITION_ZERO_QUARANTINED')
        self.assertIsNone(self.e.state['halt'])
        self.assertIn('AUSDT', self.e.state['symbol_quarantine'])
        self.assertEqual(self.b.positions.call_count, 2)
        self.assertEqual(opened[0]['algoStatus'], 'NEW')
        self.assertEqual((len(self.b.buys), len(self.b.tps)), (1, 1))
        self.assertIn('CRASH SYMBOL QUARANTINE',
                      '\n'.join(c.args[0] for c in self.e.log.call_args_list))

    def test_manual_close_verification_failure_does_not_unlock(self):
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        self.b.positions = Mock(side_effect=BrokerError('-1003'))
        self.b.closing_orders = Mock(return_value=[])
        self.e.reconcile(order)
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(order['phase'], 'PROTECTED')
        self.assertIn('VERIFY_FAILED=-1003',
                      '\n'.join(c.args[0] for c in self.e.log.call_args_list))
        self.b.positions.side_effect = None
        self.b.positions.return_value = [dict(symbol='AUSDT', positionAmt='NaN')]
        self.assertFalse(self.e._confirm_external_close(order))
        self.assertEqual(order['phase'], 'PROTECTED')

    def test_manual_close_recheck_must_be_still_flat(self):
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        self.b.positions = Mock(side_effect=[
            [dict(symbol='AUSDT', positionAmt='0')],
            [dict(symbol='AUSDT', positionAmt='2')]])
        self.b.closing_orders = Mock(return_value=[])
        self.e.reconcile(order)
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(order['phase'], 'PROTECTED')
        self.assertIn('POSITION_CHANGED',
                      '\n'.join(c.args[0] for c in self.e.log.call_args_list))

    def test_manual_close_does_not_clear_unknown_buy_for_same_symbol(self):
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.e.state['orders']['uncertain'] = {
            'symbol': 'AUSDT', 'phase': 'ENTRY_UNKNOWN', 'entry_id': 'unknown'}
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[])
        self.e.reconcile(order)
        self.assertEqual(order['phase'], 'PROTECTED')
        self.assertTrue(self.e.state['halt'])
        self.b.positions.assert_not_called()

    def test_persisted_warned_manual_close_can_release_prior_tp_halt(self):
        order = self.make_halted_cancelled_tp()
        order['phase'] = 'CLOSED_EXTERNALLY'
        order['close_verification'] = 'LIVE_POSITION_ZERO_NO_OPEN_ORDERS'
        self.e.save()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[])
        self.assertTrue(self.e.recover_closed_tp_halt())
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(order['phase'], 'CLOSED_EXTERNALLY')
        self.assertTrue(self.e.enabled.is_set())

    def test_manually_closed_flat_tp_halt_recovers_safely(self):
        order = self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[
            dict(symbol='AUSDT', positionAmt='0', positionSide='BOTH')])
        self.b.closing_orders = Mock(return_value=[])
        self.now += 10
        self.assertTrue(self.e.recover_closed_tp_halt())
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(order['phase'], 'CLOSED_EXTERNALLY')
        self.assertEqual(order['close_verification'],
                         'LIVE_POSITION_ZERO_NO_OPEN_ORDERS')
        self.assertEqual(self.e.daily_count(), 1)
        self.assertEqual(len(self.b.buys), 1)
        self.assertEqual(len(self.b.tps), 1)
        self.assertTrue(self.e.enabled.is_set())
        self.assertIn('TP HATA KİLİDİ TEMİZLENDİ',
                      self.e.state['outbox'][-1]['text'])
        self.e.save(); self.e.guard.close(); self.e = self.make()
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(self.e.state['orders']['signal-1']['phase'],
                         'CLOSED_EXTERNALLY')

    def test_tp_halt_kept_while_position_still_open(self):
        order = self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[
            dict(symbol='AUSDT', positionAmt='5', positionSide='BOTH')])
        self.b.closing_orders = Mock(return_value=[])
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.b.closing_orders.assert_not_called()
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(order['phase'], 'PROTECTED')

    def test_tp_halt_with_open_order_isolates_coin_and_unlocks_other_buys(self):
        order = self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[
            dict(symbol='AUSDT', positionAmt='0'),
            dict(symbol='BUSDT', positionAmt='5')])
        opened = [dict(symbol='AUSDT', side='SELL', type='LIMIT', status='NEW')]
        self.b.closing_orders = Mock(return_value=opened)
        with patch('crash_night.threading.Thread'):
            self.assertTrue(self.e.recover_closed_tp_halt())
        self.assertIsNone(self.e.state['halt'])
        self.assertTrue(self.e.enabled.is_set())
        self.assertEqual(order['phase'], 'CLOSED_EXTERNALLY')
        self.assertEqual(order['close_verification'], 'LIVE_POSITION_ZERO_QUARANTINED')
        self.assertIn('AUSDT', self.e.state['symbol_quarantine'])
        self.assertEqual(opened[0]['status'], 'NEW')  # Never cancel user orders.
        self.e.deliver_pending()
        self.enter(signal_id='new-ake', symbol='AUSDT')
        self.assertEqual(len(self.b.buys), 1)
        self.e.deliver_pending()
        self.enter(signal_id='new-busdt', symbol='BUSDT')
        self.assertEqual(len(self.b.buys), 2)
        self.assertEqual(self.b.buys[-1][0], 'BUSDT')

    def test_quarantine_survives_restart_and_preserves_explicit_off(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[
            dict(symbol='AUSDT', side='SELL', status='NEW')])
        with patch('crash_night.threading.Thread'):
            self.assertTrue(self.e.recover_closed_tp_halt())
        self.e.apply_control('off')
        self.e.guard.close()
        self.e = self.make()
        self.assertIn('AUSDT', self.e.state['symbol_quarantine'])
        self.assertFalse(self.e.enabled.is_set())
        self.assertTrue(self.e.state['paused'])
        self.enter(signal_id='fresh-busdt', symbol='BUSDT')
        self.assertEqual(len(self.b.buys), 1)  # Manual OFF never overridden.

    def test_quarantine_recheck_clears_after_live_flat_and_zero_orders(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[
            dict(symbol='AUSDT', side='SELL', status='NEW')])
        with patch('crash_night.threading.Thread'):
            self.assertTrue(self.e.recover_closed_tp_halt())
        with patch('crash_night.threading.Thread'):
            self.e.review_quarantined_symbols()
        self.assertIn('AUSDT', self.e.state['symbol_quarantine'])
        self.b.closing_orders.return_value = []
        with patch('crash_night.threading.Thread'):
            self.e.review_quarantined_symbols()
        self.assertNotIn('AUSDT', self.e.state['symbol_quarantine'])
        self.assertIsNone(self.e.state['halt'])
        self.e.deliver_pending()
        self.enter(signal_id='new-entry-after-clean', symbol='AUSDT')
        self.assertEqual(len(self.b.buys), 2)

    def test_unknown_bot_buy_does_not_downgrade_global_safety_hold(self):
        order = self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[
            dict(symbol='AUSDT', side='BUY', type='LIMIT', status='NEW',
                 clientOrderId='cbn'+'a'*24+'e')])
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(order['phase'], 'PROTECTED')
        self.assertNotIn('AUSDT', self.e.state['symbol_quarantine'])
        self.assertIn('UNVERIFIED_OR_BOT_PENDING_ORDER',
                      '\n'.join(c.args[0] for c in self.e.log.call_args_list))

    def test_malformed_or_mismatched_order_not_quarantined(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[dict(symbol='BUSDT', side='SELL')])
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.assertTrue(self.e.state['halt'])
        self.assertFalse(self.e.state['symbol_quarantine'])

    def test_quarantine_recheck_preserves_hold_when_position_open_or_api_fails(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[
            dict(symbol='AUSDT', side='SELL', status='NEW')])
        with patch('crash_night.threading.Thread'):
            self.assertTrue(self.e.recover_closed_tp_halt())
        self.b.positions.return_value = [dict(symbol='AUSDT', positionAmt='2')]
        self.e.review_quarantined_symbols()
        self.assertIn('AUSDT', self.e.state['symbol_quarantine'])
        self.b.positions.side_effect = BrokerError('-1003')
        self.e.review_quarantined_symbols()
        self.assertIn('AUSDT', self.e.state['symbol_quarantine'])
        self.assertFalse(self.e.state['halt'])
        self.assertEqual(len(self.b.buys), 1)

    def test_tp_halt_kept_if_live_account_query_fails_or_is_invalid(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(side_effect=BrokerError('-1003'))
        self.b.closing_orders = Mock(return_value=[])
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.b.positions.side_effect = None
        self.b.positions.return_value = {'not': 'a verified snapshot'}
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.b.positions.return_value = []
        self.b.closing_orders.return_value = {'unknown': 'open orders'}
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.assertTrue(self.e.state['halt'])

    def test_tp_halt_kept_if_another_entry_or_unverified_tp_exists(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[])
        self.e.state['orders']['uncertain'] = dict(symbol='BUSDT', phase='ENTRY_UNKNOWN')
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.e.state['orders']['uncertain']['phase'] = 'TP_UNKNOWN'
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.b.positions.assert_not_called()
        self.assertTrue(self.e.state['halt'])

    def test_halt_recovery_never_overrides_explicit_off_or_other_errors(self):
        self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[])
        self.e.apply_control('off')
        self.assertTrue(self.e.recover_closed_tp_halt())
        self.assertFalse(self.e.enabled.is_set())
        self.assertTrue(self.e.state['paused'])
        self.e.state['halt'] = 'EXECUTOR_ERROR'
        self.assertFalse(self.e.recover_closed_tp_halt())
        self.assertEqual(self.e.state['halt'], 'EXECUTOR_ERROR')

    def test_step_clears_only_verified_flat_tp_halt_after_old_close(self):
        order = self.make_halted_cancelled_tp()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[])
        self.now += 301
        self.e.step()
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(order['phase'], 'CLOSED_EXTERNALLY')
        self.b.positions.assert_called_once()
        self.b.closing_orders.assert_called_once_with('AUSDT')
        self.assertEqual(len(self.b.buys), 1)

    def test_local_budget_deferred_buy_never_halts_or_counts_daily(self):
        self.b.buy = Mock(side_effect=BrokerError('-1003',
            deferred_reason='IP_WEIGHT_SOFT_CEILING',
            deferred_seconds=11, not_sent=True))
        self.enter()
        order = self.e.state['orders']['signal-1']
        self.assertEqual(order['phase'], 'NO_FILL')
        self.assertTrue(order['entry_never_sent'])
        self.assertEqual(self.e.daily_count(), 0)
        self.assertIsNone(self.e.state['halt'])
        self.assertFalse(self.b.tps)
        self.assertEqual(self.b.buy.call_count, 1)
        self.assertIn('ALIŞ GÖNDERİLMEDİ', self.e.state['outbox'][-1]['text'])
        self.e.deliver_pending()
        self.enter(signal_id='signal-2', symbol='BUSDT')
        self.assertEqual(self.b.buy.call_count, 2)
        self.assertEqual(self.e.daily_count(), 0)

    def test_local_preflight_ceiling_retry_is_nonblocking_and_places_once(self):
        # The first preflight never reached BUY: schedule one check after the
        # minute reset, with the original upcross time and full live preflight.
        self.b.preflight = Mock(side_effect=[
            BrokerError('-1003', deferred_reason='IP_WEIGHT_SOFT_CEILING',
                        deferred_seconds=21, not_sent=True), None])
        self.enter()
        self.assertFalse(self.b.buys)
        self.assertFalse(self.e.state['outbox'])
        self.assertEqual(self.e.preflight_retries['signal-1']['attempts'], 1)
        self.assertEqual(self.e.daily_count(), 0)
        self.now += 20
        self.e.step()
        self.assertFalse(self.b.buys)  # Not yet due; safety checks continue.
        self.now += 3
        self.e.step()
        self.assertEqual(len(self.b.buys), 1)
        self.assertEqual(len(self.b.tps), 1)
        self.assertEqual(self.e.state['orders']['signal-1']['phase'], 'PROTECTED')
        self.assertFalse(self.e.preflight_retries)
        self.assertEqual(self.e.daily_count(), 1)

    def test_two_local_preflight_retries_then_final_deferral_no_buy(self):
        err = BrokerError('-1003', deferred_reason='IP_WEIGHT_SOFT_CEILING',
                          deferred_seconds=10, not_sent=True)
        self.b.preflight = Mock(side_effect=[err, err, err])
        self.enter()
        self.now += 12
        self.e.step()
        self.assertEqual(self.e.preflight_retries['signal-1']['attempts'], 2)
        self.now += 12
        self.e.step()
        self.assertFalse(self.b.buys)
        self.assertFalse(self.e.preflight_retries)
        self.assertEqual(self.e.daily_count(), 0)
        self.assertIsNone(self.e.state['halt'])
        self.assertIn('LOCAL_IP_BUDGET', self.e.state['outbox'][-1]['text'])

    def test_real_exchange_error_does_not_retry_preflight(self):
        self.b.preflight=Mock(side_effect=BrokerError('-1003'))
        self.enter()
        self.assertFalse(self.e.preflight_retries)
        self.assertFalse(self.b.buys)
        self.assertTrue(self.e.state['outbox'])

    def test_expired_local_preflight_is_never_executed_after_original_window(self):
        self.b.preflight=Mock(side_effect=BrokerError(
            '-1003',deferred_reason='IP_WEIGHT_SOFT_CEILING',
            deferred_seconds=45,not_sent=True))
        self.enter()
        self.assertEqual(len(self.e.preflight_retries),1)
        self.now += 121
        self.e.step()
        self.assertFalse(self.b.buys)
        self.assertFalse(self.e.preflight_retries)
        self.b.preflight.assert_called_once()

    def test_manual_off_cancels_pending_preflight_without_buy(self):
        self.b.preflight=Mock(side_effect=BrokerError(
            '-1003',deferred_reason='IP_WEIGHT_SOFT_CEILING',
            deferred_seconds=10,not_sent=True))
        self.enter()
        self.e.apply_control('off')
        self.now += 12
        self.e.step()
        self.assertFalse(self.b.buys)
        self.assertFalse(self.e.preflight_retries)
        self.assertFalse(self.e.enabled.is_set())

    def test_persisted_local_deferral_keeps_daily_counter_consistent(self):
        self.b.buy = Mock(side_effect=BrokerError('-1003',
            not_sent=True))
        self.enter()
        self.assertEqual(self.e.daily_count(), 0)
        self.e.apply_control('counter:2')
        self.assertEqual(self.e.daily_count(), 2)
        self.e.guard.close(); self.e=self.make()
        self.assertEqual(self.e.daily_count(), 2)

    def add_unknown_halt(self):
        self.e.state['orders']['old']=dict(
            day=day(self.now), symbol='AUSDT', entry_id='cbn'+'a'*24+'e',
            quantity='5', phase='ENTRY_UNKNOWN', warned=True, price='10', tp='10.1')
        self.e.state['halt'] = 'GİRİŞ SONUCU BELİRSİZ'
        self.e.save()
        self.b.positions=Mock(return_value=[])
        self.b.closing_orders=Mock(return_value=[])
        return self.e.state['orders']['old']

    def make_mixed_halt(self):
        tp = self.make_halted_cancelled_tp()
        old = dict(day=day(self.now), symbol='BUSDT',
                   entry_id='cbn'+'b'*24+'e', quantity='5',
                   phase='ENTRY_UNKNOWN', warned=True, price='10', tp='10.1')
        self.e.state['orders']['old'] = old
        self.e.save()
        self.b.positions = Mock(return_value=[])
        self.b.closing_orders = Mock(return_value=[])
        self.b.order = Mock(side_effect=BrokerError('-2013'))
        return tp, old

    def test_mixed_halt_manual_close_recovers_both_atomically(self):
        tp, old = self.make_mixed_halt()
        count = self.e.daily_count()
        buys, tps = len(self.b.buys), len(self.b.tps)
        self.assertTrue(self.e.recover_mixed_halt())
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(old['phase'], 'NO_FILL')
        self.assertEqual(old['entry_reconcile_proof'], 'ORDER_NOT_FOUND_FLAT')
        self.assertEqual(tp['phase'], 'CLOSED_EXTERNALLY')
        self.assertEqual(tp['close_verification'], 'LIVE_POSITION_ZERO_NO_OPEN_ORDERS')
        self.assertEqual(self.e.daily_count(), count)
        self.assertEqual((len(self.b.buys), len(self.b.tps)), (buys, tps))
        self.assertTrue(self.e.enabled.is_set())
        self.assertEqual(self.b.closing_orders.call_count, 2)

    def test_mixed_halt_kept_if_position_or_algo_order_exists(self):
        tp, old = self.make_mixed_halt()
        self.b.positions.return_value = [dict(symbol='BUSDT', positionAmt='1')]
        self.assertFalse(self.e.recover_mixed_halt())
        self.b.closing_orders.assert_not_called()
        self.b.positions.return_value = []
        self.b.closing_orders.side_effect = lambda symbol: (
            [dict(symbol=symbol, status='NEW')] if symbol == 'BUSDT' else [])
        self.assertFalse(self.e.recover_mixed_halt())
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(old['phase'], 'ENTRY_UNKNOWN')
        self.assertEqual(tp['phase'], 'PROTECTED')
        self.b.order.assert_not_called()

    def test_mixed_halt_never_unlocks_unknown_client_or_api_failure(self):
        tp, old = self.make_mixed_halt()
        self.b.order.side_effect = BrokerError('-1003')
        self.assertFalse(self.e.recover_mixed_halt())
        self.b.order.side_effect = None
        self.b.order.return_value = dict(symbol='BUSDT', clientOrderId=old['entry_id'],
                                         side='BUY', status='NEW', executedQty='0')
        self.assertFalse(self.e.recover_mixed_halt())
        self.assertEqual(old['phase'], 'ENTRY_UNKNOWN')
        self.assertEqual(tp['phase'], 'PROTECTED')
        self.assertTrue(self.e.state['halt'])

    def test_mixed_halt_auto_recheck_and_explicit_off(self):
        tp, old = self.make_mixed_halt()
        self.e.apply_control('off')
        self.now += 301
        self.e.step()
        self.assertIsNone(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())
        self.assertTrue(self.e.state['paused'])
        self.assertEqual(old['phase'], 'NO_FILL')
        self.assertEqual(tp['phase'], 'CLOSED_EXTERNALLY')
        self.assertFalse(self.b.buys[1:] if len(self.b.buys) > 1 else [])

    def test_entry_skip_reports_reason_without_duplicate_notice(self):
        self.e.state['halt'] = 'TP DOĞRULANAMADI'
        self.e.enter(dict(self.r), self.now)
        self.assertIn('CRASH ENTRY SKIP', self.e.log.call_args.args[0])
        self.assertIn('HALTED', self.e.log.call_args.args[0])
        self.assertFalse(self.e.state['outbox'])
        self.e.enter(dict(self.r), self.now)
        self.assertFalse(self.e.state['outbox'])
        self.assertFalse(self.b.buys)
        with patch('crash_night.threading.Thread') as thread:
            self.e.entry_skip('AUSDT', 'DAILY_LIMIT')
            self.e.entry_skip('AUSDT', 'DAILY_LIMIT')
            thread.assert_called_once()
        # Informational notifications must never block legitimate future BUYs.
        self.assertFalse(self.e.state['outbox'])

    def test_30_min_lock_audit_shows_pending_conditions(self):
        self.e.state['halt'] = 'EXECUTOR_ERROR'
        self.e.state['orders']['unknown'] = dict(phase='ENTRY_UNKNOWN')
        self.e.reconcile = Mock()
        self.e.expire_untracked_entries = Mock()
        self.e.step()
        logs = [c.args[0] for c in self.e.log.call_args_list]
        self.assertTrue(any('CRASH LOCK AUDIT 30M' in x
                            and 'entry_unknown=1' in x for x in logs))
        self.e.log.reset_mock()
        self.now += 1799
        self.e.step()
        self.assertFalse(any('CRASH LOCK AUDIT 30M' in c.args[0]
                             for c in self.e.log.call_args_list))
        self.now += 1
        self.e.step()
        self.assertTrue(any('CRASH LOCK AUDIT 30M' in c.args[0]
                            for c in self.e.log.call_args_list))

    def test_stale_entry_halt_safe_recovery_when_client_order_not_found(self):
        old=self.add_unknown_halt()
        self.b.order=Mock(side_effect=BrokerError('-2013'))
        self.assertTrue(self.e.recover_unknown_entry_halt())
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(old['phase'], 'NO_FILL')
        self.assertEqual(old['entry_reconcile_proof'], 'ORDER_NOT_FOUND_FLAT')
        self.b.positions.assert_called_once()
        self.b.closing_orders.assert_called_once_with('AUSDT')
        self.assertTrue(self.e.enabled.is_set())
        self.e.guard.close(); self.e=self.make()
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(self.e.state['orders']['old']['phase'], 'NO_FILL')

    def test_stale_entry_halt_recovers_filled_then_manual_close_without_tp_claim(self):
        old=self.add_unknown_halt()
        self.b.order=Mock(return_value=dict(
            symbol='AUSDT', clientOrderId=old['entry_id'],
            side='BUY', status='FILLED', executedQty='5'))
        self.assertTrue(self.e.recover_unknown_entry_halt())
        self.assertEqual(old['phase'], 'CLOSED_EXTERNALLY')
        self.assertEqual(self.e.daily_count(), 1)

    def test_stale_entry_halt_never_unlocks_open_position_order_or_api_error(self):
        old=self.add_unknown_halt()
        self.b.order=Mock(side_effect=BrokerError('-2013'))
        self.b.positions.return_value=[dict(symbol='AUSDT', positionAmt='5')]
        self.assertFalse(self.e.recover_unknown_entry_halt())
        self.b.positions.return_value=[]
        self.b.closing_orders.return_value=[dict(symbol='AUSDT', status='NEW')]
        self.assertFalse(self.e.recover_unknown_entry_halt())
        self.b.closing_orders.return_value=[]
        self.b.order.side_effect=BrokerError('-1003')
        self.assertFalse(self.e.recover_unknown_entry_halt())
        self.b.order.side_effect=BrokerError('UNKNOWN', uncertain=True)
        self.assertFalse(self.e.recover_unknown_entry_halt())
        self.assertEqual(old['phase'], 'ENTRY_UNKNOWN')
        self.assertTrue(self.e.state['halt'])

    def test_stale_entry_halt_rejects_inconsistent_entry_and_respects_off(self):
        old=self.add_unknown_halt()
        self.b.order=Mock(return_value=dict(
            symbol='AUSDT',clientOrderId=old['entry_id'],
            side='BUY',status='NEW',executedQty='0'))
        self.assertFalse(self.e.recover_unknown_entry_halt())
        self.b.order.return_value.update(status='CANCELED')
        self.e.apply_control('off')
        self.assertTrue(self.e.recover_unknown_entry_halt())
        self.assertFalse(self.e.enabled.is_set())
        self.assertTrue(self.e.state['paused'])

    def test_recover_halt_step_after_5min_verification(self):
        old=self.add_unknown_halt()
        self.b.order=Mock(side_effect=BrokerError('-2013'))
        self.e.step()
        self.assertIsNone(self.e.state['halt'])
        self.assertEqual(old['phase'], 'NO_FILL')
        self.b.positions.assert_called_once()
        self.b.order.assert_called_once()
        self.assertFalse(self.b.buys)

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
        self.budget_patch = patch('crash_rest_budget.BUDGET', Mock())
        self.budget_patch.start()
        self.addCleanup(self.budget_patch.stop)

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

    def test_preflight_reports_pending_bot_orders_without_extra_http_or_cancel(self):
        import re
        self.b.max_symbols=20
        bot=dict(symbol='BUSDT',side='BUY',type='LIMIT',
                 clientOrderId='cbn'+'a'*24+'e',positionSide='BOTH')
        manual=dict(symbol='CUSDT',side='BUY',type='LIMIT',
                    clientOrderId='manual-order',positionSide='BOTH')
        tp=dict(symbol='DUSDT',side='SELL',type='LIMIT',
                reduceOnly=True,positionSide='BOTH')
        self.b.request=Mock(side_effect=[
            {'dualSidePosition':False},[],[bot,manual,tp],[],
            {'leverage':10}])
        with patch('builtins.print') as output:
            self.assertEqual(self.b.preflight('AUSDT'),'BOTH')
        self.assertEqual(self.b.request.call_count,5)
        emitted=' '.join(str(arg) for call in output.call_args_list for arg in call.args)
        self.assertIn('bot_pending_buy=1',emitted)
        self.assertIn('normal_open=3',emitted)
        self.assertIn('no orders cancelled',emitted)
        self.assertNotIn('DELETE',[c.args[0] for c in self.b.request.call_args_list])

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

    def test_rate_limit_respects_retry_after_and_blocks_followup_http(self):
        response=Mock(status_code=429,headers={'Retry-After':'300'})
        response.json.return_value={'code':-1003,'msg':'Too many requests'}
        http=Mock(return_value=response)
        with patch.dict('sys.modules', {'requests':SimpleNamespace(request=http)}),patch('crash_night.time.time',return_value=1000):
            with self.assertRaises(BrokerError):self.b.positions()
            self.assertEqual(self.b.cooldown_until,1300)
            with self.assertRaises(BrokerError):self.b.positions()
            self.assertEqual(http.call_count,1)

    def test_ip_ban_until_timestamp_is_honored(self):
        response=Mock(status_code=418,headers={})
        response.json.return_value={'code':-1003,'msg':'IP banned until 1793220000000'}
        with patch.dict('sys.modules', {'requests':SimpleNamespace(request=Mock(return_value=response))}),patch('crash_night.time.time',return_value=1793210000):
            with self.assertRaises(BrokerError):self.b.positions()
            self.assertGreaterEqual(self.b.cooldown_until,1793220005)

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
