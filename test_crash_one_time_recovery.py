"""Offline live-recovery tests: no keys, no network, no real orders."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from crash_night import BinanceBroker, NightExecutor, day
from test_crash_external_close import AccountBroker


class OneTimeBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1793700000.0
        self.b = AccountBroker()
        self.b.is_closing_order = BinanceBroker.is_closing_order
        self.log = Mock()
        self.receipt = Mock(return_value={'ok': True})
        self.patch = patch.dict(os.environ, {'CRASH_BOOTSTRAP_LIVE_ONCE': '1'})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.e = NightExecutor(self.tmp.name, self.b, self.receipt, self.log,
                               clock=lambda: self.now, enabled=True)
        self.addCleanup(self.e.guard.close)
        self.e.state['halt'] = 'STATE_HISTORY_UNVERIFIED'
        self.e.state['paused'] = True
        self.e.enabled.clear()
        self.e.save()
        self.e.guard.close()
        self.e = NightExecutor(self.tmp.name, self.b, self.receipt, self.log,
                               clock=lambda: self.now, enabled=True)
        self.addCleanup(self.e.guard.close)

    def test_double_snapshot_clear_account_arms_only_once(self):
        self.assertTrue(self.e.auto_bootstrap_requested)
        self.e.step()
        self.assertIsNone(self.e.state['halt'])
        self.assertTrue(self.e.enabled.is_set())
        self.assertFalse(self.e.state['paused'])
        self.assertTrue(self.e.state['bootstrap_completed_20261103'])
        self.assertFalse(self.b.tp)
        self.assertTrue(any('BOOTSTRAP VERIFIED' in str(c) for c in self.log.call_args_list))
        self.e.apply_control('off')
        self.e.guard.close()
        self.e = NightExecutor(self.tmp.name, self.b, self.receipt, self.log,
                               clock=lambda: self.now, enabled=True)
        self.assertFalse(self.e.auto_bootstrap_requested)
        self.e.step()
        self.assertFalse(self.e.enabled.is_set())

    def test_manual_off_aborts_pending_bootstrap(self):
        self.e.apply_control('off')
        self.e.step()
        self.assertTrue(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())
        self.assertTrue(self.e.state['bootstrap_cancelled_20261103'])

    def test_open_buy_blocks_recovery(self):
        self.b.pending['AUSDT'] = [{'symbol': 'AUSDT', 'side': 'BUY',
                                    'positionSide': 'BOTH', 'status': 'NEW'}]
        self.e.step()
        self.assertEqual(self.e.state['halt'], 'STATE_HISTORY_UNVERIFIED')
        self.assertFalse(self.e.enabled.is_set())

    def test_open_unprotected_long_blocks_recovery(self):
        self.b.ps.append({'symbol': 'AUSDT', 'positionSide': 'BOTH',
                          'positionAmt': '5', 'entryPrice': '10'})
        self.e.step()
        self.assertTrue(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())

    def test_protected_long_can_recover(self):
        self.b.ps.append({'symbol': 'AUSDT', 'positionSide': 'BOTH',
                          'positionAmt': '5', 'entryPrice': '10'})
        self.b.pending['AUSDT'] = [{
            'symbol': 'AUSDT', 'side': 'SELL', 'positionSide': 'BOTH',
            'reduceOnly': True, 'type': 'LIMIT', 'status': 'NEW',
            'price': '10.1', 'origQty': '5', 'executedQty': '0'}]
        self.e.step()
        self.assertIsNone(self.e.state['halt'])
        self.assertTrue(self.e.enabled.is_set())

    def test_orphan_reduce_only_close_is_quarantined_not_global_halt(self):
        self.b.pending['AUSDT'] = [{
            'symbol': 'AUSDT', 'side': 'SELL', 'positionSide': 'BOTH',
            'reduceOnly': True, 'type': 'LIMIT', 'status': 'NEW',
            'price': '10.1', 'origQty': '5', 'executedQty': '0'}]
        self.e.step()
        self.assertIsNone(self.e.state['halt'])
        self.assertTrue(self.e.enabled.is_set())
        self.assertEqual(self.e.state['quarantined_symbols'], ['AUSDT'])
        self.b.pending.clear()
        self.now += 301
        self.e.refresh_quarantined_symbols()
        self.assertEqual(self.e.state['quarantined_symbols'], [])

    def test_orphan_opening_sell_still_blocks_recovery(self):
        self.b.pending['AUSDT'] = [{
            'symbol': 'AUSDT', 'side': 'SELL', 'positionSide': 'BOTH',
            'reduceOnly': False, 'type': 'LIMIT', 'status': 'NEW',
            'price': '10.1', 'origQty': '5', 'executedQty': '0'}]
        self.e.step()
        self.assertTrue(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())

    def test_quarantined_symbol_cannot_enter(self):
        self.e.state['halt'] = None
        self.e.state['paused'] = False
        self.e.enabled.set()
        self.e.state['quarantined_symbols'] = ['AUSDT']
        self.e.enter({'signal_id':'fresh', 'symbol':'AUSDT', 'entry':'10', 'tp':'10.1',
                      'v2_status':'STRONG_V2', 'v2_sent':True,
                      'tracking_status':'OPEN'}, self.now)
        self.assertFalse(self.e.state['orders'])
        self.assertFalse(self.b.tp)

    def test_api_failure_blocks_recovery_without_mutating_history(self):
        from crash_night import BrokerError
        self.b.position_errors = BrokerError('-1003')
        self.e.step()
        self.assertTrue(self.e.state['halt'])
        self.assertNotIn('bootstrap_completed_20261103', self.e.state)

    def test_inconsistent_double_snapshot_denies(self):
        calls = 0
        def inconsistent():
            nonlocal calls
            calls += 1
            if calls == 1:
                return []
            return [{'symbol': 'AUSDT', 'positionSide': 'BOTH',
                     'positionAmt': '5', 'entryPrice': '10'}]
        self.b.positions = inconsistent
        self.e.step()
        self.assertTrue(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())


class FlatCloseAutoUnhaltTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1793700000.0
        self.b = AccountBroker()
        self.emit = Mock(return_value={'ok': True})
        self.e = NightExecutor(self.tmp.name, self.b, self.emit, Mock(),
                               clock=lambda: self.now, enabled=True)
        self.addCleanup(self.e.guard.close)
        self.order = {
            'symbol': 'ANYUSDT', 'day': day(self.now), 'phase': 'PROTECTED',
            'entry_id': 'a-e', 'tp_id': 'a-t', 'filled_qty': '5',
            'quantity': '5', 'price': '10', 'tp': '10.1',
            'position_side': 'BOTH'}
        self.e.state['orders']['a'] = self.order
        self.b.tp['a-t'] = {
            'symbol': 'ANYUSDT', 'side': 'SELL', 'type': 'LIMIT',
            'positionSide': 'BOTH', 'reduceOnly': True,
            'origQty': '5', 'price': '10.1', 'status': 'CANCELED'}

    def test_transient_close_halt_clears_after_confirmed_manual_close(self):
        self.b.ps = [{'symbol': 'ANYUSDT', 'positionSide': 'BOTH',
                      'positionAmt': '5', 'entryPrice': '10'}]
        self.e.reconcile(self.order)
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(self.e.state['halt_symbol'], 'ANYUSDT')
        self.b.ps = []
        self.e.reconcile(self.order)
        self.assertEqual(self.order['phase'], 'CLOSED_EXTERNALLY')
        self.e.recover_flat_close_halt()
        self.assertIsNone(self.e.state['halt'])
        self.assertTrue(self.e.enabled.is_set())

    def test_unprotected_other_position_keeps_halt(self):
        self.e.halt('TP DOĞRULANAMADI', 'ANYUSDT')
        self.b.ps = [{'symbol': 'BUSDT', 'positionSide': 'BOTH',
                      'positionAmt': '5', 'entryPrice': '10'}]
        self.e.reconcile(self.order)
        self.assertEqual(self.order['phase'], 'CLOSED_EXTERNALLY')
        self.e.recover_flat_close_halt()
        self.assertTrue(self.e.state['halt'])

    def test_manual_close_with_reduce_only_orphan_does_not_halt(self):
        self.b.pending['ANYUSDT'] = [{
            'symbol': 'ANYUSDT', 'side':'SELL','positionSide':'BOTH',
            'reduceOnly': True, 'type':'LIMIT', 'status':'NEW',
            'price':'10.1', 'origQty':'5', 'executedQty':'0'}]
        self.e.reconcile(self.order)
        self.assertEqual(self.order['phase'], 'CLOSED_EXTERNALLY')
        self.assertFalse(self.e.state['halt'])
        self.assertEqual(self.e.state['quarantined_symbols'], ['ANYUSDT'])
        self.assertTrue(self.e.enabled.is_set())

    def test_manual_pause_is_preserved_when_close_recovered(self):
        self.e.halt('TP DOĞRULANAMADI', 'ANYUSDT')
        self.e.apply_control('off')
        self.e.reconcile(self.order)
        self.e.recover_flat_close_halt()
        self.assertIsNone(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())


if __name__ == '__main__':
    unittest.main()
