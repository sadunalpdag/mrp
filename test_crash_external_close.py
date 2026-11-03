"""Fail-closed manual Close and safety-latch regression checks; no exchange network."""
import copy
import tempfile
import unittest
from unittest.mock import Mock
from crash_night import NightExecutor, BrokerError, day
from crash_telegram_control import TelegramControl


class AccountBroker:
    def __init__(self):
        self.ps=[]
        self.tp={}
        self.pending={}
        self.max_symbols=20
        self.cooldown_until=0
        self.position_errors=None
    def order(self, symbol, client_id):
        if client_id not in self.tp:
            raise BrokerError('-2013')
        return copy.deepcopy(self.tp[client_id])
    def positions(self):
        if self.position_errors:
            raise self.position_errors
        return copy.deepcopy(self.ps)
    def closing_orders(self, symbol):
        return copy.deepcopy(self.pending.get(symbol, []))
    def open_entry_orders(self):
        return [copy.deepcopy(o) for orders in self.pending.values() for o in orders]
    def open_algo_orders(self):
        return []
    def info(self,*args):
        raise AssertionError('no live entry during recovery')
    def buy(self,*args):
        raise AssertionError('unexpected order submission')
    def tp_submit(self,*args):
        raise AssertionError('unexpected TP submission')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now=1793700000.0
        self.b=AccountBroker()
        self.out=Mock(return_value={'ok':True})
        self.e=NightExecutor(self.tmp.name,self.b,self.out,Mock(),
                             clock=lambda:self.now,enabled=True)
        self.addCleanup(self.e.guard.close)
    def add_order(self,symbol='AUSDT',qty='5',status='CANCELED',side='BOTH'):
        ident=symbol.lower().replace('usdt','')
        order={'symbol':symbol,'phase':'PROTECTED',
               'day':day(self.now),'entry_id':ident+'e','tp_id':ident+'t',
               'filled_qty':qty,'quantity':qty,'price':'10','tp':'10.1',
               'position_side':side}
        self.e.state['orders'][ident]=order
        self.b.tp[ident+'t']={'symbol':symbol,'side':'SELL','type':'LIMIT',
            'positionSide':side,'reduceOnly':side=='BOTH','origQty':qty,
            'executedQty':'0','price':'10.1','status':status}
        self.e.save()
        return order
    def position(self,symbol='AUSDT',qty='0',side='BOTH'):
        self.b.ps.append({'symbol':symbol,'positionSide':side,
                          'positionAmt':qty,'entryPrice':'10'})
    def test_full_manual_close_generalizes_to_two_coins_and_restarts(self):
        a=self.add_order()
        b=self.add_order(symbol='BUSDT')
        self.position()
        self.position(symbol='BUSDT')
        self.e.state['halt']='TP DOĞRULANAMADI\\nKod: TP_NOT_ACTIVE'
        self.e.reconcile(a)
        self.e.reconcile(b)
        self.assertEqual(a['phase'],'CLOSED_EXTERNALLY')
        self.assertEqual(b['phase'],'CLOSED_EXTERNALLY')
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(self.e.daily_count(),2)
        emitted=len(self.e.state['outbox'])
        self.e.reconcile(a)
        self.assertEqual(len(self.e.state['outbox']),emitted)
        self.e.guard.close()
        self.e=NightExecutor(self.tmp.name,self.b,self.out,Mock(),lambda:self.now,enabled=True)
        self.addCleanup(self.e.guard.close)
        self.assertEqual(self.e.state['orders']['a']['phase'],'CLOSED_EXTERNALLY')
        self.assertEqual(self.e.state['orders']['b']['phase'],'CLOSED_EXTERNALLY')
        self.assertTrue(self.e.state['halt'])
    def test_manual_close_empty_position_allows_owner_unlock_but_does_not_arm(self):
        order=self.add_order()
        self.position()
        self.e.state['halt']='TP DOĞRULANAMADI'
        self.e.reconcile(order)
        reply=self.e.apply_control('unlock')
        self.assertIn('kaldırıldı',reply)
        self.assertIsNone(self.e.state['halt'])
        self.assertFalse(self.e.enabled.is_set())
        self.assertTrue(self.e.state['paused'])
        self.assertIn('/otomatik_ac',self.e.state['outbox'][-1]['text'])
    def test_active_position_with_missing_tp_must_halt(self):
        order=self.add_order()
        self.position(qty='5')
        self.e.reconcile(order)
        self.assertTrue(self.e.state['halt'].startswith('TP DOĞRULANAMADI'))
        self.assertEqual(order['phase'],'PROTECTED')
        self.assertIn('Kilit korunuyor',self.e.apply_control('unlock'))
    def test_partial_close_is_not_flat_and_tp_mismatch_blocks_unlock(self):
        order=self.add_order()
        self.position(qty='2')
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'PROTECTED')
        self.assertTrue(self.e.state['halt'])
        self.assertIn('Kilit korunuyor',self.e.apply_control('unlock'))
    def test_binance_v3_omitted_flat_row_only_safe_when_orderbooks_empty(self):
        order=self.add_order()
        # Binance V3 omits symbols with no position or open orders.
        self.b.ps=[]
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'CLOSED_EXTERNALLY')
        self.assertFalse(self.e.state['halt'])
    def test_other_position_side_does_not_prove_flat_or_unlock(self):
        order=self.add_order(side='LONG')
        self.position(qty='-3',side='SHORT')
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'PROTECTED')
        self.assertTrue(self.e.state['halt'])
    def test_api_error_not_treated_as_flat(self):
        order=self.add_order()
        self.b.position_errors=BrokerError('-1003')
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'PROTECTED')
        self.assertFalse(self.e.state['halt'])
        self.assertIn('Kilit korunuyor',self.e.apply_control('unlock'))
    def test_open_pending_order_rejects_external_closed_claim(self):
        order=self.add_order()
        self.position()
        self.b.pending['AUSDT']=[{'symbol':'AUSDT','side':'BUY','status':'NEW'}]
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'PROTECTED')
        self.assertTrue(self.e.state['halt'])
    def test_tp_filled_is_real_tp_not_external(self):
        order=self.add_order(status='FILLED')
        self.position()
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'DONE')
    def test_not_found_tp_can_close_only_with_flat_and_empty_orderbook(self):
        order=self.add_order()
        self.position()
        self.b.tp.clear()
        self.e.reconcile(order)
        self.assertEqual(order['phase'],'CLOSED_EXTERNALLY')
    def test_unrelated_unprotected_manual_long_blocks_unlock(self):
        order=self.add_order()
        self.position()
        self.position('BUSDT',qty='4')
        self.e.state['halt']='TP DOĞRULANAMADI'
        self.e.reconcile(order)
        self.assertIn('BUSDT',self.e.apply_control('unlock'))
        self.assertTrue(self.e.state['halt'])
        self.b.pending['BUSDT']=[{'symbol':'BUSDT','side':'SELL','type':'LIMIT',
            'status':'NEW','positionSide':'BOTH','reduceOnly':True,
            'price':'10.1','origQty':'4','executedQty':'0'}]
        self.assertIn('kaldırıldı',self.e.apply_control('unlock'))
        self.assertFalse(self.e.enabled.is_set())
    def test_orphan_orders_on_flat_bot_symbol_block_unlock(self):
        order=self.add_order()
        self.position()
        self.e.state['halt']='TP DOĞRULANAMADI'
        self.e.reconcile(order)
        self.b.pending['AUSDT']=[{'symbol':'AUSDT','side':'SELL','type':'LIMIT',
            'status':'NEW','positionSide':'BOTH','reduceOnly':True,
            'price':'10.1','origQty':'5','executedQty':'0'}]
        self.assertIn('Kilit korunuyor',self.e.apply_control('unlock'))
        self.assertTrue(self.e.state['halt'])
    def test_unknown_entry_blocks_unlock_and_other_halts_are_not_reset(self):
        self.e.state['halt']='TP DOĞRULANAMADI'
        self.e.state['orders']['uncertain']={'symbol':'CUSDT','phase':'ENTRY_UNKNOWN'}
        self.assertIn('belirsiz giriş',self.e.apply_control('unlock'))
        self.e.state['orders'].clear()
        self.e.state['halt']='EXECUTOR_ERROR'
        self.assertIn('TP dışı',self.e.apply_control('unlock'))
    def test_owner_only_unlock_command(self):
        c=TelegramControl(self.tmp.name,'fake','123',None,Mock(),Mock(),Mock(),
                          clock=lambda:1000,fetch=Mock(),counter_only=True,
                          require_owner=False,allow_toggle=True)
        msg={'chat':{'id':123},'from':{'id':123,'is_bot':False},
             'date':1000,'text':'/otomatik_kilit_kaldir'}
        self.assertEqual(c.command({'message':msg}),'unlock')
        msg['from']['id']=456
        self.assertIsNone(c.command({'message':msg}))


if __name__=='__main__':
    unittest.main()
