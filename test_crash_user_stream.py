"""Offline account stream lifecycle and trading-event regression tests."""
import json
import tempfile
import unittest
from unittest.mock import Mock
from crash_night import NightExecutor, BrokerError, day
from crash_user_stream import AccountStream
from test_crash_services import FakeBroker


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.now=2000000000.
        self.b=FakeBroker()
        self.e=NightExecutor(self.tmp.name,self.b,Mock(return_value={'ok':True}),Mock(),clock=lambda:self.now)
        self.addCleanup(self.e.guard.close)
        self.o=dict(symbol='AUSDT',day=day(self.now),phase='ENTRY_UNKNOWN',entry_kind='GTC',
            entry_id='entry',tp_id='tp',quantity='5',price='10',tp='10.1',submitted_at=self.now)
        self.e.state['orders']['signal']=self.o
        self.e.next_poll=self.e.next_expiry_scan=self.now+900

    def event(self,cid='entry',status='FILLED',qty='5',**changes):
        order=dict(s='AUSDT',c=cid,S='BUY' if cid=='entry' else 'SELL',ps='BOTH',o='LIMIT',
                   X=status,z=qty,q='5',p='10' if cid=='entry' else '10.1',R=cid=='tp')
        order.update(changes)
        return dict(e='ORDER_TRADE_UPDATE',E=int(self.now*1000),o=order)

    def test_fill_immediate_tp_and_duplicate_is_not_resubmitted(self):
        self.e.on_account_event(self.event());self.e.step()
        self.assertEqual(len(self.b.tps),1)
        self.assertEqual(self.o['phase'],'PROTECTED')
        self.e.on_account_event(self.event());self.e.step()
        self.assertEqual(len(self.b.tps),1)
        self.e.on_account_event(self.event(cid='tp'));self.e.step()
        self.assertEqual(self.o['phase'],'DONE')

    def test_partial_fill_cancels_remainder_and_uses_final_rest_quantity(self):
        self.b.cancel_entry=Mock(side_effect=lambda s,c:self.b.orders.update({c:dict(status='CANCELED',executedQty='3')}))
        self.e.on_account_event(self.event(status='PARTIALLY_FILLED',qty='2'));self.e.step()
        self.b.cancel_entry.assert_called_once()
        self.assertEqual(self.b.tps[0][1],'3')

    def test_wrong_side_and_stale_events_cannot_submit_tp(self):
        self.e.on_account_event(self.event(S='SELL'));self.e.process_account_events()
        self.assertFalse(self.b.tps)
        self.e.account_resync.clear()
        old=self.event();old['E']-=60000
        self.e.on_account_event(old);self.e.process_account_events()
        self.assertFalse(self.b.tps)
        self.assertTrue(self.e.account_resync.is_set())

    def test_cooldown_keeps_events_for_later_authoritative_reconciliation(self):
        self.b.cooldown_until=self.now+300
        self.e.on_account_event(self.event());self.e.step()
        self.assertEqual(self.e.account_events.qsize(),1)
        self.assertFalse(self.b.tps)
        self.now+=301
        self.e.process_account_events()
        self.assertTrue(self.e.account_resync.is_set())
        self.assertFalse(self.b.tps)

    def test_position_delta_triggers_only_changed_manual_symbols(self):
        self.e.manual_tp_enabled=True;self.e.manual_guard=Mock()
        self.e.on_account_event(dict(e='ACCOUNT_UPDATE',E=int(self.now*1000),a={'P':[
            {'s':'BUSDT','ps':'LONG','pa':'1','ep':'10'},
            {'s':'AUSDT','ps':'BOTH','pa':'5','ep':'10'}]}))
        self.e.process_account_events()
        self.e.manual_guard.scan.assert_called_once_with(symbols={'BUSDT'})
        self.e.on_account_event(dict(e='ACCOUNT_UPDATE',E=int(self.now*1000),a={'B':[]}))
        self.e.process_account_events()
        self.assertEqual(self.e.manual_guard.scan.call_count,1)

    def test_reconnect_resync_throttles_and_uses_rest(self):
        self.b.orders['entry']=dict(status='NEW',executedQty='0')
        self.b.order=Mock(wraps=self.b.order)
        self.e.on_account_event({'e':'ACCOUNT_STREAM_RESYNC'});self.e.step()
        self.b.order.assert_called_once()
        self.e.on_account_event({'e':'ACCOUNT_STREAM_RESYNC'});self.e.step()
        self.assertEqual(self.b.order.call_count,1)

    def test_tp_cancel_warns_without_replacement(self):
        self.e.on_account_event(self.event());self.e.step()
        self.now+=1
        self.e.on_account_event(self.event(cid='tp',status='CANCELED',qty='0'));self.e.step()
        self.assertTrue(self.e.state['halt'])
        self.assertEqual(len(self.b.tps),1)

    def test_overflow_requests_resync(self):
        for _ in range(2001):self.e.on_account_event(self.event())
        self.assertTrue(self.e.account_resync.is_set())
        self.assertEqual(self.e.account_events.qsize(),2000)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.now=2000000000.
        self.b=Mock(cooldown_until=0)
        self.emit=Mock();self.log=Mock()
        self.s=AccountStream(self.b,self.emit,self.log,Mock(),clock=lambda:self.now)
        self.s.ws=Mock()

    def test_open_keepalive_and_secret_redaction(self):
        self.s.on_open(self.s.ws)
        self.assertTrue(self.s.connected)
        self.emit.assert_called_with({'e':'ACCOUNT_STREAM_RESYNC'})
        self.s.keepalive()
        self.b.request.assert_called_once_with('PUT','/fapi/v1/listenKey',signed=False)
        self.s.keepalive();self.assertEqual(self.b.request.call_count,1)
        self.s.on_error(self.s.ws,RuntimeError('secret-listen-key'))
        self.assertNotIn('secret-listen-key',str(self.log.call_args_list))

    def test_expiry_and_disconnect_reconcile(self):
        self.s.on_open(self.s.ws)
        self.s.on_message(self.s.ws,json.dumps({'e':'listenKeyExpired','listenKey':'secret'}))
        self.assertFalse(self.s.connected);self.s.ws.close.assert_called_once()
        self.s.on_close(self.s.ws)
        self.emit.assert_called_with({'e':'ACCOUNT_STREAM_RESYNC'})
        self.assertNotIn('secret',str(self.log.call_args_list))

    def test_envelope_and_keepalive_pause(self):
        event={'e':'ACCOUNT_UPDATE','E':int(self.now*1000),'a':{'P':[]}}
        self.s.on_message(self.s.ws,json.dumps({'data':event}))
        self.emit.assert_called_with(event)
        self.s.connected=True;self.b.cooldown_until=self.now+60
        self.s.keepalive();self.b.request.assert_not_called()

    def test_connect_uses_private_route_and_unsigned_session_without_logging_key(self):
        self.b.request.return_value={'listenKey':'test-secret-key'}
        ws=Mock()
        def run(**kwargs):
            self.s.on_open(ws)
            self.s.stop_event.set()
        ws.run_forever.side_effect=run
        self.s.websocket.WebSocketApp.return_value=ws
        self.s._run()
        self.b.request.assert_called_once_with('POST','/fapi/v1/listenKey',signed=False)
        url=self.s.websocket.WebSocketApp.call_args.args[0]
        self.assertTrue(url.startswith('wss://fstream.binance.com/private/ws?'))
        self.assertIn('events=ORDER_TRADE_UPDATE',url)
        self.assertNotIn('test-secret-key',str(self.log.call_args_list))

    def test_public_pause_blocks_keepalive(self):
        self.s.connected=True;self.s.blocked=lambda:True
        self.s.keepalive()
        self.b.request.assert_not_called()
