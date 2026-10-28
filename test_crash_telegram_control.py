"""Owner authentication, stale-command and execution-gate regressions; no network."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from crash_telegram_control import TelegramControl
from crash_night import NightExecutor
from test_crash_services import FakeBroker, timestamp


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.now=timestamp('2026-10-27T21:00:00')
        self.action=Mock(return_value='ok'); self.deliver=Mock(return_value={'ok':True})
        self.fetch=Mock(return_value=[])
        self.c=self.make()

    def make(self, chat='123', owner=None):
        return TelegramControl(self.tmp.name,'fake',chat,owner,self.action,self.deliver,Mock(),lambda:self.now,self.fetch)

    def update(self, uid=1, text='/otomatik_ac'):
        return dict(update_id=uid,message={'chat':{'id':123},'from':{'id':123,'is_bot':False},'date':self.now,'text':text})

    def test_counter_only_commands_authorized_and_cannot_toggle_trading(self):
        self.c.counter_only=True
        self.assertEqual(self.c.command(self.update(text='/sayac 0')),'counter:0')
        self.assertEqual(self.c.command(self.update(text='/sayac 3')),'counter:3')
        self.assertEqual(self.c.command(self.update(text='/sayac')),'status')
        for text in ('/otomatik_ac','/otomatik_kapat','/sayac -1','/sayac 1 extra'):
            self.assertIsNone(self.c.command(self.update(text=text)))
        u=self.update(text='/sayac 0');u['message']['from']['id']=456
        self.assertIsNone(self.c.command(u))

    def test_chat_scoped_counter_allows_other_users_not_other_chats(self):
        c=TelegramControl(self.tmp.name,'fake','-100999',None,self.action,self.deliver,Mock(),lambda:self.now,self.fetch,counter_only=True,require_owner=False)
        u=self.update(text='/sayac 0');u['message']['chat']['id']=-100999;u['message']['from']['id']=456
        self.assertEqual(c.command(u),'counter:0')
        u['message']['chat']['id']=-100888
        self.assertIsNone(c.command(u))
        u['message']['chat']['id']=-100999;u['message']['text']='/otomatik_ac'
        self.assertIsNone(c.command(u))

    def test_owner_only_exact_commands(self):
        for text, action in [('/otomatik_ac','on'),('/otomatik_kapat','off'),('/otomatik_durum','status')]:
            self.assertEqual(self.c.command(self.update(text=text)),action)
        for text in ['buy AUSDT','/otomatik_ac extra','please /otomatik_ac','/otomatik_ac@otherbot']:
            self.assertIsNone(self.c.command(self.update(text=text)))
        for field,value in [('from',{'id':456,'is_bot':False}),('from',{'id':123,'is_bot':True}),('chat',{'id':456})]:
            u=self.update(); u['message'][field]=value
            self.assertIsNone(self.c.command(u))

    def test_old_edited_forwarded_messages_rejected(self):
        for field,value in [('date',self.now-1),('date',self.now+10),('edit_date',self.now),
                            ('forward_origin',{'type':'user'}),('forward_from',{'id':123}),('via_bot',{'id':2}),('sender_chat',{'id':123})]:
            u=self.update(); u['message'][field]=value
            self.assertIsNone(self.c.command(u))
        u=self.update(); u['edited_message']=u.pop('message')
        self.assertIsNone(self.c.command(u))
        self.now+=91
        u=self.update(); u['message']['date']-=91
        self.assertIsNone(self.c.command(u))

    def test_group_requires_explicit_owner_and_checks_both_ids(self):
        with self.assertRaises(ValueError): self.make('-100999')
        c=self.make('-100999','123'); u=self.update(); u['message']['chat']['id']=-100999
        self.assertEqual(c.command(u),'on')
        u['message']['from']['id']=456
        self.assertIsNone(c.command(u))

    def test_offset_persisted_before_action_no_replay_after_restart(self):
        def action(value):
            self.assertEqual(json.loads(Path(self.c.path).read_text())['offset'],2)
            return 'ok'
        self.action.side_effect=action
        self.fetch.return_value=[self.update()]
        self.c.step(); self.c=self.make(); self.c.step()
        self.action.assert_called_once_with('on')

    def test_offset_failure_cannot_activate(self):
        self.fetch.return_value=[self.update()]
        self.c.save_offset=Mock(side_effect=OSError('disk'))
        with self.assertRaises(OSError): self.c.step()
        self.action.assert_not_called()

    def test_reply_failure_does_not_replay_activation(self):
        self.fetch.return_value=[self.update()]
        self.deliver.return_value={'ok':False}
        self.c.step(); self.c.step()
        self.action.assert_called_once_with('on')


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.now=timestamp('2026-10-27T21:00:00')
        self.b=FakeBroker()
        self.e=self.make()
        self.r=dict(symbol='AUSDT',signal_id='id1',entry='10',tp='10.1',v2_status='STRONG_V2',v2_sent=True,tracking_status='OPEN')

    def make(self):
        e=NightExecutor(self.tmp.name,self.b,Mock(return_value={'ok':True}),Mock(),lambda:self.now)
        self.addCleanup(e.guard.close)
        return e

    def test_starts_paused_and_restart_cannot_auto_enable(self):
        self.e.enter(self.r,self.now); self.assertFalse(self.b.buys)
        self.e.apply_control('on'); self.assertTrue(self.e.enabled.is_set())
        self.e.guard.close(); self.e=self.make()
        self.assertFalse(self.e.enabled.is_set())
        self.e.enter(self.r,self.now); self.assertFalse(self.b.buys)

    def test_explicit_on_can_enter_off_prevents_next(self):
        self.e.apply_control('on'); self.now+=1
        self.e.enter(self.r,self.now); self.assertEqual(len(self.b.buys),1)
        self.e.deliver_pending(); self.e.apply_control('off')
        self.e.enter(dict(self.r,signal_id='id2',symbol='BUSDT'),self.now)
        self.assertEqual(len(self.b.buys),1)

    def test_off_does_not_stop_tp_monitor_or_close_position(self):
        self.e.apply_control('on'); self.now+=1; self.e.enter(self.r,self.now)
        self.e.deliver_pending(); self.e.apply_control('off')
        order=self.e.state['orders']['id1']; self.b.orders[order['tp_id']]['status']='FILLED'
        self.e.step()
        self.assertEqual(order['phase'],'DONE')
        self.assertEqual(len(self.b.buys),1); self.assertEqual(len(self.b.tps),1)

    def test_on_does_not_clear_tp_error_or_daily_quota(self):
        self.e.state['halt']='TP DOĞRULANAMADI'
        message=self.e.apply_control('on')
        self.assertIn('Açılamadı',message); self.assertFalse(self.e.enabled.is_set())
        self.assertEqual(self.e.state['halt'],'TP DOĞRULANAMADI')

    def test_on_discards_buffered_crossings_and_pre_activation_event(self):
        self.e.queue.put((self.r,self.now))
        self.now+=1; self.e.apply_control('on')
        self.assertTrue(self.e.queue.empty())
        self.e.enter(self.r,self.now-1)
        self.assertFalse(self.b.buys)

    def test_off_during_preflight_prevents_post(self):
        self.e.apply_control('on')
        self.b.preflight=lambda symbol:self.e.enabled.clear()
        self.e.enter(self.r,self.now)
        self.assertFalse(self.b.buys)
