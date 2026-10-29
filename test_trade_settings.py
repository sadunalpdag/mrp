import tempfile
import unittest
from decimal import Decimal
from unittest.mock import Mock
from crash_night import NightExecutor, plan
from crash_telegram_control import TelegramControl
from test_crash_services import INFO


class TradeSettingsTests(unittest.TestCase):
    def test_commands_scoped_and_not_forwarded(self):
        with tempfile.TemporaryDirectory() as d:
            c = TelegramControl(d, 'fake', '123', None, Mock(), Mock(), Mock(), lambda:1000, Mock(), counter_only=True, require_owner=False)
            msg = {'chat':{'id':123}, 'from':{'id':456,'is_bot':False}, 'date':1000, 'text':'/limit 15'}
            self.assertEqual(c.command({'message':msg}), 'limit:15')
            msg['text'] = '/tutar 100'
            self.assertEqual(c.command({'message':msg}), 'amount:100')
            msg['chat']['id'] = 999
            self.assertIsNone(c.command({'message':msg}))
            msg['chat']['id'] = 123
            msg['forward_date'] = 999
            self.assertIsNone(c.command({'message':msg}))

    def test_settings_persist_without_resetting_counter_or_orders(self):
        with tempfile.TemporaryDirectory() as d:
            e = NightExecutor(d, Mock(), Mock(), Mock())
            e.apply_control('counter:3')
            e.apply_control('limit:15')
            e.apply_control('amount:100')
            self.assertEqual(e.daily_count(), 3)
            self.assertEqual(e.state['orders'], {})
            self.assertIn('100 USDT', e.summary())
            for bad in ('limit:0','limit:16','limit:1.5','amount:NaN','amount:Infinity','amount:-1','amount:101','amount:no'):
                self.assertIn('⚠️', e.apply_control(bad))
            self.assertEqual(e.max_daily, 15)
            self.assertEqual(e.position_usdt, Decimal('100'))
            e.guard.close()
            e = NightExecutor(d, Mock(), Mock(), Mock())
            self.assertEqual(e.max_daily, 15)
            self.assertEqual(e.position_usdt, Decimal('100'))
            self.assertEqual(e.daily_count(), 3)
            e.guard.close()

    def test_budget_rounds_down_and_keeps_exchange_minimum(self):
        record = {'entry':'3', 'tp':'3.03'}
        p = plan(record, INFO, '100')
        self.assertEqual(Decimal(p['quantity']), Decimal('33.3'))
        self.assertLessEqual(Decimal(p['price'])*Decimal(p['quantity']), 100)
        with self.assertRaisesRegex(ValueError, 'MIN_NOTIONAL'):
            plan(record, INFO, '1')
