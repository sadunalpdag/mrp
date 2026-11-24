"""Offline checks for daily equity-ratio sizing; no exchange network calls."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from crash_equity_sizing import proportional_amount, read_equity, one_third_amount, previous_day_closing_equity
from crash_night import NightExecutor

PARIS = ZoneInfo("Europe/Paris")


def stamp(local):
    return datetime.fromisoformat(local).replace(tzinfo=PARIS).timestamp()


class DummyBroker:
    pass


class SizingMathTests(unittest.TestCase):
    def test_example_three_percent_up_and_down(self):
        self.assertEqual(proportional_amount('500','1000','1030'), Decimal('515'))
        self.assertEqual(proportional_amount('500','1000','970'), Decimal('485'))
        self.assertEqual(proportional_amount('515','1030','1060.9'), Decimal('530'))
        self.assertEqual(proportional_amount('515','1030','999.1'), Decimal('500'))

    def test_half_up_and_limits(self):
        self.assertEqual(proportional_amount('500','1000','1001'), Decimal('501'))
        self.assertEqual(proportional_amount('500','1000','999'), Decimal('500'))
        self.assertEqual(proportional_amount('1000','100','10000'), Decimal('1000'))
        self.assertEqual(proportional_amount('500','1000','1'), Decimal('1'))
        for bad in ('NaN', '0', '-500', 'Infinity'):
            with self.assertRaises(ValueError):
                proportional_amount('500',bad,'1000')

    def test_stale_future_wrong_mode_or_missing_sample_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'paper.json'
            now = stamp('2026-11-13T00:02:00')
            valid = {'mode': 'PAPER_ONLY', 'sampled_at':
                     datetime.fromtimestamp(now-60, timezone.utc).isoformat(),
                     'last_equity':'1030'}
            path.write_text(json.dumps(valid),encoding='utf8')
            self.assertEqual(read_equity(path,now)[0],Decimal('1030'))
            valid['mode'] = 'LIVE'
            path.write_text(json.dumps(valid),encoding='utf8')
            self.assertIsNone(read_equity(path,now))
            valid['mode'] = 'PAPER_ONLY'
            valid['sampled_at']=datetime.fromtimestamp(now-700,timezone.utc).isoformat()
            path.write_text(json.dumps(valid),encoding='utf8')
            self.assertIsNone(read_equity(path,now))


class ThirdEquitySizingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = os.environ.copy()
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.old)))
        os.environ.update({'DATA_DIR': self.tmp.name,
                           'CRASH_AUTO_POSITION_SIZING_ENABLED': '1'})
        self.now = stamp('2026-11-24T23:30:00')
        self.e = NightExecutor(self.tmp.name, DummyBroker(), Mock(), Mock(),
                               clock=lambda: self.now, enabled=False)
        self.addCleanup(self.e.guard.close)

    def sample(self, equity, at=None):
        at = self.now if at is None else at
        (Path(self.tmp.name) / 'daily_profit_lock_paper.json').write_text(
            json.dumps({'mode': 'PAPER_ONLY',
                         'sampled_at': datetime.fromtimestamp(
                             at, timezone.utc).isoformat(),
                         'last_equity': str(equity)}), encoding='utf8')

    def test_ceiling_examples(self):
        for equity, expected in [('996', '332'), ('997', '333'),
                                 ('998', '333'), ('1000', '334'), ('1005', '335')]:
            self.assertEqual(one_third_amount(equity), Decimal(expected))
        for invalid in ('0', '-3', 'NaN', '3001'):
            with self.assertRaises(ValueError):
                one_third_amount(invalid)

    def test_roll_once_at_midnight_without_intraday_repricing(self):
        self.sample(997)
        self.e._daily_size_update()
        initial = self.e.position_usdt
        self.assertEqual(self.e.state['auto_position_sizing']['day'], '2026-11-24')
        self.assertEqual(self.e.position_usdt, initial)
        self.now = stamp('2026-11-25T00:01:00')
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(333))
        self.assertEqual(self.e.state['auto_position_sizing']['day'], '2026-11-25')
        self.now = stamp('2026-11-25T12:00:00')
        self.sample(1005)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(333))
        self.e.guard.close()
        self.e = NightExecutor(self.tmp.name, DummyBroker(), Mock(), Mock(),
                               clock=lambda: self.now, enabled=False)
        self.addCleanup(self.e.guard.close)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(333))

    def test_telegram_manual_amount_until_next_midnight(self):
        self.sample(997)
        self.e._daily_size_update()
        self.assertIn('kaydedildi', self.e.apply_control('amount:900'))
        self.assertEqual(self.e.position_usdt, Decimal(900))
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(900))
        self.now = stamp('2026-11-25T00:01:00')
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(333))
        self.assertIn('Render', self.e.apply_control('auto_size_off'))
        self.assertTrue(self.e._auto_sizing_on())

    def test_no_valid_previous_close_keeps_amount_not_block_new_buys(self):
        self.e._daily_size_update()
        self.now = stamp('2026-11-25T00:01:00')
        self.e._daily_size_update()
        self.assertFalse(self.e.auto_sizing_pending)
        self.assertEqual(self.e.state['auto_position_sizing']['reason'],
                         'WAIT_PREVIOUS_DAY_CLOSE')
        self.assertEqual(self.e.position_usdt, Decimal(500))
        self.sample(997, at=stamp('2026-11-24T23:12:00'))
        self.e._daily_size_update()
        self.now += 61
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(333))

    def test_legacy_sheets_no_cannot_disable_render_daily_sizing(self):
        self.e.state['auto_position_sizing']['enabled'] = False
        self.e._daily_size_update()
        self.assertTrue(self.e._auto_sizing_on())
        self.sample(997)
        self.now = stamp('2026-11-25T00:01:00')
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(333))


if __name__ == "__main__":
    unittest.main()
