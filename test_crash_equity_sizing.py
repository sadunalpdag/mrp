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

from crash_equity_sizing import proportional_amount, read_equity
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


class ExecutorDailySizingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old=os.environ.copy()
        self.addCleanup(lambda: (os.environ.clear(),os.environ.update(self.old)))
        os.environ.update({'DATA_DIR': self.tmp.name,
                           'CRASH_AUTO_POSITION_SIZING_ENABLED':'1'})
        self.now=stamp('2026-11-12T23:30:00')
        self.e=NightExecutor(self.tmp.name,DummyBroker(),Mock(),Mock(),
                             clock=lambda:self.now,enabled=False)
        self.addCleanup(self.e.guard.close)

    def sample(self,eq,at=None):
        at = self.now if at is None else at
        path=Path(self.tmp.name)/'daily_profit_lock_paper.json'
        path.write_text(json.dumps({
            'mode':'PAPER_ONLY', 'sampled_at':datetime.fromtimestamp(
                at,timezone.utc).isoformat(), 'last_equity':str(eq)}),
                encoding='utf8')

    def test_next_midnight_freezes_new_amount_and_restarts_same_day(self):
        self.sample(1000)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(500))
        self.assertEqual(self.e.state['auto_position_sizing']['day'],'2026-11-12')
        self.now=stamp('2026-11-13T00:00:05')
        self.e._daily_size_update()
        self.assertTrue(self.e.auto_sizing_pending)  # prior day's sample
        self.now=stamp('2026-11-13T00:00:40')
        self.sample(1030)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(515))
        self.assertFalse(self.e.auto_sizing_pending)
        self.assertEqual(self.e.state['trade_settings']['position_usdt'],'515')
        self.now=stamp('2026-11-13T20:30:00')
        self.sample(900)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(515))  # once per day
        self.e.guard.close()
        self.e=NightExecutor(self.tmp.name,DummyBroker(),Mock(),Mock(),
                             clock=lambda:self.now,enabled=False)
        self.addCleanup(self.e.guard.close)
        self.assertEqual(self.e.position_usdt,Decimal(515))
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(515))
        self.now=stamp('2026-11-14T00:01:00')
        self.sample(999.1)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(500))

    def test_missing_midnight_equity_waits_then_carries_previous_day(self):
        self.sample(1000)
        self.e._daily_size_update()
        self.now=stamp('2026-11-13T00:02:00')
        self.e._daily_size_update()
        self.assertTrue(self.e.auto_sizing_pending)
        self.now=stamp('2026-11-13T00:10:01')
        self.e._daily_size_update()
        self.assertFalse(self.e.auto_sizing_pending)
        self.assertEqual(self.e.position_usdt,Decimal(500))
        self.assertEqual(self.e.state['auto_position_sizing']['day'],'2026-11-13')
        self.now=stamp('2026-11-13T13:00:00')
        self.sample(1030)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(500))

    def test_settings_toggle_off_and_manual_override_reanchor(self):
        self.sample(1000)
        self.e._daily_size_update()
        self.assertIn('KAPALI', self.e.apply_control('auto_size_off'))
        self.assertFalse(self.e._auto_sizing_on())
        self.now=stamp('2026-11-13T00:01:00')
        self.sample(1030)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt, Decimal(500))
        self.assertIn('AÇIK', self.e.apply_control('auto_size_on'))
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(500))
        self.assertIn('Yeni giriş ayarı',self.e.apply_control('amount:400'))
        self.assertEqual(self.e.state['auto_position_sizing']['day'],None)
        self.e._daily_size_update()
        self.assertEqual(self.e.state['auto_position_sizing']['day'],'2026-11-13')
        self.now=stamp('2026-11-14T00:01:00')
        self.sample(999.1)
        self.e._daily_size_update()
        self.assertEqual(self.e.position_usdt,Decimal(388))


if __name__ == "__main__":
    unittest.main()
