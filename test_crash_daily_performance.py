"""Offline daily report tests; no Binance API or real order writes."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import crash_daily_performance as perf
import crash_sheets_dashboard as dash

PARIS = ZoneInfo("Europe/Paris")


def timestamp(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=PARIS)


class DailyPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sample(self, observed, equity, mode="PAPER_ONLY"):
        (self.root / perf.OBSERVER).write_text(json.dumps({
            "mode": mode, "sampled_at": observed.isoformat(),
            "last_equity": equity}), encoding="utf-8")

    def tick(self, observed, btc, orders=None):
        executor = SimpleNamespace(state={"orders": orders or {}})
        tickers = ([{"symbol": "BTCUSDT", "lastPrice": str(btc)}]
                   if btc is not None else [])
        return perf.update(self.root, tickers, executor, observed.timestamp() + 50)

    def test_observed_peak_daily_return_matched_btc_and_real_tp(self):
        first = timestamp(14, 10)
        orders = {
            "a": {"day": "2026-11-14", "phase": "DONE", "filled_qty": "1"},
            "b": {"day": "2026-11-14", "phase": "CLOSED_EXTERNALLY", "filled_qty": "1"},
            "c": {"day": "2026-11-14", "phase": "PROTECTED", "filled_qty": "1"},
            "d": {"day": "2026-11-14", "phase": "NO_FILL", "filled_qty": "0"},
            "e": {"day": "2026-11-14", "phase": "ENTRY_UNKNOWN"},
        }
        self.sample(first, "1000")
        result = self.tick(first, "100000", orders)
        self.assertEqual(result[1][0], "2026-11-14")
        self.assertEqual(result[1][1:6], [1000.0,1000.0,1000.0,0.0,0.0])
        self.assertEqual(result[1][14:19], [3,1,1,1,50.0])
        newer = timestamp(14, 10, 20)
        self.sample(newer, "1050")
        result = self.tick(newer, "99000", orders)
        self.assertEqual(result[1][2], 1050)
        self.assertEqual(result[1][5], 5.0)
        self.assertEqual(result[1][8], -1.0)
        self.assertEqual(result[1][9], 5.0)
        self.assertEqual(result[1][10], 6.0)
        latest = timestamp(14, 11)
        self.sample(latest, "1010")
        result = self.tick(latest, "98000", orders)
        self.assertEqual(result[1][2:6], [1050.0,1010.0,10.0,1.0])
        self.assertEqual(result[1][8:11], [-2.0, 1.0, 3.0])
        # A 5-minute Sheets refresh cannot change BTC's end point while
        # the observer equity timestamp is still the previous sample.
        repeated = perf.update(self.root, [{"symbol":"BTCUSDT","lastPrice":"50000"}],
                               SimpleNamespace(state={"orders":orders}),
                               latest.timestamp()+100)
        self.assertEqual(repeated[1][7], 98000.0)

    def test_next_paris_day_preserves_old_data_and_marks_coverage(self):
        a = timestamp(14, 22, 30)
        self.sample(a, "1000")
        self.tick(a, "100000")
        b = timestamp(15, 0, 5)
        self.sample(b, "1030")
        report = self.tick(b, "102000")
        self.assertEqual([row[0] for row in report[1:]], ["2026-11-15","2026-11-14"])
        self.assertIn("EKSİK GÜN SONU", report[2][13])
        self.assertEqual(report[1][1], 1030.0)
        self.assertEqual(report[2][3], 1000.0)

    def test_missing_or_stale_observer_never_creates_fake_row(self):
        at = timestamp(14, 10)
        self.assertIsNone(self.tick(at, "100000"))
        self.sample(at-timedelta(hours=2), "1000")
        self.assertIsNone(self.tick(at, "100000"))
        self.sample(at, "1000", mode="LIVE")
        self.assertIsNone(self.tick(at, "100000"))
        self.assertFalse((self.root / perf.FILE_NAME).exists())

    def test_missing_btc_stays_unknown_until_paired_observation(self):
        first = timestamp(14, 10)
        self.sample(first, "1000")
        report = self.tick(first, None)
        self.assertEqual(report[1][8:11], ["VERİ YOK"] * 3)
        second = timestamp(14, 10, 30)
        self.sample(second, "1020")
        report = self.tick(second, "100000")
        self.assertEqual(report[1][5], 2.0)
        self.assertEqual(report[1][8:11], [0.0,0.0,0.0])
        third = timestamp(14, 11)
        self.sample(third, "1030")
        report = self.tick(third, "101000")
        self.assertEqual(report[1][9], 0.98)
        self.assertEqual(report[1][10], -0.02)

    def test_publish_creates_only_requested_tab_with_valid_observation(self):
        now = timestamp(14, 16)
        self.sample(now, "1050")
        book = Mock(title="EMA Filter")
        book.worksheets.return_value = [
            SimpleNamespace(title="Bot Status", id=1, row_count=40),
            SimpleNamespace(title="Sinyal", id=2, col_count=len(dash.HEADERS))]
        book.add_worksheet.return_value = SimpleNamespace(
            title="Günlük Performans", id=3,
            row_count=100, col_count=len(perf.HEADERS))
        client = Mock()
        client.open_by_key.return_value = book
        with patch.object(dash, "_credentials", return_value=(client, "my-test-id")), \
             patch.object(dash, "_ensure_four_risk_columns"), \
             patch.object(dash, "_format"), \
             patch.object(dash, "_sync_external_legacy_status"):
            ok = dash.publish({"symbols": {}, "history": []},
                              [{"symbol": "BTCUSDT", "priceChangePercent": "1",
                                "lastPrice": "100000"}],
                              {}, {"connected": True}, now.timestamp()+10,
                              str(self.root),
                              executor=SimpleNamespace(state={"orders":{}}))
        self.assertTrue(ok)
        book.add_worksheet.assert_called_once()
        info = book.add_worksheet.call_args.kwargs
        self.assertEqual(info["title"], "Günlük Performans")
        data = book.values_batch_update.call_args.args[0]["data"]
        self.assertEqual(len(data), 4)
        self.assertEqual(data[2]["range"], "'Günlük Performans'!A1:T100")
        self.assertEqual(data[2]["values"][1][0], "2026-11-14")
        self.assertEqual(data[3]["range"], "'Bot Status'!A40:P40")

    def test_publish_keeps_existing_tabs_when_no_observation(self):
        now = timestamp(14, 16)
        book = Mock(title="EMA Filter")
        book.worksheets.return_value = [
            SimpleNamespace(title="Bot Status", id=1, row_count=40),
            SimpleNamespace(title="Sinyal", id=2, col_count=len(dash.HEADERS))]
        client = Mock()
        client.open_by_key.return_value = book
        with patch.object(dash, "_credentials", return_value=(client, "my-test-id")), \
             patch.object(dash, "_ensure_four_risk_columns"), \
             patch.object(dash, "_format"), \
             patch.object(dash, "_sync_external_legacy_status"):
            ok = dash.publish({"symbols": {}, "history": []}, [], {},
                              {"connected": True}, now.timestamp(),
                              str(self.root))
        self.assertTrue(ok)
        self.assertFalse(book.add_worksheet.called)
        data = book.values_batch_update.call_args.args[0]["data"]
        self.assertEqual(len(data), 3)
        self.assertEqual(data[2]["range"], "'Bot Status'!A40:P40")


if __name__ == "__main__":
    unittest.main()
