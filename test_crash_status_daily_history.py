"""Offline safety tests for Paris daily report; broker GET is mocked."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import crash_status_daily_history as history

PARIS = ZoneInfo("Europe/Paris")


def instant(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=PARIS).timestamp()


class Broker:
    def __init__(self):
        self.calls = []
        self.events = [
            {"incomeType": "REALIZED_PNL", "asset": "USDT", "income": "12"},
            {"incomeType": "COMMISSION", "asset": "USDT", "income": "-1"},
            {"incomeType": "FUNDING_FEE", "asset": "USDT", "income": "-0.2"},
            {"incomeType": "TRANSFER", "asset": "USDT", "income": "2000"},
        ]

    def request(self, method, path, params):
        self.calls.append((method, path, params))
        if method != "GET" or path != "/fapi/v1/income":
            raise AssertionError("Non read-only query")
        return list(self.events)


class DailyBotStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        (self.directory / history.LEDGER).write_text(json.dumps({
            "version": 1, "days": {
                "2026-11-13": {"peak": "1080"},
                "2026-11-14": {
                    "first_equity": "1000", "peak": "1100", "low": "970",
                    "last_equity": "1030", "btc_first": "100000",
                    "btc_last": "97000", "last_at": "2026-11-14T23:55:00+01:00"
                }
            }
        }), encoding="utf-8")
        self.broker = Broker()
        self.executor = SimpleNamespace(broker=self.broker, state={"orders": {
            "a": {"day":"2026-11-14", "phase":"DONE",
                  "filled_qty":"1", "entry_filled_at":instant(14,10),
                  "tp_filled_at":instant(14,11)},
            "b": {"day":"2026-11-14", "phase":"CLOSED_EXTERNALLY",
                  "filled_qty":"1", "entry_filled_at":instant(14,12),
                  "flat_confirmed_at":instant(14,13)},
            "c": {"day":"2026-11-14", "phase":"NO_FILL", "filled_qty":"0"},
        }})

    def test_0105_fresh_snapshot_and_stable_rows(self):
        before = history.report(self.directory, instant(15,1,4), self.executor)
        self.assertEqual(before, [history.HEADERS])
        self.assertFalse(self.broker.calls)
        rows = history.report(self.directory, instant(15,1,5), self.executor)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], "2026-11-14")
        self.assertEqual(rows[1][1:4], [1100,970,1080])
        self.assertEqual(rows[1][4], -50)
        self.assertEqual(rows[1][6], -110)
        self.assertEqual(rows[1][8:13], [-3,2,2,10.8,30])
        self.assertEqual(len(self.broker.calls), 1)
        self.assertEqual(self.broker.calls[0][0:2], ("GET","/fapi/v1/income"))
        again = history.report(self.directory, instant(15,2,0), self.executor)
        self.assertEqual(rows, again)
        self.assertEqual(len(self.broker.calls), 1)

    def test_archive_keeps_existing_date_after_new_day(self):
        history.report(self.directory, instant(15,1,5), self.executor)
        source = json.loads((self.directory / history.LEDGER).read_text(encoding="utf-8"))
        source["days"]["2026-11-15"] = {
            "first_equity":"1030", "peak":"1040", "low":"1020",
            "last_equity":"1035", "btc_first":"97000",
            "btc_last":"98000", "last_at":"2026-11-15T23:50:00+01:00",
        }
        (self.directory / history.LEDGER).write_text(json.dumps(source))
        rows = history.report(self.directory, instant(16,1,5), self.executor)
        self.assertEqual([r[0] for r in rows[1:]], ["2026-11-14", "2026-11-15"])
        self.assertEqual(rows[2][3], 1100)
        self.assertEqual(rows[1][1], 1100)

    def test_missing_event_timestamps_are_not_reported_as_zero(self):
        self.executor.state["orders"]["a"].pop("entry_filled_at")
        rows = history.report(self.directory, instant(15,1,5), self.executor)
        self.assertEqual(rows[1][9:11], ["VERİ YOK","VERİ YOK"])
        self.assertIn("eski işlemlerde", rows[1][-1])

    def test_binance_api_error_never_becomes_zero_profit(self):
        class Failing:
            def request(self, method, path, params):
                raise RuntimeError("rate deferred")
        self.executor.broker = Failing()
        rows = history.report(self.directory, instant(15,1,5), self.executor)
        self.assertEqual(rows[1][11], "VERİ YOK")

    def test_page_limit_fails_closed(self):
        self.broker.events = [{"incomeType":"REALIZED_PNL",
                               "asset":"USDT","income":"1"}] * 1000
        rows = history.report(self.directory, instant(15,1,5), self.executor)
        self.assertEqual(rows[1][11], "VERİ YOK")


if __name__ == "__main__":
    unittest.main()
