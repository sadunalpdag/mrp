"""Offline invariants for no-REST, no-trading paper scalp monitor."""
import tempfile
import unittest
from unittest.mock import patch
import scalp_shadow_week as sw


def candles(count=55, close=101.0, first=0):
    rows = []
    for i in range(count):
        ts = first + i * sw.STEP_MS
        rows.append([ts, str(close), str(close * 1.002), str(close * 0.998),
                     str(close), "100", ts + sw.STEP_MS - 1])
    return rows


class ShadowWeekTests(unittest.TestCase):
    def test_rejects_unclosed_and_broken_ohlc(self):
        rows = candles(2, first=sw.STEP_MS)
        rows.append([3 * sw.STEP_MS, "1", "0.5", "1", "1",
                     "1", 4 * sw.STEP_MS - 1])
        rows.append([4 * sw.STEP_MS, "1", "1", "1", "1",
                     "1", 5 * sw.STEP_MS - 1])
        valid = sw._bars(rows, 4 * sw.STEP_MS / 1000)
        self.assertEqual(len(valid), 2)

    def test_daily_range_requires_all_real_candles(self):
        now = 2 * 86400 + 3600
        rows = candles(96, close=100, first=86400 * 1000)
        self.assertEqual(sw.previous_utc_range(rows, now), (100.2, 99.8))
        self.assertIsNone(sw.previous_utc_range(rows[:-1], now))

    def test_break_bounce_is_proxy(self):
        b = [(0, 104, 105, 103, 104, 1),
             (sw.STEP_MS, 104, 108, 103, 107, 1),
             (2 * sw.STEP_MS, 105.5, 108, 105, 106.8, 1)]
        result = sw.break_bounce(b, (105, 95))
        self.assertEqual(result[0], "LONG")
        self.assertIn("5m teyit yok", result[1])

    def test_outcomes_use_future_candles_and_ambiguous_on_double_touch(self):
        baseline = {"status": "OPEN", "ts": 0, "side": "LONG",
                    "tp": 101, "risk": 99}
        a = dict(baseline)
        sw._settle(a, [(sw.STEP_MS, 100, 101.2, 99.8, 100.5, 1)], 3600)
        self.assertEqual(a["status"], "TP")
        b = dict(baseline)
        sw._settle(b, [(sw.STEP_MS, 100, 101.2, 98.8, 100.5, 1)], 3600)
        self.assertEqual(b["status"], "AMBIGUOUS")

    def test_no_synthetic_data_and_bounded_selection(self):
        now = 56 * sw.STEP_MS / 1000
        sample = candles(55)
        tickers = [{"symbol": "A" + str(i) + "USDT",
                    "priceChangePercent": str(50-i),
                    "quoteVolume": str(1000-i)} for i in range(35)]
        calls = []
        def getter(symbol):
            calls.append(symbol)
            return sample
        with tempfile.TemporaryDirectory() as td:
            obj = sw._scan(td, tickers, getter, now, {"signals": []})
        self.assertEqual(len(obj["sampled_coins"]["PULLBACK"]), sw.MAX_SYMBOLS)
        self.assertEqual(len(obj["sampled_coins"]["BREAK_BOUNCE"]), sw.MAX_SYMBOLS)
        self.assertEqual(set(calls), {"A" + str(i) + "USDT" for i in range(10, 30)})
        self.assertEqual(len(calls), 20)
        self.assertEqual(len(obj["signals"]), 0)
        self.assertEqual(len(sw._tab_values(obj, "PULLBACK")), 57)

    def test_exact_gainer_bands_without_backfill_and_market_filter(self):
        tickers = [{"symbol": f"A{i:02}USDT", "priceChangePercent": str(40 - i)}
                   for i in range(34)]
        self.assertEqual(sw.gainer_groups(tickers)["PULLBACK"],
                         [f"A{i:02}USDT" for i in range(10, 20)])
        self.assertEqual(sw.gainer_groups(tickers)["BREAK_BOUNCE"],
                         [f"A{i:02}USDT" for i in range(20, 30)])
        allowed = {f"A{i:02}USDT" for i in range(26)}
        selected = sw.gainer_groups(tickers, allowed)
        self.assertEqual(selected["BREAK_BOUNCE"],
                         [f"A{i:02}USDT" for i in range(20, 26)])
        self.assertNotIn("A30USDT", selected["BREAK_BOUNCE"])

    def test_four_missing_true_candles_allowed_but_five_rejected(self):
        now = 57 * sw.STEP_MS / 1000
        rows = candles(56)
        four_gaps = [r for i, r in enumerate(rows) if i not in (4, 12, 20, 28)]
        self.assertTrue(sw._ready(sw._bars(four_gaps, now), now))
        five_gaps = [r for i, r in enumerate(rows) if i not in (4, 12, 20, 28, 36)]
        self.assertFalse(sw._ready(sw._bars(five_gaps, now), now))

    def test_no_future_candles_produce_data_gap_not_win(self):
        signal = {"status": "OPEN", "ts": 0, "side": "LONG", "tp": 101, "risk": 99}
        sw._settle(signal, [], sw.HORIZON_S + 1)
        self.assertEqual(signal["status"], "DATA_GAP")

    def test_wrong_workbook_fail_closed(self):
        class Book:
            title = "Other spreadsheet"
        class Client:
            def open_by_key(self, _):
                return Book()
        import types
        import sys
        fake = types.ModuleType("crash_sheets_dashboard")
        fake._credentials = lambda: (Client(), "sheet-id")
        fake._name = lambda s: "".join(c.lower() for c in s if c.isalnum())
        with patch.dict(sys.modules, {"crash_sheets_dashboard": fake}):
            with self.assertRaisesRegex(ValueError, "WRONG_WORKBOOK"):
                sw._publish({"signals": [], "last_scan": 1, "sampled_coins": []})


if __name__ == "__main__":
    unittest.main()
