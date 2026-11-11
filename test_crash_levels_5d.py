import json
import math
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from crash_levels_5d import analyze, load_cached, STEP_MS
from crash_sheets_dashboard import HEADERS, build_rows, publish


class FiveDayLevelsTests(unittest.TestCase):
    now = 1794050000.0

    def candles(self, event="plain"):
        start = (int(self.now * 1000) // STEP_MS - 480) * STEP_MS
        result = []
        for i in range(480):
            upper = 110 - .06 * i
            close = upper - 5 + 4 * math.sin(2 * math.pi * i / 48)
            if event == "up" and i in (478, 479):
                close = upper + 3
            elif event == "down" and i in (478, 479):
                close = upper - 14
            elif event == "fake" and i in (476, 477):
                close = upper + 3
            elif event == "fake" and i in (478, 479):
                close = upper - 5
            t = start + i * STEP_MS
            result.append([t, str(close), str(close + .8),
                           str(close - .8), str(close), "1000", t + STEP_MS - 1])
        return result

    def test_five_day_upward_break_and_levels(self):
        report = analyze(self.candles("up"), self.now)
        self.assertEqual(report["coverage"], "480/480")
        self.assertEqual((report["status"], report["direction"]), ("KIRDI", "YUKARI"), report)
        self.assertIsNotNone(report["confirmed_at"])
        self.assertLess(report["fib"]["0"], report["fib"]["50"])
        self.assertLess(report["fib"]["50"], report["fib"]["100"])
        self.assertIsNotNone(report["support"])

    def test_five_day_downward_break(self):
        report = analyze(self.candles("down"), self.now)
        self.assertEqual((report["status"], report["direction"]), ("KIRDI", "AŞAĞI"), report)
        self.assertIsNotNone(report["resistance"])

    def test_false_break_has_direction_without_confirming_now(self):
        report = analyze(self.candles("fake"), self.now)
        self.assertEqual((report["status"], report["direction"]), ("SAHTE KIRILIM", "YUKARI"), report)

    def test_short_or_stale_history_fails_closed(self):
        short = analyze(self.candles()[-64:], self.now)
        self.assertEqual(short["status"], "VERİ YOK")
        self.assertEqual(short["coverage"], "64/480")
        self.assertEqual(short["fib"], {})
        stale = analyze(self.candles(), self.now + 3600)
        self.assertEqual(stale["status"], "VERİ YOK")

    def test_two_missing_bars_tolerated_but_three_are_not(self):
        data = self.candles("up")
        report = analyze(data[:100] + data[102:], self.now)
        self.assertEqual(report["coverage"], "478/480")
        self.assertIn(report["status"], ("KIRDI", "KIRILIM BEKLEME"))
        fail = analyze(data[:100] + data[103:], self.now)
        self.assertEqual(fail["status"], "VERİ YOK")

    def test_open_future_or_malformed_bars_never_count(self):
        rows = self.candles()
        last = rows[-1].copy()
        last[6] += STEP_MS
        result = analyze(rows[:-1] + [last], self.now)
        self.assertEqual(result["coverage"], "479/480")
        unclosed = rows[-1].copy()
        unclosed[0] += STEP_MS
        unclosed[6] += STEP_MS
        unclosed[4] = "99999"
        self.assertEqual(analyze(rows + [unclosed], self.now)["coverage"], "480/480")

    def test_disk_history_and_ws_merge_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "crash_research", "history_cache")
            os.makedirs(path)
            all_rows = self.candles()
            with open(os.path.join(path, "TESTUSDT.json"), "w") as fh:
                json.dump(all_rows[:-2], fh)
            joined = load_cached(td, "TESTUSDT", all_rows[-64:], self.now)
            self.assertEqual(len(joined), 480)
            result = analyze(joined, self.now)
            self.assertEqual(result["coverage"], "480/480")
            self.assertEqual(load_cached(td, "../BADUSDT", [], self.now), [])

    def test_sheets_only_new_columns_no_trade_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "crash_research", "history_cache")
            os.makedirs(path)
            with open(os.path.join(path, "TESTUSDT.json"), "w") as fh:
                json.dump(self.candles("up")[:-64], fh)
            record = {"signal_id": "s1", "v2_status": "STRONG_V2",
                      "created_at": self.now-60, "entry": 1, "tp": 1.01,
                      "failure_shadow": {"version": "tp_nonclosure_entry_gate_v2_atr", "flags": {"weak_volume": False,
                          "falling_rsi_ema": False, "repeat_lower_entry": False, "low_atr_15m": False}}}
            state = {"symbols": {"TESTUSDT": record}, "history": []}
            before = json.dumps(state, sort_keys=True)
            tickers = [{"symbol": "TESTUSDT", "priceChangePercent": "-20", "lastPrice": "1"}]
            _, rows, _, _ = build_rows(state, tickers, {}, {}, self.now,
                                       candle_get=lambda _: self.candles("up")[-64:],
                                       directory=td)
            self.assertEqual(len(HEADERS), 41)
            self.assertEqual(len(rows[1]), 41)
            self.assertEqual(rows[1][23], "🟢 KIRDI")
            self.assertEqual(rows[1][26], "YUKARI")
            self.assertEqual(rows[1][27], "480/480")
            self.assertIsInstance(rows[1][33], float)
            self.assertEqual(json.dumps(state, sort_keys=True), before)
            self.assertEqual(rows[1][13], 0)


if __name__ == "__main__":
    unittest.main()
