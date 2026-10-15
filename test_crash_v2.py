"""Offline regression checks for V2-only signal filter."""
import ast
import math
import unittest
from pathlib import Path
from unittest.mock import Mock

import crash_v2


class CrashV2Tests(unittest.TestCase):
    def rows(self, at, low="0.98", close="1.02"):
        end = int(at * 1000) // 900000 * 900000
        return [[t, "1", "1.06", low, close, "100", t + 899999]
                for t in range(end - 480*900000, end, 900000)]

    def test_complete_history_and_recovery(self):
        data = crash_v2.assess_entry(self.rows(2000000000), [], 2000000000)
        self.assertEqual(data["status"], "STRONG_V2")
        self.assertEqual(data["candles"], 480)
        self.assertGreater(data["recovery_4h_pct"], 1)

    def test_history_missing_fail_closed(self):
        rows = self.rows(2000000000)
        result = crash_v2.assess_entry(rows[:-1], [], 2000000000)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["candles"], 479)

    def test_closed_ws_tail_can_fill_gap(self):
        rows = self.rows(2000000000)
        result = crash_v2.assess_entry(rows[:-1], rows[-1:], 2000000000)
        self.assertTrue(result["qualified"])

    def test_future_candle_never_accepted(self):
        at = 2000000000
        rows = self.rows(at)
        rows[-1][6] = int(at*1000)+900000
        self.assertFalse(crash_v2.assess_entry(rows, [], at)["qualified"])

    def test_recovery_less_than_one_percent(self):
        data = crash_v2.assess_entry(self.rows(2000000000, "1", "1.005"), [], 2000000000)
        self.assertEqual(data["status"], "FILTERED_V2")

    def test_old_candidate_does_not_send_v2_telegram(self):
        tree = ast.parse(Path(__file__).with_name("ema.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_crash_send_pending")
        env = dict(CRASH_BOUNCE_FILTER_VERSION="five_day_entry_v2_only",
                   _crash_save_state=Mock(), tg_send=Mock(return_value=True))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "ema.py", "exec"), env)
        env["_crash_send_pending"]({"history":[{"signal_filter_version":"rsi_ema_score_v2",
                                               "v2_status":"STRONG_V2", "v2_sent":False}]}, 100)
        env["tg_send"].assert_not_called()

    def test_only_once_after_qualified_entry(self):
        tree = ast.parse(Path(__file__).with_name("ema.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_crash_send_pending")
        env = dict(CRASH_BOUNCE_FILTER_VERSION="five_day_entry_v2_only",
                   _crash_save_state=Mock(), tg_send=Mock(return_value=True))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "ema.py", "exec"), env)
        record = dict(signal_filter_version="five_day_entry_v2_only", v2_status="STRONG_V2",
                      v2_sent=False, entered_at=100, symbol="TESTUSDT", change_24h=-15,
                      entry="1.000", tp="1.010", observed_entry_price=1.001,
                      v2_features=dict(recovery_4h_pct=1.2, rsi14=24, candles=480,
                                       return_5d_pct=-13))
        state={"history":[record], "symbols":{"TESTUSDT":record}}
        env["_crash_send_pending"](state, 110)
        env["_crash_send_pending"](state, 115)
        env["tg_send"].assert_called_once()
        self.assertTrue(record["v2_sent"])


if __name__ == "__main__":
    unittest.main()
