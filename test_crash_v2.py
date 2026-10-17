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

    def _delivery_context(self):
        import time
        tree = ast.parse(Path(__file__).with_name("ema.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_crash_send_pending")
        receipt = Mock(return_value={"ok":True, "attempt_at":100, "accepted_at":101,
                                     "message_id":42})
        env = dict(CRASH_V2_SIGNAL_TTL_SECONDS=90,
                   _crash_save_state=Mock(), _crash_append_data=Mock(),
                   _crash_telegram_receipt=receipt, time=time)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "ema.py", "exec"), env)
        return env, receipt

    def _candidate(self):
        record = dict(signal_id="TEST-100", created_at=100, v2_delivery_phase="CANDIDATE",
                      signal_filter_version="five_day_entry_v2_only", v2_status="STRONG_V2",
                      v2_sent=False, tracking_status="WAIT_ENTRY", symbol="TESTUSDT",
                      change_24h=-15, entry="1.000", tp="1.010", last_price="0.995",
                      v2_features=dict(recovery_4h_pct=1.2, rsi14=24, candles=480,
                                       return_5d_pct=-13))
        return {"history":[record], "symbols":{"TESTUSDT":dict(record)}}, record

    def test_legacy_candidate_not_resent(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        record.pop("v2_delivery_phase")
        env["_crash_send_pending"](state, 105)
        receipt.assert_not_called()

    def test_qualified_candidate_alert_before_any_entry(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        env["_crash_send_pending"](state, 105)
        env["_crash_send_pending"](state, 110)
        receipt.assert_called_once()
        self.assertTrue(record["v2_sent"])
        self.assertEqual(record["v2_telegram_message_id"],42)
        self.assertEqual(record["v2_delivery_status"],"SENT")
        self.assertNotIn("entered_at",record)
        self.assertIn("Giriş henüz gerçekleşmedi",receipt.call_args.args[0])

    def test_fractional_strength_shown_and_does_not_block_telegram(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        weights = {
            "dip_stability":(14.3,25), "rsi_momentum":(11.7,20),
            "volume_confirmation":(9.4,20), "ema25_improvement":(8.6,15),
            "atr_tp_capacity":(5.5,10), "selling_slowdown":(3.2,5),
            "btc_context":(2.4,5)
        }
        record["v2_features"]["strength_score"] = {
            "score":55.1, "coverage_pct":100.0,
            "components":{k:{"points":v,"max_points":w} for k,(v,w) in weights.items()}
        }
        env["_crash_send_pending"](state,105)
        message = receipt.call_args.args[0]
        self.assertIn("GÜÇ PUANI: 55.1/100",message)
        self.assertIn("Dip 14.3/25",message)
        self.assertIn("RSI 11.7/20",message)
        self.assertIn("Hacim 9.4/20",message)
        self.assertIn("EMA25 8.6/15",message)
        self.assertTrue(record["v2_sent"])

    def test_stale_signal_never_sent_late(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        env["_crash_send_pending"](state, 191)
        self.assertEqual(record["v2_delivery_status"],"EXPIRED_NOT_SENT")
        receipt.assert_not_called()

    def test_timeout_marks_unknown_delivery_no_blind_retry(self):
        env, receipt = self._delivery_context()
        receipt.return_value = {"ok":False, "error":"TIMEOUT_DELIVERY_UNKNOWN",
                                "ambiguous":True}
        state, record = self._candidate()
        env["_crash_send_pending"](state, 105)
        env["_crash_send_pending"](state, 140)
        self.assertEqual(record["v2_delivery_status"],"UNKNOWN_DELIVERY")
        self.assertEqual(receipt.call_count,1)


if __name__ == "__main__":
    unittest.main()
